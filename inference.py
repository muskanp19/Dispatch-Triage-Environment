"""
inference.py — 911 Dispatch Triage Environment
===============================================
Runs one episode per difficulty level (easy → medium → hard).
The LLM agent reads natural-language incident descriptions and available
unit types, then decides which unit to dispatch to which incident each step.

Required environment variables:
    API_BASE_URL     LLM API endpoint       (default: HuggingFace Inference Router)
    MODEL_NAME       Model identifier       (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN         HuggingFace API token  (REQUIRED — no default)
    ENV_BASE_URL     Running server URL     (e.g. https://<user>-dispatch-triage-env.hf.space)

Optional:
    LOCAL_IMAGE_NAME Docker image name      (fallback when ENV_BASE_URL is not set)

Stdout format (strictly required by hackathon grader):
    [START] task=<str> env=<str> model=<str>
    [STEP]  step=<int> action=<str> reward=<float> done=<bool> error=<str|null>
    [END]   success=<bool> steps=<int> rewards=<float,...>
"""

import json
import os
import re
import sys
import textwrap
import traceback
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
# DispatchTriageEnv is the CLIENT — connects to the running server via WebSocket.
# inference.py lives at the repo root, so models/client are absolute imports.
from client import DispatchTriageEnv
from models import DispatchTriageAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str       = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str         = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_BASE_URL: Optional[str]     = os.getenv("ENV_BASE_URL", "https://muskanp-dispatch-triage-env.hf.space")
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME", "dispatch-triage-env:latest")

BENCHMARK: str          = "dispatch_triage_env"
DIFFICULTIES: List[str] = ["easy", "medium", "hard"]
MAX_STEPS: int          = 8        # safety ceiling (3 valid dispatches per episode max)
TEMPERATURE: float      = 0.2
MAX_TOKENS: int         = 512
SUCCESS_THRESHOLD: float = 0.6    # normalised score in [0.0, 1.0]

# --- Startup validation ---
if not HF_TOKEN:
    print(
        "[ERROR] HF_TOKEN environment variable is not set. "
        "Export your Hugging Face token before running inference.py.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

if not ENV_BASE_URL and not LOCAL_IMAGE_NAME:
    print(
        "[ERROR] Neither ENV_BASE_URL nor LOCAL_IMAGE_NAME is set.\n"
        "  • Set ENV_BASE_URL to your deployed HF Space URL, e.g.\n"
        "    export ENV_BASE_URL=https://<user>-dispatch-triage-env.hf.space\n"
        "  • Or set LOCAL_IMAGE_NAME to a local Docker image, e.g.\n"
        "    export LOCAL_IMAGE_NAME=dispatch-triage-env:latest",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Structured stdout logging — format is graded automatically
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """[START] line — emitted once per episode, before any steps."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """[STEP] line — emitted after every env.step() call."""
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """[END] line — emitted once per episode after the loop finishes."""
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = textwrap.dedent("""
    You are an AI-powered 911 dispatch operator.

    Each turn you receive:
      • ACTIVE INCIDENTS — each has an id, a plain-English caller description, and a location.
      • AVAILABLE UNITS  — each has an id and a type: ambulance | fire_truck | police.

    Your task is to choose ONE incident and ONE unit to dispatch this turn.

    Rules:
      1. Infer urgency purely from the description — there are no explicit severity numbers.
      2. Match unit type to the nature of the emergency:
           ambulance   → medical emergencies (cardiac arrest, unresponsive person, injuries)
           fire_truck  → fires, gas leaks, structural hazards
           police      → vehicle collisions, traffic control, crowd management
      3. Always prioritise the most life-threatening, time-critical incident first.
      4. If a description hints that one incident must be resolved before another can be
         safely attended (e.g. a gas leak blocking entry to an adjacent building where a
         cardiac arrest is occurring), resolve the blocking incident FIRST, even if the
         other sounds more urgent on the surface.
      5. Only dispatch to incidents and units that appear in the lists below.

    Respond with ONLY a single-line JSON object — no markdown, no preamble:
    {"incident_id": <int>, "unit_id": <int>, "reasoning": "<one concise sentence>"}
""").strip()


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

def _format_incidents(incidents) -> str:
    lines = []
    for inc in incidents:
        if not inc.resolved:
            dep_note = f" [depends_on: {inc.depends_on}]" if inc.depends_on else ""
            lines.append(
                f"  [id={inc.id}] {inc.location}{dep_note}\n"
                f"           {inc.description}"
            )
    return "\n".join(lines) if lines else "  (none remaining)"


def _format_units(units) -> str:
    lines = []
    for u in units:
        if u.available:
            lines.append(f"  [id={u.id}] {u.type}")
    return "\n".join(lines) if lines else "  (none remaining)"


def build_user_prompt(obs, step: int, last_message: str) -> str:
    return textwrap.dedent(f"""
        Step {step}  |  Dispatches so far: {obs.dispatch_count}
        Current score: {obs.score_so_far:.4f}
        Last server message: {last_message}

        ACTIVE INCIDENTS (unresolved):
        {_format_incidents(obs.incidents)}

        AVAILABLE UNITS:
        {_format_units(obs.units)}

        Choose your next dispatch. Reply with exactly:
        {{"incident_id": <int>, "unit_id": <int>, "reasoning": "<one sentence>"}}
    """).strip()


# ---------------------------------------------------------------------------
# LLM call + JSON parsing
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    user_prompt: str,
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Call the LLM and parse its JSON response.

    Returns:
        (incident_id, unit_id, raw_response)
        On any failure returns (None, None, error_description).
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw: str = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        err = f"LLM call failed: {exc}"
        print(f"[DEBUG] {err}", flush=True)
        return None, None, err

    # Strip accidental markdown fences
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Handle model responses that wrap JSON in extra text by finding the first {...}
    match = re.search(r"\{.*?\}", clean, re.DOTALL)
    if match:
        clean = match.group(0)

    try:
        parsed      = json.loads(clean)
        incident_id = int(parsed["incident_id"])
        unit_id     = int(parsed["unit_id"])
        return incident_id, unit_id, raw
    except Exception as exc:
        err = f"JSON parse failed: {exc}"
        print(f"[DEBUG] {err} | raw={raw!r}", flush=True)
        return None, None, raw


# ---------------------------------------------------------------------------
# Fallback action selector
# ---------------------------------------------------------------------------

def _fallback_action(obs) -> Optional[Tuple[int, int]]:
    """
    When the LLM response is unparseable, pick the first available
    (incident, unit) pair as a safe fallback.
    """
    available_incidents = [i for i in obs.incidents if not i.resolved]
    available_units     = [u for u in obs.units     if u.available]
    if available_incidents and available_units:
        return available_incidents[0].id, available_units[0].id
    return None


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env,          # SyncEnvClient wrapping DispatchTriageEnv
    llm_client: OpenAI,
    difficulty: str,
) -> None:
    """
    Run one full episode for the given difficulty level.

    Emits [START], one [STEP] per dispatch, and [END] to stdout.
    All lines follow the hackathon grader format exactly.
    """
    task_name = f"dispatch-{difficulty}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False

    try:
        # env.reset() returns StepResult; observation holds the initial state.
        result      = env.reset(difficulty=difficulty)
        obs         = result.observation
        last_message: str = obs.message

        for step in range(1, MAX_STEPS + 1):
            # Check terminal condition from the previous step result
            if result.done:
                break

            # Build no-information-leak prompt (no severity numbers, no type labels)
            user_prompt = build_user_prompt(obs, step, last_message)
            incident_id, unit_id, raw_response = call_llm(llm_client, user_prompt)

            # Fallback: use first available pair when LLM fails to parse
            error_msg: Optional[str] = None
            if incident_id is None or unit_id is None:
                error_msg = "JSON parse failed — using fallback action"
                fallback  = _fallback_action(obs)
                if fallback is None:
                    print("[DEBUG] No valid fallback action — ending episode.", flush=True)
                    break
                incident_id, unit_id = fallback

            action     = DispatchTriageAction(incident_id=incident_id, unit_id=unit_id)
            action_str = f"dispatch(incident_id={incident_id},unit_id={unit_id})"

            # Execute action on the environment server
            result       = env.step(action)
            obs          = result.observation
            reward       = result.reward if result.reward is not None else 0.0
            done         = result.done
            last_message = obs.message

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Episode ended — determine success from the final normalised score
        final_score = rewards[-1] if rewards else 0.0
        success     = final_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error ({difficulty}): {exc}", flush=True)
        traceback.print_exc(file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Build the async client, then wrap it in the synchronous adapter.
    # Using .sync() is required — EnvClient methods are async coroutines.
    if ENV_BASE_URL:
        async_env = DispatchTriageEnv(base_url=ENV_BASE_URL)
    else:
        async_env = DispatchTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)

    env = async_env.sync()

    try:
        with env:
            for difficulty in DIFFICULTIES:
                run_episode(env, llm_client, difficulty)
    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
