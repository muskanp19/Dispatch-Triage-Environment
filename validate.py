#!/usr/bin/env python3
"""
validate.py — Pre-Submission Validation Script
===============================================
Run this before submitting to catch any disqualifying issues.

Usage:
    python validate.py                      # full validation (no server needed)
    python validate.py --url <ENV_BASE_URL> # also ping a live server

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"

ROOT = Path(__file__).parent.resolve()

results: List[Tuple[str, bool, str]] = []   # (name, passed, detail)


def check(name: str) -> Callable:
    """Decorator — registers a check function and records its result."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                passed, detail = fn(*args, **kwargs)
            except Exception as exc:
                passed, detail = False, f"Exception: {exc}"
            results.append((name, passed, detail))
            status = PASS if passed else FAIL
            print(f"  {status}  {name}")
            if detail:
                prefix = "       "
                for line in detail.splitlines():
                    print(f"{prefix}{line}")
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@check("openenv.yaml exists and has required fields")
def check_openenv_yaml():
    p = ROOT / "openenv.yaml"
    if not p.exists():
        return False, "openenv.yaml not found"
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(p.read_text())
    except ImportError:
        # Fallback: naive key check
        text = p.read_text()
        required = ["spec_version", "name", "app", "port", "tasks"]
        missing  = [k for k in required if k not in text]
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "yaml library not installed — basic text check passed"
    required = ["spec_version", "name", "app", "port", "tasks"]
    missing  = [k for k in required if k not in data]
    if missing:
        return False, f"Missing keys in openenv.yaml: {missing}"
    tasks = data.get("tasks", [])
    if len(tasks) < 3:
        return False, f"Need at least 3 tasks, found {len(tasks)}"
    return True, f"spec_version={data['spec_version']} | tasks={[t['name'] for t in tasks]}"


@check("Dockerfile exists")
def check_dockerfile():
    p = ROOT / "Dockerfile"
    if not p.exists():
        return False, "Dockerfile not found at project root"
    text = p.read_text()
    checks = {
        "FROM":    "FROM" in text,
        "COPY":    "COPY" in text,
        "CMD":     "CMD"  in text,
        "port 8000": "8000" in text,
    }
    missing = [k for k, v in checks.items() if not v]
    if missing:
        return False, f"Dockerfile may be incomplete — missing: {missing}"
    return True, "Dockerfile is valid"


@check("inference.py exists at project root")
def check_inference_exists():
    p = ROOT / "inference.py"
    if not p.exists():
        return False, "inference.py not found — must be at the project root"
    return True, str(p)


@check("inference.py has [START]/[STEP]/[END] markers")
def check_inference_format():
    p = ROOT / "inference.py"
    if not p.exists():
        return False, "inference.py not found"
    text = p.read_text()
    markers = {
        "[START]":  "[START]" in text,
        "[STEP]":   "[STEP]"  in text,
        "[END]":    "[END]"   in text,
        "task=":    "task="   in text,
        "reward=":  "reward=" in text,
        "success=": "success=" in text,
        "steps=":   "steps="  in text,
        "rewards=": "rewards=" in text,
    }
    missing = [k for k, v in markers.items() if not v]
    if missing:
        return False, f"Missing log markers: {missing}"
    return True, "All required log markers present"


@check("inference.py uses OpenAI client")
def check_openai_usage():
    p = ROOT / "inference.py"
    if not p.exists():
        return False, "inference.py not found"
    text = p.read_text()
    if "from openai import OpenAI" not in text and "import openai" not in text:
        return False, "OpenAI client import not found"
    if "API_BASE_URL" not in text or "MODEL_NAME" not in text or "HF_TOKEN" not in text:
        return False, "Required env vars (API_BASE_URL / MODEL_NAME / HF_TOKEN) not referenced"
    return True, "OpenAI client + required env vars found"


@check("models.py imports and instantiates correctly")
def check_models():
    sys.path.insert(0, str(ROOT))
    try:
        import importlib
        models = importlib.import_module("models")
        action = models.DispatchTriageAction(incident_id=0, unit_id=1)
        obs    = models.DispatchTriageObservation(
            done=False, reward=0.0,
            incidents=[], units=[],
            dispatch_count=0, message="test", score_so_far=0.0,
        )
        state  = models.DispatchTriageState()
        return True, f"Action={action} | State difficulty={state.difficulty}"
    except Exception as exc:
        return False, str(exc)


@check("Environment resets and steps correctly (all 3 difficulties)")
def check_environment_logic():
    sys.path.insert(0, str(ROOT))
    try:
        env_mod = importlib.import_module("server.Dispatch_triage_env_environment")
        models  = importlib.import_module("models")
        Env     = env_mod.DispatchTriageEnvironment
        Action  = models.DispatchTriageAction
    except Exception as exc:
        return False, f"Import failed: {exc}"

    report = []
    for difficulty in ["easy", "medium", "hard"]:
        try:
            env = Env()
            obs = env.reset(difficulty=difficulty)
            assert not obs.done, "reset() returned done=True"
            assert len(obs.incidents) > 0, "no incidents in observation"
            assert len(obs.units) > 0, "no units in observation"
            assert 0.0 <= obs.score_so_far <= 1.0, f"score_so_far out of range: {obs.score_so_far}"

            # Take one valid step
            inc  = next(i for i in obs.incidents if not i.resolved)
            unit = next(u for u in obs.units if u.available)
            obs2 = env.step(Action(incident_id=inc.id, unit_id=unit.id))
            assert 0.0 <= obs2.score_so_far <= 1.0, \
                f"score_so_far out of range after step: {obs2.score_so_far}"
            report.append(f"{difficulty}: score={obs2.score_so_far:.4f} done={obs2.done}")
        except Exception as exc:
            return False, f"{difficulty} failed: {exc}"

    return True, " | ".join(report)


@check("Reward stays in [0.0, 1.0] for all difficulties (full episode)")
def check_reward_range():
    sys.path.insert(0, str(ROOT))
    try:
        env_mod = importlib.import_module("server.Dispatch_triage_env_environment")
        models  = importlib.import_module("models")
        Env     = env_mod.DispatchTriageEnvironment
        Action  = models.DispatchTriageAction
    except Exception as exc:
        return False, f"Import failed: {exc}"

    bad = []
    for difficulty in ["easy", "medium", "hard"]:
        env = Env()
        obs = env.reset(difficulty=difficulty)
        for _ in range(20):
            if obs.done:
                break
            avail_incs  = [i for i in obs.incidents if not i.resolved]
            avail_units = [u for u in obs.units if u.available]
            if not avail_incs or not avail_units:
                break
            obs = env.step(Action(incident_id=avail_incs[0].id, unit_id=avail_units[0].id))
            r = obs.score_so_far
            if not (0.0 <= r <= 1.0):
                bad.append(f"{difficulty}: reward={r}")
    if bad:
        return False, f"Out-of-range rewards: {bad}"
    return True, "All rewards in [0.0, 1.0] across easy/medium/hard"


@check("Cascade penalty reduces score (hard mode correctness)")
def check_cascade_penalty():
    sys.path.insert(0, str(ROOT))
    try:
        env_mod = importlib.import_module("server.Dispatch_triage_env_environment")
        models  = importlib.import_module("models")
        Env     = env_mod.DispatchTriageEnvironment
        Action  = models.DispatchTriageAction
    except Exception as exc:
        return False, f"Import failed: {exc}"

    # Optimal: resolve gas leak (id=1) before cardiac (id=0)
    env_opt = Env()
    obs = env_opt.reset(difficulty="hard")
    obs = env_opt.step(Action(incident_id=2, unit_id=1))   # fire  → fire_truck
    obs = env_opt.step(Action(incident_id=1, unit_id=0))   # gas   → ambulance (wrong type but no cascade)
    obs = env_opt.step(Action(incident_id=0, unit_id=2))   # cardiac after gas resolved
    optimal_score = obs.score_so_far

    # Sub-optimal: dispatch cardiac (id=0) before gas (id=1)
    env_bad = Env()
    obs2 = env_bad.reset(difficulty="hard")
    obs2 = env_bad.step(Action(incident_id=2, unit_id=1))  # fire  → fire_truck
    obs2 = env_bad.step(Action(incident_id=0, unit_id=0))  # cardiac BEFORE gas → cascade penalty
    obs2 = env_bad.step(Action(incident_id=1, unit_id=2))  # gas after cardiac
    bad_score = obs2.score_so_far

    if bad_score >= optimal_score:
        return False, (
            f"Cascade penalty not working: wrong order score ({bad_score:.4f}) "
            f">= correct order score ({optimal_score:.4f})"
        )
    return True, (
        f"Correct order: {optimal_score:.4f} | Wrong order (cascade): {bad_score:.4f} — "
        "penalty is functioning correctly"
    )


@check("3+ tasks defined with distinct difficulty levels")
def check_task_count():
    sys.path.insert(0, str(ROOT))
    try:
        env_mod   = importlib.import_module("server.Dispatch_triage_env_environment")
        scenarios = env_mod.SCENARIOS
        diffs     = list(scenarios.keys())
        if len(diffs) < 3:
            return False, f"Only {len(diffs)} difficulty levels: {diffs}"
        for d, data in scenarios.items():
            n_inc  = len(data["incidents"])
            n_unit = len(data["units"])
            n_meta = len(data["_meta"])
            if n_inc != n_meta:
                return False, f"{d}: incidents ({n_inc}) vs _meta ({n_meta}) count mismatch"
            if n_inc < 3 or n_unit < 3:
                return False, f"{d}: need ≥3 incidents and ≥3 units, got {n_inc}/{n_unit}"
        return True, f"Difficulties: {diffs} | incidents per level: {[len(v['incidents']) for v in scenarios.values()]}"
    except Exception as exc:
        return False, str(exc)


@check("pyproject.toml has required dependencies")
def check_pyproject():
    p = ROOT / "pyproject.toml"
    if not p.exists():
        return False, "pyproject.toml not found"
    text = p.read_text()
    required = ["openenv-core", "openai"]
    missing  = [dep for dep in required if dep not in text]
    if missing:
        return False, f"Missing dependencies: {missing}"
    return True, f"Found: {required}"


@check("README.md represents out dispatch triage environment")
def check_readme():
    p = ROOT / "README.md"
    if not p.exists():
        return False, "README.md not found"
    text = p.read_text().lower()
    # Must reference dispatch-specific terms
    required_terms = ["incident", "dispatch", "unit", "ambulance", "reward"]
    missing = [t for t in required_terms if t not in text]
    if missing:
        return False, f"README missing domain terms: {missing} (may still be echo template)"
    # Must NOT still contain echo-env boilerplate
    bad_terms = ["echoed_message", "message_length", "echo environment"]
    present   = [t for t in bad_terms if t in text]
    if present:
        return False, f"README still contains echo-environment template text: {present}"
    return True, "README covers the dispatch environment correctly"


# ---------------------------------------------------------------------------
# Optional: live server ping
# ---------------------------------------------------------------------------

def check_live_server(url: str) -> None:
    """Ping a running server and test reset() via HTTP."""
    import urllib.request
    import urllib.error

    print(f"\n{BOLD}[Live Server Check] {url}{RESET}")

    # Health check
    try:
        name = "GET /health returns 200"
        resp = urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=10)
        if resp.status == 200:
            results.append((name, True, f"status={resp.status}"))
            print(f"  {PASS}  {name}")
        else:
            results.append((name, False, f"status={resp.status}"))
            print(f"  {FAIL}  {name}")
    except Exception as exc:
        results.append(("GET /health returns 200", False, str(exc)))
        print(f"  {FAIL}  GET /health returns 200 — {exc}")

    # POST /reset
    try:
        name = "POST /reset responds correctly"
        payload = json.dumps({"difficulty": "easy"}).encode()
        req     = urllib.request.Request(
            f"{url.rstrip('/')}/reset",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp    = urllib.request.urlopen(req, timeout=15)
        body    = json.loads(resp.read())
        # Accept either flat observation or nested StepResult
        obs     = body.get("observation", body)
        has_inc = "incidents" in obs and len(obs["incidents"]) > 0
        has_uni = "units" in obs and len(obs["units"]) > 0
        if has_inc and has_uni:
            results.append((name, True, f"incidents={len(obs['incidents'])} units={len(obs['units'])}"))
            print(f"  {PASS}  {name}")
        else:
            results.append((name, False, f"Response missing incidents/units: {list(obs.keys())}"))
            print(f"  {FAIL}  {name}")
    except Exception as exc:
        results.append(("POST /reset responds correctly", False, str(exc)))
        print(f"  {FAIL}  POST /reset responds correctly — {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Pre-submission validation for Dispatch Triage Env")
    parser.add_argument("--url", help="Live server URL to ping (optional)", default=None)
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Dispatch Triage Env — Pre-Submission Validation{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # Run all registered checks
    print(f"{BOLD}[Static / Logic Checks]{RESET}")
    check_openenv_yaml()
    check_dockerfile()
    check_inference_exists()
    check_inference_format()
    check_openai_usage()
    check_pyproject()
    check_readme()

    print(f"\n{BOLD}[Environment Logic Checks]{RESET}")
    check_models()
    check_environment_logic()
    check_reward_range()
    check_cascade_penalty()
    check_task_count()

    # Optional live server
    if args.url:
        check_live_server(args.url)

    # Summary
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    failed = [(n, d) for n, ok, d in results if not ok]

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Results: {passed}/{total} checks passed{RESET}")

    if failed:
        print(f"\n{RED}{BOLD}  FAILED CHECKS:{RESET}")
        for name, detail in failed:
            print(f"    {RED}✗ {name}{RESET}")
            if detail:
                for line in detail.splitlines():
                    print(f"        {line}")
        print(f"\n{RED}Submission is NOT ready. Fix the issues above.{RESET}\n")
        return 1
    else:
        print(f"\n{GREEN}{BOLD}  All checks passed! Submission is ready.{RESET}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
