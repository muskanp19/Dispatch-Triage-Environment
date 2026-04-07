---
title: Dispatch Triage Env — 911 Emergency Dispatch
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - dispatch
  - triage
---

# 🚨 Dispatch Triage Env

**A multi-incident 911 emergency dispatch triage environment for OpenEnv.**

The agent receives simultaneous natural-language emergency call descriptions
and a finite pool of typed response units. It must infer urgency and correct
unit type from context alone — no explicit severity numbers or incident type
labels are ever exposed — then sequence dispatch decisions to maximise a
severity-weighted score.

---

## Environment Overview

| Property         | Value |
|------------------|-------|
| Action space     | `{incident_id: int, unit_id: int}` |
| Observation      | List of incidents (natural-language), list of units, score, message |
| Reward range     | `[0.0, 1.0]` (normalised, dense — updated after every dispatch) |
| Tasks            | `dispatch-easy`, `dispatch-medium`, `dispatch-hard` |
| Max steps        | 3 (easy), 3 (medium), 3 (hard) per episode |

---

## Difficulty Levels

| Level  | Incidents | Units | Challenge |
|--------|-----------|-------|-----------|
| easy   | 3         | 3     | Unambiguous priority — match every unit correctly |
| medium | 5         | 3     | Must sacrifice 2 calls; unit-type matching is harder |
| hard   | 7         | 3     | Cascading dependency: one incident blocks safe entry to another |

---

## Action / Observation Spaces

### Action — `DispatchTriageAction`
```python
class DispatchTriageAction(Action):
    incident_id: int   # id of the incident to dispatch to
    unit_id:     int   # id of the unit to send
```

### Observation — `DispatchTriageObservation`
```python
class DispatchTriageObservation(Observation):
    incidents:      List[Incident]   # current incident list (with resolved flags)
    units:          List[Unit]       # current unit pool (with available flags)
    dispatch_count: int              # how many dispatches have been made this episode
    score_so_far:   float            # running normalised score in [0.0, 1.0]
    message:        str              # human-readable update from the last step
    done:           bool             # True when all units are deployed or all incidents resolved
    reward:         float            # same as score_so_far (dense reward)
```

Each `Incident` exposes only `id`, `description`, `location`, `resolved`, and `depends_on`
(hard mode). Severity values and incident type labels are **never** sent to the agent.

### Unit types
| Type        | Appropriate for                                    |
|-------------|---------------------------------------------------|
| `ambulance` | Medical emergencies (cardiac arrest, unconscious) |
| `fire_truck`| Fires, gas leaks, structural hazards              |
| `police`    | Vehicle collisions, traffic control               |

---

## Scoring

```
raw_score       += incident_severity / dispatch_rank     (for each dispatch)
penalty         += 1.5    (if wrong unit type)
penalty         += 4      (cascade violation — hard mode dependency violated)
running_score    = max(0, raw_score - penalty)
final_score      = min(1.0, running_score / max_possible_score)
```

**Higher severity dispatched earlier = higher score.**  
**Wrong unit type or cascade violation = score deducted.**

---

## Quick Start

### Connect to a deployed HF Space

```python
from client import DispatchTriageEnv
from models import DispatchTriageAction

env = DispatchTriageEnv(base_url="https://<user>-dispatch-triage-env.hf.space").sync()

with env:
    result = env.reset(difficulty="medium")
    obs    = result.observation
    print(obs.message)

    # Dispatch ambulance (unit 0) to first unresolved incident
    unresolved = [i for i in obs.incidents if not i.resolved]
    available  = [u for u in obs.units     if u.available]

    result = env.step(DispatchTriageAction(
        incident_id=unresolved[0].id,
        unit_id=available[0].id,
    ))
    print(f"Score: {result.observation.score_so_far:.4f}")
    print(f"Done:  {result.done}")
```

### Run inference with an LLM agent

```bash
export HF_TOKEN=hf_...
export ENV_BASE_URL=https://<user>-dispatch-triage-env.hf.space
python inference.py
```

---

## Local Development

### Run the server locally

```bash
pip install "openenv-core[core]>=0.2.2" openai
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Run the pre-submission validator

```bash
python validate.py                             # static + logic checks
python validate.py --url http://localhost:8000 # also ping the live server
```

### Build and run with Docker

```bash
docker build -t dispatch-triage-env:latest .
docker run -p 8000:8000 dispatch-triage-env:latest
```

---

## Deploy to Hugging Face Spaces

```bash
pip install "openenv-core[core]>=0.2.2"
openenv push --repo-id <your-username>/dispatch-triage-env
```

After deployment your space will expose:
- **`/web`** — interactive UI
- **`/docs`** — OpenAPI / Swagger documentation
- **`/health`** — liveness probe
- **`/reset`** — `POST` to start a new episode
- **`/step`** — `POST` to execute a dispatch action
- **`/state`** — `GET` current episode state
- **`/ws`** — WebSocket for persistent sessions (used by `EnvClient`)

---

## Project Structure

```
Dispatch_triage_env/
├── __init__.py                              # Package exports
├── openenv.yaml                             # OpenEnv manifest (spec, tasks)
├── pyproject.toml                           # Project metadata & dependencies
├── Dockerfile                               # Container image definition
├── README.md                                # This file
├── .env                                     # Local env var template (do not commit secrets)
├── models.py                                # Action / Observation / State models
├── client.py                                # DispatchTriageEnv async+sync client
├── inference.py                             # LLM agent runner (hackathon submission)
├── validate.py                              # Pre-submission validation script
└── server/
    ├── __init__.py                          # Server module exports
    ├── Dispatch_triage_env_environment.py   # Core environment logic
    └── app.py                               # FastAPI application
```

---

## Environment Variables

| Variable          | Required | Default                              | Purpose |
|-------------------|----------|--------------------------------------|---------|
| `HF_TOKEN`        | **Yes**  | —                                    | HuggingFace API token for LLM calls |
| `API_BASE_URL`    | No       | `https://router.huggingface.co/v1`   | LLM API endpoint |
| `MODEL_NAME`      | No       | `Qwen/Qwen2.5-72B-Instruct`          | Model identifier |
| `ENV_BASE_URL`    | No*      | —                                    | Deployed HF Space URL |
| `LOCAL_IMAGE_NAME`| No*      | —                                    | Docker image (fallback if no ENV_BASE_URL) |

\* At least one of `ENV_BASE_URL` or `LOCAL_IMAGE_NAME` is required to run `inference.py`.

---

## Pre-Submission Checklist

Run `python validate.py` and ensure all checks pass:

- [ ] `openenv.yaml` exists and has `spec_version`, `name`, `app`, `port`, `tasks` (≥3)
- [ ] `Dockerfile` exists and references port 8000
- [ ] `inference.py` at project root
- [ ] `inference.py` emits `[START]`, `[STEP]`, `[END]` log format
- [ ] `inference.py` uses OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] All rewards in `[0.0, 1.0]` across all difficulties
- [ ] Cascade penalty reduces score for wrong dependency order (hard mode)
- [ ] HF Space responds to `GET /health` with 200
- [ ] `POST /reset` returns observations with incidents and units
