# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Dispatch Triage Environment.

Endpoints:
    POST /reset  — reset the environment (accepts difficulty kwarg)
    POST /step   — execute one dispatch action
    GET  /state  — read current episode state
    GET  /schema — action / observation JSON schemas
    WS   /ws     — persistent WebSocket session (used by EnvClient)
    GET  /health — liveness probe for Docker / HF Spaces

Usage (dev):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Usage (prod):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with: pip install 'openenv-core[core]>=0.2.2'"
    ) from exc

# Support two run modes:
#   1. Installed as a package  → relative imports work
#   2. Run from repo root via `uvicorn server.app:app`  → absolute imports
try:
    from ..models import DispatchTriageAction, DispatchTriageObservation
    from .Dispatch_triage_env_environment import DispatchTriageEnvironment
except ImportError:
    from models import DispatchTriageAction, DispatchTriageObservation          # type: ignore[no-redef]
    from server.Dispatch_triage_env_environment import DispatchTriageEnvironment  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = create_app(
    DispatchTriageEnvironment,     # class — enables concurrent sessions
    DispatchTriageAction,
    DispatchTriageObservation,
    env_name="Dispatch_triage_env",
    max_concurrent_envs=4,         # allow up to 4 simultaneous WebSocket sessions
)


# ---------------------------------------------------------------------------
# Entry point (uv run / python -m)
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the uvicorn server directly.

    Examples::

        uv run --project . server          # via pyproject.toml scripts entry
        python -m server.app               # direct module execution
        uvicorn server.app:app --reload    # dev with hot-reload
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dispatch Triage Env server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
