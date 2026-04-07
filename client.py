# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatch Triage Environment Client (synchronous-friendly wrapper)."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# Support two run modes:
#   1. Installed as a package  → relative imports (.models)
#   2. Imported directly from repo root (e.g. by inference.py) → absolute imports
try:
    from .models import (
        DispatchTriageAction,
        DispatchTriageObservation,
        DispatchTriageState,
        Incident,
        Unit,
    )
except ImportError:
    from models import (                    # type: ignore[no-redef]
        DispatchTriageAction,
        DispatchTriageObservation,
        DispatchTriageState,
        Incident,
        Unit,
    )


class DispatchTriageEnv(
    EnvClient[DispatchTriageAction, DispatchTriageObservation, DispatchTriageState]
):
    """
    Async client for the Dispatch Triage Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated session on the server.

    For synchronous use (e.g. in inference.py), call `.sync()` to get a
    SyncEnvClient wrapper::

        env = DispatchTriageEnv(base_url="http://localhost:8000").sync()
        with env:
            result = env.reset(difficulty="medium")
            result = env.step(DispatchTriageAction(incident_id=0, unit_id=1))
            print(result.observation.message)

    For async use::

        async with DispatchTriageEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(difficulty="hard")
            result = await env.step(DispatchTriageAction(incident_id=1, unit_id=0))
    """

    def _step_payload(self, action: DispatchTriageAction) -> Dict:
        """Convert DispatchTriageAction to JSON-serialisable dict."""
        return {
            "incident_id": action.incident_id,
            "unit_id":     action.unit_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[DispatchTriageObservation]:
        """
        Parse a raw server response dict into StepResult[DispatchTriageObservation].

        The openenv HTTP server may send the observation fields either:
          • Nested under an "observation" key  (StepResult envelope), or
          • Flat at the top level              (direct Observation serialisation)
        We handle both cases.
        """
        # Prefer the nested "observation" dict; fall back to top-level payload.
        obs_data: Dict = payload.get("observation") or payload

        incidents = [
            Incident(
                id=i["id"],
                description=i["description"],
                location=i["location"],
                assigned_unit_id=i.get("assigned_unit_id"),
                dispatch_rank=i.get("dispatch_rank"),
                resolved=i.get("resolved", False),
                depends_on=i.get("depends_on", []),
            )
            for i in obs_data.get("incidents", [])
        ]

        units = [
            Unit(id=u["id"], type=u["type"], available=u.get("available", True))
            for u in obs_data.get("units", [])
        ]

        # done / reward live at the top-level payload in the StepResult envelope
        done   = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward", 0.0))

        observation = DispatchTriageObservation(
            done=done,
            reward=reward,
            incidents=incidents,
            units=units,
            dispatch_count=obs_data.get("dispatch_count", 0),
            score_so_far=obs_data.get("score_so_far", 0.0),
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into DispatchTriageState."""
        return DispatchTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            difficulty=payload.get("difficulty", "easy"),
            total_incidents=payload.get("total_incidents", 0),
            total_units=payload.get("total_units", 0),
            max_possible_score=payload.get("max_possible_score", 1.0),
        )
