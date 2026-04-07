# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Dispatch Triage Environment.

The agent receives natural-language incident descriptions and a pool of
available units. It must infer urgency and correct unit type from context
alone — no severity numbers or incident type labels are exposed.
"""

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain constants  (internal use only — never sent to the agent)
# ---------------------------------------------------------------------------

UnitType   = Literal["fire_truck", "ambulance", "police"]
Difficulty = Literal["easy", "medium", "hard"]

# Correct unit-type mapping used internally for scoring.
# The agent must infer this from the incident description.
CORRECT_UNIT: dict = {
    "fire":           "fire_truck",
    "cardiac_arrest": "ambulance",
    "car_crash":      "police",
    "gas_leak":       "fire_truck",
}

# Penalty added to mismatch_penalties when an agent dispatches an incident
# whose dependency has not yet been resolved (hard-mode cascade).
# This is a PENALTY subtracted from the running score, not a severity boost.
CASCADE_SEVERITY_BOOST: int = 4


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Incident(BaseModel):
    """
    A single emergency call visible to the agent.

    Severity and incident type are intentionally hidden — the agent must
    reason about urgency and appropriate unit from the natural-language
    `description` alone.
    """
    id: int
    description: str                          # Plain-English caller report — only signal the agent gets
    location: str                             # Human-readable label, e.g. "Block 4A"
    assigned_unit_id: Optional[int] = None   # None until dispatched
    dispatch_rank: Optional[int] = None      # 1 = dispatched first this episode
    resolved: bool = False
    depends_on: List[int] = Field(default_factory=list)  # Hard mode: ids of blocking incidents


class Unit(BaseModel):
    """An emergency unit in the dispatch pool."""
    id: int
    type: UnitType
    available: bool = True


# ---------------------------------------------------------------------------
# Action / Observation / State
# ---------------------------------------------------------------------------

class DispatchTriageAction(Action):
    """
    The agent's decision each step: send unit `unit_id` to handle `incident_id`.
    Repeated until all units are deployed or all incidents are resolved.
    """
    incident_id: int
    unit_id: int


class DispatchTriageObservation(Observation):
    """
    Everything the agent can see after each step.

    Inherited from Observation base:
        done:   bool
        reward: Optional[float]
    """
    incidents: List[Incident]
    units: List[Unit]
    dispatch_count: int
    message: str
    score_so_far: float


class DispatchTriageState(State):
    """
    Internal state persisted across steps.

    Inherited from State base:
        episode_id: Optional[str]
        step_count: int
    """
    difficulty: Difficulty = "easy"
    total_incidents: int = 0
    total_units: int = 0
    max_possible_score: float = 1.0
