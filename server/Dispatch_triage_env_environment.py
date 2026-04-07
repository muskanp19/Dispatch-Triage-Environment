# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dispatch Triage Environment Implementation.

The agent receives simultaneous emergency call descriptions (natural language only)
and a finite pool of units. It must infer urgency and correct unit type from context,
then sequence dispatch decisions to maximise a severity-weighted score.

Severity scores and incident types are stored internally and never sent to the agent.

Scoring formula (per dispatch):
    raw_score       += incident_severity / dispatch_rank
    mismatch_penalty += 1.5  (if wrong unit type)
    cascade_penalty  += CASCADE_SEVERITY_BOOST  (if dependency not yet resolved)
    running_score    = max(0, raw_score - mismatch_penalty - cascade_penalty)
    normalised       = min(1.0, running_score / max_possible_score)
"""

import uuid
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        CASCADE_SEVERITY_BOOST,
        CORRECT_UNIT,
        DispatchTriageAction,
        DispatchTriageObservation,
        DispatchTriageState,
        Incident,
        Unit,
    )
except ImportError:
    from models import (
        CASCADE_SEVERITY_BOOST,
        CORRECT_UNIT,
        DispatchTriageAction,
        DispatchTriageObservation,
        DispatchTriageState,
        Incident,
        Unit,
    )


# ---------------------------------------------------------------------------
# Scenario definitions
#
# Each incident exposes only: id, description, location (+ depends_on in hard).
# Severity and type live in _meta — internal only, never sent to the agent.
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, dict] = {

    # ------------------------------------------------------------------
    # EASY — 3 incidents, 3 units
    # Priority order is unambiguous to any reasonable reader.
    # ------------------------------------------------------------------
    "easy": {
        "incidents": [
            Incident(
                id=0,
                description=(
                    "Caller reports an elderly woman collapsed at home, unresponsive "
                    "with no breathing detected. Family member is performing CPR. "
                    "Situation is critical."
                ),
                location="Block 2A",
            ),
            Incident(
                id=1,
                description=(
                    "Two-vehicle collision at a busy intersection. One driver is "
                    "conscious but has visible injuries and cannot exit the vehicle. "
                    "Traffic is blocked."
                ),
                location="Block 7C",
            ),
            Incident(
                id=2,
                description=(
                    "Resident reports light smoke coming from a kitchen window. "
                    "No visible flames. Occupants have self-evacuated to the street."
                ),
                location="Block 1B",
            ),
        ],
        "units": [
            Unit(id=0, type="ambulance"),
            Unit(id=1, type="police"),
            Unit(id=2, type="fire_truck"),
        ],
        "_meta": {
            0: {"severity": 9, "type": "cardiac_arrest"},
            1: {"severity": 5, "type": "car_crash"},
            2: {"severity": 3, "type": "fire"},
        },
    },

    # ------------------------------------------------------------------
    # MEDIUM — 5 incidents, 3 units
    # Agent must sacrifice the two least urgent calls and correctly
    # match unit types without explicit type labels.
    # ------------------------------------------------------------------
    "medium": {
        "incidents": [
            Incident(
                id=0,
                description=(
                    "Male in his 50s collapsed in an office lobby, coworkers report "
                    "no pulse. An AED is on site but no trained personnel are present. "
                    "Time is critical."
                ),
                location="Block 3A",
            ),
            Incident(
                id=1,
                description=(
                    "Fire on the third floor of an occupied apartment complex. "
                    "Multiple residents are still inside. Flames are visible from "
                    "the street and spreading rapidly."
                ),
                location="Block 5D",
            ),
            Incident(
                id=2,
                description=(
                    "Strong gas odour reported across an entire city block. Multiple "
                    "independent callers. Children and elderly residents are present "
                    "in adjacent buildings. Risk of ignition is high."
                ),
                location="Block 2C",
            ),
            Incident(
                id=3,
                description=(
                    "Rear-end collision on the highway. Both drivers have exited "
                    "their vehicles. One is complaining of neck pain but is standing "
                    "and alert."
                ),
                location="Block 9B",
            ),
            Incident(
                id=4,
                description=(
                    "Minor fender-bender in a parking lot. No injuries reported. "
                    "Vehicles are blocking two parking spaces. Drivers are exchanging "
                    "information calmly."
                ),
                location="Block 11E",
            ),
        ],
        "units": [
            Unit(id=0, type="ambulance"),
            Unit(id=1, type="fire_truck"),
            Unit(id=2, type="police"),
        ],
        "_meta": {
            0: {"severity": 9, "type": "cardiac_arrest"},
            1: {"severity": 8, "type": "fire"},
            2: {"severity": 7, "type": "gas_leak"},
            3: {"severity": 4, "type": "car_crash"},
            4: {"severity": 2, "type": "car_crash"},
        },
    },

    # ------------------------------------------------------------------
    # HARD — 7 incidents, 3 units, cascading dependency
    #
    # Incident 0 (cardiac arrest at Block 4A) depends_on incident 1
    # (gas leak at Block 4B). Descriptions hint at proximity and unsafe
    # entry. Optimal strategy: resolve gas leak FIRST even though the
    # cardiac call sounds more urgent — dispatching without resolving the
    # dependency triggers CASCADE_SEVERITY_BOOST as a score penalty.
    # ------------------------------------------------------------------
    "hard": {
        "incidents": [
            Incident(
                id=0,
                description=(
                    "Elderly resident unresponsive in a ground-floor apartment on "
                    "Block 4A. Neighbours report a strong gas smell in the same "
                    "building. First responders may be unable to enter safely until "
                    "the gas situation nearby is stabilised."
                ),
                location="Block 4A",
                depends_on=[1],
            ),
            Incident(
                id=1,
                description=(
                    "Utility workers have struck a gas main on Block 4B, immediately "
                    "adjacent to Block 4A. Strong odour reported, risk of ignition. "
                    "A medical emergency has been reported in the neighbouring building."
                ),
                location="Block 4B",
            ),
            Incident(
                id=2,
                description=(
                    "Active fire on the upper floors of a commercial building. "
                    "Sprinkler system is failing. Multiple people are reported trapped "
                    "and unable to use the stairwells."
                ),
                location="Block 9C",
            ),
            Incident(
                id=3,
                description=(
                    "Multi-vehicle pileup on the ring road. One car is overturned "
                    "with the driver trapped inside. Bystanders are attempting "
                    "extraction — risk of further injury."
                ),
                location="Block 2D",
            ),
            Incident(
                id=4,
                description=(
                    "Collision at a roundabout. One vehicle mounted the kerb. "
                    "Driver is conscious but disoriented, possibly concussed. "
                    "No other casualties reported."
                ),
                location="Block 6E",
            ),
            Incident(
                id=5,
                description=(
                    "Rubbish fire spreading toward a wooden fence line bordering "
                    "a residential garden. No structures involved yet. Caller is "
                    "monitoring from a safe distance."
                ),
                location="Block 1F",
            ),
            Incident(
                id=6,
                description=(
                    "Elderly man reports mild chest discomfort while walking. "
                    "He is conscious, talking clearly, and has sat down on a bench. "
                    "No collapse or loss of consciousness."
                ),
                location="Block 12G",
            ),
        ],
        "units": [
            Unit(id=0, type="ambulance"),
            Unit(id=1, type="fire_truck"),
            Unit(id=2, type="police"),
        ],
        "_meta": {
            0: {"severity": 7, "type": "cardiac_arrest"},
            1: {"severity": 6, "type": "gas_leak"},
            2: {"severity": 8, "type": "fire"},
            3: {"severity": 5, "type": "car_crash"},
            4: {"severity": 4, "type": "car_crash"},
            5: {"severity": 3, "type": "fire"},
            6: {"severity": 2, "type": "cardiac_arrest"},
        },
    },
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_max_possible_score(meta: Dict[int, dict], n_units: int) -> float:
    """
    Upper-bound score: highest-severity incidents perfectly dispatched in
    descending severity order with correct unit types and no penalties.
    """
    severities = sorted(
        [v["severity"] for v in meta.values()], reverse=True
    )[:n_units]
    return sum(sev / (rank + 1) for rank, sev in enumerate(severities))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DispatchTriageEnvironment(Environment):
    """
    Multi-incident 911 dispatch triage environment.

    The agent receives natural-language emergency call descriptions and a
    finite pool of typed units. It must infer urgency and correct unit type
    from descriptions alone, then dispatch in an order that maximises a
    severity-weighted score.

    Severity and incident types are hidden from the agent — stored in _meta only.

    Difficulty levels:
        easy   — 3 incidents, 3 units, unambiguous priority.
        medium — 5 incidents, 3 units, agent must sacrifice 2 calls.
        hard   — 7 incidents, 3 units, cascading dependency.

    Scoring:
        raw_score      += severity / dispatch_rank        (for each dispatch)
        penalty        += 1.5  for each wrong unit type
        penalty        += CASCADE_SEVERITY_BOOST  for each cascade violation
        final_score     = max(0, raw_score - penalty) / max_possible_score
                          clamped to [0.0, 1.0]
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._incidents: List[Incident] = []
        self._units: List[Unit] = []
        self._meta: Dict[int, dict] = {}
        self._dispatch_count: int = 0
        self._raw_score: float = 0.0
        self._total_penalties: float = 0.0
        self._max_possible: float = 1.0
        self._state = DispatchTriageState()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed=None,
        episode_id: Optional[str] = None,
        difficulty: str = "easy",
        **kwargs,
    ) -> DispatchTriageObservation:
        """Start a new episode.  difficulty: 'easy' | 'medium' | 'hard'"""
        if difficulty not in SCENARIOS:
            difficulty = "easy"

        scenario = SCENARIOS[difficulty]

        self._incidents = [inc.model_copy(deep=True) for inc in scenario["incidents"]]
        self._units     = [u.model_copy(deep=True)   for u   in scenario["units"]]
        self._meta      = deepcopy(scenario["_meta"])

        self._dispatch_count  = 0
        self._raw_score       = 0.0
        self._total_penalties = 0.0
        self._max_possible    = _compute_max_possible_score(self._meta, len(self._units))

        self._state = DispatchTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            difficulty=difficulty,          # type: ignore[arg-type]
            total_incidents=len(self._incidents),
            total_units=len(self._units),
            max_possible_score=self._max_possible,
        )

        return DispatchTriageObservation(
            done=False,
            reward=0.0,
            incidents=deepcopy(self._incidents),
            units=deepcopy(self._units),
            dispatch_count=0,
            score_so_far=0.0,
            message=(
                f"[{difficulty.upper()}] {len(self._incidents)} emergency calls active, "
                f"{len(self._units)} units available. "
                "Read each description carefully — dispatch the most critical calls "
                "first using the best-matched unit type."
            ),
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: DispatchTriageAction,
        timeout_s=None,
        **kwargs,
    ) -> DispatchTriageObservation:

        self._state.step_count += 1

        # --- Validate action -------------------------------------------
        incident = self._find_incident(action.incident_id)
        unit     = self._find_unit(action.unit_id)

        if incident is None:
            return self._make_obs(
                done=False,
                message=f"Invalid incident id {action.incident_id}. "
                        f"Choose from: {[i.id for i in self._incidents if not i.resolved]}",
            )
        if unit is None:
            return self._make_obs(
                done=False,
                message=f"Invalid unit id {action.unit_id}. "
                        f"Choose from: {[u.id for u in self._units if u.available]}",
            )
        if not unit.available:
            return self._make_obs(
                done=False,
                message=f"Unit {action.unit_id} ({unit.type}) is already deployed. "
                        f"Available units: {[u.id for u in self._units if u.available]}",
            )
        if incident.resolved:
            return self._make_obs(
                done=False,
                message=f"Incident {action.incident_id} is already resolved. "
                        f"Unresolved: {[i.id for i in self._incidents if not i.resolved]}",
            )

        # --- Internal meta (never exposed to agent) -------------------
        meta          = self._meta[incident.id]
        internal_type = meta["type"]
        base_severity = meta["severity"]

        # --- Cascade check (hard mode) ---------------------------------
        # If the agent dispatches an incident whose dependencies are not yet
        # resolved, a score PENALTY is applied (not a severity boost).
        # This makes resolving dependencies first the strictly correct strategy.
        cascade_active: bool = False
        unresolved_deps: List[int] = []
        if incident.depends_on:
            unresolved_deps = [
                d for d in incident.depends_on
                if not self._is_resolved(d)
            ]
            if unresolved_deps:
                self._total_penalties += CASCADE_SEVERITY_BOOST
                cascade_active = True

        # --- Unit-type match check ------------------------------------
        expected_unit  = CORRECT_UNIT[internal_type]
        type_mismatch  = (expected_unit != unit.type)
        if type_mismatch:
            self._total_penalties += 1.5

        # --- Dispatch -------------------------------------------------
        self._dispatch_count     += 1
        unit.available            = False
        incident.assigned_unit_id = unit.id
        incident.dispatch_rank    = self._dispatch_count
        incident.resolved         = True

        # --- Score update ---------------------------------------------
        self._raw_score  += base_severity / self._dispatch_count
        running_score     = max(0.0, self._raw_score - self._total_penalties)
        normalised        = (
            min(1.0, running_score / self._max_possible)
            if self._max_possible > 0 else 0.0
        )

        # --- Done? ----------------------------------------------------
        done = (
            all(not u.available for u in self._units) or
            all(inc.resolved    for inc in self._incidents)
        )

        # --- Message --------------------------------------------------
        notes: List[str] = []
        if type_mismatch:
            notes.append(
                f"Wrong unit type: sent {unit.type} but {expected_unit} expected — "
                "penalty 1.5 applied."
            )
        if cascade_active:
            notes.append(
                f"Cascade violation: incident {incident.id} dispatched before "
                f"blocking incident(s) {unresolved_deps} were resolved — "
                f"penalty {CASCADE_SEVERITY_BOOST} applied."
            )

        message = (
            f"Dispatched {unit.type} (unit {unit.id}) to incident {incident.id} "
            f"at {incident.location}."
        )
        if notes:
            message += " " + " ".join(notes)
        if done:
            unresolved = [i for i in self._incidents if not i.resolved]
            if unresolved:
                ids = ", ".join(f"incident {i.id} at {i.location}" for i in unresolved)
                message += f" Episode complete. {len(unresolved)} call(s) unattended: {ids}."
            else:
                message += " All incidents resolved!"
            message += f" Final score: {normalised:.4f}"

        return DispatchTriageObservation(
            done=done,
            reward=normalised,
            incidents=deepcopy(self._incidents),
            units=deepcopy(self._units),
            dispatch_count=self._dispatch_count,
            score_so_far=normalised,
            message=message,
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> DispatchTriageState:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_incident(self, iid: int) -> Optional[Incident]:
        return next((i for i in self._incidents if i.id == iid), None)

    def _find_unit(self, uid: int) -> Optional[Unit]:
        return next((u for u in self._units if u.id == uid), None)

    def _is_resolved(self, iid: int) -> bool:
        inc = self._find_incident(iid)
        return inc.resolved if inc is not None else False

    def _make_obs(self, *, done: bool, message: str) -> DispatchTriageObservation:
        running_score = max(0.0, self._raw_score - self._total_penalties)
        normalised    = (
            min(1.0, running_score / self._max_possible)
            if self._max_possible > 0 else 0.0
        )
        return DispatchTriageObservation(
            done=done,
            reward=normalised,
            incidents=deepcopy(self._incidents),
            units=deepcopy(self._units),
            dispatch_count=self._dispatch_count,
            score_so_far=normalised,
            message=message,
        )
