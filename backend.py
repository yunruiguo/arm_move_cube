"""Backend interface and toy backend implementation for planning experiments."""

from __future__ import annotations

from typing import Protocol

from reachability_engine import query_reachability
from world_state import WorldState

Action = dict[str, object]
COA = dict[str, object]
SimulationResult = dict[str, object]


class PlanningBackend(Protocol):
    """Minimal backend interface for simulator-agnostic planning."""

    def get_current_state(self) -> WorldState:
        """Return the current observable state."""

    def apply_action(self, action: Action) -> dict[str, object]:
        """Evaluate a single structured action."""

    def simulate_action_sequence(self, coa: COA) -> SimulationResult:
        """Evaluate a whole COA action sequence."""


class ToyWorldBackend:
    """Backend adapter for the current in-memory toy benchmark world."""

    def __init__(self, initial_state: WorldState) -> None:
        """Store the deterministic toy state."""
        self._state = initial_state

    def get_current_state(self) -> WorldState:
        """Return the current toy world state."""
        return self._state

    def apply_action(self, action: Action) -> dict[str, object]:
        """Evaluate a pick/place action without mutating the toy world."""
        object_name = str(action["object"])
        place_position = action["place_position"]
        assert isinstance(place_position, tuple)

        pick_result = query_reachability(self._state, "pick", object_name)
        place_result = query_reachability(
            self._state,
            "place",
            object_name,
            place_position=place_position,
        )

        step_success = (
            bool(pick_result["reachable"])
            and bool(pick_result["action_feasible"])
            and bool(place_result["reachable"])
            and bool(place_result["action_feasible"])
        )
        step_score = int(pick_result["score"]) + int(place_result["score"])
        step_score += 10 if step_success else -5

        return {
            "object": object_name,
            "place_position": place_position,
            "pick_result": pick_result,
            "place_result": place_result,
            "success": step_success,
            "score": step_score,
        }

    def simulate_action_sequence(self, coa: COA) -> SimulationResult:
        """Evaluate a whole COA as a sequence of independent toy actions."""
        total_score = 0
        failed_actions = 0
        completed_actions = 0
        failure_reasons: list[str] = []
        step_results: list[dict[str, object]] = []

        for action in coa["actions"]:
            step_result = self.apply_action(action)
            step_results.append(step_result)
            total_score += int(step_result["score"])

            if step_result["success"]:
                completed_actions += 1
                continue

            failed_actions += 1
            failure_reasons.append(
                f"{step_result['object']}: "
                f"{step_result['pick_result']['reason']} | "
                f"{step_result['place_result']['reason']}"
            )
            break

        success = failed_actions == 0 and completed_actions == len(coa["actions"])
        return {
            "selected_strategy": coa["name"],
            "family": coa["family"],
            "total_score": total_score,
            "success": success,
            "failed_actions": failed_actions,
            "completed_actions": completed_actions,
            "failure_reasons": "; ".join(failure_reasons) if failure_reasons else "",
            "step_results": step_results,
        }
