"""Minimal reusable COA evaluation utilities for the planning prototype."""

from __future__ import annotations

from backend import PlanningBackend
from reachability_engine import query_reachability
from world_state import WorldState

COA = dict[str, object]
EvaluationResult = dict[str, object]


def _get_state(source: PlanningBackend | WorldState) -> WorldState:
    """Return a world state from either a backend or a raw state object."""
    if hasattr(source, "get_current_state"):
        return source.get_current_state()
    return source


def _print_action_summary(
    action_label: str,
    action_result: dict[str, object],
) -> None:
    """Print a structured summary for a single action evaluation."""
    print(f"  action: {action_label}")
    print(
        "  reachability:",
        f"reachable={action_result['reachable']}, feasible={action_result['action_feasible']}",
    )
    print(
        "  path details:",
        f"path_cost={action_result['path_cost']}, path={action_result['path']}",
    )
    print("  score contribution:", action_result["score"])
    print("  reason:", action_result["reason"])


def evaluate_coa_with_simplified_mcts(
    source: PlanningBackend | WorldState,
    coa: COA,
) -> EvaluationResult:
    """Evaluate a COA with a simple deterministic rollout-style score."""
    state = _get_state(source)
    object_name = str(coa["object"])
    place_position = coa["place_position"]
    assert isinstance(place_position, tuple)

    print(f"=== evaluating coa: {coa['name']} ===")
    print("step 1/2: evaluate pick action")
    pick_result = query_reachability(state, "pick", object_name)
    _print_action_summary(f"pick {object_name}", pick_result)

    print("step 2/2: evaluate place action")
    place_result = query_reachability(
        state,
        "place",
        object_name,
        place_position=place_position,
    )
    _print_action_summary(f"place {object_name} at {place_position}", place_result)

    pick_score = int(pick_result["score"])
    place_score = int(place_result["score"])
    total_score = pick_score + place_score
    success = (
        bool(pick_result["reachable"])
        and bool(pick_result["action_feasible"])
        and bool(place_result["reachable"])
        and bool(place_result["action_feasible"])
    )

    if success:
        bonus = 10
        total_score += bonus
        reason = "COA succeeded: pick and place are both reachable and feasible."
    else:
        bonus = -5
        total_score += bonus
        reason = (
            f"COA failed: pick_reason={pick_result['reason']} | "
            f"place_reason={place_result['reason']}"
        )

    result = {
        "name": coa["name"],
        "object": object_name,
        "place_position": place_position,
        "pick_result": pick_result,
        "place_result": place_result,
        "pick_score": pick_score,
        "place_score": place_score,
        "bonus": bonus,
        "score": total_score,
        "success": success,
        "reason": reason,
    }

    print("score breakdown:")
    print(f"  pick contribution: {pick_score}")
    print(f"  place contribution: {place_score}")
    print(f"  outcome adjustment: {bonus}")
    print(f"  total score: {result['score']}")
    if not result["success"]:
        print("failure explanation:", result["reason"])
    else:
        print("success explanation:", result["reason"])
    return result


def evaluate_coas(source: PlanningBackend | WorldState, coas: list[COA]) -> list[EvaluationResult]:
    """Evaluate a list of COAs in order."""
    return [evaluate_coa_with_simplified_mcts(source, candidate_coa) for candidate_coa in coas]


def select_best_coa(results: list[EvaluationResult]) -> EvaluationResult:
    """Select the highest scoring COA deterministically."""
    return max(results, key=lambda evaluation_result: int(evaluation_result["score"]))
