"""Reusable planner helpers for the richer showcase demo."""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from contextlib import redirect_stdout

from backend import ToyWorldBackend
from benchmark_scenarios_real import RealScenario, build_real_benchmark_world
from main import build_coa, build_pick_place_action, manhattan_distance
from mcts import evaluate_coas, select_best_coa

Action = dict[str, object]
COA = dict[str, object]
PlannerPlan = dict[str, object]

SHOWCASE_PLANNER_MODES = (
    "fixed_order",
    "nearest_first",
    "clear_blocking_first",
    "decision_pipeline",
    "llm_generated",
)

SHOWCASE_OBJECT_COLORS = (
    "#D84A4A",
    "#3274D9",
    "#3FAF5A",
    "#D9A232",
    "#8E5CD9",
    "#3FB5B5",
    "#CC5FA8",
    "#777777",
)


def build_object_annotations(config: RealScenario) -> dict[str, dict[str, object]]:
    """Assign stable IDs and colors so every object keeps a human-readable identity."""
    ordered_names = list(config.fixed_order) + [
        object_name
        for object_name in sorted(config.object_grid_positions)
        if object_name not in config.fixed_order
    ]
    annotations: dict[str, dict[str, object]] = {}
    for index, object_name in enumerate(ordered_names, start=1):
        annotations[object_name] = {
            "cube_id": f"C{index:02d}",
            "color": SHOWCASE_OBJECT_COLORS[(index - 1) % len(SHOWCASE_OBJECT_COLORS)],
            "goal_region": config.object_goal_regions[object_name],
        }
    return annotations


def describe_scenario(config: RealScenario) -> dict[str, object]:
    """Convert a real scenario into a compact serializable summary."""
    object_annotations = build_object_annotations(config)
    return {
        "name": config.name,
        "description": config.description,
        "robot_grid_position": config.robot_grid_position,
        "fixed_order": list(config.fixed_order),
        "object_grid_positions": config.object_grid_positions,
        "object_sim_positions": config.object_sim_positions,
        "object_goal_regions": config.object_goal_regions,
        "goal_region_grid_positions": config.goal_region_grid_positions,
        "goal_region_sim_positions": config.goal_region_sim_positions,
        "obstacles": list(config.obstacles),
        "forbidden_zones": list(config.forbidden_zones),
        "object_annotations": object_annotations,
    }


def format_planned_actions(coa: COA) -> list[str]:
    """Convert a structured COA into readable action labels."""
    return [
        f"{action['action_type']} {action['object']} -> {action['place_position']}"
        for action in coa["actions"]
    ]


def build_rollout_subtasks(config: RealScenario, coa: COA) -> list[dict[str, object]]:
    """Convert planner actions into rollout subtasks for the real recorder."""
    goal_region_by_grid = {
        grid_position: region_name
        for region_name, grid_position in config.goal_region_grid_positions.items()
    }
    subtasks: list[dict[str, object]] = []
    for action in coa["actions"]:
        object_name = str(action["object"])
        place_position = action["place_position"]
        assert isinstance(place_position, tuple)
        goal_region_name = str(action["goal_region"]) if "goal_region" in action else str(
            goal_region_by_grid[place_position]
        )
        goal_region_sim_position = config.goal_region_sim_positions[goal_region_name]
        subtasks.append(
            {
                "object": object_name,
                "cube_id": config_summary_id(config, object_name),
                "cube_initial_position": config.object_sim_positions[object_name],
                "target_position": goal_region_sim_position,
                "expected_cube_position": (
                    goal_region_sim_position[0],
                    goal_region_sim_position[1],
                    config.object_sim_positions[object_name][2],
                ),
                "goal_region": goal_region_name,
            }
        )
    return subtasks


def config_summary_id(config: RealScenario, object_name: str) -> str:
    """Fetch the stable object ID for a named object."""
    return build_object_annotations(config)[object_name]["cube_id"]  # type: ignore[index]


def build_full_sequence_actions(
    config: RealScenario,
    object_names: list[str],
) -> list[Action]:
    """Build one pick/place action per object using the scenario's goal mapping."""
    goal_slot_positions = build_goal_slot_positions(config)
    region_counts = {region_name: 0 for region_name in goal_slot_positions}
    actions: list[Action] = []
    for object_name in object_names:
        goal_region_name = config.object_goal_regions[object_name]
        goal_slots = goal_slot_positions[goal_region_name]
        goal_position = goal_slots[min(region_counts[goal_region_name], len(goal_slots) - 1)]
        region_counts[goal_region_name] += 1
        action = build_pick_place_action(object_name, goal_position)
        action["goal_region"] = goal_region_name
        actions.append(action)
    return actions


def build_goal_slot_positions(config: RealScenario) -> dict[str, list[tuple[int, int]]]:
    """Create a few nearby goal slots per region for showcase multi-object delivery."""
    slot_offsets = {
        "left_goal": [(0, 0), (0, 1), (0, -1)],
        "staging_goal": [(0, 0), (1, 0), (-1, 0)],
        "right_goal": [(0, 0), (0, -1), (0, 1)],
    }
    slots_by_region: dict[str, list[tuple[int, int]]] = {}
    for region_name, base_position in config.goal_region_grid_positions.items():
        offsets = slot_offsets.get(region_name, [(0, 0), (1, 0), (0, 1)])
        slots_by_region[region_name] = [
            (base_position[0] + dx, base_position[1] + dy)
            for dx, dy in offsets
        ]
    return slots_by_region


def estimate_blocking_score(config: RealScenario, object_name: str) -> int:
    """Heuristic score for how likely an object is to be a useful blocker to clear first."""
    object_position = config.object_grid_positions[object_name]
    score = 0
    for obstacle in config.obstacles:
        if manhattan_distance(object_position, obstacle) <= 2:
            score += 3
    for goal_position in config.goal_region_grid_positions.values():
        if manhattan_distance(object_position, goal_position) <= 6:
            score += 1
    if object_position[0] >= config.robot_grid_position[0] + 3:
        score += 1
    if abs(object_position[1] - config.robot_grid_position[1]) <= 2:
        score += 2
    return score


def build_fixed_order_plan(config: RealScenario) -> PlannerPlan:
    """Build the naive fixed-order showcase plan."""
    actions = build_full_sequence_actions(config, list(config.fixed_order))
    coa = build_coa("fixed-order", f"fixed_order_{config.name}", actions)
    return {
        "planner_name": "fixed_order",
        "selected_strategy": coa["name"],
        "coa": coa,
        "selected_object": actions[0]["object"],
        "selected_goal_region": config.object_goal_regions[str(actions[0]["object"])],
        "planned_actions": format_planned_actions(coa),
        "planning_log": (
            "Fixed-order baseline follows the scenario's predefined sequence without "
            "reconsidering distance or blocking effects."
        ),
    }


def build_nearest_first_plan(config: RealScenario) -> PlannerPlan:
    """Build the nearest-first showcase plan using the robot's grid position."""
    object_names = sorted(
        config.object_grid_positions,
        key=lambda object_name: manhattan_distance(
            config.robot_grid_position,
            config.object_grid_positions[object_name],
        ),
    )
    actions = build_full_sequence_actions(config, object_names)
    coa = build_coa("nearest-first", f"nearest_first_{config.name}", actions)
    return {
        "planner_name": "nearest_first",
        "selected_strategy": coa["name"],
        "coa": coa,
        "selected_object": actions[0]["object"],
        "selected_goal_region": config.object_goal_regions[str(actions[0]["object"])],
        "planned_actions": format_planned_actions(coa),
        "planning_log": (
            "Nearest-first sorts every object by planner-grid Manhattan distance from the robot."
        ),
    }


def build_clear_blocking_first_plan(config: RealScenario) -> PlannerPlan:
    """Build the blocker-clearing showcase plan."""
    object_names = sorted(
        config.object_grid_positions,
        key=lambda object_name: (
            -estimate_blocking_score(config, object_name),
            manhattan_distance(
                config.robot_grid_position,
                config.object_grid_positions[object_name],
            ),
        ),
    )
    actions = build_full_sequence_actions(config, object_names)
    coa = build_coa("clear-blocking-first", f"clear_blocking_first_{config.name}", actions)
    return {
        "planner_name": "clear_blocking_first",
        "selected_strategy": coa["name"],
        "coa": coa,
        "selected_object": actions[0]["object"],
        "selected_goal_region": config.object_goal_regions[str(actions[0]["object"])],
        "planned_actions": format_planned_actions(coa),
        "planning_log": (
            "Clear-blocking-first prioritizes objects near planner obstacles or central routes, "
            "then breaks ties by distance."
        ),
    }


def build_goal_grouped_candidate(config: RealScenario) -> COA:
    """Build an additional non-trivial candidate for the decision pipeline."""
    grouped_names = sorted(
        config.object_grid_positions,
        key=lambda object_name: (
            config.object_goal_regions[object_name],
            manhattan_distance(
                config.object_grid_positions[object_name],
                config.goal_region_grid_positions[config.object_goal_regions[object_name]],
            ),
        ),
    )
    actions = build_full_sequence_actions(config, grouped_names)
    return build_coa("goal-grouped", f"goal_grouped_{config.name}", actions)


def build_candidate_coas(config: RealScenario) -> list[COA]:
    """Build deterministic candidate strategies for the showcase decision pipeline."""
    return [
        build_fixed_order_plan(config)["coa"],
        build_nearest_first_plan(config)["coa"],
        build_clear_blocking_first_plan(config)["coa"],
        build_goal_grouped_candidate(config),
    ]


def build_decision_pipeline_plan(config: RealScenario) -> PlannerPlan:
    """Build a rule-aware multi-COA showcase plan using the existing evaluator."""
    state = build_real_benchmark_world(config.name)
    backend = ToyWorldBackend(state)
    candidate_coas = build_candidate_coas(config)
    planning_buffer = io.StringIO()
    with redirect_stdout(planning_buffer):
        evaluation_results = evaluate_coas(backend, candidate_coas)
        best_result = select_best_coa(evaluation_results)

    selected_coa = next(
        candidate_coa
        for candidate_coa in candidate_coas
        if candidate_coa["name"] == best_result["name"]
    )
    ranked_results = sorted(
        evaluation_results,
        key=lambda evaluation_result: int(evaluation_result["score"]),
        reverse=True,
    )
    runner_up = ranked_results[1] if len(ranked_results) > 1 else None
    decision_rationale = (
        f"Decision pipeline selected {selected_coa['name']} as the highest-scoring COA."
        if runner_up is None
        else (
            f"Decision pipeline selected {selected_coa['name']} over {runner_up['name']} "
            f"by a score margin of {int(best_result['score']) - int(runner_up['score'])}. "
            f"Winner reason: {best_result['reason']}"
        )
    )
    return {
        "planner_name": "decision_pipeline",
        "selected_strategy": selected_coa["name"],
        "coa": selected_coa,
        "selected_object": selected_coa["actions"][0]["object"],
        "selected_goal_region": config.object_goal_regions[str(selected_coa["actions"][0]["object"])],
        "planned_actions": format_planned_actions(selected_coa),
        "planning_log": planning_buffer.getvalue().strip(),
        "coa_evaluation_summary": [
            {
                "name": evaluation_result["name"],
                "score": int(evaluation_result["score"]),
                "success": bool(evaluation_result["success"]),
                "reason": str(evaluation_result["reason"]),
            }
            for evaluation_result in ranked_results
        ],
        "decision_rationale": decision_rationale,
    }


def call_openai_for_coa(config: RealScenario, model: str) -> COA:
    """Optionally ask an LLM for a high-level ordering and convert it into a COA."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    prompt = {
        "scenario": describe_scenario(config),
        "instruction": (
            "Return JSON with keys 'name' and 'object_order'. "
            "object_order must contain every object exactly once. "
            "Prefer a sensible high-level order for tabletop delivery."
        ),
    }
    request_body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(prompt),
                    }
                ],
            }
        ],
        "text": {"format": {"type": "json_object"}},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(f"LLM request failed: {error}") from error

    output_text = payload.get("output_text", "")
    if not output_text:
        raise RuntimeError("LLM response did not contain output_text.")
    llm_result = json.loads(output_text)
    object_order = llm_result["object_order"]
    if sorted(object_order) != sorted(config.object_grid_positions):
        raise RuntimeError("LLM returned an invalid object ordering.")
    actions = build_full_sequence_actions(config, list(object_order))
    return build_coa(
        "llm-generated",
        str(llm_result.get("name", f"llm_generated_{config.name}")),
        actions,
    )


def build_llm_generated_plan(
    config: RealScenario,
    model: str | None = None,
) -> PlannerPlan:
    """Build an optional LLM-based high-level plan."""
    selected_model = model or os.environ.get("SHOWCASE_LLM_MODEL", "gpt-5.4-mini")
    coa = call_openai_for_coa(config, selected_model)
    return {
        "planner_name": "llm_generated",
        "selected_strategy": coa["name"],
        "coa": coa,
        "selected_object": coa["actions"][0]["object"],
        "selected_goal_region": config.object_goal_regions[str(coa["actions"][0]["object"])],
        "planned_actions": format_planned_actions(coa),
        "planning_log": (
            f"LLM generated a high-level object ordering using model {selected_model}."
        ),
        "llm_model": selected_model,
    }


def build_plan(
    planner_mode: str,
    config: RealScenario,
    llm_model: str | None = None,
) -> PlannerPlan:
    """Dispatch showcase planner selection."""
    if planner_mode == "fixed_order":
        return build_fixed_order_plan(config)
    if planner_mode == "nearest_first":
        return build_nearest_first_plan(config)
    if planner_mode == "clear_blocking_first":
        return build_clear_blocking_first_plan(config)
    if planner_mode == "decision_pipeline":
        return build_decision_pipeline_plan(config)
    if planner_mode == "llm_generated":
        return build_llm_generated_plan(config, model=llm_model)
    raise ValueError(f"Unsupported showcase planner mode: {planner_mode}")
