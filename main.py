"""Run the minimal decision platform end to end."""

from __future__ import annotations

from backend import PlanningBackend, ToyWorldBackend
from benchmark_scenarios import build_benchmark_world as build_default_benchmark_world
from mcts import evaluate_coas, select_best_coa
from spatial_map import FORBIDDEN, OBSTACLE, FREE, world_to_grid
from visualizer import generate_visualizations
from world_state import WorldState

Action = dict[str, object]
COA = dict[str, object]
EvaluationResult = dict[str, object]


def _get_state(source: PlanningBackend | WorldState) -> WorldState:
    """Return a world state from either a backend or a raw state object."""
    if hasattr(source, "get_current_state"):
        return source.get_current_state()
    return source


def build_benchmark_world() -> WorldState:
    """Create the default deterministic benchmark world."""
    return build_default_benchmark_world()


def build_benchmark_backend() -> ToyWorldBackend:
    """Create the default toy backend for the benchmark world."""
    return ToyWorldBackend(build_benchmark_world())


def build_toy_world() -> WorldState:
    """Backward-compatible alias for the current benchmark world."""
    return build_benchmark_world()


def summarize_world(state: WorldState) -> None:
    """Print a compact world summary for debugging."""
    print("=== world summary ===")
    print("robot position:", state.get_robot_position())
    print("objects:")
    for name, obj in state.objects.items():
        print(
            f"  - {name}: pos={obj['pos']}, type={obj['type']}, graspable={obj['graspable']}"
        )
    print("obstacles:", state.obstacles)
    print("forbidden zones:", state.forbidden_zones)
    print("goal regions:", state.goal_regions)


def summarize_grid(grid: list[list[int]]) -> None:
    """Print simple occupancy counts for the generated spatial map."""
    free_count = 0
    obstacle_count = 0
    forbidden_count = 0

    for row in grid:
        for value in row:
            if value == FREE:
                free_count += 1
            elif value == OBSTACLE:
                obstacle_count += 1
            elif value == FORBIDDEN:
                forbidden_count += 1

    print("=== spatial map summary ===")
    print(f"grid size: {len(grid[0])} x {len(grid)}")
    print(
        "cell counts:",
        f"free={free_count}, obstacle={obstacle_count}, forbidden={forbidden_count}",
    )


def manhattan_distance(start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Return the Manhattan distance between two 2D points."""
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def get_graspable_object_names(source: PlanningBackend | WorldState) -> list[str]:
    """Return graspable object names in insertion order."""
    state = _get_state(source)
    return [
        name for name, obj in state.objects.items() if bool(obj["graspable"])
    ]


def build_pick_place_action(object_name: str, place_position: tuple[int, int]) -> Action:
    """Build a simple structured pick/place action."""
    return {
        "action_type": "pick_place",
        "object": object_name,
        "place_position": place_position,
    }


def build_coa(family: str, name: str, actions: list[Action]) -> COA:
    """Build a COA that remains compatible with the current evaluator."""
    first_action = actions[0]
    return {
        "family": family,
        "name": name,
        "actions": actions,
        "object": first_action["object"],
        "place_position": first_action["place_position"],
    }


def estimate_pick_cost(source: PlanningBackend | WorldState, object_name: str) -> int:
    """Estimate how hard it is to reach an object from the robot."""
    state = _get_state(source)
    robot_position = state.get_robot_position()
    object_position = state.get_object_position(object_name)
    assert object_position is not None
    return manhattan_distance(robot_position, object_position)


def estimate_blocking_score(source: PlanningBackend | WorldState, object_name: str) -> int:
    """Estimate how likely an object is to obstruct useful routes."""
    state = _get_state(source)
    object_position = state.get_object_position(object_name)
    assert object_position is not None

    score = 0
    for obstacle in state.obstacles:
        if manhattan_distance(object_position, obstacle) <= 2:
            score += 2

    for goal_positions in state.goal_regions.values():
        for goal_position in goal_positions:
            if manhattan_distance(object_position, goal_position) <= 4:
                score += 1

    if object_position[1] >= 5:
        score += 2
    return score


def choose_goal_region_for_object(
    source: PlanningBackend | WorldState,
    object_name: str,
) -> tuple[str, tuple[int, int]]:
    """Assign an object to a sensible goal region based on its location."""
    state = _get_state(source)
    object_position = state.get_object_position(object_name)
    assert object_position is not None

    if object_position[0] <= 3:
        return "left_goal", state.goal_regions["left_goal"][0]
    if object_position[0] >= 8:
        return "right_goal", state.goal_regions["right_goal"][0]
    return "staging_goal", state.goal_regions["staging_goal"][0]


def generate_nearest_first_coa(source: PlanningBackend | WorldState) -> COA:
    """Generate a COA that prioritizes the easiest objects to reach first."""
    state = _get_state(source)
    robot_position = state.get_robot_position()
    sorted_names = sorted(
        get_graspable_object_names(source),
        key=lambda name: manhattan_distance(
            robot_position,
            state.get_object_position(name) or robot_position,
        ),
    )
    actions = [
        build_pick_place_action(sorted_names[0], state.goal_regions["left_goal"][0]),
        build_pick_place_action(sorted_names[1], state.goal_regions["staging_goal"][0]),
        build_pick_place_action(sorted_names[2], state.goal_regions["left_goal"][1]),
    ]
    return build_coa(
        "nearest-first",
        "nearest_first_progressive_delivery",
        actions,
    )


def generate_goal_grouped_coa(source: PlanningBackend | WorldState) -> COA:
    """Generate a COA that groups objects by sensible goal regions."""
    state = _get_state(source)
    region_order = ["left_goal", "right_goal", "staging_goal"]
    grouped_names: dict[str, list[str]] = {region_name: [] for region_name in region_order}

    for object_name in get_graspable_object_names(source):
        region_name, _ = choose_goal_region_for_object(source, object_name)
        grouped_names[region_name].append(object_name)

    actions: list[Action] = []
    for region_name in region_order:
        goal_position = state.goal_regions[region_name][0]
        region_objects = sorted(
            grouped_names[region_name],
            key=lambda object_name: manhattan_distance(
                state.get_object_position(object_name) or goal_position,
                goal_position,
            ),
        )
        for object_name in region_objects:
            actions.append(build_pick_place_action(object_name, goal_position))

    return build_coa(
        "goal-grouped",
        "goal_grouped_regional_sort",
        actions,
    )


def generate_clear_blocking_first_coa(source: PlanningBackend | WorldState) -> COA:
    """Generate a COA that clears likely blockers before easier deliveries."""
    state = _get_state(source)
    sorted_names = sorted(
        get_graspable_object_names(source),
        key=lambda name: (
            -estimate_blocking_score(source, name),
            estimate_pick_cost(source, name),
        ),
    )
    actions = [
        build_pick_place_action(sorted_names[0], state.goal_regions["staging_goal"][1]),
        build_pick_place_action(sorted_names[1], state.goal_regions["left_goal"][1]),
        build_pick_place_action(sorted_names[2], state.goal_regions["right_goal"][1]),
    ]
    return build_coa(
        "clear-blocking-first",
        "clear_blocking_then_deliver",
        actions,
    )


def generate_failure_probe_coa(source: PlanningBackend | WorldState) -> COA:
    """Generate a deliberately bad COA to expose failure behavior."""
    state = _get_state(source)
    statue_position = state.get_object_position("statue")
    assert statue_position is not None
    actions = [
        build_pick_place_action("statue", state.goal_regions["left_goal"][1]),
        build_pick_place_action("crate_blue", state.forbidden_zones[0]),
        build_pick_place_action("crate_yellow", statue_position),
    ]
    return build_coa(
        "failure-probe",
        "failure_probe_invalid_targets",
        actions,
    )


def generate_coas(source: PlanningBackend | WorldState) -> list[COA]:
    """Generate deterministic COAs across several strategy styles."""
    return [
        generate_nearest_first_coa(source),
        generate_goal_grouped_coa(source),
        generate_clear_blocking_first_coa(source),
        generate_failure_probe_coa(source),
    ]


def print_coas(coas: list[COA]) -> None:
    """Print candidate courses of action."""
    print("=== generated coas ===")
    for index, candidate_coa in enumerate(coas, start=1):
        print(f"{index}. [{candidate_coa['family']}] {candidate_coa['name']}")
        for action_index, action in enumerate(candidate_coa["actions"], start=1):
            print(
                f"   {action_index}. pick {action['object']} -> place at {action['place_position']}"
            )


def print_score_table(results: list[EvaluationResult]) -> None:
    """Print a structured end-of-run score summary."""
    print("=== per-coa scores ===")
    for evaluation_result in results:
        print(f"- {evaluation_result['name']}")
        print(
            f"  total={evaluation_result['score']} "
            f"(pick={evaluation_result['pick_score']}, place={evaluation_result['place_score']}, "
            f"adjustment={evaluation_result['bonus']})"
        )
        print(
            f"  success={evaluation_result['success']}, "
            f"pick_reachable={evaluation_result['pick_result']['reachable']}, "
            f"place_reachable={evaluation_result['place_result']['reachable']}"
        )
        print(f"  why={evaluation_result['reason']}")


def print_best_choice(
    best_result: EvaluationResult,
    results: list[EvaluationResult],
) -> None:
    """Print why the best COA won over the other candidates."""
    runner_up = None
    for candidate_result in sorted(
        results,
        key=lambda evaluation_result: int(evaluation_result["score"]),
        reverse=True,
    ):
        if candidate_result["name"] != best_result["name"]:
            runner_up = candidate_result
            break

    print("=== selected best coa ===")
    print("name:", best_result["name"])
    print("object:", best_result["object"])
    print("place position:", best_result["place_position"])
    print("score:", best_result["score"])
    print("success:", best_result["success"])
    print("reason:", best_result["reason"])
    print("selection rationale:")
    print(
        f"  selected because it achieved the highest score ({best_result['score']}) "
        f"with success={best_result['success']}."
    )
    print(
        f"  pick contribution={best_result['pick_score']}, "
        f"place contribution={best_result['place_score']}, "
        f"adjustment={best_result['bonus']}."
    )
    if runner_up is not None:
        score_gap = int(best_result["score"]) - int(runner_up["score"])
        print(
            f"  score gap over next best option '{runner_up['name']}': {score_gap}."
        )
        print(f"  next best reason: {runner_up['reason']}")


def main() -> None:
    """Run the full toy planning prototype."""
    backend = build_benchmark_backend()
    world_state = backend.get_current_state()
    occupancy_grid = world_to_grid(world_state)
    coas = generate_coas(backend)

    summarize_world(world_state)
    summarize_grid(occupancy_grid)
    print_coas(coas)

    print("=== evaluating coas ===")
    results = evaluate_coas(world_state, coas)

    print_score_table(results)
    best_result = select_best_coa(results)
    print_best_choice(best_result, results)
    generated_files = generate_visualizations(world_state, results, best_result)
    print("=== visualization outputs ===")
    for generated_file in generated_files:
        print(generated_file.as_posix())


if __name__ == "__main__":
    main()
