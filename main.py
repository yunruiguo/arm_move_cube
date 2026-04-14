"""Run the minimal decision platform end to end."""

from __future__ import annotations

from mcts import evaluate_coas, select_best_coa
from spatial_map import FORBIDDEN, OBSTACLE, FREE, world_to_grid
from visualizer import generate_visualizations
from world_state import WorldState

COA = dict[str, object]
EvaluationResult = dict[str, object]


def build_toy_world() -> WorldState:
    """Create a deterministic toy world for end-to-end evaluation."""
    world_state = WorldState(
        robot_position=(0, 0),
        obstacles=[
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (7, 3),
            (8, 3),
            (9, 3),
            (10, 3),
            (6, 7),
            (7, 7),
            (8, 7),
        ],
        forbidden_zones=[(11, 4), (11, 5), (12, 4), (12, 5)],
        goal_regions={
            "left_goal": [(1, 9), (2, 9)],
            "right_goal": [(12, 8), (12, 9)],
            "staging_goal": [(6, 9), (7, 9)],
        },
    )
    world_state.update_object(
        "crate_red", position=(2, 1), object_type="crate", graspable=True
    )
    world_state.update_object(
        "crate_blue", position=(5, 1), object_type="crate", graspable=True
    )
    world_state.update_object(
        "crate_green", position=(3, 6), object_type="crate", graspable=True
    )
    world_state.update_object(
        "crate_yellow", position=(8, 1), object_type="crate", graspable=True
    )
    world_state.update_object(
        "crate_orange", position=(10, 6), object_type="crate", graspable=True
    )
    world_state.update_object(
        "statue", position=(9, 9), object_type="decor", graspable=False
    )
    return world_state


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


def get_graspable_object_names(state: WorldState) -> list[str]:
    """Return graspable object names in insertion order."""
    return [
        name for name, obj in state.objects.items() if bool(obj["graspable"])
    ]


def generate_nearest_first_coas(state: WorldState) -> list[COA]:
    """Generate COAs that prioritize the nearest movable objects."""
    robot_position = state.get_robot_position()
    sorted_names = sorted(
        get_graspable_object_names(state),
        key=lambda name: manhattan_distance(
            robot_position,
            state.get_object_position(name) or robot_position,
        ),
    )
    return [
        {
            "name": "nearest_first_move_crate_red_to_left_goal",
            "object": sorted_names[0],
            "place_position": state.goal_regions["left_goal"][0],
        },
        {
            "name": "nearest_first_move_crate_blue_to_staging_goal",
            "object": sorted_names[1],
            "place_position": state.goal_regions["staging_goal"][0],
        },
    ]


def generate_clear_blocking_coas(state: WorldState) -> list[COA]:
    """Generate COAs that prioritize clearing blocking objects first."""
    return [
        {
            "name": "clear_blocking_move_crate_green_to_staging_goal",
            "object": "crate_green",
            "place_position": state.goal_regions["staging_goal"][1],
        },
        {
            "name": "clear_blocking_move_crate_green_to_forbidden_zone",
            "object": "crate_green",
            "place_position": state.forbidden_zones[0],
        },
    ]


def generate_grouped_goal_coas(state: WorldState) -> list[COA]:
    """Generate COAs that group objects by their target goal region."""
    return [
        {
            "name": "group_left_move_crate_green_to_left_goal",
            "object": "crate_green",
            "place_position": state.goal_regions["left_goal"][1],
        },
        {
            "name": "group_right_move_crate_yellow_to_right_goal",
            "object": "crate_yellow",
            "place_position": state.goal_regions["right_goal"][0],
        },
        {
            "name": "group_right_move_crate_orange_to_right_goal",
            "object": "crate_orange",
            "place_position": state.goal_regions["right_goal"][1],
        },
    ]


def generate_failure_probe_coas(state: WorldState) -> list[COA]:
    """Generate COAs that fail early and expose bad strategy choices."""
    return [
        {
            "name": "bad_pick_move_statue_to_left_goal",
            "object": "statue",
            "place_position": state.goal_regions["left_goal"][1],
        },
        {
            "name": "bad_place_move_crate_blue_to_forbidden_zone",
            "object": "crate_blue",
            "place_position": state.forbidden_zones[0],
        },
        {
            "name": "bad_place_move_crate_yellow_to_occupied_goal",
            "object": "crate_yellow",
            "place_position": state.get_object_position("statue"),
        },
    ]


def generate_coas(state: WorldState) -> list[COA]:
    """Generate deterministic COAs across several strategy styles."""
    return (
        generate_nearest_first_coas(state)
        + generate_clear_blocking_coas(state)
        + generate_grouped_goal_coas(state)
        + generate_failure_probe_coas(state)
    )


def print_coas(coas: list[COA]) -> None:
    """Print candidate courses of action."""
    print("=== generated coas ===")
    for index, candidate_coa in enumerate(coas, start=1):
        print(
            f"{index}. {candidate_coa['name']}: pick {candidate_coa['object']} -> place at {candidate_coa['place_position']}"
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
    world_state = build_toy_world()
    occupancy_grid = world_to_grid(world_state)
    coas = generate_coas(world_state)

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
