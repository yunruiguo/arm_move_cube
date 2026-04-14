"""Run the minimal decision platform end to end."""

from __future__ import annotations

from mcts import evaluate_coas, select_best_coa
from spatial_map import FORBIDDEN, OBSTACLE, FREE, world_to_grid
from world_state import WorldState


def build_toy_world() -> WorldState:
    """Create a deterministic toy world for end-to-end evaluation."""
    state = WorldState(
        robot_position=(0, 0),
        obstacles=[(4, 1), (4, 2), (4, 3), (4, 4), (2, 5), (3, 5)],
        forbidden_zones=[(8, 8), (8, 9)],
        goal_regions={
            "left_goal": [(6, 2), (6, 3)],
            "right_goal": [(7, 6), (7, 7)],
        },
    )
    state.update_object("box_a", position=(2, 2), object_type="crate", graspable=True)
    state.update_object("box_b", position=(6, 1), object_type="crate", graspable=True)
    state.update_object("statue", position=(10, 10), object_type="decor", graspable=False)
    return state


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


def generate_coas(state: WorldState) -> list[dict[str, object]]:
    """Generate a small deterministic set of candidate courses of action."""
    return [
        {
            "name": "move_box_a_to_left_goal",
            "object": "box_a",
            "place_position": state.goal_regions["left_goal"][0],
        },
        {
            "name": "move_box_b_to_right_goal",
            "object": "box_b",
            "place_position": state.goal_regions["right_goal"][0],
        },
        {
            "name": "move_statue_to_left_goal",
            "object": "statue",
            "place_position": state.goal_regions["left_goal"][1],
        },
        {
            "name": "move_box_a_to_forbidden_zone",
            "object": "box_a",
            "place_position": state.forbidden_zones[0],
        },
    ]


def print_coas(coas: list[dict[str, object]]) -> None:
    """Print candidate courses of action."""
    print("=== generated coas ===")
    for index, coa in enumerate(coas, start=1):
        print(
            f"{index}. {coa['name']}: pick {coa['object']} -> place at {coa['place_position']}"
        )


def print_score_table(results: list[dict[str, object]]) -> None:
    """Print a structured end-of-run score summary."""
    print("=== per-coa scores ===")
    for result in results:
        print(f"- {result['name']}")
        print(
            f"  total={result['score']} "
            f"(pick={result['pick_score']}, place={result['place_score']}, "
            f"adjustment={result['bonus']})"
        )
        print(
            f"  success={result['success']}, "
            f"pick_reachable={result['pick_result']['reachable']}, "
            f"place_reachable={result['place_result']['reachable']}"
        )
        print(f"  why={result['reason']}")


def print_best_choice(best: dict[str, object], results: list[dict[str, object]]) -> None:
    """Print why the best COA won over the other candidates."""
    runner_up = None
    for candidate in sorted(results, key=lambda item: int(item["score"]), reverse=True):
        if candidate["name"] != best["name"]:
            runner_up = candidate
            break

    print("=== selected best coa ===")
    print("name:", best["name"])
    print("object:", best["object"])
    print("place position:", best["place_position"])
    print("score:", best["score"])
    print("success:", best["success"])
    print("reason:", best["reason"])
    print("selection rationale:")
    print(
        f"  selected because it achieved the highest score ({best['score']}) "
        f"with success={best['success']}."
    )
    print(
        f"  pick contribution={best['pick_score']}, "
        f"place contribution={best['place_score']}, "
        f"adjustment={best['bonus']}."
    )
    if runner_up is not None:
        score_gap = int(best["score"]) - int(runner_up["score"])
        print(
            f"  score gap over next best option '{runner_up['name']}': {score_gap}."
        )
        print(f"  next best reason: {runner_up['reason']}")


def main() -> None:
    """Run the full toy planning prototype."""
    state = build_toy_world()
    grid = world_to_grid(state)
    coas = generate_coas(state)

    summarize_world(state)
    summarize_grid(grid)
    print_coas(coas)

    print("=== evaluating coas ===")
    results = evaluate_coas(state, coas)

    print_score_table(results)
    best = select_best_coa(results)
    print_best_choice(best, results)


if __name__ == "__main__":
    main()
