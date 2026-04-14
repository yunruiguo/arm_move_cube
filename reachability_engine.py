"""Minimal reachability queries that combine pathfinding and action checks."""

from __future__ import annotations

from astar import astar
from rule_checker import check_pick, check_place
from spatial_map import GRID_HEIGHT, GRID_WIDTH, in_bounds, world_to_grid
from world_state import Position, WorldState

ReachabilityResult = dict[str, object]


def get_approach_points(position: Position) -> list[Position]:
    """Return simple 4-connected approach points around a target."""
    x, y = position
    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    ]


def compute_score(reachable: bool, action_feasible: bool, path_cost: int) -> int:
    """Return a simple reachability score."""
    if not reachable:
        return -20

    if not action_feasible:
        return -10 - (2 * path_cost)

    return max(1, 24 - (2 * path_cost))


def build_reachability_result(
    reachable: bool,
    path: list[Position],
    path_cost: int,
    action_feasible: bool,
    score: int,
    reason: str,
) -> ReachabilityResult:
    """Build a consistent reachability result payload."""
    return {
        "reachable": reachable,
        "path": path,
        "path_cost": path_cost,
        "action_feasible": action_feasible,
        "score": score,
        "reason": reason,
    }


def query_reachability(
    state: WorldState,
    action_type: str,
    target: str,
    place_position: Position | None = None,
) -> ReachabilityResult:
    """Evaluate whether an action is reachable and feasible in the current world."""
    robot_position = state.get_robot_position()
    grid = world_to_grid(state)

    print(f"Reachability query: action={action_type}, target={target}, robot={robot_position}")

    if action_type == "pick":
        target_position = state.get_object_position(target)
        if target_position is None:
            reason = f"pick failed: object '{target}' does not exist."
            print(reason)
            return build_reachability_result(False, [], 0, False, -10, reason)

        print(f"Pick target position: {target_position}")
        best_path: list[Position] = []
        best_path_cost: int | None = None

        for approach in get_approach_points(target_position):
            if not in_bounds(approach, width=GRID_WIDTH, height=GRID_HEIGHT):
                print(f"Skipping approach point {approach}: out of bounds.")
                continue

            print(f"Trying pick approach point: {approach}")
            path, path_cost = astar(grid, robot_position, approach)
            if path and (best_path_cost is None or path_cost < best_path_cost):
                best_path = path
                best_path_cost = path_cost

        reachable = bool(best_path)
        rule_result = check_pick(state, target)
        action_feasible = bool(rule_result["valid"])
        path_cost = best_path_cost if best_path_cost is not None else 0
        reason = rule_result["reason"]

        if not reachable and action_feasible:
            reason = f"pick failed: no reachable approach point for '{target}'."

        score = compute_score(reachable, action_feasible, path_cost)
        print(
            f"Pick result: reachable={reachable}, feasible={action_feasible}, "
            f"path_cost={path_cost}, score={score}"
        )
        return build_reachability_result(
            reachable,
            best_path,
            path_cost,
            action_feasible,
            score,
            reason,
        )

    if action_type == "place":
        if place_position is None:
            reason = "place failed: place_position is required."
            print(reason)
            return build_reachability_result(False, [], 0, False, -10, reason)

        print(f"Place target position: {place_position}")
        path, path_cost = astar(grid, robot_position, place_position)
        reachable = bool(path)
        rule_result = check_place(state, place_position)
        action_feasible = bool(rule_result["valid"])
        reason = rule_result["reason"]

        if not reachable and action_feasible:
            reason = f"place failed: no path to target position {place_position}."

        score = compute_score(reachable, action_feasible, path_cost)
        print(
            f"Place result: reachable={reachable}, feasible={action_feasible}, "
            f"path_cost={path_cost}, score={score}"
        )
        return build_reachability_result(
            reachable,
            path,
            path_cost,
            action_feasible,
            score,
            reason,
        )

    reason = f"unsupported action type: {action_type}"
    print(reason)
    return build_reachability_result(False, [], 0, False, -10, reason)


if __name__ == "__main__":
    demo_state = WorldState(
        robot_position=(0, 0),
        obstacles=[(1, 1), (1, 2), (1, 3)],
        forbidden_zones=[(4, 4)],
    )
    demo_state.update_object("box", position=(3, 3), object_type="crate", graspable=True)
    demo_state.update_object("statue", position=(6, 6), object_type="decor", graspable=False)

    print("=== pick success example ===")
    print(query_reachability(demo_state, "pick", "box"))

    print("=== place success example ===")
    print(query_reachability(demo_state, "place", "box", place_position=(3, 0)))

    print("=== pick failure example ===")
    print(query_reachability(demo_state, "pick", "statue"))

    print("=== place failure example ===")
    print(query_reachability(demo_state, "place", "box", place_position=(4, 4)))
