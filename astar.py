"""Minimal A* pathfinding on a 2D occupancy grid."""

from __future__ import annotations

import heapq

from spatial_map import FORBIDDEN, OBSTACLE, Grid, in_bounds
from world_state import WorldState
from spatial_map import world_to_grid

Position = tuple[int, int]


def heuristic(current: Position, goal: Position) -> int:
    """Compute the Manhattan distance between two grid positions."""
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def get_neighbors(position: Position) -> list[Position]:
    """Return 4-connected neighbor positions."""
    x, y = position
    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    ]


def is_blocked(grid: Grid, position: Position) -> bool:
    """Return True when a grid cell cannot be traversed."""
    if not in_bounds(position, width=len(grid[0]), height=len(grid)):
        return True

    x, y = position
    return grid[y][x] in (OBSTACLE, FORBIDDEN)


def reconstruct_path(came_from: dict[Position, Position], goal: Position) -> list[Position]:
    """Reconstruct a path from the predecessor map."""
    path = [goal]
    current = goal

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar(grid: Grid, start: Position, goal: Position) -> tuple[list[Position], int]:
    """Find a shortest 4-connected path on a 2D occupancy grid."""
    if is_blocked(grid, start):
        print("A*: failed, start is blocked or out of bounds.")
        return [], 0

    if is_blocked(grid, goal):
        print("A*: failed, goal is blocked or out of bounds.")
        return [], 0

    open_heap: list[tuple[int, Position]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), start))

    came_from: dict[Position, Position] = {}
    g_score: dict[Position, int] = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = reconstruct_path(came_from, goal)
            path_length = len(path) - 1
            print(f"A*: success, path found with length {path_length}.")
            return path, path_length

        for neighbor in get_neighbors(current):
            if is_blocked(grid, neighbor):
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            priority = tentative_g_score + heuristic(neighbor, goal)
            heapq.heappush(open_heap, (priority, neighbor))

    print("A*: failed, no path found.")
    return [], 0


if __name__ == "__main__":
    world = WorldState(
        obstacles=[(2, 1), (2, 2), (2, 3)],
        forbidden_zones=[(1, 4)],
    )
    grid = world_to_grid(world)

    start = (0, 0)
    goal = (4, 4)
    path, path_length = astar(grid, start, goal)

    print("start:", start)
    print("goal:", goal)
    print("path:", path)
    print("path length:", path_length)
