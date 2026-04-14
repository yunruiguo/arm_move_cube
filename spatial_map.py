"""Simple occupancy grid utilities for a 2D planning world."""

from __future__ import annotations

from world_state import Position, WorldState

GRID_WIDTH = 40
GRID_HEIGHT = 40
FREE = 0
OBSTACLE = 1
FORBIDDEN = 2

Grid = list[list[int]]


def in_bounds(position: Position, width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> bool:
    """Return True when a position is inside the grid bounds."""
    x, y = position
    return 0 <= x < width and 0 <= y < height


def to_grid_index(position: Position) -> tuple[int, int]:
    """Convert an (x, y) position to (row, col) grid indices."""
    x, y = position
    return y, x


def create_empty_grid(width: int = GRID_WIDTH, height: int = GRID_HEIGHT) -> Grid:
    """Create an empty occupancy grid filled with free cells."""
    return [[FREE for _ in range(width)] for _ in range(height)]


def world_to_grid(world_state: WorldState) -> Grid:
    """Convert a WorldState into a 40x40 occupancy grid."""
    grid = create_empty_grid()

    for position in world_state.obstacles:
        if in_bounds(position):
            row, col = to_grid_index(position)
            grid[row][col] = OBSTACLE

    for obj in world_state.objects.values():
        position = obj["pos"]
        if in_bounds(position):
            row, col = to_grid_index(position)
            grid[row][col] = OBSTACLE

    for position in world_state.forbidden_zones:
        if in_bounds(position):
            row, col = to_grid_index(position)
            grid[row][col] = FORBIDDEN

    return grid


if __name__ == "__main__":
    world = WorldState(
        obstacles=[(3, 3), (4, 3)],
        forbidden_zones=[(10, 10)],
    )
    world.update_object("crate", position=(2, 2), object_type="box", graspable=True)

    grid = world_to_grid(world)
    print("grid size:", len(grid), "x", len(grid[0]))
    print("object cell:", grid[2][2])
    print("obstacle cell:", grid[3][3])
    print("forbidden cell:", grid[10][10])
