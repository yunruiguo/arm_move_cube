"""Minimal world state for a fully observable 2D planning environment."""

from __future__ import annotations

from typing import Any

Position = tuple[int, int]
ObjectState = dict[str, Any]


class WorldState:
    """Store robot, object, and region data for a simple 2D world."""

    def __init__(
        self,
        robot_position: Position = (0, 0),
        objects: dict[str, ObjectState] | None = None,
        obstacles: list[Position] | None = None,
        forbidden_zones: list[Position] | None = None,
        goal_regions: dict[str, list[Position]] | None = None,
    ) -> None:
        """Initialize the world state with explicit in-memory containers."""
        self.robot_position: Position = robot_position
        self.objects: dict[str, ObjectState] = objects.copy() if objects else {}
        self.obstacles: list[Position] = list(obstacles) if obstacles else []
        self.forbidden_zones: list[Position] = (
            list(forbidden_zones) if forbidden_zones else []
        )
        self.goal_regions: dict[str, list[Position]] = (
            {name: list(positions) for name, positions in goal_regions.items()}
            if goal_regions
            else {}
        )

    def update_robot_position(self, position: Position) -> None:
        """Update the robot's current position."""
        self.robot_position = position

    def update_object(
        self,
        name: str,
        position: Position,
        object_type: str,
        graspable: bool,
    ) -> None:
        """Create or replace an object's state."""
        self.objects[name] = {
            "pos": position,
            "type": object_type,
            "graspable": graspable,
        }

    def get_robot_position(self) -> Position:
        """Return the robot's current position."""
        return self.robot_position

    def get_object_position(self, name: str) -> Position | None:
        """Return an object's position if the object exists."""
        obj = self.objects.get(name)
        if obj is None:
            return None
        return obj["pos"]

    def get_object(self, name: str) -> ObjectState | None:
        """Return the full object record if it exists."""
        return self.objects.get(name)

    def is_occupied(self, position: Position) -> bool:
        """Return True if a position is blocked by an obstacle or object."""
        if position in self.obstacles:
            return True

        for obj in self.objects.values():
            if obj["pos"] == position:
                return True

        return False


if __name__ == "__main__":
    world = WorldState(
        robot_position=(1, 2),
        obstacles=[(5, 5)],
        forbidden_zones=[(9, 9)],
        goal_regions={"pickup_zone": [(3, 3), (3, 4)]},
    )
    world.update_object("box", position=(2, 2), object_type="crate", graspable=True)
    world.update_robot_position((2, 1))

    print("robot:", world.get_robot_position())
    print("box:", world.get_object("box"))
    print("box position:", world.get_object_position("box"))
    print("occupied (2, 2):", world.is_occupied((2, 2)))
    print("occupied (5, 5):", world.is_occupied((5, 5)))
    print("occupied (0, 0):", world.is_occupied((0, 0)))
