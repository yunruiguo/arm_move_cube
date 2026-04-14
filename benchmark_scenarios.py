"""Predefined benchmark scenarios for the planning prototype."""

from __future__ import annotations

from world_state import WorldState

ScenarioBuilder = tuple[str, callable]


def _populate_objects(world_state: WorldState, positions: dict[str, tuple[int, int]]) -> None:
    """Populate a world state with the standard benchmark object set."""
    world_state.update_object(
        "crate_red", position=positions["crate_red"], object_type="crate", graspable=True
    )
    world_state.update_object(
        "crate_blue", position=positions["crate_blue"], object_type="crate", graspable=True
    )
    world_state.update_object(
        "crate_green", position=positions["crate_green"], object_type="crate", graspable=True
    )
    world_state.update_object(
        "crate_yellow",
        position=positions["crate_yellow"],
        object_type="crate",
        graspable=True,
    )
    world_state.update_object(
        "crate_orange",
        position=positions["crate_orange"],
        object_type="crate",
        graspable=True,
    )
    world_state.update_object(
        "statue", position=positions["statue"], object_type="decor", graspable=False
    )


def build_benchmark_world() -> WorldState:
    """Create the default deterministic benchmark world."""
    world_state = WorldState(
        robot_position=(0, 0),
        obstacles=[
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (2, 7),
            (3, 7),
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
    _populate_objects(
        world_state,
        {
            "crate_red": (2, 1),
            "crate_blue": (5, 1),
            "crate_green": (3, 8),
            "crate_yellow": (8, 1),
            "crate_orange": (11, 8),
            "statue": (9, 9),
        },
    )
    return world_state


def build_corridor_challenge_world() -> WorldState:
    """Create a scenario with a narrow corridor and far-right deliveries."""
    world_state = WorldState(
        robot_position=(0, 0),
        obstacles=[
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (10, 2),
            (6, 6),
            (7, 6),
            (8, 6),
            (9, 6),
            (10, 6),
        ],
        forbidden_zones=[(11, 3), (11, 4), (12, 3), (12, 4)],
        goal_regions={
            "left_goal": [(1, 8), (2, 8)],
            "right_goal": [(12, 8), (12, 9)],
            "staging_goal": [(5, 9), (6, 9)],
        },
    )
    _populate_objects(
        world_state,
        {
            "crate_red": (1, 1),
            "crate_blue": (2, 4),
            "crate_green": (5, 8),
            "crate_yellow": (8, 1),
            "crate_orange": (10, 8),
            "statue": (9, 8),
        },
    )
    return world_state


def build_goal_pressure_world() -> WorldState:
    """Create a scenario where grouped goal assignments compete with access costs."""
    world_state = WorldState(
        robot_position=(0, 0),
        obstacles=[
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (8, 4),
            (9, 4),
            (10, 4),
            (11, 4),
            (2, 6),
            (3, 6),
            (4, 6),
            (8, 7),
            (9, 7),
        ],
        forbidden_zones=[(10, 8), (10, 9), (11, 8), (11, 9)],
        goal_regions={
            "left_goal": [(1, 9), (2, 9)],
            "right_goal": [(12, 6), (12, 7)],
            "staging_goal": [(6, 8), (7, 8)],
        },
    )
    _populate_objects(
        world_state,
        {
            "crate_red": (1, 2),
            "crate_blue": (4, 1),
            "crate_green": (3, 8),
            "crate_yellow": (7, 1),
            "crate_orange": (9, 8),
            "statue": (8, 9),
        },
    )
    return world_state


def get_benchmark_scenarios() -> list[tuple[str, callable]]:
    """Return the predefined benchmark scenarios."""
    return [
        ("default_benchmark", build_benchmark_world),
        ("corridor_challenge", build_corridor_challenge_world),
        ("goal_pressure", build_goal_pressure_world),
    ]
