"""Predefined deterministic benchmark scenarios for the real Franka runner."""

from __future__ import annotations

from dataclasses import dataclass

from world_state import WorldState


@dataclass(frozen=True)
class RealScenario:
    """One deterministic planner benchmark mapped onto a single real-cube rollout."""

    name: str
    description: str
    robot_grid_position: tuple[int, int]
    fixed_order: tuple[str, ...]
    object_grid_positions: dict[str, tuple[int, int]]
    object_sim_positions: dict[str, tuple[float, float, float]]
    object_goal_regions: dict[str, str]
    goal_region_grid_positions: dict[str, tuple[int, int]]
    goal_region_sim_positions: dict[str, tuple[float, float, float]]
    obstacles: tuple[tuple[int, int], ...] = ()
    forbidden_zones: tuple[tuple[int, int], ...] = ()


def _build_world_state(config: RealScenario) -> WorldState:
    """Convert a scenario definition into the current WorldState format."""
    state = WorldState(
        robot_position=config.robot_grid_position,
        obstacles=list(config.obstacles),
        forbidden_zones=list(config.forbidden_zones),
        goal_regions={
            region_name: [grid_position]
            for region_name, grid_position in config.goal_region_grid_positions.items()
        },
    )
    for object_name, grid_position in config.object_grid_positions.items():
        state.update_object(
            object_name,
            position=grid_position,
            object_type="cube",
            graspable=True,
        )
    return state


REAL_BENCHMARK_SCENARIOS: dict[str, RealScenario] = {
    # Stable fallback scenario: one cube, one goal region.
    "single_cube_goal": RealScenario(
        name="single_cube_goal",
        description=(
            "A minimal single-cube, single-goal tabletop delivery setup using the "
            "current stable smooth-transport configuration and a moderate far target."
        ),
        robot_grid_position=(20, 20),
        fixed_order=("cube_alpha",),
        object_grid_positions={
            "cube_alpha": (24, 18),
        },
        object_sim_positions={
            "cube_alpha": (0.46, -0.14, 0.0258),
        },
        object_goal_regions={
            "cube_alpha": "staging_goal",
        },
        goal_region_grid_positions={
            "staging_goal": (19, 23),
        },
        goal_region_sim_positions={
            "staging_goal": (-0.1466, 0.3887, 0.08),
        },
    ),
    "easy_clear": RealScenario(
        name="easy_clear",
        description="Three clear candidate cubes with open access and short routes.",
        robot_grid_position=(20, 20),
        fixed_order=("cube_alpha", "cube_beta", "cube_gamma"),
        object_grid_positions={
            "cube_alpha": (22, 20),
            "cube_beta": (24, 18),
            "cube_gamma": (24, 23),
        },
        object_sim_positions={
            "cube_alpha": (0.45, 0.00, 0.0258),
            "cube_beta": (0.52, -0.08, 0.0258),
            "cube_gamma": (0.52, 0.10, 0.0258),
        },
        object_goal_regions={
            "cube_alpha": "staging_goal",
            "cube_beta": "left_goal",
            "cube_gamma": "right_goal",
        },
        goal_region_grid_positions={
            "left_goal": (15, 16),
            "staging_goal": (16, 20),
            "right_goal": (15, 24),
        },
        goal_region_sim_positions={
            "left_goal": (-0.26, -0.20, 0.12),
            "staging_goal": (-0.22, 0.00, 0.12),
            "right_goal": (-0.26, 0.20, 0.12),
        },
    ),
    "awkward_ordering": RealScenario(
        name="awkward_ordering",
        description=(
            "The first cube in fixed order is far away, while a much easier cube sits near the robot."
        ),
        robot_grid_position=(20, 20),
        fixed_order=("cube_far", "cube_near", "cube_mid"),
        object_grid_positions={
            "cube_far": (28, 26),
            "cube_near": (21, 19),
            "cube_mid": (24, 21),
        },
        object_sim_positions={
            "cube_far": (0.62, 0.18, 0.0258),
            "cube_near": (0.41, -0.02, 0.0258),
            "cube_mid": (0.50, 0.05, 0.0258),
        },
        object_goal_regions={
            "cube_far": "right_goal",
            "cube_near": "staging_goal",
            "cube_mid": "left_goal",
        },
        goal_region_grid_positions={
            "left_goal": (15, 16),
            "staging_goal": (16, 20),
            "right_goal": (15, 24),
        },
        goal_region_sim_positions={
            "left_goal": (-0.28, -0.20, 0.12),
            "staging_goal": (-0.22, 0.00, 0.12),
            "right_goal": (-0.28, 0.20, 0.12),
        },
    ),
    "mildly_blocked": RealScenario(
        name="mildly_blocked",
        description=(
            "One nearby cube looks attractive but sits behind mild planner obstacles, so rule-aware choice matters."
        ),
        robot_grid_position=(20, 20),
        fixed_order=("cube_blocked", "cube_clear", "cube_backup"),
        object_grid_positions={
            "cube_blocked": (22, 20),
            "cube_clear": (25, 18),
            "cube_backup": (26, 23),
        },
        object_sim_positions={
            "cube_blocked": (0.46, 0.00, 0.0258),
            "cube_clear": (0.54, -0.08, 0.0258),
            "cube_backup": (0.56, 0.10, 0.0258),
        },
        object_goal_regions={
            "cube_blocked": "staging_goal",
            "cube_clear": "left_goal",
            "cube_backup": "right_goal",
        },
        goal_region_grid_positions={
            "left_goal": (15, 16),
            "staging_goal": (16, 20),
            "right_goal": (15, 24),
        },
        goal_region_sim_positions={
            "left_goal": (-0.26, -0.18, 0.12),
            "staging_goal": (-0.22, 0.00, 0.12),
            "right_goal": (-0.26, 0.22, 0.12),
        },
        obstacles=((21, 20), (21, 21), (22, 21), (23, 21)),
        forbidden_zones=((16, 21),),
    ),
    "showcase_delivery_gauntlet": RealScenario(
        name="showcase_delivery_gauntlet",
        description=(
            "A richer five-object tabletop delivery scene with mild blocking, asymmetric "
            "travel costs, and clearly suboptimal fixed ordering."
        ),
        robot_grid_position=(20, 20),
        fixed_order=(
            "cube_far_blocked",
            "cube_far_right",
            "cube_center_anchor",
            "cube_near_left",
            "cube_near_right",
        ),
        object_grid_positions={
            "cube_far_blocked": (29, 24),
            "cube_far_right": (28, 27),
            "cube_center_anchor": (24, 21),
            "cube_near_left": (21, 18),
            "cube_near_right": (22, 23),
        },
        object_sim_positions={
            "cube_far_blocked": (0.64, 0.12, 0.0258),
            "cube_far_right": (0.66, 0.20, 0.0258),
            "cube_center_anchor": (0.53, 0.02, 0.0258),
            "cube_near_left": (0.43, -0.09, 0.0258),
            "cube_near_right": (0.45, 0.09, 0.0258),
        },
        object_goal_regions={
            "cube_far_blocked": "right_goal",
            "cube_far_right": "right_goal",
            "cube_center_anchor": "staging_goal",
            "cube_near_left": "left_goal",
            "cube_near_right": "staging_goal",
        },
        goal_region_grid_positions={
            "left_goal": (15, 16),
            "staging_goal": (16, 20),
            "right_goal": (15, 24),
        },
        goal_region_sim_positions={
            "left_goal": (-0.27, -0.20, 0.12),
            "staging_goal": (-0.21, 0.00, 0.12),
            "right_goal": (-0.27, 0.22, 0.12),
        },
        obstacles=((22, 20), (23, 20), (24, 20), (25, 21), (26, 22)),
        forbidden_zones=((17, 21), (18, 21)),
    ),
}


def get_real_benchmark_scenario(name: str) -> RealScenario:
    """Return a named real benchmark scenario."""
    if name not in REAL_BENCHMARK_SCENARIOS:
        raise ValueError(
            f"Unknown real benchmark scenario '{name}'. "
            f"Expected one of: {', '.join(sorted(REAL_BENCHMARK_SCENARIOS))}."
        )
    return REAL_BENCHMARK_SCENARIOS[name]


def build_real_benchmark_world(name: str) -> WorldState:
    """Build the planner-facing world state for a named real benchmark scenario."""
    return _build_world_state(get_real_benchmark_scenario(name))
