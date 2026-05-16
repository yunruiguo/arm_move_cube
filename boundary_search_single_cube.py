"""Small helper to probe reachable single-cube target points on the real backend."""

from __future__ import annotations

from pathlib import Path

from record_franka_pick_place_animation import record_pick_place_rollout


START_POSITION = (0.46, -0.14, 0.0258)
CANDIDATES = [
    ("candidate_a", (0.06, 0.16, 0.05)),
    ("candidate_b", (0.00, 0.24, 0.05)),
    ("candidate_c", (-0.06, 0.32, 0.05)),
]


def main() -> None:
    base_output_dir = Path("/mnt/data2/outputs/showcase_demo_single_cube_boundary_search")
    for name, target in CANDIDATES:
        summary = record_pick_place_rollout(
            output_dir=base_output_dir / name,
            planner_name="boundary_search",
            selected_strategy=name,
            selected_target_object="cube_alpha",
            cube_initial_position=START_POSITION,
            target_position=target,
            scene_metadata={
                "name": name,
                "description": "single cube boundary search",
                "object_sim_positions": {"cube_alpha": list(START_POSITION)},
                "goal_region_sim_positions": {"staging_goal": list(target)},
            },
        )
        subtask = summary["subtasks"][0] if "subtasks" in summary else summary
        print(name)
        print(f"  target  = {subtask.get('end_effector_target_position')}")
        print(f"  final   = {subtask.get('physical_cube_translation')}")
        print(f"  error   = {subtask.get('position_error')}")
        print(f"  success = {subtask.get('success')}")


if __name__ == "__main__":
    main()
