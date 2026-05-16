# arm_move_cube

Simulator-grounded decision platform for robotic cube rearrangement with Isaac Lab.

![Franka five-cube decision rollout](docs/media/arm_move_cube_rollout.gif)

## Showcase

This demo connects high-level decision logic with simulator-backed execution. A Franka arm moves cubes from a plus-shaped tabletop layout into a rigid basket while preserving physical state across subtasks.

The current showcase highlights:

- multi-object task planning
- ordering constraints for blocked objects
- rigid context objects and basket collision geometry
- physics-backed position and orientation continuity
- rollout artifacts for visual inspection

The latest successful run uses the `clear_blocking_first_plus_shape` strategy:

```text
cube_north -> cube_east -> cube_south -> cube_west -> cube_center
```

The decision logic compares naive fixed order, nearest-first, and blocking-aware strategies, then selects the feasible order that clears surrounding cubes before retrieving the center cube.

## Key Files

- `multi_cube_basket_demo.py`: five-cube showcase runner and strategy evaluation.
- `record_franka_pick_place_animation.py`: Isaac Lab rollout recording, rigid context cubes, basket, manifests, and GIF generation.
- `ARM_MOVE_CUBE_SHOWCASE_TASK_DESCRIPTION.md`: detailed task description and implementation notes.
- `experiment_runner_real.py`: structured experiment runner for real Isaac Lab scenarios.

## Run

On the A100 Isaac Lab environment:

```bash
cd ~/decision_platform
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac311
python multi_cube_basket_demo.py
```

Outputs are written under:

```text
/mnt/data2/outputs/showcase_demo_five_cubes_plus_shape_planning/
```

