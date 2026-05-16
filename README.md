# arm_move_cube

Simulator-grounded decision platform for robotic cube rearrangement with Isaac Lab.

![Franka five-cube decision rollout](docs/media/arm_move_cube_rollout.gif)

## Overview

This project demonstrates high-level decision logic integrated with simulator-backed execution. A Franka robot arm moves cubes from a plus-shaped tabletop layout into a rigid basket while maintaining physics-based state consistency.

## Key Capabilities

- **Multi-object task planning** – intelligently order cube movements
- **Blocking-aware strategies** – clear obstructing objects before retrieving blocked ones
- **Rigid context handling** – model fixed basket geometry and collision constraints
- **Physics continuity** – preserve position and orientation accuracy across rollouts
- **Visual inspection** – generate rollout artifacts and GIF animations for analysis

## Strategy Selection

The system evaluates multiple planning strategies:
- Fixed naive order
- Nearest-first heuristic  
- Blocking-aware prioritization (currently preferred)

The **`clear_blocking_first_plus_shape`** strategy is the proven approach, executing moves in order:

```text
cube_north → cube_east → cube_south → cube_west → cube_center
```

This ensures surrounding cubes are cleared before attempting the center cube.

## Key Files

- **`multi_cube_basket_demo.py`** – Five-cube showcase runner and strategy comparison
- **`record_franka_pick_place_animation.py`** – Isaac Lab rollout recording, GIF generation, and context modeling
- **`ARM_MOVE_CUBE_SHOWCASE_TASK_DESCRIPTION.md`** – Detailed task specification and implementation notes
- **`experiment_runner_real.py`** – Structured runner for Isaac Lab experiments

## Setup and Execution

On the A100 Isaac Lab environment:

```bash
cd ~/decision_platform
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac311
python multi_cube_basket_demo.py
```

Outputs are written to:

```text
/mnt/data2/outputs/showcase_demo_five_cubes_plus_shape_planning/
```
