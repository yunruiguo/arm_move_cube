# Isaac Lab Decision Showcase Task Description

## Project Snapshot

This repository contains a simulator-grounded decision platform that connects high-level course-of-action planning with Isaac Lab / Isaac Sim rollout evidence. The current showcase focuses on a Franka robot manipulating tabletop cubes into a basket while preserving the separation between:

- environment and simulator backend
- world / scenario representation
- candidate strategy generation
- reachability and blocking evaluation
- low-level pick-place rollout
- visualization and result artifacts

The current codebase keeps the original toy benchmark path intact while adding real Isaac Lab demonstration scripts for multi-object manipulation.

## Current Showcase

The active showcase is a five-cube plus-shape tabletop scenario. The object set is:

- `cube_center`
- `cube_north`
- `cube_east`
- `cube_south`
- `cube_west`

The planning objective is to move cubes into a basket while respecting blocking constraints in the plus-shaped layout. The decision layer compares candidate strategies before invoking real rollout execution.

The current planner candidates are:

- `fixed_order_center_first`
- `nearest_first_plus_shape`
- `clear_blocking_first_plus_shape`

The selected strategy in the successful baseline is `clear_blocking_first_plus_shape`, which clears surrounding cubes before attempting the center cube.

## Recent Engineering Changes

The latest version adds or preserves the following capabilities:

- A multi-cube basket runner in `multi_cube_basket_demo.py`.
- Sequential Isaac rollout execution so each subtask runs in its own process and avoids long-lived simulator accumulation.
- A rigid basket representation with collision geometry for the tray and walls.
- Rigid context cubes for previously placed objects, replacing visual-only context markers.
- Physical state continuity across subtasks:
  - previous cube positions are read from physics-backed rollout results
  - previous cube orientations are stored and passed into the next subtask
- A grasp-orientation selector that can pass a preferred end-effector orientation into the default Isaac Franka pick-place controller.
- Manifest fields for:
  - physical cube translation
  - physical cube orientation
  - expected cube position
  - target end-effector orientation
  - orientation selection reason
  - phase debug timeline
- A packed plus-shape variant where cubes are placed edge-to-edge to stress the current controller and make blocking/contact limitations visible.

## Important Behavioral Notes

The successful five-cube basket demo demonstrates the full planning and execution pipeline, including real GIF output and physical state propagation.

The packed plus-shape stress test intentionally increases contact difficulty by placing cubes edge-to-edge. In the latest run, this packed variant failed on the first subtask because the default Isaac pick-place controller could not reliably extract a tightly packed cube. This is useful evidence for the next decision-focused step: high-level planning should reason not only about object order, but also about grasp approach direction.

The current grasp-orientation pathway is already wired, but the packed scenario exposes that the heuristic still needs refinement. The next natural step is to make `grasp_orientation_mode` an explicit action parameter in candidate strategies rather than an internal low-level heuristic.

## Key Output Locations On A100

The standard five-cube output directory is:

```text
/mnt/data2/outputs/showcase_demo_five_cubes_plus_shape_planning/
```

The packed plus-shape stress-test output directory is:

```text
/mnt/data2/outputs/showcase_demo_five_cubes_plus_shape_packed/
```

Each run writes:

- `combined/rollout.gif`
- `combined_summary.json`
- per-subtask `manifest.json`
- per-subtask `debug_summary.txt`
- per-subtask `rollout.gif`
- per-subtask frame directories

## Validation Commands

Local syntax validation:

```bash
cd /Users/gyr/codex/platform
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  record_franka_pick_place_animation.py \
  multi_cube_basket_demo.py
```

Remote execution on A100:

```bash
cd ~/decision_platform
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac311
python multi_cube_basket_demo.py
```

## Current Status

The code is in a useful checkpoint state for the next phase of work:

- The real Isaac Lab execution path works.
- Multi-cube physical state continuity is implemented.
- Basket and context cubes have rigid-body semantics.
- The planning showcase can explain why `clear_blocking_first` is preferred over naive orders.
- The packed stress test reveals a clear next research-engineering target: explicit grasp orientation planning.

