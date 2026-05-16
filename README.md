# 🤖 Intelligent Robot Move Cubes

**A decision-making platform for robotic cube rearrangement with physics-backed simulation.**

<div align="center">

![Franka Robot Cube Rearrangement](docs/media/arm_move_cube_rollout.gif)

*Five-cube rearrangement executed by Franka robot arm in Isaac Lab*

</div>

---

## 📋 Overview

This project implements **high-level decision logic** for robotic manipulation tasks. A Franka robot arm efficiently moves cubes from a plus-shaped tabletop layout into a rigid basket while maintaining physics accuracy and collision awareness. The system runs in simulation using [Isaac Lab](https://docs.isaacsim.io/) and demonstrates intelligent planning strategies for multi-object manipulation.

### Key Highlights

✨ **Multi-object task planning** – Intelligently order cube movements to minimize collisions  
🚫 **Blocking-aware strategies** – Clear obstructing objects before reaching blocked cubes  
🧮 **Rigid context modeling** – Account for fixed basket geometry and collision constraints  
📦 **Physics continuity** – Preserve position and orientation accuracy across rollouts  
📊 **Visual analysis** – Generate rollout artifacts and GIF animations for validation  

---

## 🎯 Strategy Selection

The system evaluates multiple planning strategies to determine the optimal execution order:

| Strategy | Description | Status |
|----------|-------------|--------|
| **Fixed Naive Order** | Execute moves in predefined sequence | Baseline |
| **Nearest-First Heuristic** | Prioritize closest cubes | Alternative |
| **Blocking-Aware Priority** | Clear blocking objects first | ✅ **Preferred** |

### Proven Approach: `clear_blocking_first_plus_shape`

The blocking-aware strategy executes moves in this order:

```
cube_north → cube_east → cube_south → cube_west → cube_center
```

This ensures surrounding cubes are cleared before attempting the center cube, preventing collisions and failed grasps.

---

## 📁 Project Structure

```
.
├── README.md                                          # This file
├── AGENTS.md                                          # Development guide
├── requirements.txt                                   # Python dependencies
├── test_env.py                                        # Environment verification
├── multi_cube_basket_demo.py                          # Main strategy showcase
├── record_franka_pick_place_animation.py              # Rollout recording & GIF generation
├── experiment_runner_real.py                          # Structured Isaac Lab runner
├── ARM_MOVE_CUBE_SHOWCASE_TASK_DESCRIPTION.md         # Detailed task specification
└── docs/
    └── media/
        └── arm_move_cube_rollout.gif                  # Demo visualization
```

### Core Files

- **`multi_cube_basket_demo.py`** – Five-cube showcase runner with strategy comparison
- **`record_franka_pick_place_animation.py`** – Isaac Lab rollout recording, GIF generation, physics context modeling
- **`experiment_runner_real.py`** – Structured runner for Isaac Lab experiments
- **`ARM_MOVE_CUBE_SHOWCASE_TASK_DESCRIPTION.md`** – Comprehensive task specification and implementation notes

---

## 🚀 Getting Started

### Prerequisites

- A100 GPU with Isaac Lab environment
- Miniconda/Conda environment with `isaac311`
- Python 3.10+

### Setup

```bash
# Navigate to project directory
cd ~/decision_platform

# Activate Isaac Lab environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac311

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Run the Demo

```bash
# Execute the five-cube showcase
python multi_cube_basket_demo.py
```

### Output Location

Results are written to:
```
/mnt/data2/outputs/showcase_demo_five_cubes_plus_shape_planning/
```

---

## 🔬 How It Works

1. **Task Definition** – Define cube positions and basket target
2. **Strategy Evaluation** – Compare planning strategies (naive, nearest-first, blocking-aware)
3. **Optimal Sequence Generation** – Select the best strategy and generate move sequence
4. **Simulator Execution** – Execute moves in Isaac Lab with physics
5. **Rollout Recording** – Capture trajectory and generate visualization GIF
6. **Analysis** – Review results for collision detection and success metrics

---

## 📊 Performance Metrics

The system tracks:
- **Move success rate** – Percentage of successful cube pickups
- **Collision avoidance** – Detection of basket/cube interactions
- **Execution time** – Total simulation time for rearrangement
- **Physics accuracy** – Position/orientation drift across moves

---

## 🛠️ Development

### Code Style

- Python 3.10+ only
- Minimal external dependencies
- Explicit, readable implementations
- Clear logging for debugging

### Running Tests

```bash
python -m pytest
```

Or for standard library tests:
```bash
python -m unittest discover
```

### Adding Features

Refer to **`AGENTS.md`** for the development workflow and phase-based delivery guidelines.

---

## 📖 Additional Resources

- **Isaac Lab Documentation** – [docs.isaacsim.io](https://docs.isaacsim.io/)
- **Task Specification** – See `ARM_MOVE_CUBE_SHOWCASE_TASK_DESCRIPTION.md`
- **Development Guide** – See `AGENTS.md`

---

## 📝 License

This project is provided as-is for research and educational purposes.

---

## 👤 Author

**yunruiguo** – [GitHub Profile](https://github.com/yunruiguo)

---

<div align="center">

**Built with** 🐍 Python • 🤖 Isaac Lab • 🎯 Robotics Planning

</div>
