"""Visualization helpers for the stronger showcase demo."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from benchmark_scenarios_real import RealScenario
from world_state import WorldState

PLANNER_COLORS = {
    "fixed_order": "#6c7a89",
    "nearest_first": "#3d7ea6",
    "clear_blocking_first": "#c97b1f",
    "decision_pipeline": "#1f9d55",
    "llm_generated": "#7f5aa2",
}


def configure_matplotlib() -> None:
    """Use a writable matplotlib cache directory."""
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "matplotlib-showcase-demo"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)


def save_scene_overview(
    state: WorldState,
    config: RealScenario,
    output_path: str | Path,
) -> Path:
    """Save a compact top-down scene overview."""
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    robot_x, robot_y = state.get_robot_position()
    axis.scatter([robot_x], [robot_y], s=180, marker="*", color="#111111", label="robot")

    goal_colors = {
        "left_goal": "#3d7ea6",
        "staging_goal": "#1f9d55",
        "right_goal": "#c97b1f",
    }
    for region_name, positions in state.goal_regions.items():
        xs = [position[0] for position in positions]
        ys = [position[1] for position in positions]
        axis.scatter(
            xs,
            ys,
            s=180,
            marker="s",
            color=goal_colors.get(region_name, "#888888"),
            label=region_name,
            alpha=0.8,
        )

    for object_name, object_data in state.objects.items():
        object_x, object_y = object_data["pos"]
        region_name = config.object_goal_regions.get(object_name, "unknown")
        axis.scatter(
            [object_x],
            [object_y],
            s=120,
            color=goal_colors.get(region_name, "#555555"),
            edgecolors="black",
            linewidths=0.8,
        )
        axis.text(object_x + 0.25, object_y + 0.15, object_name, fontsize=9)

    if state.obstacles:
        obstacle_xs = [position[0] for position in state.obstacles]
        obstacle_ys = [position[1] for position in state.obstacles]
        axis.scatter(obstacle_xs, obstacle_ys, s=90, marker="x", color="#aa3333", label="obstacle")

    if state.forbidden_zones:
        forbidden_xs = [position[0] for position in state.forbidden_zones]
        forbidden_ys = [position[1] for position in state.forbidden_zones]
        axis.scatter(
            forbidden_xs,
            forbidden_ys,
            s=90,
            marker="D",
            color="#7f1d1d",
            label="forbidden",
            alpha=0.7,
        )

    all_points = [state.get_robot_position(), *state.obstacles, *state.forbidden_zones]
    all_points.extend(obj["pos"] for obj in state.objects.values())
    all_points.extend(position for positions in state.goal_regions.values() for position in positions)
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    axis.set_xlim(min(xs) - 2, max(xs) + 2)
    axis.set_ylim(min(ys) - 2, max(ys) + 2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"Scene Overview: {config.name}", fontsize=14)
    axis.set_xlabel("Grid x")
    axis.set_ylabel("Grid y")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_strategy_comparison(
    evaluation_records: list[dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Save a compact score and action outcome comparison figure."""
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    planners = [record["planner_mode"] for record in evaluation_records]
    scores = [int(record["total_score"]) for record in evaluation_records]
    completed = [int(record["completed_actions"]) for record in evaluation_records]
    failed = [int(record["failed_actions"]) for record in evaluation_records]
    colors = [PLANNER_COLORS.get(planner, "#666666") for planner in planners]
    x_positions = list(range(len(planners)))

    figure, (score_axis, action_axis) = plt.subplots(
        2, 1, figsize=(10, 7), constrained_layout=True, height_ratios=(2.3, 1.7)
    )
    score_axis.bar(x_positions, scores, color=colors)
    score_axis.set_title("Strategy Comparison", fontsize=14)
    score_axis.set_ylabel("Total score")
    score_axis.set_xticks(x_positions, planners, rotation=10)
    score_axis.grid(axis="y", alpha=0.25)

    bar_width = 0.35
    action_axis.bar(
        [position - bar_width / 2 for position in x_positions],
        completed,
        width=bar_width,
        color="#1f9d55",
        label="completed_actions",
    )
    action_axis.bar(
        [position + bar_width / 2 for position in x_positions],
        failed,
        width=bar_width,
        color="#aa3333",
        label="failed_actions",
    )
    action_axis.set_ylabel("Action count")
    action_axis.set_xlabel("Planner")
    action_axis.set_xticks(x_positions, planners, rotation=10)
    action_axis.grid(axis="y", alpha=0.25)
    action_axis.legend(fontsize=9)

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_selected_plan_trace(
    config: RealScenario,
    selected_plan: dict[str, object],
    output_path: str | Path,
) -> Path:
    """Save a simple trace diagram for the selected action sequence."""
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    goal_colors = {
        "left_goal": "#3d7ea6",
        "staging_goal": "#1f9d55",
        "right_goal": "#c97b1f",
    }
    for region_name, goal_position in config.goal_region_grid_positions.items():
        axis.scatter(
            [goal_position[0]],
            [goal_position[1]],
            s=180,
            marker="s",
            color=goal_colors.get(region_name, "#666666"),
            alpha=0.85,
        )
        axis.text(goal_position[0] + 0.2, goal_position[1] + 0.2, region_name, fontsize=9)

    for index, action in enumerate(selected_plan["coa"]["actions"], start=1):
        object_name = str(action["object"])
        object_position = config.object_grid_positions[object_name]
        goal_region_name = str(action.get("goal_region", config.object_goal_regions[object_name]))
        goal_position = action["place_position"]
        color = goal_colors.get(goal_region_name, "#555555")
        axis.scatter([object_position[0]], [object_position[1]], s=110, color=color, edgecolors="black")
        axis.annotate(
            "",
            xy=goal_position,
            xytext=object_position,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
        )
        midpoint = ((object_position[0] + goal_position[0]) / 2, (object_position[1] + goal_position[1]) / 2)
        axis.text(midpoint[0], midpoint[1], str(index), fontsize=10, weight="bold")
        axis.text(object_position[0] + 0.2, object_position[1] - 0.35, object_name, fontsize=8)

    planner_label = selected_plan.get("planner_name", selected_plan.get("planner_mode", "selected"))
    axis.set_title(f"Selected Plan Trace: {planner_label}", fontsize=14)
    axis.set_xlabel("Grid x")
    axis.set_ylabel("Grid y")
    axis.grid(alpha=0.2)
    axis.set_aspect("equal", adjustable="box")

    all_points = list(config.object_grid_positions.values()) + list(config.goal_region_grid_positions.values())
    all_points.append(config.robot_grid_position)
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    axis.set_xlim(min(xs) - 2, max(xs) + 2)
    axis.set_ylim(min(ys) - 2, max(ys) + 2)

    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
