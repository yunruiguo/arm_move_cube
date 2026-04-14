"""Simple matplotlib visualizations for the planning prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from spatial_map import GRID_HEIGHT, GRID_WIDTH
from world_state import Position, WorldState

EvaluationResult = dict[str, object]
OBJECT_COLORS = {
    "crate_red": "#e31a1c",
    "crate_blue": "#1f78b4",
    "crate_green": "#33a02c",
    "crate_yellow": "#fdbf00",
    "crate_orange": "#ff7f00",
    "statue": "#6a3d9a",
}


def _to_plot_point(position: Position) -> tuple[float, float]:
    """Convert a grid position into plot coordinates centered in the cell."""
    x, y = position
    return x + 0.5, y + 0.5


def _compute_view_limits(
    state: WorldState,
    extra_paths: list[list[Position]] | None = None,
    padding: int = 2,
) -> tuple[float, float, float, float]:
    """Compute a padded bounding box for all relevant plotted elements."""
    positions: list[Position] = [state.get_robot_position()]
    positions.extend(state.obstacles)
    positions.extend(state.forbidden_zones)

    for obj in state.objects.values():
        positions.append(obj["pos"])

    for goal_positions in state.goal_regions.values():
        positions.extend(goal_positions)

    if extra_paths:
        for path in extra_paths:
            positions.extend(path)

    if not positions:
        return 0, GRID_WIDTH, GRID_HEIGHT, 0

    x_values = [position[0] for position in positions]
    y_values = [position[1] for position in positions]

    min_x = max(0, min(x_values) - padding)
    max_x = min(GRID_WIDTH, max(x_values) + 1 + padding)
    min_y = max(0, min(y_values) - padding)
    max_y = min(GRID_HEIGHT, max(y_values) + 1 + padding)
    return min_x, max_x, max_y, min_y


def _compute_tick_step(span: float) -> int:
    """Choose a readable tick interval for the current zoom level."""
    if span <= 12:
        return 1
    if span <= 24:
        return 2
    return 4


def _style_axes(
    ax: plt.Axes,
    title: str,
    view_limits: tuple[float, float, float, float],
) -> None:
    """Apply a consistent grid-map style to an axis."""
    min_x, max_x, max_y, min_y = view_limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.set_aspect("equal")
    x_step = _compute_tick_step(max_x - min_x)
    y_step = _compute_tick_step(max_y - min_y)
    ax.set_xticks(range(int(min_x), int(max_x) + 1, x_step))
    ax.set_yticks(range(int(min_y), int(max_y) + 1, y_step))
    ax.grid(True, color="lightgray", linewidth=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.margins(0)


def _draw_world(ax: plt.Axes, state: WorldState) -> None:
    """Draw the current world state onto an axis."""
    for obstacle in state.obstacles:
        x, y = obstacle
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="black", alpha=0.8))

    for zone in state.forbidden_zones:
        x, y = zone
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#d95f5f", alpha=0.6))

    goal_colors = ["#8dd3c7", "#80b1d3", "#fdb462", "#b3de69"]
    for index, (goal_name, positions) in enumerate(state.goal_regions.items()):
        color = goal_colors[index % len(goal_colors)]
        for goal_position in positions:
            x, y = goal_position
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=color,
                    alpha=0.35,
                    edgecolor=color,
                    linewidth=1.5,
                )
            )
        first_x, first_y = positions[0]
        ax.text(first_x + 0.08, first_y + 0.32, goal_name, fontsize=10, color="black")

    robot_x, robot_y = _to_plot_point(state.get_robot_position())
    ax.scatter(robot_x, robot_y, marker="*", s=260, color="#1f78b4", label="robot")
    ax.text(robot_x + 0.18, robot_y - 0.18, "robot", fontsize=11, color="#1f78b4")

    for name, obj in state.objects.items():
        obj_x, obj_y = _to_plot_point(obj["pos"])
        color = OBJECT_COLORS.get(
            name,
            "#33a02c" if obj["graspable"] else "#6a3d9a",
        )
        ax.scatter(obj_x, obj_y, s=130, color=color, edgecolors="white", linewidth=1.0)
        ax.text(obj_x + 0.14, obj_y - 0.14, name, fontsize=10, color=color)


def _draw_path(
    ax: plt.Axes,
    path: list[Position],
    color: str,
    label: str,
) -> None:
    """Draw a path if it exists."""
    if not path:
        return

    x_values = []
    y_values = []
    for position in path:
        plot_x, plot_y = _to_plot_point(position)
        x_values.append(plot_x)
        y_values.append(plot_y)

    ax.plot(x_values, y_values, color=color, linewidth=3.0, marker="o", markersize=5)
    ax.text(x_values[-1] + 0.15, y_values[-1], label, fontsize=10, color=color)


def _draw_target_marker(
    ax: plt.Axes,
    position: Position,
    label: str,
    color: str,
) -> None:
    """Draw a labeled target marker for a COA destination."""
    plot_x, plot_y = _to_plot_point(position)
    ax.scatter(
        plot_x,
        plot_y,
        marker="X",
        s=170,
        color=color,
        edgecolors="white",
        linewidth=1.0,
        zorder=5,
    )
    ax.text(plot_x + 0.18, plot_y + 0.2, label, fontsize=10, color=color)


def save_world_state_figure(state: WorldState, output_dir: Path) -> Path:
    """Save a static visualization of the world state."""
    output_path = output_dir / "world_state.png"
    fig, ax = plt.subplots(figsize=(10, 9))
    _style_axes(ax, "World State", _compute_view_limits(state))
    _draw_world(ax, state)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_coa_trace_figures(
    state: WorldState,
    results: list[EvaluationResult],
    output_dir: Path,
) -> list[Path]:
    """Save one trace figure per COA."""
    output_paths: list[Path] = []

    for index, result in enumerate(results, start=1):
        output_path = output_dir / f"coa_{index}_trace.png"
        fig, ax = plt.subplots(figsize=(12, 9))
        view_limits = _compute_view_limits(
            state,
            extra_paths=[result["pick_result"]["path"], result["place_result"]["path"]],
        )
        _style_axes(ax, f"COA Trace {index}: {result['name']}", view_limits)
        _draw_world(ax, state)

        pick_result = result["pick_result"]
        place_result = result["place_result"]
        place_position = result["place_position"]
        assert isinstance(place_position, tuple)
        _draw_path(ax, pick_result["path"], "#1f78b4", "1: pick")
        _draw_path(ax, place_result["path"], "#33a02c", "2: place")
        _draw_target_marker(ax, place_position, "target", "#111111")

        summary_lines = [
            f"name: {result['name']}",
            f"object: {result['object']}",
            f"target position: {result['place_position']}",
            f"total score: {result['score']}",
            f"pick score: {result['pick_score']}",
            f"place score: {result['place_score']}",
            f"adjustment: {result['bonus']}",
            f"success: {result['success']}",
            f"reason: {result['reason']}",
        ]
        summary_text = "\n".join(summary_lines)
        box_color = "#e6f4ea" if result["success"] else "#fde0dd"
        ax.text(
            1.02,
            0.98,
            summary_text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": box_color, "edgecolor": "gray", "boxstyle": "round,pad=0.4"},
        )

        if not pick_result["path"]:
            ax.text(1.02, 0.40, "pick path unavailable", transform=ax.transAxes, fontsize=9)
        if not place_result["path"]:
            ax.text(1.02, 0.35, "place path unavailable", transform=ax.transAxes, fontsize=9)

        fig.subplots_adjust(left=0.07, right=0.77, top=0.93, bottom=0.08)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_score_comparison_figure(
    results: list[EvaluationResult],
    best_result: EvaluationResult,
    output_dir: Path,
) -> Path:
    """Save a score comparison chart across all COAs."""
    output_path = output_dir / "coa_comparison.png"
    fig, ax = plt.subplots(figsize=(14, 6))

    names = [str(result["name"]) for result in results]
    total_scores = [int(result["score"]) for result in results]
    colors = []
    for result in results:
        if result["name"] == best_result["name"]:
            colors.append("#33a02c")
        elif result["success"]:
            colors.append("#80b1d3")
        else:
            colors.append("#fb9a99")

    bars = ax.bar(range(len(results)), total_scores, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("COA Score Comparison")
    ax.set_ylabel("total score")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=35, ha="right")

    for bar, result in zip(bars, results):
        label = (
            f"T={result['score']}\n"
            f"P={result['pick_score']} Pl={result['place_score']} A={result['bonus']}"
        )
        height = bar.get_height()
        offset = 1 if height >= 0 else -1
        va = "bottom" if height >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            label,
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.text(
        0.01,
        0.98,
        (
            f"best coa: {best_result['name']}\n"
            f"score: {best_result['score']}\n"
            f"reason: {best_result['reason']}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "#e6f4ea", "edgecolor": "#33a02c", "boxstyle": "round,pad=0.4"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_visualizations(
    state: WorldState,
    results: list[EvaluationResult],
    best_result: EvaluationResult,
    output_dir: str = "outputs",
) -> list[Path]:
    """Generate all visualization files for the current run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_paths = [
        save_world_state_figure(state, output_path),
        save_score_comparison_figure(results, best_result, output_path),
    ]
    generated_paths.extend(save_coa_trace_figures(state, results, output_path))
    return generated_paths
