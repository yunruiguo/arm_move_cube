"""Minimal benchmark comparison plots for real Franka experiments."""

from __future__ import annotations

from pathlib import Path


def save_benchmark_comparison_plot(
    results: list[dict[str, object]],
    output_dir: str | Path,
) -> Path:
    """Save a simple planner/scenario comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenarios = []
    planner_modes = []
    for result in results:
        scenario_name = str(result["scenario_name"])
        planner_mode = str(result["planner_mode"])
        if scenario_name not in scenarios:
            scenarios.append(scenario_name)
        if planner_mode not in planner_modes:
            planner_modes.append(planner_mode)

    scores_by_planner: dict[str, list[float]] = {}
    success_by_planner: dict[str, list[int]] = {}
    for planner_mode in planner_modes:
        planner_scores = []
        planner_successes = []
        for scenario_name in scenarios:
            matching_result = next(
                result
                for result in results
                if result["scenario_name"] == scenario_name
                and result["planner_mode"] == planner_mode
            )
            planner_scores.append(float(matching_result["total_score"]))
            planner_successes.append(1 if matching_result["success"] else 0)
        scores_by_planner[planner_mode] = planner_scores
        success_by_planner[planner_mode] = planner_successes

    planner_colors = {
        "fixed_order": "#6c7a89",
        "nearest_heuristic": "#3d7ea6",
        "decision_pipeline": "#1f9d55",
    }
    x_positions = list(range(len(scenarios)))
    bar_width = 0.22

    figure, (score_axis, success_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        constrained_layout=True,
        height_ratios=(3, 1.5),
    )

    for planner_index, planner_mode in enumerate(planner_modes):
        offsets = [
            x_position + (planner_index - (len(planner_modes) - 1) / 2) * bar_width
            for x_position in x_positions
        ]
        color = planner_colors.get(planner_mode, "#4c566a")
        score_axis.bar(
            offsets,
            scores_by_planner[planner_mode],
            width=bar_width,
            label=planner_mode,
            color=color,
        )
        success_axis.bar(
            offsets,
            success_by_planner[planner_mode],
            width=bar_width,
            color=color,
        )

    score_axis.set_title("Real Franka Benchmark Comparison")
    score_axis.set_ylabel("Total score")
    score_axis.set_xticks(x_positions, scenarios)
    score_axis.grid(axis="y", alpha=0.25)
    score_axis.legend()

    success_axis.set_ylabel("Success")
    success_axis.set_xlabel("Scenario")
    success_axis.set_xticks(x_positions, scenarios)
    success_axis.set_yticks([0, 1], ["fail", "success"])
    success_axis.set_ylim(-0.1, 1.2)
    success_axis.grid(axis="y", alpha=0.25)

    figure_path = output_path / "benchmark_comparison.png"
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return figure_path
