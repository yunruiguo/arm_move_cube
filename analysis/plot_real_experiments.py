"""Create summary tables and publication-style plots for real experiment runs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


DEFAULT_RESULTS_ROOT = Path("/data2/outputs/real_experiments")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "results_summary.csv"
DEFAULT_FIGURES_DIR = SCRIPT_DIR / "figures"
PLANNER_ORDER = ["fixed_order", "nearest_heuristic", "decision_pipeline"]
PLANNER_COLORS = {
    "fixed_order": "#6c7a89",
    "nearest_heuristic": "#3d7ea6",
    "decision_pipeline": "#1f9d55",
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a tiny CLI for real experiment analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory containing scenario/planner/result.json folders.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="CSV path for the grouped summary table.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--ignore-failed-runs",
        action="store_true",
        help="Exclude failed runs before computing grouped averages.",
    )
    return parser


def configure_matplotlib() -> None:
    """Point matplotlib at a writable cache location."""
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "matplotlib-codex-analysis"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)


def discover_result_files(results_root: Path) -> list[Path]:
    """Find all result.json files recursively under the results root."""
    if not results_root.exists():
        print(f"Results root does not exist: {results_root}")
        return []
    return sorted(results_root.rglob("result.json"))


def load_result_rows(result_paths: list[Path]) -> list[dict[str, object]]:
    """Load result files into a flat row structure."""
    rows: list[dict[str, object]] = []
    for result_path in result_paths:
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"Skipping unreadable result file {result_path}: {error}")
            continue

        try:
            rows.append(
                {
                    "scenario": str(payload["scenario_name"]),
                    "planner": str(payload["planner_name"]),
                    "success": bool(payload["success"]),
                    "total_score": int(payload["total_score"]),
                    "simulation_steps": int(payload["simulation_steps"]),
                    "completed_actions": int(payload["completed_actions"]),
                    "failed_actions": int(payload["failed_actions"]),
                    "result_path": str(result_path),
                }
            )
        except KeyError as error:
            print(f"Skipping malformed result file {result_path}: missing {error}")
        except (TypeError, ValueError) as error:
            print(f"Skipping malformed result file {result_path}: {error}")
    return rows


def build_dataframes(rows: list[dict[str, object]], ignore_failed_runs: bool):
    """Create the detailed and grouped pandas DataFrames."""
    import pandas as pd

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "planner",
            "success",
            "total_score",
            "simulation_steps",
            "completed_actions",
            "failed_actions",
            "result_path",
        ],
    )
    if dataframe.empty:
        return dataframe, dataframe

    dataframe["success"] = dataframe["success"].astype(bool)
    if ignore_failed_runs:
        dataframe = dataframe[dataframe["success"]].copy()

    summary = (
        dataframe.groupby(["scenario", "planner"], dropna=False)
        .agg(
            success_rate=("success", "mean"),
            avg_score=("total_score", "mean"),
            avg_steps=("simulation_steps", "mean"),
            avg_failed_actions=("failed_actions", "mean"),
        )
        .reset_index()
    )
    summary["success_rate"] = summary["success_rate"] * 100.0
    summary = summary.sort_values(
        by=["scenario", "planner"],
        key=lambda series: series.map(planner_sort_key) if series.name == "planner" else series,
    )
    return dataframe, summary


def planner_sort_key(planner_name: object) -> int:
    """Return a stable planner ordering for tables and plots."""
    planner_text = str(planner_name)
    return PLANNER_ORDER.index(planner_text) if planner_text in PLANNER_ORDER else len(PLANNER_ORDER)


def save_summary_csv(summary, csv_path: Path) -> Path:
    """Persist the grouped summary table."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    return csv_path


def plot_grouped_metric(summary, metric: str, ylabel: str, output_path: Path, title: str) -> Path:
    """Save one grouped bar chart for a summary metric."""
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(summary["scenario"].tolist()))
    planners = [planner for planner in PLANNER_ORDER if planner in set(summary["planner"].tolist())]
    if not planners:
        planners = list(dict.fromkeys(summary["planner"].tolist()))

    x_positions = list(range(len(scenarios)))
    bar_width = 0.22 if planners else 0.6

    figure, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    for planner_index, planner_name in enumerate(planners):
        planner_rows = summary[summary["planner"] == planner_name]
        values_by_scenario = {
            str(row["scenario"]): float(row[metric])
            for _, row in planner_rows.iterrows()
        }
        offsets = [
            x_position + (planner_index - (len(planners) - 1) / 2) * bar_width
            for x_position in x_positions
        ]
        axis.bar(
            offsets,
            [values_by_scenario.get(scenario, 0.0) for scenario in scenarios],
            width=bar_width,
            label=planner_name,
            color=PLANNER_COLORS.get(planner_name, "#4c566a"),
        )

    axis.set_title(title, fontsize=14)
    axis.set_xlabel("Scenario", fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.set_xticks(x_positions, scenarios)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def print_text_summary(summary) -> None:
    """Print a compact scenario-by-scenario summary."""
    if summary.empty:
        print("No valid experiment results were found.")
        return

    for scenario_name in summary["scenario"].drop_duplicates().tolist():
        scenario_rows = summary[summary["scenario"] == scenario_name]
        print(f"Scenario: {scenario_name}")
        for _, row in scenario_rows.iterrows():
            print(
                "  - "
                f"{row['planner']}: "
                f"score={row['avg_score']:.2f}, "
                f"steps={row['avg_steps']:.2f}, "
                f"success_rate={row['success_rate']:.1f}%, "
                f"failed_actions={row['avg_failed_actions']:.2f}"
            )


def main() -> int:
    """Run the full analysis pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    result_paths = discover_result_files(args.results_root)
    rows = load_result_rows(result_paths)
    dataframe, summary = build_dataframes(rows, ignore_failed_runs=args.ignore_failed_runs)

    if dataframe.empty or summary.empty:
        print("No result rows available for analysis.")
        return 0

    summary_csv_path = save_summary_csv(summary, args.summary_csv)
    score_figure = plot_grouped_metric(
        summary,
        metric="avg_score",
        ylabel="Average score",
        output_path=args.figures_dir / "score_comparison.png",
        title="Real Experiment Score Comparison",
    )
    steps_figure = plot_grouped_metric(
        summary,
        metric="avg_steps",
        ylabel="Average simulation steps",
        output_path=args.figures_dir / "steps_comparison.png",
        title="Real Experiment Efficiency Comparison",
    )
    success_figure = plot_grouped_metric(
        summary,
        metric="success_rate",
        ylabel="Success rate (%)",
        output_path=args.figures_dir / "success_rate.png",
        title="Real Experiment Success Rate",
    )

    print_text_summary(summary)
    print("Summary CSV:", summary_csv_path)
    print("Figures:")
    print("  -", score_figure)
    print("  -", steps_figure)
    print("  -", success_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
