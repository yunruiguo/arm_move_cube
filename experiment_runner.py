"""Minimal experiment runner for comparing planner modes across scenarios."""

from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
from pathlib import Path

from backend import PlanningBackend, ToyWorldBackend
from benchmark_scenarios import get_benchmark_scenarios
from main import (
    build_coa,
    build_pick_place_action,
    choose_goal_region_for_object,
    generate_coas,
    generate_nearest_first_coa,
    get_graspable_object_names,
)
from mcts import evaluate_coas, select_best_coa

Action = dict[str, object]
COA = dict[str, object]
ExperimentResult = dict[str, object]


def generate_fixed_order_baseline_coa(backend: PlanningBackend) -> COA:
    """Generate a simple fixed-order baseline plan."""
    actions: list[Action] = []
    for object_name in get_graspable_object_names(backend)[:3]:
        _, goal_position = choose_goal_region_for_object(backend, object_name)
        actions.append(build_pick_place_action(object_name, goal_position))

    return build_coa(
        "fixed-order-baseline",
        "fixed_order_baseline",
        actions,
    )


def score_action_sequence(backend: PlanningBackend, coa: COA) -> ExperimentResult:
    """Score a full action sequence using the current reachability checks."""
    with redirect_stdout(io.StringIO()):
        return backend.simulate_action_sequence(coa)


def run_fixed_order_mode(backend: PlanningBackend) -> ExperimentResult:
    """Run the fixed-order baseline on a scenario."""
    coa = generate_fixed_order_baseline_coa(backend)
    result = score_action_sequence(backend, coa)
    result["mode"] = "fixed-order baseline"
    return result


def run_nearest_first_mode(backend: PlanningBackend) -> ExperimentResult:
    """Run the nearest-first heuristic on a scenario."""
    coa = generate_nearest_first_coa(backend)
    result = score_action_sequence(backend, coa)
    result["mode"] = "nearest-first heuristic"
    return result


def run_multi_coa_mode(backend: PlanningBackend) -> ExperimentResult:
    """Run the current multi-COA evaluator and replay the selected plan."""
    with redirect_stdout(io.StringIO()):
        results = evaluate_coas(backend, generate_coas(backend))
        best_coa_result = select_best_coa(results)

    selected_coa = None
    for coa in generate_coas(backend):
        if coa["name"] == best_coa_result["name"]:
            selected_coa = coa
            break

    assert selected_coa is not None
    result = score_action_sequence(backend, selected_coa)
    result["mode"] = "current multi-COA evaluator"
    return result


def format_summary_table(results: list[ExperimentResult]) -> str:
    """Format a compact terminal table."""
    headers = [
        "scenario",
        "mode",
        "selected_strategy",
        "total_score",
        "success",
        "failed_actions",
        "completed_actions",
    ]
    rows = [headers]
    for result in results:
        rows.append(
            [
                str(result["scenario"]),
                str(result["mode"]),
                str(result["selected_strategy"]),
                str(result["total_score"]),
                str(result["success"]),
                str(result["failed_actions"]),
                str(result["completed_actions"]),
            ]
        )

    column_widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    formatted_lines = []
    for row_index, row in enumerate(rows):
        padded = [cell.ljust(column_widths[index]) for index, cell in enumerate(row)]
        formatted_lines.append(" | ".join(padded))
        if row_index == 0:
            formatted_lines.append("-+-".join("-" * width for width in column_widths))
    return "\n".join(formatted_lines)


def save_results_csv(results: list[ExperimentResult], output_dir: str = "outputs") -> Path:
    """Save experiment results to a CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "experiment_summary.csv"
    fieldnames = [
        "scenario",
        "mode",
        "selected_strategy",
        "family",
        "total_score",
        "success",
        "failed_actions",
        "completed_actions",
        "failure_reasons",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in fieldnames})

    return csv_path


def run_experiments() -> list[ExperimentResult]:
    """Run all experiment modes across all predefined scenarios."""
    results: list[ExperimentResult] = []
    for scenario_name, builder in get_benchmark_scenarios():
        backend = ToyWorldBackend(builder())
        scenario_results = [
            run_fixed_order_mode(ToyWorldBackend(builder())),
            run_nearest_first_mode(ToyWorldBackend(builder())),
            run_multi_coa_mode(backend),
        ]
        for result in scenario_results:
            result["scenario"] = scenario_name
            results.append(result)
    return results


def main() -> None:
    """Run the experiment suite and print a compact summary."""
    results = run_experiments()
    print("=== experiment summary ===")
    print(format_summary_table(results))
    csv_path = save_results_csv(results)
    print("=== experiment csv ===")
    print(csv_path.as_posix())


if __name__ == "__main__":
    main()
