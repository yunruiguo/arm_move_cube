"""Standard experiment entry point for real Franka rollout planner modes."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from backend import ToyWorldBackend
from benchmark_scenarios_real import (
    REAL_BENCHMARK_SCENARIOS,
    RealScenario,
    build_real_benchmark_world,
    get_real_benchmark_scenario,
)

ExperimentResult = dict[str, object]
COA = dict[str, object]
PLANNER_MODES = ("fixed_order", "nearest_heuristic", "decision_pipeline", "all")
SCENARIO_NAMES = tuple(REAL_BENCHMARK_SCENARIOS.keys()) + ("all",)
DEFAULT_BATCH_PLANNER_MODES = ("fixed_order", "nearest_heuristic", "decision_pipeline")
BATCH_CHILD_ENV = "DECISION_PLATFORM_BATCH_CHILD"
PER_RUN_TIMEOUT_SECONDS = 240


def get_default_output_dir() -> Path:
    """Read the real benchmark output directory without importing Isaac modules."""
    candidate_roots = [
        Path("/mnt/data2"),
        Path("/data2"),
        Path("/data3"),
        Path("/data1"),
    ]
    for root in candidate_roots:
        if root.exists():
            return root / "outputs" / "real_experiments"
    return Path.cwd() / "outputs" / "real_experiments"


def format_planned_actions(coa: COA) -> list[str]:
    """Convert a structured COA into short printable action labels."""
    return [
        f"{action['action_type']} {action['object']} -> {action['place_position']}"
        for action in coa["actions"]
    ]


def describe_scenario(config: RealScenario) -> dict[str, object]:
    """Build a readable scenario summary for terminal logs and manifests."""
    return {
        "name": config.name,
        "description": config.description,
        "fixed_order": list(config.fixed_order),
        "object_grid_positions": config.object_grid_positions,
        "object_sim_positions": config.object_sim_positions,
        "object_goal_regions": config.object_goal_regions,
        "goal_region_grid_positions": config.goal_region_grid_positions,
        "goal_region_sim_positions": config.goal_region_sim_positions,
        "obstacles": list(config.obstacles),
        "forbidden_zones": list(config.forbidden_zones),
    }


def select_rollout_target(config: RealScenario, object_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Map a planner-selected object onto the physical cube start and goal."""
    goal_region_name = config.object_goal_regions[object_name]
    return (
        config.object_sim_positions[object_name],
        config.goal_region_sim_positions[goal_region_name],
    )


def build_rollout_subtasks(config: RealScenario, coa: COA) -> list[dict[str, object]]:
    """Convert planner actions into sequential real rollout subtasks."""
    goal_region_by_grid = {
        grid_position: region_name
        for region_name, grid_position in config.goal_region_grid_positions.items()
    }
    subtasks: list[dict[str, object]] = []
    for action in coa["actions"]:
        object_name = str(action["object"])
        place_position = action["place_position"]
        assert isinstance(place_position, tuple)
        goal_region_name = goal_region_by_grid[place_position]
        subtasks.append(
            {
                "object": object_name,
                "cube_initial_position": config.object_sim_positions[object_name],
                "target_position": config.goal_region_sim_positions[goal_region_name],
                "goal_region": goal_region_name,
            }
        )
    return subtasks


def build_fixed_order_plan(config: RealScenario) -> dict[str, object]:
    """Build the fixed-order baseline plan."""
    from main import build_coa, build_pick_place_action

    object_name = config.fixed_order[0]
    goal_region_name = config.object_goal_regions[object_name]
    goal_grid_position = config.goal_region_grid_positions[goal_region_name]
    coa = build_coa(
        "fixed-order",
        f"fixed_order_{config.name}",
        [build_pick_place_action(object_name, goal_grid_position)],
    )
    return {
        "planner_name": "fixed_order",
        "selected_strategy": coa["name"],
        "coa": coa,
        "selected_object": object_name,
        "selected_goal_region": goal_region_name,
        "planned_actions": format_planned_actions(coa),
        "planning_log": (
            f"Fixed-order baseline followed the predefined scenario order and chose {object_name} first."
        ),
    }


def build_nearest_heuristic_plan(config: RealScenario) -> dict[str, object]:
    """Build the nearest-object heuristic plan."""
    from main import generate_nearest_first_coa

    state = build_real_benchmark_world(config.name)
    coa = generate_nearest_first_coa(state)
    object_name = str(coa["object"])
    goal_region_name = config.object_goal_regions[object_name]
    return {
        "planner_name": "nearest_heuristic",
        "selected_strategy": coa["name"],
        "coa": coa,
        "selected_object": object_name,
        "selected_goal_region": goal_region_name,
        "planned_actions": format_planned_actions(coa),
        "planning_log": (
            "Nearest heuristic ranked scenario objects by planner-space Manhattan distance "
            f"and chose {object_name} first."
        ),
    }


def build_decision_pipeline_plan(config: RealScenario) -> dict[str, object]:
    """Build the current rule-aware decision pipeline plan."""
    from main import generate_coas
    from mcts import evaluate_coas, select_best_coa

    state = build_real_benchmark_world(config.name)
    planning_buffer = io.StringIO()
    with redirect_stdout(planning_buffer):
        candidate_coas = generate_coas(state)
        evaluation_results = evaluate_coas(state, candidate_coas)
        best_result = select_best_coa(evaluation_results)

    selected_coa = next(
        candidate_coa
        for candidate_coa in candidate_coas
        if candidate_coa["name"] == best_result["name"]
    )
    object_name = str(selected_coa["object"])
    goal_region_name = config.object_goal_regions[object_name]
    ranked_results = sorted(
        evaluation_results,
        key=lambda evaluation_result: int(evaluation_result["score"]),
        reverse=True,
    )
    evaluation_summary = [
        {
            "name": evaluation_result["name"],
            "score": int(evaluation_result["score"]),
            "success": bool(evaluation_result["success"]),
            "reason": str(evaluation_result["reason"]),
        }
        for evaluation_result in ranked_results
    ]
    runner_up = ranked_results[1] if len(ranked_results) > 1 else None
    if runner_up is None:
        decision_rationale = (
            f"Decision pipeline selected {selected_coa['name']} because it was the only evaluated COA."
        )
    else:
        score_margin = int(best_result["score"]) - int(runner_up["score"])
        decision_rationale = (
            f"Decision pipeline selected {selected_coa['name']} over {runner_up['name']} "
            f"by a score margin of {score_margin}. "
            f"Winner reason: {best_result['reason']}"
        )
    return {
        "planner_name": "decision_pipeline",
        "selected_strategy": selected_coa["name"],
        "coa": selected_coa,
        "selected_object": object_name,
        "selected_goal_region": goal_region_name,
        "planned_actions": format_planned_actions(selected_coa),
        "planning_log": planning_buffer.getvalue().strip(),
        "coa_evaluation_summary": evaluation_summary,
        "decision_rationale": decision_rationale,
    }


def build_plan(planner_mode: str, config: RealScenario) -> dict[str, object]:
    """Dispatch planner mode selection."""
    if planner_mode == "fixed_order":
        return build_fixed_order_plan(config)
    if planner_mode == "nearest_heuristic":
        return build_nearest_heuristic_plan(config)
    if planner_mode == "decision_pipeline":
        return build_decision_pipeline_plan(config)
    raise ValueError(f"Unsupported planner mode: {planner_mode}")


def resolve_output_dir(planner_mode: str, scenario_name: str) -> Path:
    """Keep outputs organized by scenario and planner mode."""
    default_output_dir = get_default_output_dir()
    return default_output_dir / scenario_name / planner_mode


def evaluate_plan_record(coa: COA, scenario_name: str) -> dict[str, object]:
    """Evaluate a selected COA with the current deterministic planner backend."""
    backend = ToyWorldBackend(build_real_benchmark_world(scenario_name))
    return backend.simulate_action_sequence(coa)


def build_result_record(
    scenario_name: str,
    planner_mode: str,
    plan: dict[str, object],
    planning_record: dict[str, object],
    rollout_summary: dict[str, object],
) -> ExperimentResult:
    """Build a normalized machine-readable result record."""
    result_timestamp = datetime.now(timezone.utc).isoformat()
    overall_success = bool(planning_record["success"]) and bool(rollout_summary["success"])
    return {
        "timestamp": result_timestamp,
        "runner": "experiment_runner_real",
        "scenario_name": scenario_name,
        "planner_mode": planner_mode,
        "planner_name": plan["planner_name"],
        "selected_coa": plan["selected_strategy"],
        "selected_action_sequence": plan["planned_actions"],
        "selected_object": plan["selected_object"],
        "selected_goal_region": plan["selected_goal_region"],
        "success": overall_success,
        "planning_success": bool(planning_record["success"]),
        "rollout_success": bool(rollout_summary["success"]),
        "total_score": int(planning_record["total_score"]),
        "completed_actions": int(planning_record["completed_actions"]),
        "failed_actions": int(planning_record["failed_actions"]),
        "failure_reasons": planning_record["failure_reasons"],
        "planning_log": plan["planning_log"],
        "decision_rationale": plan.get("decision_rationale", ""),
        "coa_evaluation_summary": plan.get("coa_evaluation_summary", []),
        "scene_config": rollout_summary["scene_config"],
        "scenario_summary": rollout_summary["scene_metadata"],
        "final_state_summary": rollout_summary["final_state_summary"],
        "step_outcomes": planning_record["step_results"],
        "paths": {
            "output_dir": rollout_summary["output_dir"],
            "frames_dir": rollout_summary["frames_dir"],
            "gif_path": rollout_summary["gif_path"],
            "manifest_path": rollout_summary["manifest_path"],
        },
        "num_frames": int(rollout_summary["num_frames"]),
        "simulation_steps": int(rollout_summary["simulation_steps"]),
    }


def save_result_record(result: ExperimentResult) -> Path:
    """Persist one structured experiment result as JSON."""
    output_dir = Path(str(result["paths"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result_path


def read_result_record(result_path: Path) -> ExperimentResult:
    """Load one previously written experiment result file."""
    return json.loads(result_path.read_text(encoding="utf-8"))


def format_timestamp_now() -> str:
    """Return a short local timestamp for progress logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_file_mtime(path: Path) -> str:
    """Return a readable modification time for an existing file."""
    modified_at = datetime.fromtimestamp(path.stat().st_mtime)
    return modified_at.strftime("%Y-%m-%d %H:%M:%S")


def write_batch_summary_csv(results: list[ExperimentResult]) -> Path:
    """Write one CSV row per batch run for quick comparison."""
    output_dir = get_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_summary.csv"
    fieldnames = [
        "scenario",
        "planner_mode",
        "success",
        "total_score",
        "completed_actions",
        "failed_actions",
        "gif_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "scenario": result["scenario_name"],
                    "planner_mode": result["planner_mode"],
                    "success": result["success"],
                    "total_score": result["total_score"],
                    "completed_actions": result["completed_actions"],
                    "failed_actions": result["failed_actions"],
                    "gif_path": result["paths"]["gif_path"],
                }
            )
    return csv_path


def print_batch_summary_table(results: list[ExperimentResult], csv_path: Path) -> None:
    """Print a compact comparison table for the full benchmark batch."""
    rows = [
        [
            str(result["scenario_name"]),
            str(result["planner_mode"]),
            "yes" if result["success"] else "no",
            str(result["total_score"]),
            str(result["completed_actions"]),
            str(result["failed_actions"]),
        ]
        for result in results
    ]
    headers = ["scenario", "planner", "success", "score", "completed", "failed"]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values: list[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        )

    print("=== real benchmark summary ===")
    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))
    print("summary csv:", csv_path)


def save_batch_visualization(results: list[ExperimentResult]) -> Path:
    """Create one simple benchmark comparison figure."""
    from benchmark_visualizer_real import save_benchmark_comparison_plot

    return save_benchmark_comparison_plot(results, get_default_output_dir())


def build_failed_batch_result(
    scenario_name: str,
    planner_mode: str,
    return_code: int,
) -> ExperimentResult:
    """Create a fallback result record when a child rollout process fails."""
    failed_output_dir = resolve_output_dir(planner_mode, scenario_name)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runner": "experiment_runner_real",
        "scenario_name": scenario_name,
        "planner_mode": planner_mode,
        "planner_name": planner_mode,
        "selected_coa": "(run failed before selection)",
        "selected_action_sequence": [],
        "selected_object": "",
        "selected_goal_region": "",
        "success": False,
        "planning_success": False,
        "rollout_success": False,
        "total_score": -999,
        "completed_actions": 0,
        "failed_actions": 1,
        "failure_reasons": f"Child rollout process exited with code {return_code}.",
        "planning_log": "",
        "decision_rationale": "",
        "coa_evaluation_summary": [],
        "scene_config": {},
        "scenario_summary": {"name": scenario_name},
        "final_state_summary": {},
        "step_outcomes": [],
        "paths": {
            "output_dir": str(failed_output_dir),
            "frames_dir": str(failed_output_dir / "frames"),
            "gif_path": str(failed_output_dir / "rollout.gif"),
            "manifest_path": str(failed_output_dir / "manifest.json"),
        },
        "num_frames": 0,
        "simulation_steps": 0,
    }


def build_experiment_summary(planner_mode: str, scenario_name: str) -> ExperimentResult:
    """Run one rollout under a selected planner mode and scenario."""
    from record_franka_pick_place_animation import record_pick_place_rollout

    config = get_real_benchmark_scenario(scenario_name)
    plan = build_plan(planner_mode, config)
    planning_record = evaluate_plan_record(plan["coa"], scenario_name)
    cube_initial_position, target_position = select_rollout_target(
        config,
        str(plan["selected_object"]),
    )
    rollout_subtasks = build_rollout_subtasks(config, plan["coa"])
    rollout_summary = record_pick_place_rollout(
        output_dir=resolve_output_dir(planner_mode, scenario_name),
        planner_name=str(plan["planner_name"]),
        selected_strategy=str(plan["selected_strategy"]),
        selected_target_object=str(plan["selected_object"]),
        cube_initial_position=cube_initial_position,
        target_position=target_position,
        scene_metadata=describe_scenario(config),
        subtasks=rollout_subtasks,
    )
    result = build_result_record(
        scenario_name=scenario_name,
        planner_mode=planner_mode,
        plan=plan,
        planning_record=planning_record,
        rollout_summary=rollout_summary,
    )
    result["result_path"] = str(save_result_record(result))
    result["rollout_action_sequence"] = rollout_summary["action_sequence"]
    return result


def print_experiment_summary(result: ExperimentResult) -> None:
    """Print a compact, readable single-run experiment summary."""
    print("=== real experiment ===")
    print("timestamp:", result["timestamp"])
    print("scenario:", result["scenario_name"])
    print("planner mode:", result["planner_mode"])
    print("planner:", result["planner_name"])
    print("selected coa:", result["selected_coa"])
    print("selected object:", result["selected_object"])
    print("selected goal region:", result["selected_goal_region"])
    print("score:", result["total_score"])
    print(
        "action stats:",
        f"completed={result['completed_actions']}, failed={result['failed_actions']}",
    )
    print(
        "success:",
        result["success"],
        f"(planning={result['planning_success']}, rollout={result['rollout_success']})",
    )
    print("scene config:")
    print(json.dumps(result["scene_config"], indent=2))
    print("scenario summary:")
    print(json.dumps(result["scenario_summary"], indent=2))
    print("selected action sequence:")
    for index, action_name in enumerate(result["selected_action_sequence"], start=1):
        print(f"  {index}. {action_name}")
    print("planner step outcomes:")
    for index, step_result in enumerate(result["step_outcomes"], start=1):
        pick_result = step_result["pick_result"]
        place_result = step_result["place_result"]
        print(
            f"  {index}. object={step_result['object']} target={step_result['place_position']}"
        )
        print(
            "     outcome:",
            f"success={step_result['success']}, step_score={step_result['score']}",
        )
        print(
            "     pick:",
            f"reachable={pick_result['reachable']}, feasible={pick_result['action_feasible']}, "
            f"score={pick_result['score']}, reason={pick_result['reason']}",
        )
        print(
            "     place:",
            f"reachable={place_result['reachable']}, feasible={place_result['action_feasible']}, "
            f"score={place_result['score']}, reason={place_result['reason']}",
        )
    print("rollout phases:")
    for index, action_name in enumerate(result["rollout_action_sequence"], start=1):
        print(f"  {index}. {action_name}")
    print("planning log:")
    print(result["planning_log"] or "(no additional planning log)")
    if result["planner_mode"] == "decision_pipeline":
        print("coa evaluation summary:")
        for candidate in result["coa_evaluation_summary"]:
            print(
                "  "
                f"{candidate['name']}: score={candidate['score']}, "
                f"success={candidate['success']}, reason={candidate['reason']}"
            )
        print("selection rationale:")
        print(result["decision_rationale"] or "(no additional rationale)")
    if result["failure_reasons"]:
        print("failure reasons:", result["failure_reasons"])
        print("final rationale: planner run failed because at least one action step was infeasible.")
    else:
        print("final rationale: planner run succeeded because every planned action completed without a backend failure.")
    print("final state summary:")
    print(json.dumps(result["final_state_summary"], indent=2))
    print("simulation steps:", result["simulation_steps"])
    print("frames:", result["num_frames"])
    print("frames dir:", result["paths"]["frames_dir"])
    print("gif:", result["paths"]["gif_path"])
    print("manifest:", result["paths"]["manifest_path"])
    print("result json:", result["result_path"])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for planner mode and scenario selection."""
    parser = argparse.ArgumentParser(description="Run a real Franka planning experiment.")
    parser.add_argument(
        "--planner-mode",
        choices=PLANNER_MODES,
        default="fixed_order",
        help="Planner mode to run. Use 'all' to run every supported mode sequentially.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        default="easy_clear",
        help="Benchmark scenario to run. Use 'all' to iterate over all predefined scenarios.",
    )
    return parser.parse_args()


def run_all_combinations() -> None:
    """Run every requested scenario and planner mode in isolated subprocesses."""
    batch_results: list[ExperimentResult] = []
    combinations = [
        (scenario_name, mode_name)
        for scenario_name in REAL_BENCHMARK_SCENARIOS
        for mode_name in DEFAULT_BATCH_PLANNER_MODES
    ]
    total_runs = len(combinations)

    for run_index, (scenario_name, mode_name) in enumerate(combinations, start=1):
            print(
                f"[{format_timestamp_now()}] "
                f"[{run_index}/{total_runs}] starting rollout: "
                f"scenario={scenario_name} planner_mode={mode_name}",
                flush=True,
            )
            completed_process = subprocess.run(
                [
                    "timeout",
                    str(PER_RUN_TIMEOUT_SECONDS),
                    sys.executable,
                    __file__,
                    "--scenario",
                    scenario_name,
                    "--planner-mode",
                    mode_name,
                ],
                env={**os.environ, BATCH_CHILD_ENV: "1"},
            )
            result_path = resolve_output_dir(mode_name, scenario_name) / "result.json"
            if result_path.exists():
                result = read_result_record(result_path)
                if completed_process.returncode == 124:
                    print(
                        f"[{format_timestamp_now()}] "
                        f"[{run_index}/{total_runs}] rollout hit timeout after writing outputs: "
                        f"scenario={scenario_name} planner_mode={mode_name}",
                        flush=True,
                    )
            else:
                print(
                    f"[{format_timestamp_now()}] "
                    f"[{run_index}/{total_runs}] rollout failed: "
                    f"scenario={scenario_name} planner_mode={mode_name} "
                    f"(return code {completed_process.returncode})",
                    flush=True,
                )
                result = build_failed_batch_result(
                    scenario_name=scenario_name,
                    planner_mode=mode_name,
                    return_code=completed_process.returncode,
                )
            batch_results.append(result)
            gif_path = Path(str(result["paths"]["gif_path"]))
            print(
                f"[{format_timestamp_now()}] "
                f"[{run_index}/{total_runs}] finished rollout: "
                f"scenario={scenario_name} planner_mode={mode_name}",
                flush=True,
            )
            print(f"  result json: {result_path}", flush=True)
            if gif_path.exists():
                print(
                    f"  gif: {gif_path} "
                    f"(updated {format_file_mtime(gif_path)})",
                    flush=True,
                )
            else:
                print("  gif: (missing)", flush=True)
    csv_path = write_batch_summary_csv(batch_results)
    figure_path = save_batch_visualization(batch_results)
    print_batch_summary_table(batch_results, csv_path)
    print("comparison plot:", figure_path)


def main() -> None:
    """Run one or more real Franka experiments and print structured summaries."""
    args = parse_args()
    if args.planner_mode == "all" or args.scenario == "all":
        run_all_combinations()
        return

    from record_franka_pick_place_animation import simulation_app

    exit_code = 0
    try:
        result = build_experiment_summary(args.planner_mode, args.scenario)
        print_experiment_summary(result)
    except Exception:
        exit_code = 1
        raise
    finally:
        simulation_app.close()
        if os.environ.get(BATCH_CHILD_ENV) == "1":
            os._exit(exit_code)


if __name__ == "__main__":
    main()
