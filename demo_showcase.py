"""Showcase entry point for the stronger simulator-grounded decision demo."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
from contextlib import redirect_stdout

from benchmark_scenarios_real import get_real_benchmark_scenario, build_real_benchmark_world
from reachability_engine import query_reachability
from showcase_planners import (
    SHOWCASE_PLANNER_MODES,
    build_plan,
    build_rollout_subtasks,
    describe_scenario,
)
from showcase_visualizer import (
    save_scene_overview,
    save_selected_plan_trace,
    save_strategy_comparison,
)
from world_state import WorldState

PlannerRecord = dict[str, object]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the showcase demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="single_cube_goal",
        # Keep the default demo runnable as a simple single-cube showcase.
        help="Named real scenario to use for the showcase demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/gyr/codex/platform/outputs/showcase_demo"),
        help="Directory for demo artifacts.",
    )
    parser.add_argument(
        "--planners",
        nargs="+",
        default=["fixed_order", "nearest_first", "clear_blocking_first", "decision_pipeline"],
        help="Planner modes to compare.",
    )
    parser.add_argument(
        "--rollout-backend",
        choices=("none", "isaac"),
        default="none",
        help="Whether to attempt a real Isaac rollout artifact for the selected strategy.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional LLM model name for the llm_generated planner.",
    )
    return parser


def evaluate_plan(config_name: str, plan: dict[str, object]) -> PlannerRecord:
    """Evaluate a selected plan with a simple order-sensitive state rollout."""
    state = copy.deepcopy(build_real_benchmark_world(config_name))
    total_score = 0
    completed_actions = 0
    failed_actions = 0
    failure_reasons: list[str] = []
    step_results: list[dict[str, object]] = []
    evaluation_log = io.StringIO()

    with redirect_stdout(evaluation_log):
        for action in plan["coa"]["actions"]:
            object_name = str(action["object"])
            place_position = action["place_position"]
            assert isinstance(place_position, tuple)

            pick_result = query_reachability(state, "pick", object_name)
            object_position = state.get_object_position(object_name)
            if object_position is not None:
                state.objects.pop(object_name, None)
                state.update_robot_position(object_position)

            place_result = query_reachability(
                state,
                "place",
                object_name,
                place_position=place_position,
            )
            step_success = (
                bool(pick_result["reachable"])
                and bool(pick_result["action_feasible"])
                and bool(place_result["reachable"])
                and bool(place_result["action_feasible"])
            )
            step_score = int(pick_result["score"]) + int(place_result["score"])
            step_score += 10 if step_success else -5
            total_score += step_score

            step_result = {
                "object": object_name,
                "place_position": place_position,
                "pick_result": pick_result,
                "place_result": place_result,
                "success": step_success,
                "score": step_score,
            }
            step_results.append(step_result)

            if not step_success:
                failed_actions += 1
                failure_reasons.append(
                    f"{object_name}: {pick_result['reason']} | {place_result['reason']}"
                )
                break

            completed_actions += 1
            state.update_object(object_name, place_position, "cube", True)
            state.update_robot_position(find_retreat_position(state, place_position))

    success = failed_actions == 0 and completed_actions == len(plan["coa"]["actions"])
    return {
        "planner_mode": plan["planner_name"],
        "selected_strategy": plan["selected_strategy"],
        "total_score": int(total_score),
        "success": bool(success),
        "completed_actions": int(completed_actions),
        "failed_actions": int(failed_actions),
        "failure_reasons": "; ".join(failure_reasons),
        "planning_log": plan.get("planning_log", ""),
        "decision_rationale": plan.get("decision_rationale", ""),
        "planned_actions": plan["planned_actions"],
        "selected_object": plan["selected_object"],
        "selected_goal_region": plan["selected_goal_region"],
        "coa": plan["coa"],
        "step_outcomes": step_results,
        "coa_evaluation_summary": plan.get("coa_evaluation_summary", []),
        "evaluation_log": evaluation_log.getvalue().strip(),
    }


def find_retreat_position(state: WorldState, occupied_position: tuple[int, int]) -> tuple[int, int]:
    """Move the robot to a nearby free cell after placing an object."""
    candidate_positions = [
        (occupied_position[0] + 1, occupied_position[1]),
        (occupied_position[0] - 1, occupied_position[1]),
        (occupied_position[0], occupied_position[1] + 1),
        (occupied_position[0], occupied_position[1] - 1),
    ]
    for candidate_position in candidate_positions:
        if candidate_position in state.obstacles:
            continue
        if candidate_position in state.forbidden_zones:
            continue
        if not state.is_occupied(candidate_position):
            return candidate_position
    return occupied_position


def select_best_record(records: list[PlannerRecord]) -> PlannerRecord:
    """Select the best record by score, then by completed actions."""
    return max(
        records,
        key=lambda record: (
            int(record["total_score"]),
            int(record["completed_actions"]),
            -int(record["failed_actions"]),
        ),
    )


def write_scene_summary(output_dir: Path, scenario_summary: dict[str, object]) -> Path:
    """Write a compact human-readable scene summary artifact."""
    summary_path = output_dir / "scene_summary.md"
    lines = [
        f"# Showcase Scene: {scenario_summary['name']}",
        "",
        scenario_summary["description"],
        "",
        "## Fixed Order",
        ", ".join(scenario_summary["fixed_order"]),
        "",
        "## Objects",
    ]
    for object_name, object_position in scenario_summary["object_grid_positions"].items():
        goal_region = scenario_summary["object_goal_regions"][object_name]
        lines.append(f"- {object_name}: grid={object_position}, goal={goal_region}")
    lines.extend(
        [
            "",
            "## Goals",
        ]
    )
    for region_name, goal_position in scenario_summary["goal_region_grid_positions"].items():
        lines.append(f"- {region_name}: {goal_position}")
    if scenario_summary["obstacles"]:
        lines.extend(["", "## Obstacles", f"- {scenario_summary['obstacles']}"])
    if scenario_summary["forbidden_zones"]:
        lines.extend(["", "## Forbidden Zones", f"- {scenario_summary['forbidden_zones']}"])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def write_strategy_summary(output_dir: Path, records: list[PlannerRecord], best_record: PlannerRecord) -> Path:
    """Write a compact markdown summary for portfolio-friendly review."""
    summary_path = output_dir / "demo_summary.md"
    lines = [
        "# Decision Showcase Summary",
        "",
        f"Selected planner: **{best_record['planner_mode']}**",
        f"Selected strategy: **{best_record['selected_strategy']}**",
        f"Total score: **{best_record['total_score']}**",
        "",
        "## Candidate Strategies",
    ]
    for record in records:
        lines.append(
            f"- {record['planner_mode']}: score={record['total_score']}, "
            f"completed={record['completed_actions']}, failed={record['failed_actions']}, "
            f"success={record['success']}"
        )
    lines.extend(
        [
            "",
            "## Selected Action Sequence",
        ]
    )
    for action_label in best_record["planned_actions"]:
        lines.append(f"- {action_label}")
    if best_record["decision_rationale"]:
        lines.extend(["", "## Selection Rationale", best_record["decision_rationale"]])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def save_json_records(output_dir: Path, records: list[PlannerRecord], best_record: PlannerRecord) -> Path:
    """Persist a machine-readable showcase summary."""
    payload = {
        "records": records,
        "selected": {
            "planner_mode": best_record["planner_mode"],
            "selected_strategy": best_record["selected_strategy"],
            "total_score": best_record["total_score"],
        },
    }
    output_path = output_dir / "strategy_results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def maybe_run_real_rollout(
    output_dir: Path,
    config,
    best_record: PlannerRecord,
    scenario_summary: dict[str, object],
    rollout_backend: str,
) -> dict[str, object] | None:
    """Optionally run the real Isaac rollout artifact for the selected plan."""
    if rollout_backend != "isaac":
        return None

    from record_franka_pick_place_animation import record_pick_place_rollout

    rollout_dir = output_dir / "rollout"
    subtasks = build_rollout_subtasks(config, best_record["coa"])
    try:
        return record_pick_place_rollout(
            output_dir=rollout_dir,
            planner_name=str(best_record["planner_mode"]),
            selected_strategy=str(best_record["selected_strategy"]),
            selected_target_object=str(best_record["selected_object"]),
            cube_initial_position=config.object_sim_positions[str(best_record["selected_object"])],
            target_position=config.goal_region_sim_positions[str(best_record["selected_goal_region"])],
            scene_metadata=scenario_summary,
            subtasks=subtasks,
        )
    except Exception as error:
        failure_path = output_dir / "rollout_failure.txt"
        failure_path.write_text(str(error) + "\n", encoding="utf-8")
        return {
            "success": False,
            "error": str(error),
            "failure_path": str(failure_path),
        }


def print_showcase_summary(records: list[PlannerRecord], best_record: PlannerRecord, output_dir: Path) -> None:
    """Print a concise terminal summary for the showcase demo."""
    print("=== showcase demo ===")
    for record in records:
        print(
            f"- {record['planner_mode']}: score={record['total_score']}, "
            f"completed={record['completed_actions']}, failed={record['failed_actions']}, "
            f"success={record['success']}"
        )
    print("selected planner:", best_record["planner_mode"])
    print("selected strategy:", best_record["selected_strategy"])
    if best_record["decision_rationale"]:
        print("why selected:", best_record["decision_rationale"])
    print("artifacts dir:", output_dir)


def main() -> int:
    """Run the stronger showcase demo end to end."""
    args = build_arg_parser().parse_args()
    config = get_real_benchmark_scenario(args.scenario)
    output_dir = args.output_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    world_state = build_real_benchmark_world(config.name)
    scenario_summary = describe_scenario(config)

    available_records: list[PlannerRecord] = []
    for planner_mode in args.planners:
        try:
            plan = build_plan(planner_mode, config, llm_model=args.llm_model)
        except Exception as error:
            print(f"Skipping planner {planner_mode}: {error}")
            continue
        available_records.append(evaluate_plan(config.name, plan))

    if not available_records:
        print("No planners produced a usable showcase plan.")
        return 1

    best_record = select_best_record(available_records)

    scene_summary_path = write_scene_summary(output_dir, scenario_summary)
    scene_overview_path = save_scene_overview(world_state, config, output_dir / "scene_overview.png")
    strategy_plot_path = save_strategy_comparison(available_records, output_dir / "strategy_comparison.png")
    trace_plot_path = save_selected_plan_trace(config, best_record, output_dir / "selected_plan_trace.png")
    summary_md_path = write_strategy_summary(output_dir, available_records, best_record)
    results_json_path = save_json_records(output_dir, available_records, best_record)
    rollout_summary = maybe_run_real_rollout(
        output_dir=output_dir,
        config=config,
        best_record=best_record,
        scenario_summary=scenario_summary,
        rollout_backend=args.rollout_backend,
    )

    artifact_manifest = {
        "scene_summary": str(scene_summary_path),
        "scene_overview": str(scene_overview_path),
        "strategy_comparison": str(strategy_plot_path),
        "selected_plan_trace": str(trace_plot_path),
        "demo_summary": str(summary_md_path),
        "strategy_results": str(results_json_path),
        "rollout": rollout_summary,
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2),
        encoding="utf-8",
    )

    print_showcase_summary(available_records, best_record, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
