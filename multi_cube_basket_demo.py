"""Run a physics-consistent multi-cube basket demo via sequential rollout subprocesses."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


SUBTASK_TIMEOUT_SECONDS = 180
MANIFEST_READY_GRACE_SECONDS = 5
BASKET_SPAN = 0.30
BASKET_WALL_HALF_OFFSET = 0.155
ROBOT_REFERENCE_POSITION = (0.62, -0.22)
PLUS_SHAPE_PERIMETER = (
    "cube_north",
    "cube_east",
    "cube_south",
    "cube_west",
)


def resolve_storage_root() -> Path:
    """Match the rollout script's storage-root preference without importing Isaac."""
    for root_path in (Path("/mnt/data2"), Path("/data2"), Path("/data3"), Path("/data1")):
        if root_path.exists():
            return root_path
    return Path.cwd()


def build_basket_slots(
    basket_center: tuple[float, float, float],
    cube_rest_height: float,
    count: int,
) -> list[tuple[float, float, float]]:
    """Generate stable basket landing slots with readable spacing."""
    center_x, center_y, _center_z = basket_center
    interior_half_span = BASKET_WALL_HALF_OFFSET - 0.015
    left_x = center_x - interior_half_span + 0.03
    top_y = center_y + interior_half_span - 0.03
    row_spacing = 0.065
    col_spacing = 0.06
    slot_pattern = [
        (left_x + 0.00 * col_spacing, top_y),
        (left_x + 1.00 * col_spacing, top_y),
        (left_x + 2.00 * col_spacing, top_y),
        (left_x + 0.50 * col_spacing, top_y - row_spacing),
        (left_x + 1.50 * col_spacing, top_y - row_spacing),
        (left_x + 1.00 * col_spacing, top_y - 2.0 * row_spacing),
    ]
    slots: list[tuple[float, float, float]] = []
    for index in range(count):
        slot_x, slot_y = slot_pattern[min(index, len(slot_pattern) - 1)]
        slots.append((round(float(slot_x), 4), round(float(slot_y), 4), round(float(cube_rest_height), 4)))
    return slots


def _distance_to_robot(position: list[float]) -> float:
    """Compute a small XY distance heuristic from a fixed robot reference point."""
    return ((float(position[0]) - ROBOT_REFERENCE_POSITION[0]) ** 2 + (float(position[1]) - ROBOT_REFERENCE_POSITION[1]) ** 2) ** 0.5


def build_candidate_orders() -> list[dict[str, object]]:
    """Return deterministic planner candidates for the plus-shape planning showcase."""
    return [
        {
            "planner_name": "fixed_order",
            "strategy_name": "fixed_order_center_first",
            "order": ["cube_center", "cube_north", "cube_east", "cube_south", "cube_west"],
        },
        {
            "planner_name": "nearest_first",
            "strategy_name": "nearest_first_plus_shape",
            "order": ["cube_south", "cube_west", "cube_center", "cube_east", "cube_north"],
        },
        {
            "planner_name": "clear_blocking_first",
            "strategy_name": "clear_blocking_first_plus_shape",
            "order": ["cube_north", "cube_east", "cube_south", "cube_west", "cube_center"],
        },
    ]


def evaluate_candidate_order(candidate: dict[str, object], object_positions: dict[str, list[float]]) -> dict[str, object]:
    """Score a candidate order with a simple plus-shape blocking rule."""
    remaining = list(candidate["order"])
    score = 0.0
    step_logs: list[dict[str, object]] = []
    success = True
    failure_reason = "All subtasks remained feasible."

    for step_index, object_name in enumerate(candidate["order"], start=1):
        blocking_neighbors = [neighbor for neighbor in PLUS_SHAPE_PERIMETER if object_name == "cube_center" and neighbor in remaining]
        reachable = len(blocking_neighbors) == 0
        distance_penalty = round(_distance_to_robot(object_positions[object_name]), 3)
        score_contribution = 30.0 - distance_penalty * 10.0
        if not reachable:
            score_contribution -= 120.0
            success = False
            failure_reason = (
                f"{object_name} stays blocked while perimeter cubes remain: "
                + ", ".join(blocking_neighbors)
            )
        score += score_contribution
        step_logs.append(
            {
                "step": step_index,
                "object": object_name,
                "reachable": reachable,
                "blocking_neighbors": blocking_neighbors,
                "distance_penalty": distance_penalty,
                "score_contribution": round(score_contribution, 3),
                "reason": (
                    "Center cube is only feasible after all perimeter cubes move."
                    if not reachable
                    else "Object is planner-feasible for pickup."
                ),
            }
        )
        remaining.remove(object_name)
        if not reachable:
            break

    return {
        "planner_name": candidate["planner_name"],
        "strategy_name": candidate["strategy_name"],
        "order": list(candidate["order"]),
        "success": success,
        "score": round(score, 3),
        "step_logs": step_logs,
        "failure_reason": failure_reason,
    }


def select_planning_result(object_positions: dict[str, list[float]]) -> dict[str, object]:
    """Evaluate the candidate orders and pick the highest-scoring feasible strategy."""
    candidate_results = [
        evaluate_candidate_order(candidate, object_positions)
        for candidate in build_candidate_orders()
    ]
    candidate_results.sort(key=lambda result: (bool(result["success"]), float(result["score"])), reverse=True)
    selected = dict(candidate_results[0])
    selected["candidates"] = [dict(result) for result in candidate_results]
    return selected


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate the spawned rollout process group to avoid lingering Isaac processes."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=10)


def build_demo_definition() -> tuple[list[dict[str, object]], dict[str, object], Path, dict[str, object]]:
    """Build the five-cube tightly packed plus-shape basket demo definition."""
    cube_spacing = 0.0515
    object_initial_positions = {
        "cube_center": [0.50, 0.00, 0.0258],
        "cube_north": [0.50, round(cube_spacing, 4), 0.0258],
        "cube_east": [round(0.50 + cube_spacing, 4), 0.00, 0.0258],
        "cube_south": [0.50, round(-cube_spacing, 4), 0.0258],
        "cube_west": [round(0.50 - cube_spacing, 4), 0.00, 0.0258],
    }
    object_annotations = {
        "cube_center": {"cube_id": "C01", "color": "#D84A4A", "goal_region": "basket_goal"},
        "cube_north": {"cube_id": "C02", "color": "#3274D9", "goal_region": "basket_goal"},
        "cube_east": {"cube_id": "C03", "color": "#3FAF5A", "goal_region": "basket_goal"},
        "cube_south": {"cube_id": "C04", "color": "#D9A232", "goal_region": "basket_goal"},
        "cube_west": {"cube_id": "C05", "color": "#8E5CD9", "goal_region": "basket_goal"},
    }
    basket_target_position = [0.22, 0.46, 0.08]
    basket_slots = build_basket_slots(
        basket_center=tuple(basket_target_position),
        cube_rest_height=0.0258,
        count=len(object_initial_positions),
    )
    planning_result = select_planning_result(object_initial_positions)
    ordered_names = list(planning_result["order"])
    subtasks = [
        {
            "object": object_name,
            "cube_id": object_annotations[object_name]["cube_id"],
            "cube_initial_position": object_initial_positions[object_name],
            "target_position": [basket_slots[index][0], basket_slots[index][1], basket_target_position[2]],
            "expected_cube_position": list(basket_slots[index]),
            "goal_region": "basket_goal",
        }
        for index, object_name in enumerate(ordered_names)
    ]
    scene_metadata = {
        "scenario_name": "five_cubes_plus_shape_packed_planning",
        "object_sim_positions": dict(object_initial_positions),
        "object_sim_orientations": {
            object_name: [1.0, 0.0, 0.0, 0.0]
            for object_name in object_initial_positions
        },
        "object_goal_regions": {
            object_name: "basket_goal"
            for object_name in object_initial_positions
        },
        "goal_region_sim_positions": {
            "basket_goal": list(basket_target_position),
        },
        "target_basket_position": list(basket_target_position),
        "object_annotations": object_annotations,
        "planning_summary": planning_result,
    }
    output_dir = (
        resolve_storage_root()
        / "outputs"
        / "showcase_demo_five_cubes_plus_shape_packed"
    )
    return subtasks, scene_metadata, output_dir, planning_result


def run_subtask(
    script_path: Path,
    output_dir: Path,
    subtask: dict[str, object],
    scene_metadata: dict[str, object],
    subtask_index: int,
    total_subtasks: int,
    planner_name: str,
    selected_strategy: str,
) -> dict[str, object]:
    """Run one rollout subprocess and load its manifest."""
    subtask_output_dir = output_dir / f"subtask_{subtask_index:02d}_{subtask['object']}"
    subtask_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--mode",
        "single-subtask",
        "--output-dir",
        str(output_dir),
        "--planner-name",
        planner_name,
        "--selected-strategy",
        selected_strategy,
        "--selected-target-object",
        str(subtask["object"]),
        "--cube-initial-position",
        json.dumps(subtask["cube_initial_position"]),
        "--target-position",
        json.dumps(subtask["target_position"]),
        "--expected-cube-position",
        json.dumps(subtask["expected_cube_position"]),
        "--placement-tolerance",
        "0.12",
        "--scene-metadata-json",
        json.dumps(scene_metadata),
        "--subtask-index",
        str(subtask_index),
        "--total-subtasks",
        str(total_subtasks),
    ]
    manifest_path = subtask_output_dir / "manifest.json"
    process = subprocess.Popen(command, start_new_session=True)
    start_time = time.time()
    manifest_detected_at: float | None = None

    while True:
        return_code = process.poll()
        manifest_exists = manifest_path.exists()

        if manifest_exists and manifest_detected_at is None:
            manifest_detected_at = time.time()

        if return_code is not None:
            if return_code != 0 and not manifest_exists:
                raise subprocess.CalledProcessError(return_code, command)
            break

        if manifest_detected_at is not None:
            if time.time() - manifest_detected_at >= MANIFEST_READY_GRACE_SECONDS:
                terminate_process_group(process)
                break

        if time.time() - start_time >= SUBTASK_TIMEOUT_SECONDS:
            terminate_process_group(process)
            if not manifest_exists:
                raise TimeoutError(
                    f"Subtask {subtask_index} did not produce a manifest within "
                    f"{SUBTASK_TIMEOUT_SECONDS} seconds."
                )
            break

        time.sleep(1)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected subtask manifest at {manifest_path}")
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary["manifest_path"] = str(manifest_path)
    return summary


def combine_rollout_gifs(output_dir: Path, subtask_summaries: list[dict[str, object]]) -> Path:
    """Concatenate the per-subtask GIFs into one combined showcase GIF."""
    frames: list[Image.Image] = []
    for summary in subtask_summaries:
        gif_path = Path(str(summary["gif_path"]))
        with Image.open(gif_path) as image:
            for frame_index in range(image.n_frames):
                image.seek(frame_index)
                frames.append(image.convert("RGB").copy())
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / "rollout.gif"
    frames[0].save(combined_path, save_all=True, append_images=frames[1:], duration=80, loop=0)
    return combined_path


def main() -> None:
    """Run the five-cube plus-shape basket demo without loading Isaac in the parent process."""
    subtasks, scene_metadata, output_dir, planning_result = build_demo_definition()
    if output_dir.exists():
        for stale_path in output_dir.glob("subtask_*"):
            if stale_path.is_dir():
                shutil.rmtree(stale_path, ignore_errors=True)
        combined_dir = output_dir / "combined"
        if combined_dir.exists():
            shutil.rmtree(combined_dir, ignore_errors=True)
        summary_path = output_dir / "combined_summary.json"
        if summary_path.exists():
            summary_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve().parent / "record_franka_pick_place_animation.py"

    summaries: list[dict[str, object]] = []
    mutable_scene_metadata = json.loads(json.dumps(scene_metadata))
    for subtask_index, subtask in enumerate(subtasks, start=1):
        summary = run_subtask(
            script_path=script_path,
            output_dir=output_dir,
            subtask=subtask,
            scene_metadata=mutable_scene_metadata,
            subtask_index=subtask_index,
            total_subtasks=len(subtasks),
            planner_name=str(planning_result["planner_name"]),
            selected_strategy=str(planning_result["strategy_name"]),
        )
        summaries.append(summary)
        final_translation = summary.get("final_active_object_translation")
        if isinstance(final_translation, list) and len(final_translation) == 3:
            mutable_scene_metadata["object_sim_positions"][str(subtask["object"])] = final_translation
        final_orientation = summary.get("final_active_object_orientation")
        if isinstance(final_orientation, list) and len(final_orientation) == 4:
            mutable_scene_metadata.setdefault("object_sim_orientations", {})[str(subtask["object"])] = final_orientation
        if not summary.get("success"):
            break

    combined_gif_path = combine_rollout_gifs(output_dir, summaries)
    combined_summary = {
        "success": all(bool(summary.get("success")) for summary in summaries) and len(summaries) == len(subtasks),
        "completed_subtasks": len(summaries),
        "gif_path": str(combined_gif_path),
        "subtask_manifests": [summary["manifest_path"] for summary in summaries],
        "scene_metadata": mutable_scene_metadata,
        "planning_summary": planning_result,
    }
    summary_path = output_dir / "combined_summary.json"
    summary_path.write_text(json.dumps(combined_summary, indent=2), encoding="utf-8")
    print(json.dumps(combined_summary, indent=2))


if __name__ == "__main__":
    main()
