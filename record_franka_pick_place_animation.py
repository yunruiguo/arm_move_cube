"""Record a real Franka pick-and-place rollout to frames plus a GIF."""

from __future__ import annotations

import argparse
import json
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HEADLESS = True
MAX_SIMULATION_STEPS = 1200
MAX_SAVED_FRAMES = 180
SUBTASK_TIMEOUT_SECONDS = 180
MANIFEST_READY_GRACE_SECONDS = 5


def resolve_storage_root() -> Path:
    """Pick a large writable data volume when available."""
    candidate_roots = [
        Path("/mnt/data2"),
        Path("/data2"),
        Path("/data3"),
        Path("/data1"),
    ]
    for root in candidate_roots:
        if root.exists() and os.access(root, os.W_OK):
            return root
    return Path.cwd()


STORAGE_ROOT = resolve_storage_root()
DEFAULT_OUTPUT_DIR = STORAGE_ROOT / "decision_platform_outputs" / "franka_pick_place_animation"
DEFAULT_TEMP_ROOT = STORAGE_ROOT / "decision_platform_tmp"


def configure_runtime_environment() -> None:
    """Use stable temp/output locations and avoid accidental X/GLX startup."""
    DEFAULT_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    for env_name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[env_name] = str(DEFAULT_TEMP_ROOT)
    if HEADLESS:
        os.environ.pop("DISPLAY", None)


configure_runtime_environment()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=HEADLESS, enable_cameras=True)
simulation_app = app_launcher.app

import isaacsim
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg


def ensure_examples_path() -> None:
    """Make the official Franka examples importable in standalone mode."""
    exts_root = Path(isaacsim.__file__).resolve().parent / "exts"
    required_exts = [
        "isaacsim.robot.manipulators.examples",
        "isaacsim.robot.manipulators",
        "isaacsim.robot.surface_gripper",
        "isaacsim.robot.schema",
        "isaacsim.robot_motion.motion_generation",
        "isaacsim.robot_motion.lula",
        "isaacsim.core.experimental.utils",
        "isaacsim.core.experimental.objects",
        "isaacsim.core.experimental.prims",
        "isaacsim.core.experimental.materials",
    ]

    for ext_name in required_exts:
        ext_root = exts_root / ext_name
        package_root = ext_root / "isaacsim"
        ext_root_str = str(ext_root)
        package_root_str = str(package_root)
        if ext_root.exists() and ext_root_str not in sys.path:
            sys.path.append(ext_root_str)
        if package_root.exists() and package_root_str not in getattr(isaacsim, "__path__", []):
            isaacsim.__path__.append(package_root_str)
        pip_prebundle_root = ext_root / "pip_prebundle"
        pip_prebundle_root_str = str(pip_prebundle_root)
        if pip_prebundle_root.exists() and pip_prebundle_root_str not in sys.path:
            sys.path.append(pip_prebundle_root_str)
        usd_root = ext_root / "usd"
        usd_root_str = str(usd_root)
        if usd_root.exists() and usd_root_str not in sys.path:
            sys.path.append(usd_root_str)
        usd_schema_root = usd_root / "schema"
        usd_schema_root_str = str(usd_schema_root)
        if usd_schema_root.exists() and usd_schema_root_str not in sys.path:
            sys.path.append(usd_schema_root_str)

    nested_package_roots = {
        "isaacsim.robot.manipulators": (
            exts_root
            / "isaacsim.robot.manipulators.examples"
            / "isaacsim"
            / "robot"
            / "manipulators"
        ),
        "isaacsim.robot_motion": (
            exts_root
            / "isaacsim.robot_motion.motion_generation"
            / "isaacsim"
            / "robot_motion"
        ),
    }
    for package_name, package_root in nested_package_roots.items():
        try:
            package = importlib.import_module(package_name)
            package_root_str = str(package_root)
            if (
                package_root.exists()
                and hasattr(package, "__path__")
                and package_root_str not in package.__path__
            ):
                package.__path__.append(package_root_str)
        except Exception:
            pass


ensure_examples_path()

from isaacsim.robot.manipulators.examples.franka.pick_place.pick_place import (
    FrankaPickPlace,
)
import numpy as np


CAMERA_POSITION = (2.5, 2.5, 2.5)
CAMERA_QUAT_WORLD = (-0.3647052, -0.27984815, -0.1159169, 0.88047623)
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
CAPTURE_INTERVAL = 2
DEFAULT_PLANNER_NAME = "franka-pick-place-example"
DEFAULT_STRATEGY_NAME = "fixed_pick_place_rollout"
DEFAULT_ACTION_SEQUENCE = [
    "move_above_cube",
    "approach_cube",
    "close_gripper",
    "lift_cube",
    "move_to_goal",
    "open_gripper",
    "retreat_up",
]

Subtask = dict[str, object]

EVENT_LABELS = {
    0: "move_above_cube",
    1: "approach_cube",
    2: "close_gripper",
    3: "lift_cube",
    4: "move_to_goal",
    5: "open_gripper",
    6: "retreat_up",
}


class ShowcaseFrankaPickPlace(FrankaPickPlace):
    """Thin wrapper around the default Isaac Sim Franka pick-place controller."""

    pass


DEFAULT_DOWNWARD_ORIENTATION = [0.0, 1.0, 0.0, 0.0]
ROTATED_90_DOWNWARD_ORIENTATION = [0.0, 0.7071, 0.7071, 0.0]


def _round_vector(values: tuple[float, float, float]) -> list[float]:
    """Round 3D values for readable debug output."""
    return [round(float(value), 4) for value in values]


def _format_object_pose_lines(object_snapshots: list[dict[str, object]]) -> list[str]:
    """Format object pose records into compact human-readable lines."""
    lines: list[str] = []
    for snapshot in object_snapshots:
        lines.append(
            f"- {snapshot['name']}: path={snapshot['path']}, "
            f"translation={snapshot['translation']}"
        )
    return lines


def _euclidean_distance(
    first: list[float] | tuple[float, float, float] | None,
    second: list[float] | tuple[float, float, float] | None,
) -> float | None:
    """Compute a small 3D Euclidean distance helper for rollout checks."""
    if first is None or second is None:
        return None
    return sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)) ** 0.5


def _quaternion_alignment(
    first: np.ndarray | list[float] | tuple[float, float, float, float],
    second: np.ndarray | list[float] | tuple[float, float, float, float],
) -> float:
    """Return absolute quaternion dot-product as a simple orientation agreement score."""
    first_q = np.array(first, dtype=float)
    second_q = np.array(second, dtype=float)
    first_norm = np.linalg.norm(first_q)
    second_norm = np.linalg.norm(second_q)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    first_q = first_q / first_norm
    second_q = second_q / second_norm
    return float(abs(np.dot(first_q, second_q)))


def _normalize_quaternion(values: list[float] | tuple[float, float, float, float]) -> list[float]:
    """Normalize a quaternion into a stable [w, x, y, z] list."""
    quat = np.array(values, dtype=float)
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm == 0.0:
        return list(DEFAULT_DOWNWARD_ORIENTATION)
    quat = quat / quat_norm
    return [float(value) for value in quat.tolist()]


def _select_goal_orientation(
    scene_metadata: dict[str, object] | None,
    selected_target_object: str,
) -> tuple[list[float], str]:
    """Choose a downward grasp orientation, rotating 90 degrees when local clutter suggests a side approach.

    Heuristic:
    - If a target has stronger north/south crowding than east/west crowding, rotate the gripper 90 degrees
      around the vertical axis so the fingers approach from the orthogonal direction.
    - Otherwise keep the default downward orientation.
    """
    if not isinstance(scene_metadata, dict):
        return list(DEFAULT_DOWNWARD_ORIENTATION), "default downward orientation (no scene metadata)"

    object_positions = scene_metadata.get("object_sim_positions")
    if not isinstance(object_positions, dict):
        return list(DEFAULT_DOWNWARD_ORIENTATION), "default downward orientation (no object positions)"

    target_position = object_positions.get(selected_target_object)
    if not isinstance(target_position, list) or len(target_position) != 3:
        return list(DEFAULT_DOWNWARD_ORIENTATION), "default downward orientation (target pose unavailable)"

    target_x, target_y = float(target_position[0]), float(target_position[1])
    north = south = east = west = 0
    for object_name, other_position in object_positions.items():
        if object_name == selected_target_object:
            continue
        if not isinstance(other_position, list) or len(other_position) != 3:
            continue
        dx = float(other_position[0]) - target_x
        dy = float(other_position[1]) - target_y
        if abs(dx) <= 0.05 and 0.03 <= dy <= 0.14:
            north += 1
        if abs(dx) <= 0.05 and -0.14 <= dy <= -0.03:
            south += 1
        if abs(dy) <= 0.05 and 0.03 <= dx <= 0.14:
            east += 1
        if abs(dy) <= 0.05 and -0.14 <= dx <= -0.03:
            west += 1

    if north > 0 and south > 0 and not (east > 0 or west > 0):
        return (
            list(ROTATED_90_DOWNWARD_ORIENTATION),
            (
                "rotated 90 degrees for side approach through east-west clearance "
                f"(north={north}, south={south}, east={east}, west={west})"
            ),
        )
    if east > 0 and west > 0 and not (north > 0 or south > 0):
        return (
            list(DEFAULT_DOWNWARD_ORIENTATION),
            (
                "default downward orientation for north-south approach through side clearance "
                f"(north={north}, south={south}, east={east}, west={west})"
            ),
        )
    return (
        list(DEFAULT_DOWNWARD_ORIENTATION),
        (
            "default downward orientation "
            f"(north={north}, south={south}, east={east}, west={west})"
        ),
    )


def _install_goal_orientation(controller: Any, orientation: list[float], reason: str) -> None:
    """Install a preferred goal orientation into the default controller via the robot helper."""
    normalized = _normalize_quaternion(orientation)
    orientation_array = np.array([normalized], dtype=float)
    controller.preferred_goal_orientation = normalized
    controller.preferred_goal_orientation_reason = reason
    controller.robot.get_downward_orientation = lambda: orientation_array.copy()


def _extract_active_cube_translation(object_snapshots: list[dict[str, object]]) -> list[float] | None:
    """Read the main physical cube pose from a stage snapshot when available."""
    for snapshot in object_snapshots:
        if snapshot.get("path") == "/World/Cube":
            translation = snapshot.get("translation")
            if isinstance(translation, list):
                return translation
    return None


def _extract_controller_cube_translation(controller: Any) -> list[float] | None:
    """Read the active cube pose directly from the physics-backed controller object."""
    try:
        cube_pose = controller.cube.get_world_poses()[0].numpy()[0].tolist()
    except Exception:
        return None
    return _round_position_list(cube_pose)


def _extract_controller_cube_orientation(controller: Any) -> list[float] | None:
    """Read the active cube orientation directly from the physics-backed controller object."""
    try:
        cube_orientation = controller.cube.get_world_poses()[1].numpy()[0].tolist()
    except Exception:
        return None
    return _round_quaternion_list(cube_orientation)


def _round_position_list(values: list[float] | tuple[float, float, float] | None) -> list[float] | None:
    """Round a 3D position into a compact list."""
    if values is None:
        return None
    return [round(float(value), 4) for value in values]


def _round_quaternion_list(values: list[float] | tuple[float, float, float, float] | None) -> list[float] | None:
    """Round a quaternion into a compact readable list."""
    if values is None:
        return None
    return [round(float(value), 4) for value in values]


def _expected_cube_rest_position(
    cube_initial_position: tuple[float, float, float] | None,
    target_position: tuple[float, float, float] | None,
    explicit_expected_position: tuple[float, float, float] | list[float] | None = None,
) -> list[float]:
    """Build the expected resting cube position on the table for validation."""
    if explicit_expected_position is not None:
        return _round_vector(tuple(float(value) for value in explicit_expected_position))
    if target_position is None:
        target_position = (0.0, 0.0, 0.0)
    z_value = cube_initial_position[2] if cube_initial_position is not None else 0.0258
    return _round_vector((target_position[0], target_position[1], z_value))


def _capture_phase_debug_snapshot(
    controller: Any,
    simulation_step: int,
    target_position: tuple[float, float, float] | None,
) -> dict[str, object]:
    """Capture a compact controller/robot/object state snapshot for phase debugging."""
    current_dof_positions, current_end_effector_position, current_end_effector_orientation = (
        controller.robot.get_current_state()
    )
    cube_position = controller.cube.get_world_poses()[0].numpy()[0].tolist()
    ee_position = current_end_effector_position[0].tolist()
    ee_orientation = current_end_effector_orientation[0].tolist()
    relative_cube_position = [
        round(float(cube_value) - float(ee_value), 4)
        for cube_value, ee_value in zip(cube_position, ee_position)
    ]
    event_index = int(controller._event)
    target_orientation = getattr(controller, "preferred_goal_orientation", DEFAULT_DOWNWARD_ORIENTATION)
    target_orientation_reason = getattr(
        controller,
        "preferred_goal_orientation_reason",
        "default downward orientation",
    )
    return {
        "simulation_step": int(simulation_step),
        "event_index": event_index,
        "event_label": EVENT_LABELS.get(event_index, f"event_{event_index}"),
        "event_progress_step": int(controller._step),
        "gripper_dofs": _round_position_list(current_dof_positions[0][7:9].tolist()),
        "end_effector_position": _round_position_list(ee_position),
        "end_effector_orientation": _round_quaternion_list(ee_orientation),
        "target_end_effector_orientation": _round_quaternion_list(target_orientation),
        "target_end_effector_orientation_reason": target_orientation_reason,
        "cube_position": _round_position_list(cube_position),
        "end_effector_target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "cube_to_end_effector_target_distance": (
            round(_euclidean_distance(cube_position, target_position), 4)
            if target_position is not None
            else None
        ),
        "cube_to_target_distance": (
            round(_euclidean_distance(cube_position, target_position), 4)
            if target_position is not None
            else None
        ),
        "end_effector_to_cube_distance": round(
            _euclidean_distance(ee_position, cube_position) or 0.0,
            4,
        ),
        "cube_relative_to_end_effector": relative_cube_position,
    }


def _positions_match(
    first: list[float] | None,
    second: tuple[float, float, float] | None,
    tolerance: float = 1e-3,
) -> bool:
    """Check whether two 3D positions are effectively the same."""
    if first is None or second is None:
        return False
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second))


def _capture_stage_object_snapshots() -> list[dict[str, object]]:
    """Collect a small readable set of spawned object names and poses after reset."""
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []

    object_snapshots: list[dict[str, object]] = []
    interesting_keywords = (
        "franka",
        "cube",
        "context",
        "table",
        "ground",
        "target",
        "basket",
        "tray",
    )

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        lower_name = prim_name.lower()
        lower_path = prim_path.lower()
        if not any(keyword in lower_name or keyword in lower_path for keyword in interesting_keywords):
            continue

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue

        try:
            world_transform = xformable.ComputeLocalToWorldTransform(0.0)
            translation = _round_vector(tuple(world_transform.ExtractTranslation()))
        except Exception:
            continue

        object_snapshots.append(
            {
                "name": prim_name,
                "path": prim_path,
                "translation": translation,
            }
        )

    object_snapshots.sort(key=lambda snapshot: str(snapshot["path"]))
    return object_snapshots


def _build_annotation_map(scene_metadata: dict[str, object] | None) -> dict[str, dict[str, object]]:
    """Read stable object annotations from the scenario metadata."""
    if not scene_metadata:
        return {}
    object_annotations = scene_metadata.get("object_annotations")
    if not isinstance(object_annotations, dict):
        return {}
    return {
        str(object_name): annotation
        for object_name, annotation in object_annotations.items()
        if isinstance(annotation, dict)
    }


def _build_orientation_map(scene_metadata: dict[str, object] | None) -> dict[str, list[float]]:
    """Read stable object orientations from the scenario metadata when available."""
    if not scene_metadata:
        return {}
    object_orientations = scene_metadata.get("object_sim_orientations")
    if not isinstance(object_orientations, dict):
        return {}
    normalized: dict[str, list[float]] = {}
    for object_name, orientation in object_orientations.items():
        if isinstance(orientation, (list, tuple)) and len(orientation) == 4:
            normalized[str(object_name)] = [float(value) for value in orientation]
    return normalized


def _lookup_cube_id(scene_metadata: dict[str, object] | None, object_name: str) -> str:
    """Resolve a stable object ID for logs and manifests."""
    annotation_map = _build_annotation_map(scene_metadata)
    annotation = annotation_map.get(object_name, {})
    cube_id = annotation.get("cube_id")
    if isinstance(cube_id, str):
        return cube_id
    return object_name


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    """Convert a #RRGGBB string into integer RGB components."""
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        return (120, 120, 120)
    return tuple(int(color_hex[index : index + 2], 16) for index in (0, 2, 4))


def _build_goal_slot_offsets() -> tuple[tuple[float, float], ...]:
    """Return small XY offsets so multiple cubes stay visible inside one basket."""
    return (
        (0.0, 0.0),
        (-0.045, -0.02),
        (0.045, -0.02),
        (-0.045, 0.04),
        (0.045, 0.04),
        (0.0, 0.065),
    )


def _resolve_context_display_position(
    scene_metadata: dict[str, object] | None,
    object_name: str,
    raw_position: tuple[float, float, float] | list[float],
) -> tuple[float, float, float]:
    """Use the recorded physical position directly for context display consistency."""
    x_value, y_value, z_value = (float(value) for value in raw_position)
    return (x_value, y_value, z_value)


def write_debug_summary(summary_path: Path, summary_payload: dict[str, object]) -> None:
    """Persist a lightweight text summary next to the rollout outputs."""
    lines = [
        "Real Franka Rollout Debug Summary",
        f"planner: {summary_payload['planner_name']}",
        f"strategy: {summary_payload['selected_strategy']}",
        f"selected target object: {summary_payload['selected_target_object']}",
        f"selected cube id: {summary_payload.get('selected_cube_id')}",
        f"end-effector target position: {summary_payload.get('end_effector_target_position', summary_payload['target_position'])}",
        f"expected cube position: {summary_payload.get('expected_cube_position')}",
        f"final active object translation: {summary_payload.get('final_active_object_translation')}",
        f"position error: {summary_payload.get('position_error')}",
        f"placement success: {summary_payload.get('placement_success')}",
        f"id consistency passed: {summary_payload.get('id_consistency_passed')}",
        f"object-level subtasks: {summary_payload['object_level_subtasks']}",
        f"success: {summary_payload['success']}",
        f"simulation steps: {summary_payload['simulation_steps']}",
        f"captured frames: {summary_payload['num_frames']}",
        f"gif: {summary_payload['gif_path']}",
        "spawned objects after reset:",
        *_format_object_pose_lines(summary_payload["spawned_objects"]),
    ]
    phase_debug_timeline = summary_payload.get("phase_debug_timeline")
    if isinstance(phase_debug_timeline, list) and phase_debug_timeline:
        lines.extend(["", "phase debug timeline:"])
        for record in phase_debug_timeline:
            if not isinstance(record, dict):
                continue
            lines.append(
                "- "
                f"step={record.get('simulation_step')} "
                f"event={record.get('event_label')} "
                f"gripper={record.get('gripper_dofs')} "
                f"ee={record.get('end_effector_position')} "
                f"ee_q={record.get('end_effector_orientation')} "
                f"cube={record.get('cube_position')} "
                f"cube_to_target={record.get('cube_to_end_effector_target_distance', record.get('cube_to_target_distance'))} "
                f"ee_to_cube={record.get('end_effector_to_cube_distance')}"
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_scene_config_summary() -> dict[str, object]:
    """Describe the fixed real-world rollout scene configuration."""
    return {
        "task": "Isaac Sim Franka pick-place example",
        "headless": HEADLESS,
        "camera_position": CAMERA_POSITION,
        "camera_quat_world": CAMERA_QUAT_WORLD,
        "camera_resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
        "capture_interval": CAPTURE_INTERVAL,
        "storage_root": str(STORAGE_ROOT),
    }


def _to_numpy_vector(value: tuple[float, float, float] | None) -> Any | None:
    """Convert a simple tuple into a numpy vector only when needed."""
    if value is None:
        return None
    numpy = __import__("numpy")
    return numpy.array(value, dtype=float)


def to_numpy_image(rgb_frame: Any) -> Any:
    """Convert an Isaac RGB frame into a HxWx3 uint8 numpy array."""
    numpy = __import__("numpy")
    rgb_array = rgb_frame
    if hasattr(rgb_array, "detach"):
        rgb_array = rgb_array.detach()
    if hasattr(rgb_array, "cpu"):
        rgb_array = rgb_array.cpu()
    if hasattr(rgb_array, "numpy"):
        rgb_array = rgb_array.numpy()
    rgb_array = numpy.asarray(rgb_array)
    if rgb_array.ndim == 4:
        rgb_array = rgb_array[0]
    rgb_array = rgb_array[..., :3].astype(numpy.uint8)
    return rgb_array


def save_png_frame(rgb_array: Any, output_path: Path) -> None:
    """Save one RGB frame as PNG using Pillow."""
    from PIL import Image

    image = Image.fromarray(rgb_array, mode="RGB")
    image.save(output_path)


def save_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int = 80) -> None:
    """Build a GIF from saved PNG frames."""
    from PIL import Image

    frames = [Image.open(frame_path).convert("RGB") for frame_path in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    for frame in frames:
        frame.close()


def copy_frame_sequence(frame_paths: list[Path], destination_dir: Path, start_index: int) -> list[Path]:
    """Copy a frame sequence into a combined destination directory with new numbering."""
    import shutil

    copied_paths: list[Path] = []
    for offset, source_path in enumerate(frame_paths):
        destination_path = destination_dir / f"frame_{start_index + offset:04d}.png"
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


def annotate_frame(
    frame_path: Path,
    title: str,
    subtitle: str,
    footer_lines: list[str] | None = None,
) -> None:
    """Overlay a compact readable banner onto a saved frame."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    footer_lines = footer_lines or []
    footer_height = 18 * len(footer_lines) if footer_lines else 0
    top_box_height = 54
    draw.rounded_rectangle(
        (10, 10, image.width - 10, 10 + top_box_height),
        radius=10,
        fill=(0, 0, 0, 175),
    )
    draw.text((24, 18), title, fill=(255, 255, 255), font=font)
    draw.text((24, 34), subtitle, fill=(220, 220, 220), font=font)

    if footer_lines:
        footer_top = image.height - footer_height - 18
        draw.rounded_rectangle(
            (10, footer_top, image.width - 10, image.height - 10),
            radius=10,
            fill=(0, 0, 0, 150),
        )
        for line_index, line in enumerate(footer_lines):
            draw.text(
                (24, footer_top + 10 + line_index * 18),
                line,
                fill=(235, 235, 235),
                font=font,
            )

    image.save(frame_path)
    image.close()


def create_text_card(
    output_path: Path,
    title: str,
    body_lines: list[str],
    width: int = CAMERA_WIDTH,
    height: int = CAMERA_HEIGHT,
) -> None:
    """Create a simple intro/interstitial frame for the combined GIF."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    draw.rounded_rectangle(
        (30, 30, width - 30, height - 30),
        radius=18,
        fill=(245, 245, 245, 255),
        outline=(40, 40, 40, 255),
        width=2,
    )
    draw.text((56, 56), title, fill=(20, 20, 20), font=font)
    text_top = 92
    for line_index, line in enumerate(body_lines):
        draw.text((56, text_top + line_index * 20), line, fill=(55, 55, 55), font=font)

    image.save(output_path)
    image.close()


def parse_optional_vector(raw_value: str | None) -> tuple[float, float, float] | None:
    """Parse a JSON-encoded xyz vector when provided."""
    if not raw_value:
        return None
    values = json.loads(raw_value)
    return (float(values[0]), float(values[1]), float(values[2]))


def parse_optional_json_dict(raw_value: str | None) -> dict[str, object] | None:
    """Parse a JSON object payload when provided."""
    if not raw_value:
        return None
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object payload.")
    return parsed


def run_single_subtask_subprocess(
    output_path: Path,
    planner_name: str,
    selected_strategy: str,
    subtask: Subtask,
    placement_tolerance: float,
    scene_metadata: dict[str, object] | None,
    subtask_index: int,
    total_subtasks: int,
) -> dict[str, object]:
    """Run one real subtask in a fresh Isaac process, then load its manifest."""
    subtask_output_path = output_path / f"subtask_{subtask_index:02d}_{subtask['object']}"
    manifest_path = subtask_output_path / "manifest.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "single-subtask",
        "--output-dir",
        str(output_path),
        "--planner-name",
        planner_name,
        "--selected-strategy",
        selected_strategy,
        "--selected-target-object",
        str(subtask["object"]),
        "--cube-initial-position",
        json.dumps(list(subtask["cube_initial_position"])),
        "--target-position",
        json.dumps(list(subtask["target_position"])),
        "--expected-cube-position",
        json.dumps(list(subtask.get("expected_cube_position", subtask["target_position"]))),
        "--placement-tolerance",
        str(placement_tolerance),
        "--scene-metadata-json",
        json.dumps(scene_metadata or {}),
        "--subtask-index",
        str(subtask_index),
        "--total-subtasks",
        str(total_subtasks),
    ]
    process = subprocess.Popen(command)
    manifest_ready_deadline: float | None = None

    import time

    start_time = time.monotonic()
    while True:
        return_code = process.poll()
        if return_code is not None:
            if return_code != 0 and not manifest_path.exists():
                raise subprocess.CalledProcessError(return_code, command)
            break

        elapsed_seconds = time.monotonic() - start_time
        if manifest_path.exists():
            if manifest_ready_deadline is None:
                manifest_ready_deadline = time.monotonic() + MANIFEST_READY_GRACE_SECONDS
            elif time.monotonic() >= manifest_ready_deadline:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                print(
                    "[record_franka_pick_place_animation] subtask outputs ready; "
                    "terminated lingering process:",
                    f"{subtask_index}/{total_subtasks} {subtask['object']}",
                )
                break

        if elapsed_seconds >= SUBTASK_TIMEOUT_SECONDS:
            if not manifest_path.exists():
                process.kill()
                process.wait(timeout=10)
                raise subprocess.TimeoutExpired(command, SUBTASK_TIMEOUT_SECONDS)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            print(
                "[record_franka_pick_place_animation] subtask timed out after writing outputs:",
                f"{subtask_index}/{total_subtasks} {subtask['object']}",
            )
            break

        time.sleep(1)
    summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary["manifest_path"] = str(manifest_path)
    debug_summary_path = subtask_output_path / "debug_summary.txt"
    summary["debug_summary_path"] = str(debug_summary_path)
    return summary


def clone_scene_metadata(scene_metadata: dict[str, object] | None) -> dict[str, object] | None:
    """Clone scene metadata into a mutable structure."""
    if not scene_metadata:
        return None
    return json.loads(json.dumps(scene_metadata))


def build_frame_footer_lines(
    scene_metadata: dict[str, object] | None,
    selected_target_object: str,
) -> list[str]:
    """Build short footer lines describing all objects currently in the scene."""
    if not scene_metadata:
        return []
    object_positions = scene_metadata.get("object_sim_positions")
    if not isinstance(object_positions, dict):
        return []
    object_annotations = _build_annotation_map(scene_metadata)

    footer_lines: list[str] = []
    for object_name in sorted(object_positions):
        raw_position = object_positions[object_name]
        if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 3:
            continue
        prefix = "*" if object_name == selected_target_object else "-"
        rounded = [round(float(value), 3) for value in raw_position]
        cube_id = object_annotations.get(object_name, {}).get("cube_id", object_name)
        footer_lines.append(f"{prefix} {cube_id} {object_name}: {rounded}")
    return footer_lines


def build_intro_card_lines(
    planner_name: str,
    selected_strategy: str,
    subtasks: list[Subtask],
    scene_metadata: dict[str, object] | None,
) -> list[str]:
    """Create a clear legend page for the combined GIF."""
    lines = [
        f"planner: {planner_name}",
        f"strategy: {selected_strategy}",
        f"subtasks: {len(subtasks)}",
        "",
        "legend:",
    ]
    object_annotations = _build_annotation_map(scene_metadata)
    for object_name in sorted(object_annotations):
        annotation = object_annotations[object_name]
        cube_id = annotation.get("cube_id", object_name)
        goal_region = annotation.get("goal_region", "?")
        color = annotation.get("color", "#777777")
        lines.append(f"{cube_id} | {object_name} | goal={goal_region} | color={color}")
    lines.extend(
        [
            "",
            "note:",
            "controller target is an end-effector pose above the table.",
            "success is validated from the cube's final resting pose.",
        ]
    )
    return lines


def verify_id_consistency(
    scene_metadata: dict[str, object] | None,
    tracked_positions: dict[str, list[float]],
    tolerance: float = 1e-3,
) -> tuple[bool, list[str]]:
    """Check whether carried-over object IDs still appear at the expected positions."""
    if not scene_metadata or not tracked_positions:
        return True, []
    object_annotations = _build_annotation_map(scene_metadata)
    object_positions = scene_metadata.get("object_sim_positions")
    if not isinstance(object_positions, dict):
        return True, []

    mismatches: list[str] = []
    for object_name, annotation in object_annotations.items():
        cube_id = annotation.get("cube_id")
        if not isinstance(cube_id, str) or cube_id not in tracked_positions:
            continue
        actual_position = object_positions.get(object_name)
        expected_position = tracked_positions[cube_id]
        if not isinstance(actual_position, list):
            continue
        distance = _euclidean_distance(actual_position, expected_position)
        if distance is None or distance > tolerance:
            mismatches.append(
                f"{cube_id} {object_name}: expected={_round_position_list(expected_position)} "
                f"actual={_round_position_list(actual_position)}"
            )
    return len(mismatches) == 0, mismatches


def build_camera() -> Camera:
    """Create a fixed wide-angle camera that sees the full robot table scene."""
    camera_cfg = CameraCfg(
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        prim_path="/World/RecordingCamera",
        update_period=0.0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_POSITION,
            rot=CAMERA_QUAT_WORLD,
            convention="world",
        ),
    )
    return Camera(camera_cfg)


def spawn_context_cubes(
    scene_metadata: dict[str, object] | None,
    active_object_name: str,
) -> None:
    """Spawn rigid-body cubes for the other scenario objects."""
    if not scene_metadata:
        return

    object_sim_positions = scene_metadata.get("object_sim_positions")
    if not isinstance(object_sim_positions, dict):
        return

    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    context_root = UsdGeom.Xform.Define(stage, "/World/ContextObjects")
    annotation_map = _build_annotation_map(scene_metadata)
    orientation_map = _build_orientation_map(scene_metadata)

    for object_name, raw_position in object_sim_positions.items():
        if object_name == active_object_name:
            continue
        if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 3:
            continue

        display_position = _resolve_context_display_position(
            scene_metadata,
            str(object_name),
            raw_position,
        )

        cube_path = f"{context_root.GetPath()}/ContextCube_{object_name}"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.CreateSizeAttr(1.0)
        color_hex = str(annotation_map.get(object_name, {}).get("color", "#777777"))
        color_rgb = _hex_to_rgb(color_hex)
        cube.CreateDisplayColorAttr(
            [Gf.Vec3f(color_rgb[0] / 255.0, color_rgb[1] / 255.0, color_rgb[2] / 255.0)]
        )
        xformable = UsdGeom.Xformable(cube.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*display_position))
        orientation = orientation_map.get(str(object_name), [1.0, 0.0, 0.0, 0.0])
        xformable.AddOrientOp().Set(
            Gf.Quatf(
                float(orientation[0]),
                Gf.Vec3f(float(orientation[1]), float(orientation[2]), float(orientation[3])),
            )
        )
        xformable.AddScaleOp().Set(Gf.Vec3f(0.055, 0.055, 0.055))

        cube_prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(cube_prim)
        UsdPhysics.RigidBodyAPI.Apply(cube_prim)
        UsdPhysics.MassAPI.Apply(cube_prim).CreateMassAttr(0.05)
        try:
            physx_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(cube_prim)
            physx_body_api.CreateEnableCCDAttr(True)
        except Exception:
            pass


def spawn_target_basket(target_position: tuple[float, float, float] | None) -> None:
    """Spawn a simple visual basket/tray around the target region."""
    if target_position is None:
        return

    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    target_x, target_y, _target_z = (float(value) for value in target_position)
    table_surface_z = 0.0
    basket_root = UsdGeom.Xform.Define(stage, "/World/TargetBasket")
    basket_root_prim = basket_root.GetPrim()
    try:
        basket_body_api = UsdPhysics.RigidBodyAPI.Apply(basket_root_prim)
        basket_body_api.CreateKinematicEnabledAttr(True)
        physx_basket_api = PhysxSchema.PhysxRigidBodyAPI.Apply(basket_root_prim)
        physx_basket_api.CreateDisableGravityAttr(True)
    except Exception:
        pass

    # Double the basket floor area relative to the original showcase tray so multiple
    # cubes can be shown inside with more separation.
    basket_span = 0.30
    basket_wall_half_offset = 0.155
    basket_wall_height = 0.05
    basket_base_height = 0.01
    basket_wall_thickness = 0.01

    # A shallow tray plus four walls gives us a readable "basket" without changing physics.
    basket_parts = [
        (
            "TrayBase",
            (target_x, target_y, table_surface_z + basket_base_height / 2.0),
            (basket_span, basket_span, basket_base_height),
            "#D9A441",
        ),
        (
            "WallNorth",
            (target_x, target_y + basket_wall_half_offset, table_surface_z + 0.045),
            (basket_span, basket_wall_thickness, basket_wall_height),
            "#C8842F",
        ),
        (
            "WallSouth",
            (target_x, target_y - basket_wall_half_offset, table_surface_z + 0.045),
            (basket_span, basket_wall_thickness, basket_wall_height),
            "#C8842F",
        ),
        (
            "WallEast",
            (target_x + basket_wall_half_offset, target_y, table_surface_z + 0.045),
            (basket_wall_thickness, basket_span, basket_wall_height),
            "#C8842F",
        ),
        (
            "WallWest",
            (target_x - basket_wall_half_offset, target_y, table_surface_z + 0.045),
            (basket_wall_thickness, basket_span, basket_wall_height),
            "#C8842F",
        ),
    ]

    for part_name, translation, scale, color_hex in basket_parts:
        part_path = f"{basket_root.GetPath()}/{part_name}"
        cube = UsdGeom.Cube.Define(stage, part_path)
        cube.CreateSizeAttr(1.0)
        color_rgb = _hex_to_rgb(color_hex)
        cube.CreateDisplayColorAttr(
            [Gf.Vec3f(color_rgb[0] / 255.0, color_rgb[1] / 255.0, color_rgb[2] / 255.0)]
        )
        xform_api = UsdGeom.XformCommonAPI(cube)
        xform_api.SetTranslate(Gf.Vec3d(*translation))
        xform_api.SetScale(Gf.Vec3f(*scale))
        part_prim = cube.GetPrim()
        try:
            UsdPhysics.CollisionAPI.Apply(part_prim)
        except Exception:
            pass


def _resolve_basket_display_position(
    scene_metadata: dict[str, object] | None,
    fallback_target_position: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    """Keep the basket centered at a stable scene goal even when per-cube slots differ."""
    if scene_metadata:
        basket_position = scene_metadata.get("target_basket_position")
        if isinstance(basket_position, (list, tuple)) and len(basket_position) == 3:
            return (
                float(basket_position[0]),
                float(basket_position[1]),
                float(basket_position[2]),
            )
    return fallback_target_position


def _record_single_subtask_rollout(
    output_path: Path,
    planner_name: str,
    selected_strategy: str,
    selected_target_object: str,
    cube_initial_position: tuple[float, float, float] | None,
    target_position: tuple[float, float, float] | None,
    expected_cube_position: tuple[float, float, float] | None,
    placement_tolerance: float,
    scene_metadata: dict[str, object] | None,
    subtask_index: int,
    total_subtasks: int,
) -> dict[str, object]:
    """Run one single-object rollout for a subtask and save it into a subdirectory."""
    subtask_output_path = output_path / f"subtask_{subtask_index:02d}_{selected_target_object}"
    frames_path = subtask_output_path / "frames"
    subtask_output_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    controller = ShowcaseFrankaPickPlace()
    controller.setup_scene(
        cube_initial_position=_to_numpy_vector(cube_initial_position),
        target_position=_to_numpy_vector(target_position),
    )
    preferred_goal_orientation, preferred_goal_orientation_reason = _select_goal_orientation(
        scene_metadata,
        selected_target_object,
    )
    _install_goal_orientation(
        controller,
        preferred_goal_orientation,
        preferred_goal_orientation_reason,
    )

    dt = 0.01
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt))
    sim_utils.update_stage()
    camera = build_camera()

    sim.reset()
    controller.reset(cube_position=_to_numpy_vector(cube_initial_position))
    spawn_target_basket(_resolve_basket_display_position(scene_metadata, target_position))
    spawn_context_cubes(scene_metadata, selected_target_object)
    sim_utils.update_stage()

    spawned_objects = _capture_stage_object_snapshots()
    selected_cube_id = _lookup_cube_id(scene_metadata, selected_target_object)
    print("[record_franka_pick_place_animation] reset debug snapshot:")
    print("  object-level subtasks:", total_subtasks)
    print("  current subtask:", f"{subtask_index}/{total_subtasks}")
    print("  selected target object:", selected_target_object)
    print("  selected cube id:", selected_cube_id)
    print(
        "  end-effector target position:",
        _round_vector(target_position or (0.0, 0.0, 0.0)),
    )
    print(
        "  expected cube position:",
        _round_vector(expected_cube_position or (0.0, 0.0, 0.0)),
    )
    print(
        "  target end-effector orientation:",
        _round_quaternion_list(preferred_goal_orientation),
        f"({preferred_goal_orientation_reason})",
    )
    print("  spawned objects after reset:")
    for line in _format_object_pose_lines(spawned_objects):
        print(f"    {line}")
    print(
        "[record_franka_pick_place_animation] starting low-level execution for:",
        selected_target_object,
    )

    for _ in range(5):
        sim.step()

    saved_frames: list[Path] = []
    simulation_step = 0
    phase_debug_timeline: list[dict[str, object]] = []
    previous_event_index = int(controller._event)
    phase_debug_timeline.append(
        {
            **_capture_phase_debug_snapshot(controller, simulation_step, target_position),
            "note": "post_reset",
        }
    )

    while not controller.is_done() and simulation_step < MAX_SIMULATION_STEPS:
        controller.forward()
        sim.step()
        current_event_index = int(controller._event)
        if current_event_index != previous_event_index:
            phase_debug_timeline.append(
                {
                    **_capture_phase_debug_snapshot(controller, simulation_step, target_position),
                    "note": (
                        f"event_transition:{EVENT_LABELS.get(previous_event_index, previous_event_index)}"
                        f"->{EVENT_LABELS.get(current_event_index, current_event_index)}"
                    ),
                }
            )
            previous_event_index = current_event_index
        if current_event_index in (4, 5) and simulation_step % 10 == 0:
            phase_debug_timeline.append(
                {
                    **_capture_phase_debug_snapshot(controller, simulation_step, target_position),
                    "note": "move_to_goal_open_gripper_window",
                }
            )
        should_capture = (
            simulation_step % CAPTURE_INTERVAL == 0
            and len(saved_frames) < MAX_SAVED_FRAMES
        )
        if should_capture:
            camera.update(dt, force_recompute=True)
            rgb_frame = camera.data.output["rgb"]
            rgb_array = to_numpy_image(rgb_frame)
            frame_path = frames_path / f"frame_{len(saved_frames):04d}.png"
            save_png_frame(rgb_array, frame_path)
            saved_frames.append(frame_path)
        simulation_step += 1

    if not saved_frames:
        raise RuntimeError("No rollout frames were captured before the recording ended.")

    gif_path = subtask_output_path / "rollout.gif"
    save_gif(saved_frames, gif_path)

    final_state_summary = {
        "controller_done": controller.is_done(),
        "simulation_steps": simulation_step,
        "captured_frames": len(saved_frames),
    }
    final_object_snapshots = _capture_stage_object_snapshots()
    stage_cube_translation = _round_position_list(_extract_active_cube_translation(final_object_snapshots))
    physical_cube_translation = _extract_controller_cube_translation(controller)
    physical_cube_orientation = _extract_controller_cube_orientation(controller)
    expected_rest_position = _expected_cube_rest_position(
        cube_initial_position=cube_initial_position,
        target_position=target_position,
        explicit_expected_position=expected_cube_position,
    )
    final_active_object_translation = physical_cube_translation
    position_error = _euclidean_distance(final_active_object_translation, expected_rest_position)
    placement_success = (
        position_error is not None
        and position_error <= placement_tolerance
    )
    manifest_summary = {
        "scene_config": build_scene_config_summary(),
        "scene_metadata": scene_metadata or {},
        "planner_name": planner_name,
        "selected_strategy": selected_strategy,
        "selected_target_object": selected_target_object,
        "selected_cube_id": selected_cube_id,
        "end_effector_target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "expected_cube_position": expected_rest_position,
        "object_level_subtasks": total_subtasks,
        "current_subtask": subtask_index,
        "spawned_objects": spawned_objects,
        "final_object_snapshots": final_object_snapshots,
        "stage_cube_translation": stage_cube_translation,
        "physical_cube_translation": physical_cube_translation,
        "physical_cube_orientation": physical_cube_orientation,
        "final_active_object_translation": final_active_object_translation,
        "final_active_object_orientation": physical_cube_orientation,
        "target_end_effector_orientation": _round_quaternion_list(preferred_goal_orientation),
        "target_end_effector_orientation_reason": preferred_goal_orientation_reason,
        "position_error": round(position_error, 4) if position_error is not None else None,
        "placement_tolerance": float(placement_tolerance),
        "placement_success": bool(placement_success),
        "id_consistency_passed": True,
        "id_consistency_mismatches": [],
        "action_sequence": DEFAULT_ACTION_SEQUENCE,
        "phase_debug_timeline": phase_debug_timeline,
        "final_state_summary": final_state_summary,
        "success": bool(controller.is_done()) and bool(placement_success),
        "num_frames": len(saved_frames),
        "capture_interval": CAPTURE_INTERVAL,
        "simulation_steps": simulation_step,
        "max_simulation_steps": MAX_SIMULATION_STEPS,
        "max_saved_frames": MAX_SAVED_FRAMES,
        "output_dir": str(subtask_output_path),
        "frames_dir": str(frames_path),
        "gif_path": str(gif_path),
    }
    manifest_path = subtask_output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    manifest_summary["manifest_path"] = str(manifest_path)
    debug_summary_path = subtask_output_path / "debug_summary.txt"
    write_debug_summary(debug_summary_path, manifest_summary)
    manifest_summary["debug_summary_path"] = str(debug_summary_path)
    return manifest_summary


def record_pick_place_rollout_sequence(
    output_dir: str | Path,
    planner_name: str,
    selected_strategy: str,
    subtasks: list[Subtask],
    placement_tolerance: float = 0.06,
    scene_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run multiple single-object real rollouts and combine them into one case GIF."""
    output_path = Path(output_dir).expanduser().resolve()
    frames_path = output_path / "frames"
    output_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    combined_frame_paths: list[Path] = []
    subtask_summaries: list[dict[str, object]] = []
    mutable_scene_metadata = clone_scene_metadata(scene_metadata)
    tracked_positions_by_id: dict[str, list[float]] = {}
    intro_frame = frames_path / "frame_0000.png"
    create_text_card(
        intro_frame,
        title="Multi-object Franka rollout",
        body_lines=build_intro_card_lines(
            planner_name=planner_name,
            selected_strategy=selected_strategy,
            subtasks=subtasks,
            scene_metadata=scene_metadata,
        ),
    )
    combined_frame_paths.append(intro_frame)

    for subtask_index, subtask in enumerate(subtasks, start=1):
        id_consistency_passed, id_mismatches = verify_id_consistency(
            mutable_scene_metadata,
            tracked_positions_by_id,
        )
        subtask_summary = run_single_subtask_subprocess(
            output_path=output_path,
            planner_name=planner_name,
            selected_strategy=selected_strategy,
            subtask=subtask,
            placement_tolerance=placement_tolerance,
            scene_metadata=mutable_scene_metadata,
            subtask_index=subtask_index,
            total_subtasks=len(subtasks),
        )
        subtask_summary["id_consistency_passed"] = id_consistency_passed
        subtask_summary["id_consistency_mismatches"] = id_mismatches
        subtask_summary["success"] = bool(subtask_summary["success"]) and bool(id_consistency_passed)
        Path(subtask_summary["manifest_path"]).write_text(
            json.dumps(subtask_summary, indent=2),
            encoding="utf-8",
        )
        write_debug_summary(Path(subtask_summary["debug_summary_path"]), subtask_summary)
        source_frames = sorted(
            Path(subtask_summary["frames_dir"]).glob("frame_*.png")
        )
        copied_paths = copy_frame_sequence(source_frames, frames_path, len(combined_frame_paths))
        footer_lines = build_frame_footer_lines(
            mutable_scene_metadata,
            str(subtask["object"]),
        )
        title = (
            f"Subtask {subtask_index}/{len(subtasks)}: "
            f"{subtask.get('cube_id', subtask['object'])} {subtask['object']}"
        )
        subtitle = (
            f"goal region: {subtask['goal_region']} | "
            f"success={subtask_summary['success']} | "
            f"placement={subtask_summary.get('placement_success')}"
        )
        for copied_path in copied_paths:
            annotate_frame(
                copied_path,
                title=title,
                subtitle=subtitle,
                footer_lines=footer_lines,
            )
        combined_frame_paths.extend(copied_paths)
        subtask_summaries.append(subtask_summary)
        if mutable_scene_metadata is not None:
            object_positions = mutable_scene_metadata.get("object_sim_positions")
            if isinstance(object_positions, dict):
                final_translation = subtask_summary.get("final_active_object_translation")
                if isinstance(final_translation, list) and len(final_translation) == 3:
                    object_positions[str(subtask["object"])] = final_translation
        final_translation = subtask_summary.get("final_active_object_translation")
        if isinstance(final_translation, list) and len(final_translation) == 3:
            tracked_positions_by_id[str(subtask.get("cube_id", subtask["object"]))] = final_translation
        if not subtask_summary["success"]:
            break

    if not combined_frame_paths:
        raise RuntimeError("No subtask frames were captured for the combined rollout.")

    gif_path = output_path / "rollout.gif"
    save_gif(combined_frame_paths, gif_path)

    final_state_summary = {
        "controller_done": all(summary["success"] for summary in subtask_summaries),
        "simulation_steps": sum(int(summary["simulation_steps"]) for summary in subtask_summaries),
        "captured_frames": len(combined_frame_paths),
        "completed_subtasks": len(subtask_summaries),
    }
    summary = {
        "scene_config": build_scene_config_summary(),
        "scene_metadata": mutable_scene_metadata or scene_metadata or {},
        "planner_name": planner_name,
        "selected_strategy": selected_strategy,
        "selected_target_object": ", ".join(str(subtask["object"]) for subtask in subtasks),
        "selected_cube_ids": [str(subtask.get("cube_id", subtask["object"])) for subtask in subtasks],
        "target_position": [subtask["target_position"] for subtask in subtasks],
        "object_level_subtasks": len(subtasks),
        "spawned_objects": subtask_summaries[-1]["spawned_objects"],
        "tracked_positions_by_id": tracked_positions_by_id,
        "action_sequence": [str(subtask["object"]) for subtask in subtasks],
        "final_state_summary": final_state_summary,
        "success": all(summary["success"] for summary in subtask_summaries),
        "completed_subtasks": len(subtask_summaries),
        "failed_subtask_index": next(
            (
                index
                for index, subtask_summary in enumerate(subtask_summaries, start=1)
                if not subtask_summary["success"]
            ),
            None,
        ),
        "num_frames": len(combined_frame_paths),
        "capture_interval": CAPTURE_INTERVAL,
        "simulation_steps": final_state_summary["simulation_steps"],
        "max_simulation_steps": MAX_SIMULATION_STEPS,
        "max_saved_frames": MAX_SAVED_FRAMES * len(subtasks),
        "output_dir": str(output_path),
        "frames_dir": str(frames_path),
        "gif_path": str(gif_path),
        "subtasks": subtask_summaries,
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["manifest_path"] = str(manifest_path)
    debug_summary_path = output_path / "debug_summary.txt"
    write_debug_summary(debug_summary_path, summary)
    summary["debug_summary_path"] = str(debug_summary_path)

    print("[record_franka_pick_place_animation] combined recording complete:")
    print(f"  success: {summary['success']}")
    print(f"  planner: {summary['planner_name']}")
    print(f"  strategy: {summary['selected_strategy']}")
    print(f"  subtasks: {summary['object_level_subtasks']}")
    print(f"  frames: {summary['num_frames']}")
    print(f"  frames dir: {summary['frames_dir']}")
    print(f"  gif: {summary['gif_path']}")
    print(f"  manifest: {summary['manifest_path']}")
    print(f"  debug summary: {summary['debug_summary_path']}")
    return summary


def record_pick_place_rollout(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    planner_name: str = DEFAULT_PLANNER_NAME,
    selected_strategy: str = DEFAULT_STRATEGY_NAME,
    selected_target_object: str = "cube",
    cube_initial_position: tuple[float, float, float] | None = None,
    target_position: tuple[float, float, float] | None = None,
    expected_cube_position: tuple[float, float, float] | None = None,
    placement_tolerance: float = 0.06,
    scene_metadata: dict[str, object] | None = None,
    subtasks: list[Subtask] | None = None,
) -> dict[str, object]:
    """Run the official Franka pick-and-place controller and record the rollout."""
    if subtasks:
        return record_pick_place_rollout_sequence(
            output_dir=output_dir,
            planner_name=planner_name,
            selected_strategy=selected_strategy,
            subtasks=subtasks,
            placement_tolerance=placement_tolerance,
            scene_metadata=scene_metadata,
        )

    output_path = Path(output_dir).expanduser().resolve()
    frames_path = output_path / "frames"
    output_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    controller = ShowcaseFrankaPickPlace()
    controller.setup_scene(
        cube_initial_position=_to_numpy_vector(cube_initial_position),
        target_position=_to_numpy_vector(target_position),
    )
    preferred_goal_orientation, preferred_goal_orientation_reason = _select_goal_orientation(
        scene_metadata,
        selected_target_object,
    )
    _install_goal_orientation(
        controller,
        preferred_goal_orientation,
        preferred_goal_orientation_reason,
    )

    dt = 0.01
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt))
    sim_utils.update_stage()
    camera = build_camera()

    sim.reset()
    controller.reset(cube_position=_to_numpy_vector(cube_initial_position))
    spawn_target_basket(_resolve_basket_display_position(scene_metadata, target_position))
    spawn_context_cubes(scene_metadata, selected_target_object)
    sim_utils.update_stage()

    spawned_objects = _capture_stage_object_snapshots()
    object_level_subtasks = 1
    print("[record_franka_pick_place_animation] reset debug snapshot:")
    print("  object-level subtasks:", object_level_subtasks)
    print("  selected target object:", selected_target_object)
    print(
        "  end-effector target position:",
        _round_vector(target_position or (0.0, 0.0, 0.0)),
    )
    print(
        "  target end-effector orientation:",
        _round_quaternion_list(preferred_goal_orientation),
        f"({preferred_goal_orientation_reason})",
    )
    print("  spawned objects after reset:")
    for line in _format_object_pose_lines(spawned_objects):
        print(f"    {line}")
    print(
        "[record_franka_pick_place_animation] starting low-level execution for:",
        selected_target_object,
    )

    for _ in range(5):
        sim.step()

    saved_frames: list[Path] = []
    simulation_step = 0
    phase_debug_timeline: list[dict[str, object]] = []
    previous_event_index = int(controller._event)
    phase_debug_timeline.append(
        {
            **_capture_phase_debug_snapshot(controller, simulation_step, target_position),
            "note": "post_reset",
        }
    )

    while not controller.is_done() and simulation_step < MAX_SIMULATION_STEPS:
        controller.forward()
        sim.step()
        current_event_index = int(controller._event)
        if current_event_index != previous_event_index:
            phase_debug_timeline.append(
                {
                    **_capture_phase_debug_snapshot(controller, simulation_step, target_position),
                    "note": (
                        f"event_transition:{EVENT_LABELS.get(previous_event_index, previous_event_index)}"
                        f"->{EVENT_LABELS.get(current_event_index, current_event_index)}"
                    ),
                }
            )
            previous_event_index = current_event_index
        if current_event_index in (1, 2, 3, 4, 5, 6) and simulation_step % 10 == 0:
            phase_debug_timeline.append(
                {
                    **_capture_phase_debug_snapshot(controller, simulation_step, target_position),
                    "note": "single_rollout_diagnostic_window",
                }
            )
        should_capture = (
            simulation_step % CAPTURE_INTERVAL == 0
            and len(saved_frames) < MAX_SAVED_FRAMES
        )
        if should_capture:
            camera.update(dt, force_recompute=True)
            rgb_frame = camera.data.output["rgb"]
            rgb_array = to_numpy_image(rgb_frame)
            frame_path = frames_path / f"frame_{len(saved_frames):04d}.png"
            save_png_frame(rgb_array, frame_path)
            saved_frames.append(frame_path)
        simulation_step += 1

    if not saved_frames:
        raise RuntimeError("No rollout frames were captured before the recording ended.")

    gif_path = output_path / "rollout.gif"
    save_gif(saved_frames, gif_path)

    final_state_summary = {
        "controller_done": controller.is_done(),
        "simulation_steps": simulation_step,
        "captured_frames": len(saved_frames),
    }
    final_object_snapshots = _capture_stage_object_snapshots()
    stage_cube_translation = _round_position_list(_extract_active_cube_translation(final_object_snapshots))
    physical_cube_translation = _extract_controller_cube_translation(controller)
    physical_cube_orientation = _extract_controller_cube_orientation(controller)
    expected_rest_position = _expected_cube_rest_position(
        cube_initial_position=cube_initial_position,
        target_position=target_position,
        explicit_expected_position=expected_cube_position,
    )
    final_active_object_translation = physical_cube_translation
    position_error = _euclidean_distance(final_active_object_translation, expected_rest_position)
    placement_success = position_error is not None and position_error <= placement_tolerance
    summary = {
        "scene_config": build_scene_config_summary(),
        "scene_metadata": scene_metadata or {},
        "planner_name": planner_name,
        "selected_strategy": selected_strategy,
        "selected_target_object": selected_target_object,
        "end_effector_target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "expected_cube_position": expected_rest_position,
        "object_level_subtasks": object_level_subtasks,
        "spawned_objects": spawned_objects,
        "action_sequence": DEFAULT_ACTION_SEQUENCE,
        "phase_debug_timeline": phase_debug_timeline,
        "final_object_snapshots": final_object_snapshots,
        "stage_cube_translation": stage_cube_translation,
        "physical_cube_translation": physical_cube_translation,
        "physical_cube_orientation": physical_cube_orientation,
        "final_active_object_translation": final_active_object_translation,
        "final_active_object_orientation": physical_cube_orientation,
        "target_end_effector_orientation": _round_quaternion_list(preferred_goal_orientation),
        "target_end_effector_orientation_reason": preferred_goal_orientation_reason,
        "position_error": round(position_error, 4) if position_error is not None else None,
        "placement_tolerance": float(placement_tolerance),
        "placement_success": bool(placement_success),
        "final_state_summary": final_state_summary,
        "success": bool(controller.is_done()) and bool(placement_success),
        "num_frames": len(saved_frames),
        "capture_interval": CAPTURE_INTERVAL,
        "simulation_steps": simulation_step,
        "max_simulation_steps": MAX_SIMULATION_STEPS,
        "max_saved_frames": MAX_SAVED_FRAMES,
        "output_dir": str(output_path),
        "frames_dir": str(frames_path),
        "gif_path": str(gif_path),
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["manifest_path"] = str(manifest_path)
    debug_summary_path = output_path / "debug_summary.txt"
    write_debug_summary(debug_summary_path, summary)
    summary["debug_summary_path"] = str(debug_summary_path)

    print("[record_franka_pick_place_animation] recording complete:")
    print(f"  success: {summary['success']}")
    print(f"  planner: {summary['planner_name']}")
    print(f"  strategy: {summary['selected_strategy']}")
    print(f"  frames: {summary['num_frames']}")
    print(f"  frames dir: {summary['frames_dir']}")
    print(f"  gif: {summary['gif_path']}")
    print(f"  manifest: {summary['manifest_path']}")
    print(f"  debug summary: {summary['debug_summary_path']}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a tiny CLI for direct rollout and isolated subtask recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("default", "single-subtask"),
        default="default",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--planner-name", default=DEFAULT_PLANNER_NAME)
    parser.add_argument("--selected-strategy", default=DEFAULT_STRATEGY_NAME)
    parser.add_argument("--selected-target-object", default="cube")
    parser.add_argument("--cube-initial-position")
    parser.add_argument("--target-position")
    parser.add_argument("--expected-cube-position")
    parser.add_argument("--placement-tolerance", type=float, default=0.06)
    parser.add_argument("--scene-metadata-json")
    parser.add_argument("--subtask-index", type=int, default=1)
    parser.add_argument("--total-subtasks", type=int, default=1)
    return parser


def main() -> None:
    """Run the recording entrypoint."""
    args = build_arg_parser().parse_args()
    try:
        if args.mode == "single-subtask":
            _record_single_subtask_rollout(
                output_path=Path(args.output_dir),
                planner_name=args.planner_name,
                selected_strategy=args.selected_strategy,
                selected_target_object=args.selected_target_object,
                cube_initial_position=parse_optional_vector(args.cube_initial_position),
                target_position=parse_optional_vector(args.target_position),
                expected_cube_position=parse_optional_vector(args.expected_cube_position),
                placement_tolerance=args.placement_tolerance,
                scene_metadata=parse_optional_json_dict(args.scene_metadata_json),
                subtask_index=args.subtask_index,
                total_subtasks=args.total_subtasks,
            )
        else:
            record_pick_place_rollout(
                output_dir=args.output_dir,
                planner_name=args.planner_name,
                selected_strategy=args.selected_strategy,
                selected_target_object=args.selected_target_object,
                cube_initial_position=parse_optional_vector(args.cube_initial_position),
                target_position=parse_optional_vector(args.target_position),
                expected_cube_position=parse_optional_vector(args.expected_cube_position),
                placement_tolerance=args.placement_tolerance,
                scene_metadata=parse_optional_json_dict(args.scene_metadata_json),
            )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
