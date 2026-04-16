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


def _capture_stage_object_snapshots() -> list[dict[str, object]]:
    """Collect a small readable set of spawned object names and poses after reset."""
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return []

    object_snapshots: list[dict[str, object]] = []
    interesting_keywords = ("franka", "cube", "table", "ground", "target")

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


def write_debug_summary(summary_path: Path, summary_payload: dict[str, object]) -> None:
    """Persist a lightweight text summary next to the rollout outputs."""
    lines = [
        "Real Franka Rollout Debug Summary",
        f"planner: {summary_payload['planner_name']}",
        f"strategy: {summary_payload['selected_strategy']}",
        f"selected target object: {summary_payload['selected_target_object']}",
        f"target position: {summary_payload['target_position']}",
        f"object-level subtasks: {summary_payload['object_level_subtasks']}",
        f"success: {summary_payload['success']}",
        f"simulation steps: {summary_payload['simulation_steps']}",
        f"captured frames: {summary_payload['num_frames']}",
        f"gif: {summary_payload['gif_path']}",
        "spawned objects after reset:",
        *_format_object_pose_lines(summary_payload["spawned_objects"]),
    ]
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
        "--scene-metadata-json",
        json.dumps(scene_metadata or {}),
        "--subtask-index",
        str(subtask_index),
        "--total-subtasks",
        str(total_subtasks),
    ]
    try:
        subprocess.run(command, check=True, timeout=SUBTASK_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        if not manifest_path.exists():
            raise
        print(
            "[record_franka_pick_place_animation] subtask timed out after writing outputs:",
            f"{subtask_index}/{total_subtasks} {subtask['object']}",
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


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
    """Spawn simple visual-only cubes for the other scenario objects."""
    if not scene_metadata:
        return

    object_sim_positions = scene_metadata.get("object_sim_positions")
    if not isinstance(object_sim_positions, dict):
        return

    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    context_root = UsdGeom.Xform.Define(stage, "/World/ContextObjects")
    color_cycle = [
        Gf.Vec3f(0.85, 0.25, 0.25),
        Gf.Vec3f(0.25, 0.55, 0.90),
        Gf.Vec3f(0.25, 0.75, 0.40),
        Gf.Vec3f(0.90, 0.70, 0.20),
        Gf.Vec3f(0.70, 0.35, 0.85),
    ]

    visible_index = 0
    for object_name, raw_position in object_sim_positions.items():
        if object_name == active_object_name:
            continue
        if not isinstance(raw_position, (list, tuple)) or len(raw_position) != 3:
            continue

        cube_path = f"{context_root.GetPath()}/{object_name}"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([color_cycle[visible_index % len(color_cycle)]])
        visible_index += 1

        xform_api = UsdGeom.XformCommonAPI(cube)
        xform_api.SetTranslate(
            Gf.Vec3d(float(raw_position[0]), float(raw_position[1]), float(raw_position[2]))
        )
        xform_api.SetScale(Gf.Vec3f(0.0516, 0.0516, 0.0516))


def _record_single_subtask_rollout(
    output_path: Path,
    planner_name: str,
    selected_strategy: str,
    selected_target_object: str,
    cube_initial_position: tuple[float, float, float] | None,
    target_position: tuple[float, float, float] | None,
    scene_metadata: dict[str, object] | None,
    subtask_index: int,
    total_subtasks: int,
) -> dict[str, object]:
    """Run one single-object rollout for a subtask and save it into a subdirectory."""
    subtask_output_path = output_path / f"subtask_{subtask_index:02d}_{selected_target_object}"
    frames_path = subtask_output_path / "frames"
    subtask_output_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    controller = FrankaPickPlace()
    controller.setup_scene(
        cube_initial_position=_to_numpy_vector(cube_initial_position),
        target_position=_to_numpy_vector(target_position),
    )

    dt = 0.01
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt))
    sim_utils.update_stage()
    camera = build_camera()

    sim.reset()
    controller.reset(cube_position=_to_numpy_vector(cube_initial_position))
    spawn_context_cubes(scene_metadata, selected_target_object)
    sim_utils.update_stage()

    spawned_objects = _capture_stage_object_snapshots()
    print("[record_franka_pick_place_animation] reset debug snapshot:")
    print("  object-level subtasks:", total_subtasks)
    print("  current subtask:", f"{subtask_index}/{total_subtasks}")
    print("  selected target object:", selected_target_object)
    print("  selected target position:", _round_vector(target_position or (0.0, 0.0, 0.0)))
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

    while not controller.is_done() and simulation_step < MAX_SIMULATION_STEPS:
        controller.forward()
        sim.step()
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
    manifest_summary = {
        "scene_config": build_scene_config_summary(),
        "scene_metadata": scene_metadata or {},
        "planner_name": planner_name,
        "selected_strategy": selected_strategy,
        "selected_target_object": selected_target_object,
        "target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "object_level_subtasks": total_subtasks,
        "current_subtask": subtask_index,
        "spawned_objects": spawned_objects,
        "action_sequence": DEFAULT_ACTION_SEQUENCE,
        "final_state_summary": final_state_summary,
        "success": controller.is_done(),
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
    scene_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run multiple single-object real rollouts and combine them into one case GIF."""
    output_path = Path(output_dir).expanduser().resolve()
    frames_path = output_path / "frames"
    output_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    combined_frame_paths: list[Path] = []
    subtask_summaries: list[dict[str, object]] = []

    for subtask_index, subtask in enumerate(subtasks, start=1):
        subtask_summary = run_single_subtask_subprocess(
            output_path=output_path,
            planner_name=planner_name,
            selected_strategy=selected_strategy,
            subtask=subtask,
            scene_metadata=scene_metadata,
            subtask_index=subtask_index,
            total_subtasks=len(subtasks),
        )
        source_frames = sorted(
            Path(subtask_summary["frames_dir"]).glob("frame_*.png")
        )
        combined_frame_paths.extend(
            copy_frame_sequence(source_frames, frames_path, len(combined_frame_paths))
        )
        subtask_summaries.append(subtask_summary)

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
        "scene_metadata": scene_metadata or {},
        "planner_name": planner_name,
        "selected_strategy": selected_strategy,
        "selected_target_object": ", ".join(str(subtask["object"]) for subtask in subtasks),
        "target_position": [subtask["target_position"] for subtask in subtasks],
        "object_level_subtasks": len(subtasks),
        "spawned_objects": subtask_summaries[-1]["spawned_objects"],
        "action_sequence": [str(subtask["object"]) for subtask in subtasks],
        "final_state_summary": final_state_summary,
        "success": all(summary["success"] for summary in subtask_summaries),
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
            scene_metadata=scene_metadata,
        )

    output_path = Path(output_dir).expanduser().resolve()
    frames_path = output_path / "frames"
    output_path.mkdir(parents=True, exist_ok=True)
    frames_path.mkdir(parents=True, exist_ok=True)

    controller = FrankaPickPlace()
    controller.setup_scene(
        cube_initial_position=_to_numpy_vector(cube_initial_position),
        target_position=_to_numpy_vector(target_position),
    )

    dt = 0.01
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt))
    sim_utils.update_stage()
    camera = build_camera()

    sim.reset()
    controller.reset(cube_position=_to_numpy_vector(cube_initial_position))

    spawned_objects = _capture_stage_object_snapshots()
    object_level_subtasks = 1
    print("[record_franka_pick_place_animation] reset debug snapshot:")
    print("  object-level subtasks:", object_level_subtasks)
    print("  selected target object:", selected_target_object)
    print("  selected target position:", _round_vector(target_position or (0.0, 0.0, 0.0)))
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

    while not controller.is_done() and simulation_step < MAX_SIMULATION_STEPS:
        controller.forward()
        sim.step()
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
    summary = {
        "scene_config": build_scene_config_summary(),
        "scene_metadata": scene_metadata or {},
        "planner_name": planner_name,
        "selected_strategy": selected_strategy,
        "selected_target_object": selected_target_object,
        "target_position": _round_vector(target_position or (0.0, 0.0, 0.0)),
        "object_level_subtasks": object_level_subtasks,
        "spawned_objects": spawned_objects,
        "action_sequence": DEFAULT_ACTION_SEQUENCE,
        "final_state_summary": final_state_summary,
        "success": controller.is_done(),
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
                scene_metadata=parse_optional_json_dict(args.scene_metadata_json),
            )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
