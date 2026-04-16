"""Minimal Isaac Lab backend for loading an environment and reading state."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from world_state import WorldState


class IsaacLabBackend:
    """Read-only Isaac Lab backend that projects simulator state into WorldState."""

    DEFAULT_ENV_ID = "Isaac-Lift-Cube-Franka-v0"

    def __init__(
        self,
        env: Any | None = None,
        env_id: str = DEFAULT_ENV_ID,
        headless: bool = True,
        render_mode: str | None = None,
        grid_origin: tuple[int, int] = (20, 20),
        grid_scale: int = 20,
        debug: bool = True,
    ) -> None:
        """Create the backend and initialize the Isaac Lab environment."""
        self.env_id = env_id
        self.headless = headless
        self.render_mode = render_mode
        self.grid_origin = grid_origin
        self.grid_scale = grid_scale
        self.debug = debug
        self._latest_obs: Any = None
        self._latest_info: dict[str, Any] = {}
        self._simulation_app: Any | None = None
        self._debug_camera: Any | None = None
        if env is None:
            self._start_simulation_app()
        self._env = env if env is not None else self._create_env()

    def reset(self) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and cache the latest observation."""
        reset_output = self._env.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            self._latest_obs, info = reset_output
            self._latest_info = info if isinstance(info, dict) else {}
        else:
            self._latest_obs = reset_output
            self._latest_info = {}

        if self.debug:
            print(f"[IsaacLabBackend] reset environment: {self.env_id}")
            state = self.get_current_state()
            print(
                "[IsaacLabBackend] spawned objects after reset:",
                ", ".join(state.objects.keys()) or "(none)",
            )
            for object_name, object_data in state.objects.items():
                print(
                    f"[IsaacLabBackend] object pose after reset: "
                    f"{object_name} -> {object_data['pos']}"
                )
        return self._latest_obs, self._latest_info

    def get_current_state(self) -> WorldState:
        """Read the current simulator state and convert it to WorldState."""
        if self._latest_obs is None:
            self.reset()

        robot_pose = self._extract_robot_pose()
        cube_positions = self._extract_cube_positions()
        goal_position = self._extract_goal_position()

        state = WorldState(
            robot_position=self._project_to_grid(robot_pose),
            goal_regions={"lift_goal": [self._project_to_grid(goal_position)]},
            obstacles=[],
            forbidden_zones=[],
        )

        if not cube_positions:
            cube_positions = [(0.0, 0.0, 0.0)]

        for index, cube_position in enumerate(cube_positions, start=1):
            cube_name = "cube" if index == 1 else f"cube_{index}"
            state.update_object(
                cube_name,
                position=self._project_to_grid(cube_position),
                object_type="cube",
                graspable=True,
            )

        if self.debug:
            print("[IsaacLabBackend] extracted state:")
            print(f"  robot position: {state.get_robot_position()}")
            for name, obj in state.objects.items():
                print(f"  object {name}: pos={obj['pos']}, type={obj['type']}")
            print(f"  goal regions: {state.goal_regions}")

        return state

    def save_debug_frame(self, output_dir: str = "outputs/isaac_camera") -> str:
        """Capture one RGB debug frame and save it to disk."""
        if self._latest_obs is None:
            self.reset()

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        camera = self._get_or_create_debug_camera()
        sim = self._get_simulation_context()
        dt = self._get_simulation_dt(sim)
        self._advance_debug_rollout(sim, num_steps=5)

        camera.update(dt, force_recompute=True)
        rgb_frame = self._extract_debug_rgb_frame(camera)
        saved_path = self._save_debug_rgb_frame(
            rgb_frame,
            output_path,
            file_stem="debug_frame",
        )
        print(f"[IsaacLabBackend] saved debug frame: {saved_path}")
        return saved_path

    def save_debug_animation(
        self,
        output_dir: str = "outputs/isaac_animation",
        num_steps: int = 30,
    ) -> dict[str, object]:
        """Capture a short RGB rollout as a numbered frame sequence."""
        if self._latest_obs is None:
            self.reset()

        output_path = Path(output_dir).expanduser().resolve()
        frames_path = output_path / "frames"
        frames_path.mkdir(parents=True, exist_ok=True)

        camera = self._get_or_create_debug_camera()
        sim = self._get_simulation_context()
        dt = self._get_simulation_dt(sim)

        saved_paths: list[str] = []
        frame_extension = ""

        for step_index in range(max(1, num_steps)):
            self._advance_debug_rollout(sim, num_steps=1)
            camera.update(dt, force_recompute=True)
            rgb_frame = self._extract_debug_rgb_frame(camera)
            saved_path = self._save_debug_rgb_frame(
                rgb_frame,
                frames_path,
                file_stem=f"frame_{step_index:04d}",
            )
            saved_paths.append(saved_path)
            if not frame_extension:
                frame_extension = Path(saved_path).suffix

        summary = {
            "success": True,
            "num_frames": len(saved_paths),
            "output_dir": str(output_path),
            "frames_dir": str(frames_path),
            "frame_extension": frame_extension,
            "frames": saved_paths,
        }
        manifest_path = output_path / "manifest.json"
        manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["manifest_path"] = str(manifest_path)

        print("[IsaacLabBackend] saved debug animation:")
        print(f"  success: {summary['success']}")
        print(f"  frames: {summary['num_frames']}")
        print(f"  frames dir: {summary['frames_dir']}")
        print(f"  manifest: {summary['manifest_path']}")
        return summary

    def _create_env(self) -> Any:
        """Create the Isaac Lab environment with lightweight compatibility fallbacks."""
        try:
            gym = importlib.import_module("gymnasium")
        except ImportError as exc:
            raise ImportError(
                "IsaacLabBackend requires gymnasium plus an Isaac Lab installation."
            ) from exc

        task_modules = ("isaaclab_tasks", "omni.isaac.lab_tasks")
        for module_name in task_modules:
            try:
                importlib.import_module(module_name)
                break
            except ImportError:
                continue
        else:
            raise ImportError(
                "Isaac Lab task registrations were not found. Expected "
                "'isaaclab_tasks' or 'omni.isaac.lab_tasks'."
            )

        env_cfg = self._load_env_cfg(gym)
        kwargs = {"cfg": env_cfg}
        if self.render_mode is not None:
            kwargs["render_mode"] = self.render_mode

        try:
            env = gym.make(self.env_id, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to create Isaac Lab environment '{self.env_id}'."
            ) from exc

        if self.debug:
            print(
                f"[IsaacLabBackend] created environment {self.env_id} "
                f"with args {kwargs}"
            )
        return env

    def _get_or_create_debug_camera(self) -> Any:
        """Create a small overhead debug camera the first time it is needed."""
        if self._debug_camera is not None:
            return self._debug_camera

        sim_utils = importlib.import_module("isaaclab.sim")
        camera_module = importlib.import_module("isaaclab.sensors.camera")
        camera_cfg_cls = camera_module.CameraCfg
        camera_cls = camera_module.Camera

        camera_cfg = camera_cfg_cls(
            prim_path="/World/DebugCamera",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=2.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 20.0),
            ),
            offset=camera_cfg_cls.OffsetCfg(
                pos=(0.0, 0.0, 1.8),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )

        self._debug_camera = camera_cls(camera_cfg)
        self._initialize_debug_camera()

        if self.debug:
            print("[IsaacLabBackend] initialized debug camera at /World/DebugCamera")
        return self._debug_camera

    def _initialize_debug_camera(self) -> None:
        """Force camera initialization when it is created after the simulation starts."""
        if self._debug_camera is None:
            return
        if getattr(self._debug_camera, "_is_initialized", False):
            return

        self._debug_camera._initialize_impl()
        self._debug_camera._is_initialized = True
        self._debug_camera.reset()

    def _extract_debug_rgb_frame(self, camera: Any) -> Any:
        """Read a single RGB frame from the debug camera output."""
        output = getattr(camera.data, "output", {})
        rgb_frame = output.get("rgb")
        if rgb_frame is None:
            raise RuntimeError("IsaacLabBackend debug camera did not produce an RGB frame.")
        return rgb_frame

    def _save_debug_rgb_frame(
        self,
        rgb_frame: Any,
        output_path: Path,
        file_stem: str,
    ) -> str:
        """Save an RGB frame to disk using a lightweight image write path."""
        numpy = importlib.import_module("numpy")

        rgb_array = numpy.asarray(self._to_python_value(rgb_frame))
        if rgb_array.ndim == 4:
            rgb_array = rgb_array[0]
        if rgb_array.ndim != 3 or rgb_array.shape[-1] < 3:
            raise RuntimeError(
                "IsaacLabBackend received an unexpected RGB frame shape: "
                f"{tuple(rgb_array.shape)}"
            )

        rgb_array = rgb_array[..., :3].astype(numpy.uint8)

        png_path = output_path / f"{file_stem}.png"
        try:
            image_module = importlib.import_module("PIL.Image")
            image = image_module.fromarray(rgb_array, mode="RGB")
            image.save(png_path)
            return str(png_path)
        except ImportError:
            ppm_path = output_path / f"{file_stem}.ppm"
            height, width = rgb_array.shape[:2]
            with ppm_path.open("wb") as ppm_file:
                ppm_file.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                ppm_file.write(rgb_array.tobytes())
            return str(ppm_path)

    def _advance_debug_rollout(self, sim: Any, num_steps: int) -> None:
        """Advance the simulator a small number of steps for debug capture."""
        for _ in range(max(1, num_steps)):
            if hasattr(sim, "step"):
                sim.step()
            else:
                sim.render()

    def _start_simulation_app(self) -> None:
        """Start Isaac Sim's runtime so pxr/omni modules are available."""
        try:
            app_launcher_cls = importlib.import_module("isaaclab.app").AppLauncher
        except Exception as exc:
            raise ImportError(
                "IsaacLabBackend requires isaaclab.app.AppLauncher support."
            ) from exc

        if self.debug:
            print("[IsaacLabBackend] starting Isaac Lab AppLauncher")
        self._simulation_app = app_launcher_cls(
            headless=self.headless,
            enable_cameras=True,
        )
        self._simulation_app = self._simulation_app.app
        self._ensure_asset_root()

    def _ensure_asset_root(self) -> None:
        """Fill in the Isaac asset root if the current runtime left it unset."""
        carb = importlib.import_module("carb")
        settings = carb.settings.get_settings()
        asset_root_cloud = settings.get("/persistent/isaac/asset_root/cloud")
        asset_root_default = settings.get("/persistent/isaac/asset_root/default")

        if asset_root_cloud is None and asset_root_default:
            settings.set("/persistent/isaac/asset_root/cloud", asset_root_default)
            if self.debug:
                print(
                    "[IsaacLabBackend] set asset root cloud to default:",
                    asset_root_default,
                )

    def _load_env_cfg(self, gym: Any) -> Any:
        """Load the Isaac Lab environment config class from the gym spec."""
        env_spec = gym.spec(self.env_id)
        cfg_entry = env_spec.kwargs.get("env_cfg_entry_point")
        if not isinstance(cfg_entry, str) or ":" not in cfg_entry:
            raise RuntimeError(
                f"Gym spec for '{self.env_id}' does not expose env_cfg_entry_point."
            )

        module_name, attr_name = cfg_entry.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        cfg_cls = getattr(module, attr_name)
        env_cfg = cfg_cls()

        scene = getattr(env_cfg, "scene", None)
        if scene is not None and hasattr(scene, "num_envs"):
            scene.num_envs = 1

        if self.debug:
            print(f"[IsaacLabBackend] loaded env cfg: {cfg_entry}")
        return env_cfg

    def _get_simulation_context(self) -> Any:
        """Return the Isaac simulation context from the environment."""
        env = getattr(self._env, "unwrapped", self._env)
        sim = getattr(env, "sim", None)
        if sim is None:
            raise RuntimeError(
                "IsaacLabBackend could not access the simulator context needed for "
                "camera capture."
            )
        return sim

    def _get_simulation_dt(self, sim: Any) -> float:
        """Return a small timestep value for camera sensor updates."""
        if hasattr(sim, "get_physics_dt"):
            dt = sim.get_physics_dt()
            if isinstance(dt, (int, float)) and dt > 0:
                return float(dt)

        cfg = getattr(sim, "cfg", None)
        dt = getattr(cfg, "dt", None)
        if isinstance(dt, (int, float)) and dt > 0:
            return float(dt)

        return 1.0 / 60.0

    def _flatten_payload(
        self,
        payload: Any,
        prefix: str = "",
    ) -> dict[str, Any]:
        """Flatten nested observation dictionaries for simple key matching."""
        flattened: dict[str, Any] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                flattened.update(self._flatten_payload(value, child_prefix))
            return flattened

        if prefix:
            flattened[prefix] = payload
        return flattened

    def _find_payload_value(self, key_fragments: list[str]) -> Any | None:
        """Return the first observation/info entry whose key matches all fragments."""
        for source in (self._latest_obs, self._latest_info):
            flattened = self._flatten_payload(source)
            for key, value in flattened.items():
                normalized_key = key.lower()
                if all(fragment in normalized_key for fragment in key_fragments):
                    return value
        return None

    def _to_python_value(self, value: Any) -> Any:
        """Convert tensors and arrays into plain Python values when possible."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def _select_first_env(self, value: Any) -> Any:
        """Select the first environment if Isaac observations are batched."""
        value = self._to_python_value(value)
        if isinstance(value, list) and value:
            first_item = value[0]
            if isinstance(first_item, list):
                return first_item
        return value

    def _as_points(self, value: Any) -> list[tuple[float, ...]]:
        """Normalize an observation payload into a list of points."""
        selected = self._select_first_env(value)
        if not isinstance(selected, list):
            return []
        if selected and all(isinstance(item, (int, float)) for item in selected):
            return [tuple(float(item) for item in selected)]

        points: list[tuple[float, ...]] = []
        for item in selected:
            if isinstance(item, list) and item:
                points.append(tuple(float(component) for component in item))
        return points

    def _extract_robot_pose(self) -> tuple[float, ...]:
        """Extract an end-effector-like robot pose if available."""
        for fragments in (
            ["eef", "pos"],
            ["ee", "pos"],
            ["tcp", "pos"],
            ["hand", "pos"],
            ["robot", "pos"],
        ):
            value = self._find_payload_value(fragments)
            points = self._as_points(value)
            if points:
                return points[0]
        return (0.0, 0.0, 0.0)

    def _extract_cube_positions(self) -> list[tuple[float, ...]]:
        """Extract cube/object positions from the latest observation."""
        for fragments in (
            ["cube", "pos"],
            ["object", "pos"],
            ["cube", "position"],
            ["object", "position"],
        ):
            value = self._find_payload_value(fragments)
            points = self._as_points(value)
            if points:
                return points
        return []

    def _extract_goal_position(self) -> tuple[float, ...]:
        """Extract a target/goal position from the latest observation or info."""
        for fragments in (
            ["goal", "pos"],
            ["target", "pos"],
            ["command", "pos"],
            ["desired", "pos"],
        ):
            value = self._find_payload_value(fragments)
            points = self._as_points(value)
            if points:
                return points[0]
        return (0.1, 0.1, 0.0)

    def _project_to_grid(self, point: tuple[float, ...]) -> tuple[int, int]:
        """Project an Isaac-space point into the current 2D planner grid."""
        x_value = point[0] if len(point) >= 1 else 0.0
        y_value = point[1] if len(point) >= 2 else 0.0
        grid_x = self.grid_origin[0] + int(round(x_value * self.grid_scale))
        grid_y = self.grid_origin[1] + int(round(y_value * self.grid_scale))
        return (
            max(0, min(39, grid_x)),
            max(0, min(39, grid_y)),
        )


if __name__ == "__main__":
    class _FakeLiftCubeEnv:
        def reset(self) -> tuple[dict[str, object], dict[str, object]]:
            obs = {
                "policy": {
                    "eef_pos": [[0.05, -0.05, 0.25]],
                    "cube_pos": [[0.12, 0.08, 0.03]],
                    "goal_pos": [[0.25, 0.15, 0.20]],
                }
            }
            return obs, {"debug_source": "fake_env"}

    backend = IsaacLabBackend(env=_FakeLiftCubeEnv())
    backend.reset()
    state = backend.get_current_state()
    print("[IsaacLabBackend] demo state loaded successfully.")
    print("robot:", state.get_robot_position())
    print("objects:", state.objects)
    print("goals:", state.goal_regions)
