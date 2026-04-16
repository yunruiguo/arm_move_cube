"""Backend interfaces and simulator adapters for planning experiments."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Protocol

from reachability_engine import query_reachability
from world_state import WorldState

Action = dict[str, object]
COA = dict[str, object]
SimulationResult = dict[str, object]
ControllerHook = Callable[[Any, Action], dict[str, object]]


class PlanningBackend(Protocol):
    """Minimal backend interface for simulator-agnostic planning."""

    def get_current_state(self) -> WorldState:
        """Return the current observable state."""

    def apply_action(self, action: Action) -> dict[str, object]:
        """Evaluate a single structured action."""

    def simulate_action_sequence(self, coa: COA) -> SimulationResult:
        """Evaluate a whole COA action sequence."""


class ToyWorldBackend:
    """Backend adapter for the current in-memory toy benchmark world."""

    def __init__(self, initial_state: WorldState) -> None:
        """Store the deterministic toy state."""
        self._state = initial_state

    def get_current_state(self) -> WorldState:
        """Return the current toy world state."""
        return self._state

    def apply_action(self, action: Action) -> dict[str, object]:
        """Evaluate a pick/place action without mutating the toy world."""
        object_name = str(action["object"])
        place_position = action["place_position"]
        assert isinstance(place_position, tuple)

        pick_result = query_reachability(self._state, "pick", object_name)
        place_result = query_reachability(
            self._state,
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

        return {
            "object": object_name,
            "place_position": place_position,
            "pick_result": pick_result,
            "place_result": place_result,
            "success": step_success,
            "score": step_score,
        }

    def simulate_action_sequence(self, coa: COA) -> SimulationResult:
        """Evaluate a whole COA as a sequence of independent toy actions."""
        total_score = 0
        failed_actions = 0
        completed_actions = 0
        failure_reasons: list[str] = []
        step_results: list[dict[str, object]] = []

        for action in coa["actions"]:
            step_result = self.apply_action(action)
            step_results.append(step_result)
            total_score += int(step_result["score"])

            if step_result["success"]:
                completed_actions += 1
                continue

            failed_actions += 1
            failure_reasons.append(
                f"{step_result['object']}: "
                f"{step_result['pick_result']['reason']} | "
                f"{step_result['place_result']['reason']}"
            )
            break

        success = failed_actions == 0 and completed_actions == len(coa["actions"])
        return {
            "selected_strategy": coa["name"],
            "family": coa["family"],
            "total_score": total_score,
            "success": success,
            "failed_actions": failed_actions,
            "completed_actions": completed_actions,
            "failure_reasons": "; ".join(failure_reasons) if failure_reasons else "",
            "step_results": step_results,
        }


class IsaacLabBackend:
    """Backend adapter for the Isaac Lab lift-cube manipulation environment."""

    ENV_ID = "Isaac-Lift-Cube-Franka-v0"

    def __init__(
        self,
        env: Any | None = None,
        env_id: str = ENV_ID,
        headless: bool = True,
        render_mode: str | None = None,
        pick_controller: ControllerHook | None = None,
        place_controller: ControllerHook | None = None,
        grid_origin: tuple[int, int] = (20, 20),
        grid_scale: int = 20,
    ) -> None:
        """Create an Isaac Lab backend or wrap an already-created environment.

        The backend keeps the planning side simulator-agnostic by projecting the
        current Isaac observation into the existing 2D WorldState structure.
        Controller hooks can be injected to bridge pick/place actions into Isaac.
        """
        self._env_id = env_id
        self._headless = headless
        self._render_mode = render_mode
        self._pick_controller = pick_controller
        self._place_controller = place_controller
        self._grid_origin = grid_origin
        self._grid_scale = grid_scale
        self._env = env if env is not None else self._create_default_env()
        self._latest_obs: Any = None
        self._latest_info: dict[str, Any] = {}
        self.reset()

    def _create_default_env(self) -> Any:
        """Create the Isaac Lab gym environment with a few compatibility fallbacks."""
        try:
            gym = importlib.import_module("gymnasium")
        except ImportError as exc:
            raise ImportError(
                "IsaacLabBackend requires gymnasium. Install Isaac Lab and its gymnasium "
                "bridge, then try again."
            ) from exc

        registration_loaded = False
        for module_name in ("isaaclab_tasks", "omni.isaac.lab_tasks"):
            try:
                importlib.import_module(module_name)
                registration_loaded = True
                break
            except ImportError:
                continue

        if not registration_loaded:
            raise ImportError(
                "Isaac Lab task registrations are unavailable. Expected one of "
                "'isaaclab_tasks' or 'omni.isaac.lab_tasks' to be importable."
            )

        creation_attempts = [
            {},
            {"render_mode": self._render_mode},
            {"headless": self._headless},
            {"headless": self._headless, "render_mode": self._render_mode},
            {"num_envs": 1},
            {"num_envs": 1, "headless": self._headless},
            {"num_envs": 1, "headless": self._headless, "render_mode": self._render_mode},
        ]

        last_error: Exception | None = None
        for attempt_kwargs in creation_attempts:
            kwargs = {
                key: value
                for key, value in attempt_kwargs.items()
                if value is not None
            }
            try:
                return gym.make(self._env_id, **kwargs)
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to create Isaac Lab environment '{self._env_id}'."
        ) from last_error

    def reset(self) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and cache the latest observation payload."""
        reset_output = self._env.reset()
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            self._latest_obs, self._latest_info = reset_output
        else:
            self._latest_obs = reset_output
            self._latest_info = {}
        return self._latest_obs, self._latest_info

    def close(self) -> None:
        """Close the Isaac environment if the wrapped object supports it."""
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()

    def get_current_state(self) -> WorldState:
        """Project the current Isaac observation into the minimal 2D world model."""
        robot_point = self._extract_robot_pose()
        cube_points = self._extract_cube_positions()
        goal_point = self._extract_goal_position()

        state = WorldState(
            robot_position=self._project_to_grid(robot_point),
            obstacles=[],
            forbidden_zones=[],
            goal_regions={"lift_goal": [self._project_to_grid(goal_point)]},
        )

        if not cube_points:
            cube_points = [goal_point]

        for index, cube_point in enumerate(cube_points, start=1):
            cube_name = "cube" if index == 1 else f"cube_{index}"
            state.update_object(
                cube_name,
                position=self._project_to_grid(cube_point),
                object_type="cube",
                graspable=True,
            )

        return state

    def apply_action(self, action: Action) -> dict[str, object]:
        """Execute a structured pick/place action through Isaac controller hooks."""
        object_name = str(action["object"])
        place_position = action["place_position"]
        assert isinstance(place_position, tuple)

        pick_result = self._run_phase_controller("pick", action)
        place_result = self._run_phase_controller("place", action)

        step_success = (
            bool(pick_result["reachable"])
            and bool(pick_result["action_feasible"])
            and bool(place_result["reachable"])
            and bool(place_result["action_feasible"])
        )
        step_score = int(pick_result["score"]) + int(place_result["score"])
        step_score += 10 if step_success else -5

        return {
            "object": object_name,
            "place_position": place_position,
            "pick_result": pick_result,
            "place_result": place_result,
            "success": step_success,
            "score": step_score,
        }

    def simulate_action_sequence(self, coa: COA) -> SimulationResult:
        """Replay a COA inside Isaac Lab from a clean reset."""
        self.reset()

        total_score = 0
        failed_actions = 0
        completed_actions = 0
        failure_reasons: list[str] = []
        step_results: list[dict[str, object]] = []

        for action in coa["actions"]:
            step_result = self.apply_action(action)
            step_results.append(step_result)
            total_score += int(step_result["score"])

            if step_result["success"]:
                completed_actions += 1
                continue

            failed_actions += 1
            failure_reasons.append(
                f"{step_result['object']}: "
                f"{step_result['pick_result']['reason']} | "
                f"{step_result['place_result']['reason']}"
            )
            break

        success = failed_actions == 0 and completed_actions == len(coa["actions"])
        return {
            "selected_strategy": coa["name"],
            "family": coa["family"],
            "total_score": total_score,
            "success": success,
            "failed_actions": failed_actions,
            "completed_actions": completed_actions,
            "failure_reasons": "; ".join(failure_reasons) if failure_reasons else "",
            "step_results": step_results,
        }

    def _run_phase_controller(
        self,
        phase: str,
        action: Action,
    ) -> dict[str, object]:
        """Run a pick/place bridge and normalize the result for the planner."""
        controller = self._resolve_controller(phase)
        controller_result = controller(self._env, action)

        if "observation" in controller_result:
            self._latest_obs = controller_result["observation"]
        if "info" in controller_result:
            info_value = controller_result["info"]
            if isinstance(info_value, dict):
                self._latest_info = info_value

        success = bool(controller_result.get("success", False))
        path_cost = int(controller_result.get("path_cost", controller_result.get("steps", 0)))
        return {
            "reachable": True,
            "action_feasible": success,
            "path": controller_result.get("path", []),
            "path_cost": path_cost,
            "score": int(controller_result.get("score", 12 if success else -10)),
            "reason": str(
                controller_result.get(
                    "reason",
                    f"{phase} {'succeeded' if success else 'failed'} in Isaac Lab.",
                )
            ),
        }

    def _resolve_controller(self, phase: str) -> ControllerHook:
        """Resolve an injected or environment-provided controller bridge."""
        injected_controller = (
            self._pick_controller if phase == "pick" else self._place_controller
        )
        if injected_controller is not None:
            return injected_controller

        candidate_names = [
            f"run_{phase}_controller",
            f"{phase}_controller",
            f"execute_{phase}",
            phase,
        ]
        for owner in (self._env, getattr(self._env, "unwrapped", None)):
            if owner is None:
                continue
            for name in candidate_names:
                candidate = getattr(owner, name, None)
                if callable(candidate):
                    return lambda env, action, candidate=candidate: candidate(action)

        raise NotImplementedError(
            f"IsaacLabBackend could not find a {phase} controller bridge. Inject "
            f"`{phase}_controller=` when constructing the backend, or expose one of "
            f"{candidate_names} on the wrapped environment."
        )

    def _flatten_payload(
        self,
        payload: Any,
        prefix: str = "",
    ) -> dict[str, Any]:
        """Flatten nested dict-like Isaac observations for heuristic key lookup."""
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
        """Find the first flattened observation/info entry containing all fragments."""
        for source in (self._latest_obs, self._latest_info):
            flattened = self._flatten_payload(source)
            for key, value in flattened.items():
                normalized_key = key.lower()
                if all(fragment in normalized_key for fragment in key_fragments):
                    return value
        return None

    def _to_python_value(self, value: Any) -> Any:
        """Convert tensors and arrays into plain Python containers."""
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
        """Select the first vectorized environment entry when observations are batched."""
        value = self._to_python_value(value)
        if isinstance(value, list) and value:
            first_item = value[0]
            if isinstance(first_item, list):
                return first_item
        return value

    def _as_points(self, value: Any) -> list[tuple[float, ...]]:
        """Normalize a scalar/list/tensor payload into a list of points."""
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
        """Extract an end-effector-like robot pose from the latest Isaac observation."""
        candidate_keys = [
            ["eef", "pos"],
            ["ee", "pos"],
            ["tcp", "pos"],
            ["hand", "pos"],
            ["robot", "pos"],
        ]
        for fragments in candidate_keys:
            value = self._find_payload_value(fragments)
            points = self._as_points(value)
            if points:
                return points[0]
        return (0.0, 0.0, 0.0)

    def _extract_cube_positions(self) -> list[tuple[float, ...]]:
        """Extract cube/object positions from the latest Isaac observation."""
        candidate_keys = [
            ["cube", "pos"],
            ["object", "pos"],
            ["cube", "position"],
            ["object", "position"],
        ]
        for fragments in candidate_keys:
            value = self._find_payload_value(fragments)
            points = self._as_points(value)
            if points:
                return points
        return []

    def _extract_goal_position(self) -> tuple[float, ...]:
        """Extract the lift target/goal position from the latest Isaac payload."""
        candidate_keys = [
            ["goal", "pos"],
            ["target", "pos"],
            ["command", "pos"],
            ["desired", "pos"],
        ]
        for fragments in candidate_keys:
            value = self._find_payload_value(fragments)
            points = self._as_points(value)
            if points:
                return points[0]
        return (0.1, 0.1, 0.0)

    def _project_to_grid(self, point: tuple[float, ...]) -> tuple[int, int]:
        """Project a 3D Isaac point into the existing 2D benchmark grid."""
        x_value = point[0] if len(point) >= 1 else 0.0
        y_value = point[1] if len(point) >= 2 else 0.0
        grid_x = self._grid_origin[0] + int(round(x_value * self._grid_scale))
        grid_y = self._grid_origin[1] + int(round(y_value * self._grid_scale))
        return (
            max(0, min(39, grid_x)),
            max(0, min(39, grid_y)),
        )
