"""Launch the Isaac Lab benchmark backend with a visible GUI for manual inspection."""

from __future__ import annotations

import time

from backend_isaaclab import IsaacLabBackend


def main() -> None:
    """Start Isaac Lab with a visible window and keep it rendering until interrupted."""
    backend = IsaacLabBackend(headless=False, debug=True)
    backend.reset()
    state = backend.get_current_state()

    print("[run_isaac_gui] Isaac Lab scene loaded.")
    print(f"[run_isaac_gui] robot position: {state.get_robot_position()}")
    print(f"[run_isaac_gui] objects: {state.objects}")
    print(f"[run_isaac_gui] goal regions: {state.goal_regions}")
    print("[run_isaac_gui] Window is live. Press Ctrl+C in this terminal to exit.")

    env = getattr(backend, "_env", None)
    sim = getattr(getattr(env, "unwrapped", env), "sim", None)
    if sim is None:
        raise RuntimeError("Isaac GUI runner could not access the simulator context.")

    try:
        while True:
            if hasattr(sim, "render"):
                sim.render()
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\n[run_isaac_gui] Shutting down Isaac Lab GUI.")
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
