"""Run a real Franka pick-and-place motion demo in an Isaac Sim GUI session."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import isaacsim
from isaacsim.core.experimental.utils.stage import create_new_stage_async
from omni.kit.app import get_app


def ensure_interactive_example_path() -> None:
    """Add the Isaac Sim interactive examples extension to sys.path if needed."""
    ext_root = (
        Path(isaacsim.__file__).resolve().parent
        / "exts"
        / "isaacsim.examples.interactive"
    )
    ext_root_str = str(ext_root)
    if ext_root.exists() and ext_root_str not in sys.path:
        sys.path.append(ext_root_str)


ensure_interactive_example_path()

from isaacsim.examples.interactive.pick_place.pick_place_example import (
    FrankaPickPlaceInteractive,
)


async def step_updates(frame_count: int) -> None:
    """Advance the Omniverse app for a fixed number of update frames."""
    app = get_app()
    for _ in range(frame_count):
        await app.next_update_async()


async def load_and_execute_demo() -> FrankaPickPlaceInteractive:
    """Load the official Franka pick-and-place example and execute it once."""
    await create_new_stage_async()
    await step_updates(5)

    sample = FrankaPickPlaceInteractive()
    await sample.load_world_async()
    await step_updates(60)

    print("[run_isaac_rollout_gui] Scene loaded.")
    print("[run_isaac_rollout_gui] Starting real Franka pick-and-place execution...")

    await sample.execute_pick_place_async()

    while sample.is_executing():
        await get_app().next_update_async()

    print("[run_isaac_rollout_gui] Pick-and-place motion finished.")
    return sample


async def keep_window_alive() -> None:
    """Keep the GUI responsive until the user interrupts the process."""
    print("[run_isaac_rollout_gui] Window will stay open. Press Ctrl+C to exit.")
    while True:
        await get_app().next_update_async()


def main() -> None:
    """Launch the GUI demo, run one real motion sequence, then keep the window open."""
    event_loop = asyncio.get_event_loop()
    sample: FrankaPickPlaceInteractive | None = None

    try:
        sample = event_loop.run_until_complete(load_and_execute_demo())
        event_loop.run_until_complete(keep_window_alive())
    except KeyboardInterrupt:
        print("\n[run_isaac_rollout_gui] Shutting down Isaac Sim GUI.")
    finally:
        if sample is not None:
            sample.simulation_context_cleanup()
        simulation_app.close()


if __name__ == "__main__":
    main()
