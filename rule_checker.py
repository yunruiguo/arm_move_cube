"""Minimal action feasibility checks for the planning prototype."""

from __future__ import annotations

from world_state import Position, WorldState


def check_pick(state: WorldState, target: str) -> dict[str, str | bool]:
    """Validate whether a pick action is allowed for the target object."""
    obj = state.get_object(target)
    if obj is None:
        return {"valid": False, "reason": f"pick failed: object '{target}' does not exist."}

    if not obj["graspable"]:
        return {
            "valid": False,
            "reason": f"pick failed: object '{target}' is not graspable.",
        }

    return {"valid": True, "reason": f"pick valid: object '{target}' is graspable."}


def check_place(state: WorldState, position: Position) -> dict[str, str | bool]:
    """Validate whether a place action is allowed at the target position."""
    if position in state.forbidden_zones:
        return {
            "valid": False,
            "reason": f"place failed: target position {position} is forbidden.",
        }

    if state.is_occupied(position):
        return {
            "valid": False,
            "reason": f"place failed: target position {position} is occupied.",
        }

    return {"valid": True, "reason": f"place valid: target position {position} is free."}

