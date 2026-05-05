"""Shared intensity policy for GUI phi x 2theta caked views."""

from __future__ import annotations


GUI_CAKED_VIEW_CORRECT_SOLID_ANGLE = False


def resolve_gui_caked_view_correct_solid_angle(value: bool | None) -> bool:
    """Resolve optional per-call override to the GUI caked-view default."""

    if value is None:
        return bool(GUI_CAKED_VIEW_CORRECT_SOLID_ANGLE)
    return bool(value)
