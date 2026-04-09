"""Import-safe helpers for determining whether Analyze outputs are visible."""

from __future__ import annotations


def analysis_outputs_visible(
    *,
    control_tab_var: object = None,
    popout_open: bool = False,
    assume_visible_when_unknown: bool = True,
) -> bool:
    """Return whether Analyze-only outputs are currently visible to the user."""

    if bool(popout_open):
        return True

    getter = getattr(control_tab_var, "get", None)
    if not callable(getter):
        return bool(assume_visible_when_unknown)

    try:
        tab_key = str(getter() or "").strip().lower()
    except Exception:
        return bool(assume_visible_when_unknown)

    if not tab_key:
        return bool(assume_visible_when_unknown)
    return tab_key == "analyze"
