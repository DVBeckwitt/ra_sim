"""Import-safe helpers for Analyze quick-control actions."""

from __future__ import annotations

from collections.abc import Callable


DEFAULT_CLEAR_INTEGRATION_MESSAGE = (
    "Integration region cleared. Draw a new box to integrate again."
)


def _invoke(callback: Callable[[], object] | None) -> bool:
    if not callable(callback):
        return False
    try:
        callback()
    except Exception:
        return False
    return True


def _set_boolean_var(value: object, enabled: bool) -> bool:
    setter = getattr(value, "set", None)
    if not callable(setter):
        return False
    try:
        setter(bool(enabled))
    except Exception:
        return False
    return True


def clear_analysis_integration_region(
    *,
    show_1d_var: object = None,
    hide_drag_region: Callable[[], object] | None = None,
    disable_peak_pick: Callable[[], object] | None = None,
    clear_peak_selection: Callable[[], object] | None = None,
    apply_cleared_integration: Callable[[], object] | None = None,
    set_status_text: Callable[[str], object] | None = None,
    message: str = DEFAULT_CLEAR_INTEGRATION_MESSAGE,
) -> bool:
    """Clear the current integration region and pause its downstream effects."""

    updated = _invoke(hide_drag_region)
    updated = _invoke(disable_peak_pick) or updated
    updated = _invoke(clear_peak_selection) or updated
    updated = _set_boolean_var(show_1d_var, False) or updated
    updated = _invoke(apply_cleared_integration) or updated

    if callable(set_status_text):
        try:
            set_status_text(str(message))
        except Exception:
            pass
        else:
            updated = True

    return updated
