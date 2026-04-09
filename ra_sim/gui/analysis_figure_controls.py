"""Import-safe helpers for interactive Analyze 1D Matplotlib figures."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence


_ZOOM_BASE = 1.2
_MIN_AXIS_SPAN = 1.0e-9


def reset_analysis_axes_view(*axes: object) -> bool:
    """Autoscale each provided axis back to its full data extents."""

    updated = False
    for axis in axes:
        relim = getattr(axis, "relim", None)
        autoscale_view = getattr(axis, "autoscale_view", None)
        if not callable(relim) or not callable(autoscale_view):
            continue
        try:
            relim()
            autoscale_view()
        except Exception:
            continue
        updated = True
    return updated


def _resolve_event_axis(
    event: object,
    axes: Sequence[object],
) -> object | None:
    inaxes = getattr(event, "inaxes", None)
    for axis in axes:
        if inaxes is axis:
            return axis
    return None


def _axis_limits(
    axis: object,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    get_xlim = getattr(axis, "get_xlim", None)
    get_ylim = getattr(axis, "get_ylim", None)
    if not callable(get_xlim) or not callable(get_ylim):
        return None
    try:
        xlim = tuple(float(value) for value in get_xlim())
        ylim = tuple(float(value) for value in get_ylim())
    except Exception:
        return None
    if (
        len(xlim) != 2
        or len(ylim) != 2
        or not all(math.isfinite(value) for value in (*xlim, *ylim))
    ):
        return None
    return (xlim, ylim)


def _set_axis_limits(
    axis: object,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> bool:
    set_xlim = getattr(axis, "set_xlim", None)
    set_ylim = getattr(axis, "set_ylim", None)
    if not callable(set_xlim) or not callable(set_ylim):
        return False
    try:
        set_xlim(float(xlim[0]), float(xlim[1]))
        set_ylim(float(ylim[0]), float(ylim[1]))
    except Exception:
        return False
    return True


def _draw_idle(canvas: object) -> None:
    draw_idle = getattr(canvas, "draw_idle", None)
    if callable(draw_idle):
        try:
            draw_idle()
        except Exception:
            pass


def _scroll_step(event: object) -> float:
    try:
        step = float(getattr(event, "step", 0.0))
    except Exception:
        step = 0.0
    if math.isfinite(step) and abs(step) > 0.0:
        return step

    button = str(getattr(event, "button", "") or "").strip().lower()
    if button == "up":
        return 1.0
    if button == "down":
        return -1.0
    return 0.0


def _zoom_limits_about_anchor(
    limits: tuple[float, float],
    *,
    anchor: float,
    scale: float,
) -> tuple[float, float]:
    lower = float(limits[0])
    upper = float(limits[1])
    anchor_value = float(anchor)
    scale_value = float(scale)
    return (
        anchor_value + ((lower - anchor_value) * scale_value),
        anchor_value + ((upper - anchor_value) * scale_value),
    )


def create_analysis_figure_interactions(
    *,
    canvas: object,
    axes: Sequence[object],
    on_reset_view: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Attach direct mouse pan/zoom/reset controls to the stacked 1D axes."""

    tracked_axes = tuple(axis for axis in axes if axis is not None)
    pan_state: dict[str, object | None] = {
        "axis": None,
        "x_anchor": None,
        "y_anchor": None,
        "xlim": None,
        "ylim": None,
    }

    def _clear_pan_state() -> None:
        pan_state.update(
            axis=None,
            x_anchor=None,
            y_anchor=None,
            xlim=None,
            ylim=None,
        )

    def _handle_press(event: object) -> bool:
        if getattr(event, "button", None) != 1:
            return False

        axis = _resolve_event_axis(event, tracked_axes)
        if axis is None:
            return False

        if bool(getattr(event, "dblclick", False)):
            if callable(on_reset_view):
                on_reset_view()
            else:
                if reset_analysis_axes_view(*tracked_axes):
                    _draw_idle(canvas)
            _clear_pan_state()
            return True

        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None or ydata is None:
            return False

        limits = _axis_limits(axis)
        if limits is None:
            return False

        pan_state.update(
            axis=axis,
            x_anchor=float(xdata),
            y_anchor=float(ydata),
            xlim=limits[0],
            ylim=limits[1],
        )
        return True

    def _handle_motion(event: object) -> bool:
        axis = pan_state.get("axis")
        if axis is None:
            return False

        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None or ydata is None:
            return True

        try:
            x_anchor = float(pan_state["x_anchor"])
            y_anchor = float(pan_state["y_anchor"])
            xlim = tuple(float(value) for value in pan_state["xlim"])
            ylim = tuple(float(value) for value in pan_state["ylim"])
        except Exception:
            _clear_pan_state()
            return False

        delta_x = float(xdata) - x_anchor
        delta_y = float(ydata) - y_anchor
        updated = _set_axis_limits(
            axis,
            xlim=(float(xlim[0]) - delta_x, float(xlim[1]) - delta_x),
            ylim=(float(ylim[0]) - delta_y, float(ylim[1]) - delta_y),
        )
        if updated:
            _draw_idle(canvas)
        return bool(updated)

    def _handle_release(event: object) -> bool:
        if pan_state.get("axis") is None:
            return False
        if getattr(event, "button", None) != 1:
            return False
        _clear_pan_state()
        return True

    def _handle_scroll(event: object) -> bool:
        axis = _resolve_event_axis(event, tracked_axes)
        if axis is None:
            return False

        limits = _axis_limits(axis)
        if limits is None:
            return False
        xlim, ylim = limits

        step = _scroll_step(event)
        if not math.isfinite(step) or abs(step) <= 0.0:
            return False

        anchor_x = getattr(event, "xdata", None)
        anchor_y = getattr(event, "ydata", None)
        if anchor_x is None or not math.isfinite(float(anchor_x)):
            anchor_x = (float(xlim[0]) + float(xlim[1])) / 2.0
        if anchor_y is None or not math.isfinite(float(anchor_y)):
            anchor_y = (float(ylim[0]) + float(ylim[1])) / 2.0

        if step > 0.0:
            scale = 1.0 / (_ZOOM_BASE ** float(step))
        else:
            scale = _ZOOM_BASE ** abs(float(step))

        next_xlim = _zoom_limits_about_anchor(
            xlim,
            anchor=float(anchor_x),
            scale=float(scale),
        )
        next_ylim = _zoom_limits_about_anchor(
            ylim,
            anchor=float(anchor_y),
            scale=float(scale),
        )
        if (
            abs(float(next_xlim[1]) - float(next_xlim[0])) <= _MIN_AXIS_SPAN
            or abs(float(next_ylim[1]) - float(next_ylim[0])) <= _MIN_AXIS_SPAN
        ):
            return False
        if not _set_axis_limits(axis, xlim=next_xlim, ylim=next_ylim):
            return False
        _draw_idle(canvas)
        return True

    mpl_connect = getattr(canvas, "mpl_connect", None)
    if not callable(mpl_connect):
        return {}

    return {
        "button_press_event": mpl_connect("button_press_event", _handle_press),
        "motion_notify_event": mpl_connect("motion_notify_event", _handle_motion),
        "button_release_event": mpl_connect("button_release_event", _handle_release),
        "scroll_event": mpl_connect("scroll_event", _handle_scroll),
    }
