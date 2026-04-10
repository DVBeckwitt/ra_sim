"""Runtime canvas event orchestration for cross-feature GUI interactions."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CanvasInteractionBindings:
    """Live runtime callbacks and shared state for canvas interactions."""

    axis: Any
    geometry_runtime_state: Any
    geometry_preview_state: Any
    geometry_manual_state: Any
    peak_selection_state: Any
    peak_selection_callbacks: Any
    integration_range_drag_callbacks: Any
    manual_pick_session_active: Callable[[], bool]
    set_geometry_manual_pick_mode: Callable[..., None]
    set_geometry_preview_exclude_mode: Callable[..., None]
    toggle_geometry_manual_selection_at: Callable[[float, float], object]
    toggle_live_geometry_preview_exclusion_at: Callable[[float, float], object]
    clamp_to_axis_view: Callable[[Any, float, float], tuple[float, float]]
    apply_geometry_manual_pick_zoom: Callable[..., None]
    update_geometry_manual_pick_preview: Callable[..., None]
    place_geometry_manual_selection_at: Callable[[float, float], object]
    clear_geometry_manual_preview_artists: Callable[..., None]
    restore_geometry_manual_pick_view: Callable[..., None]
    render_current_geometry_manual_pairs: Callable[..., object]
    caked_view_enabled_factory: object
    analysis_peak_state: Any = None
    analysis_peak_callbacks: Any = None
    set_geometry_status_text: Callable[[str], None] | None = None
    draw_idle: Callable[[], None] | None = None
    begin_live_interaction: Callable[[], None] | None = None
    touch_live_interaction: Callable[[], None] | None = None
    end_live_interaction: Callable[[], None] | None = None


@dataclass(frozen=True)
class CanvasInteractionCallbacks:
    """Bound runtime callbacks for canvas click/drag event wiring."""

    on_click: Callable[[Any], bool]
    on_press: Callable[[Any], bool]
    on_motion: Callable[[Any], bool]
    on_release: Callable[[Any], bool]
    on_scroll: Callable[[Any], bool]


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def _runtime_caked_view_enabled(bindings: CanvasInteractionBindings) -> bool:
    return bool(_resolve_runtime_value(bindings.caked_view_enabled_factory))


def _draw_idle(bindings: CanvasInteractionBindings) -> None:
    if callable(bindings.draw_idle):
        bindings.draw_idle()


def _set_status_text(bindings: CanvasInteractionBindings, text: str) -> None:
    if callable(bindings.set_geometry_status_text):
        bindings.set_geometry_status_text(str(text))


def _begin_live_interaction(bindings: CanvasInteractionBindings) -> None:
    if callable(bindings.begin_live_interaction):
        bindings.begin_live_interaction()


def _touch_live_interaction(bindings: CanvasInteractionBindings) -> None:
    if callable(bindings.touch_live_interaction):
        bindings.touch_live_interaction()


def _end_live_interaction(bindings: CanvasInteractionBindings) -> None:
    if callable(bindings.end_live_interaction):
        bindings.end_live_interaction()


def _set_mode(
    setter: Callable[..., None] | None,
    enabled: bool,
    message: str | None = None,
) -> None:
    if not callable(setter):
        return
    try:
        setter(bool(enabled), message=message)
    except TypeError:
        setter(bool(enabled), message)


def _pan_session(bindings: CanvasInteractionBindings) -> dict[str, object] | None:
    session = getattr(bindings.geometry_runtime_state, "_canvas_pan_session", None)
    if isinstance(session, dict):
        return session
    return None


def _clear_pan_session(bindings: CanvasInteractionBindings) -> None:
    setattr(bindings.geometry_runtime_state, "_canvas_pan_session", None)


def _axis_limits(axis: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
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
        or not all(np.isfinite(value) for value in (*xlim, *ylim))
    ):
        return None
    return (xlim, ylim)


def _set_axis_limits(
    axis: Any,
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


def capture_axis_limits(
    axis: Any,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the finite x/y limits for one Matplotlib-like axis."""

    return _axis_limits(axis)


def _coerce_preserved_limits(
    preserved_limits: tuple[float, float] | list[float],
    default_limits: tuple[float, float] | list[float],
) -> tuple[float, float]:
    """Clamp one preserved axis window into the new data bounds."""

    try:
        preserved_start = float(preserved_limits[0])
        preserved_end = float(preserved_limits[1])
        default_start = float(default_limits[0])
        default_end = float(default_limits[1])
    except Exception:
        return (float(default_limits[0]), float(default_limits[1]))

    if not all(
        np.isfinite(value)
        for value in (
            preserved_start,
            preserved_end,
            default_start,
            default_end,
        )
    ):
        return (default_start, default_end)

    preserved_lo = min(preserved_start, preserved_end)
    preserved_hi = max(preserved_start, preserved_end)
    default_lo = min(default_start, default_end)
    default_hi = max(default_start, default_end)
    preserved_span = preserved_hi - preserved_lo
    default_span = default_hi - default_lo
    if preserved_span <= 1.0e-9 or default_span <= 1.0e-9:
        return (default_start, default_end)

    if preserved_span >= default_span:
        clamped_lo = default_lo
        clamped_hi = default_hi
    else:
        clamped_lo = preserved_lo
        clamped_hi = preserved_hi
        if clamped_lo < default_lo:
            shift = default_lo - clamped_lo
            clamped_lo += shift
            clamped_hi += shift
        if clamped_hi > default_hi:
            shift = clamped_hi - default_hi
            clamped_lo -= shift
            clamped_hi -= shift
        clamped_lo = max(clamped_lo, default_lo)
        clamped_hi = min(clamped_hi, default_hi)
        if clamped_hi - clamped_lo <= 1.0e-9:
            clamped_lo = default_lo
            clamped_hi = default_hi

    if preserved_start <= preserved_end:
        return (float(clamped_lo), float(clamped_hi))
    return (float(clamped_hi), float(clamped_lo))


def restore_axis_view(
    axis: Any,
    *,
    preserved_limits: tuple[tuple[float, float], tuple[float, float]] | None,
    default_xlim: tuple[float, float] | list[float],
    default_ylim: tuple[float, float] | list[float],
    preserve: bool,
) -> bool:
    """Apply a preserved axis view when possible, else reset to defaults."""

    resolved_xlim = (float(default_xlim[0]), float(default_xlim[1]))
    resolved_ylim = (float(default_ylim[0]), float(default_ylim[1]))
    if preserve and preserved_limits is not None:
        resolved_xlim = _coerce_preserved_limits(
            preserved_limits[0],
            resolved_xlim,
        )
        resolved_ylim = _coerce_preserved_limits(
            preserved_limits[1],
            resolved_ylim,
        )
    return _set_axis_limits(
        axis,
        xlim=resolved_xlim,
        ylim=resolved_ylim,
    )


def _event_has_axis_data(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    return bool(
        getattr(event, "inaxes", None) is bindings.axis
        and getattr(event, "xdata", None) is not None
        and getattr(event, "ydata", None) is not None
    )


def _manual_pick_click_coords(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> tuple[float, float] | None:
    if not _event_has_axis_data(bindings, event):
        return None
    col, row = bindings.clamp_to_axis_view(
        bindings.axis,
        float(event.xdata),
        float(event.ydata),
    )
    if _runtime_caked_view_enabled(bindings):
        # The live caked view exposes angular coordinates; bind them explicitly
        # as (phi, 2theta) before forwarding the helper's expected (2theta, phi).
        phi_deg = float(row)
        two_theta_deg = float(col)
        return float(two_theta_deg), float(phi_deg)
    return float(col), float(row)


def _manual_pick_zoom_active(bindings: CanvasInteractionBindings) -> bool:
    session = getattr(bindings.geometry_manual_state, "pick_session", None)
    return bool(isinstance(session, dict) and session.get("zoom_active", False))


def _start_pan_session(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    if not _event_has_axis_data(bindings, event):
        return False
    limits = _axis_limits(bindings.axis)
    if limits is None:
        return False
    x0, y0 = bindings.clamp_to_axis_view(
        bindings.axis,
        float(event.xdata),
        float(event.ydata),
    )
    setattr(
        bindings.geometry_runtime_state,
        "_canvas_pan_session",
        {
            "x_anchor": float(x0),
            "y_anchor": float(y0),
            "x_anchor_px": float(getattr(event, "x", np.nan)),
            "y_anchor_px": float(getattr(event, "y", np.nan)),
            "xlim": limits[0],
            "ylim": limits[1],
        },
    )
    return True


def _event_axis_pixel_position(axis: Any, event: Any) -> tuple[float, float] | None:
    bbox = getattr(axis, "bbox", None)
    if bbox is None:
        return None
    try:
        bbox_x0 = float(bbox.x0)
        bbox_y0 = float(bbox.y0)
        bbox_width = float(bbox.width)
        bbox_height = float(bbox.height)
        event_x = float(getattr(event, "x", np.nan))
        event_y = float(getattr(event, "y", np.nan))
    except Exception:
        return None
    if (
        not np.isfinite(bbox_width)
        or not np.isfinite(bbox_height)
        or bbox_width <= 0.0
        or bbox_height <= 0.0
        or not np.isfinite(event_x)
        or not np.isfinite(event_y)
    ):
        return None
    clamped_x = float(np.clip(event_x, bbox_x0, bbox_x0 + bbox_width))
    clamped_y = float(np.clip(event_y, bbox_y0, bbox_y0 + bbox_height))
    return clamped_x, clamped_y


def _update_pan_session(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    session = _pan_session(bindings)
    if session is None:
        return False
    try:
        x_anchor = float(session["x_anchor"])
        y_anchor = float(session["y_anchor"])
        x_anchor_px = float(session.get("x_anchor_px", np.nan))
        y_anchor_px = float(session.get("y_anchor_px", np.nan))
        xlim = tuple(float(value) for value in session["xlim"])
        ylim = tuple(float(value) for value in session["ylim"])
    except Exception:
        _clear_pan_session(bindings)
        return False

    delta_x: float | None = None
    delta_y: float | None = None
    pixel_position = _event_axis_pixel_position(bindings.axis, event)
    if (
        pixel_position is not None
        and np.isfinite(x_anchor_px)
        and np.isfinite(y_anchor_px)
    ):
        bbox = getattr(bindings.axis, "bbox", None)
        try:
            bbox_width = float(getattr(bbox, "width", np.nan))
            bbox_height = float(getattr(bbox, "height", np.nan))
        except Exception:
            bbox_width = np.nan
            bbox_height = np.nan
        if (
            np.isfinite(bbox_width)
            and np.isfinite(bbox_height)
            and bbox_width > 0.0
            and bbox_height > 0.0
        ):
            delta_x = (
                (float(pixel_position[0]) - x_anchor_px)
                * (float(xlim[1]) - float(xlim[0]))
                / float(bbox_width)
            )
            delta_y = (
                (float(pixel_position[1]) - y_anchor_px)
                * (float(ylim[1]) - float(ylim[0]))
                / float(bbox_height)
            )

    if delta_x is None or delta_y is None:
        if not _event_has_axis_data(bindings, event):
            return True
        x1, y1 = bindings.clamp_to_axis_view(
            bindings.axis,
            float(event.xdata),
            float(event.ydata),
        )
        delta_x = float(x1) - x_anchor
        delta_y = float(y1) - y_anchor

    updated = _set_axis_limits(
        bindings.axis,
        xlim=(float(xlim[0]) - delta_x, float(xlim[1]) - delta_x),
        ylim=(float(ylim[0]) - delta_y, float(ylim[1]) - delta_y),
    )
    if updated:
        _draw_idle(bindings)
    return bool(updated)


def _finish_pan_session(bindings: CanvasInteractionBindings) -> bool:
    if _pan_session(bindings) is None:
        return False
    _clear_pan_session(bindings)
    return True


def _scroll_step(event: Any) -> float:
    try:
        step = float(getattr(event, "step", 0.0))
    except Exception:
        step = 0.0
    if np.isfinite(step) and abs(step) > 0.0:
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
    lo, hi = (float(limits[0]), float(limits[1]))
    anchor_value = float(anchor)
    scale_value = float(scale)
    return (
        anchor_value + ((lo - anchor_value) * scale_value),
        anchor_value + ((hi - anchor_value) * scale_value),
    )


def manual_pick_zoom_anchor_fractions(
    axis: Any,
    event: Any,
) -> tuple[float, float]:
    """Return the normalized cursor anchor inside the current axes bbox."""

    anchor_fraction_x = 0.5
    anchor_fraction_y = 0.5
    try:
        bbox = getattr(axis, "bbox", None)
        if (
            bbox is not None
            and np.isfinite(float(bbox.width))
            and np.isfinite(float(bbox.height))
            and float(bbox.width) > 0.0
            and float(bbox.height) > 0.0
        ):
            anchor_fraction_x = float(
                np.clip(
                    (float(event.x) - float(bbox.x0)) / float(bbox.width),
                    0.0,
                    1.0,
                )
            )
            anchor_fraction_y = float(
                np.clip(
                    (float(event.y) - float(bbox.y0)) / float(bbox.height),
                    0.0,
                    1.0,
                )
            )
    except Exception:
        anchor_fraction_x = 0.5
        anchor_fraction_y = 0.5
    return anchor_fraction_x, anchor_fraction_y


def handle_runtime_canvas_click(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas click across manual-pick, preview, and HKL modes."""

    if getattr(event, "button", None) == 3:
        if bool(bindings.peak_selection_state.hkl_pick_armed):
            setattr(bindings.geometry_runtime_state, "_suppress_pan_press_once", True)
            setter = getattr(bindings.peak_selection_callbacks, "set_hkl_pick_mode", None)
            if callable(setter):
                setter(False, "HKL image-pick canceled.")
            return True
        if bool(bindings.geometry_runtime_state.manual_pick_armed):
            setattr(bindings.geometry_runtime_state, "_suppress_pan_press_once", True)
            _set_mode(
                bindings.set_geometry_manual_pick_mode,
                False,
                "Manual geometry picking canceled.",
            )
            return True
        if bool(bindings.geometry_preview_state.exclude_armed):
            setattr(bindings.geometry_runtime_state, "_suppress_pan_press_once", True)
            _set_mode(
                bindings.set_geometry_preview_exclude_mode,
                False,
                "Preview exclusion mode canceled.",
            )
            return True
        analysis_peak_state = getattr(bindings, "analysis_peak_state", None)
        if bool(getattr(analysis_peak_state, "pick_armed", False)):
            setattr(bindings.geometry_runtime_state, "_suppress_pan_press_once", True)
            callbacks = getattr(bindings, "analysis_peak_callbacks", None)
            setter = getattr(callbacks, "set_pick_mode", None)
            if callable(setter):
                setter(False, "Analysis peak picking canceled.")
                return True
        return False

    if getattr(event, "button", None) != 1:
        return False

    if bool(bindings.geometry_runtime_state.manual_pick_armed):
        manual_pick_coords = _manual_pick_click_coords(bindings, event)
        if manual_pick_coords is None:
            return False
        if bool(bindings.manual_pick_session_active()):
            return True
        bindings.toggle_geometry_manual_selection_at(
            float(manual_pick_coords[0]),
            float(manual_pick_coords[1]),
        )
        return True

    analysis_peak_state = getattr(bindings, "analysis_peak_state", None)
    if bool(getattr(analysis_peak_state, "pick_armed", False)):
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        callbacks = getattr(bindings, "analysis_peak_callbacks", None)
        click_handler = getattr(callbacks, "select_peak_from_canvas_click", None)
        if not callable(click_handler):
            return False
        return bool(click_handler(float(event.xdata), float(event.ydata)))

    if bool(bindings.peak_selection_state.hkl_pick_armed):
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        click_handler = getattr(
            bindings.peak_selection_callbacks,
            "select_peak_from_canvas_click",
            None,
        )
        if not callable(click_handler):
            return False
        return bool(click_handler(float(event.xdata), float(event.ydata)))

    if _runtime_caked_view_enabled(bindings):
        return False

    if bool(bindings.geometry_preview_state.exclude_armed):
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        bindings.toggle_live_geometry_preview_exclusion_at(
            float(event.xdata),
            float(event.ydata),
        )
        return True

    return False


def handle_runtime_canvas_press(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas button-press."""

    if bool(getattr(bindings.geometry_runtime_state, "_suppress_pan_press_once", False)):
        setattr(bindings.geometry_runtime_state, "_suppress_pan_press_once", False)
        if getattr(event, "button", None) == 3:
            return True

    if bool(bindings.peak_selection_state.suppress_drag_press_once):
        bindings.peak_selection_state.suppress_drag_press_once = False
        return True

    if getattr(event, "button", None) == 3:
        started = _start_pan_session(bindings, event)
        if started:
            _begin_live_interaction(bindings)
        return started

    if bool(bindings.geometry_runtime_state.manual_pick_armed) and bool(
        bindings.manual_pick_session_active()
    ):
        if getattr(event, "button", None) != 1:
            return False
        manual_pick_coords = _manual_pick_click_coords(bindings, event)
        if manual_pick_coords is None:
            return False
        anchor_fraction_x, anchor_fraction_y = manual_pick_zoom_anchor_fractions(
            bindings.axis,
            event,
        )
        bindings.apply_geometry_manual_pick_zoom(
            float(manual_pick_coords[0]),
            float(manual_pick_coords[1]),
            anchor_fraction_x=anchor_fraction_x,
            anchor_fraction_y=anchor_fraction_y,
        )
        bindings.update_geometry_manual_pick_preview(
            float(manual_pick_coords[0]),
            float(manual_pick_coords[1]),
            force=True,
        )
        return True

    if bool(bindings.geometry_runtime_state.manual_pick_armed):
        if getattr(event, "button", None) != 1:
            return False
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        # The paired click handler owns Qr/Qz-set selection while armed. Consuming
        # the press here prevents integration-range dragging from stealing the same
        # left click in detector or caked views.
        return True

    if bool(bindings.peak_selection_state.hkl_pick_armed):
        if getattr(event, "button", None) != 1:
            return False
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        # The paired click handler owns HKL peak selection while armed. Consuming
        # the press here prevents integration-range dragging from stealing the same
        # left click before or after the click-selection callback resolves.
        return True

    analysis_peak_state = getattr(bindings, "analysis_peak_state", None)
    if bool(getattr(analysis_peak_state, "pick_armed", False)):
        if getattr(event, "button", None) != 1:
            return False
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        return True

    on_press = getattr(bindings.integration_range_drag_callbacks, "on_press", None)
    if not callable(on_press):
        return False
    return bool(on_press(event))


def handle_runtime_canvas_motion(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas motion event."""

    if _pan_session(bindings) is not None:
        _touch_live_interaction(bindings)
        return _update_pan_session(bindings, event)
    if getattr(event, "button", None) == 3:
        return False

    if (
        bool(bindings.geometry_runtime_state.manual_pick_armed)
        and bool(bindings.manual_pick_session_active())
    ):
        if _manual_pick_zoom_active(bindings):
            manual_pick_coords = _manual_pick_click_coords(bindings, event)
            if manual_pick_coords is not None:
                bindings.update_geometry_manual_pick_preview(
                    float(manual_pick_coords[0]),
                    float(manual_pick_coords[1]),
                )
            return True
        return _manual_pick_click_coords(bindings, event) is not None

    on_motion = getattr(bindings.integration_range_drag_callbacks, "on_motion", None)
    if not callable(on_motion):
        return False
    return bool(on_motion(event))


def handle_runtime_canvas_release(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas button-release."""

    if getattr(event, "button", None) == 3 or _pan_session(bindings) is not None:
        finished = _finish_pan_session(bindings)
        if finished:
            _end_live_interaction(bindings)
        return finished

    if getattr(event, "button", None) != 1:
        return False

    if bool(bindings.geometry_runtime_state.manual_pick_armed):
        if bool(bindings.manual_pick_session_active()):
            if not _manual_pick_zoom_active(bindings):
                return True
            manual_pick_coords = _manual_pick_click_coords(bindings, event)
            if manual_pick_coords is not None:
                bindings.place_geometry_manual_selection_at(
                    float(manual_pick_coords[0]),
                    float(manual_pick_coords[1]),
                )
            else:
                bindings.clear_geometry_manual_preview_artists(redraw=False)
                bindings.restore_geometry_manual_pick_view(redraw=False)
                bindings.render_current_geometry_manual_pairs(update_status=False)
                _set_status_text(
                    bindings,
                    "Manual point placement canceled: release inside the image.",
                )
                _draw_idle(bindings)
            return True
        return _manual_pick_click_coords(bindings, event) is not None

    on_release = getattr(bindings.integration_range_drag_callbacks, "on_release", None)
    if not callable(on_release):
        return False
    return bool(on_release(event))


def handle_runtime_canvas_scroll(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas scroll-wheel zoom event."""

    if _pan_session(bindings) is not None:
        return False
    limits = _axis_limits(bindings.axis)
    if limits is None:
        return False

    step = _scroll_step(event)
    if not np.isfinite(step) or abs(step) <= 0.0:
        return False

    xlim, ylim = limits
    anchor_x = getattr(event, "xdata", None)
    anchor_y = getattr(event, "ydata", None)
    if anchor_x is None or not np.isfinite(float(anchor_x)):
        anchor_x = float(xlim[0] + xlim[1]) / 2.0
    else:
        anchor_x = float(anchor_x)
    if anchor_y is None or not np.isfinite(float(anchor_y)):
        anchor_y = float(ylim[0] + ylim[1]) / 2.0
    else:
        anchor_y = float(anchor_y)

    zoom_base = 1.2
    if step > 0.0:
        scale = 1.0 / (zoom_base**float(step))
    else:
        scale = zoom_base ** abs(float(step))

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
        abs(float(next_xlim[1]) - float(next_xlim[0])) <= 1e-9
        or abs(float(next_ylim[1]) - float(next_ylim[0])) <= 1e-9
    ):
        return False
    if not _set_axis_limits(bindings.axis, xlim=next_xlim, ylim=next_ylim):
        return False
    _touch_live_interaction(bindings)
    _draw_idle(bindings)
    _end_live_interaction(bindings)
    return True


def make_runtime_canvas_interaction_bindings_factory(
    *,
    axis: Any,
    geometry_runtime_state: Any,
    geometry_preview_state: Any,
    geometry_manual_state: Any,
    peak_selection_state: Any,
    peak_selection_callbacks: Any,
    integration_range_drag_callbacks: Any,
    manual_pick_session_active: Callable[[], bool],
    set_geometry_manual_pick_mode: Callable[..., None],
    set_geometry_preview_exclude_mode: Callable[..., None],
    toggle_geometry_manual_selection_at: Callable[[float, float], object],
    toggle_live_geometry_preview_exclusion_at: Callable[[float, float], object],
    clamp_to_axis_view: Callable[[Any, float, float], tuple[float, float]],
    apply_geometry_manual_pick_zoom: Callable[..., None],
    update_geometry_manual_pick_preview: Callable[..., None],
    place_geometry_manual_selection_at: Callable[[float, float], object],
    clear_geometry_manual_preview_artists: Callable[..., None],
    restore_geometry_manual_pick_view: Callable[..., None],
    render_current_geometry_manual_pairs: Callable[..., object],
    caked_view_enabled_factory: object,
    analysis_peak_state: Any = None,
    analysis_peak_callbacks: Any = None,
    set_geometry_status_text_factory: object | None = None,
    draw_idle_factory: object | None = None,
    begin_live_interaction_factory: object | None = None,
    touch_live_interaction_factory: object | None = None,
    end_live_interaction_factory: object | None = None,
) -> Callable[[], CanvasInteractionBindings]:
    """Return a zero-arg factory for live runtime canvas-interaction bindings."""

    def _build_bindings() -> CanvasInteractionBindings:
        return CanvasInteractionBindings(
            axis=axis,
            geometry_runtime_state=geometry_runtime_state,
            geometry_preview_state=geometry_preview_state,
            geometry_manual_state=geometry_manual_state,
            peak_selection_state=peak_selection_state,
            peak_selection_callbacks=peak_selection_callbacks,
            integration_range_drag_callbacks=integration_range_drag_callbacks,
            manual_pick_session_active=manual_pick_session_active,
            set_geometry_manual_pick_mode=set_geometry_manual_pick_mode,
            set_geometry_preview_exclude_mode=set_geometry_preview_exclude_mode,
            toggle_geometry_manual_selection_at=toggle_geometry_manual_selection_at,
            toggle_live_geometry_preview_exclusion_at=toggle_live_geometry_preview_exclusion_at,
            clamp_to_axis_view=clamp_to_axis_view,
            apply_geometry_manual_pick_zoom=apply_geometry_manual_pick_zoom,
            update_geometry_manual_pick_preview=update_geometry_manual_pick_preview,
            place_geometry_manual_selection_at=place_geometry_manual_selection_at,
            clear_geometry_manual_preview_artists=clear_geometry_manual_preview_artists,
            restore_geometry_manual_pick_view=restore_geometry_manual_pick_view,
            render_current_geometry_manual_pairs=render_current_geometry_manual_pairs,
            caked_view_enabled_factory=caked_view_enabled_factory,
            analysis_peak_state=analysis_peak_state,
            analysis_peak_callbacks=analysis_peak_callbacks,
            set_geometry_status_text=_resolve_runtime_value(
                set_geometry_status_text_factory
            ),
            draw_idle=_resolve_runtime_value(draw_idle_factory),
            begin_live_interaction=_resolve_runtime_value(
                begin_live_interaction_factory
            ),
            touch_live_interaction=_resolve_runtime_value(
                touch_live_interaction_factory
            ),
            end_live_interaction=_resolve_runtime_value(
                end_live_interaction_factory
            ),
        )

    return _build_bindings


def make_runtime_canvas_interaction_callbacks(
    bindings_factory: Callable[[], CanvasInteractionBindings],
) -> CanvasInteractionCallbacks:
    """Return bound callbacks for runtime canvas event wiring."""

    def _safe_runtime_callback(
        handler: Callable[[CanvasInteractionBindings, Any], bool],
    ) -> Callable[[Any], bool]:
        def _wrapped(event: Any) -> bool:
            bindings = bindings_factory()
            try:
                return bool(handler(bindings, event))
            except Exception:
                traceback.print_exc()
                _set_status_text(
                    bindings,
                    "Canvas interaction canceled after an internal error.",
                )
                _draw_idle(bindings)
                return False

        return _wrapped

    return CanvasInteractionCallbacks(
        on_click=_safe_runtime_callback(handle_runtime_canvas_click),
        on_press=_safe_runtime_callback(handle_runtime_canvas_press),
        on_motion=_safe_runtime_callback(handle_runtime_canvas_motion),
        on_release=_safe_runtime_callback(handle_runtime_canvas_release),
        on_scroll=_safe_runtime_callback(handle_runtime_canvas_scroll),
    )
