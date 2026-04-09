"""Workflow helpers for canvas drag-selection of 1D integration ranges."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.patches import Rectangle
from ra_sim.gui import controllers as gui_controllers

_FAST_VIEWER_DRAG_CURVE_PREVIEW_ENABLED = False

_ACTIVE_DRAG_EDGE_COLOR = "#ff7a00"
_ACTIVE_DRAG_EDGE_RGBA = (255, 122, 0, 255)
_ACTIVE_DRAG_FACE_RGBA = (1.0, 0.48, 0.0, 0.14)
_ACTIVE_DRAG_LINEWIDTH = 2.8

_SELECTED_REGION_EDGE_COLOR = "#ff00a8"
_SELECTED_REGION_EDGE_RGBA = (255, 0, 168, 255)
_SELECTED_REGION_FACE_RGBA = (1.0, 0.0, 0.66, 0.10)
_SELECTED_REGION_OVERLAY_RGBA = (1.0, 0.0, 0.66, 0.46)
_SELECTED_REGION_LINEWIDTH = 3.0


@dataclass
class IntegrationRangeDragBindings:
    """Runtime callbacks and shared state used by integration-range dragging."""

    drag_state: Any
    peak_selection_state: Any
    range_view_state: Any
    ax: Any
    drag_select_rect: Any
    integration_region_overlay: Any
    integration_region_rect: Any
    image_display: Any
    get_detector_angular_maps: Callable[[Any], tuple[object, object]]
    range_visible_factory: object
    caked_view_enabled_factory: object
    unscaled_image_present_factory: object
    ai_factory: object
    show_1d_var: object = None
    sync_peak_selection_state: Callable[[], None] | None = None
    schedule_range_update: Callable[[], None] | None = None
    last_sim_res2_factory: object = None
    draw_idle: Callable[[], None] | None = None
    set_status_text: Callable[[str], None] | None = None


@dataclass(frozen=True)
class IntegrationRangeDragCallbacks:
    """Bound callbacks for runtime integration-range dragging."""

    on_press: Callable[[Any], bool]
    on_motion: Callable[[Any], bool]
    on_release: Callable[[Any], bool]
    reset: Callable[[], None]


@dataclass
class IntegrationRangeUpdateBindings:
    """Runtime callbacks and state used by 1D integration-range updates."""

    root: Any
    simulation_runtime_state: Any
    analysis_view_state: Any
    range_view_state: Any
    display_controls_state: Any
    hkl_lookup_controls: Any = None
    integration_range_drag_callbacks: Any = None
    refresh_integration_from_cached_results: Callable[[], bool] | None = None
    refresh_display_from_controls: Callable[[], None] | None = None
    schedule_update: Callable[[], None] | None = None
    range_update_debounce_ms: int = 120


@dataclass(frozen=True)
class IntegrationRangeUpdateCallbacks:
    """Bound callbacks for integration-range update and analysis toggles."""

    schedule_range_update: Callable[..., None]
    toggle_1d_plots: Callable[[], None]
    toggle_caked_2d: Callable[[], None]
    toggle_log_display: Callable[[], None]


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def _safe_var_get(var: object) -> object:
    getter = getattr(var, "get", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _safe_var_set(var: object, value: object) -> None:
    setter = getattr(var, "set", None)
    if callable(setter):
        setter(value)


def _safe_var_trace_add(var: object, callback: Callable[..., object]) -> None:
    trace_add = getattr(var, "trace_add", None)
    if callable(trace_add):
        trace_add("write", callback)


def _activate_runtime_1d_analysis(show_1d_var: object) -> None:
    setter = getattr(show_1d_var, "set", None)
    if callable(setter):
        try:
            setter(True)
        except Exception:
            return


def _runtime_range_update_debounce_ms(
    bindings: IntegrationRangeUpdateBindings,
) -> int:
    try:
        return max(1, int(bindings.range_update_debounce_ms))
    except Exception:
        return 120


def _draw_idle(bindings: IntegrationRangeDragBindings) -> None:
    if callable(bindings.draw_idle):
        bindings.draw_idle()


def _set_status_text(
    bindings: IntegrationRangeDragBindings,
    text: str,
) -> None:
    if callable(bindings.set_status_text):
        bindings.set_status_text(str(text))


def _sync_peak_selection_state(bindings: IntegrationRangeDragBindings) -> None:
    if callable(bindings.sync_peak_selection_state):
        bindings.sync_peak_selection_state()


def _runtime_caked_view_enabled(bindings: IntegrationRangeDragBindings) -> bool:
    return bool(_resolve_runtime_value(bindings.caked_view_enabled_factory))


def _runtime_range_visible(bindings: IntegrationRangeDragBindings) -> bool:
    return bool(_resolve_runtime_value(bindings.range_visible_factory))


def _runtime_unscaled_image_present(bindings: IntegrationRangeDragBindings) -> bool:
    return bool(_resolve_runtime_value(bindings.unscaled_image_present_factory))


def _runtime_ai(bindings: IntegrationRangeDragBindings):
    return _resolve_runtime_value(bindings.ai_factory)


def _runtime_last_sim_res2(bindings: IntegrationRangeDragBindings):
    return _resolve_runtime_value(bindings.last_sim_res2_factory)


def create_integration_region_highlight_cmap(
    *,
    listed_colormap_cls: Callable[[list[tuple[float, float, float, float]]], Any],
) -> Any:
    """Return the live detector-overlay colormap for selected regions."""

    return listed_colormap_cls(
        [
            (0.0, 0.0, 0.0, 0.0),
            _SELECTED_REGION_OVERLAY_RGBA,
        ]
    )


def range_refresh_requires_pending_analysis_result(
    *,
    active_job: object,
    queued_job: object,
    cached_result: object,
) -> bool:
    """Return whether a range refresh must wait for a background caking result."""

    if cached_result is not None:
        return False
    return active_job is not None or queued_job is not None


def _clear_drag_coordinates(drag_state) -> None:
    drag_state.active = False
    drag_state.mode = None
    drag_state.x0 = None
    drag_state.y0 = None
    drag_state.x1 = None
    drag_state.y1 = None
    drag_state.tth0 = None
    drag_state.phi0 = None
    drag_state.tth1 = None
    drag_state.phi1 = None
    setattr(drag_state, "_raw_drag_preview_center", None)
    setattr(drag_state, "_raw_drag_preview_data", None)
    _set_fast_viewer_overlay_state(
        drag_state,
        curve_specs=(),
        suppress_overlay_image=False,
    )


def _set_fast_viewer_overlay_state(
    drag_state: object,
    *,
    curve_specs: object = (),
    suppress_overlay_image: bool,
) -> None:
    if not _FAST_VIEWER_DRAG_CURVE_PREVIEW_ENABLED:
        curve_specs = ()
        suppress_overlay_image = False
    setattr(
        drag_state,
        "_fast_viewer_curve_specs",
        tuple(curve_specs or ()),
    )
    setattr(
        drag_state,
        "_fast_viewer_suppress_overlay_image",
        bool(suppress_overlay_image),
    )
    try:
        version = int(getattr(drag_state, "_fast_viewer_overlay_version", 0))
    except Exception:
        version = 0
    setattr(drag_state, "_fast_viewer_overlay_version", version + 1)


def _wrap_angle_degrees(value: float) -> float:
    return float(((float(value) + 180.0) % 360.0) - 180.0)


def _canonicalize_shortest_angle_interval(
    angle0: object,
    angle1: object,
) -> tuple[float, float] | None:
    if angle0 is None or angle1 is None:
        return None
    wrapped0 = _wrap_angle_degrees(float(angle0))
    wrapped1 = _wrap_angle_degrees(float(angle1))
    positive_span = (wrapped1 - wrapped0) % 360.0
    if positive_span <= 180.0:
        return (wrapped0, wrapped1)
    return (wrapped1, wrapped0)


def detector_phi_mask(
    phi_values: object,
    phi_start: object,
    phi_end: object,
) -> np.ndarray:
    phi_array = np.asarray(phi_values, dtype=float)
    wrapped = ((phi_array + 180.0) % 360.0) - 180.0
    phi_start_val = float(phi_start)
    phi_end_val = float(phi_end)
    if phi_end_val >= phi_start_val:
        return (wrapped >= phi_start_val) & (wrapped <= phi_end_val)
    return (wrapped >= phi_start_val) | (wrapped <= phi_end_val)


def clamp_to_axis_view(ax: object, x: float, y: float) -> tuple[float, float]:
    """Clamp one display-space point to the current axes limits."""

    x_lo, x_hi = sorted(ax.get_xlim())
    y_lo, y_hi = sorted(ax.get_ylim())
    clamped_x = min(max(float(x), float(x_lo)), float(x_hi))
    clamped_y = min(max(float(y), float(y_lo)), float(y_hi))
    return float(clamped_x), float(clamped_y)


def create_drag_select_rectangle(
    ax: object,
    *,
    rectangle_cls: Callable[..., Any] = Rectangle,
) -> Any:
    """Create and attach the visible caked-view drag rectangle."""

    rect = rectangle_cls(
        (0.0, 0.0),
        0.0,
        0.0,
        linewidth=_ACTIVE_DRAG_LINEWIDTH,
        edgecolor=_ACTIVE_DRAG_EDGE_COLOR,
        facecolor=_ACTIVE_DRAG_FACE_RGBA,
        linestyle="-",
        zorder=8,
    )
    rect.set_visible(False)
    add_patch = getattr(ax, "add_patch", None)
    if callable(add_patch):
        add_patch(rect)
    return rect


def create_integration_region_rectangle(
    ax: object,
    *,
    rectangle_cls: Callable[..., Any] = Rectangle,
) -> Any:
    """Create and attach the caked-view 1D integration selection rectangle."""

    rect = rectangle_cls(
        (0.0, 0.0),
        0.0,
        0.0,
        linewidth=_SELECTED_REGION_LINEWIDTH,
        edgecolor=_SELECTED_REGION_EDGE_COLOR,
        facecolor=_SELECTED_REGION_FACE_RGBA,
        linestyle="-",
        zorder=7,
    )
    rect.set_visible(False)
    add_patch = getattr(ax, "add_patch", None)
    if callable(add_patch):
        add_patch(rect)
    return rect


def update_runtime_drag_rectangle(
    bindings: IntegrationRangeDragBindings,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> None:
    """Refresh the visible drag rectangle for one caked-view drag."""

    _set_fast_viewer_overlay_state(
        bindings.drag_state,
        curve_specs=(),
        suppress_overlay_image=False,
    )
    x_min, x_max = sorted((float(x0), float(x1)))
    y_min, y_max = sorted((float(y0), float(y1)))
    bindings.integration_region_overlay.set_visible(False)
    bindings.integration_region_rect.set_visible(False)
    bindings.drag_select_rect.set_xy((x_min, y_min))
    bindings.drag_select_rect.set_width(x_max - x_min)
    bindings.drag_select_rect.set_height(y_max - y_min)
    bindings.drag_select_rect.set_visible(True)
    _draw_idle(bindings)


def _sorted_detector_angle_bounds(
    tth0: object,
    phi0: object,
    tth1: object,
    phi1: object,
) -> tuple[float, float, float, float] | None:
    if None in (tth0, phi0, tth1, phi1):
        return None
    tth_min, tth_max = sorted((float(tth0), float(tth1)))
    phi_bounds = _canonicalize_shortest_angle_interval(phi0, phi1)
    if phi_bounds is None:
        return None
    return (float(tth_min), float(tth_max), float(phi_bounds[0]), float(phi_bounds[1]))


def _display_polar_angle_degrees(
    x_value: float,
    y_value: float,
    center_x: float,
    center_y: float,
) -> float:
    return _wrap_angle_degrees(
        np.degrees(
            np.arctan2(
                float(y_value) - float(center_y),
                float(x_value) - float(center_x),
            )
        )
    )


def _detector_preview_center(two_theta: object) -> tuple[float, float] | None:
    try:
        array = np.asarray(two_theta, dtype=float)
    except Exception:
        return None
    if array.ndim < 2 or array.size <= 0:
        return None
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return None
    masked = np.where(finite_mask, array, np.inf)
    flat_index = int(np.argmin(masked))
    row_idx, col_idx = np.unravel_index(flat_index, array.shape[:2])
    return (float(col_idx), float(row_idx))


def _preview_buffer(
    drag_state: object,
    shape: tuple[int, int],
) -> np.ndarray:
    buffer = getattr(drag_state, "_raw_drag_preview_data", None)
    if not isinstance(buffer, np.ndarray) or buffer.shape != tuple(shape):
        buffer = np.zeros(shape, dtype=float)
        setattr(drag_state, "_raw_drag_preview_data", buffer)
    else:
        buffer.fill(0.0)
    return buffer


def _stamp_preview_points(
    preview: np.ndarray,
    *,
    x_values: object,
    y_values: object,
) -> None:
    try:
        x_array = np.rint(np.asarray(x_values, dtype=float)).astype(int).reshape(-1)
        y_array = np.rint(np.asarray(y_values, dtype=float)).astype(int).reshape(-1)
    except Exception:
        return
    if x_array.size <= 0 or y_array.size <= 0:
        return
    point_count = min(int(x_array.size), int(y_array.size))
    if point_count <= 0:
        return
    x_clipped = np.clip(x_array[:point_count], 0, int(preview.shape[1]) - 1)
    y_clipped = np.clip(y_array[:point_count], 0, int(preview.shape[0]) - 1)
    for row_offset, col_offset in (
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ):
        row_idx = np.clip(y_clipped + int(row_offset), 0, int(preview.shape[0]) - 1)
        col_idx = np.clip(x_clipped + int(col_offset), 0, int(preview.shape[1]) - 1)
        preview[row_idx, col_idx] = 1.0


def _update_detector_drag_arc_preview(
    bindings: IntegrationRangeDragBindings,
    *,
    two_theta: object,
) -> bool:
    drag_state = bindings.drag_state
    if None in (drag_state.x0, drag_state.y0, drag_state.x1, drag_state.y1):
        return False

    center = getattr(drag_state, "_raw_drag_preview_center", None)
    if center is None:
        center = _detector_preview_center(two_theta)
        setattr(drag_state, "_raw_drag_preview_center", center)
    if center is None:
        return False
    center_x, center_y = center
    if not (np.isfinite(center_x) and np.isfinite(center_y)):
        return False

    try:
        detector_shape = tuple(int(value) for value in np.asarray(two_theta).shape[:2])
    except Exception:
        return False
    if len(detector_shape) != 2 or detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return False

    preview = _preview_buffer(drag_state, detector_shape)
    radius0 = float(
        np.hypot(float(drag_state.x0) - float(center_x), float(drag_state.y0) - float(center_y))
    )
    radius1 = float(
        np.hypot(float(drag_state.x1) - float(center_x), float(drag_state.y1) - float(center_y))
    )
    radius_min, radius_max = sorted((radius0, radius1))

    angle_bounds = _canonicalize_shortest_angle_interval(
        _display_polar_angle_degrees(
            float(drag_state.x0),
            float(drag_state.y0),
            float(center_x),
            float(center_y),
        ),
        _display_polar_angle_degrees(
            float(drag_state.x1),
            float(drag_state.y1),
            float(center_x),
            float(center_y),
        ),
    )
    if angle_bounds is None:
        return False
    angle_start, angle_end = angle_bounds
    span_degrees = (
        float(angle_end - angle_start)
        if float(angle_end) >= float(angle_start)
        else float(angle_end + 360.0 - angle_start)
    )
    arc_samples = max(24, int(np.ceil(max(span_degrees * 2.0, radius_max - radius_min))))
    edge_samples = max(4, int(np.ceil(abs(radius_max - radius_min))) + 1)
    angle_values = np.linspace(float(angle_start), float(angle_start) + float(span_degrees), arc_samples)
    angle_radians = np.deg2rad(angle_values)

    outer_x = float(center_x) + float(radius_max) * np.cos(angle_radians)
    outer_y = float(center_y) + float(radius_max) * np.sin(angle_radians)
    inner_x = float(center_x) + float(radius_min) * np.cos(angle_radians)
    inner_y = float(center_y) + float(radius_min) * np.sin(angle_radians)
    radial_values = np.linspace(float(radius_min), float(radius_max), edge_samples)
    start_radians = float(np.deg2rad(angle_start))
    end_radians = float(np.deg2rad(angle_start + span_degrees))

    _stamp_preview_points(preview, x_values=outer_x, y_values=outer_y)
    _stamp_preview_points(preview, x_values=inner_x, y_values=inner_y)
    _stamp_preview_points(
        preview,
        x_values=float(center_x) + radial_values * np.cos(start_radians),
        y_values=float(center_y) + radial_values * np.sin(start_radians),
    )
    _stamp_preview_points(
        preview,
        x_values=float(center_x) + radial_values * np.cos(end_radians),
        y_values=float(center_y) + radial_values * np.sin(end_radians),
    )

    if not np.any(preview):
        return False

    bindings.integration_region_rect.set_visible(False)
    bindings.drag_select_rect.set_visible(False)
    _set_fast_viewer_overlay_state(
        drag_state,
        curve_specs=(
            {
                "x_values": tuple(float(value) for value in outer_x),
                "y_values": tuple(float(value) for value in outer_y),
                "edge_rgba": _ACTIVE_DRAG_EDGE_RGBA,
                "linewidth": 2.0,
                "linestyle": "-",
                "zorder": 9.0,
            },
            {
                "x_values": tuple(float(value) for value in inner_x),
                "y_values": tuple(float(value) for value in inner_y),
                "edge_rgba": _ACTIVE_DRAG_EDGE_RGBA,
                "linewidth": 2.0,
                "linestyle": "-",
                "zorder": 9.0,
            },
            {
                "x_values": tuple(
                    float(value)
                    for value in (float(center_x) + radial_values * np.cos(start_radians))
                ),
                "y_values": tuple(
                    float(value)
                    for value in (float(center_y) + radial_values * np.sin(start_radians))
                ),
                "edge_rgba": _ACTIVE_DRAG_EDGE_RGBA,
                "linewidth": 1.6,
                "linestyle": "-",
                "zorder": 9.0,
            },
            {
                "x_values": tuple(
                    float(value)
                    for value in (float(center_x) + radial_values * np.cos(end_radians))
                ),
                "y_values": tuple(
                    float(value)
                    for value in (float(center_y) + radial_values * np.sin(end_radians))
                ),
                "edge_rgba": _ACTIVE_DRAG_EDGE_RGBA,
                "linewidth": 1.6,
                "linestyle": "-",
                "zorder": 9.0,
            },
        ),
        suppress_overlay_image=True,
    )
    bindings.integration_region_overlay.set_data(preview)
    bindings.integration_region_overlay.set_extent(bindings.image_display.get_extent())
    bindings.integration_region_overlay.set_visible(True)
    return True


def _update_detector_integration_overlay(
    bindings: IntegrationRangeDragBindings,
    *,
    ai: object,
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
) -> bool:
    _set_fast_viewer_overlay_state(
        bindings.drag_state,
        curve_specs=(),
        suppress_overlay_image=False,
    )
    bindings.integration_region_rect.set_visible(False)
    if ai is None:
        bindings.integration_region_overlay.set_visible(False)
        return False

    two_theta, phi_vals = bindings.get_detector_angular_maps(ai)
    if two_theta is None or phi_vals is None:
        bindings.integration_region_overlay.set_visible(False)
        return False

    mask = (
        (two_theta >= float(tth_min))
        & (two_theta <= float(tth_max))
        & detector_phi_mask(phi_vals, float(phi_min), float(phi_max))
    )
    if not np.any(mask):
        bindings.integration_region_overlay.set_visible(False)
        return False

    bindings.integration_region_overlay.set_data(mask.astype(float))
    bindings.integration_region_overlay.set_extent(bindings.image_display.get_extent())
    bindings.integration_region_overlay.set_visible(True)
    return True


def display_to_detector_angles(
    bindings: IntegrationRangeDragBindings,
    col: float,
    row: float,
    ai: object,
) -> tuple[float, float] | None:
    """Resolve display coordinates to detector angles using the current maps."""

    two_theta, phi_vals = bindings.get_detector_angular_maps(ai)
    if two_theta is None or phi_vals is None:
        return None

    height, width = two_theta.shape[:2]
    if int(height) <= 0 or int(width) <= 0:
        return None

    col_idx = min(max(int(round(float(col))), 0), int(width) - 1)
    row_idx = min(max(int(round(float(row))), 0), int(height) - 1)
    tth = float(two_theta[row_idx, col_idx])
    phi = float(phi_vals[row_idx, col_idx])
    if not (np.isfinite(tth) and np.isfinite(phi)):
        return None
    return (tth, phi)


def update_runtime_raw_drag_preview(
    bindings: IntegrationRangeDragBindings,
    ai: object,
) -> bool:
    """Refresh the raw-detector drag preview using the true angular footprint."""

    drag_state = bindings.drag_state
    bounds = _sorted_detector_angle_bounds(
        drag_state.tth0,
        drag_state.phi0,
        drag_state.tth1,
        drag_state.phi1,
    )
    if bounds is None:
        return False

    bindings.drag_select_rect.set_visible(False)
    two_theta, _phi_vals = bindings.get_detector_angular_maps(ai)
    updated = False
    if two_theta is not None:
        updated = _update_detector_drag_arc_preview(
            bindings,
            two_theta=two_theta,
        )
    if not updated:
        updated = _update_detector_integration_overlay(
            bindings,
            ai=ai,
            tth_min=bounds[0],
            tth_max=bounds[1],
            phi_min=bounds[2],
            phi_max=bounds[3],
        )
    _draw_idle(bindings)
    return bool(updated)


def set_runtime_integration_range_from_drag(
    bindings: IntegrationRangeDragBindings,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    preserve_wrapped_phi: bool = False,
) -> bool:
    """Apply one completed drag to the integration-range controls."""

    view_state = bindings.range_view_state
    if view_state is None:
        return False

    tth_min, tth_max = sorted((float(x0), float(x1)))
    if bool(preserve_wrapped_phi):
        phi_bounds = _canonicalize_shortest_angle_interval(float(y0), float(y1))
        if phi_bounds is None:
            return False
        phi_min, phi_max = phi_bounds
    else:
        phi_min, phi_max = sorted((float(y0), float(y1)))

    tth_lo = float(view_state.tth_min_slider.cget("from"))
    tth_hi = float(view_state.tth_min_slider.cget("to"))
    phi_lo = float(view_state.phi_min_slider.cget("from"))
    phi_hi = float(view_state.phi_min_slider.cget("to"))
    tth_min = min(max(tth_min, min(tth_lo, tth_hi)), max(tth_lo, tth_hi))
    tth_max = min(max(tth_max, min(tth_lo, tth_hi)), max(tth_lo, tth_hi))
    phi_min = min(max(phi_min, min(phi_lo, phi_hi)), max(phi_lo, phi_hi))
    phi_max = min(max(phi_max, min(phi_lo, phi_hi)), max(phi_lo, phi_hi))

    if tth_max <= tth_min:
        eps = max(abs(tth_min) * 1e-6, 1e-3)
        tth_max = min(tth_min + eps, max(tth_lo, tth_hi))
    if not bool(preserve_wrapped_phi) and phi_max <= phi_min:
        eps = max(abs(phi_min) * 1e-6, 1e-3)
        phi_max = min(phi_min + eps, max(phi_lo, phi_hi))

    view_state.tth_min_var.set(tth_min)
    view_state.tth_max_var.set(tth_max)
    view_state.phi_min_var.set(phi_min)
    view_state.phi_max_var.set(phi_max)
    _activate_runtime_1d_analysis(bindings.show_1d_var)

    if callable(bindings.schedule_range_update):
        bindings.schedule_range_update()
    _set_status_text(
        bindings,
        (
            "Integration region set: "
            f"2θ=[{tth_min:.2f}, {tth_max:.2f}]°, "
            f"φ=[{phi_min:.2f}, {phi_max:.2f}]°"
        ),
    )
    return True


def update_runtime_integration_region_visuals(
    bindings: IntegrationRangeDragBindings,
    ai: object,
    sim_res2: object,
) -> None:
    """Refresh the current raw/caked integration-region visuals from live bindings."""

    if bool(getattr(bindings.drag_state, "active", False)):
        return

    show_region = _runtime_range_visible(bindings) and _runtime_unscaled_image_present(
        bindings
    )
    if not show_region:
        _set_fast_viewer_overlay_state(
            bindings.drag_state,
            curve_specs=(),
            suppress_overlay_image=False,
        )
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_visible(False)
        return

    view_state = bindings.range_view_state
    if view_state is None:
        _set_fast_viewer_overlay_state(
            bindings.drag_state,
            curve_specs=(),
            suppress_overlay_image=False,
        )
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_visible(False)
        return

    tth_min, tth_max = sorted(
        (float(view_state.tth_min_var.get()), float(view_state.tth_max_var.get()))
    )
    phi_min = float(view_state.phi_min_var.get())
    phi_max = float(view_state.phi_max_var.get())

    if _runtime_caked_view_enabled(bindings) and sim_res2 is not None:
        _set_fast_viewer_overlay_state(
            bindings.drag_state,
            curve_specs=(),
            suppress_overlay_image=False,
        )
        bindings.integration_region_overlay.set_visible(False)
        if phi_max >= phi_min:
            bindings.integration_region_rect.set_xy((tth_min, phi_min))
            bindings.integration_region_rect.set_width(tth_max - tth_min)
            bindings.integration_region_rect.set_height(phi_max - phi_min)
            bindings.integration_region_rect.set_visible(True)
        else:
            bindings.integration_region_rect.set_visible(False)
        return

    _update_detector_integration_overlay(
        bindings,
        ai=ai,
        tth_min=tth_min,
        tth_max=tth_max,
        phi_min=phi_min,
        phi_max=phi_max,
    )


def refresh_runtime_integration_region_visuals(
    bindings: IntegrationRangeDragBindings,
) -> None:
    """Refresh integration visuals using the live AI and cached caked result."""

    if bool(getattr(bindings.drag_state, "active", False)):
        return

    update_runtime_integration_region_visuals(
        bindings,
        _runtime_ai(bindings),
        _runtime_last_sim_res2(bindings),
    )


def reset_runtime_integration_drag(
    bindings: IntegrationRangeDragBindings,
    *,
    redraw: bool = True,
) -> None:
    """Clear the current drag-selection state and hide the drag rectangle."""

    _clear_drag_coordinates(bindings.drag_state)
    bindings.drag_select_rect.set_visible(False)
    bindings.integration_region_rect.set_visible(False)
    bindings.integration_region_overlay.set_visible(False)
    if redraw:
        _draw_idle(bindings)


def handle_runtime_integration_drag_press(
    bindings: IntegrationRangeDragBindings,
    event: Any,
) -> bool:
    """Handle one canvas button-press for integration-range dragging."""

    if bool(bindings.peak_selection_state.suppress_drag_press_once):
        bindings.peak_selection_state.suppress_drag_press_once = False
        _sync_peak_selection_state(bindings)
        return True

    if getattr(event, "button", None) != 1:
        return False
    if (
        getattr(event, "inaxes", None) is not bindings.ax
        or getattr(event, "xdata", None) is None
        or getattr(event, "ydata", None) is None
    ):
        return False
    if not _runtime_unscaled_image_present(bindings):
        _set_status_text(bindings, "Run a simulation first.")
        return False

    drag_state = bindings.drag_state
    x0, y0 = clamp_to_axis_view(bindings.ax, event.xdata, event.ydata)
    if _runtime_caked_view_enabled(bindings):
        drag_state.active = True
        drag_state.mode = "caked"
        drag_state.x0 = x0
        drag_state.y0 = y0
        drag_state.x1 = x0
        drag_state.y1 = y0
        drag_state.tth0 = None
        drag_state.phi0 = None
        drag_state.tth1 = None
        drag_state.phi1 = None
        update_runtime_drag_rectangle(bindings, x0, y0, x0, y0)
        return True

    if bool(bindings.peak_selection_state.hkl_pick_armed):
        return False

    ai = _runtime_ai(bindings)
    if ai is None:
        return False

    angles = display_to_detector_angles(bindings, x0, y0, ai)
    if angles is None:
        return False
    tth0, phi0 = angles
    drag_state.active = True
    drag_state.mode = "raw"
    drag_state.x0 = x0
    drag_state.y0 = y0
    drag_state.x1 = x0
    drag_state.y1 = y0
    drag_state.tth0 = tth0
    drag_state.phi0 = phi0
    drag_state.tth1 = tth0
    drag_state.phi1 = phi0
    update_runtime_raw_drag_preview(bindings, ai)
    return True


def handle_runtime_integration_drag_motion(
    bindings: IntegrationRangeDragBindings,
    event: Any,
) -> bool:
    """Handle one canvas motion event for integration-range dragging."""

    drag_state = bindings.drag_state
    if not bool(drag_state.active):
        return False

    mode = drag_state.mode
    if mode == "caked":
        if not _runtime_caked_view_enabled(bindings):
            return False

        if (
            getattr(event, "inaxes", None) is bindings.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x1, y1 = clamp_to_axis_view(bindings.ax, event.xdata, event.ydata)
            drag_state.x1 = x1
            drag_state.y1 = y1
        elif drag_state.x1 is None or drag_state.y1 is None:
            return False

        update_runtime_drag_rectangle(
            bindings,
            drag_state.x0,
            drag_state.y0,
            drag_state.x1,
            drag_state.y1,
        )
        return True

    if mode != "raw" or _runtime_caked_view_enabled(bindings):
        return False

    ai = _runtime_ai(bindings)
    if ai is None:
        return False

    if (
        getattr(event, "inaxes", None) is bindings.ax
        and getattr(event, "xdata", None) is not None
        and getattr(event, "ydata", None) is not None
    ):
        x1, y1 = clamp_to_axis_view(bindings.ax, event.xdata, event.ydata)
        angles = display_to_detector_angles(bindings, x1, y1, ai)
        if angles is not None:
            tth1, phi1 = angles
            drag_state.x1 = x1
            drag_state.y1 = y1
            drag_state.tth1 = tth1
            drag_state.phi1 = phi1

    update_runtime_raw_drag_preview(bindings, ai)
    return True


def handle_runtime_integration_drag_release(
    bindings: IntegrationRangeDragBindings,
    event: Any,
) -> bool:
    """Handle one canvas button-release for integration-range dragging."""

    if getattr(event, "button", None) != 1:
        return False

    drag_state = bindings.drag_state
    if not bool(drag_state.active):
        return False

    mode = drag_state.mode
    drag_state.active = False

    if mode == "caked":
        applied = False
        if (
            _runtime_caked_view_enabled(bindings)
            and getattr(event, "inaxes", None) is bindings.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x1, y1 = clamp_to_axis_view(bindings.ax, event.xdata, event.ydata)
            drag_state.x1 = x1
            drag_state.y1 = y1

        if None not in (drag_state.x0, drag_state.y0, drag_state.x1, drag_state.y1):
            applied = set_runtime_integration_range_from_drag(
                bindings,
                drag_state.x0,
                drag_state.y0,
                drag_state.x1,
                drag_state.y1,
            )
        reset_runtime_integration_drag(bindings, redraw=False)
        if applied or _runtime_range_visible(bindings):
            update_runtime_integration_region_visuals(
                bindings,
                _runtime_ai(bindings),
                _runtime_last_sim_res2(bindings),
            )
        _draw_idle(bindings)
        return True

    if mode == "raw":
        ai = _runtime_ai(bindings)
        applied = False
        if (
            not _runtime_caked_view_enabled(bindings)
            and ai is not None
            and getattr(event, "inaxes", None) is bindings.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x1, y1 = clamp_to_axis_view(bindings.ax, event.xdata, event.ydata)
            angles = display_to_detector_angles(bindings, x1, y1, ai)
            if angles is not None:
                tth1, phi1 = angles
                drag_state.x1 = x1
                drag_state.y1 = y1
                drag_state.tth1 = tth1
                drag_state.phi1 = phi1

        if None not in (drag_state.tth0, drag_state.phi0, drag_state.tth1, drag_state.phi1):
            applied = set_runtime_integration_range_from_drag(
                bindings,
                drag_state.tth0,
                drag_state.phi0,
                drag_state.tth1,
                drag_state.phi1,
                preserve_wrapped_phi=True,
            )
        reset_runtime_integration_drag(bindings, redraw=False)
        if applied or ai is not None:
            update_runtime_integration_region_visuals(
                bindings,
                ai,
                _runtime_last_sim_res2(bindings),
            )
            _draw_idle(bindings)
        return True

    reset_runtime_integration_drag(bindings)
    return False


def make_runtime_integration_range_drag_bindings_factory(
    *,
    drag_state,
    peak_selection_state,
    range_view_state_factory: object,
    ax: object,
    drag_select_rect: object,
    integration_region_overlay: object,
    integration_region_rect: object,
    image_display: object,
    get_detector_angular_maps: Callable[[Any], tuple[object, object]],
    range_visible_factory: object,
    caked_view_enabled_factory: object,
    unscaled_image_present_factory: object,
    ai_factory: object,
    show_1d_var_factory: object = None,
    sync_peak_selection_state: Callable[[], None] | None = None,
    schedule_range_update_factory: object = None,
    last_sim_res2_factory: object = None,
    draw_idle_factory: object = None,
    set_status_text_factory: object = None,
):
    """Build a factory that resolves the live integration-range drag bindings."""

    def _build() -> IntegrationRangeDragBindings:
        return IntegrationRangeDragBindings(
            drag_state=drag_state,
            peak_selection_state=peak_selection_state,
            range_view_state=_resolve_runtime_value(range_view_state_factory),
            ax=ax,
            drag_select_rect=drag_select_rect,
            integration_region_overlay=integration_region_overlay,
            integration_region_rect=integration_region_rect,
            image_display=image_display,
            get_detector_angular_maps=get_detector_angular_maps,
            range_visible_factory=range_visible_factory,
            caked_view_enabled_factory=caked_view_enabled_factory,
            unscaled_image_present_factory=unscaled_image_present_factory,
            ai_factory=ai_factory,
            show_1d_var=_resolve_runtime_value(show_1d_var_factory),
            sync_peak_selection_state=sync_peak_selection_state,
            schedule_range_update=_resolve_runtime_value(schedule_range_update_factory),
            last_sim_res2_factory=last_sim_res2_factory,
            draw_idle=_resolve_runtime_value(draw_idle_factory),
            set_status_text=_resolve_runtime_value(set_status_text_factory),
        )

    return _build


def make_runtime_integration_range_drag_callbacks(
    bindings_factory: Callable[[], IntegrationRangeDragBindings],
) -> IntegrationRangeDragCallbacks:
    """Build bound callbacks for runtime integration-range dragging."""

    def _safe_runtime_callback(
        handler: Callable[[IntegrationRangeDragBindings, Any], bool],
    ) -> Callable[[Any], bool]:
        def _wrapped(event: Any) -> bool:
            bindings = bindings_factory()
            try:
                return bool(handler(bindings, event))
            except Exception:
                traceback.print_exc()
                try:
                    reset_runtime_integration_drag(bindings, redraw=False)
                except Exception:
                    pass
                _set_status_text(
                    bindings,
                    "Integration drag canceled after an internal error.",
                )
                _draw_idle(bindings)
                return False

        return _wrapped

    return IntegrationRangeDragCallbacks(
        on_press=_safe_runtime_callback(
            handle_runtime_integration_drag_press,
        ),
        on_motion=_safe_runtime_callback(
            handle_runtime_integration_drag_motion,
        ),
        on_release=_safe_runtime_callback(
            handle_runtime_integration_drag_release,
        ),
        reset=lambda: reset_runtime_integration_drag(bindings_factory()),
    )


def make_runtime_integration_region_visuals_callback(
    bindings_factory: Callable[[], IntegrationRangeDragBindings],
) -> Callable[[], None]:
    """Return a zero-arg callback that refreshes live integration visuals."""

    return lambda: refresh_runtime_integration_region_visuals(bindings_factory())


def make_runtime_integration_range_update_bindings_factory(
    *,
    root: Any,
    simulation_runtime_state: Any,
    analysis_view_state_factory: object,
    range_view_state_factory: object,
    display_controls_state: Any,
    hkl_lookup_controls_factory: object = None,
    integration_range_drag_callbacks_factory: object = None,
    refresh_integration_from_cached_results_factory: object = None,
    refresh_display_from_controls_factory: object = None,
    schedule_update_factory: object = None,
    range_update_debounce_ms_factory: object = 120,
) -> Callable[[], IntegrationRangeUpdateBindings]:
    """Build a factory that resolves the live integration-range update bindings."""

    def _build() -> IntegrationRangeUpdateBindings:
        debounce_ms = _resolve_runtime_value(range_update_debounce_ms_factory)
        try:
            normalized_debounce_ms = int(debounce_ms)
        except Exception:
            normalized_debounce_ms = 120
        return IntegrationRangeUpdateBindings(
            root=root,
            simulation_runtime_state=simulation_runtime_state,
            analysis_view_state=_resolve_runtime_value(analysis_view_state_factory),
            range_view_state=_resolve_runtime_value(range_view_state_factory),
            display_controls_state=display_controls_state,
            hkl_lookup_controls=_resolve_runtime_value(hkl_lookup_controls_factory),
            integration_range_drag_callbacks=_resolve_runtime_value(
                integration_range_drag_callbacks_factory
            ),
            refresh_integration_from_cached_results=_resolve_runtime_value(
                refresh_integration_from_cached_results_factory
            ),
            refresh_display_from_controls=_resolve_runtime_value(
                refresh_display_from_controls_factory
            ),
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            range_update_debounce_ms=normalized_debounce_ms,
        )

    return _build


def _run_runtime_scheduled_range_update(
    bindings_factory: Callable[[], IntegrationRangeUpdateBindings],
) -> None:
    bindings = bindings_factory()
    bindings.simulation_runtime_state.integration_update_pending = None

    if bool(getattr(bindings.simulation_runtime_state, "update_running", False)):
        _schedule_runtime_range_update(
            bindings_factory,
            delay_ms=_runtime_range_update_debounce_ms(bindings),
        )
        return

    refresh = bindings.refresh_integration_from_cached_results
    if callable(refresh) and refresh():
        return

    if callable(bindings.schedule_update):
        bindings.schedule_update()


def _schedule_runtime_range_update(
    bindings_factory: Callable[[], IntegrationRangeUpdateBindings],
    *,
    delay_ms: object | None = None,
) -> None:
    bindings = bindings_factory()
    pending = getattr(bindings.simulation_runtime_state, "integration_update_pending", None)
    if pending is not None:
        gui_controllers.clear_tk_after_token(bindings.root, pending)

    base_delay = _runtime_range_update_debounce_ms(bindings)
    try:
        requested_delay = int(base_delay if delay_ms is None else delay_ms)
    except Exception:
        requested_delay = base_delay
    delay = max(base_delay, requested_delay)

    after = getattr(bindings.root, "after", None)
    if not callable(after):
        _run_runtime_scheduled_range_update(bindings_factory)
        return

    bindings.simulation_runtime_state.integration_update_pending = after(
        delay,
        lambda: _run_runtime_scheduled_range_update(bindings_factory),
    )


def _sync_runtime_range_text_vars(view_state: object) -> None:
    specs = (
        (
            getattr(view_state, "tth_min_var", None),
            getattr(view_state, "tth_min_label_var", None),
            getattr(view_state, "tth_min_entry_var", None),
        ),
        (
            getattr(view_state, "tth_max_var", None),
            getattr(view_state, "tth_max_label_var", None),
            getattr(view_state, "tth_max_entry_var", None),
        ),
        (
            getattr(view_state, "phi_min_var", None),
            getattr(view_state, "phi_min_label_var", None),
            getattr(view_state, "phi_min_entry_var", None),
        ),
        (
            getattr(view_state, "phi_max_var", None),
            getattr(view_state, "phi_max_label_var", None),
            getattr(view_state, "phi_max_entry_var", None),
        ),
    )
    if any(
        value_var is None or label_var is None or entry_var is None
        for value_var, label_var, entry_var in specs
    ):
        return
    for value_var, label_var, entry_var in specs:
        value = _safe_var_get(value_var)
        try:
            numeric_value = float(value)
        except Exception:
            continue
        _safe_var_set(label_var, f"{numeric_value:.1f}")
        _safe_var_set(entry_var, f"{numeric_value:.4f}")


def _apply_runtime_range_entry(
    *,
    view_state: object,
    entry_var: object,
    value_var: object,
    slider: object,
    show_1d_var: object,
    schedule_range_update: Callable[..., object] | None,
) -> None:
    entry_value = _safe_var_get(entry_var)
    try:
        entered = float(str(entry_value).strip())
    except Exception:
        _sync_runtime_range_text_vars(view_state)
        return

    try:
        lo = float(slider.cget("from"))
        hi = float(slider.cget("to"))
    except Exception:
        _sync_runtime_range_text_vars(view_state)
        return

    clamped = min(max(entered, min(lo, hi)), max(lo, hi))
    _safe_var_set(value_var, clamped)
    _activate_runtime_1d_analysis(show_1d_var)
    _sync_runtime_range_text_vars(view_state)
    if callable(schedule_range_update):
        schedule_range_update()


def _make_runtime_range_slider_callback(
    *,
    view_state: object,
    value_var_name: str,
    show_1d_var: object,
    schedule_range_update: Callable[..., object] | None,
) -> Callable[[object], None]:
    def _on_changed(value: object) -> None:
        value_var = getattr(view_state, value_var_name, None)
        if value_var is None:
            return
        try:
            numeric_value = float(value)
        except Exception:
            return
        _safe_var_set(value_var, numeric_value)
        _activate_runtime_1d_analysis(show_1d_var)
        _sync_runtime_range_text_vars(view_state)
        if callable(schedule_range_update):
            schedule_range_update()

    return _on_changed


def create_runtime_integration_range_controls(
    *,
    parent: Any,
    views_module: Any,
    view_state: Any,
    show_1d_var: object,
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
    schedule_range_update: Callable[..., object] | None,
) -> None:
    """Create the 1D integration-range controls and wire runtime callbacks."""

    views_module.create_integration_range_controls(
        parent=parent,
        view_state=view_state,
        tth_min=tth_min,
        tth_max=tth_max,
        phi_min=phi_min,
        phi_max=phi_max,
        on_tth_min_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="tth_min_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
        ),
        on_tth_max_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="tth_max_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
        ),
        on_phi_min_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="phi_min_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
        ),
        on_phi_max_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="phi_max_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
        ),
        on_apply_entry=lambda entry_var, value_var, slider: _apply_runtime_range_entry(
            view_state=view_state,
            entry_var=entry_var,
            value_var=value_var,
            slider=slider,
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
        ),
    )

    refs = (
        getattr(view_state, "tth_min_var", None),
        getattr(view_state, "tth_max_var", None),
        getattr(view_state, "phi_min_var", None),
        getattr(view_state, "phi_max_var", None),
        getattr(view_state, "tth_min_slider", None),
        getattr(view_state, "tth_max_slider", None),
        getattr(view_state, "phi_min_slider", None),
        getattr(view_state, "phi_max_slider", None),
    )
    if any(ref is None for ref in refs):
        raise RuntimeError("Integration-range controls did not create the expected widgets.")

    for value_var in refs[:4]:
        _safe_var_trace_add(
            value_var,
            lambda *_args, _view_state=view_state: _sync_runtime_range_text_vars(
                _view_state
            ),
        )
    _sync_runtime_range_text_vars(view_state)


def _toggle_runtime_1d_plots(
    callbacks: IntegrationRangeUpdateCallbacks,
) -> None:
    callbacks.schedule_range_update()


def _toggle_runtime_caked_2d(
    bindings_factory: Callable[[], IntegrationRangeUpdateBindings],
) -> None:
    bindings = bindings_factory()
    show_caked_2d_var = getattr(bindings.analysis_view_state, "show_caked_2d_var", None)
    show_caked = bool(_safe_var_get(show_caked_2d_var))

    if not show_caked:
        bindings.simulation_runtime_state.caked_limits_user_override = False
    else:
        bindings.display_controls_state.simulation_limits_user_override = False
        hkl_lookup_controls = bindings.hkl_lookup_controls
        set_pick_mode = getattr(hkl_lookup_controls, "set_hkl_pick_mode", None)
        if callable(set_pick_mode):
            set_pick_mode(False)

    drag_callbacks = bindings.integration_range_drag_callbacks
    reset_drag = getattr(drag_callbacks, "reset", None)
    if callable(reset_drag):
        reset_drag()

    if callable(bindings.schedule_update):
        bindings.schedule_update()


def _toggle_runtime_log_display(
    bindings_factory: Callable[[], IntegrationRangeUpdateBindings],
) -> None:
    bindings = bindings_factory()
    refresh_display = getattr(bindings, "refresh_display_from_controls", None)
    if callable(refresh_display):
        refresh_display()


def make_runtime_integration_range_update_callbacks(
    bindings_factory: Callable[[], IntegrationRangeUpdateBindings],
) -> IntegrationRangeUpdateCallbacks:
    """Build bound callbacks for integration-range update scheduling and toggles."""

    def _schedule_range_update(*, delay_ms: object | None = None) -> None:
        _schedule_runtime_range_update(bindings_factory, delay_ms=delay_ms)

    callbacks = IntegrationRangeUpdateCallbacks(
        schedule_range_update=_schedule_range_update,
        toggle_1d_plots=lambda: None,
        toggle_caked_2d=lambda: _toggle_runtime_caked_2d(bindings_factory),
        toggle_log_display=lambda: _toggle_runtime_log_display(bindings_factory),
    )
    return IntegrationRangeUpdateCallbacks(
        schedule_range_update=callbacks.schedule_range_update,
        toggle_1d_plots=lambda: _toggle_runtime_1d_plots(callbacks),
        toggle_caked_2d=callbacks.toggle_caked_2d,
        toggle_log_display=callbacks.toggle_log_display,
    )
