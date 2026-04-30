"""Workflow helpers for canvas drag-selection of 1D integration ranges."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.patches import Rectangle
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui import qr_cylinder_overlay as gui_qr_cylinder_overlay
from ra_sim.simulation.exact_cake_portable import raw_phi_to_gui_phi

_ACTIVE_DRAG_EDGE_COLOR = "#ff7a00"
_ACTIVE_DRAG_FACE_RGBA = (1.0, 0.48, 0.0, 0.14)
_ACTIVE_DRAG_LINEWIDTH = 2.8

_SELECTED_REGION_EDGE_COLOR = "#ff00a8"
_SELECTED_REGION_FACE_RGBA = (1.0, 0.0, 0.66, 0.10)
_SELECTED_REGION_OVERLAY_RGBA = (1.0, 0.0, 0.66, 0.46)
_SELECTED_REGION_LINEWIDTH = 3.0

_DEFAULT_TTH_MIN = 0.0
_DEFAULT_TTH_MAX = 80.0
_DEFAULT_PHI_MIN = -15.0
_DEFAULT_PHI_MAX = 15.0
_DEFAULT_QZ_MIN = 0.0
_DEFAULT_QZ_MAX = 5.0
_DEFAULT_DELTA_QR = 0.25
_TTH_SLIDER_BOUNDS = (0.0, 90.0)
_PHI_SLIDER_BOUNDS = (-180.0, 180.0)
_DELTA_QR_SLIDER_BOUNDS = (0.0, 1.0)
_DISPLAY_COORD_EPSILON = 1.0e-9
_RUNTIME_RANGE_TEXT_FORMATS: dict[str, tuple[str, str]] = {
    "tth_min": ("{:.1f}", "{:.4f}"),
    "tth_max": ("{:.1f}", "{:.4f}"),
    "phi_min": ("{:.1f}", "{:.4f}"),
    "phi_max": ("{:.1f}", "{:.4f}"),
    "qz_min": ("{:.4f}", "{:.4f}"),
    "qz_max": ("{:.4f}", "{:.4f}"),
    "delta_qr": ("{:.4f}", "{:.4f}"),
}


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
    set_integration_overlay_image: Callable[..., None] | None = None
    caked_custom_mask_factory: object = None
    caked_custom_mask_signature_factory: object = None
    detector_geometry_signature_factory: object = None
    caked_qr_rod_drag_context_factory: object = None


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
    toggle_caked_2d: Callable[[], None] | None = None
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


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _set_widget_enabled(widget: object, enabled: bool) -> None:
    if widget is None:
        return
    state_value = "normal" if bool(enabled) else "disabled"
    state_method = getattr(widget, "state", None)
    if callable(state_method):
        try:
            state_method(["!disabled"] if bool(enabled) else ["disabled"])
            return
        except Exception:
            pass
    configure = getattr(widget, "configure", None)
    if callable(configure):
        try:
            configure(state=state_value)
            return
        except Exception:
            pass
    config = getattr(widget, "config", None)
    if callable(config):
        try:
            config(state=state_value)
        except Exception:
            pass


def _get_runtime_range_value(
    view_state: object,
    prefix: str,
    fallback: float,
) -> float:
    value_var = getattr(view_state, f"{prefix}_var", None)
    numeric_value = _coerce_float(_safe_var_get(value_var))
    if numeric_value is not None:
        setattr(view_state, f"{prefix}_value", numeric_value)
        return numeric_value

    numeric_value = _coerce_float(getattr(view_state, f"{prefix}_value", None))
    if numeric_value is not None:
        return numeric_value

    numeric_value = float(fallback)
    setattr(view_state, f"{prefix}_value", numeric_value)
    return numeric_value


def _set_runtime_range_value(
    view_state: object,
    prefix: str,
    value: float,
) -> float:
    numeric_value = float(value)
    setattr(view_state, f"{prefix}_value", numeric_value)
    _safe_var_set(getattr(view_state, f"{prefix}_var", None), numeric_value)
    return numeric_value


def _get_runtime_string_value(
    view_state: object,
    prefix: str,
    fallback: str = "",
) -> str:
    value_var = getattr(view_state, f"{prefix}_var", None)
    value = _safe_var_get(value_var)
    if value is not None:
        text = str(value)
        setattr(view_state, f"{prefix}_value", text)
        return text

    cached_value = getattr(view_state, f"{prefix}_value", None)
    if cached_value is not None:
        return str(cached_value)

    text = str(fallback)
    setattr(view_state, f"{prefix}_value", text)
    return text


def _set_runtime_string_value(
    view_state: object,
    prefix: str,
    value: object,
) -> str:
    text = str(value)
    setattr(view_state, f"{prefix}_value", text)
    _safe_var_set(getattr(view_state, f"{prefix}_var", None), text)
    return text


def _runtime_range_boolean(
    view_state: object,
    prefix: str,
    fallback: bool = False,
) -> bool:
    value_var = getattr(view_state, f"{prefix}_var", None)
    value = _safe_var_get(value_var)
    if value is not None:
        normalized = bool(value)
        setattr(view_state, f"{prefix}_value", normalized)
        return normalized

    cached_value = getattr(view_state, f"{prefix}_value", None)
    if cached_value is not None:
        normalized = bool(cached_value)
        setattr(view_state, f"{prefix}_value", normalized)
        return normalized

    normalized = bool(fallback)
    setattr(view_state, f"{prefix}_value", normalized)
    return normalized


def _get_runtime_slider_bounds(
    view_state: object,
    prefix: str,
    fallback_lo: float,
    fallback_hi: float,
) -> tuple[float, float]:
    slider = getattr(view_state, f"{prefix}_slider", None)
    if slider is None:
        return float(fallback_lo), float(fallback_hi)

    try:
        lower = float(slider.cget("from"))
        upper = float(slider.cget("to"))
    except Exception:
        return float(fallback_lo), float(fallback_hi)
    return lower, upper


def _set_runtime_slider_bounds(
    view_state: object,
    prefix: str,
    lower_bound: float,
    upper_bound: float,
) -> None:
    slider = getattr(view_state, f"{prefix}_slider", None)
    if slider is None:
        return
    lower = float(lower_bound)
    upper = float(upper_bound)
    if upper < lower:
        lower, upper = upper, lower
    try:
        slider.configure(from_=lower, to=upper)
    except Exception:
        pass


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


def _set_integration_overlay_image(
    bindings: IntegrationRangeDragBindings,
    image: object,
    *,
    source_signature: object | None = None,
) -> None:
    if callable(bindings.set_integration_overlay_image):
        if source_signature is None:
            bindings.set_integration_overlay_image(image)
        else:
            bindings.set_integration_overlay_image(
                image,
                source_signature=source_signature,
            )
        return
    bindings.integration_region_overlay.set_data(image)
    extent = _detector_display_extent(bindings)
    if extent is None:
        getter = getattr(bindings.image_display, "get_extent", None)
        if callable(getter):
            try:
                extent = getter()
            except Exception:
                extent = None
    if extent is not None:
        bindings.integration_region_overlay.set_extent(extent)


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


def _runtime_caked_custom_mask(bindings: IntegrationRangeDragBindings):
    resolved = _resolve_runtime_value(bindings.caked_custom_mask_factory)
    if resolved is None:
        return None
    try:
        mask = np.asarray(resolved, dtype=bool)
    except Exception:
        return None
    return mask if mask.ndim == 2 else None


def _runtime_caked_custom_mask_signature(
    bindings: IntegrationRangeDragBindings,
) -> object | None:
    return _resolve_runtime_value(getattr(bindings, "caked_custom_mask_signature_factory", None))


def _runtime_detector_geometry_signature(
    bindings: IntegrationRangeDragBindings,
) -> object | None:
    return _resolve_runtime_value(getattr(bindings, "detector_geometry_signature_factory", None))


def _rounded_overlay_bounds_signature(
    *,
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
) -> tuple[float, float, float, float]:
    return tuple(round(float(value), 10) for value in (tth_min, tth_max, phi_min, phi_max))


def _detector_overlay_source_signature(
    *,
    detector_shape: tuple[int, int],
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
    detector_geometry_signature: object | None,
) -> tuple[
    str,
    tuple[int, int],
    tuple[float, float, float, float],
    object | None,
]:
    return (
        "detector_integration_overlay",
        tuple(int(value) for value in detector_shape),
        _rounded_overlay_bounds_signature(
            tth_min=tth_min,
            tth_max=tth_max,
            phi_min=phi_min,
            phi_max=phi_max,
        ),
        detector_geometry_signature,
    )


def _caked_overlay_source_signature(
    *,
    rect_mask_shape: tuple[int, int],
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
    custom_mask_signature: object | None,
) -> tuple[str, tuple[int, int], tuple[float, float, float, float], object | None]:
    return (
        "caked_integration_overlay",
        tuple(int(value) for value in rect_mask_shape),
        _rounded_overlay_bounds_signature(
            tth_min=tth_min,
            tth_max=tth_max,
            phi_min=phi_min,
            phi_max=phi_max,
        ),
        custom_mask_signature,
    )


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


def _detector_display_extent(
    bindings: IntegrationRangeDragBindings,
) -> tuple[float, float, float, float] | None:
    image_display = getattr(bindings, "image_display", None)
    extent = getattr(image_display, "_ra_sim_source_extent", None)
    if extent is None:
        getter = getattr(image_display, "get_extent", None)
        if not callable(getter):
            return None
        try:
            extent = getter()
        except Exception:
            return None
    try:
        extent = np.asarray(extent, dtype=float).reshape(-1)
    except Exception:
        return None
    if extent.size < 4 or not np.all(np.isfinite(extent[:4])):
        return None
    return tuple(float(value) for value in extent[:4])


def _display_axis_values_to_detector_indices(
    values: object,
    *,
    axis_start: float,
    axis_end: float,
    size: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if int(size) <= 0:
        return array
    span = float(axis_end) - float(axis_start)
    if not np.isfinite(span) or abs(span) <= _DISPLAY_COORD_EPSILON:
        return array
    indices = ((array - float(axis_start)) / span) * float(size) - 0.5
    return np.clip(indices, 0.0, float(size - 1))


def _display_coords_to_detector_indices(
    bindings: IntegrationRangeDragBindings,
    *,
    x_values: object,
    y_values: object,
    detector_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    extent = _detector_display_extent(bindings)
    if extent is None:
        return x_array, y_array
    height, width = (int(detector_shape[0]), int(detector_shape[1]))
    return (
        _display_axis_values_to_detector_indices(
            x_array,
            axis_start=float(extent[0]),
            axis_end=float(extent[1]),
            size=width,
        ),
        _display_axis_values_to_detector_indices(
            y_array,
            axis_start=float(extent[2]),
            axis_end=float(extent[3]),
            size=height,
        ),
    )


def _detector_index_to_display_axis_value(
    index_value: float,
    *,
    axis_start: float,
    axis_end: float,
    size: int,
) -> float | None:
    if int(size) <= 0 or not np.isfinite(index_value):
        return None
    span = float(axis_end) - float(axis_start)
    if not np.isfinite(span) or abs(span) <= _DISPLAY_COORD_EPSILON:
        return None
    world_value = float(axis_start) + ((float(index_value) + 0.5) / float(size)) * span
    return float(world_value) if np.isfinite(world_value) else None


def _detector_indices_to_display_coords(
    bindings: IntegrationRangeDragBindings,
    *,
    col_idx: float,
    row_idx: float,
    detector_shape: tuple[int, int],
) -> tuple[float, float] | None:
    extent = _detector_display_extent(bindings)
    if extent is None:
        if not (np.isfinite(col_idx) and np.isfinite(row_idx)):
            return None
        return float(col_idx), float(row_idx)
    height, width = (int(detector_shape[0]), int(detector_shape[1]))
    col_value = _detector_index_to_display_axis_value(
        float(col_idx),
        axis_start=float(extent[0]),
        axis_end=float(extent[1]),
        size=width,
    )
    row_value = _detector_index_to_display_axis_value(
        float(row_idx),
        axis_start=float(extent[2]),
        axis_end=float(extent[3]),
        size=height,
    )
    if col_value is None or row_value is None:
        return None
    return float(col_value), float(row_value)


def _detector_preview_center(
    bindings: IntegrationRangeDragBindings,
    two_theta: object,
) -> tuple[float, float] | None:
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
    display_coords = _detector_indices_to_display_coords(
        bindings,
        col_idx=float(col_idx),
        row_idx=float(row_idx),
        detector_shape=tuple(int(value) for value in array.shape[:2]),
    )
    if display_coords is None:
        return None
    return float(display_coords[0]), float(display_coords[1])


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


def _next_drag_preview_source_signature(
    drag_state: object,
    *,
    detector_shape: tuple[int, int],
) -> tuple[str, tuple[int, int], int]:
    revision = getattr(drag_state, "_raw_drag_preview_revision", None)
    try:
        next_revision = int(revision) + 1
    except Exception:
        next_revision = 1
    setattr(drag_state, "_raw_drag_preview_revision", next_revision)
    return ("raw_drag_preview", tuple(int(value) for value in detector_shape), next_revision)


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
    x_array = x_array[:point_count]
    y_array = y_array[:point_count]
    finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
    if not np.any(finite_mask):
        return
    x_clipped = np.clip(x_array[finite_mask], 0, int(preview.shape[1]) - 1)
    y_clipped = np.clip(y_array[finite_mask], 0, int(preview.shape[0]) - 1)
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
        center = _detector_preview_center(bindings, two_theta)
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
    angle_values = np.linspace(
        float(angle_start), float(angle_start) + float(span_degrees), arc_samples
    )
    angle_radians = np.deg2rad(angle_values)

    outer_x = float(center_x) + float(radius_max) * np.cos(angle_radians)
    outer_y = float(center_y) + float(radius_max) * np.sin(angle_radians)
    inner_x = float(center_x) + float(radius_min) * np.cos(angle_radians)
    inner_y = float(center_y) + float(radius_min) * np.sin(angle_radians)
    radial_values = np.linspace(float(radius_min), float(radius_max), edge_samples)
    start_radians = float(np.deg2rad(angle_start))
    end_radians = float(np.deg2rad(angle_start + span_degrees))

    outer_col_idx, outer_row_idx = _display_coords_to_detector_indices(
        bindings,
        x_values=outer_x,
        y_values=outer_y,
        detector_shape=detector_shape,
    )
    inner_col_idx, inner_row_idx = _display_coords_to_detector_indices(
        bindings,
        x_values=inner_x,
        y_values=inner_y,
        detector_shape=detector_shape,
    )
    start_x = float(center_x) + radial_values * np.cos(start_radians)
    start_y = float(center_y) + radial_values * np.sin(start_radians)
    end_x = float(center_x) + radial_values * np.cos(end_radians)
    end_y = float(center_y) + radial_values * np.sin(end_radians)
    start_col_idx, start_row_idx = _display_coords_to_detector_indices(
        bindings,
        x_values=start_x,
        y_values=start_y,
        detector_shape=detector_shape,
    )
    end_col_idx, end_row_idx = _display_coords_to_detector_indices(
        bindings,
        x_values=end_x,
        y_values=end_y,
        detector_shape=detector_shape,
    )

    _stamp_preview_points(preview, x_values=outer_col_idx, y_values=outer_row_idx)
    _stamp_preview_points(preview, x_values=inner_col_idx, y_values=inner_row_idx)
    _stamp_preview_points(
        preview,
        x_values=start_col_idx,
        y_values=start_row_idx,
    )
    _stamp_preview_points(
        preview,
        x_values=end_col_idx,
        y_values=end_row_idx,
    )

    if not np.any(preview):
        return False

    bindings.integration_region_rect.set_visible(False)
    bindings.drag_select_rect.set_visible(False)
    _set_integration_overlay_image(
        bindings,
        preview,
        source_signature=_next_drag_preview_source_signature(
            drag_state,
            detector_shape=detector_shape,
        ),
    )
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

    _set_integration_overlay_image(
        bindings,
        mask.astype(float),
        source_signature=_detector_overlay_source_signature(
            detector_shape=tuple(int(value) for value in mask.shape),
            tth_min=tth_min,
            tth_max=tth_max,
            phi_min=phi_min,
            phi_max=phi_max,
            detector_geometry_signature=_runtime_detector_geometry_signature(bindings),
        ),
    )
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

    col_idx_array, row_idx_array = _display_coords_to_detector_indices(
        bindings,
        x_values=(float(col),),
        y_values=(float(row),),
        detector_shape=(int(height), int(width)),
    )
    if col_idx_array.size <= 0 or row_idx_array.size <= 0:
        return None

    col_idx = min(max(int(round(float(col_idx_array.reshape(-1)[0]))), 0), int(width) - 1)
    row_idx = min(max(int(round(float(row_idx_array.reshape(-1)[0]))), 0), int(height) - 1)
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


def qz_bounds_from_caked_drag_for_qr_rod(
    projected_samples,
    *,
    x0,
    y0,
    x1,
    y1,
    phi_min: float | None = None,
    phi_max: float | None = None,
    phi_windows: Sequence[tuple[object, object]] | None = None,
) -> tuple[float, float] | None:
    """Return Qz bounds selected by a caked drag across projected Qr-rod samples."""

    try:
        two_theta = np.asarray(projected_samples.get("two_theta"), dtype=float).reshape(-1)
        phi = np.asarray(projected_samples.get("phi"), dtype=float).reshape(-1)
        qz = np.asarray(projected_samples.get("qz"), dtype=float).reshape(-1)
    except Exception:
        return None
    if two_theta.size <= 0 or two_theta.shape != phi.shape or two_theta.shape != qz.shape:
        return None

    try:
        x_lo, x_hi = sorted((float(x0), float(x1)))
        y_lo, y_hi = sorted((float(y0), float(y1)))
    except Exception:
        return None
    if not all(np.isfinite(value) for value in (x_lo, x_hi, y_lo, y_hi)):
        return None

    if phi_windows is not None:
        normalized_phi_windows = gui_qr_cylinder_overlay.normalize_caked_phi_windows(
            phi_windows=phi_windows,
        )
        if normalized_phi_windows is None:
            return None
        phi_window = gui_qr_cylinder_overlay.caked_phi_window_mask(
            phi,
            normalized_phi_windows,
        )
    elif phi_min is not None and phi_max is not None:
        normalized_phi_windows = gui_qr_cylinder_overlay.normalize_caked_phi_windows(
            phi_min=phi_min,
            phi_max=phi_max,
        )
        if normalized_phi_windows is None:
            return None
        phi_window = gui_qr_cylinder_overlay.caked_phi_window_mask(
            phi,
            normalized_phi_windows,
        )
    else:
        phi_window = np.ones(phi.shape, dtype=bool)

    selected = (
        np.isfinite(two_theta)
        & np.isfinite(phi)
        & np.isfinite(qz)
        & phi_window
        & (two_theta >= x_lo)
        & (two_theta <= x_hi)
        & (phi >= y_lo)
        & (phi <= y_hi)
    )
    if not np.any(selected):
        return None
    selected_qz = qz[selected]
    return float(np.min(selected_qz)), float(np.max(selected_qz))


def qz_bounds_from_caked_drag_for_qr_rod_bins(
    *,
    selected_entry: Mapping[str, object],
    config: gui_qr_cylinder_overlay.QrCylinderOverlayRenderConfig,
    projection_context: Mapping[str, object] | None,
    radial_axis: object,
    azimuth_axis: object,
    delta_qr: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    phi_windows: Sequence[tuple[object, object]] | None = None,
) -> tuple[float, float] | None:
    """Return Qz bounds selected by caked bins using LUT-transpose detector support."""

    try:
        qr0 = float(selected_entry["qr"])
        delta_qr_value = float(delta_qr)
        radial_values = np.asarray(radial_axis, dtype=float).reshape(-1)
        azimuth_values = np.asarray(azimuth_axis, dtype=float).reshape(-1)
        x_lo, x_hi = sorted((float(x0), float(x1)))
        y_lo, y_hi = sorted((float(y0), float(y1)))
    except Exception:
        return None
    if (
        not np.isfinite(qr0)
        or qr0 < 0.0
        or not np.isfinite(delta_qr_value)
        or delta_qr_value <= 0.0
        or radial_values.size <= 0
        or azimuth_values.size <= 0
        or not all(np.isfinite(value) for value in (x_lo, x_hi, y_lo, y_hi))
        or not np.all(np.isfinite(radial_values))
        or not np.all(np.isfinite(azimuth_values))
    ):
        return None

    if phi_windows is not None:
        normalized_phi_windows = gui_qr_cylinder_overlay.normalize_caked_phi_windows(
            phi_windows=phi_windows,
        )
        if normalized_phi_windows is None:
            return None
        phi_window_mask = gui_qr_cylinder_overlay.caked_phi_window_mask(
            azimuth_values,
            normalized_phi_windows,
        )
    else:
        phi_window_mask = np.ones(azimuth_values.shape, dtype=bool)

    radial_mask = (radial_values >= x_lo) & (radial_values <= x_hi)
    phi_drag_mask = (azimuth_values >= y_lo) & (azimuth_values <= y_hi)
    azimuth_mask = phi_drag_mask & phi_window_mask
    if not np.any(radial_mask) or not np.any(azimuth_mask):
        return None

    resolved_projection = gui_qr_cylinder_overlay._resolve_caked_projection_context(  # noqa: SLF001
        projection_context,
    )
    if resolved_projection is None:
        return None
    detector_shape = tuple(int(v) for v in resolved_projection["detector_shape"])
    bundle = resolved_projection["transform_bundle"]
    lut_orientation = gui_qr_cylinder_overlay._resolve_detector_to_caked_lut(  # noqa: SLF001
        bundle,
        detector_shape=detector_shape,
        n_radial=int(radial_values.size),
        n_azimuth=int(azimuth_values.size),
    )
    if lut_orientation is None:
        return None
    _detector_to_caked, caked_to_detector, _lut_signature = lut_orientation

    display_dragged = np.asarray(
        np.outer(azimuth_mask, radial_mask),
        dtype=np.float32,
    )
    raw_to_gui = np.asarray(
        resolved_projection["raw_to_gui_row_permutation"],
        dtype=np.int64,
    ).reshape(-1)
    if raw_to_gui.size != azimuth_values.size:
        return None
    raw_dragged = np.zeros_like(display_dragged, dtype=np.float32)
    raw_dragged[raw_to_gui, :] = display_dragged

    detector_size = int(detector_shape[0]) * int(detector_shape[1])
    detector_weight = gui_qr_cylinder_overlay._matrix_vector_product(  # noqa: SLF001
        caked_to_detector,
        raw_dragged.reshape(-1),
        expected_size=detector_size,
    )
    if detector_weight is None:
        return None
    finite_weight = detector_weight[np.isfinite(detector_weight)]
    if finite_weight.size <= 0:
        return None
    max_weight = float(np.max(finite_weight))
    if not np.isfinite(max_weight) or max_weight <= 0.0:
        return None
    weight_threshold = max(
        gui_qr_cylinder_overlay._SELECTED_QR_ROD_LUT_ABS_EPS,  # noqa: SLF001
        gui_qr_cylinder_overlay._SELECTED_QR_ROD_LUT_REL_EPS * max_weight,  # noqa: SLF001
    )
    detector_support = (detector_weight > weight_threshold).reshape(detector_shape)
    if not np.any(detector_support):
        return None

    q_maps = gui_qr_cylinder_overlay.detector_qr_qz_maps_for_projection(
        config=config,
        detector_shape=detector_shape,
    )
    if q_maps is None:
        return None
    qr_map, qz_map, valid_q = q_maps
    if (
        qr_map.shape != detector_shape
        or qz_map.shape != detector_shape
        or valid_q.shape != detector_shape
    ):
        return None

    qr_lo = max(0.0, qr0 - delta_qr_value)
    qr_hi = qr0 + delta_qr_value
    selected = (
        detector_support
        & np.asarray(valid_q, dtype=bool)
        & np.isfinite(qr_map)
        & np.isfinite(qz_map)
        & (qr_map >= qr_lo)
        & (qr_map <= qr_hi)
    )
    if not np.any(selected):
        return None
    selected_qz = np.asarray(qz_map, dtype=float)[selected]
    return float(np.min(selected_qz)), float(np.max(selected_qz))


def _runtime_qr_rod_drag_context(
    bindings: IntegrationRangeDragBindings,
) -> Mapping[str, object] | None:
    context = _resolve_runtime_value(getattr(bindings, "caked_qr_rod_drag_context_factory", None))
    return context if isinstance(context, Mapping) else None


def _projected_qr_rod_samples_from_context(
    context: Mapping[str, object],
) -> Mapping[str, object] | None:
    projected_samples = context.get("projected_samples")
    if isinstance(projected_samples, Mapping):
        return projected_samples
    try:
        selected_entry = context["selected_entry"]
        config = context["config"]
        projection_context = context["projection_context"]
    except Exception:
        return None
    return gui_qr_cylinder_overlay.project_selected_qr_rod_caked_samples(
        selected_entry=selected_entry,
        config=config,
        projection_context=projection_context,
    )


def _runtime_selected_qr_phi_windows(
    view_state: object,
) -> tuple[tuple[float, float], ...] | None:
    phi_min = _get_runtime_range_value(view_state, "phi_min", _DEFAULT_PHI_MIN)
    phi_max = _get_runtime_range_value(view_state, "phi_max", _DEFAULT_PHI_MAX)
    mirror = _runtime_range_boolean(
        view_state,
        "integrate_selected_qr_rod",
        False,
    ) and _runtime_range_boolean(view_state, "mirror_selected_qr_phi", False)

    if mirror:
        return gui_qr_cylinder_overlay.mirrored_abs_phi_windows(phi_min, phi_max)

    return gui_qr_cylinder_overlay.normalize_caked_phi_windows(
        phi_min=phi_min,
        phi_max=phi_max,
    )


def update_runtime_qr_rod_drag_preview(bindings: IntegrationRangeDragBindings) -> bool:
    """Refresh the selected-Qr rod drag preview as a temporary Q-space mask."""

    drag_state = bindings.drag_state
    if None in (drag_state.x0, drag_state.y0, drag_state.x1, drag_state.y1):
        return False

    context = _runtime_qr_rod_drag_context(bindings)
    if context is None:
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_visible(False)
        bindings.drag_select_rect.set_visible(False)
        _draw_idle(bindings)
        return False

    view_state = bindings.range_view_state
    if view_state is None:
        return False
    delta_qr = _get_runtime_range_value(view_state, "delta_qr", _DEFAULT_DELTA_QR)
    phi_min = _get_runtime_range_value(view_state, "phi_min", _DEFAULT_PHI_MIN)
    phi_max = _get_runtime_range_value(view_state, "phi_max", _DEFAULT_PHI_MAX)
    phi_windows = _runtime_selected_qr_phi_windows(view_state)
    if phi_windows is None:
        return False
    try:
        selected_entry = context["selected_entry"]
        config = context["config"]
        projection_context = context["projection_context"]
        radial_axis = np.asarray(context["radial_axis"], dtype=float).reshape(-1)
        azimuth_axis = np.asarray(context["azimuth_axis"], dtype=float).reshape(-1)
    except Exception:
        return False

    qz_bounds = qz_bounds_from_caked_drag_for_qr_rod_bins(
        selected_entry=selected_entry,
        config=config,
        projection_context=projection_context,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        delta_qr=delta_qr,
        x0=drag_state.x0,
        x1=drag_state.x1,
        y0=drag_state.y0,
        y1=drag_state.y1,
        phi_windows=phi_windows,
    )
    if qz_bounds is None:
        projected_samples = _projected_qr_rod_samples_from_context(context)
        if projected_samples is None:
            bindings.integration_region_overlay.set_visible(False)
            bindings.integration_region_rect.set_visible(False)
            bindings.drag_select_rect.set_visible(False)
            _draw_idle(bindings)
            return False
        qz_bounds = qz_bounds_from_caked_drag_for_qr_rod(
            projected_samples,
            x0=drag_state.x0,
            y0=drag_state.y0,
            x1=drag_state.x1,
            y1=drag_state.y1,
            phi_min=phi_min,
            phi_max=phi_max,
            phi_windows=phi_windows,
        )
    if qz_bounds is None:
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_visible(False)
        bindings.drag_select_rect.set_visible(False)
        _draw_idle(bindings)
        return False

    qz_lo, qz_hi = qz_bounds
    try:
        mask_payload = gui_qr_cylinder_overlay.build_selected_qr_rod_qz_caked_mask(
            selected_entry=selected_entry,
            config=config,
            projection_context=projection_context,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            qz_min=qz_lo,
            qz_max=qz_hi,
            phi_min=phi_min,
            phi_max=phi_max,
            phi_windows=phi_windows,
            delta_qr=delta_qr,
        )
    except Exception:
        return False
    if not isinstance(mask_payload, Mapping):
        return False
    mask = np.asarray(mask_payload.get("mask"), dtype=bool)

    if not np.any(mask):
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_visible(False)
        bindings.drag_select_rect.set_visible(False)
        _draw_idle(bindings)
        return False

    _set_integration_overlay_image(
        bindings,
        mask.astype(float),
        source_signature=(
            "caked_selected_qr_rod_drag_preview",
            mask_payload.get("signature"),
        ),
    )
    bindings.integration_region_overlay.set_visible(True)
    bindings.integration_region_rect.set_visible(False)
    bindings.drag_select_rect.set_visible(False)
    _draw_idle(bindings)
    return True


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

    tth_lo, tth_hi = _get_runtime_slider_bounds(
        view_state,
        "tth_min",
        *_TTH_SLIDER_BOUNDS,
    )
    phi_lo, phi_hi = _get_runtime_slider_bounds(
        view_state,
        "phi_min",
        *_PHI_SLIDER_BOUNDS,
    )
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

    tth_min = _set_runtime_range_value(view_state, "tth_min", tth_min)
    tth_max = _set_runtime_range_value(view_state, "tth_max", tth_max)
    phi_min = _set_runtime_range_value(view_state, "phi_min", phi_min)
    phi_max = _set_runtime_range_value(view_state, "phi_max", phi_max)
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


def _caked_rect_mask(
    sim_res2: object,
    *,
    tth_min: float,
    tth_max: float,
    phi_min: float,
    phi_max: float,
) -> np.ndarray | None:
    radial_axis = np.asarray(getattr(sim_res2, "radial", ()), dtype=float).reshape(-1)
    raw_azimuth_axis = np.asarray(getattr(sim_res2, "azimuthal", ()), dtype=float).reshape(-1)
    if radial_axis.size <= 0 or raw_azimuth_axis.size <= 0:
        return None

    radial_mask = (radial_axis >= 0.0) & (radial_axis <= 90.0)
    if not np.any(radial_mask):
        return None

    gui_azimuth = np.asarray(raw_phi_to_gui_phi(raw_azimuth_axis), dtype=float)
    order = np.argsort(gui_azimuth, kind="stable")
    gui_azimuth = gui_azimuth[order]
    radial_axis = radial_axis[radial_mask]
    azimuth_mask = detector_phi_mask(gui_azimuth, float(phi_min), float(phi_max))
    radial_clip_mask = (radial_axis >= float(tth_min)) & (radial_axis <= float(tth_max))
    return np.asarray(np.outer(azimuth_mask, radial_clip_mask), dtype=bool)


def _validated_runtime_caked_custom_mask(
    bindings: IntegrationRangeDragBindings,
    sim_res2: object,
    *,
    require_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    custom_mask = _runtime_caked_custom_mask(bindings)
    if custom_mask is None:
        return None

    caked_reference_res2 = sim_res2 if sim_res2 is not None else _runtime_last_sim_res2(bindings)
    if caked_reference_res2 is None:
        return None

    radial_axis = np.asarray(getattr(caked_reference_res2, "radial", ()), dtype=float).reshape(-1)
    raw_azimuth_axis = np.asarray(
        getattr(caked_reference_res2, "azimuthal", ()),
        dtype=float,
    ).reshape(-1)
    if radial_axis.size <= 0 or raw_azimuth_axis.size <= 0:
        return None
    radial_mask = (radial_axis >= 0.0) & (radial_axis <= 90.0)
    if not np.any(radial_mask):
        return None
    expected_shape = (
        int(raw_azimuth_axis.size),
        int(np.count_nonzero(radial_mask)),
    )
    if require_shape is not None:
        expected_shape = tuple(int(value) for value in require_shape)
    return custom_mask if tuple(custom_mask.shape) == tuple(expected_shape) else None


def update_runtime_integration_region_visuals(
    bindings: IntegrationRangeDragBindings,
    ai: object,
    sim_res2: object,
) -> None:
    """Refresh the current raw/caked integration-region visuals from live bindings."""

    if bool(getattr(bindings.drag_state, "active", False)):
        return

    show_region = _runtime_range_visible(bindings) and _runtime_unscaled_image_present(bindings)
    if not show_region:
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_visible(False)
        return

    view_state = bindings.range_view_state
    if view_state is None:
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_visible(False)
        return

    tth_min, tth_max = sorted(
        (
            _get_runtime_range_value(view_state, "tth_min", _DEFAULT_TTH_MIN),
            _get_runtime_range_value(view_state, "tth_max", _DEFAULT_TTH_MAX),
        )
    )
    phi_min = _get_runtime_range_value(view_state, "phi_min", _DEFAULT_PHI_MIN)
    phi_max = _get_runtime_range_value(view_state, "phi_max", _DEFAULT_PHI_MAX)
    rod_mode_enabled = _runtime_range_boolean(
        view_state,
        "integrate_selected_qr_rod",
        False,
    )

    if _runtime_caked_view_enabled(bindings) and sim_res2 is not None:
        if rod_mode_enabled:
            custom_mask = _validated_runtime_caked_custom_mask(bindings, sim_res2)
            if custom_mask is not None and np.any(custom_mask):
                _set_integration_overlay_image(
                    bindings,
                    custom_mask.astype(float),
                    source_signature=(
                        "caked_selected_qr_rod_overlay",
                        tuple(int(value) for value in custom_mask.shape),
                        _runtime_caked_custom_mask_signature(bindings),
                    ),
                )
                bindings.integration_region_overlay.set_visible(True)
                bindings.integration_region_rect.set_visible(False)
            else:
                bindings.integration_region_overlay.set_visible(False)
                bindings.integration_region_rect.set_visible(False)
        else:
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
        if _runtime_range_boolean(
            bindings.range_view_state,
            "integrate_selected_qr_rod",
            False,
        ):
            drag_state.active = True
            drag_state.mode = "caked_qr_rod"
            drag_state.x0 = x0
            drag_state.y0 = y0
            drag_state.x1 = x0
            drag_state.y1 = y0
            drag_state.tth0 = None
            drag_state.phi0 = None
            drag_state.tth1 = None
            drag_state.phi1 = None
            update_runtime_qr_rod_drag_preview(bindings)
            return True
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

    if mode == "caked_qr_rod":
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

        update_runtime_qr_rod_drag_preview(bindings)
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

    if mode == "caked_qr_rod":
        if (
            _runtime_caked_view_enabled(bindings)
            and getattr(event, "inaxes", None) is bindings.ax
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x1, y1 = clamp_to_axis_view(bindings.ax, event.xdata, event.ydata)
            drag_state.x1 = x1
            drag_state.y1 = y1

        context = _runtime_qr_rod_drag_context(bindings)
        qz_bounds = None
        if (
            context is not None
            and None not in (drag_state.x0, drag_state.y0, drag_state.x1, drag_state.y1)
        ):
            try:
                phi_min = _get_runtime_range_value(
                    bindings.range_view_state,
                    "phi_min",
                    _DEFAULT_PHI_MIN,
                )
                phi_max = _get_runtime_range_value(
                    bindings.range_view_state,
                    "phi_max",
                    _DEFAULT_PHI_MAX,
                )
                phi_windows = _runtime_selected_qr_phi_windows(bindings.range_view_state)
                delta_qr = _get_runtime_range_value(
                    bindings.range_view_state,
                    "delta_qr",
                    _DEFAULT_DELTA_QR,
                )
                qz_bounds = qz_bounds_from_caked_drag_for_qr_rod_bins(
                    selected_entry=context["selected_entry"],
                    config=context["config"],
                    projection_context=context["projection_context"],
                    radial_axis=context["radial_axis"],
                    azimuth_axis=context["azimuth_axis"],
                    delta_qr=delta_qr,
                    x0=drag_state.x0,
                    x1=drag_state.x1,
                    y0=drag_state.y0,
                    y1=drag_state.y1,
                    phi_windows=phi_windows,
                )
                if qz_bounds is None:
                    projected_samples = _projected_qr_rod_samples_from_context(context)
                    qz_bounds = qz_bounds_from_caked_drag_for_qr_rod(
                        projected_samples,
                        x0=drag_state.x0,
                        y0=drag_state.y0,
                        x1=drag_state.x1,
                        y1=drag_state.y1,
                        phi_min=phi_min,
                        phi_max=phi_max,
                        phi_windows=phi_windows,
                    )
            except Exception:
                qz_bounds = None

        reset_runtime_integration_drag(bindings, redraw=False)
        if qz_bounds is None or bindings.range_view_state is None:
            update_runtime_integration_region_visuals(
                bindings,
                _runtime_ai(bindings),
                _runtime_last_sim_res2(bindings),
            )
            _set_status_text(bindings, "Drag across the selected Qr rod to set a Qz range.")
            _draw_idle(bindings)
            return True

        qz_lo, qz_hi = sorted((float(qz_bounds[0]), float(qz_bounds[1])))
        _set_runtime_range_value(bindings.range_view_state, "qz_min", qz_lo)
        _set_runtime_range_value(bindings.range_view_state, "qz_max", qz_hi)
        _activate_runtime_1d_analysis(bindings.show_1d_var)
        _sync_runtime_range_text_vars(bindings.range_view_state)
        if callable(bindings.schedule_range_update):
            bindings.schedule_range_update()
        _set_status_text(
            bindings,
            f"Selected Qr rod Qz range set: Qz=[{qz_lo:.4f}, {qz_hi:.4f}] A^-1",
        )
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
    set_integration_overlay_image_factory: object = None,
    caked_custom_mask_factory: object = None,
    caked_custom_mask_signature_factory: object = None,
    detector_geometry_signature_factory: object = None,
    caked_qr_rod_drag_context_factory: object = None,
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
            set_integration_overlay_image=_resolve_runtime_value(
                set_integration_overlay_image_factory
            ),
            caked_custom_mask_factory=caked_custom_mask_factory,
            caked_custom_mask_signature_factory=caked_custom_mask_signature_factory,
            detector_geometry_signature_factory=detector_geometry_signature_factory,
            caked_qr_rod_drag_context_factory=caked_qr_rod_drag_context_factory,
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
    toggle_caked_2d_factory: object = None,
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
            toggle_caked_2d=_resolve_runtime_value(toggle_caked_2d_factory),
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
    specs = []
    for prefix, (label_format, entry_format) in _RUNTIME_RANGE_TEXT_FORMATS.items():
        specs.append(
            (
                prefix,
                getattr(view_state, f"{prefix}_var", None),
                getattr(view_state, f"{prefix}_label_var", None),
                getattr(view_state, f"{prefix}_entry_var", None),
                label_format,
                entry_format,
            )
        )
    for prefix, value_var, label_var, entry_var, label_format, entry_format in specs:
        if value_var is None or label_var is None or entry_var is None:
            continue
        value = _safe_var_get(value_var)
        try:
            numeric_value = float(value)
        except Exception:
            continue
        setattr(view_state, f"{prefix}_value", numeric_value)
        _safe_var_set(label_var, label_format.format(numeric_value))
        _safe_var_set(entry_var, entry_format.format(numeric_value))


def _sync_runtime_selected_qr_rod_mode_state(view_state: object) -> None:
    rod_mode_enabled = _runtime_range_boolean(view_state, "integrate_selected_qr_rod", False)

    for widget_name in (
        "tth_min_slider",
        "tth_min_entry",
        "tth_max_slider",
        "tth_max_entry",
    ):
        _set_widget_enabled(getattr(view_state, widget_name, None), not rod_mode_enabled)

    for widget_name in (
        "phi_min_slider",
        "phi_min_entry",
        "phi_max_slider",
        "phi_max_entry",
    ):
        _set_widget_enabled(getattr(view_state, widget_name, None), True)

    for widget_name in (
        "selected_qr_rod_combobox",
        "mirror_selected_qr_phi_checkbutton",
        "qz_min_slider",
        "qz_min_entry",
        "qz_max_slider",
        "qz_max_entry",
        "delta_qr_slider",
        "delta_qr_entry",
    ):
        _set_widget_enabled(getattr(view_state, widget_name, None), rod_mode_enabled)


def _refresh_runtime_region_visuals(
    refresh_region_visuals: Callable[[], object] | None,
) -> None:
    if not callable(refresh_region_visuals):
        return
    try:
        refresh_region_visuals()
    except Exception:
        pass


def _apply_runtime_range_entry(
    *,
    view_state: object,
    entry_var: object,
    value_var: object,
    slider: object,
    show_1d_var: object,
    schedule_range_update: Callable[..., object] | None,
    refresh_region_visuals: Callable[[], object] | None = None,
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
    _refresh_runtime_region_visuals(refresh_region_visuals)


def _make_runtime_range_slider_callback(
    *,
    view_state: object,
    value_var_name: str,
    show_1d_var: object,
    schedule_range_update: Callable[..., object] | None,
    refresh_region_visuals: Callable[[], object] | None = None,
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
        _refresh_runtime_region_visuals(refresh_region_visuals)

    return _on_changed


def _set_runtime_selected_qr_rod_options(
    view_state: object,
    options: list[tuple[str, str]],
) -> None:
    labels = [str(label) for _key, label in options]
    keys = [str(key) for key, _label in options]
    label_by_key = {str(key): str(label) for key, label in options}
    key_by_label = {str(label): str(key) for key, label in options}

    setattr(view_state, "selected_qr_rod_options", keys)
    setattr(view_state, "selected_qr_rod_option_labels", label_by_key)
    setattr(view_state, "selected_qr_rod_key_by_label", key_by_label)

    combobox = getattr(view_state, "selected_qr_rod_combobox", None)
    if combobox is not None:
        try:
            combobox.configure(
                values=labels,
                state=("readonly" if labels else "disabled"),
            )
        except Exception:
            pass

    current_key = _get_runtime_string_value(view_state, "selected_qr_rod_key", "")
    current_label = label_by_key.get(current_key, "")
    _safe_var_set(getattr(view_state, "selected_qr_rod_display_var", None), current_label)


def _toggle_runtime_integrate_selected_qr_rod(
    *,
    view_state: object,
    show_1d_var: object,
    schedule_range_update: Callable[..., object] | None,
    disable_peak_pick: Callable[[], object] | None = None,
    refresh_region_visuals: Callable[[], object] | None = None,
) -> None:
    _activate_runtime_1d_analysis(show_1d_var)
    if _runtime_range_boolean(view_state, "integrate_selected_qr_rod", False):
        if callable(disable_peak_pick):
            disable_peak_pick()
    _sync_runtime_selected_qr_rod_mode_state(view_state)
    _sync_runtime_range_text_vars(view_state)
    if callable(schedule_range_update):
        schedule_range_update()
    _refresh_runtime_region_visuals(refresh_region_visuals)


def _toggle_runtime_mirror_selected_qr_phi(
    *,
    view_state: object,
    show_1d_var: object,
    schedule_range_update: Callable[..., object] | None,
    refresh_region_visuals: Callable[[], object] | None = None,
) -> None:
    _activate_runtime_1d_analysis(show_1d_var)
    _sync_runtime_selected_qr_rod_mode_state(view_state)
    if callable(schedule_range_update):
        schedule_range_update()
    _refresh_runtime_region_visuals(refresh_region_visuals)


def _select_runtime_selected_qr_rod(
    *,
    view_state: object,
    display_value: object,
    show_1d_var: object,
    schedule_range_update: Callable[..., object] | None,
    refresh_region_visuals: Callable[[], object] | None = None,
) -> None:
    label = str(display_value or "")
    encoded_key = (getattr(view_state, "selected_qr_rod_key_by_label", {}) or {}).get(label, "")
    _safe_var_set(getattr(view_state, "selected_qr_rod_display_var", None), label)
    _set_runtime_string_value(view_state, "selected_qr_rod_key", encoded_key)
    _activate_runtime_1d_analysis(show_1d_var)
    if callable(schedule_range_update):
        schedule_range_update()
    _refresh_runtime_region_visuals(refresh_region_visuals)


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
    integrate_selected_qr_rod: bool = False,
    mirror_selected_qr_phi: bool = False,
    selected_qr_rod_key: str = "",
    selected_qr_rod_options: list[tuple[str, str]] | None = None,
    qz_min: float = _DEFAULT_QZ_MIN,
    qz_max: float = _DEFAULT_QZ_MAX,
    delta_qr: float = _DEFAULT_DELTA_QR,
    schedule_range_update: Callable[..., object] | None,
    disable_peak_pick: Callable[[], object] | None = None,
    refresh_region_visuals: Callable[[], object] | None = None,
) -> None:
    """Create the 1D integration-range controls and wire runtime callbacks."""

    _set_runtime_range_value(view_state, "tth_min", float(tth_min))
    _set_runtime_range_value(view_state, "tth_max", float(tth_max))
    _set_runtime_range_value(view_state, "phi_min", float(phi_min))
    _set_runtime_range_value(view_state, "phi_max", float(phi_max))
    setattr(view_state, "integrate_selected_qr_rod_value", bool(integrate_selected_qr_rod))
    _safe_var_set(
        getattr(view_state, "integrate_selected_qr_rod_var", None),
        bool(integrate_selected_qr_rod),
    )
    setattr(view_state, "mirror_selected_qr_phi_value", bool(mirror_selected_qr_phi))
    _safe_var_set(
        getattr(view_state, "mirror_selected_qr_phi_var", None),
        bool(mirror_selected_qr_phi),
    )
    _set_runtime_string_value(view_state, "selected_qr_rod_key", str(selected_qr_rod_key))
    _set_runtime_range_value(view_state, "qz_min", float(qz_min))
    _set_runtime_range_value(view_state, "qz_max", float(qz_max))
    _set_runtime_range_value(view_state, "delta_qr", float(delta_qr))

    views_module.create_integration_range_controls(
        parent=parent,
        view_state=view_state,
        tth_min=tth_min,
        tth_max=tth_max,
        phi_min=phi_min,
        phi_max=phi_max,
        qz_min=float(qz_min),
        qz_max=float(qz_max),
        delta_qr=float(delta_qr),
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
            refresh_region_visuals=refresh_region_visuals,
        ),
        on_phi_max_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="phi_max_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            refresh_region_visuals=refresh_region_visuals,
        ),
        on_qz_min_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="qz_min_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            refresh_region_visuals=refresh_region_visuals,
        ),
        on_qz_max_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="qz_max_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            refresh_region_visuals=refresh_region_visuals,
        ),
        on_delta_qr_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="delta_qr_var",
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            refresh_region_visuals=refresh_region_visuals,
        ),
        integrate_selected_qr_rod=bool(integrate_selected_qr_rod),
        mirror_selected_qr_phi=bool(mirror_selected_qr_phi),
        selected_qr_rod_key=str(selected_qr_rod_key),
        selected_qr_rod_options=list(selected_qr_rod_options or ()),
        on_toggle_integrate_selected_qr_rod=lambda: _toggle_runtime_integrate_selected_qr_rod(
            view_state=view_state,
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            disable_peak_pick=disable_peak_pick,
            refresh_region_visuals=refresh_region_visuals,
        ),
        on_toggle_mirror_selected_qr_phi=lambda: _toggle_runtime_mirror_selected_qr_phi(
            view_state=view_state,
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            refresh_region_visuals=refresh_region_visuals,
        ),
        on_selected_qr_rod_changed=lambda value: _select_runtime_selected_qr_rod(
            view_state=view_state,
            display_value=value,
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            refresh_region_visuals=refresh_region_visuals,
        ),
        on_apply_entry=lambda entry_var, value_var, slider: _apply_runtime_range_entry(
            view_state=view_state,
            entry_var=entry_var,
            value_var=value_var,
            slider=slider,
            show_1d_var=show_1d_var,
            schedule_range_update=schedule_range_update,
            refresh_region_visuals=(
                refresh_region_visuals
                if value_var
                in (
                    getattr(view_state, "phi_min_var", None),
                    getattr(view_state, "phi_max_var", None),
                    getattr(view_state, "qz_min_var", None),
                    getattr(view_state, "qz_max_var", None),
                    getattr(view_state, "delta_qr_var", None),
                )
                else None
            ),
        ),
    )

    refs = (
        getattr(view_state, "tth_min_var", None),
        getattr(view_state, "tth_max_var", None),
        getattr(view_state, "phi_min_var", None),
        getattr(view_state, "phi_max_var", None),
        getattr(view_state, "integrate_selected_qr_rod_var", None),
        getattr(view_state, "mirror_selected_qr_phi_var", None),
        getattr(view_state, "selected_qr_rod_key_var", None),
        getattr(view_state, "selected_qr_rod_display_var", None),
        getattr(view_state, "qz_min_var", None),
        getattr(view_state, "qz_max_var", None),
        getattr(view_state, "delta_qr_var", None),
        getattr(view_state, "tth_min_slider", None),
        getattr(view_state, "tth_max_slider", None),
        getattr(view_state, "phi_min_slider", None),
        getattr(view_state, "phi_max_slider", None),
        getattr(view_state, "selected_qr_rod_combobox", None),
        getattr(view_state, "mirror_selected_qr_phi_checkbutton", None),
        getattr(view_state, "qz_min_slider", None),
        getattr(view_state, "qz_max_slider", None),
        getattr(view_state, "delta_qr_slider", None),
        getattr(view_state, "delta_qr_entry", None),
    )
    if any(ref is None for ref in refs):
        raise RuntimeError("Integration-range controls did not create the expected widgets.")

    for value_var in (
        getattr(view_state, "tth_min_var", None),
        getattr(view_state, "tth_max_var", None),
        getattr(view_state, "phi_min_var", None),
        getattr(view_state, "phi_max_var", None),
        getattr(view_state, "qz_min_var", None),
        getattr(view_state, "qz_max_var", None),
        getattr(view_state, "delta_qr_var", None),
    ):
        if value_var is None:
            continue
        _safe_var_trace_add(
            value_var,
            lambda *_args, _view_state=view_state: _sync_runtime_range_text_vars(_view_state),
        )
    _safe_var_trace_add(
        getattr(view_state, "integrate_selected_qr_rod_var", None),
        lambda *_args, _view_state=view_state: _sync_runtime_selected_qr_rod_mode_state(
            _view_state
        ),
    )
    _set_runtime_selected_qr_rod_options(view_state, list(selected_qr_rod_options or ()))
    _sync_runtime_range_text_vars(view_state)
    _sync_runtime_selected_qr_rod_mode_state(view_state)


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

    if callable(bindings.toggle_caked_2d):
        bindings.toggle_caked_2d()
        return

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
