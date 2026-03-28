"""Workflow helpers for canvas drag-selection of 1D integration ranges."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.patches import Rectangle


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
    schedule_update: Callable[[], None] | None = None
    range_update_debounce_ms: int = 120


@dataclass(frozen=True)
class IntegrationRangeUpdateCallbacks:
    """Bound callbacks for integration-range update and analysis toggles."""

    schedule_range_update: Callable[..., None]
    toggle_1d_plots: Callable[[], None]
    toggle_caked_2d: Callable[[], None]
    toggle_log_radial: Callable[[], None]
    toggle_log_azimuth: Callable[[], None]


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
        linewidth=1.8,
        edgecolor="yellow",
        facecolor="none",
        linestyle="-",
        zorder=6,
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
        linewidth=2.0,
        edgecolor="cyan",
        facecolor="none",
        linestyle="--",
        zorder=5,
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
    bindings.drag_select_rect.set_xy((x_min, y_min))
    bindings.drag_select_rect.set_width(x_max - x_min)
    bindings.drag_select_rect.set_height(y_max - y_min)
    bindings.drag_select_rect.set_visible(True)
    _draw_idle(bindings)


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
    """Refresh the raw-detector integration overlay for the current drag."""

    drag_state = bindings.drag_state
    if None in (drag_state.tth0, drag_state.phi0, drag_state.tth1, drag_state.phi1):
        return False

    two_theta, phi_vals = bindings.get_detector_angular_maps(ai)
    if two_theta is None or phi_vals is None:
        return False

    tth_min, tth_max = sorted((float(drag_state.tth0), float(drag_state.tth1)))
    phi_min, phi_max = sorted((float(drag_state.phi0), float(drag_state.phi1)))
    mask = (
        (two_theta >= tth_min)
        & (two_theta <= tth_max)
        & (phi_vals >= phi_min)
        & (phi_vals <= phi_max)
    )

    bindings.drag_select_rect.set_visible(False)
    bindings.integration_region_rect.set_visible(False)
    if np.any(mask):
        bindings.integration_region_overlay.set_data(mask.astype(float))
        bindings.integration_region_overlay.set_extent(bindings.image_display.get_extent())
        bindings.integration_region_overlay.set_visible(True)
    else:
        bindings.integration_region_overlay.set_visible(False)
    _draw_idle(bindings)
    return True


def set_runtime_integration_range_from_drag(
    bindings: IntegrationRangeDragBindings,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    """Apply one completed drag to the integration-range controls."""

    view_state = bindings.range_view_state
    if view_state is None:
        return False

    tth_min, tth_max = sorted((float(x0), float(x1)))
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
    if phi_max <= phi_min:
        eps = max(abs(phi_min) * 1e-6, 1e-3)
        phi_max = min(phi_min + eps, max(phi_lo, phi_hi))

    view_state.tth_min_var.set(tth_min)
    view_state.tth_max_var.set(tth_max)
    view_state.phi_min_var.set(phi_min)
    view_state.phi_max_var.set(phi_max)

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

    show_region = _runtime_range_visible(bindings) and _runtime_unscaled_image_present(
        bindings
    )
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
        (float(view_state.tth_min_var.get()), float(view_state.tth_max_var.get()))
    )
    phi_min, phi_max = sorted(
        (float(view_state.phi_min_var.get()), float(view_state.phi_max_var.get()))
    )

    if _runtime_caked_view_enabled(bindings) and sim_res2 is not None:
        bindings.integration_region_overlay.set_visible(False)
        bindings.integration_region_rect.set_xy((tth_min, phi_min))
        bindings.integration_region_rect.set_width(tth_max - tth_min)
        bindings.integration_region_rect.set_height(phi_max - phi_min)
        bindings.integration_region_rect.set_visible(True)
        return

    bindings.integration_region_rect.set_visible(False)
    if ai is None:
        bindings.integration_region_overlay.set_visible(False)
        return

    two_theta, phi_vals = bindings.get_detector_angular_maps(ai)
    if two_theta is None or phi_vals is None:
        bindings.integration_region_overlay.set_visible(False)
        return

    mask = (
        (two_theta >= tth_min)
        & (two_theta <= tth_max)
        & (phi_vals >= phi_min)
        & (phi_vals <= phi_max)
    )
    if not np.any(mask):
        bindings.integration_region_overlay.set_visible(False)
        return

    bindings.integration_region_overlay.set_data(mask.astype(float))
    bindings.integration_region_overlay.set_extent(bindings.image_display.get_extent())
    bindings.integration_region_overlay.set_visible(True)


def refresh_runtime_integration_region_visuals(
    bindings: IntegrationRangeDragBindings,
) -> None:
    """Refresh integration visuals using the live AI and cached caked result."""

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
            set_runtime_integration_range_from_drag(
                bindings,
                drag_state.x0,
                drag_state.y0,
                drag_state.x1,
                drag_state.y1,
            )
        reset_runtime_integration_drag(bindings)
        return True

    if mode == "raw":
        ai = _runtime_ai(bindings)
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
            set_runtime_integration_range_from_drag(
                bindings,
                drag_state.tth0,
                drag_state.phi0,
                drag_state.tth1,
                drag_state.phi1,
            )
        elif ai is not None:
            update_runtime_integration_region_visuals(
                bindings,
                ai,
                _runtime_last_sim_res2(bindings),
            )
            _draw_idle(bindings)

        reset_runtime_integration_drag(bindings)
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

    return IntegrationRangeDragCallbacks(
        on_press=lambda event: handle_runtime_integration_drag_press(
            bindings_factory(),
            event,
        ),
        on_motion=lambda event: handle_runtime_integration_drag_motion(
            bindings_factory(),
            event,
        ),
        on_release=lambda event: handle_runtime_integration_drag_release(
            bindings_factory(),
            event,
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
    after_cancel = getattr(bindings.root, "after_cancel", None)
    if pending is not None and callable(after_cancel):
        after_cancel(pending)

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
    _sync_runtime_range_text_vars(view_state)
    if callable(schedule_range_update):
        schedule_range_update()


def _make_runtime_range_slider_callback(
    *,
    view_state: object,
    value_var_name: str,
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
        _sync_runtime_range_text_vars(view_state)
        if callable(schedule_range_update):
            schedule_range_update()

    return _on_changed


def create_runtime_integration_range_controls(
    *,
    parent: Any,
    views_module: Any,
    view_state: Any,
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
            schedule_range_update=schedule_range_update,
        ),
        on_tth_max_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="tth_max_var",
            schedule_range_update=schedule_range_update,
        ),
        on_phi_min_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="phi_min_var",
            schedule_range_update=schedule_range_update,
        ),
        on_phi_max_changed=_make_runtime_range_slider_callback(
            view_state=view_state,
            value_var_name="phi_max_var",
            schedule_range_update=schedule_range_update,
        ),
        on_apply_entry=lambda entry_var, value_var, slider: _apply_runtime_range_entry(
            view_state=view_state,
            entry_var=entry_var,
            value_var=value_var,
            slider=slider,
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
        toggle_log_radial=lambda: None,
        toggle_log_azimuth=lambda: None,
    )
    return IntegrationRangeUpdateCallbacks(
        schedule_range_update=callbacks.schedule_range_update,
        toggle_1d_plots=lambda: _toggle_runtime_1d_plots(callbacks),
        toggle_caked_2d=callbacks.toggle_caked_2d,
        toggle_log_radial=lambda: _toggle_runtime_1d_plots(callbacks),
        toggle_log_azimuth=lambda: _toggle_runtime_1d_plots(callbacks),
    )
