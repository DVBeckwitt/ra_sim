"""Runtime canvas event orchestration for cross-feature GUI interactions."""

from __future__ import annotations

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
    set_geometry_status_text: Callable[[str], None] | None = None
    draw_idle: Callable[[], None] | None = None


@dataclass(frozen=True)
class CanvasInteractionCallbacks:
    """Bound runtime callbacks for canvas click/drag event wiring."""

    on_click: Callable[[Any], bool]
    on_press: Callable[[Any], bool]
    on_motion: Callable[[Any], bool]
    on_release: Callable[[Any], bool]


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
            setter = getattr(bindings.peak_selection_callbacks, "set_hkl_pick_mode", None)
            if callable(setter):
                setter(False, "HKL image-pick canceled.")
            return True
        if bool(bindings.geometry_runtime_state.manual_pick_armed):
            _set_mode(
                bindings.set_geometry_manual_pick_mode,
                False,
                "Manual geometry picking canceled.",
            )
            return True
        if bool(bindings.geometry_preview_state.exclude_armed):
            _set_mode(
                bindings.set_geometry_preview_exclude_mode,
                False,
                "Preview exclusion mode canceled.",
            )
            return True
        return False

    if getattr(event, "button", None) != 1:
        return False

    if bool(bindings.geometry_runtime_state.manual_pick_armed):
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        if bool(bindings.manual_pick_session_active()):
            return True
        bindings.toggle_geometry_manual_selection_at(
            float(event.xdata),
            float(event.ydata),
        )
        return True

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

    if not bool(bindings.peak_selection_state.hkl_pick_armed):
        return False

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


def handle_runtime_canvas_press(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas button-press."""

    if bool(bindings.peak_selection_state.suppress_drag_press_once):
        bindings.peak_selection_state.suppress_drag_press_once = False
        return True

    if bool(bindings.geometry_runtime_state.manual_pick_armed) and bool(
        bindings.manual_pick_session_active()
    ):
        if getattr(event, "button", None) != 1:
            return False
        if (
            getattr(event, "inaxes", None) is not bindings.axis
            or getattr(event, "xdata", None) is None
            or getattr(event, "ydata", None) is None
        ):
            return False
        x0, y0 = bindings.clamp_to_axis_view(
            bindings.axis,
            event.xdata,
            event.ydata,
        )
        anchor_fraction_x, anchor_fraction_y = manual_pick_zoom_anchor_fractions(
            bindings.axis,
            event,
        )
        bindings.apply_geometry_manual_pick_zoom(
            x0,
            y0,
            anchor_fraction_x=anchor_fraction_x,
            anchor_fraction_y=anchor_fraction_y,
        )
        bindings.update_geometry_manual_pick_preview(x0, y0, force=True)
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

    on_press = getattr(bindings.integration_range_drag_callbacks, "on_press", None)
    if not callable(on_press):
        return False
    return bool(on_press(event))


def handle_runtime_canvas_motion(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas motion event."""

    if (
        bool(bindings.geometry_runtime_state.manual_pick_armed)
        and bool(bindings.manual_pick_session_active())
        and bool(bindings.geometry_manual_state.pick_session.get("zoom_active", False))
    ):
        if (
            getattr(event, "inaxes", None) is bindings.axis
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x1, y1 = bindings.clamp_to_axis_view(
                bindings.axis,
                event.xdata,
                event.ydata,
            )
            bindings.update_geometry_manual_pick_preview(x1, y1)
        return True

    on_motion = getattr(bindings.integration_range_drag_callbacks, "on_motion", None)
    if not callable(on_motion):
        return False
    return bool(on_motion(event))


def handle_runtime_canvas_release(
    bindings: CanvasInteractionBindings,
    event: Any,
) -> bool:
    """Handle one runtime canvas button-release."""

    if getattr(event, "button", None) != 1:
        return False

    if (
        bool(bindings.geometry_runtime_state.manual_pick_armed)
        and bool(bindings.manual_pick_session_active())
        and bool(bindings.geometry_manual_state.pick_session.get("zoom_active", False))
    ):
        if (
            getattr(event, "inaxes", None) is bindings.axis
            and getattr(event, "xdata", None) is not None
            and getattr(event, "ydata", None) is not None
        ):
            x_sel, y_sel = bindings.clamp_to_axis_view(
                bindings.axis,
                event.xdata,
                event.ydata,
            )
            bindings.place_geometry_manual_selection_at(float(x_sel), float(y_sel))
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

    on_release = getattr(bindings.integration_range_drag_callbacks, "on_release", None)
    if not callable(on_release):
        return False
    return bool(on_release(event))


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
    set_geometry_status_text_factory: object | None = None,
    draw_idle_factory: object | None = None,
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
            set_geometry_status_text=_resolve_runtime_value(
                set_geometry_status_text_factory
            ),
            draw_idle=_resolve_runtime_value(draw_idle_factory),
        )

    return _build_bindings


def make_runtime_canvas_interaction_callbacks(
    bindings_factory: Callable[[], CanvasInteractionBindings],
) -> CanvasInteractionCallbacks:
    """Return bound callbacks for runtime canvas event wiring."""

    return CanvasInteractionCallbacks(
        on_click=lambda event: handle_runtime_canvas_click(bindings_factory(), event),
        on_press=lambda event: handle_runtime_canvas_press(bindings_factory(), event),
        on_motion=lambda event: handle_runtime_canvas_motion(bindings_factory(), event),
        on_release=lambda event: handle_runtime_canvas_release(bindings_factory(), event),
    )
