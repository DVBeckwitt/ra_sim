"""Import-safe helpers for fast-viewer and redraw acceleration workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def parse_fast_viewer_env_flag(raw_value: object) -> bool:
    """Return whether one environment value enables the fast viewer."""

    return str(raw_value or "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


@dataclass(frozen=True)
class RuntimeFastViewerWorkflow:
    """Bound helpers for the optional fast-viewer runtime surface."""

    active: Callable[[], bool]
    requested_enabled: Callable[[], bool]
    suspend_reason: Callable[[], str | None]
    set_requested_enabled: Callable[[bool], None]
    refresh_runtime_mode: Callable[..., bool]
    request_main_canvas_redraw: Callable[..., None]
    request_overlay_canvas_redraw: Callable[..., None]
    refresh_status_text: Callable[[], None]
    toggle: Callable[[], None]
    shutdown: Callable[[], None]


def _resolve_runtime_value(value_or_factory: object) -> object:
    if callable(value_or_factory):
        try:
            return value_or_factory()
        except Exception:
            return None
    return value_or_factory


def build_runtime_fast_viewer_workflow(
    *,
    fast_plot_viewer_module,
    tk_module,
    ttk_module,
    canvas_frame,
    matplotlib_canvas,
    ax,
    image_artist,
    background_artist,
    overlay_artist,
    marker_artist_factory: object = None,
    display_controls_view_state_factory: object = None,
    fast_toggle_var_factory: object = None,
    canvas_interaction_callbacks_factory: object = None,
    bind_canvas_interactions: Callable[..., None],
    set_canvas: Callable[[object], None],
    set_progress_text: Callable[[str], None] | None = None,
    refresh_run_status_bar: Callable[[], None] | None = None,
    manual_pick_armed_factory: object = None,
    manual_pick_session_active_factory: object = None,
    geometry_preview_exclude_armed_factory: object = None,
    live_geometry_preview_enabled_factory: object = None,
    qr_overlay_var_factory: object = None,
    overlay_artist_groups_factory: object = None,
    defer_overlay_redraw_factory: object = None,
    placeholder_text: str = (
        "Fast viewer active in a separate window.\n"
        "The embedded Matplotlib canvas is paused until fast-viewer mode is turned off."
    ),
    draw_interval_s: float = 0.08,
    requested_enabled: bool = False,
) -> RuntimeFastViewerWorkflow:
    """Build the fast-viewer runtime controller with late-bound dependencies."""

    matplotlib_canvas_widget = matplotlib_canvas.get_tk_widget()
    placeholder_var = tk_module.StringVar(value=str(placeholder_text))
    placeholder_label = ttk_module.Label(
        canvas_frame,
        textvariable=placeholder_var,
        anchor=tk_module.CENTER,
        justify=tk_module.CENTER,
        wraplength=520,
        padding=(24, 24),
    )

    fast_image_viewer = None
    fast_canvas_proxy = None
    fast_requested = bool(requested_enabled)
    fast_suspend_reason = None
    bound_canvas_ids: set[int] = set()

    def _display_controls_view_state() -> object | None:
        return _resolve_runtime_value(display_controls_view_state_factory)

    def _refresh_run_status_bar() -> None:
        if callable(refresh_run_status_bar):
            try:
                refresh_run_status_bar()
            except Exception:
                pass

    def _set_status_text(text: str) -> None:
        view_state = _display_controls_view_state()
        status_var = getattr(view_state, "fast_viewer_status_var", None)
        if status_var is None:
            return
        try:
            status_var.set(str(text))
        except Exception:
            pass

    def _show_placeholder() -> None:
        try:
            matplotlib_canvas_widget.pack_forget()
        except Exception:
            pass
        try:
            if str(placeholder_label.winfo_manager()) != "pack":
                placeholder_label.pack(
                    side=tk_module.TOP,
                    fill=tk_module.BOTH,
                    expand=True,
                )
        except Exception:
            pass

    def _hide_placeholder() -> None:
        try:
            placeholder_label.pack_forget()
        except Exception:
            pass
        try:
            if str(matplotlib_canvas_widget.winfo_manager()) != "pack":
                matplotlib_canvas_widget.pack(
                    side=tk_module.TOP,
                    fill=tk_module.BOTH,
                    expand=True,
                )
        except Exception:
            pass

    def _marker_artist() -> object | None:
        return _resolve_runtime_value(marker_artist_factory)

    def _ensure_canvas_interaction_bindings(target_canvas) -> None:
        callbacks = _resolve_runtime_value(canvas_interaction_callbacks_factory)
        if target_canvas is None or callbacks is None:
            return
        target_id = int(id(target_canvas))
        if target_id in bound_canvas_ids:
            return
        bind_canvas_interactions(canvas=target_canvas, callbacks=callbacks)
        bound_canvas_ids.add(target_id)

    def _sync_from_matplotlib() -> None:
        proxy = fast_canvas_proxy
        if proxy is None:
            return
        try:
            proxy.sync_from_matplotlib(
                ax=ax,
                image_artist=image_artist,
                background_artist=background_artist,
                overlay_artist=overlay_artist,
                marker_artist=_marker_artist(),
            )
        except Exception:
            pass

    def _active() -> bool:
        return fast_canvas_proxy is not None and fast_image_viewer is not None

    def _requested_enabled() -> bool:
        return bool(fast_requested)

    def _suspend_reason() -> str | None:
        return fast_suspend_reason

    def _refresh_status_text() -> None:
        if _active():
            _set_status_text("Fast viewer active. Embedded canvas paused.")
            return
        if bool(fast_requested) and isinstance(fast_suspend_reason, str):
            _set_status_text(
                f"Fast viewer paused: {fast_suspend_reason}. Using embedded canvas."
            )
            return
        if bool(fast_requested):
            _set_status_text("Fast viewer requested.")
            return
        _set_status_text("Fast viewer off.")

    def _set_requested_enabled(enabled: bool) -> None:
        nonlocal fast_requested

        fast_requested = bool(enabled)
        toggle_var = _resolve_runtime_value(fast_toggle_var_factory)
        if toggle_var is not None:
            try:
                if bool(toggle_var.get()) != bool(enabled):
                    toggle_var.set(bool(enabled))
            except Exception:
                pass

    def _suspend_reason_text() -> str | None:
        try:
            if bool(_resolve_runtime_value(manual_pick_armed_factory)):
                return "manual geometry picking is active"
        except Exception:
            pass
        try:
            manual_pick_active = _resolve_runtime_value(
                manual_pick_session_active_factory
            )
            if bool(manual_pick_active):
                return "manual geometry picking is active"
        except Exception:
            pass
        try:
            if bool(_resolve_runtime_value(geometry_preview_exclude_armed_factory)):
                return "geometry preview exclusion is active"
        except Exception:
            pass
        try:
            if bool(_resolve_runtime_value(live_geometry_preview_enabled_factory)):
                return "live geometry preview is active"
        except Exception:
            pass
        try:
            qr_overlay_var = _resolve_runtime_value(qr_overlay_var_factory)
            if qr_overlay_var is not None and bool(qr_overlay_var.get()):
                return "QR-cylinder overlays are enabled"
        except Exception:
            pass
        try:
            artist_groups = tuple(_resolve_runtime_value(overlay_artist_groups_factory) or ())
        except Exception:
            artist_groups = ()
        if any(bool(group) for group in artist_groups):
            return "geometry overlays are visible"
        return None

    def _enable(*, announce: bool = True) -> bool:
        nonlocal fast_image_viewer, fast_canvas_proxy

        if _active():
            set_canvas(fast_canvas_proxy)
            _show_placeholder()
            _ensure_canvas_interaction_bindings(fast_canvas_proxy)
            _sync_from_matplotlib()
            _refresh_status_text()
            return True

        viewer = fast_plot_viewer_module.FastPlotViewer(title="RA-SIM Fast Viewer")
        if not bool(getattr(viewer, "available", False)):
            error_text = str(getattr(viewer, "error_message", "Unavailable"))
            _set_requested_enabled(False)
            _set_status_text(f"Fast viewer unavailable: {error_text}")
            if announce and callable(set_progress_text):
                set_progress_text(f"Fast viewer unavailable: {error_text}")
            return False

        proxy = fast_plot_viewer_module.MatplotlibCanvasProxy(
            matplotlib_canvas,
            viewer,
            draw_interval_s=float(draw_interval_s),
            render_matplotlib=False,
            event_axes=ax,
        )
        proxy.set_sync_callback(_sync_from_matplotlib)
        fast_image_viewer = viewer
        fast_canvas_proxy = proxy
        set_canvas(proxy)
        _show_placeholder()
        _ensure_canvas_interaction_bindings(proxy)
        _sync_from_matplotlib()
        proxy.process_fast_events()
        _refresh_status_text()
        if announce and callable(set_progress_text):
            set_progress_text(
                "Fast viewer enabled. Using external primary image renderer."
            )
        _refresh_run_status_bar()
        return True

    def _disable(*, announce: bool = True) -> None:
        nonlocal fast_image_viewer, fast_canvas_proxy

        viewer = fast_image_viewer
        fast_image_viewer = None
        fast_canvas_proxy = None
        set_canvas(matplotlib_canvas)
        _hide_placeholder()
        matplotlib_canvas.draw()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        _refresh_status_text()
        if announce and callable(set_progress_text):
            set_progress_text("Fast viewer disabled.")
        _refresh_run_status_bar()

    def _refresh_runtime_mode(*, announce: bool = False) -> bool:
        nonlocal fast_suspend_reason

        requested = bool(fast_requested)
        suspend_reason = _suspend_reason_text() if requested else None
        fast_suspend_reason = suspend_reason

        if requested and suspend_reason is None:
            if not _active():
                if not _enable(announce=announce):
                    fast_suspend_reason = None
                    _refresh_status_text()
                    _refresh_run_status_bar()
                    return False
            _refresh_status_text()
            _refresh_run_status_bar()
            return True

        if _active():
            _disable(announce=False)
        else:
            _hide_placeholder()

        _refresh_status_text()
        _refresh_run_status_bar()
        if announce and requested and isinstance(suspend_reason, str):
            if callable(set_progress_text):
                set_progress_text(f"Fast viewer paused while {suspend_reason}.")
        elif announce and not requested:
            if callable(set_progress_text):
                set_progress_text("Fast viewer disabled.")
        return False

    def _request_main_canvas_redraw(*, force_matplotlib: bool = False) -> None:
        _refresh_runtime_mode(announce=False)
        if _active():
            _sync_from_matplotlib()
            try:
                fast_canvas_proxy.process_fast_events()
            except Exception:
                pass
            if not force_matplotlib:
                return
        if force_matplotlib:
            matplotlib_canvas.draw()
        else:
            matplotlib_canvas.draw_idle()

    def _request_overlay_canvas_redraw(*, force: bool = False) -> None:
        _refresh_runtime_mode(announce=False)
        if not force and bool(_resolve_runtime_value(defer_overlay_redraw_factory)):
            return
        _request_main_canvas_redraw(
            force_matplotlib=bool(force and not _active())
        )

    def _toggle() -> None:
        toggle_var = _resolve_runtime_value(fast_toggle_var_factory)
        enabled = bool(toggle_var.get()) if toggle_var is not None else False
        _set_requested_enabled(enabled)
        _refresh_runtime_mode(announce=True)

    def _shutdown() -> None:
        nonlocal fast_image_viewer, fast_canvas_proxy

        viewer = fast_image_viewer
        fast_image_viewer = None
        fast_canvas_proxy = None
        set_canvas(matplotlib_canvas)
        _hide_placeholder()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    return RuntimeFastViewerWorkflow(
        active=_active,
        requested_enabled=_requested_enabled,
        suspend_reason=_suspend_reason,
        set_requested_enabled=_set_requested_enabled,
        refresh_runtime_mode=_refresh_runtime_mode,
        request_main_canvas_redraw=_request_main_canvas_redraw,
        request_overlay_canvas_redraw=_request_overlay_canvas_redraw,
        refresh_status_text=_refresh_status_text,
        toggle=_toggle,
        shutdown=_shutdown,
    )
