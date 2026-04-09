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
    reset_view: Callable[[], None]
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
    overlay_model_factory: object = None,
    layer_versions_factory: object = None,
    display_controls_view_state_factory: object = None,
    fast_toggle_var_factory: object = None,
    canvas_interaction_callbacks_factory: object = None,
    bind_canvas_interactions: Callable[..., None],
    set_canvas: Callable[[object], None],
    set_progress_text: Callable[[str], None] | None = None,
    refresh_run_status_bar: Callable[[], None] | None = None,
    manual_pick_armed_factory: object = None,
    hkl_pick_armed_factory: object = None,
    manual_pick_session_active_factory: object = None,
    geometry_preview_exclude_armed_factory: object = None,
    live_geometry_preview_enabled_factory: object = None,
    qr_overlay_var_factory: object = None,
    overlay_artist_groups_factory: object = None,
    defer_overlay_redraw_factory: object = None,
    integration_drag_active_factory: object = None,
    placeholder_text: str = (
        "Fast viewer active in a separate window.\n"
        "The embedded Matplotlib canvas is paused until fast-viewer mode is turned off."
    ),
    draw_interval_s: float = 0.08,
    requested_enabled: bool = False,
    control_locked: bool = False,
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
    fast_surface_mode = None
    fast_requested = bool(requested_enabled)
    fast_suspend_reason = None
    fast_unavailable_reason = None
    fast_input_overlay = None
    fast_input_overlay_destroy_token = None
    fast_input_overlay_pending_release_grab = False
    bound_canvas_ids: set[int] = set()
    overlay_idle_alpha = 0.0
    overlay_active_alpha = 0.01

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

    def _set_fast_viewer_control_enabled(enabled: bool) -> None:
        view_state = _display_controls_view_state()
        checkbutton = getattr(view_state, "fast_viewer_checkbutton", None)
        if checkbutton is None:
            return
        enabled = bool(enabled) and not bool(control_locked)
        try:
            state = getattr(checkbutton, "state", None)
            if callable(state):
                state(["!disabled"] if bool(enabled) else ["disabled"])
                return
        except Exception:
            pass
        for attr_name in ("configure", "config"):
            try:
                setter = getattr(checkbutton, attr_name, None)
                if callable(setter):
                    setter(state=("normal" if bool(enabled) else "disabled"))
                    return
            except Exception:
                pass

    def _hide_matplotlib_canvas() -> None:
        try:
            matplotlib_canvas_widget.pack_forget()
        except Exception:
            pass

    def _show_matplotlib_canvas() -> None:
        _hide_placeholder()
        try:
            if str(matplotlib_canvas_widget.winfo_manager()) != "pack":
                matplotlib_canvas_widget.pack(
                    side=tk_module.TOP,
                    fill=tk_module.BOTH,
                    expand=True,
                )
        except Exception:
            pass

    def _show_placeholder() -> None:
        _hide_matplotlib_canvas()
        try:
            if str(placeholder_label.winfo_manager()) != "pack":
                placeholder_label.pack(
                    side=tk_module.TOP,
                    fill=tk_module.BOTH,
                    expand=True,
                )
        except Exception:
            pass

    def _show_embedded_fast_viewer() -> None:
        _hide_placeholder()
        try:
            matplotlib_canvas_widget.pack_forget()
        except Exception:
            pass
        viewer = fast_image_viewer
        if viewer is None:
            return
        try:
            resize = getattr(viewer, "resize_to_tk_host", None)
            if callable(resize):
                resize(canvas_frame)
        except Exception:
            pass
        _sync_fast_input_overlay_geometry()

    def _hide_placeholder() -> None:
        try:
            placeholder_label.pack_forget()
        except Exception:
            pass

    def _marker_artist() -> object | None:
        return _resolve_runtime_value(marker_artist_factory)

    def _overlay_model() -> object | None:
        return _resolve_runtime_value(overlay_model_factory)

    def _layer_versions() -> dict[str, object]:
        resolved = _resolve_runtime_value(layer_versions_factory)
        if isinstance(resolved, dict):
            return dict(resolved)
        return {}

    def _ensure_canvas_interaction_bindings(target_canvas) -> None:
        callbacks = _resolve_runtime_value(canvas_interaction_callbacks_factory)
        if target_canvas is None or callbacks is None:
            return
        target_id = int(id(target_canvas))
        if target_id in bound_canvas_ids:
            return
        bind_canvas_interactions(canvas=target_canvas, callbacks=callbacks)
        bound_canvas_ids.add(target_id)

    def _sync_from_matplotlib(*, force_view_range: bool = False) -> None:
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
                overlay_model=_overlay_model(),
                layer_versions=_layer_versions(),
                force_view_range=bool(force_view_range),
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
        control_suspend_reason = None if _active() else _suspend_reason_text()

        if _active():
            _set_fast_viewer_control_enabled(True)
            if fast_surface_mode == "embedded":
                _set_status_text(
                    "Fast viewer active in plot area. Matplotlib canvas paused."
                )
            else:
                _set_status_text("Fast viewer active in separate window.")
            return
        if isinstance(fast_unavailable_reason, str):
            _set_fast_viewer_control_enabled(False)
            _set_status_text(f"Fast viewer unavailable: {fast_unavailable_reason}.")
            return
        if isinstance(control_suspend_reason, str):
            _set_fast_viewer_control_enabled(False)
            if bool(fast_requested):
                _set_status_text(
                    "Fast viewer unavailable while "
                    f"{control_suspend_reason}. It will resume automatically "
                    "when that mode ends."
                )
            else:
                _set_status_text(
                    f"Fast viewer unavailable while {control_suspend_reason}."
                )
            return
        _set_fast_viewer_control_enabled(True)
        if bool(fast_requested) and isinstance(fast_suspend_reason, str):
            _set_status_text(
                f"Fast viewer paused: {fast_suspend_reason}. Using embedded canvas."
            )
            return
        if bool(fast_requested):
            _set_status_text("Fast viewer requested.")
            return
        _set_status_text("")

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
            if bool(_resolve_runtime_value(hkl_pick_armed_factory)):
                return "HKL image-pick is active"
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

    def _embed_fast_viewer(viewer) -> bool:
        mount = getattr(viewer, "mount_into_tk", None)
        if not callable(mount):
            return False
        try:
            if not bool(mount(canvas_frame)):
                return False
        except Exception:
            return False
        _show_embedded_fast_viewer()
        return True

    def _current_fast_viewer_overlay_geometry() -> tuple[int, int, int, int] | None:
        viewer = fast_image_viewer
        if viewer is None:
            return None
        geometry_getter = getattr(viewer, "viewport_global_geometry", None)
        if callable(geometry_getter):
            try:
                geometry = geometry_getter()
            except Exception:
                geometry = None
            if (
                isinstance(geometry, tuple)
                and len(geometry) == 4
                and all(isinstance(value, int) for value in geometry)
            ):
                x_pos, y_pos, width, height = geometry
                if width > 0 and height > 0:
                    return geometry
        try:
            return (
                int(canvas_frame.winfo_rootx()),
                int(canvas_frame.winfo_rooty()),
                max(int(canvas_frame.winfo_width()), 1),
                max(int(canvas_frame.winfo_height()), 1),
            )
        except Exception:
            return None

    def _fast_input_overlay_drag_active() -> bool:
        return bool(_resolve_runtime_value(integration_drag_active_factory))

    def _set_fast_input_overlay_alpha(alpha: float) -> None:
        overlay = fast_input_overlay
        if overlay is None:
            return
        alpha_value = float(max(0.0, min(1.0, alpha)))
        for attr_name in ("attributes", "wm_attributes"):
            setter = getattr(overlay, attr_name, None)
            if not callable(setter):
                continue
            try:
                setter("-alpha", alpha_value)
            except Exception:
                pass

    def _sync_fast_input_overlay_visibility() -> None:
        overlay = fast_input_overlay
        if overlay is None:
            return
        target_alpha = (
            overlay_active_alpha
            if fast_surface_mode == "embedded"
            else overlay_idle_alpha
        )
        _set_fast_input_overlay_alpha(target_alpha)

    def _clear_fast_input_overlay_destroy() -> None:
        nonlocal fast_input_overlay_destroy_token
        if fast_input_overlay_destroy_token is None:
            return
        overlay = fast_input_overlay
        try:
            cancel = getattr(overlay, "after_cancel", None) if overlay is not None else None
            if callable(cancel):
                cancel(fast_input_overlay_destroy_token)
        except Exception:
            pass
        fast_input_overlay_destroy_token = None

    def _schedule_fast_input_overlay_destroy(
        *,
        release_grab: bool = False,
    ) -> None:
        nonlocal fast_input_overlay_destroy_token
        nonlocal fast_input_overlay_pending_release_grab

        overlay = fast_input_overlay
        if overlay is None:
            return
        fast_input_overlay_pending_release_grab = bool(
            fast_input_overlay_pending_release_grab or release_grab
        )
        _clear_fast_input_overlay_destroy()

        def _finish_destroy() -> None:
            nonlocal fast_input_overlay_destroy_token
            nonlocal fast_input_overlay_pending_release_grab

            fast_input_overlay_destroy_token = None
            if fast_input_overlay_pending_release_grab:
                try:
                    grab_release = getattr(overlay, "grab_release", None)
                    if callable(grab_release):
                        grab_release()
                except Exception:
                    pass
            fast_input_overlay_pending_release_grab = False
            _destroy_fast_input_overlay()

        after = getattr(overlay, "after", None)
        if callable(after):
            try:
                fast_input_overlay_destroy_token = after(0, _finish_destroy)
                return
            except Exception:
                fast_input_overlay_destroy_token = None
        _finish_destroy()

    def _dispatch_fast_input_overlay_event(
        event_name: str,
        event: object,
        *,
        button: int | None,
        dblclick: bool = False,
    ) -> None:
        overlay = fast_input_overlay
        proxy = fast_canvas_proxy
        if overlay is None or proxy is None or fast_surface_mode != "embedded":
            return
        try:
            width = max(int(overlay.winfo_width()), 1)
            height = max(int(overlay.winfo_height()), 1)
            x_value = float(getattr(event, "x", 0.0))
            y_value = float(getattr(event, "y", 0.0))
        except Exception:
            return
        try:
            dispatcher = getattr(proxy, "dispatch_host_overlay_event", None)
            if callable(dispatcher):
                dispatcher(
                    event_name,
                    x_pixel=x_value,
                    y_pixel=y_value,
                    width=width,
                    height=height,
                    button=button,
                    dblclick=bool(dblclick),
                )
        except Exception:
            pass

    def _destroy_fast_input_overlay() -> None:
        nonlocal fast_input_overlay
        nonlocal fast_input_overlay_pending_release_grab

        overlay = fast_input_overlay
        _clear_fast_input_overlay_destroy()
        fast_input_overlay = None
        fast_input_overlay_pending_release_grab = False
        if overlay is None:
            return
        try:
            grab_release = getattr(overlay, "grab_release", None)
            if callable(grab_release):
                grab_release()
        except Exception:
            pass
        try:
            destroy = getattr(overlay, "destroy", None)
            if callable(destroy):
                destroy()
        except Exception:
            pass

    def _sync_fast_input_overlay_geometry() -> None:
        overlay = fast_input_overlay
        if overlay is None or fast_surface_mode != "embedded":
            return
        geometry = _current_fast_viewer_overlay_geometry()
        if geometry is None:
            try:
                withdraw = getattr(overlay, "withdraw", None)
                if callable(withdraw):
                    withdraw()
            except Exception:
                pass
            return
        x_pos, y_pos, width, height = geometry
        try:
            geometry_setter = getattr(overlay, "geometry", None)
            if callable(geometry_setter):
                geometry_setter(f"{int(width)}x{int(height)}+{int(x_pos)}+{int(y_pos)}")
        except Exception:
            return
        for attr_name in ("deiconify", "lift"):
            try:
                setter = getattr(overlay, attr_name, None)
                if callable(setter):
                    setter()
            except Exception:
                pass
        _sync_fast_input_overlay_visibility()

    def _ensure_fast_input_overlay() -> None:
        nonlocal fast_input_overlay

        if fast_surface_mode != "embedded" or fast_canvas_proxy is None:
            _destroy_fast_input_overlay()
            return
        if fast_input_overlay is not None:
            _sync_fast_input_overlay_geometry()
            return
        overlay_cls = getattr(tk_module, "Toplevel", None)
        if overlay_cls is None:
            return
        try:
            parent_window = canvas_frame.winfo_toplevel()
        except Exception:
            parent_window = None
        try:
            overlay = (
                overlay_cls(parent_window)
                if parent_window is not None
                else overlay_cls()
            )
        except Exception:
            return
        for attr_name, value in (
            ("overrideredirect", True),
            ("transient", parent_window),
        ):
            try:
                setter = getattr(overlay, attr_name, None)
                if callable(setter):
                    setter(value)
            except TypeError:
                continue
            except Exception:
                continue
        for attr_name in ("attributes", "wm_attributes"):
            setter = getattr(overlay, attr_name, None)
            if not callable(setter):
                continue
            try:
                setter("-alpha", overlay_active_alpha)
            except Exception:
                pass
            try:
                setter("-topmost", True)
            except Exception:
                pass
        try:
            configure = getattr(overlay, "configure", None)
            if callable(configure):
                configure(background="black")
        except Exception:
            pass

        def _wrap_dispatch(
            event_name: str,
            *,
            button: int | None,
            dblclick: bool = False,
            grab_on_press: bool = False,
            release_grab: bool = False,
        ) -> Callable[[object], None]:
            def _handler(event) -> None:
                try:
                    if grab_on_press:
                        grab_set = getattr(overlay, "grab_set", None)
                        if callable(grab_set):
                            grab_set()
                except Exception:
                    pass
                _dispatch_fast_input_overlay_event(
                    event_name,
                    event,
                    button=button,
                    dblclick=dblclick,
                )
                if release_grab:
                    _schedule_fast_input_overlay_destroy(release_grab=True)
                else:
                    _sync_fast_input_overlay_visibility()

            return _handler

        binder = getattr(overlay, "bind", None)
        if callable(binder):
            try:
                binder(
                    "<ButtonPress-1>",
                    _wrap_dispatch(
                        "button_press_event",
                        button=1,
                        grab_on_press=True,
                    ),
                    add="+",
                )
                binder(
                    "<Double-Button-1>",
                    _wrap_dispatch(
                        "button_press_event",
                        button=1,
                        dblclick=True,
                        grab_on_press=True,
                    ),
                    add="+",
                )
                binder(
                    "<B1-Motion>",
                    _wrap_dispatch("motion_notify_event", button=None),
                    add="+",
                )
                binder(
                    "<Motion>",
                    _wrap_dispatch("motion_notify_event", button=None),
                    add="+",
                )
                binder(
                    "<ButtonRelease-1>",
                    _wrap_dispatch(
                        "button_release_event",
                        button=1,
                        release_grab=True,
                    ),
                    add="+",
                )
                binder(
                    "<ButtonPress-3>",
                    _wrap_dispatch("button_press_event", button=3),
                    add="+",
                )
                binder(
                    "<ButtonRelease-3>",
                    _wrap_dispatch("button_release_event", button=3),
                    add="+",
                )
            except Exception:
                pass

        fast_input_overlay = overlay
        _sync_fast_input_overlay_geometry()

    def _fast_viewer_event_local_position(event: object) -> tuple[float, float] | None:
        geometry = _current_fast_viewer_overlay_geometry()
        if geometry is None:
            return None
        x_pos, y_pos, width, height = geometry
        try:
            root_x = getattr(event, "x_root", None)
            root_y = getattr(event, "y_root", None)
            if root_x is None or root_y is None:
                root_x = float(getattr(event, "x", 0.0)) + float(canvas_frame.winfo_rootx())
                root_y = float(getattr(event, "y", 0.0)) + float(canvas_frame.winfo_rooty())
            else:
                root_x = float(root_x)
                root_y = float(root_y)
        except Exception:
            return None
        local_x = float(root_x) - float(x_pos)
        local_y = float(root_y) - float(y_pos)
        if local_x < 0.0 or local_y < 0.0:
            return None
        if local_x > float(width) or local_y > float(height):
            return None
        return (local_x, local_y)

    def _dispatch_fast_input_host_press(
        event_name: str,
        event: object,
        *,
        button: int | None,
        dblclick: bool = False,
    ) -> None:
        if fast_surface_mode != "embedded" or fast_canvas_proxy is None:
            return
        local_position = _fast_viewer_event_local_position(event)
        if local_position is None:
            return
        _ensure_fast_input_overlay()
        overlay = fast_input_overlay
        if overlay is None:
            return
        try:
            if button == 1:
                grab_set = getattr(overlay, "grab_set", None)
                if callable(grab_set):
                    grab_set()
        except Exception:
            pass
        _dispatch_fast_input_overlay_event(
            event_name,
            type("_HostOverlayEvent", (), {"x": local_position[0], "y": local_position[1]})(),
            button=button,
            dblclick=dblclick,
        )
        _sync_fast_input_overlay_visibility()

    def _resize_embedded_fast_viewer() -> None:
        if fast_surface_mode != "embedded":
            return
        viewer = fast_image_viewer
        if viewer is None:
            return
        try:
            resize = getattr(viewer, "resize_to_tk_host", None)
            if callable(resize):
                resize(canvas_frame)
        except Exception:
            pass
        _sync_fast_input_overlay_geometry()

    top_level_binder = getattr(getattr(canvas_frame, "winfo_toplevel", lambda: None)(), "bind", None)
    frame_binder = getattr(canvas_frame, "bind", None)
    if callable(frame_binder):
        try:
            frame_binder(
                "<Configure>",
                lambda _event=None: _resize_embedded_fast_viewer(),
                add="+",
            )
        except Exception:
            pass
    if callable(top_level_binder):
        try:
            top_level_binder(
                "<Configure>",
                lambda _event=None: _sync_fast_input_overlay_geometry(),
                add="+",
            )
        except Exception:
            pass

    def _enable(*, announce: bool = True) -> bool:
        nonlocal fast_canvas_proxy, fast_image_viewer, fast_surface_mode
        nonlocal fast_unavailable_reason

        if _active():
            set_canvas(fast_canvas_proxy)
            if fast_surface_mode == "embedded":
                _show_embedded_fast_viewer()
            else:
                _destroy_fast_input_overlay()
                _show_placeholder()
            _ensure_canvas_interaction_bindings(fast_canvas_proxy)
            _sync_from_matplotlib()
            _refresh_status_text()
            return True

        viewer = fast_plot_viewer_module.FastPlotViewer(
            title="RA-SIM Fast Viewer",
            show_window=False,
        )
        if not bool(getattr(viewer, "available", False)):
            error_text = str(getattr(viewer, "error_message", "Unavailable"))
            fast_unavailable_reason = error_text
            _set_requested_enabled(False)
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
        fast_unavailable_reason = None
        fast_surface_mode = "embedded" if _embed_fast_viewer(viewer) else "window"
        if fast_surface_mode == "embedded":
            _show_embedded_fast_viewer()
        else:
            _destroy_fast_input_overlay()
            try:
                show_window = getattr(viewer, "show_window", None)
                if callable(show_window):
                    show_window()
            except Exception:
                pass
            _show_placeholder()
        set_canvas(proxy)
        _ensure_canvas_interaction_bindings(proxy)
        _sync_from_matplotlib()
        proxy.process_fast_events()
        _refresh_status_text()
        if announce and callable(set_progress_text):
            if fast_surface_mode == "embedded":
                set_progress_text("Fast viewer enabled in plot area.")
            else:
                set_progress_text(
                    "Fast viewer enabled. Using external primary image renderer."
                )
        _refresh_run_status_bar()
        return True

    def _disable(*, announce: bool = True) -> None:
        nonlocal fast_canvas_proxy, fast_image_viewer, fast_surface_mode

        viewer = fast_image_viewer
        fast_image_viewer = None
        fast_canvas_proxy = None
        fast_surface_mode = None
        _destroy_fast_input_overlay()
        set_canvas(matplotlib_canvas)
        _show_matplotlib_canvas()
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
            _sync_fast_input_overlay_visibility()
            _refresh_status_text()
            _refresh_run_status_bar()
            return True

        if _active():
            _disable(announce=False)
        else:
            _show_matplotlib_canvas()

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

    def _reset_view() -> None:
        _refresh_runtime_mode(announce=False)
        if not _active():
            try:
                matplotlib_canvas.draw_idle()
            except Exception:
                pass
            return
        _sync_from_matplotlib(force_view_range=True)
        try:
            fast_canvas_proxy.process_fast_events()
        except Exception:
            pass

    def _toggle() -> None:
        toggle_var = _resolve_runtime_value(fast_toggle_var_factory)
        enabled = bool(toggle_var.get()) if toggle_var is not None else False
        _set_requested_enabled(enabled)
        _refresh_runtime_mode(announce=True)

    def _shutdown() -> None:
        nonlocal fast_canvas_proxy, fast_image_viewer, fast_surface_mode

        viewer = fast_image_viewer
        fast_image_viewer = None
        fast_canvas_proxy = None
        fast_surface_mode = None
        _destroy_fast_input_overlay()
        set_canvas(matplotlib_canvas)
        _show_matplotlib_canvas()
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
        reset_view=_reset_view,
        refresh_status_text=_refresh_status_text,
        toggle=_toggle,
        shutdown=_shutdown,
    )
