import numpy as np
from types import SimpleNamespace

from ra_sim.gui import integration_range_drag, state


class _FakeVar:
    def __init__(self, value=0.0):
        self._value = value
        self.trace_calls = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def trace_add(self, mode, callback):
        self.trace_calls.append((mode, callback))
        return f"trace-{len(self.trace_calls)}"


class _FakeSlider:
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def cget(self, key):
        if key == "from":
            return self.minimum
        if key == "to":
            return self.maximum
        raise KeyError(key)


class _FakeRect:
    def __init__(self, xy=None, width=None, height=None, **kwargs):
        self.init_xy = tuple(xy) if xy is not None else None
        self.init_width = None if width is None else float(width)
        self.init_height = None if height is None else float(height)
        self.init_kwargs = dict(kwargs)
        self.xy = self.init_xy
        self.width = self.init_width
        self.height = self.init_height
        self.visible = False

    def set_xy(self, xy):
        self.xy = tuple(xy)

    def set_width(self, width):
        self.width = float(width)

    def set_height(self, height):
        self.height = float(height)

    def set_visible(self, visible):
        self.visible = bool(visible)


class _FakeOverlay:
    def __init__(self):
        self.data = None
        self.extent = None
        self.visible = False

    def set_data(self, data):
        self.data = np.asarray(data, dtype=float)

    def set_extent(self, extent):
        self.extent = tuple(extent)

    def set_visible(self, visible):
        self.visible = bool(visible)


class _FakeAxis:
    def __init__(self, *, xlim=(0.0, 40.0), ylim=(-20.0, 20.0)):
        self._xlim = tuple(xlim)
        self._ylim = tuple(ylim)
        self.patches = []

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def add_patch(self, patch):
        self.patches.append(patch)


class _FakeImageDisplay:
    def __init__(self, extent=(0.0, 3.0, 2.0, 0.0)):
        self._extent = tuple(extent)

    def get_extent(self):
        return self._extent


class _FakeEvent:
    def __init__(self, *, button=1, inaxes=None, xdata=None, ydata=None):
        self.button = button
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata


class _FakeRoot:
    def __init__(self):
        self.next_token = 0
        self.after_calls = []
        self.after_cancel_calls = []

    def after(self, delay, callback):
        self.next_token += 1
        token = f"after-{self.next_token}"
        self.after_calls.append((delay, callback, token))
        return token

    def after_cancel(self, token):
        self.after_cancel_calls.append(token)


def _range_view_state() -> state.IntegrationRangeControlsViewState:
    return state.IntegrationRangeControlsViewState(
        tth_min_var=_FakeVar(0.0),
        tth_max_var=_FakeVar(0.0),
        phi_min_var=_FakeVar(0.0),
        phi_max_var=_FakeVar(0.0),
        tth_min_slider=_FakeSlider(0.0, 40.0),
        phi_min_slider=_FakeSlider(-20.0, 20.0),
    )


def test_create_runtime_drag_rectangles_attach_hidden_overlays() -> None:
    axis = _FakeAxis()

    drag_rect = integration_range_drag.create_drag_select_rectangle(
        axis,
        rectangle_cls=_FakeRect,
    )
    region_rect = integration_range_drag.create_integration_region_rectangle(
        axis,
        rectangle_cls=_FakeRect,
    )

    assert axis.patches == [drag_rect, region_rect]
    assert drag_rect.init_xy == (0.0, 0.0)
    assert drag_rect.init_width == 0.0
    assert drag_rect.init_height == 0.0
    assert drag_rect.init_kwargs["edgecolor"] == "yellow"
    assert drag_rect.init_kwargs["zorder"] == 6
    assert drag_rect.visible is False
    assert region_rect.init_xy == (0.0, 0.0)
    assert region_rect.init_width == 0.0
    assert region_rect.init_height == 0.0
    assert region_rect.init_kwargs["edgecolor"] == "cyan"
    assert region_rect.init_kwargs["linestyle"] == "--"
    assert region_rect.visible is False


def test_integration_range_drag_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {"view": 0, "schedule": 0, "draw": 0, "status": 0}

    monkeypatch.setattr(
        integration_range_drag,
        "IntegrationRangeDragBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def build_view():
        counters["view"] += 1
        return f"view-{counters['view']}"

    def build_schedule():
        counters["schedule"] += 1
        idx = counters["schedule"]
        return lambda: f"schedule-{idx}"

    def build_draw():
        counters["draw"] += 1
        idx = counters["draw"]
        return lambda: f"draw-{idx}"

    def build_status():
        counters["status"] += 1
        idx = counters["status"]
        return lambda text: f"status-{idx}:{text}"

    factory = integration_range_drag.make_runtime_integration_range_drag_bindings_factory(
        drag_state="drag-state",
        peak_selection_state="peak-state",
        range_view_state_factory=build_view,
        ax="axis",
        drag_select_rect="drag-rect",
        integration_region_overlay="overlay",
        integration_region_rect="overlay-rect",
        image_display="image-display",
        get_detector_angular_maps=lambda ai: ai,
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: "ai",
        sync_peak_selection_state=lambda: None,
        schedule_range_update_factory=build_schedule,
        last_sim_res2_factory=lambda: "res2",
        draw_idle_factory=build_draw,
        set_status_text_factory=build_status,
    )

    assert factory()["drag_state"] == "drag-state"
    assert factory()["drag_state"] == "drag-state"
    assert calls[0]["range_view_state"] == "view-1"
    assert calls[1]["range_view_state"] == "view-2"
    assert calls[0]["ax"] == "axis"
    assert calls[0]["range_visible_factory"]() is True
    assert callable(calls[0]["schedule_range_update"])
    assert callable(calls[0]["draw_idle"])
    assert callable(calls[0]["set_status_text"])
    assert "update_integration_region_visuals" not in calls[0]
    assert calls[0]["schedule_range_update"] is not calls[1]["schedule_range_update"]
    assert calls[0]["draw_idle"] is not calls[1]["draw_idle"]
    assert calls[0]["set_status_text"] is not calls[1]["set_status_text"]


def test_integration_range_update_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {
        "analysis": 0,
        "range": 0,
        "drag": 0,
        "lookup": 0,
        "refresh": 0,
        "schedule": 0,
        "debounce": 0,
    }

    monkeypatch.setattr(
        integration_range_drag,
        "IntegrationRangeUpdateBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def _bump(key: str, prefix: str) -> str:
        counters[key] += 1
        return f"{prefix}-{counters[key]}"

    def _next_debounce() -> int:
        counters["debounce"] += 1
        return 90 + counters["debounce"]

    factory = integration_range_drag.make_runtime_integration_range_update_bindings_factory(
        root="root",
        simulation_runtime_state="simulation-state",
        analysis_view_state_factory=lambda: _bump("analysis", "analysis"),
        range_view_state_factory=lambda: _bump("range", "range"),
        display_controls_state="display-state",
        hkl_lookup_controls_factory=lambda: _bump("lookup", "lookup"),
        integration_range_drag_callbacks_factory=lambda: _bump("drag", "drag"),
        refresh_integration_from_cached_results_factory=lambda: _bump(
            "refresh",
            "refresh",
        ),
        schedule_update_factory=lambda: _bump("schedule", "schedule"),
        range_update_debounce_ms_factory=_next_debounce,
    )

    first = factory()
    second = factory()

    assert first["root"] == "root"
    assert first["simulation_runtime_state"] == "simulation-state"
    assert first["analysis_view_state"] == "analysis-1"
    assert second["analysis_view_state"] == "analysis-2"
    assert first["range_view_state"] == "range-1"
    assert second["range_view_state"] == "range-2"
    assert first["display_controls_state"] == "display-state"
    assert first["hkl_lookup_controls"] == "lookup-1"
    assert second["hkl_lookup_controls"] == "lookup-2"
    assert first["integration_range_drag_callbacks"] == "drag-1"
    assert second["integration_range_drag_callbacks"] == "drag-2"
    assert first["refresh_integration_from_cached_results"] == "refresh-1"
    assert second["schedule_update"] == "schedule-2"
    assert first["range_update_debounce_ms"] == 91
    assert second["range_update_debounce_ms"] == 92


def test_create_runtime_integration_range_controls_wires_callbacks_and_text_sync() -> None:
    schedule_calls = []
    callback_refs = {}
    view_state = state.IntegrationRangeControlsViewState()

    def _create_controls(**kwargs):
        callback_refs.update(kwargs)
        kwargs["view_state"].tth_min_var = _FakeVar(kwargs["tth_min"])
        kwargs["view_state"].tth_max_var = _FakeVar(kwargs["tth_max"])
        kwargs["view_state"].phi_min_var = _FakeVar(kwargs["phi_min"])
        kwargs["view_state"].phi_max_var = _FakeVar(kwargs["phi_max"])
        kwargs["view_state"].tth_min_label_var = _FakeVar("")
        kwargs["view_state"].tth_max_label_var = _FakeVar("")
        kwargs["view_state"].phi_min_label_var = _FakeVar("")
        kwargs["view_state"].phi_max_label_var = _FakeVar("")
        kwargs["view_state"].tth_min_entry_var = _FakeVar("")
        kwargs["view_state"].tth_max_entry_var = _FakeVar("")
        kwargs["view_state"].phi_min_entry_var = _FakeVar("")
        kwargs["view_state"].phi_max_entry_var = _FakeVar("")
        kwargs["view_state"].tth_min_slider = _FakeSlider(0.0, 60.0)
        kwargs["view_state"].tth_max_slider = _FakeSlider(0.0, 60.0)
        kwargs["view_state"].phi_min_slider = _FakeSlider(-15.0, 15.0)
        kwargs["view_state"].phi_max_slider = _FakeSlider(-15.0, 15.0)

    integration_range_drag.create_runtime_integration_range_controls(
        parent="parent",
        views_module=SimpleNamespace(create_integration_range_controls=_create_controls),
        view_state=view_state,
        tth_min=0.0,
        tth_max=60.0,
        phi_min=-15.0,
        phi_max=15.0,
        schedule_range_update=lambda: schedule_calls.append("range"),
    )

    assert callback_refs["parent"] == "parent"
    assert view_state.tth_min_label_var.get() == "0.0"
    assert view_state.tth_max_entry_var.get() == "60.0000"
    assert view_state.phi_min_entry_var.get() == "-15.0000"
    assert len(view_state.tth_min_var.trace_calls) == 1
    assert len(view_state.phi_max_var.trace_calls) == 1

    callback_refs["on_tth_min_changed"]("12.5")
    assert view_state.tth_min_var.get() == 12.5
    assert view_state.tth_min_label_var.get() == "12.5"
    assert view_state.tth_min_entry_var.get() == "12.5000"

    view_state.phi_max_entry_var.set("45.0")
    callback_refs["on_apply_entry"](
        view_state.phi_max_entry_var,
        view_state.phi_max_var,
        view_state.phi_max_slider,
    )
    assert view_state.phi_max_var.get() == 15.0
    assert view_state.phi_max_label_var.get() == "15.0"
    assert view_state.phi_max_entry_var.get() == "15.0000"
    assert schedule_calls == ["range", "range"]

    view_state.phi_min_var.set(-7.25)
    trace_callback = view_state.phi_min_var.trace_calls[0][1]
    trace_callback()
    assert view_state.phi_min_label_var.get() == "-7.2"
    assert view_state.phi_min_entry_var.get() == "-7.2500"


def test_integration_range_update_callbacks_schedule_reschedule_and_toggle_modes() -> None:
    root = _FakeRoot()
    schedule_update_calls = []
    hkl_pick_calls = []
    drag_reset_calls = []
    refresh_results = [False]
    sim_state = SimpleNamespace(
        integration_update_pending=None,
        update_running=False,
        caked_limits_user_override=True,
    )
    analysis_view_state = SimpleNamespace(
        show_1d_var=_FakeVar(False),
        show_caked_2d_var=_FakeVar(False),
    )
    display_controls_state = SimpleNamespace(simulation_limits_user_override=True)
    bindings = integration_range_drag.IntegrationRangeUpdateBindings(
        root=root,
        simulation_runtime_state=sim_state,
        analysis_view_state=analysis_view_state,
        range_view_state=state.IntegrationRangeControlsViewState(),
        display_controls_state=display_controls_state,
        hkl_lookup_controls=SimpleNamespace(
            set_hkl_pick_mode=lambda enabled: hkl_pick_calls.append(enabled)
        ),
        integration_range_drag_callbacks=SimpleNamespace(
            reset=lambda: drag_reset_calls.append(True)
        ),
        refresh_integration_from_cached_results=lambda: refresh_results[-1],
        schedule_update=lambda: schedule_update_calls.append(True),
        range_update_debounce_ms=120,
    )
    callbacks = integration_range_drag.make_runtime_integration_range_update_callbacks(
        lambda: bindings
    )

    callbacks.schedule_range_update(delay_ms=50)
    assert root.after_calls[0][0] == 120
    assert sim_state.integration_update_pending == "after-1"

    sim_state.update_running = True
    root.after_calls[0][1]()
    assert sim_state.integration_update_pending == "after-2"
    assert root.after_cancel_calls == []
    assert root.after_calls[1][0] == 120
    assert schedule_update_calls == []

    sim_state.update_running = False
    root.after_calls[1][1]()
    assert sim_state.integration_update_pending is None
    assert schedule_update_calls == [True]

    callbacks.toggle_1d_plots()
    callbacks.toggle_log_radial()
    assert root.after_cancel_calls == ["after-3"]
    assert root.after_calls[-1][0] == 120

    analysis_view_state.show_caked_2d_var.set(True)
    display_controls_state.simulation_limits_user_override = True
    callbacks.toggle_caked_2d()
    assert analysis_view_state.show_1d_var.get() is True
    assert display_controls_state.simulation_limits_user_override is False
    assert hkl_pick_calls == [False]
    assert drag_reset_calls == [True]
    assert schedule_update_calls == [True, True]

    sim_state.caked_limits_user_override = True
    analysis_view_state.show_caked_2d_var.set(False)
    callbacks.toggle_caked_2d()
    assert analysis_view_state.show_1d_var.get() is False
    assert sim_state.caked_limits_user_override is False
    assert drag_reset_calls == [True, True]
    assert schedule_update_calls == [True, True, True]


def test_update_runtime_integration_region_visuals_hides_when_range_hidden() -> None:
    overlay = _FakeOverlay()
    overlay.visible = True
    overlay_rect = _FakeRect()
    overlay_rect.visible = True

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=_range_view_state(),
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: False,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=object(),
        sim_res2=object(),
    )

    assert overlay.visible is False
    assert overlay_rect.visible is False


def test_update_runtime_integration_region_visuals_updates_caked_rectangle() -> None:
    view_state = _range_view_state()
    view_state.tth_min_var.set(8.0)
    view_state.tth_max_var.set(3.0)
    view_state.phi_min_var.set(7.0)
    view_state.phi_max_var.set(-1.0)
    overlay = _FakeOverlay()
    overlay.visible = True
    overlay_rect = _FakeRect()

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: (None, None),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=None,
        sim_res2=object(),
    )

    assert overlay.visible is False
    assert overlay_rect.visible is True
    assert overlay_rect.xy == (3.0, -1.0)
    assert overlay_rect.width == 5.0
    assert overlay_rect.height == 8.0


def test_update_runtime_integration_region_visuals_updates_raw_overlay() -> None:
    view_state = _range_view_state()
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=state.IntegrationRangeDragState(),
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=_FakeAxis(),
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
    )

    integration_range_drag.update_runtime_integration_region_visuals(
        bindings,
        ai=ai,
        sim_res2=None,
    )

    assert overlay_rect.visible is False
    assert overlay.visible is True
    assert overlay.extent == (0.0, 2.0, 2.0, 0.0)
    assert int(np.sum(overlay.data)) == 6


def test_runtime_integration_region_visuals_callback_uses_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    bindings = object()

    monkeypatch.setattr(
        integration_range_drag,
        "refresh_runtime_integration_region_visuals",
        lambda bound: calls.append(bound),
    )

    callback = integration_range_drag.make_runtime_integration_region_visuals_callback(
        lambda: bindings
    )
    callback()

    assert calls == [bindings]


def test_integration_range_drag_runtime_helpers_handle_suppress_and_caked_drag() -> None:
    axis = _FakeAxis()
    drag_state = state.IntegrationRangeDragState()
    peak_state = state.PeakSelectionState(suppress_drag_press_once=True)
    view_state = _range_view_state()
    drag_rect = _FakeRect()
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    status_messages = []
    sync_calls = []
    draw_calls = []
    schedule_calls = []

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=peak_state,
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(),
        get_detector_angular_maps=lambda ai: ai,
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: True,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: None,
        sync_peak_selection_state=lambda: sync_calls.append(True),
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: None,
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
    )

    suppressed = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=5.0, ydata=-10.0),
    )
    assert suppressed is True
    assert peak_state.suppress_drag_press_once is False
    assert sync_calls == [True]

    started = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=5.0, ydata=-10.0),
    )
    assert started is True
    assert drag_state.active is True
    assert drag_state.mode == "caked"
    assert drag_rect.visible is True
    assert drag_rect.xy == (5.0, -10.0)

    moved = integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=8.0, ydata=-4.0),
    )
    assert moved is True
    assert drag_state.x1 == 8.0
    assert drag_state.y1 == -4.0
    assert drag_rect.width == 3.0
    assert drag_rect.height == 6.0

    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=8.0, ydata=-4.0),
    )
    assert released is True
    assert view_state.tth_min_var.get() == 5.0
    assert view_state.tth_max_var.get() == 8.0
    assert view_state.phi_min_var.get() == -10.0
    assert view_state.phi_max_var.get() == -4.0
    assert schedule_calls == [True]
    assert status_messages[-1] == "Integration region set: 2θ=[5.00, 8.00]°, φ=[-10.00, -4.00]°"
    assert drag_state.active is False
    assert drag_state.mode is None
    assert drag_rect.visible is False
    assert len(draw_calls) >= 3


def test_raw_release_with_incomplete_drag_restores_current_region_visuals() -> None:
    axis = _FakeAxis(xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    drag_state = state.IntegrationRangeDragState(
        active=True,
        mode="raw",
        tth0=10.0,
        phi0=-10.0,
        tth1=None,
        phi1=None,
    )
    view_state = _range_view_state()
    view_state.tth_min_var.set(10.0)
    view_state.tth_max_var.set(22.0)
    view_state.phi_min_var.set(-10.0)
    view_state.phi_max_var.set(2.0)
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    draw_calls = []
    ai = object()
    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=state.PeakSelectionState(),
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=_FakeRect(),
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        last_sim_res2_factory=lambda: "res2",
        draw_idle=lambda: draw_calls.append(True),
    )

    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=None, xdata=None, ydata=None),
    )

    assert released is True
    assert overlay.visible is True
    assert int(np.sum(overlay.data)) == 6
    assert drag_state.active is False
    assert drag_state.mode is None
    assert len(draw_calls) >= 2


def test_integration_range_drag_runtime_helpers_handle_raw_drag_and_callback_bundle(
    monkeypatch,
) -> None:
    axis = _FakeAxis(xlim=(0.0, 2.0), ylim=(0.0, 2.0))
    drag_state = state.IntegrationRangeDragState()
    peak_state = state.PeakSelectionState()
    view_state = _range_view_state()
    drag_rect = _FakeRect()
    overlay = _FakeOverlay()
    overlay_rect = _FakeRect()
    status_messages = []
    schedule_calls = []
    draw_calls = []

    two_theta = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=float,
    )
    phi_vals = np.asarray(
        [
            [-10.0, -9.0, -8.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
        ],
        dtype=float,
    )
    ai = object()

    bindings = integration_range_drag.IntegrationRangeDragBindings(
        drag_state=drag_state,
        peak_selection_state=peak_state,
        range_view_state=view_state,
        ax=axis,
        drag_select_rect=drag_rect,
        integration_region_overlay=overlay,
        integration_region_rect=overlay_rect,
        image_display=_FakeImageDisplay(extent=(0.0, 2.0, 2.0, 0.0)),
        get_detector_angular_maps=lambda ai_arg: (
            (two_theta, phi_vals) if ai_arg is ai else (None, None)
        ),
        range_visible_factory=lambda: True,
        caked_view_enabled_factory=lambda: False,
        unscaled_image_present_factory=lambda: True,
        ai_factory=lambda: ai,
        sync_peak_selection_state=lambda: None,
        schedule_range_update=lambda: schedule_calls.append(True),
        last_sim_res2_factory=lambda: "res2",
        draw_idle=lambda: draw_calls.append(True),
        set_status_text=status_messages.append,
    )

    started = integration_range_drag.handle_runtime_integration_drag_press(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=0.2, ydata=1.1),
    )
    assert started is True
    assert drag_state.active is True
    assert drag_state.mode == "raw"
    assert np.isclose(drag_state.tth0, 20.0)
    assert np.isclose(drag_state.phi0, 0.0)
    assert overlay.visible is True
    assert overlay_rect.visible is False
    assert drag_rect.visible is False

    moved = integration_range_drag.handle_runtime_integration_drag_motion(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=2.0),
    )
    assert moved is True
    assert np.isclose(drag_state.tth1, 32.0)
    assert np.isclose(drag_state.phi1, 12.0)
    assert overlay.extent == (0.0, 2.0, 2.0, 0.0)
    assert int(np.sum(overlay.data)) > 0

    released = integration_range_drag.handle_runtime_integration_drag_release(
        bindings,
        _FakeEvent(button=1, inaxes=axis, xdata=1.8, ydata=2.0),
    )
    assert released is True
    assert view_state.tth_min_var.get() == 20.0
    assert view_state.tth_max_var.get() == 32.0
    assert view_state.phi_min_var.get() == 0.0
    assert view_state.phi_max_var.get() == 12.0
    assert schedule_calls == [True]
    assert status_messages[-1] == "Integration region set: 2θ=[20.00, 32.00]°, φ=[0.00, 12.00]°"
    assert drag_state.active is False
    assert drag_state.mode is None
    assert drag_rect.visible is False
    assert len(draw_calls) >= 3

    callback_calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        integration_range_drag,
        "handle_runtime_integration_drag_press",
        lambda bindings_arg, event: callback_calls.append(("press", bindings_arg, event))
        or True,
    )
    monkeypatch.setattr(
        integration_range_drag,
        "handle_runtime_integration_drag_motion",
        lambda bindings_arg, event: callback_calls.append(("motion", bindings_arg, event))
        or False,
    )
    monkeypatch.setattr(
        integration_range_drag,
        "handle_runtime_integration_drag_release",
        lambda bindings_arg, event: callback_calls.append(("release", bindings_arg, event))
        or True,
    )
    monkeypatch.setattr(
        integration_range_drag,
        "reset_runtime_integration_drag",
        lambda bindings_arg, *, redraw=True: callback_calls.append(
            ("reset", bindings_arg, redraw)
        ),
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = integration_range_drag.make_runtime_integration_range_drag_callbacks(
        build_bindings
    )

    press_event = _FakeEvent(button=1)
    motion_event = _FakeEvent(button=1)
    release_event = _FakeEvent(button=1)
    assert callbacks.on_press(press_event) is True
    assert callbacks.on_motion(motion_event) is False
    assert callbacks.on_release(release_event) is True
    callbacks.reset()

    assert callback_calls == [
        ("press", "bindings-1", press_event),
        ("motion", "bindings-2", motion_event),
        ("release", "bindings-3", release_event),
        ("reset", "bindings-4", True),
    ]
