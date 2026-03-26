import numpy as np

from ra_sim.gui import integration_range_drag, state


class _FakeVar:
    def __init__(self, value=0.0):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


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
