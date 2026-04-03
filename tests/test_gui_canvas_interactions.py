from types import SimpleNamespace

import pytest

from ra_sim.gui import canvas_interactions, state


class _FakeBbox:
    def __init__(self, *, x0=10.0, y0=20.0, width=200.0, height=100.0):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height


class _FakeAxis:
    def __init__(self, *, xlim=(0.0, 100.0), ylim=(100.0, 0.0)):
        self.bbox = _FakeBbox()
        self._xlim = tuple(float(value) for value in xlim)
        self._ylim = tuple(float(value) for value in ylim)
        self.set_xlim_calls = []
        self.set_ylim_calls = []

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xlim(self, left, right):
        self._xlim = (float(left), float(right))
        self.set_xlim_calls.append(self._xlim)

    def set_ylim(self, bottom, top):
        self._ylim = (float(bottom), float(top))
        self.set_ylim_calls.append(self._ylim)


class _FakeEvent:
    def __init__(
        self,
        *,
        button=1,
        inaxes=None,
        xdata=None,
        ydata=None,
        x=0.0,
        y=0.0,
        step=0.0,
    ):
        self.button = button
        self.inaxes = inaxes
        self.xdata = xdata
        self.ydata = ydata
        self.x = x
        self.y = y
        self.step = step


class _PeakCallbacks:
    def __init__(self):
        self.calls = []

    def set_hkl_pick_mode(self, enabled, message=None):
        self.calls.append(("set", bool(enabled), message))

    def select_peak_from_canvas_click(self, col, row):
        self.calls.append(("click", float(col), float(row)))
        return True


class _AnalysisPeakCallbacks:
    def __init__(self):
        self.calls = []

    def set_pick_mode(self, enabled, message=None):
        self.calls.append(("set", bool(enabled), message))

    def select_peak_from_canvas_click(self, col, row):
        self.calls.append(("click", float(col), float(row)))
        return True


class _DragCallbacks:
    def __init__(self):
        self.calls = []

    def on_press(self, event):
        self.calls.append(("press", event))
        return True

    def on_motion(self, event):
        self.calls.append(("motion", event))
        return False

    def on_release(self, event):
        self.calls.append(("release", event))
        return True


def test_canvas_interaction_binding_factory_builds_live_bindings(monkeypatch) -> None:
    calls = []
    counters = {"status": 0, "draw": 0}

    monkeypatch.setattr(
        canvas_interactions,
        "CanvasInteractionBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def build_status():
        counters["status"] += 1
        idx = counters["status"]
        return lambda text: f"status-{idx}:{text}"

    def build_draw():
        counters["draw"] += 1
        idx = counters["draw"]
        return lambda: f"draw-{idx}"

    factory = canvas_interactions.make_runtime_canvas_interaction_bindings_factory(
        axis="axis",
        geometry_runtime_state="geometry-runtime",
        geometry_preview_state="preview-state",
        geometry_manual_state="manual-state",
        peak_selection_state="peak-state",
        peak_selection_callbacks="peak-callbacks",
        integration_range_drag_callbacks="drag-callbacks",
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis, x, y: (x, y),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: False,
        set_geometry_status_text_factory=build_status,
        draw_idle_factory=build_draw,
    )

    assert factory()["axis"] == "axis"
    assert factory()["axis"] == "axis"
    assert calls[0]["geometry_runtime_state"] == "geometry-runtime"
    assert calls[0]["set_geometry_status_text"] is not calls[1]["set_geometry_status_text"]
    assert calls[0]["draw_idle"] is not calls[1]["draw_idle"]


def test_canvas_click_routes_cancel_manual_preview_and_hkl_paths() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState()
    geometry_runtime = state.GeometryRuntimeState()
    preview_state = state.GeometryPreviewState()
    manual_state = state.ManualGeometryState()
    calls = []

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        geometry_preview_state=preview_state,
        geometry_manual_state=manual_state,
        peak_selection_state=peak_state,
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda enabled, message=None: calls.append(
            ("manual_mode", bool(enabled), message)
        ),
        set_geometry_preview_exclude_mode=lambda enabled, message=None: calls.append(
            ("exclude_mode", bool(enabled), message)
        ),
        toggle_geometry_manual_selection_at=lambda col, row: calls.append(
            ("manual_toggle", float(col), float(row))
        ),
        toggle_live_geometry_preview_exclusion_at=lambda col, row: calls.append(
            ("exclude_toggle", float(col), float(row))
        ),
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: False,
    )

    peak_state.hkl_pick_armed = True
    assert (
        canvas_interactions.handle_runtime_canvas_click(
            bindings,
            _FakeEvent(button=3),
        )
        is True
    )
    assert peak_callbacks.calls == [("set", False, "HKL image-pick canceled.")]

    peak_state.hkl_pick_armed = False
    geometry_runtime.manual_pick_armed = True
    assert (
        canvas_interactions.handle_runtime_canvas_click(
            bindings,
            _FakeEvent(button=3),
        )
        is True
    )
    assert calls[-1] == ("manual_mode", False, "Manual geometry picking canceled.")

    geometry_runtime.manual_pick_armed = False
    preview_state.exclude_armed = True
    assert (
        canvas_interactions.handle_runtime_canvas_click(
            bindings,
            _FakeEvent(button=3),
        )
        is True
    )
    assert calls[-1] == ("exclude_mode", False, "Preview exclusion mode canceled.")

    preview_state.exclude_armed = False
    geometry_runtime.manual_pick_armed = True
    assert (
        canvas_interactions.handle_runtime_canvas_click(
            bindings,
            _FakeEvent(button=1, inaxes=axis, xdata=5.5, ydata=6.5),
        )
        is True
    )
    assert calls[-1] == ("manual_toggle", 5.5, 6.5)

    geometry_runtime.manual_pick_armed = False
    preview_state.exclude_armed = True
    assert (
        canvas_interactions.handle_runtime_canvas_click(
            bindings,
            _FakeEvent(button=1, inaxes=axis, xdata=7.5, ydata=8.5),
        )
        is True
    )
    assert calls[-1] == ("exclude_toggle", 7.5, 8.5)

    preview_state.exclude_armed = False
    peak_state.hkl_pick_armed = True
    assert (
        canvas_interactions.handle_runtime_canvas_click(
            bindings,
            _FakeEvent(button=1, inaxes=axis, xdata=9.5, ydata=10.5),
        )
        is True
    )
    assert peak_callbacks.calls[-1] == ("click", 9.5, 10.5)


def test_canvas_click_routes_analysis_peak_pick_before_caked_drag() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    analysis_callbacks = _AnalysisPeakCallbacks()
    drag_callbacks = _DragCallbacks()

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(),
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: True,
        analysis_peak_state=state.AnalysisPeakSelectionState(pick_armed=True),
        analysis_peak_callbacks=analysis_callbacks,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=14.5, ydata=-7.5)

    assert canvas_interactions.handle_runtime_canvas_click(bindings, event) is True
    assert canvas_interactions.handle_runtime_canvas_press(bindings, event) is True
    assert analysis_callbacks.calls == [("click", 14.5, -7.5)]
    assert drag_callbacks.calls == []


def test_canvas_right_click_cancels_analysis_peak_pick_mode() -> None:
    axis = _FakeAxis()
    analysis_callbacks = _AnalysisPeakCallbacks()
    geometry_runtime = state.GeometryRuntimeState()

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(),
        peak_selection_callbacks=_PeakCallbacks(),
        integration_range_drag_callbacks=_DragCallbacks(),
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: True,
        analysis_peak_state=state.AnalysisPeakSelectionState(pick_armed=True),
        analysis_peak_callbacks=analysis_callbacks,
    )

    assert canvas_interactions.handle_runtime_canvas_click(bindings, _FakeEvent(button=3)) is True
    assert analysis_callbacks.calls == [("set", False, "Analysis peak picking canceled.")]
    assert getattr(geometry_runtime, "_suppress_pan_press_once", False) is True


def test_canvas_first_manual_pick_click_does_not_immediately_place_background_point() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState()
    geometry_runtime = state.GeometryRuntimeState(manual_pick_armed=True)
    preview_state = state.GeometryPreviewState()
    manual_state = state.ManualGeometryState(pick_session={})
    calls = []

    def _toggle_selection(col, row):
        calls.append(("toggle", float(col), float(row)))
        manual_state.pick_session = {
            "group_key": ("q", 1),
            "group_entries": [{"label": "sim-1"}],
            "pending_entries": [],
            "zoom_active": False,
        }
        peak_state.suppress_drag_press_once = True

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        geometry_preview_state=preview_state,
        geometry_manual_state=manual_state,
        peak_selection_state=peak_state,
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: bool(manual_state.pick_session.get("group_key")),
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=_toggle_selection,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda col, row, **kwargs: calls.append(
            ("zoom", float(col), float(row), kwargs)
        ),
        update_geometry_manual_pick_preview=lambda col, row, **kwargs: calls.append(
            ("preview", float(col), float(row), kwargs)
        ),
        place_geometry_manual_selection_at=lambda col, row: calls.append(
            ("place", float(col), float(row))
        ),
        clear_geometry_manual_preview_artists=lambda **kwargs: calls.append(
            ("clear", kwargs)
        ),
        restore_geometry_manual_pick_view=lambda **kwargs: calls.append(
            ("restore", kwargs)
        ),
        render_current_geometry_manual_pairs=lambda **kwargs: calls.append(
            ("render", kwargs)
        )
        or True,
        caked_view_enabled_factory=lambda: False,
        set_geometry_status_text=lambda text: calls.append(("status", text)),
        draw_idle=lambda: calls.append(("draw",)),
    )

    click_event = _FakeEvent(
        button=1,
        inaxes=axis,
        xdata=12.0,
        ydata=18.0,
        x=110.0,
        y=70.0,
    )
    release_event = _FakeEvent(
        button=1,
        inaxes=axis,
        xdata=12.0,
        ydata=18.0,
    )

    assert canvas_interactions.handle_runtime_canvas_click(bindings, click_event) is True
    assert peak_state.suppress_drag_press_once is True
    assert canvas_interactions.handle_runtime_canvas_press(bindings, click_event) is True
    assert peak_state.suppress_drag_press_once is False
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True

    assert calls == [
        ("toggle", 12.0, 18.0),
    ]
    assert drag_callbacks.calls == [("release", release_event)]


def test_canvas_press_does_not_start_integration_drag_while_manual_qr_pick_is_armed() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState()
    geometry_runtime = state.GeometryRuntimeState(manual_pick_armed=True)
    preview_state = state.GeometryPreviewState()
    manual_state = state.ManualGeometryState()
    calls = []

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        geometry_preview_state=preview_state,
        geometry_manual_state=manual_state,
        peak_selection_state=peak_state,
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda col, row: calls.append(
            ("toggle", float(col), float(row))
        ),
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: True,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=12.0, ydata=18.0)

    assert canvas_interactions.handle_runtime_canvas_click(bindings, event) is True
    assert canvas_interactions.handle_runtime_canvas_press(bindings, event) is True

    assert calls == [("toggle", 12.0, 18.0)]
    assert drag_callbacks.calls == []


def test_canvas_press_does_not_start_integration_drag_while_hkl_pick_is_armed() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(hkl_pick_armed=True),
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: False,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=12.0, ydata=18.0)

    assert canvas_interactions.handle_runtime_canvas_press(bindings, event) is True
    assert peak_callbacks.calls == []
    assert drag_callbacks.calls == []


def test_canvas_press_motion_and_release_prefer_manual_pick_session() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState()
    geometry_runtime = state.GeometryRuntimeState(manual_pick_armed=True)
    preview_state = state.GeometryPreviewState()
    manual_state = state.ManualGeometryState(pick_session={"zoom_active": True})
    calls = []

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        geometry_preview_state=preview_state,
        geometry_manual_state=manual_state,
        peak_selection_state=peak_state,
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: True,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda col, row, **kwargs: calls.append(
            ("zoom", float(col), float(row), kwargs)
        ),
        update_geometry_manual_pick_preview=lambda col, row, **kwargs: calls.append(
            ("preview", float(col), float(row), kwargs)
        ),
        place_geometry_manual_selection_at=lambda col, row: calls.append(
            ("place", float(col), float(row))
        ),
        clear_geometry_manual_preview_artists=lambda **kwargs: calls.append(
            ("clear", kwargs)
        ),
        restore_geometry_manual_pick_view=lambda **kwargs: calls.append(
            ("restore", kwargs)
        ),
        render_current_geometry_manual_pairs=lambda **kwargs: calls.append(
            ("render", kwargs)
        )
        or True,
        caked_view_enabled_factory=lambda: False,
        set_geometry_status_text=lambda text: calls.append(("status", text)),
        draw_idle=lambda: calls.append(("draw",)),
    )

    press_event = _FakeEvent(
        button=1,
        inaxes=axis,
        xdata=15.0,
        ydata=25.0,
        x=110.0,
        y=70.0,
    )
    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert calls[0][0] == "zoom"
    assert calls[0][1:3] == (15.0, 25.0)
    assert calls[0][3]["anchor_fraction_x"] == 0.5
    assert calls[0][3]["anchor_fraction_y"] == 0.5
    assert calls[1] == ("preview", 15.0, 25.0, {"force": True})

    motion_event = _FakeEvent(button=1, inaxes=axis, xdata=16.0, ydata=26.0)
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert calls[2] == ("preview", 16.0, 26.0, {})

    release_event = _FakeEvent(button=1, inaxes=axis, xdata=17.0, ydata=27.0)
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert calls[3] == ("place", 17.0, 27.0)

    calls.clear()
    cancel_release = _FakeEvent(button=1, inaxes=None, xdata=None, ydata=None)
    assert (
        canvas_interactions.handle_runtime_canvas_release(bindings, cancel_release)
        is True
    )
    assert calls == [
        ("clear", {"redraw": False}),
        ("restore", {"redraw": False}),
        ("render", {"update_status": False}),
        ("status", "Manual point placement canceled: release inside the image."),
        ("draw",),
    ]


def test_canvas_interaction_callback_bundle_delegates_live_bindings(monkeypatch) -> None:
    callback_calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_click",
        lambda bindings_arg, event: callback_calls.append(("click", bindings_arg, event))
        or True,
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_press",
        lambda bindings_arg, event: callback_calls.append(("press", bindings_arg, event))
        or False,
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_motion",
        lambda bindings_arg, event: callback_calls.append(("motion", bindings_arg, event))
        or True,
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_release",
        lambda bindings_arg, event: callback_calls.append(
            ("release", bindings_arg, event)
        )
        or False,
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_scroll",
        lambda bindings_arg, event: callback_calls.append(("scroll", bindings_arg, event))
        or True,
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = canvas_interactions.make_runtime_canvas_interaction_callbacks(
        build_bindings
    )
    click_event = _FakeEvent(button=1)
    press_event = _FakeEvent(button=1)
    motion_event = _FakeEvent(button=1)
    release_event = _FakeEvent(button=1)
    scroll_event = _FakeEvent(button="up", step=1.0)

    assert callbacks.on_click(click_event) is True
    assert callbacks.on_press(press_event) is False
    assert callbacks.on_motion(motion_event) is True
    assert callbacks.on_release(release_event) is False
    assert callbacks.on_scroll(scroll_event) is True

    assert callback_calls == [
        ("click", "bindings-1", click_event),
        ("press", "bindings-2", press_event),
        ("motion", "bindings-3", motion_event),
        ("release", "bindings-4", release_event),
        ("scroll", "bindings-5", scroll_event),
    ]


def test_canvas_interaction_callback_bundle_catches_runtime_errors(monkeypatch) -> None:
    status_messages = []
    draw_calls = []
    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=_FakeAxis(),
        geometry_runtime_state=state.GeometryRuntimeState(),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(),
        peak_selection_callbacks=_PeakCallbacks(),
        integration_range_drag_callbacks=_DragCallbacks(),
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args, **_kwargs: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args, **_kwargs: None,
        clamp_to_axis_view=lambda axis, x, y: (x, y),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args, **_kwargs: None,
        clear_geometry_manual_preview_artists=lambda *_args, **_kwargs: None,
        restore_geometry_manual_pick_view=lambda *_args, **_kwargs: None,
        render_current_geometry_manual_pairs=lambda *_args, **_kwargs: None,
        caked_view_enabled_factory=lambda: False,
        set_geometry_status_text=lambda text: status_messages.append(str(text)),
        draw_idle=lambda: draw_calls.append(True),
    )

    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_release",
        lambda _bindings_arg, _event: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(canvas_interactions.traceback, "print_exc", lambda: None)

    callbacks = canvas_interactions.make_runtime_canvas_interaction_callbacks(
        lambda: bindings
    )

    assert callbacks.on_release(_FakeEvent(button=1)) is False
    assert status_messages == ["Canvas interaction canceled after an internal error."]
    assert draw_calls == [True]


def test_canvas_handlers_delegate_to_drag_callbacks_outside_manual_pick_session() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(manual_pick_armed=False),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(),
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: False,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=1.0, ydata=2.0)
    assert canvas_interactions.handle_runtime_canvas_press(bindings, event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, event) is False
    assert canvas_interactions.handle_runtime_canvas_release(bindings, event) is True
    assert [name for name, _event in drag_callbacks.calls] == [
        "press",
        "motion",
        "release",
    ]


def test_canvas_right_drag_pans_detector_view() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    drag_callbacks = _DragCallbacks()
    draw_calls = []
    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(),
        peak_selection_callbacks=_PeakCallbacks(),
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: False,
        draw_idle=lambda: draw_calls.append(True),
    )

    press_event = _FakeEvent(button=3, inaxes=axis, xdata=4.0, ydata=5.0)
    motion_event = _FakeEvent(button=3, inaxes=axis, xdata=6.0, ydata=9.0)
    release_event = _FakeEvent(button=3, inaxes=axis, xdata=6.0, ydata=9.0)

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert axis.get_xlim() == (-2.0, 8.0)
    assert axis.get_ylim() == (16.0, -4.0)
    assert draw_calls == [True]
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is False
    assert drag_callbacks.calls == []


def test_canvas_scroll_zooms_caked_view_about_cursor() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(-180.0, 180.0))
    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(),
        peak_selection_callbacks=_PeakCallbacks(),
        integration_range_drag_callbacks=_DragCallbacks(),
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: True,
        draw_idle=lambda: None,
    )

    zoom_in = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
    )
    zoom_out = _FakeEvent(
        button="down",
        step=-1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    xlim_after_zoom_in = axis.get_xlim()
    ylim_after_zoom_in = axis.get_ylim()
    assert xlim_after_zoom_in == pytest.approx((4.166666666666668, 87.5))
    assert ylim_after_zoom_in == pytest.approx((-155.0, 145.0))

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_out) is True
    assert axis.get_xlim() == pytest.approx((0.0, 100.0))
    assert axis.get_ylim() == pytest.approx((-180.0, 180.0))


def test_restore_axis_view_keeps_existing_zoom_within_new_bounds() -> None:
    axis = _FakeAxis(xlim=(20.0, 40.0), ylim=(90.0, 60.0))

    restored = canvas_interactions.restore_axis_view(
        axis,
        preserved_limits=canvas_interactions.capture_axis_limits(axis),
        default_xlim=(0.0, 100.0),
        default_ylim=(100.0, 0.0),
        preserve=True,
    )

    assert restored is True
    assert axis.get_xlim() == pytest.approx((20.0, 40.0))
    assert axis.get_ylim() == pytest.approx((90.0, 60.0))


def test_restore_axis_view_clamps_preserved_window_into_smaller_bounds() -> None:
    axis = _FakeAxis(xlim=(80.0, 120.0), ylim=(30.0, -10.0))

    restored = canvas_interactions.restore_axis_view(
        axis,
        preserved_limits=canvas_interactions.capture_axis_limits(axis),
        default_xlim=(10.0, 90.0),
        default_ylim=(20.0, -20.0),
        preserve=True,
    )

    assert restored is True
    assert axis.get_xlim() == pytest.approx((50.0, 90.0))
    assert axis.get_ylim() == pytest.approx((20.0, -20.0))


def test_restore_axis_view_uses_defaults_when_preserve_disabled() -> None:
    axis = _FakeAxis(xlim=(20.0, 40.0), ylim=(90.0, 60.0))

    restored = canvas_interactions.restore_axis_view(
        axis,
        preserved_limits=canvas_interactions.capture_axis_limits(axis),
        default_xlim=(0.0, 100.0),
        default_ylim=(100.0, 0.0),
        preserve=False,
    )

    assert restored is True
    assert axis.get_xlim() == pytest.approx((0.0, 100.0))
    assert axis.get_ylim() == pytest.approx((100.0, 0.0))
