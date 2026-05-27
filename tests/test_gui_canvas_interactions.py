import json
from types import SimpleNamespace

import pytest

from ra_sim.gui import canvas_interactions, state

try:
    from matplotlib.backend_bases import MouseButton
except Exception:  # pragma: no cover - Matplotlib is available in normal test runs.
    MouseButton = None


_RIGHT_BUTTON_FORMS: list[object] = [3, "3", "right", "button3", "MouseButton.RIGHT"]
if MouseButton is not None:
    _RIGHT_BUTTON_FORMS.append(MouseButton.RIGHT)


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


def _basic_canvas_bindings(
    *,
    axis,
    geometry_runtime_state=None,
    geometry_preview_state=None,
    geometry_manual_state=None,
    peak_selection_state=None,
    peak_selection_callbacks=None,
    integration_range_drag_callbacks=None,
    manual_pick_session_active=None,
    caked_view_enabled=False,
    set_geometry_manual_pick_mode=None,
    set_geometry_preview_exclude_mode=None,
    begin_live_interaction=None,
    touch_live_interaction=None,
    end_live_interaction=None,
    draw_idle=None,
    preview_view_limits=None,
    commit_preview_view=None,
    analysis_peak_state=None,
    analysis_peak_callbacks=None,
    beam_center_pick_armed=None,
    beam_center_pick_session_active=None,
    set_beam_center_pick_mode=None,
    start_beam_center_pick_at=None,
    update_beam_center_pick_preview=None,
    commit_beam_center_pick_at=None,
    cancel_beam_center_pick=None,
):
    return canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime_state or state.GeometryRuntimeState(),
        geometry_preview_state=geometry_preview_state or state.GeometryPreviewState(),
        geometry_manual_state=geometry_manual_state or state.ManualGeometryState(),
        peak_selection_state=peak_selection_state or state.PeakSelectionState(),
        peak_selection_callbacks=peak_selection_callbacks or _PeakCallbacks(),
        integration_range_drag_callbacks=integration_range_drag_callbacks or _DragCallbacks(),
        manual_pick_session_active=manual_pick_session_active or (lambda: False),
        set_geometry_manual_pick_mode=(
            set_geometry_manual_pick_mode or (lambda *_args, **_kwargs: None)
        ),
        set_geometry_preview_exclude_mode=(
            set_geometry_preview_exclude_mode or (lambda *_args, **_kwargs: None)
        ),
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=(
            caked_view_enabled
            if callable(caked_view_enabled)
            else (lambda: bool(caked_view_enabled))
        ),
        analysis_peak_state=analysis_peak_state,
        analysis_peak_callbacks=analysis_peak_callbacks,
        draw_idle=draw_idle,
        begin_live_interaction=begin_live_interaction,
        touch_live_interaction=touch_live_interaction,
        end_live_interaction=end_live_interaction,
        beam_center_pick_armed=beam_center_pick_armed,
        beam_center_pick_session_active=beam_center_pick_session_active,
        set_beam_center_pick_mode=set_beam_center_pick_mode,
        start_beam_center_pick_at=start_beam_center_pick_at,
        update_beam_center_pick_preview=update_beam_center_pick_preview,
        commit_beam_center_pick_at=commit_beam_center_pick_at,
        cancel_beam_center_pick=cancel_beam_center_pick,
        preview_view_limits=preview_view_limits,
        commit_preview_view=commit_preview_view,
    )


def test_canvas_interaction_binding_factory_builds_live_bindings(monkeypatch) -> None:
    calls = []
    counters = {
        "status": 0,
        "draw": 0,
        "begin": 0,
        "touch": 0,
        "end": 0,
        "preview": 0,
        "commit": 0,
        "clear": 0,
    }

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

    def build_begin():
        counters["begin"] += 1
        idx = counters["begin"]
        return lambda: f"begin-{idx}"

    def build_touch():
        counters["touch"] += 1
        idx = counters["touch"]
        return lambda: f"touch-{idx}"

    def build_end():
        counters["end"] += 1
        idx = counters["end"]
        return lambda: f"end-{idx}"

    def build_preview():
        counters["preview"] += 1
        idx = counters["preview"]
        return lambda xlim, ylim: f"preview-{idx}:{xlim}:{ylim}"

    def build_commit():
        counters["commit"] += 1
        idx = counters["commit"]
        return lambda: f"commit-{idx}"

    def build_clear():
        counters["clear"] += 1
        idx = counters["clear"]
        return lambda: f"clear-{idx}"

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
        begin_live_interaction_factory=build_begin,
        touch_live_interaction_factory=build_touch,
        end_live_interaction_factory=build_end,
        preview_view_limits_factory=build_preview,
        commit_preview_view_factory=build_commit,
        clear_preview_view_factory=build_clear,
        beam_center_pick_armed_factory=lambda: True,
        beam_center_pick_session_active=lambda: False,
        set_beam_center_pick_mode=lambda *_args, **_kwargs: None,
        start_beam_center_pick_at=lambda *_args, **_kwargs: None,
        update_beam_center_pick_preview=lambda *_args, **_kwargs: None,
        commit_beam_center_pick_at=lambda *_args, **_kwargs: None,
        cancel_beam_center_pick=lambda *_args, **_kwargs: None,
    )

    assert factory()["axis"] == "axis"
    assert factory()["axis"] == "axis"
    assert calls[0]["geometry_runtime_state"] == "geometry-runtime"
    assert calls[0]["set_geometry_status_text"] is not calls[1]["set_geometry_status_text"]
    assert calls[0]["draw_idle"] is not calls[1]["draw_idle"]
    assert calls[0]["begin_live_interaction"] is not calls[1]["begin_live_interaction"]
    assert calls[0]["touch_live_interaction"] is not calls[1]["touch_live_interaction"]
    assert calls[0]["end_live_interaction"] is not calls[1]["end_live_interaction"]
    assert calls[0]["preview_view_limits"] is not calls[1]["preview_view_limits"]
    assert calls[0]["commit_preview_view"] is not calls[1]["commit_preview_view"]
    assert calls[0]["clear_preview_view"] is not calls[1]["clear_preview_view"]
    assert calls[0]["beam_center_pick_armed"] is True
    assert callable(calls[0]["beam_center_pick_session_active"])
    assert callable(calls[0]["set_beam_center_pick_mode"])
    assert callable(calls[0]["start_beam_center_pick_at"])
    assert callable(calls[0]["update_beam_center_pick_preview"])
    assert callable(calls[0]["commit_beam_center_pick_at"])
    assert callable(calls[0]["cancel_beam_center_pick"])


def test_beam_center_pick_press_motion_release_uses_priority_canvas_route(
    monkeypatch,
    tmp_path,
) -> None:
    from ra_sim.gui._runtime import runtime_session

    axis = _FakeAxis()
    geometry_runtime = state.GeometryRuntimeState()
    geometry_runtime.manual_pick_armed = True
    geometry_runtime.beam_center_pick_armed = True
    peak_state = state.PeakSelectionState()
    peak_state.hkl_pick_armed = True
    drag_callbacks = _DragCallbacks()
    calls = []
    session_active = {"value": False}
    scheduled_reads = {}

    class _TraceVar:
        def __init__(self, value):
            self.value = float(value)
            self.callbacks = []

        def get(self):
            return self.value

        def set(self, value):
            self.value = float(value)
            for callback in list(self.callbacks):
                callback("name", "index", "write")

        def trace_add(self, _mode, callback):
            self.callbacks.append(callback)
            return f"trace-{len(self.callbacks)}"

    class _Entry:
        def __init__(self, value=""):
            self.value = str(value)
            self.bindings = {}

        def bind(self, event_name, callback, add=None):
            self.bindings.setdefault(event_name, []).append((callback, add))

        def get(self):
            return self.value

        def cget(self, key):
            if key == "textvariable":
                return ""
            raise KeyError(key)

        def delete(self, *_args):
            self.value = ""

        def insert(self, _index, text):
            self.value = str(text)

    class _SliderRow:
        def __init__(self, scale, entry):
            self.children = [scale, entry]

        def winfo_children(self):
            return list(self.children)

    class _VisibleScale:
        def __init__(self, value_var, entry):
            self.value_var = value_var
            self.entry = entry
            self.value = float(value_var.get())
            self.bounds = {"from": 0.0, "to": 3000.0}
            self.master = _SliderRow(self, entry)

        def cget(self, key):
            return self.bounds[key]

        def configure(self, **kwargs):
            if "from_" in kwargs:
                self.bounds["from"] = float(kwargs["from_"])
            if "to" in kwargs:
                self.bounds["to"] = float(kwargs["to"])

        def set(self, value):
            self.value = float(value)
            self.value_var.set(self.value)
            self.entry.value = str(int(round(self.value)))

        def get(self):
            return self.value

    class _Marker:
        def __init__(self):
            self.xdata = []
            self.ydata = []
            self.visible = False

        def set_xdata(self, values):
            self.xdata = list(values)

        def set_ydata(self, values):
            self.ydata = list(values)

        def set_visible(self, value):
            self.visible = bool(value)

        def get_xdata(self):
            return list(self.xdata)

        def get_ydata(self):
            return list(self.ydata)

    class _Root:
        def __init__(self):
            self.callbacks = []

        def after(self, _delay_ms, callback):
            self.callbacks.append(callback)
            return f"after-{len(self.callbacks)}"

        def update(self):
            while self.callbacks:
                callback = self.callbacks.pop(0)
                callback()

        def update_idletasks(self):
            pass

    row_var = _TraceVar(0.0)
    col_var = _TraceVar(0.0)
    row_entry = _Entry("0")
    col_entry = _Entry("0")
    row_scale = _VisibleScale(row_var, row_entry)
    col_scale = _VisibleScale(col_var, col_entry)
    marker = _Marker()
    root = _Root()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RA_SIM_TRACE_BEAM_CENTER", "1")
    monkeypatch.setattr(runtime_session, "_beam_center_trace_counter", 0, raising=False)
    monkeypatch.setattr(runtime_session, "_beam_center_trace_hooks_attached", False, raising=False)
    monkeypatch.setattr(runtime_session, "_beam_center_trace_hook_refs", [], raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_debug_expected_beam_center_after_pick",
        None,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_debug_beam_center_overwrite_reported",
        False,
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "center_x_var", row_var, raising=False)
    monkeypatch.setattr(runtime_session, "center_y_var", col_var, raising=False)
    monkeypatch.setattr(runtime_session, "center_x_scale", row_scale, raising=False)
    monkeypatch.setattr(runtime_session, "center_y_scale", col_scale, raising=False)
    monkeypatch.setattr(runtime_session, "center_marker", marker, raising=False)
    monkeypatch.setattr(runtime_session, "root", root, raising=False)
    monkeypatch.setattr(runtime_session, "image_size", 3000.0, raising=False)
    monkeypatch.setattr(runtime_session, "DISPLAY_ROTATE_K", -1, raising=False)
    monkeypatch.setattr(runtime_session, "geometry_runtime_state", geometry_runtime, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "background_runtime_state",
        SimpleNamespace(current_background_index=0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            profile_cache={},
            primary_relative_hit_table_cache_center=None,
            secondary_relative_hit_table_cache_center=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_beam_center_coordinate_shape",
        lambda: (3000, 3000),
    )
    monkeypatch.setattr(
        runtime_session,
        "_beam_center_pick_session_active",
        lambda: bool(session_active["value"]),
    )
    monkeypatch.setattr(
        runtime_session,
        "_beam_center_spot_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        runtime_session,
        "_current_geometry_fit_caked_roi_preview_enabled",
        lambda: False,
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "detector")
    monkeypatch.setattr(runtime_session, "_set_beam_center_pick_status", lambda _text: None)
    monkeypatch.setattr(runtime_session, "_restore_beam_center_pick_view", lambda **_kw: None)
    monkeypatch.setattr(
        runtime_session,
        "_clear_geometry_manual_preview_artists",
        lambda **_kw: None,
    )
    monkeypatch.setattr(runtime_session, "_set_beam_center_pick_cursor", lambda _enabled: None)
    monkeypatch.setattr(runtime_session, "_update_beam_center_pick_button_label", lambda: None)
    monkeypatch.setattr(runtime_session, "_clear_geometry_fit_dataset_cache", lambda: None)
    monkeypatch.setattr(runtime_session, "_invalidate_simulation_cache", lambda: None)

    def _runtime_preview(display_col, display_row, **_kwargs):
        geometry_runtime.beam_center_pick_session["refined_col"] = float(display_col)
        geometry_runtime.beam_center_pick_session["refined_row"] = float(display_row)
        return True

    def _runtime_schedule_update():
        runtime_session._trace_beam_center("update.scheduled.before")

        def _after_callback():
            center_pair = (float(row_var.get()), float(col_var.get()))
            scheduled_reads["runtime"] = center_pair
            scheduled_reads["simulation"] = center_pair
            scheduled_reads["remap"] = center_pair
            runtime_session._trace_beam_center_writer(
                "update.after_runtime_read",
                new_pair=center_pair,
            )
            runtime_session._trace_beam_center_writer(
                "remap.after_read",
                new_pair=center_pair,
                current_center_pair=center_pair,
            )

        token = root.after(0, _after_callback)
        runtime_session._trace_beam_center("update.scheduled.after", queued_token=token)

    monkeypatch.setattr(
        runtime_session,
        "_update_beam_center_pick_preview",
        _runtime_preview,
    )
    monkeypatch.setattr(runtime_session, "schedule_update", _runtime_schedule_update)
    runtime_session._attach_beam_center_widget_trace_hooks()

    def _start(col, row, **kwargs):
        session_active["value"] = True
        geometry_runtime.beam_center_pick_session = {
            "refined_col": float(col),
            "refined_row": float(row),
        }
        calls.append(("start", float(col), float(row), dict(kwargs)))
        return True

    def _commit(col, row):
        calls.append(("commit", float(col), float(row)))
        return runtime_session._commit_beam_center_pick_at(float(col), float(row))

    bindings = _basic_canvas_bindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        peak_selection_state=peak_state,
        integration_range_drag_callbacks=drag_callbacks,
        manual_pick_session_active=lambda: True,
        beam_center_pick_armed=lambda: True,
        beam_center_pick_session_active=lambda: bool(session_active["value"]),
        start_beam_center_pick_at=_start,
        update_beam_center_pick_preview=(
            lambda col, row: calls.append(("preview", float(col), float(row)))
        ),
        commit_beam_center_pick_at=_commit,
    )

    press = _FakeEvent(button=1, inaxes=axis, xdata=1456.0, ydata=1607.0, x=110.0, y=70.0)
    assert canvas_interactions.handle_runtime_canvas_press(bindings, press) is True
    assert calls[0][0:3] == ("start", 1456.0, 1607.0)
    assert calls[0][3]["anchor_fraction_x"] == pytest.approx(0.5)
    assert calls[0][3]["anchor_fraction_y"] == pytest.approx(0.5)

    motion = _FakeEvent(button=1, inaxes=axis, xdata=1456.0, ydata=1607.0)
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion) is True
    assert calls[1] == ("preview", 1456.0, 1607.0)

    release = _FakeEvent(button=1, inaxes=axis, xdata=1456.0, ydata=1607.0)
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release) is True
    assert calls[2] == ("commit", 1456.0, 1607.0)
    root.update()
    root.update_idletasks()
    runtime_session._trace_beam_center_after_idle_drain()

    assert row_var.get() == pytest.approx(1607.0)
    assert col_var.get() == pytest.approx(1544.0)
    assert row_scale.get() == pytest.approx(1607.0)
    assert col_scale.get() == pytest.approx(1544.0)
    assert float(row_entry.get()) == pytest.approx(1607.0)
    assert float(col_entry.get()) == pytest.approx(1544.0)
    assert scheduled_reads["runtime"] == pytest.approx((1607.0, 1544.0))
    assert scheduled_reads["simulation"] == pytest.approx((1607.0, 1544.0))
    assert scheduled_reads["remap"] == pytest.approx((1607.0, 1544.0))
    assert marker.xdata == pytest.approx([1456.0])
    assert marker.ydata == pytest.approx([1607.0])
    assert marker.visible is True

    trace_path = tmp_path / "debug" / "beam_center_trace.jsonl"
    trace_records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "BEAM_CENTER_OVERWRITE" not in {
        str(record.get("event_name")) for record in trace_records
    }
    assert any(record.get("event_name") == "tk.after_idle_drain" for record in trace_records)
    assert drag_callbacks.calls == []


def test_beam_center_pick_right_click_cancels_before_other_modes() -> None:
    axis = _FakeAxis()
    geometry_runtime = state.GeometryRuntimeState()
    geometry_runtime.manual_pick_armed = True
    peak_state = state.PeakSelectionState()
    peak_state.hkl_pick_armed = True
    peak_callbacks = _PeakCallbacks()
    calls = []
    bindings = _basic_canvas_bindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        peak_selection_state=peak_state,
        peak_selection_callbacks=peak_callbacks,
        beam_center_pick_armed=lambda: True,
        set_beam_center_pick_mode=(
            lambda enabled, message=None: calls.append(
                ("beam", bool(enabled), message)
            )
        ),
        set_geometry_manual_pick_mode=(
            lambda enabled, message=None: calls.append(
                ("manual", bool(enabled), message)
            )
        ),
    )

    event = _FakeEvent(button=3, inaxes=axis, xdata=10.0, ydata=11.0)
    assert canvas_interactions.handle_runtime_canvas_click(bindings, event) is True
    assert calls == [("beam", False, "Beam center picking canceled.")]
    assert peak_callbacks.calls == []
    assert getattr(geometry_runtime, "_suppress_pan_press_once") is True


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


def test_canvas_click_routes_hkl_pick_before_caked_short_circuit() -> None:
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
        caked_view_enabled_factory=lambda: True,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=14.5, ydata=-7.5)

    assert canvas_interactions.handle_runtime_canvas_click(bindings, event) is True
    assert canvas_interactions.handle_runtime_canvas_press(bindings, event) is True
    assert peak_callbacks.calls == [("click", 14.5, -7.5)]
    assert drag_callbacks.calls == []


def test_canvas_click_routes_hkl_pick_through_detector_coord_normalization() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(hkl_pick_armed=True),
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=_DragCallbacks(),
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x) + 1.25, float(y) - 2.5),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: False,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=14.5, ydata=-7.5)

    assert canvas_interactions.handle_runtime_canvas_click(bindings, event) is True
    assert peak_callbacks.calls == [("click", 15.75, -10.0)]


def test_canvas_click_routes_hkl_pick_through_caked_coord_normalization() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()

    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=state.ManualGeometryState(),
        peak_selection_state=state.PeakSelectionState(hkl_pick_armed=True),
        peak_selection_callbacks=peak_callbacks,
        integration_range_drag_callbacks=_DragCallbacks(),
        manual_pick_session_active=lambda: False,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x) + 3.0, float(y) - 4.0),
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda *_args, **_kwargs: None,
        place_geometry_manual_selection_at=lambda *_args: None,
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: True,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=14.5, ydata=-7.5)

    assert canvas_interactions.handle_runtime_canvas_click(bindings, event) is True
    assert peak_callbacks.calls == [("click", 17.5, -11.5)]


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
            "zoom_active": True,
            "zoom_center": (12.0, 18.0),
            "saved_xlim": (1.0, 2.0),
            "saved_ylim": (3.0, 4.0),
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
        clear_geometry_manual_preview_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        restore_geometry_manual_pick_view=lambda **kwargs: calls.append(("restore", kwargs)),
        render_current_geometry_manual_pairs=lambda **kwargs: (
            calls.append(("render", kwargs)) or True
        ),
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
    assert getattr(geometry_runtime, "_manual_pick_skip_release_once", False) is True
    assert peak_state.suppress_drag_press_once is True
    assert canvas_interactions.handle_runtime_canvas_press(bindings, click_event) is True
    assert peak_state.suppress_drag_press_once is False
    assert manual_state.pick_session["zoom_active"] is False
    assert manual_state.pick_session["zoom_center"] is None
    assert manual_state.pick_session["saved_xlim"] is None
    assert manual_state.pick_session["saved_ylim"] is None
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert getattr(geometry_runtime, "_manual_pick_skip_release_once", False) is False

    assert calls == [
        ("toggle", 12.0, 18.0),
    ]
    assert drag_callbacks.calls == []


def test_canvas_detector_view_single_group_fallback_places_on_release() -> None:
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
            "place_current_release": True,
            "zoom_active": False,
            "zoom_center": None,
            "saved_xlim": None,
            "saved_ylim": None,
        }

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
        clear_geometry_manual_preview_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        restore_geometry_manual_pick_view=lambda **kwargs: calls.append(("restore", kwargs)),
        render_current_geometry_manual_pairs=lambda **kwargs: (
            calls.append(("render", kwargs)) or True
        ),
        caked_view_enabled_factory=lambda: False,
        set_geometry_status_text=lambda text: calls.append(("status", text)),
        draw_idle=lambda: calls.append(("draw",)),
    )

    click_event = _FakeEvent(
        button=1,
        inaxes=axis,
        xdata=120.0,
        ydata=130.0,
        x=110.0,
        y=70.0,
    )
    release_event = _FakeEvent(
        button=1,
        inaxes=axis,
        xdata=121.0,
        ydata=131.0,
    )

    assert canvas_interactions.handle_runtime_canvas_click(bindings, click_event) is True
    assert getattr(geometry_runtime, "_manual_pick_skip_release_once", False) is False
    assert canvas_interactions.handle_runtime_canvas_press(bindings, click_event) is True
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True

    assert calls == [
        ("toggle", 120.0, 130.0),
        ("place", 121.0, 131.0),
    ]
    assert drag_callbacks.calls == []


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


def test_canvas_detector_view_manual_pick_session_places_on_release_without_zoom_preview() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState()
    geometry_runtime = state.GeometryRuntimeState(manual_pick_armed=True)
    preview_state = state.GeometryPreviewState()
    manual_state = state.ManualGeometryState(
        pick_session={
            "group_key": ("q", 1),
            "zoom_active": True,
            "zoom_center": (15.0, 25.0),
            "saved_xlim": (10.0, 20.0),
            "saved_ylim": (30.0, 40.0),
        }
    )
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
        clear_geometry_manual_preview_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        restore_geometry_manual_pick_view=lambda **kwargs: calls.append(("restore", kwargs)),
        render_current_geometry_manual_pairs=lambda **kwargs: (
            calls.append(("render", kwargs)) or True
        ),
        caked_view_enabled_factory=lambda: False,
        set_geometry_status_text=lambda text: calls.append(("status", text)),
        draw_idle=lambda: calls.append(("draw",)),
    )

    click_event = _FakeEvent(
        button=1,
        inaxes=axis,
        xdata=15.0,
        ydata=25.0,
        x=110.0,
        y=70.0,
    )
    motion_event = _FakeEvent(button=1, inaxes=axis, xdata=16.0, ydata=26.0)
    release_event = _FakeEvent(button=1, inaxes=axis, xdata=17.0, ydata=27.0)
    cancel_release = _FakeEvent(button=1, inaxes=None, xdata=None, ydata=None)

    assert canvas_interactions.handle_runtime_canvas_click(bindings, click_event) is True
    assert calls == []

    assert canvas_interactions.handle_runtime_canvas_press(bindings, click_event) is True
    assert manual_state.pick_session["zoom_active"] is False
    assert manual_state.pick_session["zoom_center"] is None
    assert manual_state.pick_session["saved_xlim"] is None
    assert manual_state.pick_session["saved_ylim"] is None
    assert calls == []

    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert calls == []

    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert calls == [("place", 17.0, 27.0)]

    calls.clear()
    assert canvas_interactions.handle_runtime_canvas_release(bindings, cancel_release) is True
    assert calls == []
    assert drag_callbacks.calls == []


def test_canvas_drag_move_saved_manual_peak_places_refined_release() -> None:
    axis = _FakeAxis()
    peak_callbacks = _PeakCallbacks()
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState()
    geometry_runtime = state.GeometryRuntimeState(
        manual_pick_armed=False,
        manual_drag_move_enabled=True,
    )
    preview_state = state.GeometryPreviewState()
    manual_state = state.ManualGeometryState(pick_session={})
    calls = []

    def _toggle_selection(col, row, **kwargs):
        calls.append(("toggle", float(col), float(row), dict(kwargs)))
        if not bool(kwargs.get("saved_pair_move_only")):
            return False
        manual_state.pick_session = {
            "group_key": ("q", 1),
            "group_entries": [{"label": "saved"}],
            "pending_entries": [],
            "target_count": 1,
            "moving_saved_pair": True,
        }
        return True

    def _place(col, row):
        calls.append(("place", float(col), float(row)))
        manual_state.pick_session = {}

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
        apply_geometry_manual_pick_zoom=lambda *_args, **_kwargs: None,
        update_geometry_manual_pick_preview=lambda col, row, **kwargs: calls.append(
            ("preview", float(col), float(row), dict(kwargs))
        ),
        place_geometry_manual_selection_at=_place,
        clear_geometry_manual_preview_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **kwargs: (
            calls.append(("render", kwargs)) or True
        ),
        caked_view_enabled_factory=lambda: False,
        begin_live_interaction=lambda: calls.append(("begin",)),
        touch_live_interaction=lambda: calls.append(("touch",)),
        end_live_interaction=lambda: calls.append(("end",)),
    )

    press_event = _FakeEvent(button=1, inaxes=axis, xdata=12.0, ydata=18.0)
    motion_event = _FakeEvent(button=1, inaxes=axis, xdata=14.0, ydata=20.0)
    release_event = _FakeEvent(button=1, inaxes=axis, xdata=15.0, ydata=21.0)

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert getattr(geometry_runtime, "_manual_drag_move_active", False) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert getattr(geometry_runtime, "_manual_drag_move_active", False) is False
    assert calls == [
        ("toggle", 12.0, 18.0, {"saved_pair_move_only": True}),
        ("begin",),
        ("preview", 12.0, 18.0, {"force": True}),
        ("touch",),
        ("preview", 14.0, 20.0, {}),
        ("place", 15.0, 21.0),
        ("end",),
    ]
    assert drag_callbacks.calls == []


def test_canvas_click_places_manual_qr_pick_session_in_caked_space() -> None:
    axis = _FakeAxis()
    manual_state = state.ManualGeometryState(
        pick_session={"group_key": ("q", 1), "zoom_active": False}
    )
    calls = []
    bindings = canvas_interactions.CanvasInteractionBindings(
        axis=axis,
        geometry_runtime_state=state.GeometryRuntimeState(manual_pick_armed=True),
        geometry_preview_state=state.GeometryPreviewState(),
        geometry_manual_state=manual_state,
        peak_selection_state=state.PeakSelectionState(),
        peak_selection_callbacks=_PeakCallbacks(),
        integration_range_drag_callbacks=_DragCallbacks(),
        manual_pick_session_active=lambda: True,
        set_geometry_manual_pick_mode=lambda *_args, **_kwargs: None,
        set_geometry_preview_exclude_mode=lambda *_args, **_kwargs: None,
        toggle_geometry_manual_selection_at=lambda *_args: None,
        toggle_live_geometry_preview_exclusion_at=lambda *_args: None,
        clamp_to_axis_view=lambda axis_arg, x, y: (float(x), float(y)),
        apply_geometry_manual_pick_zoom=lambda col, row, **kwargs: (
            calls.append(("zoom", float(col), float(row), kwargs))
            or manual_state.pick_session.update({"zoom_active": True})
        ),
        update_geometry_manual_pick_preview=lambda col, row, **kwargs: calls.append(
            ("preview", float(col), float(row), kwargs)
        ),
        place_geometry_manual_selection_at=lambda col, row: calls.append(
            ("place", float(col), float(row))
        ),
        clear_geometry_manual_preview_artists=lambda **_kwargs: None,
        restore_geometry_manual_pick_view=lambda **_kwargs: None,
        render_current_geometry_manual_pairs=lambda **_kwargs: True,
        caked_view_enabled_factory=lambda: True,
    )

    two_theta_deg = 18.0
    phi_deg = -5.0
    press_event = _FakeEvent(
        button=1,
        inaxes=axis,
        xdata=two_theta_deg,
        ydata=phi_deg,
        x=110.0,
        y=70.0,
    )
    release_event = _FakeEvent(button=1, inaxes=axis, xdata=19.0, ydata=-4.0)

    assert canvas_interactions.handle_runtime_canvas_click(bindings, press_event) is True
    assert calls == []

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert calls[0][0] == "zoom"
    assert calls[0][1:3] == (two_theta_deg, phi_deg)
    assert calls[1] == ("preview", two_theta_deg, phi_deg, {"force": True})

    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert calls[2] == ("place", 19.0, -4.0)


def test_canvas_interaction_callback_bundle_delegates_live_bindings(monkeypatch) -> None:
    callback_calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_click",
        lambda bindings_arg, event: callback_calls.append(("click", bindings_arg, event)) or True,
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_press",
        lambda bindings_arg, event: callback_calls.append(("press", bindings_arg, event)) or False,
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_motion",
        lambda bindings_arg, event: callback_calls.append(("motion", bindings_arg, event)) or True,
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_release",
        lambda bindings_arg, event: (
            callback_calls.append(("release", bindings_arg, event)) or False
        ),
    )
    monkeypatch.setattr(
        canvas_interactions,
        "handle_runtime_canvas_scroll",
        lambda bindings_arg, event: callback_calls.append(("scroll", bindings_arg, event)) or True,
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = canvas_interactions.make_runtime_canvas_interaction_callbacks(build_bindings)
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

    callbacks = canvas_interactions.make_runtime_canvas_interaction_callbacks(lambda: bindings)

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
    interaction_events = []
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
        begin_live_interaction=lambda: interaction_events.append("begin"),
        touch_live_interaction=lambda: interaction_events.append("touch"),
        end_live_interaction=lambda: interaction_events.append("end"),
    )

    press_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=4.0,
        ydata=5.0,
        x=90.0,
        y=95.0,
    )
    motion_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )
    release_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert axis.get_xlim() == (-2.0, 8.0)
    assert axis.get_ylim() == (16.0, -4.0)
    assert draw_calls == [True]
    assert interaction_events == ["begin", "touch"]
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert interaction_events == ["begin", "touch", "end"]
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is False
    assert drag_callbacks.calls == []


def _assert_right_drag_ignores_stale_left_drag_suppression(
    *,
    caked_view_enabled: bool,
) -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState(suppress_drag_press_once=True)
    interaction_events = []
    draw_calls = []
    bindings = _basic_canvas_bindings(
        axis=axis,
        peak_selection_state=peak_state,
        integration_range_drag_callbacks=drag_callbacks,
        caked_view_enabled=caked_view_enabled,
        draw_idle=lambda: draw_calls.append(True),
        begin_live_interaction=lambda: interaction_events.append("begin"),
        touch_live_interaction=lambda: interaction_events.append("touch"),
        end_live_interaction=lambda: interaction_events.append("end"),
    )

    press_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=4.0,
        ydata=5.0,
        x=90.0,
        y=95.0,
    )
    motion_event = _FakeEvent(
        button=None,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )
    release_event = _FakeEvent(
        button=None,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session") is not None
    assert peak_state.suppress_drag_press_once is False
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert axis.get_xlim() == pytest.approx((-2.0, 8.0))
    assert axis.get_ylim() == pytest.approx((16.0, -4.0))
    assert draw_calls == [True]
    assert interaction_events == ["begin", "touch"]
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session") is None
    assert interaction_events == ["begin", "touch", "end"]
    assert drag_callbacks.calls == []


def test_canvas_right_drag_ignores_stale_left_drag_suppression_detector_view() -> None:
    _assert_right_drag_ignores_stale_left_drag_suppression(caked_view_enabled=False)


def test_canvas_right_drag_ignores_stale_left_drag_suppression_caked_view() -> None:
    _assert_right_drag_ignores_stale_left_drag_suppression(caked_view_enabled=True)


def _assert_pixel_only_right_drag_starts_pan(
    *,
    caked_view_enabled: bool,
    stale_left_drag_suppression: bool = False,
) -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState(
        suppress_drag_press_once=bool(stale_left_drag_suppression)
    )
    interaction_events = []
    draw_calls = []
    bindings = _basic_canvas_bindings(
        axis=axis,
        peak_selection_state=peak_state,
        integration_range_drag_callbacks=drag_callbacks,
        caked_view_enabled=caked_view_enabled,
        draw_idle=lambda: draw_calls.append(True),
        begin_live_interaction=lambda: interaction_events.append("begin"),
        touch_live_interaction=lambda: interaction_events.append("touch"),
        end_live_interaction=lambda: interaction_events.append("end"),
    )

    press_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=None,
        ydata=None,
        x=90.0,
        y=95.0,
    )
    motion_event = _FakeEvent(
        button=None,
        inaxes=axis,
        xdata=None,
        ydata=None,
        x=130.0,
        y=75.0,
    )
    release_event = _FakeEvent(
        button=None,
        inaxes=axis,
        xdata=None,
        ydata=None,
        x=130.0,
        y=75.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session") is not None
    assert peak_state.suppress_drag_press_once is False
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert axis.get_xlim() == pytest.approx((-2.0, 8.0))
    assert axis.get_ylim() == pytest.approx((16.0, -4.0))
    assert draw_calls == [True]
    assert interaction_events == ["begin", "touch"]
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session") is None
    assert interaction_events == ["begin", "touch", "end"]
    assert drag_callbacks.calls == []


def test_canvas_right_drag_starts_detector_pan_from_press_pixels_without_axis_data() -> None:
    _assert_pixel_only_right_drag_starts_pan(caked_view_enabled=False)


def test_canvas_right_drag_starts_caked_pan_from_press_pixels_without_axis_data() -> None:
    _assert_pixel_only_right_drag_starts_pan(caked_view_enabled=True)


def test_canvas_right_drag_starts_pan_from_press_pixels_inside_axis_without_inaxes() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    bindings = _basic_canvas_bindings(
        axis=axis,
        draw_idle=lambda: None,
    )

    press_event = _FakeEvent(
        button=3,
        inaxes=None,
        xdata=None,
        ydata=None,
        x=90.0,
        y=95.0,
    )
    motion_event = _FakeEvent(
        button=None,
        inaxes=None,
        xdata=None,
        ydata=None,
        x=130.0,
        y=75.0,
    )
    release_event = _FakeEvent(
        button=None,
        inaxes=None,
        xdata=None,
        ydata=None,
        x=130.0,
        y=75.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert axis.get_xlim() == pytest.approx((-2.0, 8.0))
    assert axis.get_ylim() == pytest.approx((16.0, -4.0))
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True


def test_canvas_right_drag_with_stale_suppression_starts_from_press_pixels() -> None:
    _assert_pixel_only_right_drag_starts_pan(
        caked_view_enabled=False,
        stale_left_drag_suppression=True,
    )


def test_canvas_right_drag_rejects_press_pixels_outside_axis_without_axis_data() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    drag_callbacks = _DragCallbacks()
    bindings = _basic_canvas_bindings(
        axis=axis,
        integration_range_drag_callbacks=drag_callbacks,
    )

    press_event = _FakeEvent(
        button=3,
        inaxes=None,
        xdata=None,
        ydata=None,
        x=0.0,
        y=0.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is False
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session", None) is None
    assert drag_callbacks.calls == []


def test_canvas_left_drag_suppression_still_blocks_left_press() -> None:
    axis = _FakeAxis()
    drag_callbacks = _DragCallbacks()
    peak_state = state.PeakSelectionState(suppress_drag_press_once=True)
    bindings = _basic_canvas_bindings(
        axis=axis,
        peak_selection_state=peak_state,
        integration_range_drag_callbacks=drag_callbacks,
    )

    event = _FakeEvent(button=1, inaxes=axis, xdata=1.0, ydata=2.0)

    assert canvas_interactions.handle_runtime_canvas_press(bindings, event) is True
    assert peak_state.suppress_drag_press_once is False
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session", None) is None
    assert drag_callbacks.calls == []


def test_canvas_right_click_cancel_does_not_start_pan_on_same_press() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    geometry_runtime = state.GeometryRuntimeState(manual_pick_armed=True)
    mode_calls = []
    bindings = _basic_canvas_bindings(
        axis=axis,
        geometry_runtime_state=geometry_runtime,
        set_geometry_manual_pick_mode=lambda enabled, message=None: mode_calls.append(
            (bool(enabled), message)
        ),
    )
    callbacks = canvas_interactions.make_runtime_canvas_interaction_callbacks(lambda: bindings)
    event = _FakeEvent(button=3, inaxes=axis, xdata=4.0, ydata=5.0, x=90.0, y=95.0)

    assert callbacks.on_click(event) is True
    assert callbacks.on_press(event) is True
    assert mode_calls == [(False, "Manual geometry picking canceled.")]
    assert getattr(geometry_runtime, "_suppress_pan_press_once", False) is False
    assert getattr(geometry_runtime, "_canvas_pan_session", None) is None


def test_canvas_right_drag_pans_caked_view_with_string_button_value() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(-180.0, 180.0))
    drag_callbacks = _DragCallbacks()
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
        caked_view_enabled_factory=lambda: True,
        draw_idle=lambda: None,
    )

    press_event = _FakeEvent(
        button="MouseButton.RIGHT",
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=60.0,
        y=60.0,
    )
    motion_event = _FakeEvent(
        button="MouseButton.RIGHT",
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=100.0,
        y=40.0,
    )
    release_event = _FakeEvent(
        button="MouseButton.RIGHT",
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=100.0,
        y=40.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert axis.get_xlim() == pytest.approx((-20.0, 80.0))
    assert axis.get_ylim() == pytest.approx((-108.0, 252.0))
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert drag_callbacks.calls == []


@pytest.mark.parametrize("button", _RIGHT_BUTTON_FORMS)
def test_canvas_right_button_forms_start_pan(button) -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    drag_callbacks = _DragCallbacks()
    bindings = _basic_canvas_bindings(
        axis=axis,
        integration_range_drag_callbacks=drag_callbacks,
    )

    press_event = _FakeEvent(
        button=button,
        inaxes=axis,
        xdata=4.0,
        ydata=5.0,
        x=90.0,
        y=95.0,
    )
    motion_event = _FakeEvent(
        button=None,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )
    release_event = _FakeEvent(button=None, inaxes=axis, xdata=6.0, ydata=9.0)

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session") is not None
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert axis.get_xlim() == pytest.approx((-2.0, 8.0))
    assert axis.get_ylim() == pytest.approx((16.0, -4.0))
    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert getattr(bindings.geometry_runtime_state, "_canvas_pan_session") is None
    assert drag_callbacks.calls == []


def test_canvas_right_drag_pan_uses_pointer_pixels_across_multiple_motion_events() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
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
        caked_view_enabled_factory=lambda: False,
        draw_idle=lambda: None,
    )

    press_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=4.0,
        ydata=5.0,
        x=90.0,
        y=95.0,
    )
    first_motion = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )
    # After the first pan update, the shifted axes would report the same data
    # coordinates here under the old implementation, which made drag motion
    # stall. Pixel anchoring should continue moving the view.
    second_motion = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=170.0,
        y=55.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, first_motion) is True
    assert axis.get_xlim() == pytest.approx((-2.0, 8.0))
    assert axis.get_ylim() == pytest.approx((16.0, -4.0))

    assert canvas_interactions.handle_runtime_canvas_motion(bindings, second_motion) is True
    assert axis.get_xlim() == pytest.approx((-4.0, 6.0))
    assert axis.get_ylim() == pytest.approx((12.0, -8.0))


def test_canvas_right_drag_pan_previews_limits_and_commits_once_on_release() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(20.0, 0.0))
    preview_limits = []
    commit_calls = []
    interaction_events = []

    def _commit_preview() -> bool:
        if not preview_limits:
            return False
        xlim, ylim = preview_limits[-1]
        commit_calls.append((xlim, ylim))
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        return True

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
        caked_view_enabled_factory=lambda: False,
        draw_idle=lambda: None,
        begin_live_interaction=lambda: interaction_events.append("begin"),
        touch_live_interaction=lambda: interaction_events.append("touch"),
        end_live_interaction=lambda: interaction_events.append("end"),
        preview_view_limits=lambda xlim, ylim: preview_limits.append((xlim, ylim)) or True,
        commit_preview_view=_commit_preview,
    )

    press_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=4.0,
        ydata=5.0,
        x=90.0,
        y=95.0,
    )
    motion_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )
    release_event = _FakeEvent(
        button=3,
        inaxes=axis,
        xdata=6.0,
        ydata=9.0,
        x=130.0,
        y=75.0,
    )

    assert canvas_interactions.handle_runtime_canvas_press(bindings, press_event) is True
    assert canvas_interactions.handle_runtime_canvas_motion(bindings, motion_event) is True
    assert preview_limits == [((-2.0, 8.0), (16.0, -4.0))]
    assert axis.set_xlim_calls == []
    assert axis.set_ylim_calls == []
    assert interaction_events == ["begin", "touch"]

    assert canvas_interactions.handle_runtime_canvas_release(bindings, release_event) is True
    assert commit_calls == [((-2.0, 8.0), (16.0, -4.0))]
    assert axis.set_xlim_calls == [(-2.0, 8.0)]
    assert axis.set_ylim_calls == [(16.0, -4.0)]
    assert interaction_events == ["begin", "touch", "end"]


def test_canvas_scroll_zooms_caked_view_about_cursor() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(-180.0, 180.0))
    interaction_events = []
    anchor_x_px = 60.0
    anchor_y_px = 61.66666666666667
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
        touch_live_interaction=lambda: interaction_events.append("touch"),
        end_live_interaction=lambda: interaction_events.append("end"),
    )

    zoom_in = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=anchor_x_px,
        y=anchor_y_px,
    )
    zoom_out = _FakeEvent(
        button="down",
        step=-1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=anchor_x_px,
        y=anchor_y_px,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    xlim_after_zoom_in = axis.get_xlim()
    ylim_after_zoom_in = axis.get_ylim()
    assert xlim_after_zoom_in == pytest.approx((4.166666666666668, 87.5))
    assert ylim_after_zoom_in == pytest.approx((-155.0, 145.0))
    assert interaction_events == ["touch", "end"]

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_out) is True
    assert axis.get_xlim() == pytest.approx((0.0, 100.0))
    assert axis.get_ylim() == pytest.approx((-180.0, 180.0))
    assert interaction_events == ["touch", "end", "touch", "end"]


def test_canvas_scroll_uses_pending_preview_limits_for_zoom_bursts() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(-180.0, 180.0))
    interaction_events = []
    preview_limits = []
    anchor_x_px = 60.0
    anchor_y_px = 61.66666666666667

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
        touch_live_interaction=lambda: interaction_events.append("touch"),
        end_live_interaction=lambda: interaction_events.append("end"),
        preview_view_limits=lambda xlim, ylim: preview_limits.append((xlim, ylim)) or True,
    )

    zoom_in = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=anchor_x_px,
        y=anchor_y_px,
    )
    zoom_out = _FakeEvent(
        button="down",
        step=-1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=anchor_x_px,
        y=anchor_y_px,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    assert axis.set_xlim_calls == []
    assert axis.set_ylim_calls == []
    assert axis.get_xlim() == pytest.approx((0.0, 100.0))
    assert axis.get_ylim() == pytest.approx((-180.0, 180.0))
    assert preview_limits[0][0] == pytest.approx((4.166666666666668, 87.5))
    assert preview_limits[0][1] == pytest.approx((-155.0, 145.0))

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_out) is True
    assert preview_limits[1][0] == pytest.approx((0.0, 100.0))
    assert preview_limits[1][1] == pytest.approx((-180.0, 180.0))
    assert axis.set_xlim_calls == []
    assert axis.set_ylim_calls == []
    assert interaction_events == ["touch", "end", "touch", "end"]


def test_canvas_scroll_zooms_detector_view_about_cursor_on_inverted_y_axis() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(100.0, 0.0))
    interaction_events = []

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
        caked_view_enabled_factory=lambda: False,
        draw_idle=lambda: None,
        touch_live_interaction=lambda: interaction_events.append("touch"),
        end_live_interaction=lambda: interaction_events.append("end"),
    )

    zoom_in = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=75.0,
        x=60.0,
        y=45.0,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    assert axis.get_xlim() == pytest.approx((4.166666666666664, 87.5))
    assert axis.get_ylim() == pytest.approx((95.83333333333334, 12.5))
    assert interaction_events == ["touch", "end"]


def test_canvas_scroll_detector_view_after_pan_uses_cursor_anchor() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(8.0, -2.0))

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
        caked_view_enabled_factory=lambda: False,
        detector_view_limits=((0.0, 10.0), (10.0, 0.0)),
        draw_idle=lambda: None,
    )

    zoom_in = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=5.0,
        ydata=3.0,
        x=100.0,
        y=60.0,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    assert axis.get_xlim() == pytest.approx((0.833333333333333, 9.166666666666668))
    assert axis.get_ylim() == pytest.approx((7.166666666666667, -1.166666666666667))


def test_canvas_scroll_detector_view_preview_after_pan_uses_cursor_pixels() -> None:
    axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(8.0, -2.0))
    preview_limits = []

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
        caked_view_enabled_factory=lambda: False,
        detector_view_limits=((0.0, 10.0), (10.0, 0.0)),
        draw_idle=lambda: None,
        preview_view_limits=lambda xlim, ylim: preview_limits.append((xlim, ylim)) or True,
    )

    first_zoom = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=5.0,
        ydata=3.0,
        x=100.0,
        y=60.0,
    )
    second_zoom = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        # Preview path leaves legacy event.xdata/ydata stale until commit.
        xdata=5.0,
        ydata=3.0,
        x=150.0,
        y=40.0,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, first_zoom) is True
    assert preview_limits[0][0] == pytest.approx((0.833333333333333, 9.166666666666668))
    assert preview_limits[0][1] == pytest.approx((7.166666666666667, -1.166666666666667))

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, second_zoom) is True
    assert preview_limits[1][0] == pytest.approx((1.8055555555555554, 8.750000000000002))
    assert preview_limits[1][1] == pytest.approx((6.888888888888889, -0.055555555555556246))
    assert axis.set_xlim_calls == []
    assert axis.set_ylim_calls == []


def test_canvas_scroll_detector_view_preview_bursts_follow_cursor_pixels() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(100.0, 0.0))
    preview_limits = []

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
        caked_view_enabled_factory=lambda: False,
        draw_idle=lambda: None,
        preview_view_limits=lambda xlim, ylim: preview_limits.append((xlim, ylim)) or True,
    )

    first_zoom = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=75.0,
        x=60.0,
        y=45.0,
    )
    second_zoom = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        # Preview path has not committed axis limits yet, so legacy xdata/ydata stay stale.
        xdata=25.0,
        ydata=75.0,
        x=160.0,
        y=95.0,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, first_zoom) is True
    assert preview_limits[0][0] == pytest.approx((4.166666666666664, 87.5))
    assert preview_limits[0][1] == pytest.approx((95.83333333333334, 12.5))

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, second_zoom) is True
    assert preview_limits[1][0] == pytest.approx((14.583333333333329, 84.02777777777777))
    assert preview_limits[1][1] == pytest.approx((85.41666666666669, 15.972222222222221))
    assert axis.set_xlim_calls == []
    assert axis.set_ylim_calls == []


def test_canvas_scroll_inverted_y_caked_view_still_zooms_about_cursor() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(100.0, 0.0))

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
        xdata=40.0,
        ydata=80.0,
        x=160.0,
        y=45.0,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    assert axis.get_xlim() == pytest.approx((6.666666666666664, 90.0))
    assert axis.get_ylim() == pytest.approx((96.66666666666667, 13.333333333333329))


def test_canvas_scroll_prefers_axis_data_when_preview_is_not_pending() -> None:
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
        xdata=40.0,
        ydata=-20.0,
        x=160.0,
        y=61.66666666666667,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    assert axis.get_xlim() == pytest.approx((6.666666666666664, 90.0))
    assert axis.get_ylim() == pytest.approx((-153.33333333333334, 146.66666666666669))


def test_canvas_scroll_uses_cursor_pixels_when_preview_keeps_axis_data_stale() -> None:
    axis = _FakeAxis(xlim=(0.0, 100.0), ylim=(-180.0, 180.0))
    preview_limits = []

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
        preview_view_limits=lambda xlim, ylim: preview_limits.append((xlim, ylim)) or True,
    )

    first_zoom = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        xdata=25.0,
        ydata=-30.0,
        x=60.0,
        y=61.66666666666667,
    )
    second_zoom = _FakeEvent(
        button="up",
        step=1.0,
        inaxes=axis,
        # Preview path has not committed axis limits yet, so legacy xdata/ydata stay stale.
        xdata=25.0,
        ydata=-30.0,
        x=160.0,
        y=61.66666666666667,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, first_zoom) is True
    assert preview_limits[0][0] == pytest.approx((4.166666666666668, 87.5))
    assert preview_limits[0][1] == pytest.approx((-155.0, 145.0))

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, second_zoom) is True
    assert preview_limits[1][0] == pytest.approx((14.583333333333329, 84.02777777777779))
    assert preview_limits[1][1] == pytest.approx((-134.16666666666669, 115.83333333333334))
    assert axis.set_xlim_calls == []
    assert axis.set_ylim_calls == []


def test_canvas_scroll_outside_axis_falls_back_to_center_anchor() -> None:
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
        inaxes=None,
        xdata=None,
        ydata=None,
        x=5.0,
        y=5.0,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    assert axis.get_xlim() == pytest.approx((8.333333333333329, 91.66666666666667))
    assert axis.get_ylim() == pytest.approx((-150.0, 150.0))


def test_canvas_scroll_in_axis_with_missing_coords_uses_cursor_pixels() -> None:
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
        xdata=None,
        ydata=None,
        x=160.0,
        y=61.66666666666667,
    )

    assert canvas_interactions.handle_runtime_canvas_scroll(bindings, zoom_in) is True
    assert axis.get_xlim() == pytest.approx((12.5, 95.83333333333334))
    assert axis.get_ylim() == pytest.approx((-155.0, 145.0))


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
