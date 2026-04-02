from types import SimpleNamespace

from ra_sim.gui import runtime_display_acceleration


class _FakeVar:
    def __init__(self, value=None) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _FakeWidget:
    def __init__(self, *_args, **_kwargs) -> None:
        self.manager = ""
        self.configure_calls = []
        self.bind_calls = []
        self._bindings = {}
        self.width = 640
        self.height = 480
        self.widget_id = 1234

    def pack(self, **_kwargs) -> None:
        self.manager = "pack"

    def pack_forget(self) -> None:
        self.manager = ""

    def winfo_manager(self) -> str:
        return self.manager

    def configure(self, **kwargs) -> None:
        self.configure_calls.append(dict(kwargs))

    config = configure

    def bind(self, event_name, callback, add=None):
        self.bind_calls.append((event_name, bool(add)))
        self._bindings.setdefault(str(event_name), []).append(callback)
        return f"{event_name}-{len(self._bindings[str(event_name)])}"

    def emit(self, event_name) -> None:
        for callback in list(self._bindings.get(str(event_name), [])):
            callback(None)

    def update_idletasks(self) -> None:
        return None

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height

    def winfo_id(self) -> int:
        return self.widget_id


class _FakeCheckbutton(_FakeWidget):
    def __init__(self) -> None:
        super().__init__()
        self.disabled = False

    def state(self, states) -> None:
        for item in list(states):
            if item == "disabled":
                self.disabled = True
            elif item == "!disabled":
                self.disabled = False


class _FakeTkModule:
    TOP = "top"
    BOTH = "both"
    CENTER = "center"

    StringVar = _FakeVar


class _FakeTtkModule:
    Label = _FakeWidget


class _FakeCanvasWidget(_FakeWidget):
    pass


class _FakeMatplotlibCanvas:
    def __init__(self) -> None:
        self.widget = _FakeCanvasWidget()
        self.widget.pack()
        self.draw_calls = 0
        self.draw_idle_calls = 0

    def get_tk_widget(self):
        return self.widget

    def draw(self) -> None:
        self.draw_calls += 1

    def draw_idle(self) -> None:
        self.draw_idle_calls += 1


class _UnavailableFastPlotViewerModule:
    class FastPlotViewer:
        def __init__(self, *args, **kwargs) -> None:
            self.available = False
            self.error_message = "pyqtgraph missing"


class _AvailableFastPlotViewerModule:
    def __init__(self) -> None:
        self.created_viewers = []

    class FastPlotViewer:
        def __init__(self, *_args, module=None, **_kwargs) -> None:
            self.available = True
            self.error_message = None
            self.mount_calls = []
            self.resize_calls = []
            self.show_window_calls = 0
            self.closed = False
            if module is not None:
                module.created_viewers.append(self)

        def mount_into_tk(self, host) -> bool:
            self.mount_calls.append(host)
            return True

        def resize_to_tk_host(self, host) -> None:
            self.resize_calls.append(host)

        def show_window(self) -> None:
            self.show_window_calls += 1

        def close(self) -> None:
            self.closed = True

        def process_events(self) -> None:
            return None

    class MatplotlibCanvasProxy:
        def __init__(self, canvas, fast_viewer, **_kwargs) -> None:
            self.canvas = canvas
            self.fast_viewer = fast_viewer
            self.sync_callback = None
            self.sync_calls = []

        def set_sync_callback(self, callback) -> None:
            self.sync_callback = callback

        def process_fast_events(self) -> None:
            return None

        def sync_from_matplotlib(self, **kwargs) -> None:
            self.sync_calls.append(dict(kwargs))
            return None


def _build_workflow(
    *,
    fast_plot_viewer_module=None,
    requested_enabled: bool = False,
    manual_pick_armed_factory=None,
    control_locked: bool = False,
) -> tuple[object, SimpleNamespace, SimpleNamespace]:
    display_view_state = SimpleNamespace(
        fast_viewer_var=_FakeVar(bool(requested_enabled)),
        fast_viewer_checkbutton=_FakeCheckbutton(),
        fast_viewer_status_var=_FakeVar(""),
    )
    canvas_frame = _FakeWidget()
    matplotlib_canvas = _FakeMatplotlibCanvas()
    set_canvas_calls = []
    if isinstance(fast_plot_viewer_module, _AvailableFastPlotViewerModule):
        viewer_module = SimpleNamespace(
            FastPlotViewer=lambda *args, **kwargs: (
                _AvailableFastPlotViewerModule.FastPlotViewer(
                    *args,
                    module=fast_plot_viewer_module,
                    **kwargs,
                )
            ),
            MatplotlibCanvasProxy=_AvailableFastPlotViewerModule.MatplotlibCanvasProxy,
        )
    else:
        viewer_module = (
            _UnavailableFastPlotViewerModule()
            if fast_plot_viewer_module is None
            else fast_plot_viewer_module
        )
    workflow = runtime_display_acceleration.build_runtime_fast_viewer_workflow(
        fast_plot_viewer_module=viewer_module,
        tk_module=_FakeTkModule,
        ttk_module=_FakeTtkModule,
        canvas_frame=canvas_frame,
        matplotlib_canvas=matplotlib_canvas,
        ax=object(),
        image_artist=object(),
        background_artist=object(),
        overlay_artist=object(),
        display_controls_view_state_factory=lambda: display_view_state,
        fast_toggle_var_factory=lambda: display_view_state.fast_viewer_var,
        canvas_interaction_callbacks_factory=lambda: None,
        bind_canvas_interactions=lambda **_kwargs: None,
        set_canvas=lambda _canvas: set_canvas_calls.append(_canvas),
        manual_pick_armed_factory=(
            (lambda: False)
            if manual_pick_armed_factory is None
            else manual_pick_armed_factory
        ),
        requested_enabled=bool(requested_enabled),
        control_locked=bool(control_locked),
    )
    return workflow, display_view_state, SimpleNamespace(
        canvas_frame=canvas_frame,
        matplotlib_canvas=matplotlib_canvas,
        set_canvas_calls=set_canvas_calls,
        fast_plot_viewer_module=fast_plot_viewer_module,
    )


def test_fast_viewer_control_disables_when_launch_is_blocked() -> None:
    workflow, display_view_state, _refs = _build_workflow(
        manual_pick_armed_factory=lambda: True,
    )

    workflow.refresh_runtime_mode(announce=False)

    assert display_view_state.fast_viewer_checkbutton.disabled is True
    assert (
        display_view_state.fast_viewer_status_var.get()
        == "Fast viewer unavailable while manual geometry picking is active."
    )


def test_fast_viewer_control_reenables_after_blocker_clears() -> None:
    blocked = {"value": True}
    workflow, display_view_state, _refs = _build_workflow(
        manual_pick_armed_factory=lambda: bool(blocked["value"]),
    )

    workflow.refresh_runtime_mode(announce=False)
    blocked["value"] = False
    workflow.refresh_runtime_mode(announce=False)

    assert display_view_state.fast_viewer_checkbutton.disabled is False
    assert (
        display_view_state.fast_viewer_status_var.get()
        == "Replace the plot area with a faster viewer."
    )


def test_fast_viewer_open_failure_disables_control_with_note() -> None:
    workflow, display_view_state, _refs = _build_workflow(requested_enabled=True)

    assert workflow.refresh_runtime_mode(announce=False) is False
    assert workflow.requested_enabled() is False
    assert display_view_state.fast_viewer_checkbutton.disabled is True
    assert (
        display_view_state.fast_viewer_status_var.get()
        == "Fast viewer unavailable: pyqtgraph missing."
    )


def test_fast_viewer_replaces_plot_area_when_embedding_succeeds() -> None:
    viewer_module = _AvailableFastPlotViewerModule()
    workflow, display_view_state, refs = _build_workflow(
        fast_plot_viewer_module=viewer_module,
        requested_enabled=True,
    )

    assert workflow.refresh_runtime_mode(announce=False) is True

    viewer = viewer_module.created_viewers[0]
    assert viewer.mount_calls == [refs.canvas_frame]
    assert refs.matplotlib_canvas.widget.winfo_manager() == ""
    assert display_view_state.fast_viewer_status_var.get() == (
        "Fast viewer active in plot area. Matplotlib canvas paused."
    )

    refs.canvas_frame.emit("<Configure>")

    assert viewer.resize_calls


def test_fast_viewer_control_can_be_locked_while_viewer_is_active() -> None:
    viewer_module = _AvailableFastPlotViewerModule()
    workflow, display_view_state, _refs = _build_workflow(
        fast_plot_viewer_module=viewer_module,
        requested_enabled=True,
        control_locked=True,
    )

    assert workflow.refresh_runtime_mode(announce=False) is True
    assert display_view_state.fast_viewer_checkbutton.disabled is True
    assert display_view_state.fast_viewer_status_var.get() == (
        "Fast viewer active in plot area. Matplotlib canvas paused."
    )


def test_fast_viewer_disable_restores_embedded_matplotlib_canvas() -> None:
    viewer_module = _AvailableFastPlotViewerModule()
    workflow, display_view_state, refs = _build_workflow(
        fast_plot_viewer_module=viewer_module,
        requested_enabled=True,
    )
    assert workflow.refresh_runtime_mode(announce=False) is True

    workflow.set_requested_enabled(False)
    assert workflow.refresh_runtime_mode(announce=False) is False

    viewer = viewer_module.created_viewers[0]
    assert viewer.closed is True
    assert refs.matplotlib_canvas.widget.winfo_manager() == "pack"
    assert refs.matplotlib_canvas.draw_calls == 1
    assert display_view_state.fast_viewer_status_var.get() == (
        "Replace the plot area with a faster viewer."
    )


def test_reset_view_forces_fast_viewer_range_sync_when_active() -> None:
    viewer_module = _AvailableFastPlotViewerModule()
    workflow, _display_view_state, refs = _build_workflow(
        fast_plot_viewer_module=viewer_module,
        requested_enabled=True,
    )
    assert workflow.refresh_runtime_mode(announce=False) is True

    proxy = refs.set_canvas_calls[-1]
    workflow.reset_view()

    assert proxy.sync_calls
    assert proxy.sync_calls[-1]["force_view_range"] is True


def test_reset_view_draws_matplotlib_when_fast_viewer_is_inactive() -> None:
    workflow, _display_view_state, refs = _build_workflow(requested_enabled=False)

    workflow.reset_view()

    assert refs.matplotlib_canvas.draw_idle_calls == 1
