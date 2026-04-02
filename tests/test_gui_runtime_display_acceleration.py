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

    def pack(self, **_kwargs) -> None:
        self.manager = "pack"

    def pack_forget(self) -> None:
        self.manager = ""

    def winfo_manager(self) -> str:
        return self.manager

    def configure(self, **kwargs) -> None:
        self.configure_calls.append(dict(kwargs))

    config = configure


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


def _build_workflow(
    *,
    fast_plot_viewer_module=None,
    requested_enabled: bool = False,
    manual_pick_armed_factory=None,
) -> tuple[object, SimpleNamespace]:
    display_view_state = SimpleNamespace(
        fast_viewer_var=_FakeVar(bool(requested_enabled)),
        fast_viewer_checkbutton=_FakeCheckbutton(),
        fast_viewer_status_var=_FakeVar(""),
    )
    workflow = runtime_display_acceleration.build_runtime_fast_viewer_workflow(
        fast_plot_viewer_module=(
            _UnavailableFastPlotViewerModule()
            if fast_plot_viewer_module is None
            else fast_plot_viewer_module
        ),
        tk_module=_FakeTkModule,
        ttk_module=_FakeTtkModule,
        canvas_frame=object(),
        matplotlib_canvas=_FakeMatplotlibCanvas(),
        ax=object(),
        image_artist=object(),
        background_artist=object(),
        overlay_artist=object(),
        display_controls_view_state_factory=lambda: display_view_state,
        fast_toggle_var_factory=lambda: display_view_state.fast_viewer_var,
        canvas_interaction_callbacks_factory=lambda: None,
        bind_canvas_interactions=lambda **_kwargs: None,
        set_canvas=lambda _canvas: None,
        manual_pick_armed_factory=(
            (lambda: False)
            if manual_pick_armed_factory is None
            else manual_pick_armed_factory
        ),
        requested_enabled=bool(requested_enabled),
    )
    return workflow, display_view_state


def test_fast_viewer_control_disables_when_launch_is_blocked() -> None:
    workflow, display_view_state = _build_workflow(
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
    workflow, display_view_state = _build_workflow(
        manual_pick_armed_factory=lambda: bool(blocked["value"]),
    )

    workflow.refresh_runtime_mode(announce=False)
    blocked["value"] = False
    workflow.refresh_runtime_mode(announce=False)

    assert display_view_state.fast_viewer_checkbutton.disabled is False
    assert (
        display_view_state.fast_viewer_status_var.get()
        == "Open in a separate window for faster image interaction."
    )


def test_fast_viewer_open_failure_disables_control_with_note() -> None:
    workflow, display_view_state = _build_workflow(requested_enabled=True)

    assert workflow.refresh_runtime_mode(announce=False) is False
    assert workflow.requested_enabled() is False
    assert display_view_state.fast_viewer_checkbutton.disabled is True
    assert (
        display_view_state.fast_viewer_status_var.get()
        == "Fast viewer unavailable: pyqtgraph missing."
    )
