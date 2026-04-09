from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import analysis_figure_controls


class _FakeAxis:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def relim(self) -> None:
        self.calls.append("relim")

    def autoscale_view(self) -> None:
        self.calls.append("autoscale_view")


class _FakeFrame:
    created: list["_FakeFrame"] = []

    def __init__(self, parent) -> None:
        self.parent = parent
        self.pack_calls: list[dict[str, object]] = []
        self.__class__.created.append(self)

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(kwargs)


class _FakeButton:
    created: list["_FakeButton"] = []

    def __init__(self, parent, **kwargs) -> None:
        self.parent = parent
        self.kwargs = kwargs
        self.command = kwargs.get("command")
        self.pack_calls: list[dict[str, object]] = []
        self.__class__.created.append(self)

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(kwargs)


class _FakeToolbar:
    created: list["_FakeToolbar"] = []

    def __init__(self, canvas, parent, *, pack_toolbar=False) -> None:
        self.canvas = canvas
        self.parent = parent
        self.pack_toolbar = pack_toolbar
        self.updated = False
        self.pack_calls: list[dict[str, object]] = []
        self.__class__.created.append(self)

    def update(self) -> None:
        self.updated = True

    def pack(self, **kwargs) -> None:
        self.pack_calls.append(kwargs)


def test_reset_analysis_axes_view_autoscales_all_supported_axes() -> None:
    radial_axis = _FakeAxis()
    azimuth_axis = _FakeAxis()

    updated = analysis_figure_controls.reset_analysis_axes_view(
        radial_axis,
        object(),
        azimuth_axis,
    )

    assert updated is True
    assert radial_axis.calls == ["relim", "autoscale_view"]
    assert azimuth_axis.calls == ["relim", "autoscale_view"]


def test_create_analysis_figure_toolbar_builds_navigation_and_reset_controls() -> None:
    _FakeFrame.created = []
    _FakeButton.created = []
    _FakeToolbar.created = []

    reset_calls: list[str] = []
    ttk_module = SimpleNamespace(Frame=_FakeFrame, Button=_FakeButton)
    backend_tkagg_module = SimpleNamespace(NavigationToolbar2Tk=_FakeToolbar)
    canvas = object()
    parent = object()

    frame, toolbar, reset_button = (
        analysis_figure_controls.create_analysis_figure_toolbar(
            parent=parent,
            canvas=canvas,
            ttk_module=ttk_module,
            backend_tkagg_module=backend_tkagg_module,
            on_reset_view=lambda: reset_calls.append("reset"),
        )
    )

    assert frame is _FakeFrame.created[0]
    assert frame.parent is parent
    assert frame.pack_calls == [{"side": "top", "fill": "x"}]

    assert toolbar is _FakeToolbar.created[0]
    assert toolbar.canvas is canvas
    assert toolbar.parent is frame
    assert toolbar.pack_toolbar is False
    assert toolbar.updated is True
    assert toolbar.pack_calls == [{"side": "left", "fill": "x", "expand": True}]

    assert reset_button is _FakeButton.created[0]
    assert reset_button.parent is frame
    assert reset_button.kwargs["text"] == "Reset View"
    assert reset_button.pack_calls == [
        {"side": "right", "padx": (6, 0), "pady": 2}
    ]

    reset_button.command()
    assert reset_calls == ["reset"]
