from __future__ import annotations

from types import SimpleNamespace

from ra_sim.gui import analysis_figure_controls


class _FakeAxis:
    def __init__(
        self,
        *,
        xlim: tuple[float, float] = (0.0, 10.0),
        ylim: tuple[float, float] = (0.0, 20.0),
    ) -> None:
        self.calls: list[str] = []
        self.xlim = tuple(float(value) for value in xlim)
        self.ylim = tuple(float(value) for value in ylim)

    def relim(self) -> None:
        self.calls.append("relim")

    def autoscale_view(self) -> None:
        self.calls.append("autoscale_view")

    def get_xlim(self) -> tuple[float, float]:
        return self.xlim

    def get_ylim(self) -> tuple[float, float]:
        return self.ylim

    def set_xlim(self, x0: float, x1: float) -> None:
        self.xlim = (float(x0), float(x1))

    def set_ylim(self, y0: float, y1: float) -> None:
        self.ylim = (float(y0), float(y1))


class _FakeCanvas:
    def __init__(self) -> None:
        self.connections: dict[str, object] = {}
        self.draw_idle_calls = 0
        self._next_cid = 1

    def mpl_connect(self, event_name: str, callback) -> int:
        cid = self._next_cid
        self._next_cid += 1
        self.connections[event_name] = callback
        return cid

    def draw_idle(self) -> None:
        self.draw_idle_calls += 1


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


def test_create_analysis_figure_interactions_pans_only_the_active_axis() -> None:
    radial_axis = _FakeAxis(xlim=(0.0, 10.0), ylim=(10.0, 30.0))
    azimuth_axis = _FakeAxis(xlim=(100.0, 140.0), ylim=(-5.0, 5.0))
    canvas = _FakeCanvas()

    connections = analysis_figure_controls.create_analysis_figure_interactions(
        canvas=canvas,
        axes=(radial_axis, azimuth_axis),
    )

    assert set(connections) == {
        "button_press_event",
        "motion_notify_event",
        "button_release_event",
        "scroll_event",
    }

    canvas.connections["button_press_event"](
        SimpleNamespace(
            button=1,
            dblclick=False,
            inaxes=radial_axis,
            xdata=6.0,
            ydata=18.0,
        )
    )
    canvas.connections["motion_notify_event"](
        SimpleNamespace(
            inaxes=radial_axis,
            xdata=8.5,
            ydata=23.5,
        )
    )
    canvas.connections["button_release_event"](
        SimpleNamespace(
            button=1,
            inaxes=radial_axis,
            xdata=8.5,
            ydata=23.5,
        )
    )

    assert radial_axis.xlim == (-2.5, 7.5)
    assert radial_axis.ylim == (4.5, 24.5)
    assert azimuth_axis.xlim == (100.0, 140.0)
    assert azimuth_axis.ylim == (-5.0, 5.0)
    assert canvas.draw_idle_calls == 1


def test_create_analysis_figure_interactions_scroll_zoom_and_double_click_reset() -> None:
    radial_axis = _FakeAxis(xlim=(0.0, 12.0), ylim=(0.0, 24.0))
    azimuth_axis = _FakeAxis(xlim=(-50.0, 50.0), ylim=(10.0, 110.0))
    canvas = _FakeCanvas()
    reset_calls: list[str] = []

    analysis_figure_controls.create_analysis_figure_interactions(
        canvas=canvas,
        axes=(radial_axis, azimuth_axis),
        on_reset_view=lambda: reset_calls.append("reset"),
    )

    canvas.connections["scroll_event"](
        SimpleNamespace(
            inaxes=azimuth_axis,
            step=1.0,
            button="up",
            xdata=10.0,
            ydata=70.0,
        )
    )

    assert abs(azimuth_axis.xlim[1] - azimuth_axis.xlim[0]) < 100.0
    assert abs(azimuth_axis.ylim[1] - azimuth_axis.ylim[0]) < 100.0
    assert radial_axis.xlim == (0.0, 12.0)
    assert radial_axis.ylim == (0.0, 24.0)

    canvas.connections["button_press_event"](
        SimpleNamespace(
            button=1,
            dblclick=True,
            inaxes=radial_axis,
            xdata=3.0,
            ydata=5.0,
        )
    )

    assert reset_calls == ["reset"]
