from types import SimpleNamespace

from ra_sim.gui import main_matplotlib_interaction


class _FakeWidget:
    def __init__(self) -> None:
        self._after_callbacks = {}
        self._next_after_token = 0

    def after(self, _delay, callback):
        self._next_after_token += 1
        token = f"after-{self._next_after_token}"
        self._after_callbacks[token] = callback
        return token

    def after_cancel(self, token) -> None:
        self._after_callbacks.pop(token, None)

    def run_after_callbacks(self) -> None:
        callbacks = list(self._after_callbacks.items())
        self._after_callbacks = {}
        for _token, callback in callbacks:
            callback()


def test_request_main_matplotlib_redraw_coalesces_pending_draws() -> None:
    widget = _FakeWidget()
    runtime_state = SimpleNamespace(
        main_matplotlib_redraw_token=None,
        main_matplotlib_last_draw_ts=None,
    )
    clock = {"value": 100.0}
    draw_calls = []

    def _draw() -> bool:
        draw_calls.append(clock["value"])
        return True

    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.033,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=False,
        )
        is True
    )
    assert draw_calls == [100.0]
    assert runtime_state.main_matplotlib_redraw_token is None

    clock["value"] = 100.01
    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.033,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=False,
        )
        is True
    )
    first_token = runtime_state.main_matplotlib_redraw_token
    assert first_token is not None
    assert draw_calls == [100.0]

    clock["value"] = 100.02
    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.033,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=False,
        )
        is False
    )
    assert runtime_state.main_matplotlib_redraw_token == first_token
    assert draw_calls == [100.0]

    clock["value"] = 100.04
    widget.run_after_callbacks()
    assert runtime_state.main_matplotlib_redraw_token is None
    assert draw_calls == [100.0, 100.04]


def test_force_main_matplotlib_redraw_cancels_pending_token() -> None:
    widget = _FakeWidget()
    runtime_state = SimpleNamespace(
        main_matplotlib_redraw_token=None,
        main_matplotlib_last_draw_ts=50.0,
    )
    clock = {"value": 50.01}
    draw_calls = []

    def _draw() -> bool:
        draw_calls.append(clock["value"])
        return True

    main_matplotlib_interaction.request_main_matplotlib_redraw(
        widget=widget,
        runtime_state=runtime_state,
        interval_s=0.05,
        perf_counter_fn=lambda: clock["value"],
        draw_now=_draw,
        force=False,
    )
    assert runtime_state.main_matplotlib_redraw_token is not None
    assert draw_calls == []

    clock["value"] = 50.02
    assert (
        main_matplotlib_interaction.request_main_matplotlib_redraw(
            widget=widget,
            runtime_state=runtime_state,
            interval_s=0.05,
            perf_counter_fn=lambda: clock["value"],
            draw_now=_draw,
            force=True,
        )
        is True
    )
    assert runtime_state.main_matplotlib_redraw_token is None
    assert draw_calls == [50.02]


def test_suspend_and_restore_main_matplotlib_overlays_are_one_shot() -> None:
    runtime_state = SimpleNamespace(main_matplotlib_overlays_suspended=False)
    events = []

    assert (
        main_matplotlib_interaction.suspend_main_matplotlib_overlays(
            runtime_state,
            suspend_callback=lambda: events.append("suspend"),
        )
        is True
    )
    assert runtime_state.main_matplotlib_overlays_suspended is True
    assert events == ["suspend"]

    assert (
        main_matplotlib_interaction.suspend_main_matplotlib_overlays(
            runtime_state,
            suspend_callback=lambda: events.append("suspend-again"),
        )
        is False
    )
    assert events == ["suspend"]

    assert (
        main_matplotlib_interaction.restore_main_matplotlib_overlays(
            runtime_state,
            restore_callback=lambda: events.append("restore"),
        )
        is True
    )
    assert runtime_state.main_matplotlib_overlays_suspended is False
    assert events == ["suspend", "restore"]

    assert (
        main_matplotlib_interaction.restore_main_matplotlib_overlays(
            runtime_state,
            restore_callback=lambda: events.append("restore-again"),
        )
        is False
    )
    assert events == ["suspend", "restore"]
