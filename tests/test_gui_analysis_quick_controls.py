from __future__ import annotations

from ra_sim.gui import analysis_quick_controls


class _FakeVar:
    def __init__(self, value: bool) -> None:
        self.value = bool(value)
        self.set_calls: list[bool] = []

    def get(self) -> bool:
        return self.value

    def set(self, value: bool) -> None:
        self.value = bool(value)
        self.set_calls.append(self.value)


def test_clear_analysis_integration_region_clears_region_and_effects() -> None:
    events: list[object] = []
    show_1d_var = _FakeVar(True)

    updated = analysis_quick_controls.clear_analysis_integration_region(
        show_1d_var=show_1d_var,
        hide_drag_region=lambda: events.append("hide"),
        disable_peak_pick=lambda: events.append("stop-pick"),
        clear_peak_selection=lambda: events.append("clear-peaks"),
        apply_cleared_integration=lambda: events.append("apply-clear"),
        set_status_text=lambda text: events.append(("status", text)),
    )

    assert updated is True
    assert show_1d_var.value is False
    assert show_1d_var.set_calls == [False]
    assert events == [
        "hide",
        "stop-pick",
        "clear-peaks",
        "apply-clear",
        (
            "status",
            analysis_quick_controls.DEFAULT_CLEAR_INTEGRATION_MESSAGE,
        ),
    ]


def test_clear_analysis_integration_region_tolerates_missing_or_failing_callbacks() -> None:
    class _BadVar:
        def set(self, _value: bool) -> None:
            raise RuntimeError("boom")

    updated = analysis_quick_controls.clear_analysis_integration_region(
        show_1d_var=_BadVar(),
        hide_drag_region=lambda: (_ for _ in ()).throw(RuntimeError("hide")),
        disable_peak_pick=None,
        clear_peak_selection=lambda: None,
        apply_cleared_integration=lambda: (_ for _ in ()).throw(RuntimeError("apply")),
        set_status_text=lambda _text: (_ for _ in ()).throw(RuntimeError("status")),
    )

    assert updated is True
