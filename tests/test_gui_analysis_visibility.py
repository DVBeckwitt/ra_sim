from __future__ import annotations

from ra_sim.gui import analysis_visibility


class _FakeVar:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value


def test_analysis_outputs_visible_when_analyze_tab_selected() -> None:
    assert analysis_visibility.analysis_outputs_visible(
        control_tab_var=_FakeVar("analyze"),
        popout_open=False,
        assume_visible_when_unknown=False,
    ) is True


def test_analysis_outputs_visible_when_popout_open_even_off_tab() -> None:
    assert analysis_visibility.analysis_outputs_visible(
        control_tab_var=_FakeVar("setup"),
        popout_open=True,
        assume_visible_when_unknown=False,
    ) is True


def test_analysis_outputs_visible_is_false_when_off_tab_and_not_popped_out() -> None:
    assert analysis_visibility.analysis_outputs_visible(
        control_tab_var=_FakeVar("match"),
        popout_open=False,
        assume_visible_when_unknown=False,
    ) is False


def test_analysis_outputs_visible_can_fall_back_to_visible_when_state_missing() -> None:
    assert analysis_visibility.analysis_outputs_visible(
        control_tab_var=None,
        popout_open=False,
        assume_visible_when_unknown=True,
    ) is True
