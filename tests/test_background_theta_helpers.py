from __future__ import annotations

from ra_sim.gui import background_theta, state_io


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def test_parse_background_theta_values_accepts_common_delimiters() -> None:
    assert background_theta.parse_background_theta_values("1.0, 2.5  3.25;4.0") == [
        1.0,
        2.5,
        3.25,
        4.0,
    ]

    try:
        background_theta.parse_background_theta_values("1.0, bad", expected_count=2)
    except ValueError as exc:
        assert "Invalid theta value" in str(exc)
    else:
        raise AssertionError("Expected invalid theta token to raise ValueError")


def test_background_theta_for_index_adds_shared_offset_for_multiple_backgrounds() -> None:
    theta_for_index = background_theta.background_theta_for_index

    theta_initial_var = _Var(6.0)
    background_theta_list_var = _Var("4.0, 7.5")
    geometry_fit_background_selection_var = _Var("all")
    geometry_theta_offset_var = _Var("0.25")

    assert (
        theta_for_index(
            0,
            osc_files=["bg0.osc", "bg1.osc"],
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            current_background_index=0,
            strict_count=True,
        )
        == 4.25
    )
    assert (
        theta_for_index(
            1,
            osc_files=["bg0.osc", "bg1.osc"],
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            current_background_index=0,
            strict_count=True,
        )
        == 7.75
    )


def test_parse_geometry_fit_background_indices_supports_current_all_and_ranges() -> None:
    parse_selection = background_theta.parse_geometry_fit_background_indices

    assert parse_selection("current", total_count=4, current_index=2) == [2]
    assert parse_selection("all", total_count=3, current_index=1) == [0, 1, 2]
    assert parse_selection("1, 3-4", total_count=4, current_index=0) == [0, 2, 3]


def test_geometry_fit_shared_theta_offset_depends_on_selected_fit_backgrounds() -> None:
    selection_var = _Var("current")

    assert background_theta.current_geometry_fit_background_indices(
        osc_files=["bg0.osc", "bg1.osc", "bg2.osc"],
        current_background_index=1,
        geometry_fit_background_selection_var=selection_var,
        strict=True,
    ) == [1]
    assert (
        background_theta.geometry_fit_uses_shared_theta_offset(
            osc_files=["bg0.osc", "bg1.osc", "bg2.osc"],
            current_background_index=1,
            geometry_fit_background_selection_var=selection_var,
        )
        is False
    )

    selection_var = _Var("1,3")
    assert background_theta.current_geometry_fit_background_indices(
        osc_files=["bg0.osc", "bg1.osc", "bg2.osc"],
        current_background_index=1,
        geometry_fit_background_selection_var=selection_var,
        strict=True,
    ) == [0, 2]
    assert (
        background_theta.geometry_fit_uses_shared_theta_offset(
            osc_files=["bg0.osc", "bg1.osc", "bg2.osc"],
            current_background_index=1,
            geometry_fit_background_selection_var=selection_var,
        )
        is True
    )


def test_apply_geometry_fit_background_selection_skips_redraw_when_live_theta_is_unchanged() -> None:
    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    theta_initial_var = _Var(4.0)

    assert (
        background_theta.apply_geometry_fit_background_selection(
            osc_files=["bg0.osc", "bg1.osc"],
            current_background_index=0,
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=_Var("4.0, 7.5"),
            geometry_theta_offset_var=_Var("0.0"),
            geometry_fit_background_selection_var=_Var("current"),
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=lambda: None,
            schedule_update=_schedule_update,
            trigger_update=True,
        )
        is True
    )
    assert theta_initial_var.get() == 4.0
    assert scheduled["count"] == 0


def test_apply_geometry_fit_background_selection_redraws_when_live_theta_changes() -> None:
    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    theta_initial_var = _Var(4.25)

    assert (
        background_theta.apply_geometry_fit_background_selection(
            osc_files=["bg0.osc", "bg1.osc"],
            current_background_index=0,
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=_Var("4.0, 7.5"),
            geometry_theta_offset_var=_Var("0.25"),
            geometry_fit_background_selection_var=_Var("current"),
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=lambda: None,
            schedule_update=_schedule_update,
            trigger_update=True,
        )
        is True
    )
    assert theta_initial_var.get() == 4.0
    assert scheduled["count"] == 1


def test_apply_geometry_fit_background_selection_preserves_live_theta_when_theta_list_is_blank() -> None:
    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    theta_initial_var = _Var(12.75)

    assert (
        background_theta.apply_geometry_fit_background_selection(
            osc_files=["bg0.osc", "bg1.osc"],
            current_background_index=0,
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=_Var(""),
            geometry_theta_offset_var=_Var("0.0"),
            geometry_fit_background_selection_var=_Var("current"),
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=lambda: None,
            schedule_update=_schedule_update,
            trigger_update=True,
        )
        is True
    )
    assert theta_initial_var.get() == 12.75
    assert scheduled["count"] == 0


def test_apply_geometry_fit_background_selection_can_skip_live_theta_sync() -> None:
    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    theta_initial_var = _Var(12.75)
    selection_var = _Var("current")

    assert (
        background_theta.apply_geometry_fit_background_selection(
            osc_files=["bg0.osc", "bg1.osc"],
            current_background_index=0,
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=_Var("4.0, 7.5"),
            geometry_theta_offset_var=_Var("0.25"),
            geometry_fit_background_selection_var=selection_var,
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=lambda: None,
            schedule_update=_schedule_update,
            trigger_update=True,
            sync_live_theta=False,
        )
        is True
    )
    assert selection_var.get() == "current"
    assert theta_initial_var.get() == 12.75
    assert scheduled["count"] == 0


def test_apply_background_theta_metadata_can_skip_live_theta_sync() -> None:
    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    theta_initial_var = _Var(12.75)
    background_theta_list_var = _Var("4.0, 7.5")

    assert (
        background_theta.apply_background_theta_metadata(
            osc_files=["bg0.osc", "bg1.osc"],
            current_background_index=0,
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=_Var("0.25"),
            geometry_fit_background_selection_var=_Var("all"),
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=lambda: None,
            schedule_update=_schedule_update,
            trigger_update=False,
            sync_live_theta=False,
        )
        is True
    )
    assert background_theta_list_var.get() == "4, 7.5"
    assert theta_initial_var.get() == 12.75
    assert scheduled["count"] == 0


def test_apply_gui_state_background_theta_compatibility_seeds_legacy_theta_list() -> None:
    osc_files = ["bg0.osc", "bg1.osc"]
    theta_initial_var = _Var(12.75)
    background_theta_list_var = _Var("6, 6")
    geometry_theta_offset_var = _Var("1.5")
    geometry_fit_background_selection_var = _Var("current")

    state_io.apply_gui_state_background_theta_compatibility(
        {"theta_initial_var": 12.75},
        osc_files=osc_files,
        theta_initial_var=theta_initial_var,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        format_background_theta_values=background_theta.format_background_theta_values,
        default_geometry_fit_background_selection=lambda: (
            background_theta.default_geometry_fit_background_selection(
                osc_files=osc_files
            )
        ),
    )

    assert background_theta_list_var.get() == "12.75, 12.75"
    assert geometry_theta_offset_var.get() == "0.0"
    assert geometry_fit_background_selection_var.get() == "all"


def test_apply_gui_state_background_theta_compatibility_preserves_saved_metadata() -> None:
    osc_files = ["bg0.osc", "bg1.osc"]
    theta_initial_var = _Var(12.75)
    background_theta_list_var = _Var("4, 7.5")
    geometry_theta_offset_var = _Var("0.25")
    geometry_fit_background_selection_var = _Var("current")

    state_io.apply_gui_state_background_theta_compatibility(
        {
            "theta_initial_var": 12.75,
            "background_theta_list_var": "4, 7.5",
            "geometry_theta_offset_var": "0.25",
            "geometry_fit_background_selection_var": "current",
        },
        osc_files=osc_files,
        theta_initial_var=theta_initial_var,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        format_background_theta_values=background_theta.format_background_theta_values,
        default_geometry_fit_background_selection=lambda: (
            background_theta.default_geometry_fit_background_selection(
                osc_files=osc_files
            )
        ),
    )

    assert background_theta_list_var.get() == "4, 7.5"
    assert geometry_theta_offset_var.get() == "0.25"
    assert geometry_fit_background_selection_var.get() == "current"
