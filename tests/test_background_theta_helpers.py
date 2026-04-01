from __future__ import annotations

from types import SimpleNamespace

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


def test_background_theta_base_for_index_ignores_shared_offset() -> None:
    theta_base_for_index = background_theta.background_theta_base_for_index

    assert (
        theta_base_for_index(
            1,
            osc_files=["bg0.osc", "bg1.osc"],
            theta_initial_var=_Var(6.0),
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=_Var("4.0, 7.5"),
            strict_count=True,
        )
        == 7.5
    )


def test_sync_live_theta_to_background_theta_list_updates_current_background_only() -> None:
    theta_initial_var = _Var(9.25)
    background_theta_list_var = _Var("4.0, 7.5, 10.0")

    assert (
        background_theta.sync_live_theta_to_background_theta_list(
            osc_files=["bg0.osc", "bg1.osc", "bg2.osc"],
            current_background_index=1,
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=background_theta_list_var,
        )
        is True
    )
    assert background_theta_list_var.get() == "4, 9.25, 10"


def test_parse_geometry_fit_background_indices_supports_current_all_and_ranges() -> None:
    parse_selection = background_theta.parse_geometry_fit_background_indices

    assert parse_selection("current", total_count=4, current_index=2) == [2]
    assert parse_selection("all", total_count=3, current_index=1) == [0, 1, 2]
    assert parse_selection("1, 3-4", total_count=4, current_index=0) == [0, 2, 3]


def test_serialize_geometry_fit_background_selection_canonicalizes_visual_selector_output() -> None:
    assert (
        background_theta.serialize_geometry_fit_background_selection(
            selected_indices=[0, 1, 2],
            total_count=3,
            current_index=1,
        )
        == "all"
    )
    assert (
        background_theta.serialize_geometry_fit_background_selection(
            selected_indices=[2],
            total_count=4,
            current_index=2,
        )
        == "current"
    )
    assert (
        background_theta.serialize_geometry_fit_background_selection(
            selected_indices=[3, 0, 2],
            total_count=4,
            current_index=1,
        )
        == "1,3,4"
    )
    assert (
        background_theta.serialize_geometry_fit_background_selection(
            selected_indices=[],
            total_count=4,
            current_index=1,
        )
        == "current"
    )


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


def test_apply_geometry_fit_background_selection_redraws_when_shared_offset_changes_effective_theta() -> None:
    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    theta_initial_var = _Var(4.0)
    selection_var = _Var("current")
    theta_controls = {}

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
            theta_controls=theta_controls,
            set_background_file_status_text=lambda: None,
            schedule_update=_schedule_update,
            trigger_update=True,
        )
        is True
    )
    selection_var.set("all")
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
            theta_controls=theta_controls,
            set_background_file_status_text=lambda: None,
            schedule_update=_schedule_update,
            trigger_update=True,
        )
        is True
    )
    assert theta_initial_var.get() == 4.0
    assert selection_var.get() == "all"
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


def test_apply_background_theta_metadata_syncs_live_theta_to_base_value() -> None:
    theta_initial_var = _Var(0.0)

    assert (
        background_theta.apply_background_theta_metadata(
            osc_files=["bg0.osc", "bg1.osc"],
            current_background_index=0,
            theta_initial_var=theta_initial_var,
            defaults={"theta_initial": 6.0},
            theta_initial=6.0,
            background_theta_list_var=_Var("4.0, 7.5"),
            geometry_theta_offset_var=_Var("0.25"),
            geometry_fit_background_selection_var=_Var("all"),
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=lambda: None,
            schedule_update=None,
            trigger_update=False,
            sync_live_theta=True,
        )
        is True
    )
    assert theta_initial_var.get() == 4.0


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


def test_background_theta_runtime_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {
        "osc_files": 0,
        "index": 0,
        "theta_var": 0,
        "theta_list": 0,
        "theta_offset": 0,
        "selection": 0,
        "checkbutton": 0,
        "controls": 0,
        "status": 0,
        "schedule": 0,
        "progress": 0,
        "progress_geometry": 0,
    }

    monkeypatch.setattr(
        background_theta,
        "BackgroundThetaRuntimeBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def _bump(key: str, prefix: str):
        counters[key] += 1
        return f"{prefix}-{counters[key]}"

    factory = background_theta.make_runtime_background_theta_bindings_factory(
        osc_files_factory=lambda: [f"bg-{_bump('osc_files', 'file')}.osc"],
        current_background_index_factory=lambda: counters.__setitem__(
            "index",
            counters["index"] + 1,
        )
        or counters["index"],
        theta_initial_var_factory=lambda: _bump("theta_var", "theta-var"),
        defaults={"theta_initial": 6.0},
        theta_initial=7.0,
        background_theta_list_var_factory=lambda: _bump("theta_list", "theta-list"),
        geometry_theta_offset_var_factory=lambda: _bump(
            "theta_offset",
            "theta-offset",
        ),
        geometry_fit_background_selection_var_factory=lambda: _bump(
            "selection",
            "selection",
        ),
        fit_theta_checkbutton_factory=lambda: _bump("checkbutton", "checkbutton"),
        theta_controls_factory=lambda: _bump("controls", "controls"),
        set_background_file_status_text_factory=lambda: _bump("status", "status"),
        schedule_update_factory=lambda: _bump("schedule", "schedule"),
        progress_label_factory=lambda: _bump("progress", "progress"),
        progress_label_geometry_factory=lambda: _bump(
            "progress_geometry",
            "progress-geometry",
        ),
    )

    first = factory()
    second = factory()

    assert first["osc_files"] == ("bg-file-1.osc",)
    assert second["osc_files"] == ("bg-file-2.osc",)
    assert first["current_background_index"] == 1
    assert second["current_background_index"] == 2
    assert first["defaults"] == {"theta_initial": 6.0}
    assert first["theta_initial"] == 7.0
    assert first["theta_initial_var"] == "theta-var-1"
    assert second["geometry_fit_background_selection_var"] == "selection-2"
    assert first["fit_theta_checkbutton"] == "checkbutton-1"
    assert second["theta_controls"] == "controls-2"
    assert first["set_background_file_status_text"] == "status-1"
    assert second["schedule_update"] == "schedule-2"
    assert first["progress_label"] == "progress-1"
    assert second["progress_label_geometry"] == "progress-geometry-2"


def test_background_theta_runtime_callbacks_delegate_to_live_helpers(
    monkeypatch,
) -> None:
    calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        background_theta,
        "runtime_current_geometry_fit_background_indices",
        lambda bindings, *, strict=False: calls.append(
            ("indices", bindings, strict)
        )
        or ["indices"],
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_geometry_fit_uses_shared_theta_offset",
        lambda bindings, selected_indices=None: calls.append(
            ("shared", bindings, selected_indices)
        )
        or True,
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_current_geometry_theta_offset",
        lambda bindings, *, strict=False: calls.append(
            ("offset", bindings, strict)
        )
        or 1.25,
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_current_background_theta_values",
        lambda bindings, *, strict_count=False: calls.append(
            ("values", bindings, strict_count)
        )
        or [3.0, 4.0],
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_background_theta_for_index",
        lambda bindings, index, *, strict_count=False: calls.append(
            ("theta-for-index", bindings, index, strict_count)
        )
        or 7.5,
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_sync_background_theta_controls",
        lambda bindings, *, preserve_existing=True, trigger_update=False: calls.append(
            ("sync-controls", bindings, preserve_existing, trigger_update)
        ),
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_apply_background_theta_metadata",
        lambda bindings, *, trigger_update=True, sync_live_theta=True: calls.append(
            ("apply-metadata", bindings, trigger_update, sync_live_theta)
        )
        or False,
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_apply_geometry_fit_background_selection",
        lambda bindings, *, trigger_update=False, sync_live_theta=True: calls.append(
            ("apply-selection", bindings, trigger_update, sync_live_theta)
        )
        or True,
    )
    monkeypatch.setattr(
        background_theta,
        "runtime_sync_geometry_fit_background_selection",
        lambda bindings, *, preserve_existing=True: calls.append(
            ("sync-selection", bindings, preserve_existing)
        ),
    )

    def _bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = background_theta.make_runtime_background_theta_callbacks(_bindings)

    assert callbacks.current_geometry_fit_background_indices(strict=True) == ["indices"]
    assert callbacks.geometry_fit_uses_shared_theta_offset([0, 1]) is True
    assert callbacks.current_geometry_theta_offset(strict=True) == 1.25
    assert callbacks.current_background_theta_values(strict_count=True) == [3.0, 4.0]
    assert callbacks.background_theta_for_index(2, strict_count=True) == 7.5
    callbacks.sync_background_theta_controls(
        preserve_existing=False,
        trigger_update=True,
    )
    assert (
        callbacks.apply_background_theta_metadata(
            trigger_update=False,
            sync_live_theta=False,
        )
        is False
    )
    assert (
        callbacks.apply_geometry_fit_background_selection(
            trigger_update=True,
            sync_live_theta=False,
        )
        is True
    )
    callbacks.sync_geometry_fit_background_selection(preserve_existing=False)

    assert calls == [
        ("indices", "bindings-1", True),
        ("shared", "bindings-2", [0, 1]),
        ("offset", "bindings-3", True),
        ("values", "bindings-4", True),
        ("theta-for-index", "bindings-5", 2, True),
        ("sync-controls", "bindings-6", False, True),
        ("apply-metadata", "bindings-7", False, False),
        ("apply-selection", "bindings-8", True, False),
        ("sync-selection", "bindings-9", False),
    ]
