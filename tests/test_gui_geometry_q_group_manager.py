from ra_sim.gui import geometry_q_group_manager, state


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _entry(group_key, *, peak_count, total_intensity, source="primary"):
    return {
        "key": group_key,
        "source_label": source,
        "qr": 1.25,
        "qz": 0.5,
        "gz_index": group_key[3],
        "total_intensity": total_intensity,
        "peak_count": peak_count,
        "hkl_preview": [(1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)],
    }


def test_geometry_q_group_manager_formats_lines_and_builds_status_text() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key2})
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )

    line = geometry_q_group_manager.format_geometry_q_group_line(
        q_group_state.cached_entries[0]
    )
    status = geometry_q_group_manager.build_geometry_q_group_window_status_text(
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 4}}},
        current_geometry_fit_var_names=["gamma"],
    )

    assert "primary" in line
    assert "HKL=" in line
    assert ", ..." in line
    assert (
        geometry_q_group_manager.current_geometry_auto_match_min_matches(
            {},
            ["gamma", "chi"],
        )
        == 6
    )
    assert (
        geometry_q_group_manager.geometry_q_group_excluded_count(
            preview_state,
            q_group_state,
        )
        == 1
    )
    assert "Included Qr/Qz groups: 1/2" in status
    assert "Selected peaks: 2/5" in status
    assert "Need >= 4  short 2" in status
    assert "Intensity=10.000/30.000" in status


def test_refresh_geometry_q_group_window_uses_cached_entries_and_view_helpers(
    monkeypatch,
) -> None:
    calls = {}
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(("q_group", "primary", 1, 0), peak_count=2, total_intensity=10.0)]
    )

    def _refresh(**kwargs):
        calls["kwargs"] = kwargs
        return True

    monkeypatch.setattr(
        geometry_q_group_manager.gui_views,
        "refresh_geometry_q_group_window",
        _refresh,
    )

    ok = geometry_q_group_manager.refresh_geometry_q_group_window(
        view_state=state.GeometryQGroupViewState(window=object()),
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 1}}},
        current_geometry_fit_var_names=["gamma"],
        on_toggle=lambda key, row_var: (key, row_var),
    )

    assert ok is True
    assert calls["kwargs"]["entries"] == q_group_state.cached_entries
    assert calls["kwargs"]["format_line"] is geometry_q_group_manager.format_geometry_q_group_line
    assert "ready" in calls["kwargs"]["status_text"]


def test_geometry_q_group_manager_toggle_and_bulk_enable_update_preview_state() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )

    assert (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change(
            preview_state,
            key1,
            _FakeVar(False),
        )
        == "Excluded"
    )
    assert preview_state.excluded_q_groups == {key1}

    assert (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change(
            preview_state,
            key1,
            _FakeVar(True),
        )
        == "Included"
    )
    assert preview_state.excluded_q_groups == set()

    action, count = geometry_q_group_manager.set_all_geometry_q_groups_enabled(
        preview_state,
        q_group_state,
        enabled=False,
    )
    assert (action, count) == ("Excluded", 2)
    assert preview_state.excluded_q_groups == {key1, key2}

    action, count = geometry_q_group_manager.set_all_geometry_q_groups_enabled(
        preview_state,
        q_group_state,
        enabled=True,
    )
    assert (action, count) == ("Included", 2)
    assert preview_state.excluded_q_groups == set()


def test_geometry_q_group_manager_save_load_helpers_round_trip() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    key3 = ("q_group", "primary", 3, 2)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key2})
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )

    export_rows = geometry_q_group_manager.build_geometry_q_group_export_rows(
        preview_state=preview_state,
        q_group_state=q_group_state,
    )
    payload = geometry_q_group_manager.build_geometry_q_group_save_payload(
        export_rows,
        saved_at="2026-03-26T12:00:00",
    )
    saved_state, error = geometry_q_group_manager.load_geometry_q_group_saved_state(
        payload
    )

    assert error is None
    assert payload["included_count"] == 1
    assert export_rows[0]["included"] is True
    assert export_rows[1]["included"] is False
    assert "Qr=" in export_rows[0]["display_label"]
    assert saved_state == {key1: True, key2: False}

    target_preview_state = state.GeometryPreviewState()
    target_q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
            _entry(key3, peak_count=1, total_intensity=5.0),
        ]
    )

    summary, error = geometry_q_group_manager.apply_loaded_geometry_q_group_saved_state(
        preview_state=target_preview_state,
        q_group_state=target_q_group_state,
        saved_state=saved_state,
    )

    assert error is None
    assert summary == {
        "matched_total": 2,
        "included_total": 1,
        "current_only": 1,
        "saved_only": 0,
    }
    assert target_preview_state.excluded_q_groups == {key2, key3}


def test_geometry_q_group_manager_load_helpers_report_validation_errors() -> None:
    saved_state, error = geometry_q_group_manager.load_geometry_q_group_saved_state("bad")
    assert saved_state is None
    assert error == "Invalid Qr/Qz peak list file: expected a JSON object."

    summary, error = geometry_q_group_manager.apply_loaded_geometry_q_group_saved_state(
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        saved_state={},
    )
    assert summary is None
    assert error == "Loaded Qr/Qz peak list does not contain any valid rows."
