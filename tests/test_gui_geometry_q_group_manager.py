from datetime import datetime
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


def test_geometry_q_group_manager_checkbox_side_effects_update_status() -> None:
    key1 = ("q_group", "primary", 1, 0)
    preview_state = state.GeometryPreviewState()
    events = []

    changed = (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change_with_side_effects(
            preview_state=preview_state,
            group_key=key1,
            row_var=_FakeVar(False),
            invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
            update_geometry_preview_exclude_button_label=lambda: events.append("label"),
            update_geometry_q_group_window_status=lambda: events.append("status"),
            live_geometry_preview_enabled=lambda: False,
            refresh_live_geometry_preview=lambda: events.append("refresh_live"),
            set_status_text=lambda text: events.append(text),
        )
    )

    assert changed is True
    assert preview_state.excluded_q_groups == {key1}
    assert events == [
        "invalidate",
        "label",
        "status",
        "Excluded one Qr/Qz group for geometry fitting.",
    ]


def test_geometry_q_group_manager_bulk_enable_side_effects_cover_empty_and_live_refresh() -> None:
    preview_state = state.GeometryPreviewState()
    empty_state = state.GeometryQGroupState()
    empty_messages = []

    changed = (
        geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
            preview_state=preview_state,
            q_group_state=empty_state,
            enabled=False,
            invalidate_geometry_manual_pick_cache=lambda: empty_messages.append(
                "invalidate"
            ),
            update_geometry_preview_exclude_button_label=lambda: empty_messages.append(
                "label"
            ),
            refresh_geometry_q_group_window=lambda: empty_messages.append("refresh"),
            live_geometry_preview_enabled=lambda: False,
            refresh_live_geometry_preview=lambda: empty_messages.append("live"),
            set_status_text=lambda text: empty_messages.append(text),
        )
    )

    assert changed is False
    assert empty_messages == [
        'No listed Qr/Qz groups are available. Press "Update Listed Peaks" first.'
    ]

    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )
    events = []

    changed = (
        geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
            preview_state=preview_state,
            q_group_state=q_group_state,
            enabled=False,
            invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
            update_geometry_preview_exclude_button_label=lambda: events.append("label"),
            refresh_geometry_q_group_window=lambda: events.append("refresh"),
            live_geometry_preview_enabled=lambda: True,
            refresh_live_geometry_preview=lambda: events.append("live"),
            set_status_text=lambda text: events.append(text),
        )
    )

    assert changed is True
    assert preview_state.excluded_q_groups == {key1, key2}
    assert events == ["invalidate", "label", "refresh", "live"]


def test_geometry_q_group_manager_request_update_side_effects_marks_refresh() -> None:
    q_group_state = state.GeometryQGroupState()
    events = []

    geometry_q_group_manager.request_geometry_q_group_window_update_with_side_effects(
        q_group_state=q_group_state,
        clear_last_simulation_signature=lambda: events.append("clear_signature"),
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        set_status_text=lambda text: events.append(text),
        schedule_update=lambda: events.append("schedule"),
    )

    assert q_group_state.refresh_requested is True
    assert events == [
        "clear_signature",
        "invalidate",
        "Updating listed Qr/Qz peaks from the current simulation...",
        "schedule",
    ]


def test_geometry_q_group_manager_save_dialog_workflow_writes_payload_and_reports_status() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key2})
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )
    captured = {}
    messages = []

    def _asksaveasfilename(**kwargs):
        captured["dialog"] = kwargs
        return "C:/tmp/groups.json"

    saved = geometry_q_group_manager.save_geometry_q_group_selection_with_dialog(
        preview_state=preview_state,
        q_group_state=q_group_state,
        file_dialog_dir="C:/dialogs",
        asksaveasfilename=_asksaveasfilename,
        set_status_text=lambda text: messages.append(text),
        save_payload=lambda path, payload: captured.update(
            {"path": path, "payload": payload}
        ),
        now=lambda: datetime(2026, 3, 26, 12, 34, 56),
    )

    assert saved is True
    assert captured["dialog"]["initialdir"] == "C:/dialogs"
    assert captured["dialog"]["initialfile"] == "geometry_q_groups_20260326_123456.json"
    assert captured["path"] == "C:/tmp/groups.json"
    assert captured["payload"]["saved_at"] == "2026-03-26T12:34:56"
    assert captured["payload"]["included_count"] == 1
    assert messages == ["Saved 2 Qr/Qz groups to C:/tmp/groups.json"]


def test_geometry_q_group_manager_load_dialog_workflow_applies_state_and_refreshes() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    key3 = ("q_group", "primary", 3, 2)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
            _entry(key3, peak_count=1, total_intensity=5.0),
        ]
    )
    events = []
    payload = geometry_q_group_manager.build_geometry_q_group_save_payload(
        [
            {"key": ["q_group", "primary", 1, 0], "included": True},
            {"key": ["q_group", "secondary", 2, 1], "included": False},
        ],
        saved_at="2026-03-26T12:00:00",
    )

    def _askopenfilename(**kwargs):
        events.append(("dialog", kwargs))
        return "C:/tmp/selector.json"

    loaded = geometry_q_group_manager.load_geometry_q_group_selection_with_dialog(
        preview_state=preview_state,
        q_group_state=q_group_state,
        file_dialog_dir="C:/dialogs",
        askopenfilename=_askopenfilename,
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        refresh_geometry_q_group_window=lambda: events.append("refresh"),
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: events.append("live"),
        set_status_text=lambda text: events.append(text),
        load_payload=lambda path: events.append(("load", path)) or payload,
    )

    assert loaded is True
    assert preview_state.excluded_q_groups == {key2, key3}
    assert events[0][0] == "dialog"
    assert events[0][1]["initialdir"] == "C:/dialogs"
    assert events[1] == ("load", "C:/tmp/selector.json")
    assert events[2:5] == ["label", "refresh", "live"]
    assert (
        events[5]
        == "Loaded Qr/Qz peak list from selector.json: matched 2, enabled 1, current-only excluded 1, saved-only missing 0."
    )
