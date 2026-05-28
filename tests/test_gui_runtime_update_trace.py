from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from ra_sim.gui import runtime_update_trace


def test_initial_update_decision_trace_contains_runtime_classifier_fields() -> None:
    trace = runtime_update_trace.initial_update_decision_trace()

    assert trace == {
        "update_action": None,
        "update_reason": None,
        "requires_worker": None,
        "missing_contribution_count": None,
        "center_remap_used": None,
        "primary_prune_cache_mode": None,
        "qr_selector_entries_retained": None,
        "qr_selector_entries_refreshed": None,
        "qr_selector_refresh_deferred": None,
        "source_row_snapshots_retained": None,
        "q_group_content_signature_changed": None,
        "geometry_fitter_handoff_valid": None,
        "qr_selector_branch_identity_retained": None,
        "detector_projection_cache_refreshed": None,
        "caked_projection_cache_invalidated": None,
        "center_remap_fallback_reason": None,
        "classifier_update_action": None,
        "classifier_update_reason": None,
        "classifier_requires_worker": None,
        "classifier_requires_analysis": None,
        "classifier_missing_contribution_count": None,
        "effective_update_action": None,
    }


def test_initial_update_decision_trace_returns_independent_dicts() -> None:
    first = runtime_update_trace.initial_update_decision_trace()
    second = runtime_update_trace.initial_update_decision_trace()

    first["update_action"] = "full_simulation"

    assert second["update_action"] is None


def test_set_update_decision_trace_coerces_runtime_fields() -> None:
    trace = runtime_update_trace.initial_update_decision_trace()

    runtime_update_trace.set_update_decision_trace(
        trace,
        update_action="primary_prune_reuse",
        update_reason=123,
        requires_worker=0,
        missing_contribution_count="4",
        center_remap_used=1,
        primary_prune_cache_mode=Path("reuse"),
    )

    assert trace["update_action"] == "primary_prune_reuse"
    assert trace["update_reason"] == "123"
    assert trace["requires_worker"] is False
    assert trace["missing_contribution_count"] == 4
    assert trace["center_remap_used"] is True
    assert trace["primary_prune_cache_mode"] == str(Path("reuse"))


def test_set_qr_selector_trace_derives_policy_defaults() -> None:
    trace = runtime_update_trace.initial_update_decision_trace()
    policy = SimpleNamespace(
        retain_geometry_q_group_entries=True,
        defer_q_group_refresh_until_rows_available=True,
        retain_source_row_snapshots=True,
        retain_intersection_caches=True,
        retain_manual_pick_cache=True,
        require_q_group_refresh_after_apply=False,
    )

    runtime_update_trace.set_qr_selector_trace(
        trace,
        policy=policy,
        qr_selector_entries_refreshed=False,
        q_group_content_signature_changed=True,
    )

    assert trace["qr_selector_entries_retained"] is True
    assert trace["qr_selector_refresh_deferred"] is True
    assert trace["source_row_snapshots_retained"] is True
    assert trace["geometry_fitter_handoff_valid"] is True
    assert trace["qr_selector_entries_refreshed"] is False
    assert trace["q_group_content_signature_changed"] is True


def test_set_detector_center_remap_trace_records_remap_fields() -> None:
    trace = runtime_update_trace.initial_update_decision_trace()

    runtime_update_trace.set_detector_center_remap_trace(
        trace,
        qr_selector_branch_identity_retained=True,
        detector_projection_cache_refreshed=False,
        caked_projection_cache_invalidated=True,
        geometry_fitter_handoff_valid=False,
        center_remap_fallback_reason=Path("fallback"),
    )

    assert trace["qr_selector_branch_identity_retained"] is True
    assert trace["detector_projection_cache_refreshed"] is False
    assert trace["caked_projection_cache_invalidated"] is True
    assert trace["geometry_fitter_handoff_valid"] is False
    assert trace["center_remap_fallback_reason"] == str(Path("fallback"))


def test_private_update_decision_action_and_defaults() -> None:
    trace = runtime_update_trace.initial_update_decision_trace()

    assert runtime_update_trace._update_decision_action(trace) == ""

    runtime_update_trace.ensure_update_decision_defaults(trace, "display_refresh")

    assert runtime_update_trace._update_decision_action(trace) == "display_only"
    assert trace["update_reason"] == "display_refresh"
    assert trace["requires_worker"] is False
    assert trace["missing_contribution_count"] == 0
    assert trace["center_remap_used"] is False
    assert trace["primary_prune_cache_mode"] == "none"


def test_ensure_update_decision_defaults_preserves_existing_action() -> None:
    trace = runtime_update_trace.initial_update_decision_trace()
    runtime_update_trace.set_update_decision_trace(
        trace,
        update_action="full_simulation",
        update_reason="source_changed",
        requires_worker=True,
    )

    runtime_update_trace.ensure_update_decision_defaults(trace, "display_refresh")

    assert trace["update_action"] == "full_simulation"
    assert trace["update_reason"] == "source_changed"
    assert trace["requires_worker"] is True


def test_set_classifier_decision_trace_records_classifier_and_effective_action() -> None:
    trace = runtime_update_trace.initial_update_decision_trace()
    decision = SimpleNamespace(
        action=SimpleNamespace(value="primary_prune_fill"),
        reason="primary_filter_changed_cache_fill",
        requires_worker=True,
        requires_analysis=False,
        missing_contribution_keys={"rod-a", "rod-b"},
    )
    effective_action = SimpleNamespace(value="full_simulation")

    runtime_update_trace.set_classifier_decision_trace(
        trace,
        decision,
        effective_action=effective_action,
    )

    assert trace["classifier_update_action"] == "primary_prune_fill"
    assert trace["classifier_update_reason"] == "primary_filter_changed_cache_fill"
    assert trace["classifier_requires_worker"] is True
    assert trace["classifier_requires_analysis"] is False
    assert trace["classifier_missing_contribution_count"] == 2
    assert trace["effective_update_action"] == "full_simulation"


def test_resolve_runtime_update_trace_path_uses_daily_stamp() -> None:
    path = runtime_update_trace.resolve_runtime_update_trace_path(
        "C:/tmp/downloads",
        current_time=datetime(2026, 3, 31, 12, 15, 0),
    )

    assert path == Path("C:/tmp/downloads/runtime_update_trace_20260331.log")


def test_format_runtime_update_trace_line_formats_scalar_and_sequence_fields() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_start",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "phase": "applying",
            "update_id": 7,
            "ready": True,
            "cost": 1.25,
            "values": [1, 2, 3],
            "message": "line1\nline2",
        },
    )

    assert line == (
        "2026-03-31T12:15:00.123 pid=4242 event=do_update_start "
        "cost=1.250000 message=line1\\nline2 phase=applying ready=true "
        "update_id=7 values=[1,2,3]"
    )


def test_update_trace_includes_optimization_fields() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "full_simulation",
            "update_reason": "image_signature_changed",
            "requires_worker": True,
            "missing_contribution_count": 0,
            "center_remap_used": False,
            "primary_prune_cache_mode": "none",
        },
    )

    assert "update_action=full_simulation" in line
    assert "update_reason=image_signature_changed" in line
    assert "requires_worker=true" in line
    assert "missing_contribution_count=0" in line
    assert "center_remap_used=false" in line
    assert "primary_prune_cache_mode=none" in line


def test_trace_reports_primary_prune_reuse() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "primary_prune_reuse",
            "requires_worker": False,
            "primary_prune_cache_mode": "reuse",
            "missing_contribution_count": 0,
        },
    )

    assert "update_action=primary_prune_reuse" in line
    assert "requires_worker=false" in line
    assert "primary_prune_cache_mode=reuse" in line
    assert "missing_contribution_count=0" in line


def test_trace_reports_primary_prune_fill() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_extend_primary_cache_in_background",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "primary_prune_fill",
            "requires_worker": True,
            "primary_prune_cache_mode": "fill",
            "missing_contribution_count": 2,
        },
    )

    assert "update_action=primary_prune_fill" in line
    assert "requires_worker=true" in line
    assert "primary_prune_cache_mode=fill" in line
    assert "missing_contribution_count=2" in line


def test_prune_trace_reports_qr_selector_retention_or_refresh() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "primary_prune_reuse",
            "primary_prune_cache_mode": "reuse",
            "qr_selector_entries_retained": True,
            "qr_selector_entries_refreshed": False,
            "qr_selector_refresh_deferred": False,
            "source_row_snapshots_retained": True,
            "q_group_content_signature_changed": False,
            "geometry_fitter_handoff_valid": True,
        },
    )

    assert "update_action=primary_prune_reuse" in line
    assert "primary_prune_cache_mode=reuse" in line
    assert "qr_selector_entries_retained=true" in line
    assert "qr_selector_entries_refreshed=false" in line
    assert "qr_selector_refresh_deferred=false" in line
    assert "source_row_snapshots_retained=true" in line
    assert "q_group_content_signature_changed=false" in line
    assert "geometry_fitter_handoff_valid=true" in line


def test_trace_reports_display_only() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "display_only",
            "requires_worker": False,
            "update_reason": "display_dependency_changed",
        },
    )

    assert "update_action=display_only" in line
    assert "requires_worker=false" in line
    assert "update_reason=display_dependency_changed" in line


def test_trace_reports_combine_only() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "combine_only",
            "requires_worker": False,
            "update_reason": "combine_dependency_changed",
        },
    )

    assert "update_action=combine_only" in line
    assert "requires_worker=false" in line
    assert "update_reason=combine_dependency_changed" in line


def test_trace_reports_detector_center_remap() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "detector_center_remap",
            "requires_worker": False,
            "center_remap_used": True,
            "update_reason": "detector_center_changed_exact_cache_available",
        },
    )

    assert "update_action=detector_center_remap" in line
    assert "requires_worker=false" in line
    assert "center_remap_used=true" in line
    assert "update_reason=detector_center_changed_exact_cache_available" in line


def test_center_remap_trace_reports_projection_and_handoff_state() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "detector_center_remap",
            "requires_worker": False,
            "center_remap_used": True,
            "qr_selector_branch_identity_retained": True,
            "source_row_snapshots_retained": True,
            "detector_projection_cache_refreshed": True,
            "caked_projection_cache_invalidated": True,
            "geometry_fitter_handoff_valid": False,
            "center_remap_fallback_reason": "none",
        },
    )

    assert "update_action=detector_center_remap" in line
    assert "requires_worker=false" in line
    assert "center_remap_used=true" in line
    assert "qr_selector_branch_identity_retained=true" in line
    assert "source_row_snapshots_retained=true" in line
    assert "detector_projection_cache_refreshed=true" in line
    assert "caked_projection_cache_invalidated=true" in line
    assert "geometry_fitter_handoff_valid=false" in line
    assert "center_remap_fallback_reason=none" in line


def test_mixed_update_trace_reports_full_simulation_not_center_remap() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_return_waiting_for_simulation",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "full_simulation",
            "requires_worker": True,
            "center_remap_used": False,
            "center_remap_fallback_reason": "physics_dependency_changed",
            "primary_prune_cache_mode": "full",
        },
    )

    assert "update_action=full_simulation" in line
    assert "requires_worker=true" in line
    assert "center_remap_used=false" in line
    assert "center_remap_fallback_reason=physics_dependency_changed" in line
    assert "primary_prune_cache_mode=full" in line


def test_stale_worker_trace_reports_ignored_result() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_discard_stale_ready_fast_path_result",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "ready_job_kind": "full",
            "ready_timing_update_id": 4,
            "update_action": "display_only",
        },
    )

    assert "event=do_update_discard_stale_ready_fast_path_result" in line
    assert "ready_job_kind=full" in line
    assert "ready_timing_update_id=4" in line
    assert "update_action=display_only" in line


def test_prune_fill_trace_keeps_refresh_deferred_across_display_update() -> None:
    line = runtime_update_trace.format_runtime_update_trace_line(
        "do_update_complete",
        timestamp=datetime(2026, 3, 31, 12, 15, 0, 123000),
        pid=4242,
        fields={
            "update_action": "display_only",
            "requires_worker": False,
            "qr_selector_entries_retained": True,
            "qr_selector_refresh_deferred": True,
            "q_group_content_signature_changed": True,
            "geometry_fitter_handoff_valid": False,
        },
    )

    assert "update_action=display_only" in line
    assert "requires_worker=false" in line
    assert "qr_selector_entries_retained=true" in line
    assert "qr_selector_refresh_deferred=true" in line
    assert "q_group_content_signature_changed=true" in line
    assert "geometry_fitter_handoff_valid=false" in line


def test_append_runtime_update_trace_line_writes_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_update_trace,
        "runtime_update_trace_logging_enabled",
        lambda: True,
    )
    trace_path = tmp_path / "runtime_update_trace_20260331.log"

    runtime_update_trace.append_runtime_update_trace_line(
        trace_path,
        "schedule_update",
        queued=True,
        update_id=9,
    )

    text = trace_path.read_text(encoding="utf-8")
    assert "event=schedule_update" in text
    assert "queued=true" in text
    assert "update_id=9" in text
