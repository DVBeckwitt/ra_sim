from datetime import datetime
from pathlib import Path

from ra_sim.gui import runtime_update_trace


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
