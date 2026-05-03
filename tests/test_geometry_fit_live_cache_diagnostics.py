from __future__ import annotations

from collections.abc import Mapping

from ra_sim.gui import geometry_fit
from ra_sim.gui._runtime import runtime_session


def _run_empty_live_cache_rebuild(
    cache_metadata: Mapping[str, object],
    *,
    required_pairs: list[dict[str, object]] | None = None,
) -> tuple[geometry_fit.GeometryFitSourceRowRebuildResult, dict[str, object]]:
    events: list[tuple[str, dict[str, object]]] = []

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=True,
        build_live_rows=lambda: {"rows": [], "cache_metadata": dict(cache_metadata)},
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: ([], None),
        logged_cache_matches_params=lambda _metadata, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: ([], [], [], []),
        simulate_hit_tables=lambda _params, **_kwargs: [],
        last_runtime_simulation_diagnostics=lambda: {"status": "empty_hit_tables"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs or [],
        live_cache_inventory={"source_snapshot_count": 0},
        stage_callback=lambda stage, payload: events.append((str(stage), dict(payload))),
    )

    ready_payload = next(
        payload
        for stage, payload in events
        if stage == "source_cache_live_runtime_cache_validation_ready"
    )
    return result, ready_payload


def test_live_runtime_cache_empty_log_includes_stored_hit_row_counts() -> None:
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "stored_max_positions_table_count": 2,
            "stored_max_positions_hit_row_count": 1285,
            "stored_peak_table_lattice_count": 2,
            "stored_source_reflection_indices_count": 1285,
            "built_stored_hit_table_peak_count": 0,
        }
    )

    assert payload["status"] == "empty_live_runtime_cache"
    assert payload["stored_max_positions_table_count"] == 2
    assert payload["stored_max_positions_hit_row_count"] == 1285
    assert payload["stored_peak_table_lattice_count"] == 2
    assert payload["stored_source_reflection_indices_count"] == 1285
    assert payload["built_stored_hit_table_peak_count"] == 0
    assert payload["reason"] == "stored_hit_tables_built_zero_rows"


def test_live_runtime_cache_diagnostics_reports_projection_drop_count() -> None:
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "stored_max_positions_hit_row_count": 40,
            "built_stored_hit_table_peak_count": 40,
            "projected_row_count": 0,
        }
    )

    assert payload["projected_row_count"] == 0
    assert payload["reason"] == "projection_dropped_all_rows"


def test_live_runtime_cache_diagnostics_reports_enabled_filter_drop_count() -> None:
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "stored_max_positions_hit_row_count": 40,
            "built_stored_hit_table_peak_count": 40,
            "projected_row_count": 40,
            "after_enabled_q_group_filter_count": 0,
        }
    )

    assert payload["after_enabled_q_group_filter_count"] == 0
    assert payload["reason"] == "enabled_q_group_filter_dropped_all_rows"


def test_live_runtime_cache_diagnostics_reports_required_pair_sources() -> None:
    required_pairs = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_label": "primary",
        },
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("q_group", "disordered_phase", 1, 0),
            "source_label": "disordered_phase",
        },
    ]
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "stored_max_positions_hit_row_count": 40,
            "built_stored_hit_table_peak_count": 40,
            "projected_row_count": 40,
            "after_enabled_q_group_filter_count": 40,
            "after_required_pair_source_filter_count": 0,
        },
        required_pairs=required_pairs,
    )

    assert payload["required_pair_count"] == 2
    assert payload["required_pair_sources"] == ["disordered_phase", "primary"]
    assert payload["after_required_pair_source_filter_count"] == 0
    assert payload["reason"] == "required_pair_source_filter_dropped_all_rows"


def test_live_runtime_cache_diagnostics_reports_source_counts() -> None:
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "stored_max_positions_hit_row_count": 40,
            "built_stored_hit_table_peak_count": 40,
            "projected_row_count": 40,
            "after_enabled_q_group_filter_count": 0,
            "source_counts_before_filter": {
                "primary": 20,
                "disordered_phase": 20,
            },
            "source_counts_after_filter": {},
        }
    )

    assert payload["source_counts_before_filter"] == {
        "disordered_phase": 20,
        "primary": 20,
    }
    assert payload["source_counts_after_filter"] == {}


def test_actual_live_cache_validation_ready_event_includes_drop_counts() -> None:
    message = runtime_session._format_source_cache_worker_event_message(
        "source_cache_live_runtime_cache_validation_ready",
        {
            "status": "empty_live_runtime_cache",
            "row_count": 0,
            "stored_hit_rows": 1285,
            "built_rows": 0,
            "projection_rows": 0,
            "enabled_filter_rows": 0,
            "required_pair_sources": ["primary", "disordered_phase"],
            "source_counts_before_filter": {
                "primary": 20,
                "disordered_phase": 20,
            },
            "source_counts_after_filter": {},
            "reason": "stored_hit_tables_built_zero_rows",
        },
    )

    assert "source_cache_live_runtime_cache_validation_ready" in message
    assert "stored_hit_rows=1285" in message
    assert "built_rows=0" in message
    assert "projection_rows=0" in message
    assert "enabled_filter_rows=0" in message
    assert "required_pair_sources=primary,disordered_phase" in message
    assert "source_counts_before_filter=" in message
    assert "source_counts_after_filter=" in message
    assert "reason=stored_hit_tables_built_zero_rows" in message


def test_diagnostics_not_only_computed_but_logged() -> None:
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "stored_max_positions_table_count": 2,
            "stored_max_positions_hit_row_count": 1285,
            "built_stored_hit_table_peak_count": 0,
            "projected_row_count": 0,
            "after_enabled_q_group_filter_count": 0,
            "source_counts_before_filter": {
                "primary": 20,
                "disordered_phase": 20,
            },
            "source_counts_after_filter": {},
        }
    )

    message = runtime_session._format_source_cache_worker_event_message(
        "source_cache_live_runtime_cache_validation_ready",
        payload,
    )

    assert "stored_hit_rows=1285" in message
    assert "built_rows=0" in message
    assert "projection_rows=0" in message
    assert "enabled_filter_rows=0" in message
    assert "source_counts_before_filter=" in message
    assert "reason=stored_hit_tables_built_zero_rows" in message


def test_live_cache_diagnostics_include_live_rows_raw_count() -> None:
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "live_rows_raw_count": 1285,
            "live_rows_payload_count": 0,
            "live_rows_signature_match": False,
            "reason": "requested_signature_mismatch",
        }
    )
    message = runtime_session._format_source_cache_worker_event_message(
        "source_cache_live_runtime_cache_validation_ready",
        payload,
    )

    assert payload["live_rows_raw_count"] == 1285
    assert payload["live_rows_payload_count"] == 0
    assert "live_rows_raw_count=1285" in message
    assert "live_rows_payload_count=0" in message


def test_live_cache_diagnostics_include_live_rows_source_counts() -> None:
    _result, payload = _run_empty_live_cache_rebuild(
        {
            "live_rows_raw_count": 40,
            "live_rows_source_counts": {
                "primary": 20,
                "disordered_phase": 20,
            },
            "source_counts_before_filter": {
                "primary": 20,
                "disordered_phase": 20,
            },
            "source_counts_after_filter": {},
        }
    )
    message = runtime_session._format_source_cache_worker_event_message(
        "source_cache_live_runtime_cache_validation_ready",
        payload,
    )

    assert payload["live_rows_source_counts"] == {
        "disordered_phase": 20,
        "primary": 20,
    }
    assert "live_rows_source_counts={disordered_phase=20,primary=20}" in message


def test_live_cache_diagnostics_distinguish_signature_mismatch_from_missing_rows() -> None:
    _result, mismatch_payload = _run_empty_live_cache_rebuild(
        {
            "live_rows_raw_count": 40,
            "live_rows_payload_count": 0,
            "live_rows_signature_match": False,
            "live_rows_signature_reason": "requested_signature_mismatch",
            "reason": "requested_signature_mismatch",
        }
    )
    _result, missing_payload = _run_empty_live_cache_rebuild(
        {
            "live_rows_raw_count": 0,
            "live_rows_payload_count": 0,
            "stored_max_positions_hit_row_count": 0,
        }
    )

    assert mismatch_payload["reason"] == "requested_signature_mismatch"
    assert mismatch_payload["live_rows_raw_count"] == 40
    assert mismatch_payload["live_rows_signature_match"] is False
    assert missing_payload["reason"] == "stored_hit_tables_missing_or_empty"
    assert missing_payload["live_rows_raw_count"] == 0


def test_live_trace_text_includes_phase4d1_markers_and_live_row_diagnostics() -> None:
    message = runtime_session._format_source_cache_worker_event_message(
        "source_cache_live_runtime_cache_validation_ready",
        {
            "geometry_fit_live_handoff_patch_marker": "phase4d1",
            "status": "valid",
            "row_count": 40,
            "live_rows_raw_count": 40,
            "live_rows_payload_count": 40,
            "live_rows_signature_match": True,
            "live_rows_cache_source": "q_group_snapshot",
            "live_rows_source_counts": {
                "primary": 21,
                "disordered_phase": 19,
            },
            "q_group_cached_entries": 40,
            "manual_picker_candidates": 0,
            "live_preview_rows_count": 0,
            "live_rows_by_background_current_count": 40,
            "live_rows_by_background_keys": [0],
            "requested_signature_keys": [0],
            "live_rows_signature_by_background_keys": [0],
            "reason": "accepted",
        },
    )

    assert "geometry_fit_live_handoff_patch_marker=phase4d1" in message
    assert "live_rows_raw_count=40" in message
    assert "live_rows_signature_match=true" in message
    assert "live_rows_source_counts={disordered_phase=19,primary=21}" in message
    assert "q_group_cached_entries=40" in message
    assert "live_rows_by_background_current_count=40" in message


def test_fresh_rebuild_wrapper_marker_is_formatted() -> None:
    message = runtime_session._format_source_cache_worker_event_message(
        "source_cache_fresh_rebuild_consumer_wrapper",
        {
            "fresh_rebuild_consumer_wrapper": "deduped",
            "cache_source": "fresh_simulation",
        },
    )

    assert "fresh_rebuild_consumer_wrapper=deduped" in message
