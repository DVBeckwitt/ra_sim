from __future__ import annotations

from collections.abc import Sequence

from ra_sim.gui import geometry_fit


DISORDERED = "disordered_phase"


def _row(source_label: str) -> dict[str, object]:
    return {
        "source_label": source_label,
        "q_group_key": ("q_group", source_label, 1, 4),
        "hkl": (-1, 1, 4),
        "source_branch_index": 0,
        "sim_col": 300.0,
        "sim_row": 400.0,
    }


def _disordered_pair() -> dict[str, object]:
    return {
        "pair_id": "bg0:disordered:pair0",
        "source_label": DISORDERED,
        "q_group_key": ("q_group", DISORDERED, 1, 4),
        "hkl": (-1, 1, 4),
        "source_branch_index": 0,
    }


def _run_fresh_with_rows(
    rows: Sequence[object],
) -> tuple[geometry_fit.GeometryFitSourceRowRebuildResult, list[tuple[str, dict[str, object]]]]:
    events: list[tuple[str, dict[str, object]]] = []
    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        memory_cache_signature=("sig", 0),
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: ([], None),
        logged_cache_matches_params=lambda _metadata, _params: {"matches": False},
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (list(rows), [], [], []),
        simulate_hit_tables=lambda _params, **_kwargs: [object()],
        last_runtime_simulation_diagnostics=lambda: {
            "status": "ready",
            "targeted_simulation_supported": True,
            "targeted_simulation_used": True,
        },
        project_rows=lambda source_rows: list(source_rows or ()),
        required_pairs=[_disordered_pair()],
        stage_callback=lambda stage, payload: events.append((str(stage), dict(payload))),
    )
    return result, events


def test_disordered_fit_pair_does_not_fallback_to_primary_only_fresh_simulation() -> None:
    result, events = _run_fresh_with_rows([_row("primary")])

    assert result.rebuild_source is None
    assert "source_cache_fresh_simulation_required_source_missing" in [stage for stage, _ in events]


def test_disordered_fit_pair_requires_disordered_source_rows() -> None:
    result, _events = _run_fresh_with_rows([_row(DISORDERED)])

    assert result.rebuild_source == "fresh_simulation"
    assert result.projected_rows
    assert result.projected_rows[0]["source_label"] == DISORDERED


def test_disordered_fit_preflight_reports_clear_error_when_rows_missing() -> None:
    result, events = _run_fresh_with_rows([_row("primary")])
    missing_payload = next(
        payload
        for stage, payload in events
        if stage == "source_cache_fresh_simulation_required_source_missing"
    )

    assert result.diagnostics["status"] == "required_source_rows_unavailable"
    assert missing_payload["status"] == "required_source_rows_unavailable"
    assert missing_payload["reason"] == "required_source_rows_unavailable"
    assert missing_payload["missing_required_sources"] == [DISORDERED]
