from __future__ import annotations

from collections.abc import Sequence

from ra_sim.gui import geometry_fit


DISORDERED = "disordered_phase"
PROJECTION_SIGNATURE = {"mode": "detector", "background_index": 0, "available": True}
REQUESTED_SIGNATURE = ("sig", 0)


def _row(source_label: str = "primary", *, detector: bool = True) -> dict[str, object]:
    row = {
        "source_label": source_label,
        "q_group_key": ("q_group", source_label, 1, 4),
        "hkl": (-1, 1, 4),
        "source_branch_index": 0,
    }
    if detector:
        row.update({"sim_col": 300.0, "sim_row": 400.0})
    return row


def _pair(source_label: str = "primary") -> dict[str, object]:
    return {
        "pair_id": f"bg0:{source_label}:pair0",
        "source_label": source_label,
        "q_group_key": ("q_group", source_label, 1, 4),
        "hkl": (-1, 1, 4),
        "source_branch_index": 0,
    }


def _run_rebuild(
    *,
    required_source: str = "primary",
    targeted_payload: dict[str, object] | None = None,
    live_rows: Sequence[object] | None = None,
    memory_hit_tables: Sequence[object] | None = None,
    logged_matches: bool = False,
    logged_hit_tables: Sequence[object] | None = None,
    fresh_hit_tables: Sequence[object] | None = None,
    built_rows: Sequence[object] | None = None,
) -> tuple[geometry_fit.GeometryFitSourceRowRebuildResult, list[tuple[str, dict[str, object]]]]:
    events: list[tuple[str, dict[str, object]]] = []
    rows_from_build = list(built_rows or [])

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=REQUESTED_SIGNATURE,
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=live_rows is not None,
        build_live_rows=(
            (lambda: {"rows": list(live_rows or []), "cache_metadata": {}})
            if live_rows is not None
            else None
        ),
        get_memory_intersection_cache=lambda: list(memory_hit_tables or []),
        memory_cache_signature=REQUESTED_SIGNATURE,
        load_logged_intersection_cache_metadata=lambda: {"status": "logged"},
        load_logged_intersection_cache=lambda: (
            list(logged_hit_tables or []),
            {"status": "logged"},
        ),
        logged_cache_matches_params=lambda _metadata, _params: {"matches": bool(logged_matches)},
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (rows_from_build, [], [], []),
        simulate_hit_tables=lambda _params, **_kwargs: list(fresh_hit_tables or []),
        last_runtime_simulation_diagnostics=lambda: {
            "status": "ready",
            "targeted_simulation_supported": True,
            "targeted_simulation_used": bool(fresh_hit_tables),
        },
        project_rows=lambda rows: list(rows or ()),
        required_pairs=[_pair(required_source)],
        projection_view_mode="detector",
        projection_view_signature=PROJECTION_SIGNATURE,
        get_targeted_projected_cache=(
            (lambda _digest: targeted_payload) if targeted_payload is not None else None
        ),
        live_cache_inventory={"source_snapshot_count": 1},
        stage_callback=lambda stage, payload: events.append((str(stage), dict(payload))),
    )
    return result, events


def _targeted_payload(rows: Sequence[object]) -> dict[str, object]:
    return {
        "requested_signature": geometry_fit._geometry_fit_cache_jsonable(REQUESTED_SIGNATURE),
        "requested_signature_summary": "sig-summary",
        "projection_view_signature": dict(PROJECTION_SIGNATURE),
        "consumer": "geometry_fit_dataset",
        "projected_rows": [dict(row) for row in rows if isinstance(row, dict)],
        "stored_rows": [dict(row) for row in rows if isinstance(row, dict)],
    }


def test_source_cache_rung_targeted_cache_hit() -> None:
    result, events = _run_rebuild(
        targeted_payload=_targeted_payload([_row("primary")]),
    )

    assert result.rebuild_source == "targeted_projected_cache"
    assert "source_cache_targeted_projected_cache_ready" in [stage for stage, _ in events]


def test_source_cache_rung_targeted_cache_miss_then_live_cache() -> None:
    result, events = _run_rebuild(live_rows=[_row("primary")])
    stages = [stage for stage, _ in events]

    assert result.rebuild_source == "live_runtime_cache"
    assert "source_cache_targeted_projected_cache_miss" in stages
    assert "source_cache_live_runtime_cache_accepted" in stages


def test_source_cache_rung_live_cache_accepted() -> None:
    result, events = _run_rebuild(live_rows=[_row("primary")])
    ready_payload = next(
        payload
        for stage, payload in events
        if stage == "source_cache_live_runtime_cache_validation_ready"
    )

    assert result.rebuild_source == "live_runtime_cache"
    assert ready_payload["status"] == "valid"
    assert ready_payload["reason"] == "ready"


def test_source_cache_rung_live_cache_rejected() -> None:
    result, events = _run_rebuild(live_rows=[_row("primary", detector=False)])
    rejected_payload = next(
        payload for stage, payload in events if stage == "source_cache_live_runtime_cache_rejected"
    )

    assert result.rebuild_source is None
    assert rejected_payload["reason"] == "no_finite_detector_position"


def test_source_cache_rung_memory_cache_miss() -> None:
    _result, events = _run_rebuild()

    assert "source_cache_memory_intersection_cache_miss" in [stage for stage, _ in events]


def test_source_cache_rung_logged_cache_miss() -> None:
    _result, events = _run_rebuild()

    assert "source_cache_logged_intersection_cache_miss" in [stage for stage, _ in events]


def test_source_cache_rung_fresh_simulation_fallback_source_safety() -> None:
    result, events = _run_rebuild(
        required_source=DISORDERED,
        fresh_hit_tables=[object()],
        built_rows=[_row("primary")],
    )
    missing_payload = next(
        payload
        for stage, payload in events
        if stage == "source_cache_fresh_simulation_required_source_missing"
    )

    assert result.rebuild_source is None
    assert result.diagnostics["status"] == "required_source_rows_unavailable"
    assert missing_payload["status"] == "required_source_rows_unavailable"
    assert missing_payload["reason"] == "required_source_rows_unavailable"
