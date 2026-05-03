from __future__ import annotations

from ra_sim.gui import geometry_fit
from ra_sim.gui._runtime import runtime_session


DISORDERED = "disordered_phase"


def _row(
    source_label: str = "primary",
    *,
    group_key: object | None = None,
    hkl: tuple[int, int, int] | None = (-1, 1, 4),
    branch: int | None = 0,
    coords: dict[str, object] | None = None,
) -> dict[str, object]:
    group = group_key if group_key is not None else ("q_group", source_label, 1, 4)
    row = {
        "source_label": source_label,
        "phase_label": "Disordered phase" if source_label == DISORDERED else "Primary",
        "structure_role": "disordered" if source_label == DISORDERED else "primary",
        "group_key": group,
        "q_group_key": group,
    }
    if hkl is not None:
        row["hkl"] = hkl
    if branch is not None:
        row["source_branch_index"] = int(branch)
    row.update(
        {"detector_display_x": 300.0, "detector_display_y": 400.0} if coords is None else coords
    )
    return row


def _pair(
    source_label: str = "primary",
    *,
    group_key: object | None = None,
    hkl: tuple[int, int, int] | None = (-1, 1, 4),
    branch: int | None = 0,
) -> dict[str, object]:
    group = group_key if group_key is not None else ("q_group", source_label, 1, 4)
    pair = {
        "pair_id": f"bg0:{source_label}:pair0",
        "source_label": source_label,
        "group_key": group,
        "q_group_key": group,
    }
    if hkl is not None:
        pair["hkl"] = hkl
    if branch is not None:
        pair["source_branch_index"] = int(branch)
    return pair


def _run_rebuild_with_live_rows(rows: list[dict[str, object]], required_pair: dict[str, object]):
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
        build_live_rows=lambda: {
            "rows": rows,
            "cache_metadata": {
                "live_rows_signature_match": True,
                "live_rows_source_counts": {"primary": 1, DISORDERED: 1},
            },
        },
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: ([], None),
        logged_cache_matches_params=lambda _metadata, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: ([], [], [], []),
        simulate_hit_tables=lambda _params, **_kwargs: [],
        last_runtime_simulation_diagnostics=lambda: {"status": "unused"},
        project_rows=lambda source_rows: list(source_rows or ()),
        required_pairs=[required_pair],
        live_cache_inventory={"source_snapshot_count": 1},
        stage_callback=lambda stage, payload: events.append((str(stage), dict(payload))),
    )
    return result, events


def test_live_cache_accepts_ready_primary_rows_with_matching_signature() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [_row("primary")],
        required_pairs=[_pair("primary")],
    )

    assert validation["valid"] is True
    assert validation["status"] == "valid"
    assert validation["reason"] == "ready"


def test_live_cache_accepts_ready_disordered_rows_with_matching_group_hkl() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [_row(DISORDERED)],
        required_pairs=[_pair(DISORDERED)],
    )

    assert validation["valid"] is True
    assert validation["resolved_pairs"][0]["resolution_kind"] == "source_group_hkl"


def test_live_cache_does_not_require_trusted_full_identity_for_q_group_rows() -> None:
    row = _row(DISORDERED)
    pair = _pair(DISORDERED)
    assert "source_reflection_is_full" not in row

    validation = geometry_fit.validate_geometry_fit_live_source_rows([row], required_pairs=[pair])

    assert validation["valid"] is True
    assert validation["validator_canonical_row_count"] == 0
    assert validation["validator_row_schema_valid_count"] == 1


def test_live_cache_source_group_satisfies_pair_without_hkl_when_unique() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [_row(DISORDERED, hkl=None)],
        required_pairs=[_pair(DISORDERED, hkl=None)],
    )

    assert validation["valid"] is True
    assert validation["resolved_pairs"][0]["resolution_kind"] == "source_group"


def test_live_cache_missing_branch_metadata_does_not_reject_unique_group_match() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [_row(DISORDERED, branch=None)],
        required_pairs=[_pair(DISORDERED, branch=1)],
    )

    assert validation["valid"] is True


def test_live_cache_accepts_detector_coordinate_aliases() -> None:
    for coords in (
        {"display_col": 301.0, "display_row": 401.0},
        {"sim_col": 302.0, "sim_row": 402.0},
        {"x": 303.0, "y": 403.0},
    ):
        validation = geometry_fit.validate_geometry_fit_live_source_rows(
            [_row(DISORDERED, coords=coords)],
            required_pairs=[_pair(DISORDERED)],
        )
        assert validation["valid"] is True
        assert validation["validator_finite_detector_rows"] == 1


def test_live_cache_rejects_required_source_missing_with_exact_reason() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [_row("primary")],
        required_pairs=[_pair(DISORDERED)],
    )

    assert validation["valid"] is False
    assert validation["status"] == "invalid"
    assert validation["reason"] == "required_source_missing"


def test_live_cache_rejects_missing_detector_position_with_exact_reason() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [_row(DISORDERED, coords={})],
        required_pairs=[_pair(DISORDERED)],
    )

    assert validation["valid"] is False
    assert validation["reason"] == "no_finite_detector_position"


def test_live_cache_never_reports_invalid_with_reason_ready() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [],
        required_pairs=[_pair(DISORDERED)],
    )

    assert validation["status"] == "invalid"
    assert validation["reason"] != "ready"


def test_live_cache_acceptance_prevents_fresh_simulation_start() -> None:
    result, events = _run_rebuild_with_live_rows(
        [_row("primary"), _row(DISORDERED)],
        _pair(DISORDERED),
    )
    kinds = [kind for kind, _payload in events]
    ready_payload = next(
        payload
        for kind, payload in events
        if kind == "source_cache_live_runtime_cache_validation_ready"
    )

    assert result.rebuild_source == "live_runtime_cache"
    assert ready_payload["status"] == "valid"
    assert ready_payload["reason"] == "ready"
    message = runtime_session._format_source_cache_worker_event_message(
        "source_cache_live_runtime_cache_validation_ready",
        ready_payload,
    )
    assert "status=valid" in message
    assert "reason=ready" in message
    assert "source_cache_live_runtime_cache_accepted" in kinds
    assert "source_cache_targeted_fresh_simulation_start" not in kinds
