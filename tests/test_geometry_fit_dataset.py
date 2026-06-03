from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from ra_sim.gui import geometry_fit_dataset
from ra_sim.gui.geometry_fit_coordinates import finite_float


def _pair_enabled(entry):
    return isinstance(entry, Mapping) and not bool(entry.get("geometry_fit_disabled", False))


def _background_detector_pair_for_frame(entry, frame):
    if frame != "native_detector":
        return None
    if "background_detector_x" not in entry or "background_detector_y" not in entry:
        return None
    return (float(entry["background_detector_x"]), float(entry["background_detector_y"]))


def _entry_source_label(entry):
    return str(entry.get("source_label", "primary"))


def _truth_by_order_key(rows):
    return {
        (int(row["background_index"]), int(row["pair_index"])): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }


def _bindings(*, rows, refresh_pair_entry=None, pick_uses_caked_space=None):
    return SimpleNamespace(
        geometry_manual_pairs_for_index=lambda background_idx: rows,
        geometry_manual_refresh_pair_entry=refresh_pair_entry,
        pick_uses_caked_space=pick_uses_caked_space,
    )


def _collect(
    *,
    rows,
    background_idx=2,
    theta_base=10.0,
    base_fit_params=None,
    manual_fit_requires_caked_space=False,
    refresh_pair_entry=None,
    pick_uses_caked_space=None,
    truth_calls=None,
):
    def _build_truth_pairs(background_idx, saved_rows, *, refresh_pair_entry=None):
        if truth_calls is not None:
            truth_calls.append(
                {
                    "background_idx": background_idx,
                    "saved_rows": [dict(row) for row in saved_rows],
                    "refresh_pair_entry": refresh_pair_entry,
                }
            )
        return [
            {
                "background_index": int(background_idx),
                "pair_index": pair_index,
                "raw_id": row.get("id"),
            }
            for pair_index, row in enumerate(saved_rows)
        ]

    return geometry_fit_dataset.collect_geometry_manual_dataset_inputs(
        background_idx=background_idx,
        theta_base=theta_base,
        base_fit_params=base_fit_params,
        manual_dataset_bindings=_bindings(
            rows=rows,
            refresh_pair_entry=refresh_pair_entry,
            pick_uses_caked_space=pick_uses_caked_space,
        ),
        manual_fit_requires_caked_space=manual_fit_requires_caked_space,
        pair_enabled=_pair_enabled,
        background_detector_pair_for_frame=_background_detector_pair_for_frame,
        entry_source_label=_entry_source_label,
        build_manual_picker_truth_pairs=_build_truth_pairs,
        truth_by_order_key=_truth_by_order_key,
        finite_float=finite_float,
    )


def _stable_group_identity(value):
    if isinstance(value, Mapping):
        return tuple((str(key), _stable_group_identity(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_stable_group_identity(item) for item in value)
    return tuple(value) if isinstance(value, tuple) else value


def _provider_coverage_key(entry):
    if not isinstance(entry, Mapping):
        return None
    hkl = entry.get("normalized_hkl", entry.get("hkl"))
    if not isinstance(hkl, (list, tuple)) or len(hkl) < 3:
        return None
    branch = entry.get("physical_branch_slot", entry.get("source_branch_index"))
    try:
        branch = int(branch)
    except Exception:
        branch = None
    if branch not in {0, 1}:
        branch = None
    return (
        tuple(int(value) for value in hkl[:3]),
        branch,
        _stable_group_identity(entry.get("q_group_key")),
    )


def _provider_coverage_payload(key):
    if key is None:
        return None
    return {
        "hkl": tuple(int(value) for value in key[0]),
        "branch_slot": key[1],
        "branch_index": int(key[1]) if key[1] in {0, 1} else None,
        "q_group_key": key[2],
    }


def _provider_coverage_alias_keys(entry):
    keys = set()
    direct_key = _provider_coverage_key(entry)
    if direct_key is not None:
        keys.add(direct_key)
    if isinstance(entry, Mapping):
        for alias in entry.get("source_coverage_aliases") or ():
            alias_key = _provider_coverage_key(alias)
            if alias_key is not None:
                keys.add(alias_key)
    return keys


def _cache_jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _cache_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_cache_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_cache_jsonable(item) for item in value]
    return value


def _point_list(point):
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        return [float(point[0]), float(point[1])]
    except Exception:
        return None


def _put_simulated_point_fields(entry, point, frame):
    if frame == "caked_2theta_phi":
        entry["simulated_two_theta_deg"] = float(point[0])
        entry["simulated_phi_deg"] = float(point[1])


def _provider_coverage_deps(
    *,
    provider_backed_source_row_for_target=None,
    source_coverage_filter_diagnostics=None,
):
    return geometry_fit_dataset.GeometryProviderCoverageDeps(
        normalize_source_coverage_key=_provider_coverage_key,
        source_coverage_alias_keys=_provider_coverage_alias_keys,
        source_coverage_key_payload=_provider_coverage_payload,
        source_coverage_filter_diagnostics=(
            source_coverage_filter_diagnostics or (lambda _rows: {})
        ),
        cache_jsonable=_cache_jsonable,
        point_list=_point_list,
        normalize_point_frame=lambda frame: str(frame or "unknown"),
        put_simulated_point_fields=_put_simulated_point_fields,
        coerce_nonnegative_index=lambda value: int(value)
        if value is not None and int(value) >= 0
        else None,
        normalized_hkl=lambda value: tuple(int(item) for item in value[:3])
        if isinstance(value, (list, tuple)) and len(value) >= 3
        else None,
        source_branch_index=lambda entry: (
            int(entry["source_branch_index"])
            if isinstance(entry, Mapping)
            and entry.get("source_branch_index") in {0, 1, "0", "1"}
            else None
        ),
        stable_group_identity=_stable_group_identity,
        group_identity=lambda entry: _stable_group_identity(entry.get("q_group_key"))
        if isinstance(entry, Mapping)
        else None,
        group_identity_is_q_group=lambda value: isinstance(value, tuple)
        and len(value) >= 4
        and str(value[0]) == "q_group",
        source_row_reuses_manual_caked_target=lambda _target, _row: False,
        provider_backed_source_row_for_target=(
            provider_backed_source_row_for_target or (lambda **_kwargs: None)
        ),
        pairs_use_caked_fit_space=lambda _entries: False,
        zero_qr_coverage_branch_slot="00l_collapsed",
    )


def _dynamic_trial_deps(
    *,
    simulated_peaks_for_params=None,
    last_simulation_diagnostics=None,
    project_source_rows_for_current_view=None,
):
    return geometry_fit_dataset.GeometryDynamicTrialDeps(
        cache_jsonable=_cache_jsonable,
        normalize_source_coverage_key=_provider_coverage_key,
        source_coverage_alias_keys=_provider_coverage_alias_keys,
        source_coverage_key_payload=_provider_coverage_payload,
        group_identity=lambda entry: _stable_group_identity(entry.get("q_group_key"))
        if isinstance(entry, Mapping)
        else None,
        stable_group_identity=_stable_group_identity,
        group_identity_is_q_group=lambda value: isinstance(value, tuple)
        and len(value) >= 4
        and str(value[0]) == "q_group",
        normalized_hkl=_normalized_hkl,
        source_branch_index=_source_branch_index,
        caked_angle_pair=lambda row, *, x_keys, y_keys: next(
            (
                (float(row[x_key]), float(row[y_key]))
                for x_key, y_key in zip(x_keys, y_keys, strict=True)
                if isinstance(row, Mapping)
                and row.get(x_key) is not None
                and row.get(y_key) is not None
            ),
            None,
        ),
        point_list=_point_list,
        put_simulated_point_fields=_put_simulated_point_fields,
        coerce_nonnegative_index=_coerce_nonnegative_index,
        finite_float=finite_float,
        source_row_reuses_manual_caked_target=lambda _target, _row: False,
        simulated_peaks_for_params=simulated_peaks_for_params,
        last_simulation_diagnostics=last_simulation_diagnostics,
        project_source_rows_for_current_view=(
            project_source_rows_for_current_view or (lambda rows: rows)
        ),
        zero_qr_coverage_branch_slot="00l_collapsed",
    )


def _coerce_nonnegative_index(value):
    try:
        index = int(value)
    except Exception:
        return None
    return index if index >= 0 else None


def _normalized_hkl(value):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return tuple(int(item) for item in value[:3])
    except Exception:
        return None


def _source_branch_index(entry):
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("physical_branch_slot", entry.get("source_branch_index"))
    try:
        branch = int(value)
    except Exception:
        return None
    return branch if branch in {0, 1} else None


def _entry_point(entry, x_key, y_key):
    if not isinstance(entry, Mapping):
        return None
    try:
        point = (float(entry[x_key]), float(entry[y_key]))
    except Exception:
        return None
    if not (math.isfinite(point[0]) and math.isfinite(point[1])):
        return None
    return point


def _candidate_point_for_frame(candidate, *, frame_name):
    if frame_name in {
        "measured_display",
        "refined_sim_display",
        "current_view_display",
    }:
        return _entry_point(candidate, "sim_col", "sim_row") or _entry_point(
            candidate, "display_col", "display_row"
        )
    if frame_name in {"measured_detector", "refined_sim_native"}:
        return _entry_point(candidate, "sim_col_raw", "sim_row_raw")
    if frame_name == "refined_sim_caked":
        return _entry_point(candidate, "caked_x", "caked_y")
    return None


def _source_entry_hkl_matches(entry, candidate):
    entry_hkl = _normalized_hkl(entry.get("hkl") if isinstance(entry, Mapping) else None)
    candidate_hkl = _normalized_hkl(
        candidate.get("hkl") if isinstance(candidate, Mapping) else None
    )
    if candidate_hkl is None:
        return False
    if entry_hkl is None:
        return True
    if candidate_hkl == entry_hkl:
        return True
    entry_group = _stable_group_identity(entry.get("q_group_key"))
    candidate_group = _stable_group_identity(candidate.get("q_group_key"))
    return (
        isinstance(entry_group, tuple)
        and len(entry_group) >= 4
        and str(entry_group[0]) == "q_group"
        and entry_group == candidate_group
    )


def _source_candidate_deps(
    *,
    legacy_candidate_pool=None,
    legacy_candidate_pool_source="source_hkl",
):
    def _compact(entry):
        if not isinstance(entry, Mapping):
            return None
        keys = (
            "id",
            "hkl",
            "source_table_index",
            "source_row_index",
            "source_reflection_index",
            "source_branch_index",
            "source_peak_index",
            "q_group_key",
            "sim_col",
            "sim_row",
        )
        return {key: _cache_jsonable(entry.get(key)) for key in keys if key in entry}

    def _trace_inventory(candidates):
        return [_compact(candidate) for candidate in candidates or () if isinstance(candidate, Mapping)]

    def _legacy_dense_working_entry(entry, raw_saved_entry):
        if not isinstance(entry, Mapping):
            return None
        working = dict(entry)
        source = raw_saved_entry if isinstance(raw_saved_entry, Mapping) else entry
        if source.get("source_reflection_index") is not None:
            working["legacy_source_reflection_index"] = source.get("source_reflection_index")
        if source.get("source_peak_index") is not None:
            working["legacy_source_peak_index"] = source.get("source_peak_index")
        for key in (
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
        ):
            working.pop(key, None)
        return working

    def _apply_override(entry, live_row, *, prefer_caked_display=False):
        merged = dict(entry or {})
        if isinstance(live_row, Mapping):
            merged.update(dict(live_row))
        return merged

    return geometry_fit_dataset.GeometrySourceCandidateDeps(
        coerce_nonnegative_index=_coerce_nonnegative_index,
        trusted_full_reflection_identity=lambda entry: bool(
            isinstance(entry, Mapping)
            and _coerce_nonnegative_index(entry.get("source_reflection_index")) is not None
            and (
                str(entry.get("source_reflection_namespace", "")).lower()
                in {"full", "full_reflection", "miller"}
                or bool(entry.get("source_reflection_is_full", False))
            )
        ),
        source_branch_index=_source_branch_index,
        normalized_hkl=_normalized_hkl,
        source_entry_hkl_matches=_source_entry_hkl_matches,
        entry_source_label=lambda entry: str(entry.get("source_label", "primary"))
        if isinstance(entry, Mapping)
        else "primary",
        stable_group_identity=_stable_group_identity,
        is_zero_qr_00l=lambda entry: bool(
            isinstance(entry, Mapping)
            and (
                (_normalized_hkl(entry.get("hkl")) or (1, 1, 1))[0:2] == (0, 0)
                or (
                    isinstance(entry.get("q_group_key"), tuple)
                    and len(entry["q_group_key"]) >= 3
                    and int(entry["q_group_key"][2]) == 0
                )
            )
        ),
        entry_display_point=lambda entry: _entry_point(entry, "display_col", "display_row"),
        entry_saved_simulated_current_view_point=lambda entry: _entry_point(
            entry, "refined_sim_x", "refined_sim_y"
        ),
        legacy_saved_simulated_detector_hint=lambda entry: _entry_point(
            entry, "refined_sim_native_x", "refined_sim_native_y"
        )
        or _entry_point(entry, "refined_sim_x", "refined_sim_y"),
        candidate_current_view_point=lambda candidate: _candidate_point_for_frame(
            candidate, frame_name="measured_display"
        ),
        candidate_current_view_frame=lambda candidate: "current_view_display"
        if _candidate_point_for_frame(candidate, frame_name="measured_display") is not None
        else None,
        candidate_point_for_frame=_candidate_point_for_frame,
        compact_source_resolution_entry_payload=_compact,
        trace_candidate_inventory=_trace_inventory,
        legacy_dense_working_entry=_legacy_dense_working_entry,
        resolve_source_entry_candidate_pool=lambda entry: (
            [dict(candidate) for candidate in (legacy_candidate_pool or ())],
            legacy_candidate_pool_source,
        ),
        legacy_branch_hint_resolution=lambda entry: (1, "phi_deg", None),
        legacy_geometry_hint=lambda entry, candidates: (
            "measured_display",
            _entry_point(entry, "display_col", "display_row"),
            0.0,
        ),
        canonicalize_live_source_entry=lambda entry: dict(entry)
        if isinstance(entry, Mapping)
        else None,
        is_canonical_live_source_entry=lambda entry: (
            isinstance(entry, Mapping) and _source_branch_index(entry) in {0, 1},
            None
            if isinstance(entry, Mapping) and _source_branch_index(entry) in {0, 1}
            else "missing_branch",
        ),
        apply_refined_simulated_override=_apply_override,
        cache_jsonable=_cache_jsonable,
        background_current_view_frame=lambda entry: "current_view_display"
        if _entry_point(entry, "display_col", "display_row") is not None
        else None,
    )


def test_geometry_fit_dataset_module_keeps_internal_import_boundary() -> None:
    source = Path(geometry_fit_dataset.__file__).read_text(encoding="utf-8")

    for forbidden in (
        "from ra_sim.gui import geometry_fit",
        "import ra_sim.gui.geometry_fit",
        "runtime_session",
        "geometry_fit_worker",
        "fitting.optimization",
        "tkinter",
        "matplotlib",
    ):
        assert forbidden not in source


def test_collect_geometry_manual_dataset_inputs_filters_disabled_pairs() -> None:
    snapshot = _collect(
        rows=[
            {"id": "disabled", "geometry_fit_disabled": True},
            {"id": "enabled", "source_label": "primary"},
        ],
        pick_uses_caked_space=lambda: True,
    )

    assert [entry["id"] for entry in snapshot.raw_selected_entries] == ["enabled"]
    assert [entry["id"] for entry in snapshot.selected_entries] == ["enabled"]
    assert snapshot.use_caked_display is True


def test_collect_geometry_manual_dataset_inputs_raises_without_enabled_pairs() -> None:
    with pytest.raises(RuntimeError, match="background 3 has no saved manual geometry pairs"):
        _collect(rows=[{"id": "disabled", "geometry_fit_disabled": True}])


def test_collect_geometry_manual_dataset_inputs_refreshes_and_preserves_raw_entry() -> None:
    def _refresh_pair_entry(entry):
        return {
            "id": f"{entry['id']}-refreshed",
            "background_detector_x": 11.0,
            "background_detector_y": 12.0,
        }

    snapshot = _collect(
        rows=[{"id": "raw", "background_detector_x": 1.0, "background_detector_y": 2.0}],
        refresh_pair_entry=_refresh_pair_entry,
    )

    selected_input = snapshot.selected_entry_inputs[0]
    assert selected_input["raw_saved_entry"] == {
        "id": "raw",
        "background_detector_x": 1.0,
        "background_detector_y": 2.0,
    }
    assert selected_input["entry"] == {
        "id": "raw-refreshed",
        "background_detector_x": 11.0,
        "background_detector_y": 12.0,
        "background_detector_frame_provenance": "geometry_manual_refresh_pair_entry",
    }


def test_collect_geometry_manual_dataset_inputs_adds_source_label_and_required_payload() -> None:
    snapshot = _collect(rows=[{"id": "pair-1", "source_label": "aux"}])

    assert snapshot.selected_entries == [{"id": "pair-1", "source_label": "aux"}]
    assert snapshot.required_pairs_callback_payload() == [{"id": "pair-1"}]


def test_collect_geometry_manual_dataset_inputs_builds_truth_from_raw_entries() -> None:
    truth_calls: list[dict[str, object]] = []

    def _refresh_pair_entry(entry):
        return {"id": f"{entry['id']}-refreshed"}

    snapshot = _collect(
        rows=[{"id": "raw"}],
        refresh_pair_entry=_refresh_pair_entry,
        truth_calls=truth_calls,
    )

    assert truth_calls == [
        {
            "background_idx": 2,
            "saved_rows": [{"id": "raw"}],
            "refresh_pair_entry": _refresh_pair_entry,
        }
    ]
    assert snapshot.manual_picker_truth_pairs == [
        {"background_index": 2, "pair_index": 0, "raw_id": "raw"}
    ]
    assert snapshot.manual_picker_truth_by_order == {
        (2, 0): {"background_index": 2, "pair_index": 0, "raw_id": "raw"}
    }


def test_collect_geometry_manual_dataset_inputs_sets_fit_params_and_references() -> None:
    snapshot = _collect(
        rows=[{"id": "pair"}],
        theta_base=10.0,
        base_fit_params={
            "theta_offset": 1.5,
            "a": "3.2",
            "c": float("nan"),
            "lambda": 1.54,
        },
    )

    assert snapshot.baseline_fit_params_i["theta_offset"] == pytest.approx(1.5)
    assert snapshot.baseline_fit_params_i["a"] == "3.2"
    assert math.isnan(snapshot.baseline_fit_params_i["c"])
    assert snapshot.baseline_fit_params_i["lambda"] == pytest.approx(1.54)
    assert snapshot.params_i["theta_initial"] == pytest.approx(11.5)
    assert snapshot.theta_offset == pytest.approx(1.5)
    assert snapshot.theta_initial == pytest.approx(11.5)
    assert snapshot.reference_a == pytest.approx(3.2)
    assert snapshot.reference_c is None
    assert snapshot.reference_lambda == pytest.approx(1.54)


def test_collect_geometry_manual_dataset_inputs_ignores_caked_pick_callback_errors() -> None:
    def _pick_uses_caked_space():
        raise RuntimeError("not ready")

    snapshot = _collect(rows=[{"id": "pair"}], pick_uses_caked_space=_pick_uses_caked_space)

    assert snapshot.use_caked_display is False


def test_augment_source_rows_with_provider_coverage_materializes_missing_provider_row() -> None:
    provider_calls: list[dict[str, object]] = []

    def _provider_row_for_target(*, pair_idx, entry, raw_saved_entry):
        provider_calls.append(
            {
                "pair_idx": pair_idx,
                "entry": dict(entry),
                "raw_saved_entry": dict(raw_saved_entry or {}),
            }
        )
        return {
            "row_origin": "manual_picker_saved_source_coverage",
            "provider_backed_live_source_row": True,
            "hkl": tuple(entry["hkl"]),
            "source_branch_index": int(entry["source_branch_index"]),
            "q_group_key": tuple(entry["q_group_key"]),
            "display_col": 12.0,
            "display_row": 13.0,
        }

    rows, diagnostics = geometry_fit_dataset.augment_source_rows_with_provider_coverage(
        [],
        {},
        selected_entry_inputs=[
            {
                "entry": {
                    "hkl": (1, 1, 0),
                    "source_branch_index": 1,
                    "q_group_key": ("q_group", "primary", 1, 10),
                },
                "raw_saved_entry": {"saved": True},
            }
        ],
        selected_entries=[
            {
                "hkl": (1, 1, 0),
                "source_branch_index": 1,
                "q_group_key": ("q_group", "primary", 1, 10),
            }
        ],
        manual_picker_truth_by_order={},
        background_idx=0,
        use_caked_display=False,
        deps=_provider_coverage_deps(
            provider_backed_source_row_for_target=_provider_row_for_target
        ),
    )

    assert provider_calls == [
        {
            "pair_idx": 0,
            "entry": {
                "hkl": (1, 1, 0),
                "source_branch_index": 1,
                "q_group_key": ("q_group", "primary", 1, 10),
            },
            "raw_saved_entry": {"saved": True},
        }
    ]
    assert len(rows) == 1
    assert rows[0]["provider_backed_live_source_row"] is True
    assert diagnostics["source_coverage_materialization"] == {
        "provider_backed_row_count": 1,
        "provider_backed_fresh_row_count": 0,
        "point_missing_count": 0,
        "saved_coordinate_materialization_allowed": True,
        "provider_backed_keys": [
            {
                "hkl": (1, 1, 0),
                "branch_slot": 1,
                "branch_index": 1,
                "q_group_key": ("q_group", "primary", 1, 10),
            }
        ],
        "point_missing": [],
    }


def test_augment_source_rows_with_provider_coverage_promotes_fresh_row_when_provider_required() -> None:
    rows, diagnostics = geometry_fit_dataset.augment_source_rows_with_provider_coverage(
        [
            {
                "hkl": (1, 1, 0),
                "source_branch_index": 1,
                "source_table_index": 4,
                "source_row_index": 8,
                "source_peak_index": 99,
                "q_group_key": ("q_group", "primary", 1, 10),
                "display_col": 21.0,
                "display_row": 22.0,
            }
        ],
        {},
        selected_entry_inputs=[
            {
                "entry": {
                    "hkl": (1, 1, 0),
                    "source_branch_index": 1,
                    "source_table_index": 4,
                    "source_row_index": 8,
                    "source_peak_index": 3,
                    "q_group_key": ("q_group", "primary", 1, 10),
                    "manual_selected_simulated_point": (21.0, 22.0),
                },
                "raw_saved_entry": {},
            }
        ],
        selected_entries=[
            {
                "hkl": (1, 1, 0),
                "source_branch_index": 1,
                "q_group_key": ("q_group", "primary", 1, 10),
            }
        ],
        manual_picker_truth_by_order={},
        background_idx=2,
        use_caked_display=True,
        deps=_provider_coverage_deps(),
        require_provider_backed_rows=True,
    )

    assert len(rows) == 1
    promoted = rows[0]
    assert promoted["provider_backed_live_source_row"] is True
    assert promoted["provider_backed_live_source_row_reason"] == (
        "geometry_fit_dataset_required_source_coverage"
    )
    assert promoted["trial_source_peak_index"] == 99
    assert promoted["source_peak_index"] == 3
    assert promoted["source_kind"] == "sim_visual_caked_deg"
    assert promoted["actual_source"] == "sim_visual_caked_deg"
    assert promoted["projection_frame"] == "caked_display"
    assert promoted["coordinate_provenance"] == "trial_geometry_projection"
    assert promoted["is_dynamic_trial_row"] is True
    assert promoted["background_index"] == 2
    assert promoted["overlay_match_index"] == 0
    assert diagnostics["source_coverage_materialization"][
        "provider_backed_fresh_row_count"
    ] == 1


def test_augment_source_rows_with_provider_coverage_records_missing_dynamic_trial_when_materialization_disabled() -> None:
    provider_calls: list[dict[str, object]] = []

    rows, diagnostics = geometry_fit_dataset.augment_source_rows_with_provider_coverage(
        [],
        {},
        selected_entry_inputs=[
            {
                "entry": {
                    "hkl": (1, 1, 0),
                    "source_branch_index": 0,
                    "q_group_key": ("q_group", "primary", 1, 10),
                },
                "raw_saved_entry": {},
            }
        ],
        selected_entries=[],
        manual_picker_truth_by_order={},
        background_idx=0,
        use_caked_display=False,
        deps=_provider_coverage_deps(
            provider_backed_source_row_for_target=lambda **kwargs: provider_calls.append(
                dict(kwargs)
            )
        ),
        allow_saved_coordinate_materialization=False,
    )

    assert rows == []
    assert provider_calls == []
    assert diagnostics["source_coverage_materialization"]["point_missing"] == [
        {
            "pair_index": 0,
            "target_key": {
                "hkl": (1, 1, 0),
                "branch_slot": 0,
                "branch_index": 0,
                "q_group_key": ("q_group", "primary", 1, 10),
            },
            "reason": "missing_dynamic_trial_source_row",
        }
    ]
    assert diagnostics["missing_dynamic_trial_source_row_count"] == 1


def test_trial_row_is_dynamic_rejects_saved_source_coverage() -> None:
    dynamic_row = {
        "source_kind": "sim_visual_caked_deg",
        "actual_source": "sim_visual_caked_deg",
        "projection_frame": "caked_display",
        "coordinate_provenance": "trial_geometry_projection",
        "is_dynamic_trial_row": True,
    }
    saved_row = {
        **dynamic_row,
        "row_origin": "manual_picker_saved_source_coverage",
    }

    assert geometry_fit_dataset._trial_row_is_dynamic(dynamic_row) is True
    assert geometry_fit_dataset._trial_row_is_dynamic(saved_row) is False


def test_trial_stage_summary_counts_dynamic_stale_and_missing_required_pairs() -> None:
    selected_inputs = [
        {
            "entry": {
                "pair_id": "dyn",
                "hkl": (1, 1, 0),
                "source_branch_index": 0,
                "q_group_key": ("q_group", "primary", 1, 10),
            }
        },
        {
            "entry": {
                "pair_id": "stale",
                "hkl": (1, 1, 0),
                "source_branch_index": 1,
                "q_group_key": ("q_group", "primary", 1, 10),
            }
        },
        {
            "entry": {
                "pair_id": "missing",
                "hkl": (2, 0, 0),
                "source_branch_index": 0,
                "q_group_key": ("q_group", "primary", 2, 20),
            }
        },
    ]
    rows = [
        {
            "hkl": (1, 1, 0),
            "source_branch_index": 0,
            "q_group_key": ("q_group", "primary", 1, 10),
            "actual_source": "sim_visual_caked_deg",
            "source_kind": "sim_visual_caked_deg",
            "projection_frame": "caked_display",
            "coordinate_provenance": "trial_geometry_projection",
            "is_dynamic_trial_row": True,
        },
        {
            "hkl": (1, 1, 0),
            "source_branch_index": 1,
            "q_group_key": ("q_group", "primary", 1, 10),
            "actual_source": "sim_visual_caked_deg",
            "source_kind": "sim_visual_caked_deg",
            "projection_frame": "caked_display",
            "coordinate_provenance": "stale_saved_projection",
        },
    ]

    summary = geometry_fit_dataset._trial_stage_summary(
        "after_mark",
        rows,
        selected_entry_inputs=selected_inputs,
        background_idx=4,
        deps=_dynamic_trial_deps(),
    )

    assert summary["row_count"] == 2
    assert summary["required_pair_count"] == 3
    assert summary["dynamic_required_pair_count"] == 1
    assert summary["stale_required_pair_count"] == 1
    assert summary["missing_required_pair_count"] == 1
    assert [row["drop_reason"] for row in summary["per_required_pair"]] == [
        None,
        "stale_qr_coordinate_provenance",
        "missing_dynamic_trial_source_row",
    ]


def test_mark_dynamic_trial_source_rows_sets_caked_dynamic_metadata() -> None:
    rows = geometry_fit_dataset._mark_dynamic_trial_source_rows(
        [
            {
                "hkl": (1, 1, 0),
                "caked_x": 14.0,
                "caked_y": -22.0,
            }
        ],
        deps=_dynamic_trial_deps(),
    )

    assert rows == [
        {
            "hkl": (1, 1, 0),
            "caked_x": 14.0,
            "caked_y": -22.0,
            "source_kind": "sim_visual_caked_deg",
            "actual_source": "sim_visual_caked_deg",
            "expected_source": "sim_visual_caked_deg",
            "projection_frame": "caked_display",
            "coordinate_provenance": "trial_geometry_projection",
            "is_dynamic_trial_row": True,
            "simulated_two_theta_deg": 14.0,
            "simulated_phi_deg": -22.0,
            "sim_visual_caked_deg": (14.0, -22.0),
            "sim_visual_deg": (14.0, -22.0),
            "sim_caked": (14.0, -22.0),
        }
    ]


def test_trial_candidate_sort_key_prefers_exact_target_identity() -> None:
    target = {
        "hkl": (1, 1, 0),
        "source_reflection_index": 3,
        "source_table_index": 1,
        "source_row_index": 9,
        "source_peak_index": 0,
        "manual_selected_simulated_point": (30.0, 40.0),
    }
    target_key = (
        (1, 1, 0),
        0,
        ("q_group", "primary", 1, 10),
    )
    exact_row = {
        **target,
        "q_group_key": ("q_group", "primary", 1, 10),
        "source_branch_index": 0,
        "sim_visual_caked_deg": (100.0, 100.0),
    }
    nearby_row = {
        "hkl": (1, 1, 0),
        "q_group_key": ("q_group", "primary", 1, 10),
        "source_branch_index": 0,
        "sim_visual_caked_deg": (30.0, 40.0),
    }

    exact_key = geometry_fit_dataset._trial_candidate_sort_key(
        target,
        target_key,
        exact_row,
        4,
        deps=_dynamic_trial_deps(),
    )
    nearby_key = geometry_fit_dataset._trial_candidate_sort_key(
        target,
        target_key,
        nearby_row,
        1,
        deps=_dynamic_trial_deps(),
    )

    assert exact_key < nearby_key


def test_build_dynamic_trial_completion_row_materializes_required_pair() -> None:
    row, reason = geometry_fit_dataset._build_dynamic_trial_completion_row(
        {
            "pair_id": "required",
            "hkl": (1, 1, 0),
            "source_branch_index": 0,
            "q_group_key": ("q_group", "primary", 1, 10),
            "source_table_index": 8,
        },
        ((1, 1, 0), 0, ("q_group", "primary", 1, 10)),
        active_params={"a": 3.0, "c": 5.0, "lambda": 1.0},
        candidate_rows=[
            {
                "hkl": (1, 1, 0),
                "source_branch_index": 1,
                "q_group_key": ("q_group", "primary", 1, 10),
                "actual_source": "sim_visual_caked_deg",
                "source_kind": "sim_visual_caked_deg",
                "projection_frame": "caked_display",
                "coordinate_provenance": "trial_geometry_projection",
                "is_dynamic_trial_row": True,
                "caked_x": 30.0,
                "caked_y": 12.5,
            }
        ],
        background_idx=7,
        deps=_dynamic_trial_deps(),
    )

    assert reason == "mirrored_dynamic_q_group_branch"
    assert row is not None
    assert row["background_index"] == 7
    assert row["pair_id"] == "required"
    assert row["dynamic_completion_reason"] == "mirrored_dynamic_q_group_branch"
    assert row["physical_branch_slot"] == 0
    assert row["source_branch_index"] == 0
    assert row["source_table_index"] == 8
    assert row["source_coverage_aliases"] == [
        {
            "hkl": (1, 1, 0),
            "branch_slot": 0,
            "branch_index": 0,
            "q_group_key": ("q_group", "primary", 1, 10),
        }
    ]
    assert row["phi_deg"] == pytest.approx(-12.5)


def test_build_dynamic_trial_completion_row_reports_missing_caked_point() -> None:
    row, reason = geometry_fit_dataset._build_dynamic_trial_completion_row(
        {
            "hkl": (1, 1, 0),
            "source_branch_index": 0,
            "q_group_key": ("q_group", "primary", 1, 10),
        },
        ((1, 1, 0), 0, ("q_group", "primary", 1, 10)),
        active_params={"a": 3.0, "c": 5.0, "lambda": 1.0},
        candidate_rows=[],
        background_idx=0,
        deps=_dynamic_trial_deps(),
    )

    assert row is None
    assert reason == "missing_dynamic_sibling_branch"


def test_supplement_dynamic_trial_rows_from_candidate_pool_adds_missing_required_pair() -> None:
    selected_inputs = [
        {
            "entry": {
                "pair_id": "required",
                "hkl": (1, 1, 0),
                "source_branch_index": 0,
                "q_group_key": ("q_group", "primary", 1, 10),
                "manual_selected_simulated_point": (12.0, 4.0),
            }
        }
    ]

    supplemental, diagnostics = (
        geometry_fit_dataset._supplement_dynamic_trial_rows_from_candidate_pool(
            [],
            active_params={"a": 3.0, "c": 5.0, "lambda": 1.0},
            selected_entry_inputs=selected_inputs,
            background_idx=2,
            deps=_dynamic_trial_deps(
                simulated_peaks_for_params=lambda _params, prefer_cache=False: [
                    {
                        "hkl": (1, 1, 0),
                        "source_branch_index": 0,
                        "q_group_key": ("q_group", "primary", 1, 10),
                        "caked_x": 12.0,
                        "caked_y": 4.0,
                    }
                ]
            ),
        )
    )

    assert len(supplemental) == 1
    assert supplemental[0]["pair_id"] == "required"
    assert supplemental[0]["background_index"] == 2
    assert supplemental[0]["fit_qr_branch_key"] == {
        "q_group_key": ["q_group", "primary", 1, 10],
        "hkl": [1, 1, 0],
        "physical_branch_slot": 0,
        "source_branch_index": 0,
        "source_peak_index": None,
    }
    assert diagnostics["attempted"] is True
    assert diagnostics["candidate_row_count"] == 1
    assert diagnostics["supplemental_row_count"] == 1
    assert diagnostics["missing_before_count"] == 1
    assert diagnostics["missing_after_count"] == 0


def test_supplement_dynamic_trial_rows_from_candidate_pool_sets_fit_qr_branch_key() -> None:
    supplemental, diagnostics = (
        geometry_fit_dataset._supplement_dynamic_trial_rows_from_candidate_pool(
            [],
            active_params={"a": 3.0, "c": 5.0, "lambda": 1.0},
            selected_entry_inputs=[
                {
                    "entry": {
                        "pair_id": "required",
                        "hkl": (1, 1, 0),
                        "source_branch_index": 0,
                        "q_group_key": ("q_group", "primary", 1, 10),
                    }
                }
            ],
            background_idx=3,
            deps=_dynamic_trial_deps(
                simulated_peaks_for_params=lambda _params, prefer_cache=False: [
                    {
                        "hkl": (1, 1, 0),
                        "source_branch_index": 0,
                        "source_peak_index": 2,
                        "q_group_key": ("q_group", "primary", 1, 10),
                        "caked_x": 25.0,
                        "caked_y": 6.0,
                    }
                ]
            ),
        )
    )

    assert len(supplemental) == 1
    assert supplemental[0]["fit_qr_branch_key"] == {
        "q_group_key": ["q_group", "primary", 1, 10],
        "hkl": [1, 1, 0],
        "physical_branch_slot": 0,
        "source_branch_index": 0,
        "source_peak_index": 2,
    }
    assert diagnostics["candidate_row_count"] == 1
    assert diagnostics["missing_after_count"] == 0


def test_augment_source_rows_with_provider_coverage_records_point_missing_when_provider_row_unavailable() -> None:
    rows, diagnostics = geometry_fit_dataset.augment_source_rows_with_provider_coverage(
        [],
        {},
        selected_entry_inputs=[
            {
                "entry": {
                    "hkl": (1, 1, 0),
                    "source_branch_index": 0,
                    "q_group_key": ("q_group", "primary", 1, 10),
                },
                "raw_saved_entry": {},
            }
        ],
        selected_entries=[],
        manual_picker_truth_by_order={},
        background_idx=0,
        use_caked_display=False,
        deps=_provider_coverage_deps(),
    )

    assert rows == []
    assert diagnostics["source_coverage_materialization"]["point_missing"] == [
        {
            "pair_index": 0,
            "target_key": {
                "hkl": (1, 1, 0),
                "branch_slot": 0,
                "branch_index": 0,
                "q_group_key": ("q_group", "primary", 1, 10),
            },
            "reason": "coverage_source_present_point_missing",
        }
    ]
    assert diagnostics["coverage_source_present_point_missing_count"] == 1


def test_augment_source_rows_with_provider_coverage_updates_targeted_performance_gate() -> None:
    coverage_diagnostics = {
        "candidate_rows_after_hkl_filter": 2,
        "missing_required_branch_group_keys": [{"hkl": (1, 1, 0)}],
    }

    rows, diagnostics = geometry_fit_dataset.augment_source_rows_with_provider_coverage(
        [
            {
                "hkl": (1, 1, 0),
                "source_branch_index": 0,
                "q_group_key": ("q_group", "primary", 1, 10),
            }
        ],
        {"targeted_performance_gate": {"existing": True}},
        selected_entry_inputs=[],
        selected_entries=[],
        manual_picker_truth_by_order={},
        background_idx=0,
        use_caked_display=False,
        deps=_provider_coverage_deps(
            source_coverage_filter_diagnostics=lambda _rows: coverage_diagnostics
        ),
    )

    assert len(rows) == 1
    assert diagnostics["candidate_rows_after_hkl_filter"] == 2
    assert diagnostics["missing_required_branch_group_keys"] == [{"hkl": (1, 1, 0)}]
    assert diagnostics["targeted_performance_gate"] == {
        "existing": True,
        "candidate_rows_after_hkl_filter": 2,
        "missing_required_branch_group_keys": [{"hkl": (1, 1, 0)}],
    }


def test_source_row_key_preserves_source_locator_identity() -> None:
    assert geometry_fit_dataset._source_row_key(
        {
            "source_table_index": "4",
            "source_row_index": "8",
            "source_peak_index": 99,
        }
    ) == (4, 8)
    assert geometry_fit_dataset._source_row_key({"source_table_index": 4}) is None


def test_source_reflection_row_key_preserves_hkl_and_branch_identity() -> None:
    deps = _source_candidate_deps()

    assert geometry_fit_dataset._source_reflection_row_key(
        {
            "source_reflection_index": "7",
            "source_reflection_namespace": "full",
            "source_row_index": "2",
            "hkl": (1, 0, 0),
            "source_branch_index": 1,
        },
        deps=deps,
    ) == (7, 2)
    assert (
        geometry_fit_dataset._source_reflection_row_key(
            {
                "source_reflection_index": "7",
                "source_reflection_namespace": "candidate_pool",
                "source_row_index": "2",
            },
            deps=deps,
        )
        is None
    )


def test_source_locator_identity_match_accepts_exact_payload_match() -> None:
    deps = _source_candidate_deps()
    saved = {
        "source_reflection_index": 7,
        "source_reflection_namespace": "full",
        "source_reflection_is_full": True,
        "source_table_index": 3,
        "source_row_index": 2,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    candidate = {**saved, "hkl": (1, 0, 0), "sim_col": 10.0, "sim_row": 20.0}

    assert geometry_fit_dataset._source_locator_identity_match(
        saved,
        candidate,
        deps=deps,
    )


def test_source_locator_identity_match_rejects_mismatched_branch_or_source() -> None:
    deps = _source_candidate_deps()
    saved = {
        "source_reflection_index": 7,
        "source_table_index": 3,
        "source_row_index": 2,
        "source_branch_index": 1,
    }

    assert not geometry_fit_dataset._source_locator_identity_match(
        saved,
        {**saved, "source_branch_index": 0},
        deps=deps,
    )
    assert not geometry_fit_dataset._source_locator_identity_match(
        saved,
        {**saved, "source_table_index": 4},
        deps=deps,
    )


def test_filter_hkl_candidates_keeps_matching_hkl_only() -> None:
    deps = _source_candidate_deps()

    filtered, missing, mismatched = geometry_fit_dataset._filter_hkl_candidates(
        {"hkl": (1, 1, 0)},
        [
            {"id": "match", "hkl": (1, 1, 0)},
            {"id": "missing"},
            {"id": "mismatch", "hkl": (2, 0, 0)},
        ],
        deps=deps,
    )

    assert [candidate["id"] for candidate in filtered] == ["match"]
    assert [candidate["id"] for candidate in missing] == ["missing"]
    assert [candidate["id"] for candidate in mismatched] == ["mismatch"]


def test_filter_group_candidates_keeps_matching_group_identity_only() -> None:
    deps = _source_candidate_deps()

    filtered = geometry_fit_dataset._filter_group_candidates(
        {"source_label": "primary", "q_group_key": ("q_group", "primary", 2, 10)},
        [
            {
                "id": "match",
                "source_label": "primary",
                "q_group_key": ("q_group", "primary", 2, 10),
            },
            {
                "id": "wrong-group",
                "source_label": "primary",
                "q_group_key": ("q_group", "primary", 3, 10),
            },
            {
                "id": "wrong-source",
                "source_label": "aux",
                "q_group_key": ("q_group", "primary", 2, 10),
            },
        ],
        deps=deps,
    )

    assert [candidate["id"] for candidate in filtered] == ["match"]


def test_filter_source_branch_candidates_keeps_matching_branch_only() -> None:
    deps = _source_candidate_deps()

    filtered = geometry_fit_dataset._filter_source_branch_candidates(
        {"hkl": (1, 1, 0), "source_branch_index": 1},
        [
            {"id": "branch-0", "source_branch_index": 0},
            {"id": "branch-1", "source_branch_index": 1},
        ],
        deps=deps,
    )

    assert [candidate["id"] for candidate in filtered] == ["branch-1"]


def test_select_source_candidate_prefers_exact_locator_match() -> None:
    deps = _source_candidate_deps()
    saved = {
        "hkl": (1, 1, 0),
        "source_label": "primary",
        "q_group_key": ("q_group", "primary", 2, 10),
        "source_table_index": 3,
        "source_row_index": 7,
        "source_branch_index": 1,
        "display_col": 10.0,
        "display_row": 20.0,
    }

    selection = geometry_fit_dataset._select_source_candidate(
        saved,
        [
            {
                "id": "sorts-first",
                "hkl": (1, 1, 0),
                "source_label": "primary",
                "q_group_key": ("q_group", "primary", 2, 10),
                "source_table_index": 2,
                "source_row_index": 6,
                "source_branch_index": 1,
                "sim_col": 10.0,
                "sim_row": 20.0,
            },
            {
                "id": "identity-match",
                "hkl": (1, 1, 0),
                "source_label": "primary",
                "q_group_key": ("q_group", "primary", 2, 10),
                "source_table_index": 3,
                "source_row_index": 7,
                "source_branch_index": 1,
                "sim_col": 10.0,
                "sim_row": 20.0,
            },
        ],
        deps=deps,
        saved_identity_entry=saved,
        tie_tolerance=0.0,
    )

    assert selection["selected"]["id"] == "identity-match"
    assert selection["selection_tie_breaker"] == "saved_source_identity"


def test_select_source_candidate_rejects_ambiguous_equal_candidates() -> None:
    deps = _source_candidate_deps()
    entry = {
        "hkl": (1, 1, 0),
        "source_label": "primary",
        "q_group_key": ("q_group", "primary", 2, 10),
        "source_branch_index": 1,
        "display_col": 10.0,
        "display_row": 20.0,
    }

    selection = geometry_fit_dataset._select_source_candidate(
        entry,
        [
            {
                "id": "a",
                "hkl": (1, 1, 0),
                "source_label": "primary",
                "q_group_key": ("q_group", "primary", 2, 10),
                "source_table_index": 2,
                "source_row_index": 6,
                "source_branch_index": 1,
                "sim_col": 10.0,
                "sim_row": 20.0,
            },
            {
                "id": "b",
                "hkl": (1, 1, 0),
                "source_label": "primary",
                "q_group_key": ("q_group", "primary", 2, 10),
                "source_table_index": 3,
                "source_row_index": 7,
                "source_branch_index": 1,
                "sim_col": 10.0,
                "sim_row": 20.0,
            },
        ],
        deps=deps,
        tie_tolerance=0.0,
        fail_on_ambiguous_tie=True,
    )

    assert selection["selected"] is None
    assert selection["failure_reason"] == "ambiguous_geometry_tie"
    assert len(selection["tied_candidates"]) == 2


def test_resolve_legacy_dense_source_entry_preserves_existing_dense_fallback() -> None:
    candidate_pool = [
        {
            "id": "wrong-branch",
            "hkl": (1, 1, 0),
            "source_branch_index": 0,
            "source_table_index": 5,
            "source_row_index": 0,
            "source_reflection_index": 5,
            "source_reflection_namespace": "full",
            "source_reflection_is_full": True,
            "sim_col": 10.0,
            "sim_row": 20.0,
        },
        {
            "id": "selected",
            "hkl": (1, 1, 0),
            "source_branch_index": 1,
            "source_table_index": 5,
            "source_row_index": 0,
            "source_reflection_index": 5,
            "source_reflection_namespace": "full",
            "source_reflection_is_full": True,
            "sim_col": 10.0,
            "sim_row": 20.0,
        },
    ]
    deps = _source_candidate_deps(legacy_candidate_pool=candidate_pool)
    entry = {
        "hkl": (1, 1, 0),
        "source_table_index": 5,
        "source_row_index": 0,
        "source_peak_index": 5,
        "display_col": 10.0,
        "display_row": 20.0,
        "phi_deg": 45.0,
    }

    resolved, kind, diagnostics = geometry_fit_dataset._resolve_legacy_dense_source_entry(
        entry,
        raw_saved_entry=dict(entry),
        deps=deps,
    )

    assert resolved["id"] == "selected"
    assert resolved["source_branch_index"] == 1
    assert kind == "legacy_dense_hkl_rebind"
    assert "legacy_failure_reason" not in diagnostics
    assert diagnostics["legacy_candidate_count_after_branch"] == 1
