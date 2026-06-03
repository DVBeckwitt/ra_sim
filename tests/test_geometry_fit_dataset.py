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
