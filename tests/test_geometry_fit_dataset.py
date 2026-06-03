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
