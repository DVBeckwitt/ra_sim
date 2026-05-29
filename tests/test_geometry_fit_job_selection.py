from __future__ import annotations

import numpy as np

from ra_sim.gui._runtime.geometry_fit_job import (
    resolve_geometry_fit_selection,
    snapshot_geometry_fit_background_inputs,
)


def _selection_snapshot(**overrides):
    kwargs = {
        "params": {"theta_initial": 2.0},
        "var_names": [],
        "current_background_index": 0,
        "theta_initial_value": 1.0,
        "total_background_count": 2,
        "geometry_manual_pairs_for_index": lambda _idx: [{"pair": 1}],
        "apply_geometry_fit_background_selection": lambda **_kwargs: True,
        "current_geometry_fit_background_indices": lambda **_kwargs: [0],
        "geometry_fit_uses_shared_theta_offset": lambda _indices: False,
        "apply_background_theta_metadata": lambda **_kwargs: True,
        "current_background_theta_values": lambda **_kwargs: [],
        "current_geometry_theta_offset": lambda **_kwargs: 0.0,
        "effective_geometry_fit_background_indices": lambda requested, **_kwargs: list(
            requested
        ),
    }
    kwargs.update(overrides)
    return resolve_geometry_fit_selection(**kwargs)


def test_resolve_geometry_fit_selection_keeps_detector_current_background_primary() -> None:
    snapshot = _selection_snapshot()

    assert snapshot.preserve_live_theta is True
    assert snapshot.selected_background_indices == [0]
    assert snapshot.required_indices == [0]
    assert snapshot.primary_index == 0
    assert snapshot.fit_params_snapshot["theta_initial"] == 2.0
    assert snapshot.fit_params_snapshot["theta_offset"] == 0.0
    assert snapshot.background_theta_values == [2.0]


def test_resolve_geometry_fit_selection_applies_shared_theta_metadata() -> None:
    snapshot = _selection_snapshot(
        var_names=["gamma"],
        current_background_index=1,
        current_geometry_fit_background_indices=lambda **_kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda _indices: True,
        current_background_theta_values=lambda **_kwargs: [10.0, 20.0],
        current_geometry_theta_offset=lambda **_kwargs: 3.0,
    )

    assert snapshot.uses_shared_theta is True
    assert snapshot.joint_background_mode is True
    assert snapshot.build_all_selected_backgrounds is True
    assert snapshot.required_indices == [0, 1]
    assert snapshot.primary_index == 1
    assert snapshot.theta_offset == 3.0
    assert snapshot.fit_params_snapshot["theta_initial"] == 20.0
    assert snapshot.fit_params_snapshot["theta_offset"] == 3.0


def test_resolve_geometry_fit_selection_fails_closed_on_selection_error() -> None:
    def _raise_selection(**_kwargs):
        raise ValueError("bad selection")

    snapshot = _selection_snapshot(
        current_geometry_fit_background_indices=_raise_selection,
        geometry_fit_uses_shared_theta_offset=lambda _indices: True,
        current_background_theta_values=lambda **_kwargs: [10.0, 20.0],
        current_geometry_theta_offset=lambda **_kwargs: 3.0,
    )

    assert snapshot.selection_error == "bad selection"
    assert snapshot.required_indices == []
    assert snapshot.uses_shared_theta is False


def test_resolve_geometry_fit_selection_handles_unapplied_selection() -> None:
    def _unexpected_selection_read(**_kwargs):
        raise AssertionError("selection indices should not be read")

    snapshot = _selection_snapshot(
        current_background_index=1,
        total_background_count=3,
        apply_geometry_fit_background_selection=lambda **_kwargs: False,
        current_geometry_fit_background_indices=_unexpected_selection_read,
    )

    assert snapshot.selection_applied is False
    assert snapshot.selection_error is None
    assert snapshot.selected_background_indices == []
    assert snapshot.required_indices == []
    assert snapshot.primary_index == 1
    assert snapshot.fit_params_snapshot["theta_initial"] == 2.0
    assert snapshot.fit_params_snapshot["theta_offset"] == 0.0


def test_resolve_geometry_fit_selection_records_theta_metadata_not_applied() -> None:
    def _unexpected_theta_values(**_kwargs):
        raise AssertionError("theta values should not be read")

    snapshot = _selection_snapshot(
        var_names=["theta_initial"],
        current_background_index=1,
        current_geometry_fit_background_indices=lambda **_kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda _indices: True,
        apply_background_theta_metadata=lambda **_kwargs: False,
        current_background_theta_values=_unexpected_theta_values,
    )

    assert snapshot.theta_metadata_applied is False
    assert snapshot.background_theta_error is None
    assert snapshot.background_theta_values == []
    assert snapshot.theta_offset == 0.0
    assert snapshot.required_indices == [0, 1]
    assert snapshot.fit_params_snapshot["theta_initial"] == 2.0
    assert snapshot.fit_params_snapshot["theta_offset"] == 0.0


def test_resolve_geometry_fit_selection_records_background_theta_error() -> None:
    def _raise_theta_values(**_kwargs):
        raise RuntimeError("theta values unavailable")

    snapshot = _selection_snapshot(
        var_names=["theta_offset"],
        current_geometry_fit_background_indices=lambda **_kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda _indices: True,
        current_background_theta_values=_raise_theta_values,
        current_geometry_theta_offset=lambda **_kwargs: 3.0,
    )

    assert snapshot.theta_metadata_applied is True
    assert snapshot.background_theta_error == "theta values unavailable"
    assert snapshot.background_theta_values == []
    assert snapshot.theta_offset == 0.0
    assert snapshot.required_indices == [0, 1]
    assert snapshot.fit_params_snapshot["theta_initial"] == 2.0
    assert snapshot.fit_params_snapshot["theta_offset"] == 0.0


def test_resolve_geometry_fit_selection_records_skipped_empty_backgrounds() -> None:
    snapshot = _selection_snapshot(
        total_background_count=3,
        geometry_manual_pairs_for_index=lambda idx: [] if idx == 1 else [{"pair": idx}],
        current_geometry_fit_background_indices=lambda **_kwargs: [0, 1, 2],
        effective_geometry_fit_background_indices=lambda requested, **_kwargs: [
            idx for idx in requested if idx != 1
        ],
    )

    assert snapshot.selection_applied is True
    assert snapshot.selection_error is None
    assert snapshot.selected_background_indices == [0, 2]
    assert snapshot.skipped_empty_indices == {1}
    assert snapshot.required_indices == [0]


def test_snapshot_geometry_fit_background_inputs_copies_mutable_payloads() -> None:
    native_background = np.asarray([[1.0]], dtype=float)
    display_background = np.asarray([[2.0]], dtype=float)
    manual_pair = {"pair": 1, "label": "manual"}
    source_row_snapshots = {0: {"rows": [{"source": "snapshot"}]}}
    fit_params = {"theta_initial": 2.0, "theta_offset": 0.5}

    snapshot = snapshot_geometry_fit_background_inputs(
        required_indices=[0],
        selected_background_indices=[0],
        skipped_empty_indices=set(),
        current_background_index=0,
        theta_initial_value=1.0,
        fit_params_snapshot=fit_params,
        background_theta_values=[2.0],
        uses_shared_theta=False,
        joint_background_mode=False,
        build_all_selected_backgrounds=False,
        primary_index=0,
        osc_files=["bg0.osc"],
        load_background_by_index=lambda _idx: (native_background, display_background),
        geometry_manual_pairs_for_index=lambda _idx: [manual_pair],
        source_row_snapshots=source_row_snapshots,
        background_label_for_index=lambda _idx: "background 1",
        source_snapshot_signature_for_background=lambda idx, params: (
            "sig",
            idx,
            params["theta_initial"],
        ),
        signature_summary=lambda signature: repr(signature),
    )

    native_background[0, 0] = 99.0
    display_background[0, 0] = 88.0
    manual_pair["pair"] = 99
    source_row_snapshots[0]["rows"][0]["source"] = "mutated"
    fit_params["theta_initial"] = 99.0

    assert snapshot.background_images[0]["native"][0, 0] == 1.0
    assert snapshot.background_images[0]["display"][0, 0] == 2.0
    assert snapshot.manual_pairs_by_background[0][0]["pair"] == 1
    assert snapshot.source_snapshots[0]["rows"][0]["source"] == "snapshot"
    assert snapshot.requested_signature_summaries[0] == "('sig', 0, 2.5)"
    assert snapshot.theta_base_by_background[0] == 2.0
    assert snapshot.theta_initial_by_background[0] == 2.5
