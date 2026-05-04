from __future__ import annotations

import numpy as np

from ra_sim.fitting import optimization
from ra_sim.fitting.geometry_objective_cache import (
    GeometryObjectiveSignature,
    center_remap_residual_shape_and_order_match,
    geometry_objective_cache_decision,
    geometry_objective_cache_trace_payload,
)


def _signature(**overrides: tuple[object, ...]) -> GeometryObjectiveSignature:
    parts: dict[str, tuple[object, ...]] = {
        "physics_sig": ("physics-a",),
        "detector_center_sig": (100.0, 101.0),
        "dataset_sig": ("dataset-a",),
        "point_provider_sig": ("provider-a",),
        "qr_branch_identity_sig": ("qr-branch-a",),
        "source_row_identity_sig": ("source-row-a",),
        "manual_selection_sig": ("manual-a",),
        "refined_peak_sig": ("refined-a",),
        "objective_mode_sig": ("dynamic-refined",),
        "active_fit_parameter_sig": ("center_x", "center_y"),
    }
    parts.update(overrides)
    return GeometryObjectiveSignature(**parts)


def _decision_for_changed_field(field: str, value: tuple[object, ...]):
    previous = _signature()
    current = _signature(**{field: value})
    return geometry_objective_cache_decision(
        previous,
        current,
        exact_center_remap_cache_available=True,
    )


def test_geometry_objective_center_only_reuses_cache_when_all_identity_signatures_match():
    previous = _signature()
    current = _signature(detector_center_sig=(102.0, 101.0))

    decision = geometry_objective_cache_decision(
        previous,
        current,
        exact_center_remap_cache_available=True,
    )

    assert decision.can_reuse is True
    assert decision.mode == "center_remap"
    assert decision.reject_reason is None
    assert decision.changed_fields == ("detector_center_sig",)


def test_geometry_objective_cache_not_reused_across_qr_branch_identity_change():
    decision = _decision_for_changed_field("qr_branch_identity_sig", ("qr-branch-b",))

    assert decision.can_reuse is False
    assert decision.mode == "full_simulation"
    assert decision.reject_reason == "qr_branch_identity_changed"


def test_geometry_objective_cache_not_reused_across_source_row_identity_change():
    decision = _decision_for_changed_field("source_row_identity_sig", ("source-row-b",))

    assert decision.can_reuse is False
    assert decision.reject_reason == "source_row_identity_changed"


def test_geometry_objective_cache_not_reused_across_manual_selection_change():
    decision = _decision_for_changed_field("manual_selection_sig", ("manual-b",))

    assert decision.can_reuse is False
    assert decision.reject_reason == "manual_selection_changed"


def test_geometry_objective_cache_not_reused_across_refined_peak_change():
    decision = _decision_for_changed_field("refined_peak_sig", ("refined-b",))

    assert decision.can_reuse is False
    assert decision.reject_reason == "refined_peak_changed"


def test_geometry_objective_cache_not_reused_across_point_provider_signature_change():
    decision = _decision_for_changed_field("point_provider_sig", ("provider-b",))

    assert decision.can_reuse is False
    assert decision.reject_reason == "point_provider_changed"


def test_geometry_objective_cache_not_reused_across_objective_mode_change():
    decision = _decision_for_changed_field("objective_mode_sig", ("pixel-objective",))

    assert decision.can_reuse is False
    assert decision.reject_reason == "objective_mode_changed"


def test_geometry_objective_cache_not_reused_across_active_fit_parameter_change():
    decision = _decision_for_changed_field(
        "active_fit_parameter_sig",
        ("center_x", "center_y", "corto_detector"),
    )

    assert decision.can_reuse is False
    assert decision.reject_reason == "active_fit_parameter_changed"


def test_geometry_objective_distance_change_runs_full_simulation():
    decision = _decision_for_changed_field("physics_sig", ("distance-changed",))
    trace = geometry_objective_cache_trace_payload(decision)

    assert decision.can_reuse is False
    assert decision.mode == "full_simulation"
    assert decision.reject_reason == "physics_changed"
    assert trace["objective_process_peaks_called"] is True


def test_geometry_objective_orientation_change_runs_full_simulation():
    decision = _decision_for_changed_field("physics_sig", ("orientation-changed",))
    trace = geometry_objective_cache_trace_payload(decision)

    assert decision.can_reuse is False
    assert decision.reject_reason == "physics_changed"
    assert trace["objective_cache_hit"] is False


def test_center_remap_residual_shape_and_order_match_full_path():
    full_path_labels = (
        "branch[0].delta_two_theta_deg",
        "branch[0].wrapped_delta_phi_deg",
        "branch[1].delta_two_theta_deg",
        "branch[1].wrapped_delta_phi_deg",
    )
    remap_labels = tuple(full_path_labels)

    assert center_remap_residual_shape_and_order_match(full_path_labels, remap_labels)
    assert not center_remap_residual_shape_and_order_match(
        full_path_labels,
        tuple(reversed(remap_labels)),
    )


def test_objective_cache_trace_reports_reject_reason():
    decision = _decision_for_changed_field("manual_selection_sig", ("manual-b",))
    trace = geometry_objective_cache_trace_payload(
        decision,
        residual_component_count=4,
    )

    assert trace["objective_cache_mode"] == "full_simulation"
    assert trace["objective_cache_hit"] is False
    assert trace["objective_cache_reject_reason"] == "manual_selection_changed"
    assert trace["objective_signature_changed_fields"] == ["manual_selection_sig"]
    assert trace["objective_residual_component_count"] == 4


def test_geometry_objective_cache_rejects_after_refined_peak_update_in_dry_run():
    cache: dict[str, GeometryObjectiveSignature] = {}

    def evaluate(signature: GeometryObjectiveSignature):
        decision = geometry_objective_cache_decision(
            cache.get("signature"),
            signature,
            exact_center_remap_cache_available=True,
        )
        if not decision.can_reuse:
            cache["signature"] = signature
        return geometry_objective_cache_trace_payload(decision)

    first = evaluate(_signature())
    second = evaluate(_signature(detector_center_sig=(102.0, 101.0)))
    third = evaluate(
        _signature(
            detector_center_sig=(103.0, 101.0),
            refined_peak_sig=("refined-updated",),
        )
    )

    assert first["objective_cache_reject_reason"] == "initial_evaluation"
    assert second["objective_cache_mode"] == "center_remap"
    assert second["objective_cache_hit"] is True
    assert third["objective_cache_mode"] == "full_simulation"
    assert third["objective_cache_reject_reason"] == "refined_peak_changed"


def test_geometry_objective_source_rows_cache_key_includes_identity_signature():
    builder_calls: list[int] = []

    def build_source_rows(*, local_params):
        builder_calls.append(len(builder_calls))
        return {
            "rows": [
                {
                    "source_row_index": len(builder_calls),
                    "hkl": (1, 0, 0),
                }
            ],
            "source": "test-builder",
        }

    subset = optimization.ReflectionSimulationSubset(
        miller=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.asarray([10.0], dtype=np.float64),
        measured_entries=[],
        original_indices=np.asarray([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = optimization.GeometryFitDatasetContext(
        dataset_index=7,
        label="background-7",
        theta_initial=0.0,
        subset=subset,
        qr_fit_trial_source_rows_builder=build_source_rows,
        qr_fit_trial_source_rows_builder_kind="test-builder-kind",
    )
    shared_cache: dict[tuple[object, ...], dict[str, object]] = {}

    first = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 1.0},
        params_signature="same-params",
        fit_context={
            "prediction_source_rows_cache": shared_cache,
            "objective_signature": _signature(qr_branch_identity_sig=("qr-a",)),
        },
    )
    second = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 1.0},
        params_signature="same-params",
        fit_context={
            "prediction_source_rows_cache": shared_cache,
            "objective_signature": _signature(qr_branch_identity_sig=("qr-a",)),
        },
    )
    third = optimization._build_trial_qr_source_rows_payload(
        dataset_ctx,
        trial_params={"center_x": 1.0},
        params_signature="same-params",
        fit_context={
            "prediction_source_rows_cache": shared_cache,
            "objective_signature": _signature(qr_branch_identity_sig=("qr-b",)),
        },
    )

    assert len(builder_calls) == 2
    assert len(shared_cache) == 2
    assert first["source_rows_rebuilt_or_reused"] == "rebuilt_for_trial_params"
    assert second["source_rows_rebuilt_or_reused"] == "reused_for_same_params_signature"
    assert third["source_rows_rebuilt_or_reused"] == "rebuilt_for_trial_params"
