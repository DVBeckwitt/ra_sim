from __future__ import annotations

import csv
import math
from types import SimpleNamespace

import numpy as np
import pytest

import ra_sim.gui.peak_sensitivity as peak_sensitivity
import scripts.debug.run_q_group_peak_sensitivity as sensitivity_cli
from ra_sim.gui.peak_sensitivity import (
    PeakObservation,
    PeakSensitivityEvaluationError,
    PeakSensitivityParameter,
    assemble_finite_difference_result,
    refine_record_in_caked_payload,
    run_peak_sensitivity,
    wrapped_phi_delta,
    write_sensitivity_artifacts,
)


def _parameter(name: str = "theta_initial", *, step: float = 0.5, scale: float | None = None):
    return PeakSensitivityParameter(
        name=name,
        baseline_value=10.0,
        step=step,
        units="deg",
        scale_for_normalized_output=step if scale is None else scale,
    )


def test_default_metric_preserves_refined_max() -> None:
    assert peak_sensitivity.DEFAULT_METRIC == peak_sensitivity.METRIC_REFINED_MAX


def _obs(
    branch: str,
    *,
    tth: float,
    phi: float,
    hkl: tuple[int, int, int] = (1, 0, 1),
    source_row_index: int = 0,
) -> PeakObservation:
    return PeakObservation(
        group_key=("q_group", "primary", 1, 1),
        branch_id=branch,
        hkl=hkl,
        source_reflection_index=7,
        source_table_index=2,
        source_row_index=source_row_index,
        two_theta_deg=tth,
        phi_deg=phi,
    )


def _source_record(
    *,
    group_key: tuple[object, ...] = ("q_group", "primary", 1, 1),
    hkl: tuple[int, int, int] = (1, 0, 1),
    source_reflection_index: int = 7,
    source_table_index: int = 2,
    source_row_index: int = 0,
    source_branch_index: int = 0,
    source_peak_index: int = 0,
) -> dict[str, object]:
    return {
        "q_group_key": group_key,
        "hkl": hkl,
        "source_reflection_index": source_reflection_index,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_table_index": source_table_index,
        "source_row_index": source_row_index,
        "source_branch_index": source_branch_index,
        "source_peak_index": source_peak_index,
    }


def test_source_rows_callback_uses_peak_sensitivity_consumer() -> None:
    seen: dict[str, object] = {}

    def callback(background_index, params, *, consumer):
        seen["background_index"] = background_index
        seen["params"] = params
        seen["consumer"] = consumer
        return [{"hkl": (1, 0, 1)}]

    rows = peak_sensitivity._call_source_rows_for_background(
        callback,
        3,
        {"theta_initial": 5.0},
    )

    assert rows == [{"hkl": (1, 0, 1)}]
    assert seen == {
        "background_index": 3,
        "params": {"theta_initial": 5.0},
        "consumer": "peak_sensitivity",
    }


def test_source_rows_callback_receives_required_pairs() -> None:
    seen: dict[str, object] = {}
    required_pairs = [{"hkl": (1, 0, 1), "source_row_index": 2}]

    def callback(background_index, params, *, consumer, required_pairs):
        seen["background_index"] = background_index
        seen["params"] = params
        seen["consumer"] = consumer
        seen["required_pairs"] = required_pairs
        return [{"hkl": (1, 0, 1)}]

    rows = peak_sensitivity._call_source_rows_for_background(
        callback,
        3,
        {"theta_initial": 5.0},
        required_pairs=required_pairs,
    )

    assert rows == [{"hkl": (1, 0, 1)}]
    assert seen == {
        "background_index": 3,
        "params": {"theta_initial": 5.0},
        "consumer": "peak_sensitivity",
        "required_pairs": required_pairs,
    }


def test_missing_trusted_reflection_fields_restore_on_stable_identity_match() -> None:
    record = {
        "q_group_key": ("q_group", "primary", 1, 1),
        "hkl": (1, 0, 1),
        "source_table_index": 2,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }

    restored = peak_sensitivity._restore_trusted_reflection_provenance(
        record,
        group_key=("q_group", "primary", 1, 1),
        required_pairs=[_source_record()],
    )

    assert restored["source_reflection_index"] == 7
    assert restored["source_reflection_namespace"] == "full_reflection"
    assert restored["source_reflection_is_full"] is True


def test_missing_hkl_does_not_restore_trusted_reflection_fields() -> None:
    record = {
        "q_group_key": ("q_group", "primary", 1, 1),
        "source_table_index": 2,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }

    restored = peak_sensitivity._restore_trusted_reflection_provenance(
        record,
        group_key=("q_group", "primary", 1, 1),
        required_pairs=[_source_record()],
    )

    assert "source_reflection_index" not in restored
    assert "source_reflection_namespace" not in restored
    assert "source_reflection_is_full" not in restored


def test_untrusted_complete_reflection_fields_are_not_restored() -> None:
    record = {
        "q_group_key": ("q_group", "primary", 1, 1),
        "hkl": (1, 0, 1),
        "source_table_index": 2,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }
    untrusted_pair = {
        **_source_record(),
        "source_reflection_namespace": "legacy_table",
        "source_reflection_is_full": False,
    }

    restored = peak_sensitivity._restore_trusted_reflection_provenance(
        record,
        group_key=("q_group", "primary", 1, 1),
        required_pairs=[untrusted_pair],
    )

    assert "source_reflection_index" not in restored
    assert "source_reflection_namespace" not in restored
    assert "source_reflection_is_full" not in restored


def test_missing_trusted_reflection_fields_do_not_restore_on_identity_mismatch() -> None:
    record = {
        "q_group_key": ("q_group", "primary", 1, 1),
        "hkl": (1, 0, 1),
        "source_table_index": 2,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 99,
    }

    restored = peak_sensitivity._restore_trusted_reflection_provenance(
        record,
        group_key=("q_group", "primary", 1, 1),
        required_pairs=[_source_record()],
    )

    assert "source_reflection_index" not in restored
    assert "source_reflection_namespace" not in restored
    assert "source_reflection_is_full" not in restored


def test_restored_reflection_provenance_allows_ok_sensitivity_status() -> None:
    baseline_record = {
        **_source_record(),
        "two_theta_deg": 10.0,
        "phi_deg": 20.0,
    }
    plus_record = {
        key: value
        for key, value in baseline_record.items()
        if key
        not in {
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
        }
    }
    plus_record.update({"two_theta_deg": 11.0, "phi_deg": 21.0})
    minus_record = dict(plus_record, two_theta_deg=9.0, phi_deg=19.0)
    required_pairs = [baseline_record]
    plus_record = peak_sensitivity._restore_trusted_reflection_provenance(
        plus_record,
        group_key=("q_group", "primary", 1, 1),
        required_pairs=required_pairs,
    )
    minus_record = peak_sensitivity._restore_trusted_reflection_provenance(
        minus_record,
        group_key=("q_group", "primary", 1, 1),
        required_pairs=required_pairs,
    )

    result = assemble_finite_difference_result(
        baseline_observations=[
            peak_sensitivity._record_to_observation(
                baseline_record,
                group_key=("q_group", "primary", 1, 1),
                branch_id="+x",
            )
        ],
        plus_observations={
            "theta_initial": [
                peak_sensitivity._record_to_observation(
                    plus_record,
                    group_key=("q_group", "primary", 1, 1),
                    branch_id="+x",
                )
            ]
        },
        minus_observations={
            "theta_initial": [
                peak_sensitivity._record_to_observation(
                    minus_record,
                    group_key=("q_group", "primary", 1, 1),
                    branch_id="+x",
                )
            ]
        },
        parameters=[_parameter(step=0.5)],
    )

    diag = result.diagnostics[("+x", "two_theta_deg", "theta_initial")]
    assert diag["identity_changed"] is False
    assert diag["status"] == "ok"
    assert result.jacobian[("+x", "two_theta_deg")]["theta_initial"] == 2.0


def test_run_peak_sensitivity_passes_baseline_records_as_required_pairs(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    group_key = ("q_group", "primary", 1, 1)
    baseline_record = {
        **_source_record(group_key=group_key),
        "two_theta_deg": 10.0,
        "phi_deg": 20.0,
    }

    class FakeEvaluator:
        def __init__(self, _state_path):
            self.context = SimpleNamespace(saved_state={}, state_path="fake-state.json")
            self.metadata = {}
            self._last_eval_metadata = {}

        @property
        def baseline_params(self):
            return {"theta_initial": 10.0}

        def evaluate_peak_observations(
            self,
            param_overrides,
            group_key_arg,
            branch_ids=None,
            required_pairs=None,
        ):
            calls.append(
                {
                    "param_overrides": dict(param_overrides),
                    "group_key": group_key_arg,
                    "branch_ids": branch_ids,
                    "required_pairs": required_pairs,
                }
            )
            if not param_overrides:
                return [
                    peak_sensitivity._record_to_observation(
                        baseline_record,
                        group_key=group_key,
                        branch_id="+x",
                    )
                ]
            record = {
                key: value
                for key, value in baseline_record.items()
                if key
                not in {
                    "source_reflection_index",
                    "source_reflection_namespace",
                    "source_reflection_is_full",
                }
            }
            delta = float(param_overrides.get("theta_initial", 10.0)) - 10.0
            record["two_theta_deg"] = 10.0 + delta
            record["phi_deg"] = 20.0 + delta
            record = peak_sensitivity._restore_trusted_reflection_provenance(
                record,
                group_key=group_key,
                required_pairs=required_pairs,
            )
            return [
                peak_sensitivity._record_to_observation(
                    record,
                    group_key=group_key,
                    branch_id="+x",
                )
            ]

    monkeypatch.setattr(peak_sensitivity, "PeakSensitivityEvaluator", FakeEvaluator)

    result = run_peak_sensitivity(
        state_path="fake-state.json",
        group_key=group_key,
        parameter_names=["theta_initial"],
    )

    assert calls[0]["required_pairs"] is None
    assert calls[1]["required_pairs"] == [baseline_record]
    assert calls[2]["required_pairs"] == [baseline_record]
    assert result.selected_metric == "refined_max"
    assert result.diagnostics[("+x", "two_theta_deg", "theta_initial")]["status"] == "ok"


def test_central_finite_difference_matrix_assembly() -> None:
    result = assemble_finite_difference_result(
        baseline_observations=[_obs("+x", tth=10.0, phi=20.0)],
        plus_observations={"theta_initial": [_obs("+x", tth=11.0, phi=21.0)]},
        minus_observations={"theta_initial": [_obs("+x", tth=9.0, phi=19.0)]},
        parameters=[_parameter(step=0.5)],
    )

    assert result.jacobian[("+x", "two_theta_deg")]["theta_initial"] == 2.0
    assert result.jacobian[("+x", "phi_deg")]["theta_initial"] == 2.0
    assert result.diagnostics[("+x", "two_theta_deg", "theta_initial")]["status"] == "ok"


def test_phi_wraparound_uses_short_angular_difference() -> None:
    assert wrapped_phi_delta(-179.0, 179.0) == 2.0
    result = assemble_finite_difference_result(
        baseline_observations=[_obs("+x", tth=10.0, phi=179.0)],
        plus_observations={"theta_initial": [_obs("+x", tth=10.0, phi=-179.0)]},
        minus_observations={"theta_initial": [_obs("+x", tth=10.0, phi=178.0)]},
        parameters=[_parameter(step=0.5)],
    )

    assert result.jacobian[("+x", "phi_deg")]["theta_initial"] == 3.0


def test_missing_branch_gives_nan_and_status() -> None:
    result = assemble_finite_difference_result(
        baseline_observations=[_obs("+x", tth=10.0, phi=20.0)],
        plus_observations={"theta_initial": []},
        minus_observations={"theta_initial": [_obs("+x", tth=9.0, phi=19.0)]},
        parameters=[_parameter(step=0.5)],
    )

    value = result.jacobian[("+x", "two_theta_deg")]["theta_initial"]
    diag = result.diagnostics[("+x", "two_theta_deg", "theta_initial")]
    assert math.isnan(value)
    assert diag["branch_missing"] is True
    assert diag["status"] == "missing_plus"
    assert math.isnan(diag["one_sided_plus"])
    assert diag["one_sided_minus"] == 2.0


def test_caked_payload_missing_gives_nan_and_status() -> None:
    missing = _obs("+x", tth=math.nan, phi=math.nan)
    missing = PeakObservation(**{**missing.__dict__, "status": "missing_caked_payload"})
    result = assemble_finite_difference_result(
        baseline_observations=[missing],
        plus_observations={"theta_initial": [missing]},
        minus_observations={"theta_initial": [missing]},
        parameters=[_parameter(step=0.5)],
    )

    value = result.jacobian[("+x", "two_theta_deg")]["theta_initial"]
    diag = result.diagnostics[("+x", "two_theta_deg", "theta_initial")]
    assert math.isnan(value)
    assert diag["status"] == "missing_caked_payload"


def test_unknown_observation_status_maps_to_stable_status() -> None:
    baseline = PeakObservation(
        **{**_obs("+x", tth=10.0, phi=20.0).__dict__, "status": "adapter_weird"}
    )
    result = assemble_finite_difference_result(
        baseline_observations=[baseline],
        plus_observations={"theta_initial": [_obs("+x", tth=11.0, phi=21.0)]},
        minus_observations={"theta_initial": [_obs("+x", tth=9.0, phi=19.0)]},
        parameters=[_parameter(step=0.5)],
    )

    value = result.jacobian[("+x", "two_theta_deg")]["theta_initial"]
    diag = result.diagnostics[("+x", "two_theta_deg", "theta_initial")]
    assert math.isnan(value)
    assert diag["status"] == "nonfinite_peak"
    assert diag["observation_status"] == "adapter_weird"
    assert diag["status"] in peak_sensitivity.STABLE_DIAGNOSTIC_STATUSES


def test_identity_change_is_flagged_but_derivative_is_computed() -> None:
    result = assemble_finite_difference_result(
        baseline_observations=[_obs("+x", tth=10.0, phi=20.0, source_row_index=0)],
        plus_observations={"theta_initial": [_obs("+x", tth=11.0, phi=21.0, source_row_index=9)]},
        minus_observations={"theta_initial": [_obs("+x", tth=9.0, phi=19.0, source_row_index=0)]},
        parameters=[_parameter(step=0.5)],
    )

    diag = result.diagnostics[("+x", "two_theta_deg", "theta_initial")]
    assert result.jacobian[("+x", "two_theta_deg")]["theta_initial"] == 2.0
    assert diag["identity_changed"] is True
    assert diag["status"] == "identity_changed"


def test_normalized_sensitivity_uses_parameter_scale() -> None:
    result = assemble_finite_difference_result(
        baseline_observations=[_obs("+x", tth=10.0, phi=20.0)],
        plus_observations={"theta_initial": [_obs("+x", tth=12.0, phi=20.0)]},
        minus_observations={"theta_initial": [_obs("+x", tth=8.0, phi=20.0)]},
        parameters=[_parameter(step=1.0, scale=0.25)],
    )

    assert result.jacobian[("+x", "two_theta_deg")]["theta_initial"] == 2.0
    assert result.normalized_jacobian[("+x", "two_theta_deg")]["theta_initial"] == 0.5


def _shape_obs(
    branch: str,
    *,
    com_tth: float,
    com_phi: float,
    provenance: str = "stable",
    status: str = "ok",
) -> peak_sensitivity.ShapeMetricObservation:
    return peak_sensitivity.ShapeMetricObservation(
        group_key=("q_group", "primary", 1, 1),
        branch_id=branch,
        metric="ray_cloud_com",
        values={
            "com_two_theta_deg": com_tth,
            "com_phi_deg": com_phi,
            "sigma_two_theta_deg": 1.0,
            "sigma_phi_deg": 2.0,
            "cov_two_theta_phi": 0.5,
            "major_sigma_deg": 2.0,
            "minor_sigma_deg": 1.0,
            "axis_angle_deg": 30.0,
            "delta_com_vs_max_two_theta": 0.5,
            "delta_com_vs_max_phi": 1.0,
        },
        point_count=5,
        total_weight=10.0,
        provenance_digest=provenance,
        status=status,
    )


def test_run_peak_sensitivity_passes_required_pairs_to_shape_evaluator(monkeypatch) -> None:
    peak_calls: list[dict[str, object]] = []
    shape_calls: list[dict[str, object]] = []
    group_key = ("q_group", "primary", 1, 1)
    baseline_record = {
        **_source_record(group_key=group_key),
        "two_theta_deg": 10.0,
        "phi_deg": 20.0,
    }

    class FakeEvaluator:
        def __init__(self, _state_path):
            self.context = SimpleNamespace(saved_state={}, state_path="fake-state.json")
            self.metadata = {}
            self._last_eval_metadata = {}

        @property
        def baseline_params(self):
            return {"theta_initial": 10.0}

        def evaluate_peak_observations(
            self,
            param_overrides,
            group_key_arg,
            branch_ids=None,
            required_pairs=None,
        ):
            peak_calls.append(
                {
                    "param_overrides": dict(param_overrides),
                    "group_key": group_key_arg,
                    "branch_ids": branch_ids,
                    "required_pairs": required_pairs,
                }
            )
            if not param_overrides:
                return [
                    peak_sensitivity._record_to_observation(
                        baseline_record,
                        group_key=group_key,
                        branch_id="+x",
                    )
                ]
            delta = float(param_overrides["theta_initial"]) - 10.0
            return [
                peak_sensitivity._record_to_observation(
                    {
                        **baseline_record,
                        "two_theta_deg": 10.0 + delta,
                        "phi_deg": 20.0 + delta,
                    },
                    group_key=group_key,
                    branch_id="+x",
                )
            ]

        def evaluate_shape_observations(
            self,
            param_overrides,
            group_key_arg,
            *,
            metric,
            refined_max_observations,
            branch_ids,
            reference_phi_by_branch,
            options,
            required_pairs=None,
        ):
            shape_calls.append(
                {
                    "param_overrides": dict(param_overrides),
                    "group_key": group_key_arg,
                    "branch_ids": branch_ids,
                    "required_pairs": required_pairs,
                    "metric": metric,
                    "min_cloud_points": options.min_cloud_points,
                }
            )
            delta = float(param_overrides.get("theta_initial", 10.0)) - 10.0
            return [_shape_obs("+x", com_tth=10.0 + delta, com_phi=20.0 + delta)]

    monkeypatch.setattr(peak_sensitivity, "PeakSensitivityEvaluator", FakeEvaluator)

    result = run_peak_sensitivity(
        state_path="fake-state.json",
        group_key=group_key,
        parameter_names=["theta_initial"],
        metric="ray_cloud_com",
        min_cloud_points=7,
    )

    assert peak_calls[0]["required_pairs"] is None
    assert peak_calls[1]["required_pairs"] == [baseline_record]
    assert peak_calls[2]["required_pairs"] == [baseline_record]
    assert [call["required_pairs"] for call in shape_calls] == [
        [baseline_record],
        [baseline_record],
        [baseline_record],
    ]
    assert [call["min_cloud_points"] for call in shape_calls] == [7, 7, 7]
    assert result.selected_metric == "ray_cloud_com"
    assert (
        result.shape_results["ray_cloud_com"].diagnostics[
            ("ray_cloud_com", "+x", "com_two_theta_deg", "theta_initial")
        ]["status"]
        == "ok"
    )


def test_weighted_com_on_synthetic_ray_cloud_gives_known_center() -> None:
    rows = [
        {"two_theta_deg": 10.0, "phi_deg": 20.0, "intensity": 1.0},
        {"two_theta_deg": 12.0, "phi_deg": 20.0, "intensity": 1.0},
        {"two_theta_deg": 12.0, "phi_deg": 20.0, "intensity": 2.0},
    ]

    values, point_count, total_weight, status, warning = (
        peak_sensitivity.compute_weighted_caked_shape_metrics(
            rows,
            reference_phi_deg=20.0,
            refined_max_two_theta_deg=11.0,
            refined_max_phi_deg=20.0,
        )
    )

    assert status == "ok"
    assert warning is True
    assert point_count == 3
    assert total_weight == 4.0
    assert values["com_two_theta_deg"] == 11.5
    assert values["com_phi_deg"] == 20.0


def test_ray_cloud_weight_falls_back_when_intensity_is_not_positive() -> None:
    rows = [
        {"two_theta_deg": 10.0, "phi_deg": 20.0, "intensity": 0.0, "weight": 1.0},
        {"two_theta_deg": 12.0, "phi_deg": 20.0, "intensity": -1.0, "weight": 1.0},
        {"two_theta_deg": 12.0, "phi_deg": 20.0, "weight": 2.0},
    ]

    values, point_count, total_weight, status, _warning = (
        peak_sensitivity.compute_weighted_caked_shape_metrics(
            rows,
            reference_phi_deg=20.0,
            refined_max_two_theta_deg=11.0,
            refined_max_phi_deg=20.0,
        )
    )

    assert status == "ok"
    assert point_count == 3
    assert total_weight == 4.0
    assert values["com_two_theta_deg"] == 11.5


def test_missing_refined_max_keeps_valid_com_and_nan_offsets() -> None:
    rows = [
        {"two_theta_deg": 10.0, "phi_deg": 20.0, "intensity": 1.0},
        {"two_theta_deg": 12.0, "phi_deg": 20.0, "intensity": 1.0},
        {"two_theta_deg": 12.0, "phi_deg": 20.0, "intensity": 2.0},
    ]

    values, _point_count, _total_weight, status, _warning = (
        peak_sensitivity.compute_weighted_caked_shape_metrics(
            rows,
            reference_phi_deg=20.0,
            refined_max_two_theta_deg=math.nan,
            refined_max_phi_deg=math.nan,
        )
    )

    assert status == "ok"
    assert values["com_two_theta_deg"] == 11.5
    assert values["com_phi_deg"] == 20.0
    assert math.isnan(values["delta_com_vs_max_two_theta"])
    assert math.isnan(values["delta_com_vs_max_phi"])


def test_wrapped_phi_com_works_across_boundary() -> None:
    rows = [
        {"two_theta_deg": 10.0, "phi_deg": 179.0, "intensity": 1.0},
        {"two_theta_deg": 10.0, "phi_deg": -179.0, "intensity": 1.0},
        {"two_theta_deg": 10.0, "phi_deg": -178.0, "intensity": 1.0},
    ]

    values, _point_count, _total_weight, status, _warning = (
        peak_sensitivity.compute_weighted_caked_shape_metrics(
            rows,
            reference_phi_deg=179.0,
            refined_max_two_theta_deg=10.0,
            refined_max_phi_deg=179.0,
        )
    )

    assert status == "ok"
    assert math.isclose(values["com_phi_deg"], -179.33333333333334)


def test_covariance_major_minor_axes_for_synthetic_ellipse() -> None:
    center_tth = 10.0
    center_phi = 20.0
    angle = math.radians(30.0)
    major_radius = math.sqrt(8.0)
    minor_radius = math.sqrt(2.0)
    major_vec = (math.cos(angle), math.sin(angle))
    minor_vec = (-math.sin(angle), math.cos(angle))
    rows = []
    for sign in (-1.0, 1.0):
        rows.append(
            {
                "two_theta_deg": center_tth + sign * major_radius * major_vec[0],
                "phi_deg": center_phi + sign * major_radius * major_vec[1],
                "intensity": 1.0,
            }
        )
        rows.append(
            {
                "two_theta_deg": center_tth + sign * minor_radius * minor_vec[0],
                "phi_deg": center_phi + sign * minor_radius * minor_vec[1],
                "intensity": 1.0,
            }
        )

    values, _point_count, _total_weight, status, warning = (
        peak_sensitivity.compute_weighted_caked_shape_metrics(
            rows,
            reference_phi_deg=center_phi,
            refined_max_two_theta_deg=center_tth,
            refined_max_phi_deg=center_phi,
        )
    )

    assert status == "ok"
    assert warning is True
    assert math.isclose(values["major_sigma_deg"], 2.0)
    assert math.isclose(values["minor_sigma_deg"], 1.0)
    assert math.isclose(values["axis_angle_deg"], 30.0)


def test_com_relative_to_refined_max_is_correct() -> None:
    rows = [
        {"two_theta_deg": 11.0, "phi_deg": 22.0, "intensity": 1.0},
        {"two_theta_deg": 11.0, "phi_deg": 22.0, "intensity": 1.0},
        {"two_theta_deg": 11.0, "phi_deg": 22.0, "intensity": 1.0},
    ]

    values, _point_count, _total_weight, status, _warning = (
        peak_sensitivity.compute_weighted_caked_shape_metrics(
            rows,
            reference_phi_deg=20.0,
            refined_max_two_theta_deg=10.0,
            refined_max_phi_deg=20.0,
        )
    )

    assert status == "ok"
    assert values["delta_com_vs_max_two_theta"] == 1.0
    assert values["delta_com_vs_max_phi"] == 2.0


def test_finite_difference_derivative_of_synthetic_com_shift_is_correct() -> None:
    result = peak_sensitivity.assemble_shape_finite_difference_result(
        metric="ray_cloud_com",
        baseline_observations=[_shape_obs("+x", com_tth=10.0, com_phi=179.0)],
        plus_observations={"theta_initial": [_shape_obs("+x", com_tth=11.0, com_phi=-179.0)]},
        minus_observations={"theta_initial": [_shape_obs("+x", com_tth=9.0, com_phi=178.0)]},
        parameters=[_parameter(step=0.5)],
    )

    assert result.jacobian[("ray_cloud_com", "+x", "com_two_theta_deg")]["theta_initial"] == 2.0
    assert result.jacobian[("ray_cloud_com", "+x", "com_phi_deg")]["theta_initial"] == 3.0
    assert (
        result.diagnostics[("ray_cloud_com", "+x", "com_phi_deg", "theta_initial")]["status"]
        == "ok"
    )


def test_image_roi_com_recovers_gaussian_center_after_background_subtraction() -> None:
    radial = np.linspace(9.0, 11.0, 81)
    azimuth = np.linspace(-1.0, 3.0, 121)
    rr, pp = np.meshgrid(radial, azimuth)
    image = 5.0 + 100.0 * np.exp(-(((rr - 10.2) ** 2) / 0.08 + ((pp - 1.3) ** 2) / 0.12))

    values, _point_count, _total_weight, status, _warning = (
        peak_sensitivity.compute_image_roi_shape_metrics(
            caked_image=image,
            radial_axis=radial,
            azimuth_axis=azimuth,
            refined_max_two_theta_deg=10.0,
            refined_max_phi_deg=1.0,
            reference_phi_deg=1.0,
            roi_two_theta_half_width=0.8,
            roi_phi_half_width=1.0,
            background_percentile=0.0,
            min_total_weight=1.0e-9,
        )
    )

    assert status == "ok"
    assert math.isclose(values["com_two_theta_deg"], 10.2, abs_tol=0.03)
    assert math.isclose(values["com_phi_deg"], 1.3, abs_tol=0.03)


def test_image_roi_insufficient_weight_gives_nan_status() -> None:
    image = np.full((5, 5), 7.0, dtype=np.float64)

    values, _point_count, total_weight, status, _warning = (
        peak_sensitivity.compute_image_roi_shape_metrics(
            caked_image=image,
            radial_axis=np.linspace(1.0, 5.0, 5),
            azimuth_axis=np.linspace(-2.0, 2.0, 5),
            refined_max_two_theta_deg=3.0,
            refined_max_phi_deg=0.0,
            reference_phi_deg=0.0,
            roi_two_theta_half_width=2.0,
            roi_phi_half_width=2.0,
            background_percentile=50.0,
        )
    )

    assert status == "insufficient_weight"
    assert total_weight == 0.0
    assert math.isnan(values["com_two_theta_deg"])


def test_insufficient_cloud_points_gives_nan_status() -> None:
    rows = [
        {"two_theta_deg": 10.0, "phi_deg": 20.0, "intensity": 1.0},
        {"two_theta_deg": 11.0, "phi_deg": 21.0, "intensity": 1.0},
    ]

    values, point_count, _total_weight, status, _warning = (
        peak_sensitivity.compute_weighted_caked_shape_metrics(
            rows,
            reference_phi_deg=20.0,
            refined_max_two_theta_deg=10.0,
            refined_max_phi_deg=20.0,
        )
    )

    assert point_count == 2
    assert status == "insufficient_cloud_points"
    assert math.isnan(values["com_phi_deg"])


def test_shape_branch_provenance_stays_stable_across_baseline_plus_minus() -> None:
    result = peak_sensitivity.assemble_shape_finite_difference_result(
        metric="ray_cloud_com",
        baseline_observations=[_shape_obs("+x", com_tth=10.0, com_phi=20.0)],
        plus_observations={"theta_initial": [_shape_obs("+x", com_tth=11.0, com_phi=21.0)]},
        minus_observations={"theta_initial": [_shape_obs("+x", com_tth=9.0, com_phi=19.0)]},
        parameters=[_parameter(step=0.5)],
    )

    diag = result.diagnostics[("ray_cloud_com", "+x", "com_two_theta_deg", "theta_initial")]
    assert diag["status"] == "ok"
    assert diag["identity_changed"] is False
    assert result.jacobian[("ray_cloud_com", "+x", "com_two_theta_deg")]["theta_initial"] == 2.0


def test_csv_output_has_stable_row_and_column_order(tmp_path) -> None:
    baseline = [_obs("+x", tth=10.0, phi=20.0), _obs("-x", tth=30.0, phi=40.0)]
    parameters = [_parameter("theta_initial", step=1.0), _parameter("chi", step=1.0)]
    result = assemble_finite_difference_result(
        baseline_observations=baseline,
        plus_observations={
            "theta_initial": [_obs("+x", tth=11.0, phi=21.0), _obs("-x", tth=31.0, phi=41.0)],
            "chi": [_obs("+x", tth=12.0, phi=22.0), _obs("-x", tth=32.0, phi=42.0)],
        },
        minus_observations={
            "theta_initial": [_obs("+x", tth=9.0, phi=19.0), _obs("-x", tth=29.0, phi=39.0)],
            "chi": [_obs("+x", tth=8.0, phi=18.0), _obs("-x", tth=28.0, phi=38.0)],
        },
        parameters=parameters,
    )

    paths = write_sensitivity_artifacts(result, tmp_path, metadata={"kind": "test"})
    with paths["sensitivity_matrix"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    with paths["normalized_sensitivity_matrix"].open(
        newline="",
        encoding="utf-8",
    ) as handle:
        normalized_rows = list(csv.reader(handle))
    with paths["sensitivity_long"].open(newline="", encoding="utf-8") as handle:
        long_rows = list(csv.DictReader(handle))

    assert rows[0] == ["branch_id", "coordinate", "theta_initial", "chi"]
    assert rows[1][:2] == ["+x", "two_theta_deg"]
    assert rows[2][:2] == ["+x", "phi_deg"]
    assert rows[3][:2] == ["-x", "two_theta_deg"]
    assert rows[4][:2] == ["-x", "phi_deg"]
    assert normalized_rows[0] == rows[0]
    assert normalized_rows[1][:2] == ["+x", "two_theta_deg"]
    assert (
        float(normalized_rows[1][2])
        == result.normalized_jacobian[("+x", "two_theta_deg")]["theta_initial"]
    )
    assert "baseline_gui_mismatch" in long_rows[0]


def test_real_detector_image_accepts_experimental_image_key() -> None:
    image = np.ones((3, 4), dtype=np.float64)
    context = SimpleNamespace(dataset={"spec": {"experimental_image": image}})

    found, diagnostics = peak_sensitivity._real_detector_image_from_context(context)

    assert found is not None
    assert found.shape == (3, 4)
    assert diagnostics["real_runtime_image_status"] == "ok"
    assert diagnostics["real_runtime_image_key"] == "experimental_image"


def test_build_caked_payload_does_not_use_sparse_source_rows() -> None:
    context = SimpleNamespace(
        params={"image_size": 4},
        modules=SimpleNamespace(),
        dataset={},
        prepare_result=None,
    )

    payload, diagnostics = peak_sensitivity._build_caked_payload(context, {"image_size": 4})

    assert payload.image is None
    assert diagnostics["caked_payload_status"] == "missing_caked_payload"
    assert diagnostics["direct_simulation_fallback_used"] is False
    assert "source_row_sparse_image_fallback_used" not in diagnostics
    assert not hasattr(peak_sensitivity, "_simulate_image_fallback")


def test_existing_caked_payload_reused_for_baseline_equivalent_params() -> None:
    existing_image = np.ones((2, 3), dtype=np.float64)
    owner = SimpleNamespace(
        last_caked_image_unscaled=existing_image,
        last_caked_radial_values=np.array([1.0, 2.0, 3.0]),
        last_caked_azimuth_values=np.array([-1.0, 1.0]),
        last_caked_transform_bundle=SimpleNamespace(detector_shape=(4, 5)),
    )
    params = {
        "center": [1.0, 1.0],
        "center_x": 1.0,
        "center_y": 1.0,
        "corto_detector": 10.0,
        "pixel_size": 1.0,
        "pixel_size_m": 1.0,
        "lambda": 1.54,
    }
    context = SimpleNamespace(
        prepare_result=owner,
        projection_callbacks=None,
        manual_dataset_bindings=SimpleNamespace(),
        bindings=None,
        dataset={},
        params=params,
        modules=SimpleNamespace(),
    )

    payload, diagnostics = peak_sensitivity._build_caked_payload(context, dict(params))

    assert payload.image is existing_image
    assert diagnostics["payload_source"] == "existing_runtime_caked_payload"
    assert diagnostics["existing_caked_payload_reused"] is True
    assert diagnostics["existing_caked_payload_reuse_reason"] == "matches_baseline_cake_geometry"
    assert diagnostics["cake_geometry_signature"]["detector_shape"] == [4, 5]


def test_existing_caked_payload_not_reused_when_cake_geometry_changes() -> None:
    class FakeIntegrator:
        def __init__(self, **_kwargs):
            pass

        def integrate2d(self, *_args, **_kwargs):
            return SimpleNamespace(azimuthal=np.array([-1.0, 1.0]))

    class FakeExactCake:
        FastAzimuthalIntegrator = FakeIntegrator

        @staticmethod
        def prepare_gui_phi_display(_result):
            return (
                np.full((2, 3), 7.0, dtype=np.float64),
                np.array([1.0, 2.0, 3.0], dtype=np.float64),
                np.array([-1.0, 1.0], dtype=np.float64),
            )

        @staticmethod
        def resolve_cake_transform_bundle(*_args, **_kwargs):
            return object()

    owner = SimpleNamespace(
        last_caked_image_unscaled=np.zeros((2, 3), dtype=np.float64),
        last_caked_radial_values=np.array([1.0, 2.0, 3.0]),
        last_caked_azimuth_values=np.array([-1.0, 1.0]),
        last_caked_transform_bundle=object(),
    )
    baseline_params = {
        "center": [1.0, 1.0],
        "center_x": 1.0,
        "center_y": 1.0,
        "corto_detector": 10.0,
        "pixel_size": 1.0,
        "pixel_size_m": 1.0,
        "lambda": 1.54,
    }
    changed_params = {
        **baseline_params,
        "center": [2.0, 1.0],
        "center_x": 2.0,
    }
    context = SimpleNamespace(
        prepare_result=owner,
        projection_callbacks=None,
        manual_dataset_bindings=SimpleNamespace(),
        bindings=None,
        dataset={"spec": {"experimental_image": np.ones((4, 4), dtype=np.float64)}},
        params=baseline_params,
        modules=SimpleNamespace(exact_cake_portable=FakeExactCake),
    )

    payload, diagnostics = peak_sensitivity._build_caked_payload(context, changed_params)

    assert np.all(payload.image == 7.0)
    assert diagnostics["payload_source"] == "real_runtime_image"
    assert diagnostics["existing_caked_payload_reused"] is False
    assert diagnostics["existing_caked_payload_reuse_reason"] == "cake_geometry_changed"
    assert diagnostics["cake_geometry_signature"]["center_x"] == 2.0


def test_runtime_baseline_not_replaced_by_saved_gui_peak(monkeypatch) -> None:
    group_key = ("q_group", "primary", 1, 1)

    class FakeEvaluator:
        def __init__(self, _state_path):
            self.context = SimpleNamespace(
                saved_state={
                    "geometry": {
                        "manual_pairs": [
                            {
                                "entries": [
                                    {
                                        "q_group_key": list(group_key),
                                        "branch_id": "+x",
                                        "refined_sim_caked_x": 99.0,
                                        "refined_sim_caked_y": 20.0,
                                    }
                                ]
                            }
                        ]
                    }
                },
                state_path="fake-state.json",
            )
            self.metadata = {}
            self._last_eval_metadata = {}

        @property
        def baseline_params(self):
            return {"theta_initial": 10.0}

        def evaluate_peak_observations(
            self,
            param_overrides,
            group_key_arg,
            branch_ids=None,
            required_pairs=None,
        ):
            if not param_overrides:
                self._last_eval_metadata = {}
                return [_obs("+x", tth=10.0, phi=20.0)]
            value = float(param_overrides["theta_initial"])
            self._last_eval_metadata = {}
            tth = 11.0 if value > 10.0 else 9.0
            return [_obs("+x", tth=tth, phi=20.0)]

    monkeypatch.setattr(peak_sensitivity, "PeakSensitivityEvaluator", FakeEvaluator)

    result = run_peak_sensitivity(
        state_path="fake-state.json",
        group_key=group_key,
        parameter_names=["theta_initial"],
        metric="refined_max",
    )

    assert result.baseline_observations[0].two_theta_deg == 10.0
    assert result.metadata["baseline_coordinate_source"] == "runtime_evaluator"
    assert result.metadata["baseline_gui_smoke_check"]["status"] == "mismatch"
    assert result.metadata["baseline_gui_smoke_check"]["unreliable"] is True
    diag = result.diagnostics[("+x", "two_theta_deg", "theta_initial")]
    assert diag["baseline_gui_mismatch"] is True
    assert diag["status"] == "peak_jump"


def test_baseline_eval_failure_raises(monkeypatch) -> None:
    class FakeEvaluator:
        def __init__(self, _state_path):
            self.context = SimpleNamespace(saved_state={}, state_path="fake-state.json")
            self.metadata = {}
            self._last_eval_metadata = {}

        @property
        def baseline_params(self):
            return {"theta_initial": 10.0}

        def evaluate_peak_observations(
            self,
            _param_overrides,
            _group_key,
            branch_ids=None,
            required_pairs=None,
        ):
            self._last_eval_metadata = {"eval_error": "RuntimeError: boom"}
            return []

    monkeypatch.setattr(peak_sensitivity, "PeakSensitivityEvaluator", FakeEvaluator)

    with pytest.raises(PeakSensitivityEvaluationError, match="RuntimeError: boom"):
        run_peak_sensitivity(
            state_path="fake-state.json",
            group_key=("q_group", "primary", 1, 1),
            parameter_names=["theta_initial"],
        )


def test_empty_baseline_raises(monkeypatch) -> None:
    class FakeEvaluator:
        def __init__(self, _state_path):
            self.context = SimpleNamespace(saved_state={}, state_path="fake-state.json")
            self.metadata = {}
            self._last_eval_metadata = {}

        @property
        def baseline_params(self):
            return {"theta_initial": 10.0}

        def evaluate_peak_observations(
            self,
            _param_overrides,
            _group_key,
            branch_ids=None,
            required_pairs=None,
        ):
            self._last_eval_metadata = {}
            return []

    monkeypatch.setattr(peak_sensitivity, "PeakSensitivityEvaluator", FakeEvaluator)

    with pytest.raises(PeakSensitivityEvaluationError, match="no observations"):
        run_peak_sensitivity(
            state_path="fake-state.json",
            group_key=("q_group", "primary", 1, 1),
            parameter_names=["theta_initial"],
        )


def test_missing_caked_payload_baseline_raises(monkeypatch) -> None:
    class FakeEvaluator:
        def __init__(self, _state_path):
            self.context = SimpleNamespace(saved_state={}, state_path="fake-state.json")
            self.metadata = {}
            self._last_eval_metadata = {}

        @property
        def baseline_params(self):
            return {"theta_initial": 10.0}

        def evaluate_peak_observations(
            self,
            _param_overrides,
            _group_key,
            branch_ids=None,
            required_pairs=None,
        ):
            self._last_eval_metadata = {}
            return [
                PeakObservation(
                    group_key=("q_group", "primary", 1, 1),
                    branch_id="+x",
                    two_theta_deg=math.nan,
                    phi_deg=math.nan,
                    status="missing_caked_payload",
                )
            ]

    monkeypatch.setattr(peak_sensitivity, "PeakSensitivityEvaluator", FakeEvaluator)

    with pytest.raises(PeakSensitivityEvaluationError, match="missing caked payload"):
        run_peak_sensitivity(
            state_path="fake-state.json",
            group_key=("q_group", "primary", 1, 1),
            parameter_names=["theta_initial"],
        )


def test_cli_baseline_failure_prints_eval_error(monkeypatch, capsys, tmp_path) -> None:
    def _raise_eval_error(**_kwargs):
        raise PeakSensitivityEvaluationError("baseline failed")

    monkeypatch.setattr(sensitivity_cli, "run_peak_sensitivity", _raise_eval_error)

    code = sensitivity_cli.main(
        [
            "--state",
            "fake-state.json",
            "--group-key",
            "q_group,primary,1,1",
            "--outdir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert '"status": "eval_error"' in captured.out
    assert '"error": "baseline failed"' in captured.out


def test_cli_passes_min_cloud_points(monkeypatch, capsys, tmp_path) -> None:
    seen: dict[str, object] = {}

    def _run_peak_sensitivity(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            baseline_observations=[],
            metadata={"status": "ok", "baseline_shape_metrics": {}},
            selected_metric="ray_cloud_com",
        )

    def _write_sensitivity_artifacts(_result, _outdir, **_kwargs):
        return {"metadata": tmp_path / "metadata.json"}

    monkeypatch.setattr(sensitivity_cli, "run_peak_sensitivity", _run_peak_sensitivity)
    monkeypatch.setattr(
        sensitivity_cli,
        "write_sensitivity_artifacts",
        _write_sensitivity_artifacts,
    )

    code = sensitivity_cli.main(
        [
            "--state",
            "fake-state.json",
            "--group-key",
            "q_group,primary,1,1",
            "--metric",
            "ray_cloud_com",
            "--min-cloud-points",
            "7",
            "--outdir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert '"status": "ok"' in captured.out
    assert seen["metric"] == "ray_cloud_com"
    assert seen["min_cloud_points"] == 7


def test_synthetic_caked_refinement_recovers_known_local_maximum() -> None:
    image = np.zeros((7, 7), dtype=np.float64)
    image[4, 3] = 100.0
    radial_axis = np.linspace(10.0, 16.0, 7)
    azimuth_axis = np.linspace(-3.0, 3.0, 7)
    record = {"two_theta_deg": 13.0, "phi_deg": 0.0, "caked_x": 13.0, "caked_y": 0.0}

    refined, point = refine_record_in_caked_payload(
        record,
        caked_image=image,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )

    assert point is not None
    assert refined is not None
    assert refined["two_theta_deg"] == 13.0
    assert refined["phi_deg"] == 1.0
    assert refined["refined_by"] == "caked_peak_center"
