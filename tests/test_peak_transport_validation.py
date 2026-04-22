from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from ra_sim.gui import peak_transport_validation as transport
from ra_sim.gui.peak_sensitivity import ShapeMetricObservation, wrapped_phi_delta
from ra_sim.simulation import exact_cake_portable


def _options() -> transport.TransportValidationOptions:
    return transport.TransportValidationOptions(
        min_cloud_points=3,
        tol_two_theta_deg=0.01,
        tol_phi_deg=0.01,
        tol_shape=0.01,
    )


def _bundle(*, dist: float = 1.0):
    ai = exact_cake_portable.FastAzimuthalIntegrator(
        dist=float(dist),
        poni1=10.0e-3,
        poni2=10.0e-3,
        pixel1=1.0e-3,
        pixel2=1.0e-3,
    )
    shape = (21, 21)
    radial, raw_azimuth = exact_cake_portable.build_angle_axes(
        npt_rad=200,
        npt_azim=180,
        tth_min_deg=0.0,
        tth_max_deg=exact_cake_portable.detector_two_theta_max_deg(shape, ai.geometry),
        azimuth_min_deg=-180.0,
        azimuth_max_deg=180.0,
    )
    gui_azimuth = exact_cake_portable.raw_phi_to_gui_phi(raw_azimuth)
    bundle = exact_cake_portable.resolve_cake_transform_bundle(
        ai,
        shape,
        radial,
        gui_azimuth_deg=np.asarray(gui_azimuth[np.argsort(gui_azimuth, kind="stable")]),
        raw_azimuth_deg=raw_azimuth,
        require_gui_display_match=True,
    )
    assert bundle is not None
    return ai, bundle


def _fake_evaluator():
    return SimpleNamespace(
        context=SimpleNamespace(modules=SimpleNamespace(exact_cake_portable=exact_cake_portable))
    )


def _decision_by_scope(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["decision_scope"]): row for row in rows}


def _shape_values(tth: float, phi: float) -> dict[str, float]:
    return {
        "com_two_theta_deg": float(tth),
        "com_phi_deg": float(phi),
        "sigma_two_theta_deg": 0.1,
        "sigma_phi_deg": 0.1,
        "cov_two_theta_phi": 0.0,
        "major_sigma_deg": 0.1,
        "minor_sigma_deg": 0.1,
        "axis_angle_deg": 0.0,
        "delta_com_vs_max_two_theta": 0.0,
        "delta_com_vs_max_phi": 0.0,
    }


def _shape_obs(branch: str, *, tth: float = 1.0, phi: float = 2.0, status: str = "ok"):
    return ShapeMetricObservation(
        group_key=("q_group", "primary", 1, 5),
        branch_id=branch,
        metric="ray_cloud_com",
        values=_shape_values(tth, phi),
        point_count=3,
        total_weight=3.0,
        status=status,
        provenance_digest="abc",
    )


def _transport_comp(
    *,
    tth: float = 1.0,
    phi: float = 2.0,
    point_count: int = 3,
    status: str = "ok",
):
    return transport._TransportComputation(
        values=_shape_values(tth, phi),
        point_count=point_count,
        total_weight=float(point_count),
        status=status,
        low_point_warning=False,
    )


def _decision_rows(
    *,
    comparison: str,
    metric: str = "image_roi_com",
    parameter: str = "corto_detector",
    branch: str = "+x",
    moved: bool = False,
    mismatch: bool = False,
    fallback: bool = False,
    point_count: int = 3,
):
    rows: list[dict[str, object]] = []
    for side in ("plus", "minus"):
        for coordinate in transport.DECISION_COORDINATES:
            baseline = 1.0
            full = 1.1 if moved and coordinate == "com_two_theta_deg" else 1.0
            transported = full
            error = 0.0 if not mismatch else 0.1
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "branch_id": branch,
                    "parameter": parameter,
                    "side": side,
                    "coordinate": coordinate,
                    "baseline_value": baseline,
                    "full_recompute_value": full,
                    "transported_value": transported if not mismatch else baseline,
                    "abs_transport_error": abs(error),
                    "pass": not mismatch and not fallback and point_count >= 3,
                    "point_count": point_count,
                    "baseline_transport_mismatch": False,
                    "full_recompute_status": "ok",
                    "transport_status": ("ok" if point_count >= 3 else "insufficient_cloud_points"),
                    "direct_simulation_fallback_used": fallback,
                    "sparse_source_row_fallback_used": False,
                }
            )
    return rows


def _manual_projected_com(samples, bundle, *, reference_phi: float = 0.0):
    weights: list[float] = []
    two_theta_values: list[float] = []
    phi_values: list[float] = []
    for sample in samples:
        two_theta, phi = exact_cake_portable.detector_pixel_to_caked_bin(
            bundle,
            float(sample["_bundle_detector_col"]),
            float(sample["_bundle_detector_row"]),
        )
        weight = float(sample["weight"])
        weights.append(weight)
        two_theta_values.append(float(two_theta))
        phi_values.append(float(phi))
    weight_arr = np.asarray(weights, dtype=np.float64)
    two_theta_arr = np.asarray(two_theta_values, dtype=np.float64)
    phi_unwrapped = np.asarray(
        [reference_phi + wrapped_phi_delta(phi, reference_phi) for phi in phi_values],
        dtype=np.float64,
    )
    total = float(np.sum(weight_arr))
    com_two_theta = float(np.sum(weight_arr * two_theta_arr) / total)
    com_phi = float((np.sum(weight_arr * phi_unwrapped) / total + 180.0) % 360.0 - 180.0)
    return com_two_theta, com_phi


def test_detector_pixel_transport_matches_full_recompute_for_fixed_image() -> None:
    opts = transport.TransportValidationOptions(
        roi_two_theta_half_width=1.0,
        roi_phi_half_width=180.0,
        background_percentile=0.0,
        min_cloud_points=3,
        tol_two_theta_deg=1.0e-12,
        tol_phi_deg=1.0e-12,
    )
    yy, xx = np.indices((21, 21), dtype=np.float64)
    image = 1.0 + 100.0 * np.exp(-(((xx - 12.0) ** 2) + ((yy - 9.0) ** 2)) / 8.0)
    ai, baseline_bundle = _bundle(dist=1.0)
    _pert_ai, perturbed_bundle = _bundle(dist=1.01)
    samples = transport.build_image_roi_transport_samples(
        image,
        refined_two_theta_deg=0.2,
        refined_phi_deg=0.0,
        ai=ai,
        bundle=baseline_bundle,
        evaluator=None,
        branch_id="+x",
        options=opts,
    )

    transported = transport._transport_from_samples(
        _fake_evaluator(),
        samples,
        perturbed_bundle,
        reference_phi_deg=0.0,
        refined_max_two_theta_deg=0.2,
        refined_max_phi_deg=0.0,
        metric="image_roi_com",
        options=opts,
    )
    expected_tth, expected_phi = _manual_projected_com(samples, perturbed_bundle)

    assert transported.status == "ok"
    assert math.isclose(
        transported.values["com_two_theta_deg"],
        expected_tth,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        transported.values["com_phi_deg"],
        expected_phi,
        abs_tol=1.0e-12,
    )


def test_same_roi_full_from_samples_does_not_recake(monkeypatch) -> None:
    def fail_recake(*_args, **_kwargs):
        raise AssertionError("recake forbidden")

    monkeypatch.setattr(transport.ps, "_build_caked_payload", fail_recake)
    opts = _options()
    _ai, bundle = _bundle(dist=1.0)
    bundle_payload = transport._TransportBundle(
        bundle=bundle,
        ai=None,
        detector_shape=(21, 21),
        diagnostics={"transport_used_integrate2d": False},
    )
    samples = [
        {"_bundle_detector_col": 10.0, "_bundle_detector_row": 10.0, "weight": 1.0},
        {"_bundle_detector_col": 11.0, "_bundle_detector_row": 10.0, "weight": 1.0},
        {"_bundle_detector_col": 10.0, "_bundle_detector_row": 11.0, "weight": 1.0},
    ]

    result = transport._same_roi_full_from_samples(
        _fake_evaluator(),
        samples,
        bundle_payload,
        reference_phi_deg=0.0,
        refined_max_two_theta_deg=0.0,
        refined_max_phi_deg=0.0,
        options=opts,
    )

    assert result.status == "ok"


def test_theta_initial_transport_zero_when_image_and_cake_geometry_unchanged() -> None:
    rows = _decision_rows(
        comparison="transport_vs_new_roi_full",
        metric="image_roi_com",
        parameter="theta_initial",
        moved=False,
    )

    decisions = _decision_by_scope(
        transport.build_transport_decision_rows(rows, options=_options())
    )

    assert decisions["com"]["recommendation"] == "keep_fixed"
    assert decisions["com"]["can_transport"] is True
    assert decisions["com"]["plus_pass"] is True
    assert decisions["com"]["minus_pass"] is True


def test_source_row_transport_detects_when_theta_changes_source_rows() -> None:
    baseline = _shape_obs("+x", tth=10.0, phi=20.0)
    transported = _transport_comp(tth=10.0, phi=20.0)
    full_plus = _shape_obs("+x", tth=10.5, phi=20.0)
    full_minus = _shape_obs("+x", tth=9.5, phi=20.0)
    rows = []
    for side, full in (("plus", full_plus), ("minus", full_minus)):
        rows.extend(
            transport._comparison_rows(
                comparison="transport_vs_full_recompute",
                metric="ray_cloud_com",
                group_key=("q_group", "primary", 1, 5),
                branch_id="+x",
                parameter_name="theta_initial",
                side=side,
                baseline=baseline,
                full=full,
                transported=transported,
                options=_options(),
                baseline_transport_mismatches=set(),
                full_metadata={},
                transport_metadata={},
                full_recompute_roi_definition="perturbed_source_rows",
                transport_roi_definition="baseline_source_rows_fixed",
            )
        )

    decisions = _decision_by_scope(
        transport.build_transport_decision_rows(rows, options=_options())
    )

    assert decisions["com"]["recommendation"] == "full_recompute_required"
    assert decisions["com"]["can_transport"] is False
    assert decisions["shape"]["recommendation"] == "keep_fixed"
    assert decisions["shape"]["can_transport"] is True


def test_source_row_transport_passes_when_theta_does_not_change_rows() -> None:
    rows = _decision_rows(
        comparison="transport_vs_full_recompute",
        metric="ray_cloud_com",
        parameter="theta_initial",
        moved=False,
    )

    decision = _decision_by_scope(
        transport.build_transport_decision_rows(rows, options=_options())
    )["com"]

    assert decision["recommendation"] == "keep_fixed"
    assert decision["can_transport"] is True


def test_insufficient_cloud_points_blocks_ray_cloud_transport() -> None:
    rows = _decision_rows(
        comparison="transport_vs_full_recompute",
        metric="ray_cloud_com",
        parameter="theta_initial",
        point_count=2,
    )

    decision = _decision_by_scope(
        transport.build_transport_decision_rows(rows, options=_options())
    )["com"]

    assert decision["recommendation"] == "invalid_insufficient_points"
    assert decision["can_transport"] is False


def test_phi_wrap_in_transport_error() -> None:
    delta = transport._shape_delta(179.0, -179.0, coordinate="com_phi_deg")

    assert math.isclose(abs(delta), 2.0)


def test_no_sparse_or_direct_fallback_allowed() -> None:
    rows = _decision_rows(
        comparison="transport_vs_full_recompute",
        metric="ray_cloud_com",
        parameter="corto_detector",
        moved=True,
        fallback=True,
    )

    decision = _decision_by_scope(
        transport.build_transport_decision_rows(rows, options=_options())
    )["com"]

    assert decision["recommendation"] == "invalid_fallback_used"
    assert decision["can_transport"] is False


def test_roi_membership_change_is_reported_separately() -> None:
    same_roi = _decision_rows(
        comparison="transport_vs_same_roi_full",
        metric="image_roi_com",
        parameter="corto_detector",
        moved=True,
        mismatch=False,
    )
    new_roi = _decision_rows(
        comparison="transport_vs_new_roi_full",
        metric="image_roi_com",
        parameter="corto_detector",
        moved=True,
        mismatch=True,
    )

    decisions = transport.build_transport_decision_rows(
        [*same_roi, *new_roi],
        options=_options(),
    )
    by_comparison = {(row["comparison"], row["decision_scope"]): row for row in decisions}

    assert (
        by_comparison[("transport_vs_same_roi_full", "com")]["recommendation"] == "transport_points"
    )
    assert (
        by_comparison[("transport_vs_new_roi_full", "com")]["recommendation"]
        == "full_recompute_required"
    )


def test_com_decision_can_pass_when_shape_baseline_mismatches() -> None:
    rows = _decision_rows(
        comparison="transport_vs_same_roi_full",
        metric="image_roi_com",
        parameter="corto_detector",
        moved=True,
        mismatch=False,
    )
    for row in rows:
        if str(row["coordinate"]) in transport.SHAPE_DECISION_COORDINATES:
            row["baseline_transport_mismatch"] = True
            row["pass"] = False
            row["status"] = "baseline_transport_mismatch"

    decisions = _decision_by_scope(
        transport.build_transport_decision_rows(rows, options=_options())
    )

    assert decisions["com"]["recommendation"] == "transport_points"
    assert decisions["com"]["can_transport"] is True
    assert decisions["shape"]["recommendation"] == "invalid_baseline_mismatch"
    assert decisions["shape"]["can_transport"] is False
