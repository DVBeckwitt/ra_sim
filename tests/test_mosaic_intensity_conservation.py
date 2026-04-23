from __future__ import annotations

import csv
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ra_sim.simulation import diffraction
from ra_sim.simulation.mosaic_normalization import (
    MosaicNormalizationResult,
    apply_reflection_intensity_to_mosaic,
    build_unscaled_mosaic_for_hkl,
    build_unscaled_mosaic_for_request_hkl,
    integrate_mosaic_mass,
    summarize_mosaic_mass,
)
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)


BI2SE3_A = 4.143
BI2SE3_C = 28.636
CU_KA1_WAVELENGTH = 1.540592865402115
BRANCHING_CANDIDATES = ((1, 0, 1), (0, 1, 5), (1, 1, 0), (2, 1, 10))
ARTIFACT_DIR = Path("artifacts/mosaic_normalization")
REQUIRED_SUMMARY_KEYS = {
    "hkl",
    "branch_count",
    "total_mass",
    "per_branch_mass",
    "sample_count",
    "negative_count",
    "nan_count",
    "integration_mode",
}


def _validation_request(*, solve_q_steps: int = 8192) -> SimulationRequest:
    geometry = DetectorGeometry(
        image_size=32,
        av=BI2SE3_A,
        cv=BI2SE3_C,
        lambda_angstrom=CU_KA1_WAVELENGTH,
        distance_m=0.1,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        center=np.array([16.0, 16.0], dtype=np.float64),
        theta_initial_deg=0.0,
        cor_angle_deg=0.0,
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    beam = BeamSamples(
        beam_x_array=np.zeros(1, dtype=np.float64),
        beam_y_array=np.zeros(1, dtype=np.float64),
        theta_array=np.zeros(1, dtype=np.float64),
        phi_array=np.zeros(1, dtype=np.float64),
        wavelength_array=np.array([CU_KA1_WAVELENGTH], dtype=np.float64),
    )
    mosaic = MosaicParams(
        sigma_mosaic_deg=0.5,
        gamma_mosaic_deg=0.4,
        eta=0.2,
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=1.0e-5,
    )
    return SimulationRequest(
        miller=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty(0, dtype=np.float64),
        geometry=geometry,
        beam=beam,
        mosaic=mosaic,
        debye_waller=DebyeWallerParams(x=0.0, y=0.0),
        n2=1.0 + 0.0j,
        image_buffer=None,
    )


def _artifact_name(prefix: str, result: MosaicNormalizationResult) -> str:
    hkl = "_".join(str(int(v)) if float(v).is_integer() else f"{v:g}" for v in result.hkl)
    return f"{prefix}_{hkl}"


def _sample_rows(result: MosaicNormalizationResult, limit: int = 50) -> list[dict[str, object]]:
    weights = result.scaled_weights if result.scaled_weights is not None else result.weights
    rows = []
    for idx in range(min(limit, result.qx.size)):
        rows.append(
            {
                "qx": float(result.qx[idx]),
                "qy": float(result.qy[idx]),
                "qz": float(result.qz[idx]),
                "weight_or_density": float(weights[idx]),
                "area_if_present": None,
                "branch_id": int(result.branch_id[idx]),
            }
        )
    return rows


def _write_debug_artifact(
    name: str,
    result: MosaicNormalizationResult,
    *,
    force: bool = False,
) -> None:
    if not force and os.environ.get("RA_SIM_WRITE_MOSAIC_NORMALIZATION_DEBUG") != "1":
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize_mosaic_mass(result),
        "first_50_sample_rows": _sample_rows(result),
    }
    base = ARTIFACT_DIR / _artifact_name(name, result)
    base.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("qx", "qy", "qz", "weight_or_density", "area_if_present", "branch_id"),
        )
        writer.writeheader()
        writer.writerows(_sample_rows(result, limit=result.qx.size))


def _assert_conserved(
    name: str,
    result: MosaicNormalizationResult,
    *,
    write_on_failure: bool = True,
) -> dict[str, object]:
    try:
        summary = summarize_mosaic_mass(result)
        total_mass = integrate_mosaic_mass(result)
        assert result.status == "ok"
        assert np.isfinite(result.raw_mass)
        assert abs(total_mass - 1.0) < 1.0e-3
        assert summary["total_mass"] == pytest.approx(1.0, abs=1.0e-3)
        assert summary["nan_count"] == 0
        assert summary["negative_count"] == 0
    except AssertionError:
        if write_on_failure:
            _write_debug_artifact(name, result, force=True)
        raise
    _write_debug_artifact(name, result)
    return summary


def _branching_result() -> tuple[
    tuple[int, int, int], MosaicNormalizationResult, dict[str, object]
]:
    request = _validation_request()
    for hkl in BRANCHING_CANDIDATES:
        result = build_unscaled_mosaic_for_request_hkl(request, hkl)
        summary = _assert_conserved(f"branching_candidate_{hkl}", result)
        if summary["branch_count"] >= 2:
            return hkl, result, summary
    raise AssertionError("No configured branching HKL found among candidate reflections.")


def test_solve_q_column_3_is_integrated_quadrature_weight() -> None:
    uniform_source = inspect.getsource(diffraction._solve_q_uniform_full_circle.py_func)
    adaptive_source = inspect.getsource(diffraction._interval_mass_error.py_func)

    assert "all_int = sigma_arr * ds" in uniform_source
    assert "mass = circle_r * dphi * (f_a + 4.0 * f_m + f_b) / 6.0" in adaptive_source


@pytest.mark.parametrize("hkl", [(0, 0, 3), (0, 0, 12)])
def test_mosaic_mass_conserved_for_00l(hkl: tuple[int, int, int]) -> None:
    result = build_unscaled_mosaic_for_request_hkl(_validation_request(), hkl)
    summary = _assert_conserved(f"00l_{hkl}", result)

    assert summary["branch_count"] == 1
    assert sum(summary["per_branch_mass"]) == pytest.approx(integrate_mosaic_mass(result))


def test_mosaic_mass_conserved_for_branching_hkl() -> None:
    _hkl, result, summary = _branching_result()
    per_branch_mass = np.asarray(summary["per_branch_mass"], dtype=np.float64)
    unique_branches = np.unique(result.branch_id)

    assert summary["branch_count"] >= 2
    assert all(np.any(result.branch_id == branch) for branch in unique_branches)
    assert np.all(np.isfinite(per_branch_mass))
    assert np.all(per_branch_mass >= 0.0)
    assert float(np.sum(per_branch_mass)) == pytest.approx(integrate_mosaic_mass(result), abs=1e-6)
    assert not np.allclose(per_branch_mass, np.ones_like(per_branch_mass), atol=1.0e-3)


def test_scaled_reflection_intensity_conserved_before_detector_projection() -> None:
    reflection_intensity = 123.456
    request = _validation_request()
    results = [
        build_unscaled_mosaic_for_request_hkl(request, (0, 0, 3)),
        _branching_result()[1],
    ]

    for result in results:
        _assert_conserved("scaled_unscaled_input", result)
        scaled = apply_reflection_intensity_to_mosaic(result, reflection_intensity)
        summary = summarize_mosaic_mass(scaled)
        per_branch_mass = np.asarray(summary["per_branch_mass"], dtype=np.float64)

        assert integrate_mosaic_mass(scaled) == pytest.approx(reflection_intensity, rel=1.0e-3)
        assert float(np.sum(per_branch_mass)) == pytest.approx(
            integrate_mosaic_mass(scaled),
            abs=1.0e-6,
        )


def test_mosaic_weights_are_not_normalized_per_branch() -> None:
    _hkl, result, summary = _branching_result()
    per_branch_mass = np.asarray(summary["per_branch_mass"], dtype=np.float64)

    assert integrate_mosaic_mass(result) == pytest.approx(1.0, abs=1.0e-3)
    if summary["branch_count"] > 1:
        assert not np.allclose(per_branch_mass, np.ones_like(per_branch_mass), atol=1.0e-3)
    assert float(np.sum(per_branch_mass)) == pytest.approx(1.0, abs=1.0e-6)


def test_mosaic_mass_debug_summary_is_json_serializable() -> None:
    summaries = [
        summarize_mosaic_mass(
            build_unscaled_mosaic_for_hkl((0, 0, 3), request=_validation_request())
        ),
        summarize_mosaic_mass(_branching_result()[1]),
    ]

    for summary in summaries:
        assert REQUIRED_SUMMARY_KEYS.issubset(summary)
        json.dumps(summary, allow_nan=False)


def test_branch_normalized_negative_control_fails_conservation() -> None:
    _hkl, result, _summary = _branching_result()
    qx_parts = []
    qy_parts = []
    qz_parts = []
    weight_parts = []
    branch_parts = []

    for branch in np.unique(result.branch_id):
        mask = result.branch_id == branch
        branch_weights = result.weights[mask]
        branch_mass = float(np.sum(branch_weights))
        assert branch_mass > 0.0
        qx_parts.append(result.qx[mask])
        qy_parts.append(result.qy[mask])
        qz_parts.append(result.qz[mask])
        weight_parts.append(branch_weights / branch_mass)
        branch_parts.append(result.branch_id[mask])

    per_branch_normalized = replace(
        result,
        qx=np.concatenate(qx_parts),
        qy=np.concatenate(qy_parts),
        qz=np.concatenate(qz_parts),
        weights=np.concatenate(weight_parts),
        branch_id=np.concatenate(branch_parts),
    )

    assert integrate_mosaic_mass(per_branch_normalized) > 1.0
    with pytest.raises(AssertionError):
        _assert_conserved(
            "negative_control_per_branch_normalized",
            per_branch_normalized,
            write_on_failure=False,
        )
