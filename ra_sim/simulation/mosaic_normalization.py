"""Validation helpers for pre-detector mosaic mass conservation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, exp, pi, sin, sqrt
from typing import Literal

import numpy as np

from ra_sim.simulation.diffraction import DEFAULT_SOLVE_Q_STEPS
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)


MosaicMassStatus = Literal[
    "ok",
    "invalid_mass_zero",
    "invalid_mass_negative",
    "invalid_mass_nan",
    "invalid_mass_inf",
]
MosaicIntegrationMode = Literal["weights", "density_times_area"]


@dataclass(frozen=True)
class MosaicNormalizationResult:
    hkl: tuple[float, float, float]
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    weights: np.ndarray
    branch_id: np.ndarray
    raw_mass: float
    status: MosaicMassStatus
    integration_mode: MosaicIntegrationMode = "weights"
    scaled_weights: np.ndarray | None = None
    solve_q_status: int = 0


def _empty_result(
    hkl: tuple[float, float, float],
    *,
    raw_mass: float,
    status: MosaicMassStatus,
    solve_q_status: int = 0,
) -> MosaicNormalizationResult:
    empty_f = np.empty(0, dtype=np.float64)
    empty_i = np.empty(0, dtype=np.int64)
    return MosaicNormalizationResult(
        hkl=hkl,
        qx=empty_f.copy(),
        qy=empty_f.copy(),
        qz=empty_f.copy(),
        weights=empty_f.copy(),
        branch_id=empty_i,
        raw_mass=float(raw_mass),
        status=status,
        solve_q_status=int(solve_q_status),
    )


def _mass_status(weights: np.ndarray, raw_mass: float) -> MosaicMassStatus:
    if np.any(np.isinf(weights)) or np.isinf(raw_mass):
        return "invalid_mass_inf"
    if np.any(np.isnan(weights)) or np.isnan(raw_mass):
        return "invalid_mass_nan"
    if raw_mass < 0.0:
        return "invalid_mass_negative"
    if raw_mass == 0.0:
        return "invalid_mass_zero"
    return "ok"


def _hkl_tuple(hkl) -> tuple[float, float, float]:
    arr = np.asarray(hkl, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError("hkl must contain exactly three values")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _g_vector(hkl: tuple[float, float, float], av: float, cv: float) -> np.ndarray:
    h_val, k_val, l_val = hkl
    gz = 2.0 * pi * (l_val / float(cv))
    gr = 4.0 * pi / float(av) * sqrt((h_val * h_val + h_val * k_val + k_val * k_val) / 3.0)
    return np.array([0.0, gr, gz], dtype=np.float64)


def _wrap_to_pi(value: np.ndarray | float) -> np.ndarray | float:
    return (value + pi) % (2.0 * pi) - pi


def _mosaic_density(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    g_vec: np.ndarray,
    sigma: float,
    gamma_pv: float,
    eta_pv: float,
) -> np.ndarray:
    g_mag = float(np.linalg.norm(g_vec))
    if g_mag < 1.0e-14:
        return np.zeros_like(qx, dtype=np.float64)

    sigma_eff = max(float(sigma), 1.0e-12)
    gamma_eff = max(float(gamma_pv), 1.0e-12)

    qr = np.sqrt(qx * qx + qy * qy)
    gr = sqrt(g_vec[0] * g_vec[0] + g_vec[1] * g_vec[1])
    theta0 = atan2(g_vec[2], gr)
    theta = np.arctan2(qz, qr)
    dtheta = _wrap_to_pi(theta - theta0)

    gauss = (1.0 / (sigma_eff * sqrt(2.0 * pi))) * np.exp(
        -0.5 * (dtheta / sigma_eff) * (dtheta / sigma_eff)
    )
    lorentz = (1.0 / (pi * gamma_eff)) / (1.0 + (dtheta / gamma_eff) ** 2)
    omega = (1.0 - eta_pv) * gauss + eta_pv * lorentz
    return omega / (2.0 * pi * g_mag * g_mag)


def _incident_wavevector_from_request(
    request: SimulationRequest,
    sample_index: int,
) -> tuple[np.ndarray, float]:
    beam = request.beam
    wavelength = float(np.asarray(beam.wavelength_array, dtype=np.float64)[sample_index])
    if (not np.isfinite(wavelength)) or wavelength <= 0.0:
        return np.zeros(3, dtype=np.float64), 0.0

    eps2_real = float(np.real(request.n2 * request.n2))
    if not np.isfinite(eps2_real) or eps2_real <= 0.0:
        eps2_real = 1.0

    k_scat = (2.0 * pi / wavelength) * sqrt(eps2_real)
    dtheta = float(np.asarray(beam.theta_array, dtype=np.float64)[sample_index])
    dphi = float(np.asarray(beam.phi_array, dtype=np.float64)[sample_index])
    k_in = np.array(
        [
            k_scat * cos(dtheta) * sin(dphi),
            k_scat * cos(dtheta) * cos(dphi),
            -k_scat * sin(dtheta),
        ],
        dtype=np.float64,
    )
    return k_in, k_scat


def _solve_mosaic_ring_uniform(
    k_in_crystal: np.ndarray,
    k_scat: float,
    g_vec: np.ndarray,
    sigma: float,
    gamma_pv: float,
    eta_pv: float,
    n_steps: int,
) -> tuple[np.ndarray, int]:
    if n_steps <= 0:
        return np.empty((0, 4), dtype=np.float64), 0
    if k_scat <= 0.0 or not np.isfinite(k_scat):
        return np.empty((0, 4), dtype=np.float64), -2

    g_sq = float(np.dot(g_vec, g_vec))
    if g_sq < 1.0e-14:
        return np.empty((0, 4), dtype=np.float64), -1

    axis = -np.asarray(k_in_crystal, dtype=np.float64)
    axis_sq = float(np.dot(axis, axis))
    if axis_sq < 1.0e-14:
        return np.empty((0, 4), dtype=np.float64), -2
    axis_len = sqrt(axis_sq)

    c_val = (g_sq + axis_sq - k_scat * k_scat) / (2.0 * axis_len)
    circle_r_sq = g_sq - c_val * c_val
    if circle_r_sq < 0.0:
        return np.empty((0, 4), dtype=np.float64), -3
    circle_r = sqrt(max(circle_r_sq, 0.0))

    axis_hat = axis / axis_len
    origin = c_val * axis_hat
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, axis_hat))) > 0.9999:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = seed - float(np.dot(seed, axis_hat)) * axis_hat
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm < 1.0e-14:
        return np.empty((0, 4), dtype=np.float64), -4
    e1 /= e1_norm
    e2 = np.cross(axis_hat, e1)
    e2_norm = float(np.linalg.norm(e2))
    if e2_norm < 1.0e-14:
        return np.empty((0, 4), dtype=np.float64), -5
    e2 /= e2_norm

    phi = (2.0 * pi / float(n_steps)) * np.arange(n_steps, dtype=np.float64)
    qx = origin[0] + circle_r * (np.cos(phi) * e1[0] + np.sin(phi) * e2[0])
    qy = origin[1] + circle_r * (np.cos(phi) * e1[1] + np.sin(phi) * e2[1])
    qz = origin[2] + circle_r * (np.cos(phi) * e1[2] + np.sin(phi) * e2[2])

    density = _mosaic_density(qx, qy, qz, g_vec, sigma, gamma_pv, eta_pv)
    weights = density * circle_r * (2.0 * pi / float(n_steps))
    valid = np.isfinite(weights) & (weights > exp(-100.0))
    return np.column_stack((qx[valid], qy[valid], qz[valid], weights[valid])), 0


def _branch_ids_for_current_frame(
    hkl: tuple[float, float, float],
    qx: np.ndarray,
    *,
    eps: float,
) -> np.ndarray:
    h_val, k_val, _l_val = hkl
    branch_id = np.zeros(qx.shape[0], dtype=np.int64)
    if h_val == 0.0 and k_val == 0.0:
        return branch_id

    # Current diffraction frame builds reciprocal vectors as G_vec=[0, Gr, Gz].
    # With that frame, non-specular detector-side branches split by qx sign.
    eps_abs = abs(float(eps))
    branch_id[qx > eps_abs] = 1
    branch_id[np.abs(qx) < eps_abs] = 0
    return branch_id


def _request_from_parts(
    *,
    geometry: DetectorGeometry,
    beam: BeamSamples,
    mosaic: MosaicParams,
    n2: complex,
    thickness: float,
    optics_mode: int | None,
) -> SimulationRequest:
    return SimulationRequest(
        miller=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty(0, dtype=np.float64),
        geometry=geometry,
        beam=beam,
        mosaic=mosaic,
        debye_waller=DebyeWallerParams(x=0.0, y=0.0),
        n2=n2,
        image_buffer=None,
        thickness=float(thickness),
        optics_mode=optics_mode,
    )


def build_unscaled_mosaic_for_request_hkl(
    request: SimulationRequest,
    hkl,
    *,
    sample_index: int | None = None,
    branch_eps: float = 1.0e-12,
) -> MosaicNormalizationResult:
    """Build normalized pre-detector mosaic weights for one request/HKL.

    The output is unscaled by reflection intensity. ``raw_mass`` preserves the
    pre-normalization quadrature sum. ``solve_q`` column 3 is an integrated
    quadrature weight: uniform paths store ``density * arc_length`` and the
    adaptive path stores Simpson interval mass, so integration is ``sum(weights)``.
    """

    hkl_t = _hkl_tuple(hkl)
    beam = request.beam
    beam_arrays = (
        np.asarray(beam.beam_x_array, dtype=np.float64).reshape(-1),
        np.asarray(beam.beam_y_array, dtype=np.float64).reshape(-1),
        np.asarray(beam.theta_array, dtype=np.float64).reshape(-1),
        np.asarray(beam.phi_array, dtype=np.float64).reshape(-1),
        np.asarray(beam.wavelength_array, dtype=np.float64).reshape(-1),
    )
    sample_count = int(beam_arrays[0].size)
    if sample_count == 0 or any(arr.size != sample_count for arr in beam_arrays[1:]):
        return _empty_result(hkl_t, raw_mass=0.0, status="invalid_mass_zero")

    if sample_index is None:
        angle_metric = beam_arrays[2] * beam_arrays[2] + beam_arrays[3] * beam_arrays[3]
        beam_metric = beam_arrays[0] * beam_arrays[0] + beam_arrays[1] * beam_arrays[1]
        order = np.lexsort((beam_metric, angle_metric))
        sample_index_i = int(order[0])
    else:
        sample_index_i = int(sample_index)
        if sample_index_i < 0 or sample_index_i >= sample_count:
            raise IndexError("sample_index is outside the request beam sample range")

    geometry = request.geometry
    mosaic = request.mosaic
    g_vec = _g_vector(hkl_t, geometry.av, geometry.cv)
    k_in_crystal, k_scat = _incident_wavevector_from_request(request, sample_index_i)
    q_points, solve_status = _solve_mosaic_ring_uniform(
        k_in_crystal,
        k_scat,
        g_vec,
        float(mosaic.sigma_mosaic_deg) * (pi / 180.0),
        float(mosaic.gamma_mosaic_deg) * (pi / 180.0),
        float(mosaic.eta),
        int(mosaic.solve_q_steps or DEFAULT_SOLVE_Q_STEPS),
    )
    q_arr = np.asarray(q_points, dtype=np.float64)
    if q_arr.size == 0:
        return _empty_result(
            hkl_t,
            raw_mass=0.0,
            status="invalid_mass_zero",
            solve_q_status=int(solve_status),
        )

    raw_weights = np.asarray(q_arr[:, 3], dtype=np.float64).reshape(-1)
    raw_mass = float(np.sum(raw_weights))
    status = _mass_status(raw_weights, raw_mass)
    if status != "ok":
        return MosaicNormalizationResult(
            hkl=hkl_t,
            qx=np.asarray(q_arr[:, 0], dtype=np.float64).copy(),
            qy=np.asarray(q_arr[:, 1], dtype=np.float64).copy(),
            qz=np.asarray(q_arr[:, 2], dtype=np.float64).copy(),
            weights=raw_weights.copy(),
            branch_id=_branch_ids_for_current_frame(hkl_t, q_arr[:, 0], eps=branch_eps),
            raw_mass=raw_mass,
            status=status,
            solve_q_status=int(solve_status),
        )

    weights = raw_weights / raw_mass
    return MosaicNormalizationResult(
        hkl=hkl_t,
        qx=np.asarray(q_arr[:, 0], dtype=np.float64).copy(),
        qy=np.asarray(q_arr[:, 1], dtype=np.float64).copy(),
        qz=np.asarray(q_arr[:, 2], dtype=np.float64).copy(),
        weights=np.asarray(weights, dtype=np.float64),
        branch_id=_branch_ids_for_current_frame(hkl_t, q_arr[:, 0], eps=branch_eps),
        raw_mass=raw_mass,
        status="ok",
        solve_q_status=int(solve_status),
    )


def build_unscaled_mosaic_for_hkl(
    hkl,
    *,
    request: SimulationRequest | None = None,
    geometry: DetectorGeometry | None = None,
    beam: BeamSamples | None = None,
    mosaic: MosaicParams | None = None,
    n2: complex = 1.0 + 0.0j,
    thickness: float = 0.0,
    optics_mode: int | None = None,
    sample_index: int | None = None,
    branch_eps: float = 1.0e-12,
) -> MosaicNormalizationResult:
    """Build normalized mosaic weights with reflection intensity still unscaled."""

    if request is None:
        if geometry is None or beam is None or mosaic is None:
            raise ValueError("request or geometry, beam, and mosaic must be provided")
        request = _request_from_parts(
            geometry=geometry,
            beam=beam,
            mosaic=mosaic,
            n2=n2,
            thickness=thickness,
            optics_mode=optics_mode,
        )
    return build_unscaled_mosaic_for_request_hkl(
        request,
        hkl,
        sample_index=sample_index,
        branch_eps=branch_eps,
    )


def _active_weights(mosaic_result: MosaicNormalizationResult) -> np.ndarray:
    if mosaic_result.scaled_weights is not None:
        return np.asarray(mosaic_result.scaled_weights, dtype=np.float64)
    return np.asarray(mosaic_result.weights, dtype=np.float64)


def integrate_mosaic_mass(mosaic_result: MosaicNormalizationResult) -> float:
    """Integrate current mosaic mass without applying an area factor twice."""

    if mosaic_result.integration_mode != "weights":
        raise ValueError("Only integrated quadrature weights are currently supported")
    return float(np.sum(_active_weights(mosaic_result)))


def _json_float(value: float) -> float | None:
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def summarize_mosaic_mass(mosaic_result: MosaicNormalizationResult) -> dict[str, object]:
    values = _active_weights(mosaic_result)
    branch_id = np.asarray(mosaic_result.branch_id, dtype=np.int64)
    finite_values = values[np.isfinite(values)]
    unique_branches = np.unique(branch_id) if branch_id.size else np.empty(0, dtype=np.int64)
    per_branch_mass = [
        _json_float(float(np.sum(values[branch_id == branch]))) for branch in unique_branches
    ]

    min_value = None
    max_value = None
    if finite_values.size:
        min_value = _json_float(float(np.min(finite_values)))
        max_value = _json_float(float(np.max(finite_values)))

    return {
        "hkl": [float(v) for v in mosaic_result.hkl],
        "branch_count": int(unique_branches.size),
        "total_mass": _json_float(integrate_mosaic_mass(mosaic_result)),
        "per_branch_mass": per_branch_mass,
        "sample_count": int(values.size),
        "min_weight_or_density": min_value,
        "max_weight_or_density": max_value,
        "negative_count": int(np.sum(values < 0.0)),
        "nan_count": int(np.sum(np.isnan(values))),
        "integration_mode": mosaic_result.integration_mode,
        "raw_mass": _json_float(mosaic_result.raw_mass),
        "status": mosaic_result.status,
        "solve_q_status": int(mosaic_result.solve_q_status),
    }


def apply_reflection_intensity_to_mosaic(
    mosaic_result: MosaicNormalizationResult,
    reflection_intensity: float,
) -> MosaicNormalizationResult:
    """Scale normalized weights by reflection strength without renormalizing."""

    scaled = np.asarray(mosaic_result.weights, dtype=np.float64) * float(reflection_intensity)
    return replace(mosaic_result, scaled_weights=scaled)
