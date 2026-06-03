#!/usr/bin/env python3
"""
Plot the Fig. 2 near-Qz=0 diagnostic from inside the RA-SIM repo.

This version explicitly includes the *pure optical Parratt reflectivity* in
addition to the HT-derived curves.

Curves shown:

    Born-scaled HT p=0:
        (Qc / 2 Qz)^4 * S_HT0(L)

    Fresnel-corrected HT p=0:
        R_F(Qz) * S_HT0(L)

    Parratt-envelope HT p=0:
        R_Parratt(Qz) * S_HT0(L)

    Pure Parratt reflectivity:
        R_Parratt(Qz)

where

    S_HT0(L) = I_HT(L; p=0) / I_HT(0; p=0).

The pure Parratt curve contains no HT structure term and is independent of p.

Run from the repo root, for example:

    python scripts/plot_fig2_born_scaled_ht_near_zero.py --show

Fast redraw from an existing model-curve CSV:

    python scripts/plot_fig2_born_scaled_ht_near_zero.py \
        --from-csv artifacts/l_axis_reflectivity_diagnostics/model_curves_L_Qz.csv \
        --show

Direct recomputation from repo HT machinery:

    python scripts/plot_fig2_born_scaled_ht_near_zero.py \
        --ht-bandwidth-mode interpolate-only \
        --bandwidth-fwhm 0.05 \
        --n-wavelength-samples 241 \
        --show

All important numerical values are centralized in DEFAULTS and exposed as CLI options.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# User-adjustable defaults
# =============================================================================


@dataclass(frozen=True)
class Defaults:
    # Material and optical values.
    material: str = "Bi2Se3"
    a_angstrom: float = 4.1380
    c_angstrom: float = 28.6360
    qc_inv_angstrom: float = 0.0517
    film_density_e_per_a3: float = 1.8870
    thickness_nm: float = 50.0
    substrate: str = "SiO2"
    substrate_qc_inv_angstrom: float = 0.0305
    substrate_density_e_per_a3: float = 0.6567
    top_roughness_angstrom: float = 0.0
    bottom_roughness_angstrom: float = 0.0

    # X-ray / broadening values.
    lambda0_angstrom: float = 1.5418
    bandwidth_fwhm: float = 0.005
    n_wavelength_samples: int = 20

    # Angular divergence is treated as a Gaussian FWHM in the specular angle.
    divergence_fwhm_deg: float = 0.5
    n_divergence_samples: int = 50
    ht_structure_scale: float = 1.0
    ht_over_q2_scale: float = 1.0

    # HT finite-stack values.
    p_user: float = 0.0
    ht_bandwidth_mode: str = "interpolate-only"  # "interpolate-only" or "resimulate"
    L_step_dense: float = 0.001
    L_max_padded: float = 1.0

    # Plot range.
    L_min: float = 0.0
    L_max: float = 0.50
    L_step: float = 0.0005

    # Avoid division by zero for Born asymptote. This is only a plotting floor.
    qz_floor_for_born: float = 1.0e-8

    # Output.
    out_dir: str = "artifacts/fig2_born_scaled_ht_near_zero"
    out_name: str = "fig2_born_scaled_ht_near_zero_with_pure_parratt.png"
    dpi: int = 200


DEFAULTS = Defaults()

DEFAULT_Y_MIN_LOG10 = -12.0
DEFAULT_Y_MAX_LOG10 = 30.0
DEFAULT_BRAGG_MARKER_L = (3.0, 6.0, 9.0)
DEFAULT_AUTOFIT_L_MIN = 1.0
DEFAULT_AUTOFIT_L_MAX = 10.0
DEFAULT_AUTOFIT_PEAK_CENTERS = (3.0, 6.0, 9.0)
DEFAULT_AUTOFIT_PEAK_HALF_WIDTH = 0.35
DEFAULT_STITCH_X1_Q_OVER_QC = 2.0
DEFAULT_STITCH_X2_Q_OVER_QC = 5.0
DEFAULT_STITCH_FIT_X_MIN_Q_OVER_QC = 5.0
DEFAULT_STITCH_FIT_X_MAX_Q_OVER_QC = 10.0
DEFAULT_STITCH_EXCLUDE_CENTERS_L = (3.0, 6.0, 9.0)
DEFAULT_STITCH_EXCLUDE_HALF_WIDTH_L = 0.35

HT_STRUCTURE_LABEL = "HT structure term p=0"
HT_OVER_Q2_LABEL = "HT structure term p=0 over Qz^2"
STITCH_UNSCALED_HT_OVER_Q2_LABEL = "Unscaled HT/Qz^2"
STITCH_SCALED_HT_OVER_Q2_LABEL = "Scaled HT/Qz^2"
STITCHED_PARRATT_HT_OVER_Q2_LABEL = "Stitched Parratt + HT/Qz^2"
STITCH_WEIGHT_LABEL = "Stitch weight"
BORN_HT_LABEL = "Born-scaled HT p=0"
FRESNEL_HT_LABEL = "Fresnel-corrected HT p=0"
PARRATT_ENV_HT_LABEL = "Parratt-envelope HT p=0"
PURE_PARRATT_LABEL = "Pure Parratt reflectivity"
FIG2_CURVE_ORDER = (
    HT_STRUCTURE_LABEL,
    HT_OVER_Q2_LABEL,
    BORN_HT_LABEL,
    FRESNEL_HT_LABEL,
    PARRATT_ENV_HT_LABEL,
    PURE_PARRATT_LABEL,
)
STITCH_CURVE_ORDER = (
    PURE_PARRATT_LABEL,
    STITCH_SCALED_HT_OVER_Q2_LABEL,
    STITCHED_PARRATT_HT_OVER_Q2_LABEL,
)
STITCH_DIAGNOSTIC_CURVE_ORDER = (
    PURE_PARRATT_LABEL,
    STITCH_UNSCALED_HT_OVER_Q2_LABEL,
    STITCH_SCALED_HT_OVER_Q2_LABEL,
    STITCHED_PARRATT_HT_OVER_Q2_LABEL,
    STITCH_WEIGHT_LABEL,
)
DEFAULT_GUI_VISIBLE_CURVE_LABELS = (
    HT_STRUCTURE_LABEL,
    HT_OVER_Q2_LABEL,
    PURE_PARRATT_LABEL,
)
GUI_CURVE_ORDER = (
    HT_STRUCTURE_LABEL,
    HT_OVER_Q2_LABEL,
    PURE_PARRATT_LABEL,
    STITCHED_PARRATT_HT_OVER_Q2_LABEL,
)
DEFAULT_GUI_CURVE_VISIBILITY = {
    label: label in DEFAULT_GUI_VISIBLE_CURVE_LABELS for label in GUI_CURVE_ORDER
}

FIG2_CURVE_DISPLAY_LABELS = {
    HT_STRUCTURE_LABEL: r"$S_{\mathrm{HT},0}(L)$",
    HT_OVER_Q2_LABEL: r"$S_{\mathrm{HT},0}(L) / Q_z^2$",
    STITCH_UNSCALED_HT_OVER_Q2_LABEL: r"Unscaled $S_{\mathrm{HT},0}(L) / Q_z^2$",
    STITCH_SCALED_HT_OVER_Q2_LABEL: r"Scaled $S_{\mathrm{HT},0}(L) / Q_z^2$",
    STITCHED_PARRATT_HT_OVER_Q2_LABEL: r"Stitched $R_{\mathrm{Parratt}} + S_{\mathrm{HT},0}/Q_z^2$",
    STITCH_WEIGHT_LABEL: "Stitch weight",
    BORN_HT_LABEL: r"$\left(Q_c / 2Q_z\right)^4 S_{\mathrm{HT},0}(L)$",
    FRESNEL_HT_LABEL: r"$R_F(Q_z) S_{\mathrm{HT},0}(L)$",
    PARRATT_ENV_HT_LABEL: r"$R_{\mathrm{Parratt}}(Q_z) S_{\mathrm{HT},0}(L)$",
    PURE_PARRATT_LABEL: r"$R_{\mathrm{Parratt}}(Q_z)$",
}

# Okabe-Ito colorblind-friendly palette plus line styles.
FIG2_CURVE_STYLE = {
    HT_STRUCTURE_LABEL: dict(color="#CC79A7", linestyle="-", linewidth=1.6),
    HT_OVER_Q2_LABEL: dict(color="#56B4E9", linestyle="--", linewidth=1.7),
    STITCH_UNSCALED_HT_OVER_Q2_LABEL: dict(color="#56B4E9", linestyle=":", linewidth=1.3),
    STITCH_SCALED_HT_OVER_Q2_LABEL: dict(color="#0072B2", linestyle="--", linewidth=1.8),
    STITCHED_PARRATT_HT_OVER_Q2_LABEL: dict(color="#009E73", linestyle="-", linewidth=2.1),
    STITCH_WEIGHT_LABEL: dict(color="#E69F00", linestyle="-.", linewidth=1.4),
    BORN_HT_LABEL: dict(color="#0072B2", linestyle="-", linewidth=1.8),
    FRESNEL_HT_LABEL: dict(color="#D55E00", linestyle="--", linewidth=1.8),
    PARRATT_ENV_HT_LABEL: dict(color="#009E73", linestyle="-.", linewidth=1.8),
    PURE_PARRATT_LABEL: dict(color="#000000", linestyle=":", linewidth=2.0),
}


# Classical electron radius in angstrom. Used by Qc^2 = 16*pi*r_e*rho.
CLASSICAL_ELECTRON_RADIUS_ANGSTROM = 2.8179403262e-5


def density_e_per_a3_from_qc(qc_inv_angstrom: float) -> float:
    qc = float(qc_inv_angstrom)
    if qc < 0.0:
        raise ValueError("Critical wavevector must be non-negative.")
    return qc**2 / (16.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_ANGSTROM)


def qc_from_density_e_per_a3(density_e_per_a3: float) -> float:
    density = float(density_e_per_a3)
    if density < 0.0:
        raise ValueError("Electron density must be non-negative.")
    return float(np.sqrt(16.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_ANGSTROM * density))


def parse_float_list(text: str | None) -> tuple[float, ...]:
    if text is None or str(text).strip() == "":
        return ()
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip())


def surface_cell_area_hex(a_angstrom: float) -> float:
    return 0.5 * np.sqrt(3.0) * float(a_angstrom) ** 2


def miceli_cell_scale(a_angstrom: float) -> float:
    a_cell = surface_cell_area_hex(a_angstrom)
    return 16.0 * np.pi**2 * CLASSICAL_ELECTRON_RADIUS_ANGSTROM**2 / a_cell**2


def bragg_exclusion_mask(
    L: np.ndarray,
    centers: tuple[float, ...] = DEFAULT_STITCH_EXCLUDE_CENTERS_L,
    half_width: float = DEFAULT_STITCH_EXCLUDE_HALF_WIDTH_L,
) -> np.ndarray:
    l_values = np.asarray(L, dtype=float)
    mask = np.ones_like(l_values, dtype=bool)
    for center in centers:
        mask &= np.abs(l_values - float(center)) > float(half_width)
    return mask


def fit_log_median_scale(model: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    model_values = np.asarray(model, dtype=float)
    target_values = np.asarray(target, dtype=float)
    mask_values = np.asarray(mask, dtype=bool)
    valid = (
        mask_values
        & np.isfinite(model_values)
        & np.isfinite(target_values)
        & (model_values > 0.0)
        & (target_values > 0.0)
    )
    if np.count_nonzero(valid) < 4:
        raise ValueError("Not enough valid points for scale estimate.")
    return float(np.exp(np.median(np.log(target_values[valid]) - np.log(model_values[valid]))))


def smooth_stitch_weight(
    qz_over_qc: np.ndarray,
    x1: float = DEFAULT_STITCH_X1_Q_OVER_QC,
    x2: float = DEFAULT_STITCH_X2_Q_OVER_QC,
) -> np.ndarray:
    left = float(x1)
    right = float(x2)
    if right <= left:
        raise ValueError("stitch-x2-q-over-qc must be larger than stitch-x1-q-over-qc.")
    x = np.asarray(qz_over_qc, dtype=float)
    weight = np.ones_like(x, dtype=float)
    weight[x >= right] = 0.0
    mid = (x > left) & (x < right)
    weight[mid] = 0.5 * (1.0 + np.cos(np.pi * (x[mid] - left) / (right - left)))
    return weight


def log_blend(
    low_q_curve: np.ndarray,
    high_q_curve: np.ndarray,
    weight: np.ndarray,
    floor: float = 1.0e-300,
) -> np.ndarray:
    low = np.maximum(np.asarray(low_q_curve, dtype=float), float(floor))
    high = np.maximum(np.asarray(high_q_curve, dtype=float), float(floor))
    blend_weight = np.asarray(weight, dtype=float)
    return np.exp(blend_weight * np.log(low) + (1.0 - blend_weight) * np.log(high))


def validated_l_limits(
    l_min: float, l_max: float, *, min_width: float = 0.001
) -> tuple[float, float]:
    return _validated_ordered_limits(
        l_min,
        l_max,
        default_min=DEFAULTS.L_min,
        default_max=DEFAULTS.L_max,
        min_width=min_width,
    )


def log10_or_default(value: float | None, default: float) -> float:
    if value is None:
        return float(default)
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        return float(default)
    return float(np.log10(number))


def validated_y_log_limits(
    y_min_log: float, y_max_log: float, *, min_width: float = 0.1
) -> tuple[float, float]:
    return _validated_ordered_limits(
        y_min_log,
        y_max_log,
        default_min=DEFAULT_Y_MIN_LOG10,
        default_max=DEFAULT_Y_MAX_LOG10,
        min_width=min_width,
    )


def _validated_ordered_limits(
    low: float,
    high: float,
    *,
    default_min: float,
    default_max: float,
    min_width: float,
) -> tuple[float, float]:
    left = float(low)
    right = float(high)
    width = float(min_width)
    if not np.isfinite(left):
        left = float(default_min)
    if not np.isfinite(right):
        right = float(default_max)
    if right <= left:
        right = left + width
    return left, right


def expanded_slider_bounds(
    current_min: float, current_max: float, value: float
) -> tuple[float, float]:
    left = float(current_min)
    right = float(current_max)
    typed = float(value)
    return min(left, typed), max(right, typed)


@dataclass
class Fig2GuiParameterState:
    film_qc_inv_angstrom: float
    film_density_e_per_a3: float
    substrate_qc_inv_angstrom: float
    substrate_density_e_per_a3: float
    thickness_angstrom: float
    top_roughness_angstrom: float
    bottom_roughness_angstrom: float
    bandwidth_percent: float
    n_wavelength_samples: float
    divergence_fwhm_deg: float
    n_divergence_samples: float
    ht_structure_scale: float
    ht_over_q2_scale: float

    @classmethod
    def from_args(cls, args) -> "Fig2GuiParameterState":
        return cls(
            film_qc_inv_angstrom=float(args.qc_inv_angstrom),
            film_density_e_per_a3=float(args.film_density_e_per_a3),
            substrate_qc_inv_angstrom=float(args.substrate_qc_inv_angstrom),
            substrate_density_e_per_a3=float(args.substrate_density_e_per_a3),
            thickness_angstrom=float(args.thickness_nm) * 10.0,
            top_roughness_angstrom=float(args.top_roughness_angstrom),
            bottom_roughness_angstrom=float(args.bottom_roughness_angstrom),
            bandwidth_percent=float(args.bandwidth_fwhm) * 100.0,
            n_wavelength_samples=float(args.n_wavelength_samples),
            divergence_fwhm_deg=float(args.divergence_fwhm_deg),
            n_divergence_samples=float(args.n_divergence_samples),
            ht_structure_scale=float(args.ht_structure_scale),
            ht_over_q2_scale=float(args.ht_over_q2_scale),
        )

    def sync_density_from_qc(self, material_kind: str) -> None:
        if material_kind == "film":
            self.film_density_e_per_a3 = density_e_per_a3_from_qc(self.film_qc_inv_angstrom)
            return
        if material_kind == "substrate":
            self.substrate_density_e_per_a3 = density_e_per_a3_from_qc(
                self.substrate_qc_inv_angstrom
            )
            return
        raise ValueError(f"Unknown material kind: {material_kind}")

    def sync_qc_from_density(self, material_kind: str) -> None:
        if material_kind == "film":
            self.film_qc_inv_angstrom = qc_from_density_e_per_a3(self.film_density_e_per_a3)
            return
        if material_kind == "substrate":
            self.substrate_qc_inv_angstrom = qc_from_density_e_per_a3(
                self.substrate_density_e_per_a3
            )
            return
        raise ValueError(f"Unknown material kind: {material_kind}")

    def updated_args(self, args):
        updated = argparse.Namespace(**vars(args))
        updated.qc_inv_angstrom = float(self.film_qc_inv_angstrom)
        updated.film_density_e_per_a3 = float(self.film_density_e_per_a3)
        updated.substrate_qc_inv_angstrom = float(self.substrate_qc_inv_angstrom)
        updated.substrate_density_e_per_a3 = float(self.substrate_density_e_per_a3)
        updated.thickness_nm = float(self.thickness_angstrom) / 10.0
        updated.top_roughness_angstrom = float(self.top_roughness_angstrom)
        updated.bottom_roughness_angstrom = float(self.bottom_roughness_angstrom)
        updated.bandwidth_fwhm = float(self.bandwidth_percent) / 100.0
        updated.n_wavelength_samples = max(1, int(round(float(self.n_wavelength_samples))))
        updated.divergence_fwhm_deg = float(self.divergence_fwhm_deg)
        updated.n_divergence_samples = max(1, int(round(float(self.n_divergence_samples))))
        updated.ht_structure_scale = float(self.ht_structure_scale)
        updated.ht_over_q2_scale = float(self.ht_over_q2_scale)
        return updated


@dataclass
class LiveComputeGate:
    generation: int = 0
    active_generation: int | None = None
    queued_compute: tuple[int, object] | None = None

    def next_generation(self) -> int:
        self.generation += 1
        return self.generation

    def is_current(self, generation: int) -> bool:
        return int(generation) == self.generation

    def request_live_compute(self, args) -> tuple[int, object] | None:
        generation = self.next_generation()
        if self.active_generation is None:
            self.active_generation = generation
            return generation, args
        self.queued_compute = generation, args
        return None

    def finish_live_compute(self, generation: int) -> tuple[int, object] | None:
        if int(generation) != self.active_generation:
            return None
        if self.queued_compute is None:
            self.active_generation = None
            return None

        queued = self.queued_compute
        self.active_generation = queued[0]
        self.queued_compute = None
        return queued


# =============================================================================
# Optical helper functions
# =============================================================================


@dataclass(frozen=True)
class Layer:
    name: str
    qc_inv_angstrom: float
    thickness_angstrom: float | None = None
    beta: float = 0.0
    roughness_angstrom: float = 0.0


def make_air_film_substrate_stack(args) -> list[Layer]:
    substrate_qc = 0.0 if args.substrate.lower() == "air" else float(args.substrate_qc_inv_angstrom)
    return [
        Layer("air", 0.0, None),
        Layer(
            args.material,
            float(args.qc_inv_angstrom),
            float(args.thickness_nm) * 10.0,
            roughness_angstrom=float(args.top_roughness_angstrom),
        ),
        Layer(
            args.substrate,
            substrate_qc,
            None,
            roughness_angstrom=float(args.bottom_roughness_angstrom),
        ),
    ]


def fresnel_reflectivity_single_interface(qz: np.ndarray, qc: float) -> np.ndarray:
    qz = np.asarray(qz, dtype=np.complex128)
    qp = np.sqrt(qz**2 - float(qc) ** 2 + 0j)
    r = (qz - qp) / (qz + qp)
    return np.abs(r) ** 2


def born_fresnel_asymptote(qz: np.ndarray, qc: float) -> np.ndarray:
    qz = np.asarray(qz, dtype=float)
    return (float(qc) / (2.0 * qz)) ** 4


def parratt_reflectivity(
    qz: np.ndarray, layers: list[Layer], wavelength_angstrom: float
) -> np.ndarray:
    """General Parratt recursion for a stack of layers.

    layers[0] is the incident medium.
    layers[1:-1] are finite layers.
    layers[-1] is the semi-infinite substrate.
    """
    qz = np.asarray(qz, dtype=float)
    k0 = 2.0 * np.pi / float(wavelength_angstrom)

    # Qz = 2 k0 sin(alpha)
    alpha = np.arcsin(np.clip(qz / (2.0 * k0), 0.0, 1.0))
    kx = k0 * np.cos(alpha)

    kz = []
    for layer in layers:
        qc = float(layer.qc_inv_angstrom)
        # Qc = 2 k0 sqrt(2 delta) -> delta = 0.5*(Qc/(2k0))^2
        delta = 0.5 * (qc / (2.0 * k0)) ** 2
        n = 1.0 - delta + 1j * float(layer.beta)
        kz.append(np.sqrt((n * k0) ** 2 - kx**2 + 0j))

    r_eff = np.zeros_like(qz, dtype=np.complex128)

    for j in range(len(layers) - 2, -1, -1):
        kz_j = kz[j]
        kz_next = kz[j + 1]

        r_j = (kz_j - kz_next) / (kz_j + kz_next)

        sigma = float(layers[j + 1].roughness_angstrom)
        if sigma > 0.0:
            r_j *= np.exp(-2.0 * kz_j * kz_next * sigma**2)

        d_next = layers[j + 1].thickness_angstrom
        phase = 1.0 if d_next is None else np.exp(2j * kz_next * float(d_next))

        r_eff = (r_j + r_eff * phase) / (1.0 + r_j * r_eff * phase)

    return np.abs(r_eff) ** 2


# =============================================================================
# Small numerical helpers
# =============================================================================


def qz_from_L(L: np.ndarray, c_angstrom: float) -> np.ndarray:
    return 2.0 * np.pi * np.asarray(L, dtype=float) / float(c_angstrom)


def L_from_qz(qz: np.ndarray, c_angstrom: float) -> np.ndarray:
    return np.asarray(qz, dtype=float) * float(c_angstrom) / (2.0 * np.pi)


def gaussian_samples(center: float, fwhm: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gaussian samples and weights.

    If n=1 or fwhm=0, return the center with unit weight.
    """
    if int(n) <= 1 or float(fwhm) == 0.0:
        return np.array([float(center)]), np.array([1.0])

    sigma = float(fwhm) / 2.354820045
    eps = np.linspace(-4.0 * sigma, 4.0 * sigma, int(n))
    weights = np.exp(-0.5 * (eps / sigma) ** 2)
    weights /= weights.sum()
    return float(center) + eps, weights


def stable_zero_reference(L: np.ndarray, I: np.ndarray) -> float:
    L = np.asarray(L, dtype=float)
    I = np.asarray(I, dtype=float)
    idx = int(np.nanargmin(np.abs(L)))
    ref = float(I[idx])
    if not np.isfinite(ref) or ref <= 0.0:
        near = np.abs(L) < 0.02
        ref = float(np.nanmax(I[near])) if np.any(near) else float(np.nanmax(I))
    if not np.isfinite(ref) or ref <= 0.0:
        raise ValueError("Could not find a positive finite HT reference near L=0.")
    return ref


def ht_over_q2_positive_qz_division(
    structure: np.ndarray,
    qz: np.ndarray,
    *,
    qz_min: float,
    scale: float = 1.0,
) -> np.ndarray:
    structure_values, qz_values = np.broadcast_arrays(
        np.asarray(structure, dtype=float),
        np.asarray(qz, dtype=float),
    )
    out = np.full(structure_values.shape, np.nan, dtype=float)
    valid = np.isfinite(structure_values) & np.isfinite(qz_values) & (qz_values > float(qz_min))
    np.divide(
        structure_values * float(scale),
        qz_values**2,
        out=out,
        where=valid,
    )
    return out


def divergence_safe_ht_over_q2_average(
    structure_samples: np.ndarray,
    qz_samples: np.ndarray,
    divergence_weights: np.ndarray,
    *,
    qz_min: float,
) -> np.ndarray:
    structure_values = np.asarray(structure_samples, dtype=float)
    qz_values = np.asarray(qz_samples, dtype=float)
    weights = np.asarray(divergence_weights, dtype=float)
    if structure_values.shape != qz_values.shape:
        raise ValueError("structure_samples and qz_samples must have the same shape.")
    if structure_values.ndim != 2:
        raise ValueError("structure_samples and qz_samples must be 2D arrays.")
    if weights.ndim != 1 or weights.shape[0] != structure_values.shape[0]:
        raise ValueError("divergence_weights must have one value per divergence sample.")

    per_sample = ht_over_q2_positive_qz_division(
        structure_values,
        qz_values,
        qz_min=qz_min,
    )
    weights_2d = weights[:, np.newaxis]
    valid = np.isfinite(per_sample) & np.isfinite(weights_2d) & (weights_2d > 0.0)
    weighted_sum = np.where(valid, per_sample * weights_2d, 0.0).sum(axis=0)
    weight_sum = np.where(valid, weights_2d, 0.0).sum(axis=0)
    out = np.full(structure_values.shape[1], np.nan, dtype=float)
    np.divide(weighted_sum, weight_sum, out=out, where=weight_sum > 0.0)
    return out


def pseudo_voigt_profile(
    L: np.ndarray,
    center: float,
    amplitude: float,
    width: float,
    eta: float,
) -> np.ndarray:
    width = max(float(width), np.finfo(float).eps)
    mixing = float(np.clip(eta, 0.0, 1.0))
    x = (np.asarray(L, dtype=np.float64) - float(center)) / width
    gaussian = np.exp(-0.5 * x**2)
    lorentzian = 1.0 / (1.0 + x**2)
    return float(amplitude) * ((1.0 - mixing) * gaussian + mixing * lorentzian)


def fit_pseudo_voigt_peak_sum(
    L: np.ndarray,
    observed: np.ndarray,
    background: np.ndarray,
    *,
    peak_centers: tuple[float, ...] = DEFAULT_AUTOFIT_PEAK_CENTERS,
    peak_half_width: float = DEFAULT_AUTOFIT_PEAK_HALF_WIDTH,
) -> np.ndarray:
    from scipy.optimize import OptimizeWarning, curve_fit

    l_values = np.asarray(L, dtype=np.float64)
    observed_values = np.asarray(observed, dtype=np.float64)
    background_values = np.asarray(background, dtype=np.float64)
    peak_sum = np.zeros_like(l_values, dtype=np.float64)

    for center in peak_centers:
        center = float(center)
        local = (
            (l_values >= center - peak_half_width)
            & (l_values <= center + peak_half_width)
            & np.isfinite(l_values)
            & np.isfinite(observed_values)
            & np.isfinite(background_values)
        )
        if np.count_nonzero(local) < 5:
            continue

        local_l = l_values[local]
        residual = np.maximum(observed_values[local] - background_values[local], 0.0)
        amplitude0 = float(np.nanmax(residual))
        if not np.isfinite(amplitude0) or amplitude0 <= 0.0:
            continue

        def _model(x, amplitude, peak_center, width, eta):
            return pseudo_voigt_profile(x, peak_center, amplitude, width, eta)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                params, _cov = curve_fit(
                    _model,
                    local_l,
                    residual,
                    p0=(amplitude0, center, 0.08, 0.5),
                    bounds=(
                        (0.0, center - 0.15, 0.01, 0.0),
                        (np.inf, center + 0.15, peak_half_width, 1.0),
                    ),
                    method="trf",
                    maxfev=5000,
                )
        except (RuntimeError, ValueError):
            continue

        amplitude, peak_center, width, eta = params
        peak_sum += pseudo_voigt_profile(l_values, peak_center, amplitude, width, eta)

    return peak_sum


def _curve_grid_for_label(curves: pd.DataFrame, label: str) -> pd.DataFrame:
    sub = curves[curves["label"] == label]
    if len(sub) == 0:
        raise ValueError(f"Curve not found: {label}")
    return sub.drop_duplicates(subset=["L"]).sort_values("L")


def _curve_arrays_for_label(curves: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    grid = _curve_grid_for_label(curves, label)
    return np.asarray(grid["L"], dtype=np.float64), np.asarray(grid["intensity"], dtype=np.float64)


def curve_labels_for_plot_mode(args) -> tuple[str, ...]:
    mode = getattr(args, "plot_mode", "fig2")
    if mode == "stitch":
        return STITCH_CURVE_ORDER
    if mode == "stitch-diagnostics":
        return STITCH_DIAGNOSTIC_CURVE_ORDER
    return FIG2_CURVE_ORDER


def stitch_requested(args) -> bool:
    return (
        bool(getattr(args, "stitch", False))
        or getattr(args, "plot_mode", "fig2") != "fig2"
        or bool(getattr(args, "write_stitch_diagnostics", False))
    )


def stitch_fit_mask(L: np.ndarray, qz: np.ndarray, args) -> np.ndarray:
    qz_over_qc = np.asarray(qz, dtype=float) / float(args.qc_inv_angstrom)
    return (
        (qz_over_qc >= float(args.fit_x_min_q_over_qc))
        & (qz_over_qc <= float(args.fit_x_max_q_over_qc))
        & bragg_exclusion_mask(
            np.asarray(L, dtype=float),
            centers=parse_float_list(args.fit_exclude_centers_L),
            half_width=float(args.fit_exclude_half_width_L),
        )
    )


def stitch_fit_l_limits(args) -> tuple[float, float]:
    fit_qz_min = float(args.fit_x_min_q_over_qc) * float(args.qc_inv_angstrom)
    fit_qz_max = float(args.fit_x_max_q_over_qc) * float(args.qc_inv_angstrom)
    return validated_l_limits(
        L_from_qz(fit_qz_min, args.c_angstrom),
        L_from_qz(fit_qz_max, args.c_angstrom),
    )


def compute_l_grid_limits(args) -> tuple[float, float]:
    visible_l_min, visible_l_max = validated_l_limits(args.L_min, args.L_max)
    if stitch_requested(args) and args.scale_mode == "log-median":
        fit_l_min, fit_l_max = stitch_fit_l_limits(args)
        return min(visible_l_min, fit_l_min), max(visible_l_max, fit_l_max)
    return visible_l_min, visible_l_max


def raw_miceli_stitch_requested(args) -> bool:
    return stitch_requested(args) and args.stitch_branch == "raw-miceli-ht-q2"


def build_stitch_curve_frames(
    L: np.ndarray,
    qz: np.ndarray,
    pure_parratt: np.ndarray,
    normalized_ht_over_q2: np.ndarray,
    raw_miceli_ht_over_q2: np.ndarray | None,
    args,
) -> list[pd.DataFrame]:
    l_values = np.asarray(L, dtype=float)
    qz_values = np.asarray(qz, dtype=float)
    pure_values = np.asarray(pure_parratt, dtype=float)
    normalized_values = np.asarray(normalized_ht_over_q2, dtype=float)
    if args.stitch_branch == "normalized-ht-q2":
        stitch_high_unscaled = normalized_values
    elif args.stitch_branch == "raw-miceli-ht-q2":
        if raw_miceli_ht_over_q2 is None:
            raise ValueError("raw-miceli-ht-q2 stitch branch is unavailable for these curves.")
        stitch_high_unscaled = np.asarray(raw_miceli_ht_over_q2, dtype=float)
    else:
        raise ValueError(f"Unknown stitch branch: {args.stitch_branch}")

    qz_over_qc = qz_values / float(args.qc_inv_angstrom)
    fit_mask = stitch_fit_mask(l_values, qz_values, args)
    if args.scale_mode == "manual":
        stitch_scale = float(args.manual_stitch_scale)
    elif args.scale_mode == "miceli-cell":
        stitch_scale = 1.0
    elif args.scale_mode == "log-median":
        stitch_scale = fit_log_median_scale(stitch_high_unscaled, pure_values, fit_mask)
    else:
        raise ValueError(f"Unknown scale mode: {args.scale_mode}")

    scaled_ht_over_q2 = stitch_scale * stitch_high_unscaled
    stitch_weight = smooth_stitch_weight(
        qz_over_qc,
        x1=float(args.stitch_x1_q_over_qc),
        x2=float(args.stitch_x2_q_over_qc),
    )
    stitched = log_blend(pure_values, scaled_ht_over_q2, stitch_weight)
    return [
        pd.DataFrame(
            {
                "L": l_values,
                "Qz_Ainv": qz_values,
                "intensity": intensity,
                "label": label,
            }
        )
        for label, intensity in (
            (STITCH_UNSCALED_HT_OVER_Q2_LABEL, stitch_high_unscaled),
            (STITCH_SCALED_HT_OVER_Q2_LABEL, scaled_ht_over_q2),
            (STITCHED_PARRATT_HT_OVER_Q2_LABEL, stitched),
            (STITCH_WEIGHT_LABEL, stitch_weight),
        )
    ]


def write_stitch_diagnostics(curves: pd.DataFrame, args, out_dir: Path) -> Path:
    scaled_grid = _curve_grid_for_label(curves, STITCH_SCALED_HT_OVER_Q2_LABEL)
    scaled_l = np.asarray(scaled_grid["L"], dtype=float)
    scaled_y = np.asarray(scaled_grid["intensity"], dtype=float)
    unscaled_l, unscaled_y = _curve_arrays_for_label(curves, STITCH_UNSCALED_HT_OVER_Q2_LABEL)
    pure_l, pure_y = _curve_arrays_for_label(curves, PURE_PARRATT_LABEL)
    qz_values = np.asarray(scaled_grid["Qz_Ainv"], dtype=float)
    pure_interp = np.interp(scaled_l, pure_l, pure_y)
    unscaled_interp = np.interp(scaled_l, unscaled_l, unscaled_y)
    fit_mask = stitch_fit_mask(scaled_l, qz_values, args)
    valid_residual = (
        fit_mask
        & np.isfinite(scaled_y)
        & np.isfinite(pure_interp)
        & (scaled_y > 0.0)
        & (pure_interp > 0.0)
    )
    residual = np.log10(scaled_y[valid_residual] / pure_interp[valid_residual])
    if residual.size == 0:
        median_residual = np.nan
        mad_residual = np.nan
    else:
        median_residual = float(np.median(residual))
        mad_residual = float(np.median(np.abs(residual - median_residual)))
    valid_scale = (
        fit_mask
        & np.isfinite(scaled_y)
        & np.isfinite(unscaled_interp)
        & (scaled_y > 0.0)
        & (unscaled_interp > 0.0)
    )
    stitch_scale = (
        float(np.median(scaled_y[valid_scale] / unscaled_interp[valid_scale]))
        if np.any(valid_scale)
        else np.nan
    )
    diagnostics = pd.DataFrame(
        [
            {
                "material": args.material,
                "stitch_branch": args.stitch_branch,
                "scale_mode": args.scale_mode,
                "stitch_scale": stitch_scale,
                "fit_x_min_q_over_qc": args.fit_x_min_q_over_qc,
                "fit_x_max_q_over_qc": args.fit_x_max_q_over_qc,
                "stitch_x1_q_over_qc": args.stitch_x1_q_over_qc,
                "stitch_x2_q_over_qc": args.stitch_x2_q_over_qc,
                "fit_exclude_centers_L": args.fit_exclude_centers_L,
                "fit_exclude_half_width_L": args.fit_exclude_half_width_L,
                "n_fit_points": int(np.count_nonzero(valid_residual)),
                "median_log10_residual": median_residual,
                "mad_log10_residual": mad_residual,
            }
        ]
    )
    out_dir = Path(out_dir)
    out_path = out_dir / "stitch_diagnostics.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_path, index=False)
    if np.isfinite(mad_residual) and mad_residual > 0.3:
        print(
            "WARNING: scaled HT/Qz^2 does not overlap pure Parratt well in the chosen "
            "window; stitch is a visual composite, not a strong physical dovetail."
        )
    return out_path


def write_stitch_diagnostic_plot(curves: pd.DataFrame, args, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    pure_l, pure_y = _curve_arrays_for_label(curves, PURE_PARRATT_LABEL)
    unscaled_l, unscaled_y = _curve_arrays_for_label(curves, STITCH_UNSCALED_HT_OVER_Q2_LABEL)
    scaled_l, scaled_y = _curve_arrays_for_label(curves, STITCH_SCALED_HT_OVER_Q2_LABEL)
    stitched_l, stitched_y = _curve_arrays_for_label(curves, STITCHED_PARRATT_HT_OVER_Q2_LABEL)

    pure_interp = np.interp(scaled_l, pure_l, pure_y)
    ratio = np.full_like(scaled_y, np.nan, dtype=float)
    valid_ratio = (
        np.isfinite(scaled_y) & np.isfinite(pure_interp) & (scaled_y > 0.0) & (pure_interp > 0.0)
    )
    np.divide(scaled_y, pure_interp, out=ratio, where=valid_ratio)

    out_dir = Path(out_dir)
    out_path = out_dir / "fig_stitch_scale_diagnostic.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax, ratio_ax) = plt.subplots(2, 1, figsize=(9.4, 7.0), sharex=True)
    for l_values, intensity, label in (
        (pure_l, pure_y, PURE_PARRATT_LABEL),
        (unscaled_l, unscaled_y, STITCH_UNSCALED_HT_OVER_Q2_LABEL),
        (scaled_l, scaled_y, STITCH_SCALED_HT_OVER_Q2_LABEL),
        (stitched_l, stitched_y, STITCHED_PARRATT_HT_OVER_Q2_LABEL),
    ):
        ax.plot(l_values, intensity, label=FIG2_CURVE_DISPLAY_LABELS[label])
    fit_l_min, fit_l_max = stitch_fit_l_limits(args)
    ax.axvspan(fit_l_min, fit_l_max, color="0.8", alpha=0.25, label="Scale window")
    for center in parse_float_list(args.fit_exclude_centers_L):
        ax.axvspan(
            float(center) - float(args.fit_exclude_half_width_L),
            float(center) + float(args.fit_exclude_half_width_L),
            color="0.7",
            alpha=0.18,
        )
    ax.set_yscale("log")
    ax.set_ylabel(r"$I/I_0$ or $R$")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.25)

    ratio_ax.plot(scaled_l, ratio, color="#D55E00", label=r"Scaled HT/$R_P$")
    ratio_ax.axhline(1.0, color="0.3", linestyle=":", linewidth=1.0)
    ratio_ax.set_yscale("log")
    ratio_ax.set_xlabel(r"$L$ (r.l.u.)")
    ratio_ax.set_ylabel("Ratio")
    ratio_ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    return out_path


def curves_span_l_range(
    curves: pd.DataFrame,
    l_min: float,
    l_max: float,
    *,
    labels: tuple[str, ...] = (HT_OVER_Q2_LABEL, PURE_PARRATT_LABEL),
) -> bool:
    tolerance = max(1.0e-9, abs(float(l_max) - float(l_min)) * 1.0e-10)
    for label in labels:
        try:
            l_values, _intensity = _curve_arrays_for_label(curves, label)
        except ValueError:
            return False
        finite_l = l_values[np.isfinite(l_values)]
        if finite_l.size == 0:
            return False
        if (
            float(np.nanmin(finite_l)) - tolerance > l_min
            or float(np.nanmax(finite_l)) + tolerance < l_max
        ):
            return False
    return True


def autofit_args_for_l_range(
    args,
    fit_l_min: float = DEFAULT_AUTOFIT_L_MIN,
    fit_l_max: float = DEFAULT_AUTOFIT_L_MAX,
):
    fit_args = argparse.Namespace(**vars(args))
    fit_args.L_min, fit_args.L_max = validated_l_limits(fit_l_min, fit_l_max)
    fit_args.from_csv = None
    return fit_args


def _least_squares_scale(model: np.ndarray, target: np.ndarray) -> float:
    model = np.asarray(model, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(model) & np.isfinite(target) & (model > 0.0) & (target > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Need at least two finite positive points for HT/Qz² auto-fit.")
    model = model[valid]
    target = target[valid]
    denominator = float(np.sum(model**2))
    if denominator <= 0.0:
        raise ValueError("HT/Qz² auto-fit denominator is zero.")
    scale = float(np.sum(model * target) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("HT/Qz² auto-fit produced a non-positive scale.")
    return scale


def fit_ht_over_q2_scale_to_parratt(
    curves: pd.DataFrame,
    *,
    current_scale: float = 1.0,
    fit_l_min: float = DEFAULT_AUTOFIT_L_MIN,
    fit_l_max: float = DEFAULT_AUTOFIT_L_MAX,
    peak_centers: tuple[float, ...] = DEFAULT_AUTOFIT_PEAK_CENTERS,
    peak_half_width: float = DEFAULT_AUTOFIT_PEAK_HALF_WIDTH,
) -> float:
    current_scale = float(current_scale)
    if not np.isfinite(current_scale) or current_scale <= 0.0:
        raise ValueError("Current HT/Qz² scale must be positive for auto-fit.")

    ht_l, ht_y = _curve_arrays_for_label(curves, HT_OVER_Q2_LABEL)
    parratt_l, parratt_y = _curve_arrays_for_label(curves, PURE_PARRATT_LABEL)
    if not curves_span_l_range(curves, fit_l_min, fit_l_max):
        raise ValueError(
            f"Auto-fit requires curves spanning {fit_l_min:.1f} <= L <= {fit_l_max:.1f}."
        )

    in_range = (ht_l >= fit_l_min) & (ht_l <= fit_l_max)
    l_fit = ht_l[in_range]
    ht_fit = ht_y[in_range] / current_scale
    parratt_fit = np.interp(l_fit, parratt_l, parratt_y)
    finite = np.isfinite(l_fit) & np.isfinite(ht_fit) & np.isfinite(parratt_fit)
    finite &= ht_fit > 0.0
    finite &= parratt_fit > 0.0
    if np.count_nonzero(finite) < 10:
        raise ValueError("Need at least ten finite positive points for HT/Qz² auto-fit.")

    l_fit = l_fit[finite]
    ht_fit = ht_fit[finite]
    parratt_fit = parratt_fit[finite]

    peak_excluded = np.ones_like(l_fit, dtype=bool)
    for center in peak_centers:
        peak_excluded &= np.abs(l_fit - float(center)) > peak_half_width
    background_mask = (
        peak_excluded if np.count_nonzero(peak_excluded) >= 10 else np.ones_like(l_fit, dtype=bool)
    )
    preliminary_scale = _least_squares_scale(ht_fit[background_mask], parratt_fit[background_mask])
    background_in_ht_units = parratt_fit / preliminary_scale
    fitted_peaks = fit_pseudo_voigt_peak_sum(
        l_fit,
        ht_fit,
        background_in_ht_units,
        peak_centers=peak_centers,
        peak_half_width=peak_half_width,
    )
    ht_without_peaks = np.maximum(ht_fit - fitted_peaks, 0.0)
    peak_subtracted_valid = (
        np.isfinite(ht_without_peaks)
        & np.isfinite(parratt_fit)
        & (ht_without_peaks > 0.0)
        & (parratt_fit > 0.0)
    )
    if np.count_nonzero(peak_subtracted_valid) < 2:
        return preliminary_scale
    return _least_squares_scale(ht_without_peaks, parratt_fit)


def find_repo_root(start: Path) -> Path:
    here = start.resolve()
    candidates = [here] + list(here.parents)
    for candidate in candidates:
        if (candidate / "ra_sim").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find repo root. Run this script from inside the repo, "
        "or place it under the repo's scripts/ directory."
    )


# =============================================================================
# Pure Parratt averaging
# =============================================================================


def average_pure_parratt_for_L(L_nom: np.ndarray, args) -> np.ndarray:
    """Compute pure Parratt reflectivity on the requested nominal L grid.

    This function includes wavelength bandwidth and optional divergence.
    It contains no HT structure term.
    """
    L_nom = np.asarray(L_nom, dtype=float)
    qz_nom = qz_from_L(L_nom, args.c_angstrom)
    layers = make_air_film_substrate_stack(args)

    lambda_samples, lambda_weights = gaussian_samples(
        args.lambda0_angstrom,
        args.bandwidth_fwhm * args.lambda0_angstrom,
        args.n_wavelength_samples,
    )
    divergence_samples_deg, divergence_weights = gaussian_samples(
        0.0,
        args.divergence_fwhm_deg,
        args.n_divergence_samples,
    )
    divergence_samples_rad = np.deg2rad(divergence_samples_deg)

    out = np.zeros_like(L_nom, dtype=float)
    theta_nom = np.arcsin(np.clip(qz_nom * args.lambda0_angstrom / (4.0 * np.pi), 0.0, 1.0))

    for lam_j, w_lam in zip(lambda_samples, lambda_weights):
        for dtheta, w_ang in zip(divergence_samples_rad, divergence_weights):
            theta_true = np.maximum(theta_nom + dtheta, 0.0)
            qz_true = 4.0 * np.pi * np.sin(theta_true) / float(lam_j)
            out += (
                float(w_lam)
                * float(w_ang)
                * parratt_reflectivity(qz_true, layers, wavelength_angstrom=float(lam_j))
            )

    return out


# =============================================================================
# CSV loading mode
# =============================================================================


def load_curves_from_csv(path: Path, args) -> pd.DataFrame:
    """Load already-generated model curves.

    If the CSV does not include pure Parratt, this function computes pure Parratt
    locally on the same L grid using the current CLI/default optical parameters.
    """
    df = pd.read_csv(path)

    parts = []

    if {"curve_name", "L", "Qz_Ainv", "intensity"}.issubset(df.columns):
        names = {
            "HT_structure_term_p0": HT_STRUCTURE_LABEL,
            "S_HT0": HT_STRUCTURE_LABEL,
            "Born_scaled_HT_p0": BORN_HT_LABEL,
            "Fresnel_corrected_HT_p0": FRESNEL_HT_LABEL,
            "Parratt_envelope_HT_p0": PARRATT_ENV_HT_LABEL,
            "Parratt_reflectivity": PURE_PARRATT_LABEL,
            "Parratt_reflectivity_air_film_SiO2": PURE_PARRATT_LABEL,
        }
        for curve_name, label in names.items():
            sub = df[df["curve_name"] == curve_name].copy()
            if len(sub) == 0:
                continue
            sub["label"] = label
            parts.append(sub[["L", "Qz_Ainv", "intensity", "label"]])

    elif {"L", "Qz_Ainv"}.issubset(df.columns):
        wide_map = {
            "HT_structure_term_p0": HT_STRUCTURE_LABEL,
            "S_HT0": HT_STRUCTURE_LABEL,
            "Born_scaled_HT_p0": BORN_HT_LABEL,
            "R_Born_HT": BORN_HT_LABEL,
            "Fresnel_corrected_HT_p0": FRESNEL_HT_LABEL,
            "R_Fresnel_HT": FRESNEL_HT_LABEL,
            "Parratt_envelope_HT_p0": PARRATT_ENV_HT_LABEL,
            "R_ParrattEnvelope_HT": PARRATT_ENV_HT_LABEL,
            "R_Parratt_sharp": PURE_PARRATT_LABEL,
            "Parratt_reflectivity": PURE_PARRATT_LABEL,
        }
        for col, label in wide_map.items():
            if col in df.columns:
                parts.append(
                    pd.DataFrame(
                        {
                            "L": df["L"],
                            "Qz_Ainv": df["Qz_Ainv"],
                            "intensity": df[col],
                            "label": label,
                        }
                    )
                )

    else:
        raise ValueError(
            f"CSV {path} does not look like a supported model curve CSV. "
            f"Columns: {list(df.columns)}"
        )

    if not any((p["label"].iloc[0] == HT_OVER_Q2_LABEL) for p in parts):
        for part in parts:
            if part["label"].iloc[0] != HT_STRUCTURE_LABEL:
                continue
            grid = part.drop_duplicates(subset=["L", "Qz_Ainv"]).sort_values("L")
            parts.append(
                pd.DataFrame(
                    {
                        "L": grid["L"].to_numpy(),
                        "Qz_Ainv": grid["Qz_Ainv"].to_numpy(),
                        "intensity": ht_over_q2_positive_qz_division(
                            np.asarray(grid["intensity"], dtype=float),
                            np.asarray(grid["Qz_Ainv"], dtype=float),
                            qz_min=float(args.qz_floor_for_born),
                            scale=float(args.ht_over_q2_scale),
                        ),
                        "label": HT_OVER_Q2_LABEL,
                    }
                )
            )
            break

    for part in parts:
        if part["label"].iloc[0] == HT_STRUCTURE_LABEL:
            part["intensity"] = part["intensity"] * float(args.ht_structure_scale)

    if not any((p["label"].iloc[0] == PURE_PARRATT_LABEL) for p in parts):
        # Compute it on the grid from the loaded CSV.
        grid = df[["L", "Qz_Ainv"]].drop_duplicates().sort_values("L")
        if args.L_max is not None:
            grid = grid[grid["L"] <= float(args.L_max)]
        R = average_pure_parratt_for_L(grid["L"].to_numpy(), args)
        parts.append(
            pd.DataFrame(
                {
                    "L": grid["L"].to_numpy(),
                    "Qz_Ainv": grid["Qz_Ainv"].to_numpy(),
                    "intensity": R,
                    "label": PURE_PARRATT_LABEL,
                }
            )
        )

    out = pd.concat(parts, ignore_index=True)
    if args.L_max is not None:
        out = out[out["L"] <= float(args.L_max)].copy()
    if stitch_requested(args) and STITCHED_PARRATT_HT_OVER_Q2_LABEL not in set(out["label"]):
        ht_grid = (
            out[out["label"] == HT_OVER_Q2_LABEL]
            .drop_duplicates(subset=["L", "Qz_Ainv"])
            .sort_values("L")
        )
        pure_l, pure_y = _curve_arrays_for_label(out, PURE_PARRATT_LABEL)
        pure_interp = np.interp(np.asarray(ht_grid["L"], dtype=float), pure_l, pure_y)
        out = pd.concat(
            [
                out,
                *build_stitch_curve_frames(
                    np.asarray(ht_grid["L"], dtype=float),
                    np.asarray(ht_grid["Qz_Ainv"], dtype=float),
                    pure_interp,
                    np.asarray(ht_grid["intensity"], dtype=float),
                    None,
                    args,
                ),
            ],
            ignore_index=True,
        )
    return out


# =============================================================================
# Direct HT compute mode
# =============================================================================


def import_ht_module(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from ra_sim.utils.stacking_fault import ht_Iinf_dict
    except Exception as exc:
        raise RuntimeError(
            "Could not import ra_sim.utils.stacking_fault.ht_Iinf_dict. "
            "If this fails because CifFile is missing, install PyCifRW in the repo "
            "environment or use --from-csv with an existing model_curves_L_Qz.csv."
        ) from exc
    return ht_Iinf_dict


def get_hk_curve(curves, h: int = 0, k: int = 0):
    for key in [(h, k), f"{h},{k}", f"({h}, {k})"]:
        if key in curves:
            return curves[key]
    raise KeyError(f"No curve for hk=({h},{k}); available keys={list(curves)}")


def simulate_ht_p0_dense(
    *,
    ht_Iinf_dict,
    cif_path: Path,
    a_angstrom: float,
    c_angstrom: float,
    lambda_angstrom: float,
    p_user: float,
    stack_layers: int,
    L_step_dense: float,
    L_max_padded: float,
):
    curves = ht_Iinf_dict(
        cif_path=str(cif_path),
        hk_list=[(0, 0)],
        p=float(p_user),
        L_step=float(L_step_dense),
        L_max=float(L_max_padded),
        lambda_=float(lambda_angstrom),
        a_lattice=float(a_angstrom),
        c_lattice=float(c_angstrom),
        phase_delta_expression=None,
        phi_l_divisor=1.0,
        finite_stack=True,
        stack_layers=int(stack_layers),
    )
    curve = get_hk_curve(curves, 0, 0)
    return np.asarray(curve["L"], dtype=float), np.asarray(curve["I"], dtype=float)


def compute_curves(args) -> pd.DataFrame:
    repo_root = find_repo_root(Path.cwd())
    ht_Iinf_dict = import_ht_module(repo_root)

    cif_path = Path(args.cif_path)
    if not cif_path.is_absolute():
        cif_path = repo_root / cif_path
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    stack_layers = int(args.stack_layers)
    if stack_layers <= 0:
        stack_layers = max(1, int(round(args.thickness_nm * 10.0 / args.c_angstrom)))

    L_grid_min, L_grid_max = compute_l_grid_limits(args)
    L_nom = np.arange(L_grid_min, L_grid_max + 0.5 * args.L_step, args.L_step)
    qz_nom = qz_from_L(L_nom, args.c_angstrom)
    compute_raw_miceli_branch = raw_miceli_stitch_requested(args)

    lambda_samples, lambda_weights = gaussian_samples(
        args.lambda0_angstrom,
        args.bandwidth_fwhm * args.lambda0_angstrom,
        args.n_wavelength_samples,
    )
    divergence_samples_deg, divergence_weights = gaussian_samples(
        0.0,
        args.divergence_fwhm_deg,
        args.n_divergence_samples,
    )
    divergence_samples_rad = np.deg2rad(divergence_samples_deg)

    # Ensure the HT grid is padded enough for wavelength shifts.
    min_lambda = float(np.min(lambda_samples))
    needed_L_max = L_grid_max * args.lambda0_angstrom / min_lambda + 0.05
    L_max_padded = max(float(args.L_max_padded), needed_L_max)

    structure = np.zeros_like(L_nom, dtype=float)
    ht_over_q2 = np.zeros_like(L_nom, dtype=float)
    ht_over_q2_weight = np.zeros_like(L_nom, dtype=float)
    raw_miceli_ht_over_q2 = None
    raw_miceli_ht_over_q2_weight = None
    if compute_raw_miceli_branch:
        raw_miceli_ht_over_q2 = np.zeros_like(L_nom, dtype=float)
        raw_miceli_ht_over_q2_weight = np.zeros_like(L_nom, dtype=float)
    born = np.zeros_like(L_nom, dtype=float)
    fresnel = np.zeros_like(L_nom, dtype=float)
    parratt_env = np.zeros_like(L_nom, dtype=float)
    pure_parratt = np.zeros_like(L_nom, dtype=float)

    cached_ht = None
    if args.ht_bandwidth_mode == "interpolate-only":
        cached_ht = simulate_ht_p0_dense(
            ht_Iinf_dict=ht_Iinf_dict,
            cif_path=cif_path,
            a_angstrom=args.a_angstrom,
            c_angstrom=args.c_angstrom,
            lambda_angstrom=args.lambda0_angstrom,
            p_user=args.p_user,
            stack_layers=stack_layers,
            L_step_dense=args.L_step_dense,
            L_max_padded=L_max_padded,
        )
    elif args.ht_bandwidth_mode != "resimulate":
        raise ValueError("--ht-bandwidth-mode must be 'interpolate-only' or 'resimulate'")

    layers = make_air_film_substrate_stack(args)
    miceli_scale = miceli_cell_scale(args.a_angstrom) if compute_raw_miceli_branch else None

    theta_nom = np.arcsin(np.clip(qz_nom * args.lambda0_angstrom / (4.0 * np.pi), 0.0, 1.0))

    for lam_j, w_lam in zip(lambda_samples, lambda_weights):
        if cached_ht is None:
            L_dense, I_dense = simulate_ht_p0_dense(
                ht_Iinf_dict=ht_Iinf_dict,
                cif_path=cif_path,
                a_angstrom=args.a_angstrom,
                c_angstrom=args.c_angstrom,
                lambda_angstrom=lam_j,
                p_user=args.p_user,
                stack_layers=stack_layers,
                L_step_dense=args.L_step_dense,
                L_max_padded=L_max_padded,
            )
        else:
            L_dense, I_dense = cached_ht

        I0 = stable_zero_reference(L_dense, I_dense)
        ht_structure_samples = []
        raw_ht_samples = [] if compute_raw_miceli_branch else None
        ht_qz_samples = []

        for dtheta, w_ang in zip(divergence_samples_rad, divergence_weights):
            theta_raw = theta_nom + dtheta
            theta_true = np.maximum(theta_raw, 0.0)
            qz_true = 4.0 * np.pi * np.sin(theta_true) / float(lam_j)
            qz_ht = 4.0 * np.pi * np.sin(theta_raw) / float(lam_j)
            L_true = L_from_qz(qz_true, args.c_angstrom)

            I_true = np.interp(L_true, L_dense, I_dense, left=I_dense[0], right=I_dense[-1])
            S = I_true / I0
            ht_structure_samples.append(S)
            if raw_ht_samples is not None:
                raw_ht_samples.append(I_true)
            ht_qz_samples.append(qz_ht)

            qz_safe = np.maximum(qz_true, args.qz_floor_for_born)
            weight = float(w_lam) * float(w_ang)

            Rf = fresnel_reflectivity_single_interface(qz_safe, args.qc_inv_angstrom)
            Rp = parratt_reflectivity(qz_safe, layers, wavelength_angstrom=float(lam_j))

            structure += weight * S
            born += weight * born_fresnel_asymptote(qz_safe, args.qc_inv_angstrom) * S
            fresnel += weight * Rf * S
            parratt_env += weight * Rp * S
            pure_parratt += weight * Rp

        ht_over_q2_lam = divergence_safe_ht_over_q2_average(
            np.vstack(ht_structure_samples),
            np.vstack(ht_qz_samples),
            divergence_weights,
            qz_min=float(args.qz_floor_for_born),
        )
        valid_ht = np.isfinite(ht_over_q2_lam)
        ht_over_q2[valid_ht] += float(w_lam) * ht_over_q2_lam[valid_ht]
        ht_over_q2_weight[valid_ht] += float(w_lam)
        if compute_raw_miceli_branch:
            raw_miceli_ht_over_q2_lam = divergence_safe_ht_over_q2_average(
                miceli_scale * np.vstack(raw_ht_samples),
                np.vstack(ht_qz_samples),
                divergence_weights,
                qz_min=float(args.qz_floor_for_born),
            )
            valid_raw_ht = np.isfinite(raw_miceli_ht_over_q2_lam)
            raw_miceli_ht_over_q2[valid_raw_ht] += (
                float(w_lam) * raw_miceli_ht_over_q2_lam[valid_raw_ht]
            )
            raw_miceli_ht_over_q2_weight[valid_raw_ht] += float(w_lam)

    ht_over_q2_values = np.full_like(ht_over_q2, np.nan, dtype=float)
    np.divide(
        ht_over_q2,
        ht_over_q2_weight,
        out=ht_over_q2_values,
        where=ht_over_q2_weight > 0.0,
    )
    ht_over_q2 = ht_over_q2_values * float(args.ht_over_q2_scale)
    raw_miceli_ht_over_q2_values = None
    if compute_raw_miceli_branch:
        raw_miceli_ht_over_q2_values = np.full_like(raw_miceli_ht_over_q2, np.nan, dtype=float)
        np.divide(
            raw_miceli_ht_over_q2,
            raw_miceli_ht_over_q2_weight,
            out=raw_miceli_ht_over_q2_values,
            where=raw_miceli_ht_over_q2_weight > 0.0,
        )

    stitch_parts = []
    if stitch_requested(args):
        stitch_parts = build_stitch_curve_frames(
            L_nom,
            qz_nom,
            pure_parratt,
            ht_over_q2_values,
            raw_miceli_ht_over_q2_values,
            args,
        )

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "L": L_nom,
                    "Qz_Ainv": qz_nom,
                    "intensity": structure * float(args.ht_structure_scale),
                    "label": HT_STRUCTURE_LABEL,
                }
            ),
            pd.DataFrame(
                {
                    "L": L_nom,
                    "Qz_Ainv": qz_nom,
                    "intensity": ht_over_q2,
                    "label": HT_OVER_Q2_LABEL,
                }
            ),
            pd.DataFrame(
                {"L": L_nom, "Qz_Ainv": qz_nom, "intensity": born, "label": BORN_HT_LABEL}
            ),
            pd.DataFrame(
                {
                    "L": L_nom,
                    "Qz_Ainv": qz_nom,
                    "intensity": fresnel,
                    "label": FRESNEL_HT_LABEL,
                }
            ),
            pd.DataFrame(
                {
                    "L": L_nom,
                    "Qz_Ainv": qz_nom,
                    "intensity": parratt_env,
                    "label": PARRATT_ENV_HT_LABEL,
                }
            ),
            pd.DataFrame(
                {
                    "L": L_nom,
                    "Qz_Ainv": qz_nom,
                    "intensity": pure_parratt,
                    "label": PURE_PARRATT_LABEL,
                }
            ),
            *stitch_parts,
        ],
        ignore_index=True,
    )


# =============================================================================
# Plot
# =============================================================================


def draw_fig2_curves(
    ax, curves: pd.DataFrame, args, *, visible_curve_labels=None, use_log_y_axis: bool = True
) -> None:
    visible_labels = None if visible_curve_labels is None else set(visible_curve_labels)
    ordered_labels = list(curve_labels_for_plot_mode(args))
    if visible_curve_labels is not None:
        for label in visible_curve_labels:
            if label in FIG2_CURVE_STYLE and label not in ordered_labels:
                ordered_labels.append(label)
    for label in ordered_labels:
        if visible_labels is not None and label not in visible_labels:
            continue
        sub = curves[curves["label"] == label].sort_values("L")
        if len(sub) == 0:
            continue
        y = np.asarray(sub["intensity"], dtype=float)
        finite_positive = y[np.isfinite(y) & (y > 0)]
        y_floor = float(np.nanmin(finite_positive)) * 1e-3 if finite_positive.size else 1e-300
        if label == HT_OVER_Q2_LABEL:
            y_plot = np.where(np.isfinite(y) & (y > 0), y, np.nan)
        else:
            y_plot = np.where(np.isfinite(y) & (y > 0), y, y_floor)
        ax.plot(
            sub["L"],
            y_plot,
            label=FIG2_CURVE_DISPLAY_LABELS[label],
            **FIG2_CURVE_STYLE[label],
        )

    Lc = args.qc_inv_angstrom * args.c_angstrom / (2.0 * np.pi)
    ax.axvline(Lc, color="0.4", linestyle=":", linewidth=1.1, label=f"$L_c={Lc:.3f}$")

    l_min, l_max = validated_l_limits(args.L_min, args.L_max)
    if getattr(args, "show_bragg_markers", False):
        for index, marker_l in enumerate(getattr(args, "bragg_marker_L", DEFAULT_BRAGG_MARKER_L)):
            marker = float(marker_l)
            if marker < l_min or marker > l_max:
                continue
            ax.axvline(
                marker,
                color="0.65",
                linestyle="--",
                linewidth=0.9,
                label=r"$(00L)$ markers" if index == 0 else "_nolegend_",
            )

    ax.set_xlim(l_min, l_max)
    ax.set_yscale("log" if use_log_y_axis else "linear")
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(args.y_min, args.y_max)

    ax.set_xlabel(r"$L$ (r.l.u.)")
    ax.set_ylabel(r"$I/I_0$ or $R$")
    ax.set_title(r"Fig. 2 diagnostic: $R(Q_z)S_{\mathrm{HT},0}(L)$ near $Q_z=0$")

    def L_to_qz(values):
        return qz_from_L(values, args.c_angstrom)

    def qz_to_L(values):
        return L_from_qz(values, args.c_angstrom)

    secax = ax.secondary_xaxis("top", functions=(L_to_qz, qz_to_L))
    secax.set_xlabel(r"$Q_z$ ($\mathrm{\AA^{-1}}$)")

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def apply_l_limits_to_axis(axis, canvas, l_min: float, l_max: float) -> tuple[float, float]:
    left, right = validated_l_limits(l_min, l_max)
    axis.set_xlim(left, right)
    _redraw_canvas(canvas)
    return left, right


def apply_y_log_limits_to_axis(
    axis, canvas, y_min_log: float, y_max_log: float
) -> tuple[float, float]:
    bottom, top = validated_y_log_limits(y_min_log, y_max_log)
    axis.set_ylim(10.0**bottom, 10.0**top)
    _redraw_canvas(canvas)
    return bottom, top


def _redraw_canvas(canvas) -> None:
    if hasattr(canvas, "draw_idle"):
        canvas.draw_idle()
    else:
        canvas.draw()


def plot_fig2(curves: pd.DataFrame, args) -> Path:
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    csv_path = out_dir / (Path(args.out_name).stem + "_curves.csv")
    curves.to_csv(csv_path, index=False)
    if getattr(args, "write_stitch_diagnostics", False):
        write_stitch_diagnostics(curves, args, out_dir)
        write_stitch_diagnostic_plot(curves, args, out_dir)

    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    draw_fig2_curves(ax, curves, args)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved figure: {out_path}")
    print(f"Saved curve CSV: {csv_path}")
    return out_path


def _argv_has_option(argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def parse_args(argv=None):
    d = DEFAULTS
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Plot Fig. 2: near-Qz=0 Born-scaled HT, Fresnel-corrected HT, Parratt-envelope HT, and pure Parratt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        help="Optional existing model_curves_L_Qz.csv to redraw without recomputing HT.",
    )
    parser.add_argument("--out-dir", type=str, default=d.out_dir)
    parser.add_argument("--out-name", type=str, default=d.out_name)
    parser.add_argument(
        "--show", action="store_true", help="Show the plot interactively with plt.show()."
    )
    parser.add_argument(
        "--gui", action="store_true", help="Open an interactive Tk parameter explorer."
    )
    parser.add_argument("--dpi", type=int, default=d.dpi)

    parser.add_argument("--material", type=str, default=d.material)
    parser.add_argument("--cif-path", type=str, default="tests/fixtures/Bi2Se3.cif")
    parser.add_argument("--a-angstrom", type=float, default=d.a_angstrom)
    parser.add_argument("--c-angstrom", type=float, default=d.c_angstrom)
    parser.add_argument("--qc-inv-angstrom", type=float, default=d.qc_inv_angstrom)
    parser.add_argument("--film-density-e-per-a3", type=float, default=d.film_density_e_per_a3)
    parser.add_argument("--thickness-nm", type=float, default=d.thickness_nm)
    parser.add_argument("--substrate", type=str, default=d.substrate)
    parser.add_argument(
        "--substrate-qc-inv-angstrom", type=float, default=d.substrate_qc_inv_angstrom
    )
    parser.add_argument(
        "--substrate-density-e-per-a3", type=float, default=d.substrate_density_e_per_a3
    )
    parser.add_argument("--top-roughness-angstrom", type=float, default=d.top_roughness_angstrom)
    parser.add_argument(
        "--bottom-roughness-angstrom", type=float, default=d.bottom_roughness_angstrom
    )

    parser.add_argument("--lambda0-angstrom", type=float, default=d.lambda0_angstrom)
    parser.add_argument("--bandwidth-fwhm", type=float, default=d.bandwidth_fwhm)
    parser.add_argument("--n-wavelength-samples", type=int, default=d.n_wavelength_samples)
    parser.add_argument("--divergence-fwhm-deg", type=float, default=d.divergence_fwhm_deg)
    parser.add_argument("--n-divergence-samples", type=int, default=d.n_divergence_samples)
    parser.add_argument("--ht-structure-scale", type=float, default=d.ht_structure_scale)
    parser.add_argument("--ht-over-q2-scale", type=float, default=d.ht_over_q2_scale)

    parser.add_argument("--p-user", type=float, default=d.p_user)
    parser.add_argument(
        "--stack-layers", type=int, default=0, help="0 means round(thickness_nm*10/c_angstrom)."
    )
    parser.add_argument(
        "--ht-bandwidth-mode",
        choices=["interpolate-only", "resimulate"],
        default=d.ht_bandwidth_mode,
    )
    parser.add_argument("--L-step-dense", type=float, default=d.L_step_dense)
    parser.add_argument("--L-max-padded", type=float, default=d.L_max_padded)

    parser.add_argument("--L-min", type=float, default=d.L_min)
    parser.add_argument("--L-max", type=float, default=d.L_max)
    parser.add_argument("--L-step", type=float, default=d.L_step)
    parser.add_argument("--qz-floor-for-born", type=float, default=d.qz_floor_for_born)

    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)

    parser.add_argument(
        "--plot-mode",
        choices=["fig2", "stitch", "stitch-diagnostics"],
        default="fig2",
        help="Select the plotted curve set.",
    )
    parser.add_argument("--stitch", action="store_true", help="Compute stitched Parratt/HT curves.")
    parser.add_argument(
        "--stitch-x1-q-over-qc",
        type=float,
        default=DEFAULT_STITCH_X1_Q_OVER_QC,
        help="Low-Q stitch transition start in Qz/Qc.",
    )
    parser.add_argument(
        "--stitch-x2-q-over-qc",
        type=float,
        default=DEFAULT_STITCH_X2_Q_OVER_QC,
        help="High-Q stitch transition end in Qz/Qc.",
    )
    parser.add_argument(
        "--stitch-branch",
        choices=["normalized-ht-q2", "raw-miceli-ht-q2"],
        default="normalized-ht-q2",
        help="High-Q HT branch used for stitching.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["log-median", "manual", "miceli-cell"],
        default="log-median",
        help="Scale estimate for the stitch high-Q branch.",
    )
    parser.add_argument("--manual-stitch-scale", type=float, default=1.0)
    parser.add_argument(
        "--fit-x-min-q-over-qc",
        type=float,
        default=DEFAULT_STITCH_FIT_X_MIN_Q_OVER_QC,
    )
    parser.add_argument(
        "--fit-x-max-q-over-qc",
        type=float,
        default=DEFAULT_STITCH_FIT_X_MAX_Q_OVER_QC,
    )
    parser.add_argument(
        "--fit-exclude-centers-L",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_STITCH_EXCLUDE_CENTERS_L),
    )
    parser.add_argument(
        "--fit-exclude-half-width-L",
        type=float,
        default=DEFAULT_STITCH_EXCLUDE_HALF_WIDTH_L,
    )
    parser.add_argument(
        "--write-stitch-diagnostics",
        action="store_true",
        help="Write stitch scale diagnostics CSV and diagnostic figure.",
    )

    parser.add_argument(
        "--show-bragg-markers",
        action="store_true",
        help="Show reference marker lines for expected (00L) Bragg positions.",
    )
    parser.add_argument(
        "--bragg-marker-L",
        dest="bragg_marker_L",
        type=float,
        nargs="*",
        default=list(DEFAULT_BRAGG_MARKER_L),
        help="L positions for optional (00L) Bragg marker lines.",
    )
    parser.add_argument(
        "--bragg-view",
        action="store_true",
        help="Use a high-L range and Bragg markers for inspecting (00L) peaks.",
    )

    args = parser.parse_args(raw_argv)
    if args.bragg_view:
        args.show_bragg_markers = True
        if not _argv_has_option(raw_argv, "--L-min"):
            args.L_min = 0.0
        if not _argv_has_option(raw_argv, "--L-max"):
            args.L_max = 9.0
    return args


def run_fig2_gui(args) -> None:
    import threading
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Fig. 2 Reflectivity Parameter Explorer")
    root.geometry("1180x760")
    root.minsize(920, 620)

    current_args = args
    current_curves = None
    syncing = False
    compute_gate = LiveComputeGate()
    auto_fit_generation = 0

    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    controls = ttk.Frame(root, padding=(12, 12, 10, 12))
    controls.grid(row=0, column=0, sticky="ns")
    controls.columnconfigure(0, weight=1)
    controls.rowconfigure(0, weight=1)

    control_canvas = tk.Canvas(controls, borderwidth=0, highlightthickness=0)
    control_scrollbar = ttk.Scrollbar(
        controls,
        orient=tk.VERTICAL,
        command=control_canvas.yview,
    )
    control_canvas.configure(yscrollcommand=control_scrollbar.set)
    control_canvas.grid(row=0, column=0, sticky="nsew")
    control_scrollbar.grid(row=0, column=1, sticky="ns")

    control_body = ttk.Frame(control_canvas, padding=(8, 8, 8, 8))
    control_body.columnconfigure(0, weight=1)
    control_body_window = control_canvas.create_window(
        (0, 0),
        window=control_body,
        anchor="nw",
    )

    def _sync_control_scroll_region(_event=None) -> None:
        control_canvas.configure(scrollregion=control_canvas.bbox("all"))

    def _sync_control_body_width(event) -> None:
        control_canvas.itemconfigure(control_body_window, width=event.width)

    control_body.bind("<Configure>", _sync_control_scroll_region)
    control_canvas.bind("<Configure>", _sync_control_body_width)

    def _add_section(parent, title: str, row: int):
        section = ttk.LabelFrame(parent, text=title, padding=(8, 6, 8, 8))
        section.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        section.columnconfigure(1, weight=1)
        return section

    view_axes_section = _add_section(control_body, "Axes and limits", 0)
    curve_visibility_section = _add_section(control_body, "Displayed curves", 1)
    curve_scale_section = _add_section(control_body, "Scale factors", 2)
    beam_bandwidth_section = _add_section(control_body, "Spectral bandwidth", 3)
    beam_divergence_section = _add_section(control_body, "Angular divergence", 4)
    sample_material_section = _add_section(control_body, "Film and substrate", 5)
    sample_geometry_section = _add_section(control_body, "Film geometry", 6)

    figure_frame = ttk.Frame(root, padding=(0, 10, 12, 12))
    figure_frame.grid(row=0, column=1, sticky="nsew")
    figure_frame.columnconfigure(0, weight=1)
    figure_frame.rowconfigure(0, weight=1)

    fig = Figure(figsize=(8.8, 5.8), dpi=100)
    current_axis = fig.add_subplot()
    canvas = FigureCanvasTkAgg(fig, master=figure_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    toolbar = NavigationToolbar2Tk(canvas, figure_frame, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row=1, column=0, sticky="ew")

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(root, textvariable=status_var, anchor="w", padding=(12, 4)).grid(
        row=1, column=0, columnspan=2, sticky="ew"
    )

    state = Fig2GuiParameterState.from_args(args)

    l_min_var = tk.DoubleVar(value=float(args.L_min))
    l_max_var = tk.DoubleVar(value=float(args.L_max))
    y_min_log_var = tk.DoubleVar(value=log10_or_default(args.y_min, DEFAULT_Y_MIN_LOG10))
    y_max_log_var = tk.DoubleVar(value=log10_or_default(args.y_max, DEFAULT_Y_MAX_LOG10))
    y_log_var = tk.BooleanVar(value=True)
    film_qc_var = tk.DoubleVar(value=state.film_qc_inv_angstrom)
    film_density_var = tk.DoubleVar(value=state.film_density_e_per_a3)
    substrate_qc_var = tk.DoubleVar(value=state.substrate_qc_inv_angstrom)
    substrate_density_var = tk.DoubleVar(value=state.substrate_density_e_per_a3)
    thickness_var = tk.DoubleVar(value=state.thickness_angstrom)
    top_roughness_var = tk.DoubleVar(value=state.top_roughness_angstrom)
    bottom_roughness_var = tk.DoubleVar(value=state.bottom_roughness_angstrom)
    bandwidth_percent_var = tk.DoubleVar(value=state.bandwidth_percent)
    n_wavelength_samples_var = tk.DoubleVar(value=state.n_wavelength_samples)
    divergence_fwhm_deg_var = tk.DoubleVar(value=state.divergence_fwhm_deg)
    n_divergence_samples_var = tk.DoubleVar(value=state.n_divergence_samples)
    ht_over_q2_scale_var = tk.DoubleVar(value=state.ht_over_q2_scale)
    curve_visible_vars = {
        label: tk.BooleanVar(value=DEFAULT_GUI_CURVE_VISIBILITY[label]) for label in GUI_CURVE_ORDER
    }

    readouts: dict[str, tk.StringVar] = {}
    scales = {}
    parameter_readout_specs = (
        ("film_qc", film_qc_var, 4),
        ("film_density", film_density_var, 4),
        ("substrate_qc", substrate_qc_var, 4),
        ("substrate_density", substrate_density_var, 4),
        ("thickness", thickness_var, 1),
        ("top_roughness", top_roughness_var, 1),
        ("bottom_roughness", bottom_roughness_var, 1),
        ("bandwidth_percent", bandwidth_percent_var, 2),
        ("n_wavelength_samples", n_wavelength_samples_var, 0),
        ("divergence_fwhm_deg", divergence_fwhm_deg_var, 3),
        ("n_divergence_samples", n_divergence_samples_var, 0),
        ("ht_over_q2_scale", ht_over_q2_scale_var, 3),
    )
    state_var_specs = (
        ("film_qc_inv_angstrom", film_qc_var),
        ("film_density_e_per_a3", film_density_var),
        ("substrate_qc_inv_angstrom", substrate_qc_var),
        ("substrate_density_e_per_a3", substrate_density_var),
        ("thickness_angstrom", thickness_var),
        ("top_roughness_angstrom", top_roughness_var),
        ("bottom_roughness_angstrom", bottom_roughness_var),
        ("bandwidth_percent", bandwidth_percent_var),
        ("n_wavelength_samples", n_wavelength_samples_var),
        ("divergence_fwhm_deg", divergence_fwhm_deg_var),
        ("n_divergence_samples", n_divergence_samples_var),
        ("ht_over_q2_scale", ht_over_q2_scale_var),
    )

    def _set_status(message: str) -> None:
        status_var.set(message)

    def _format_value(value: float, digits: int) -> str:
        return f"{float(value):.{digits}f}"

    def _update_readout(name: str, value: float, digits: int) -> None:
        if name in readouts:
            readouts[name].set(_format_value(value, digits))

    def _entry_value_or_restore(name: str, variable, digits: int) -> float | None:
        try:
            return float(readouts[name].get().strip())
        except ValueError:
            _update_readout(name, variable.get(), digits)
            _set_status(f"Invalid numeric value for {name}.")
            return None

    def _expand_scale_to_value(name: str, value: float) -> None:
        scale = scales[name]
        left, right = expanded_slider_bounds(scale.cget("from"), scale.cget("to"), value)
        scale.configure(from_=left, to=right)

    def _add_slider(
        *,
        parent,
        row: int,
        name: str,
        label: str,
        variable,
        from_: float,
        to: float,
        digits: int,
        command,
        entry_command,
    ) -> None:
        readouts[name] = tk.StringVar(value=_format_value(variable.get(), digits))
        scale_min, scale_max = expanded_slider_bounds(from_, to, variable.get())
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(4, 0))
        scale = ttk.Scale(
            parent,
            from_=scale_min,
            to=scale_max,
            orient=tk.HORIZONTAL,
            variable=variable,
            command=command,
            length=210,
        )
        scales[name] = scale
        scale.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(4, 0))
        entry = ttk.Entry(parent, textvariable=readouts[name], width=10, justify="right")
        entry.grid(row=row, column=2, sticky="e", pady=(4, 0))

        def _apply_entry(_event=None):
            entry_command(name, variable, digits)
            return "break"

        entry.bind("<Return>", _apply_entry)
        entry.bind("<FocusOut>", _apply_entry)

    def _sync_parameter_vars_from_state() -> None:
        _set_vars_silently((variable, getattr(state, attr)) for attr, variable in state_var_specs)
        _refresh_parameter_readouts()

    def _refresh_parameter_readouts() -> None:
        for name, variable, digits in parameter_readout_specs:
            _update_readout(name, variable.get(), digits)

    def _capture_state_from_vars() -> None:
        for attr, variable in state_var_specs:
            setattr(state, attr, float(variable.get()))

    def _set_vars_silently(pairs) -> None:
        nonlocal syncing
        syncing = True
        try:
            for variable, value in pairs:
                variable.set(value)
        finally:
            syncing = False

    def _sync_qc_change(material_kind: str, qc_attr: str, density_attr: str, qc_var, density_var):
        setattr(state, qc_attr, float(qc_var.get()))
        state.sync_density_from_qc(material_kind)
        _set_vars_silently(((density_var, getattr(state, density_attr)),))
        _refresh_parameter_readouts()
        _schedule_live_recompute()

    def _sync_density_change(
        material_kind: str, qc_attr: str, density_attr: str, qc_var, density_var
    ):
        setattr(state, density_attr, float(density_var.get()))
        state.sync_qc_from_density(material_kind)
        _set_vars_silently(((qc_var, getattr(state, qc_attr)),))
        _refresh_parameter_readouts()
        _schedule_live_recompute()

    def _on_qc_changed(material_kind: str, qc_attr: str, density_attr: str, qc_var, density_var):
        def _handler(_value: str) -> None:
            if syncing:
                return
            _sync_qc_change(material_kind, qc_attr, density_attr, qc_var, density_var)

        return _handler

    def _on_density_changed(
        material_kind: str, qc_attr: str, density_attr: str, qc_var, density_var
    ):
        def _handler(_value: str) -> None:
            if syncing:
                return
            _sync_density_change(material_kind, qc_attr, density_attr, qc_var, density_var)

        return _handler

    def _on_plain_parameter_changed(name: str, variable, digits: int):
        def _handler(_value: str) -> None:
            if syncing:
                return
            _update_readout(name, variable.get(), digits)
            _schedule_live_recompute()

        return _handler

    def _apply_l_entry(name: str, variable, digits: int) -> None:
        value = _entry_value_or_restore(name, variable, digits)
        if value is None:
            return
        _expand_scale_to_value(name, value)
        _set_vars_silently(((variable, value),))
        _on_l_limits_changed()

    def _apply_y_entry(name: str, variable, digits: int) -> None:
        value = _entry_value_or_restore(name, variable, digits)
        if value is None:
            return
        _expand_scale_to_value(name, value)
        _set_vars_silently(((variable, value),))
        _on_y_limits_changed()

    def _apply_plain_entry(name: str, variable, digits: int, *, expand_scale: bool = False) -> None:
        value = _entry_value_or_restore(name, variable, digits)
        if value is None:
            return
        if expand_scale:
            _expand_scale_to_value(name, value)
        _set_vars_silently(((variable, value),))
        _update_readout(name, variable.get(), digits)
        _schedule_live_recompute()

    def _apply_qc_entry(
        name: str,
        variable,
        digits: int,
        material_kind: str,
        qc_attr: str,
        density_attr: str,
        density_var,
    ) -> None:
        value = _entry_value_or_restore(name, variable, digits)
        if value is None:
            return
        _set_vars_silently(((variable, value),))
        _sync_qc_change(material_kind, qc_attr, density_attr, variable, density_var)

    def _apply_density_entry(
        name: str,
        variable,
        digits: int,
        material_kind: str,
        qc_attr: str,
        density_attr: str,
        qc_var,
    ) -> None:
        value = _entry_value_or_restore(name, variable, digits)
        if value is None:
            return
        _set_vars_silently(((variable, value),))
        _sync_density_change(material_kind, qc_attr, density_attr, qc_var, variable)

    def _refresh_l_readouts() -> None:
        _update_readout("l_min", l_min_var.get(), 3)
        _update_readout("l_max", l_max_var.get(), 3)

    def _refresh_y_readouts() -> None:
        _update_readout("y_min_log", y_min_log_var.get(), 2)
        _update_readout("y_max_log", y_max_log_var.get(), 2)

    def _on_l_limits_changed(_value: str | None = None) -> None:
        if syncing:
            return
        left, right = apply_l_limits_to_axis(current_axis, canvas, l_min_var.get(), l_max_var.get())
        if "l_min" in scales:
            _expand_scale_to_value("l_min", left)
        if "l_max" in scales:
            _expand_scale_to_value("l_max", right)
        _set_vars_silently(((l_min_var, left), (l_max_var, right)))
        current_args.L_min = left
        current_args.L_max = right
        _refresh_l_readouts()
        _set_status("Updated visible L range.")
        _schedule_live_recompute()

    def _on_y_limits_changed(_value: str | None = None) -> None:
        if syncing:
            return
        bottom, top = apply_y_log_limits_to_axis(
            current_axis, canvas, y_min_log_var.get(), y_max_log_var.get()
        )
        if "y_min_log" in scales:
            _expand_scale_to_value("y_min_log", bottom)
        if "y_max_log" in scales:
            _expand_scale_to_value("y_max_log", top)
        _set_vars_silently(((y_min_log_var, bottom), (y_max_log_var, top)))
        current_args.y_min = 10.0**bottom
        current_args.y_max = 10.0**top
        _refresh_y_readouts()
        _set_status("Updated visible Y range.")

    def _selected_curve_labels() -> tuple[str, ...]:
        return tuple(label for label in GUI_CURVE_ORDER if curve_visible_vars[label].get())

    def _stitch_curve_visible() -> bool:
        return curve_visible_vars[STITCHED_PARRATT_HT_OVER_Q2_LABEL].get()

    def _draw_curves(curves: pd.DataFrame, draw_args) -> None:
        nonlocal current_axis
        fig.clear()
        current_axis = fig.add_subplot()
        draw_fig2_curves(
            current_axis,
            curves,
            draw_args,
            visible_curve_labels=_selected_curve_labels(),
            use_log_y_axis=y_log_var.get(),
        )
        fig.tight_layout()
        canvas.draw_idle()

    def _on_y_scale_changed() -> None:
        if current_curves is None:
            _set_status("Y-axis scale will apply after curves load.")
            return
        _draw_curves(current_curves, current_args)
        _set_status("Updated Y-axis scale.")

    def _stitch_curve_missing(curves: pd.DataFrame) -> bool:
        return STITCHED_PARRATT_HT_OVER_Q2_LABEL not in set(curves["label"])

    def _on_curve_visibility_changed() -> None:
        if current_curves is None:
            _set_status("Curve visibility will apply after curves load.")
            return
        if _stitch_curve_visible() and _stitch_curve_missing(current_curves):
            _schedule_live_recompute()
            return
        _draw_curves(current_curves, current_args)
        _set_status("Updated curve visibility.")

    def _auto_fit_ht_over_q2_scale() -> None:
        _capture_state_from_vars()
        source_args = state.updated_args(current_args)
        source_args.L_min, source_args.L_max = validated_l_limits(l_min_var.get(), l_max_var.get())
        source_args.from_csv = None

        if current_curves is None or not curves_span_l_range(
            current_curves,
            DEFAULT_AUTOFIT_L_MIN,
            DEFAULT_AUTOFIT_L_MAX,
        ):
            _start_auto_fit_ht_over_q2_scale(source_args)
            return

        try:
            fitted_scale = fit_ht_over_q2_scale_to_parratt(
                current_curves,
                current_scale=ht_over_q2_scale_var.get(),
            )
        except ValueError as exc:
            _set_status(f"Auto-fit HT/Qz² failed: {exc}")
            return

        _finish_auto_fit_ht_over_q2_scale(None, fitted_scale)

    def _start_auto_fit_ht_over_q2_scale(source_args) -> None:
        nonlocal auto_fit_generation
        auto_fit_generation += 1
        generation = auto_fit_generation
        fit_args = autofit_args_for_l_range(source_args)
        _set_save_enabled(False)
        _set_status("Auto-fitting HT/Qz² scale...")

        def _worker() -> None:
            try:
                fit_curves = _load_or_compute(fit_args)
                fitted_scale = fit_ht_over_q2_scale_to_parratt(
                    fit_curves,
                    current_scale=fit_args.ht_over_q2_scale,
                )
            except Exception as exc:
                root.after(
                    0,
                    lambda exc=exc, generation=generation: _finish_auto_fit_ht_over_q2_scale_error(
                        generation, exc
                    ),
                )
                return
            root.after(
                0,
                lambda generation=generation, fitted_scale=fitted_scale: (
                    _finish_auto_fit_ht_over_q2_scale(generation, fitted_scale)
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_auto_fit_ht_over_q2_scale(
        generation: int | None,
        fitted_scale: float,
    ) -> None:
        if generation is not None and generation != auto_fit_generation:
            return
        _set_save_enabled(True)
        if "ht_over_q2_scale" in scales:
            _expand_scale_to_value("ht_over_q2_scale", fitted_scale)
        _set_vars_silently(((ht_over_q2_scale_var, fitted_scale),))
        _update_readout("ht_over_q2_scale", fitted_scale, 3)
        _set_status(f"Auto-fit HT/Qz² scale = {fitted_scale:.4g}.")
        _schedule_live_recompute()

    def _finish_auto_fit_ht_over_q2_scale_error(generation: int, exc: Exception) -> None:
        if generation != auto_fit_generation:
            return
        _set_save_enabled(True)
        _set_status(f"Auto-fit HT/Qz² failed: {exc}")

    def _set_save_enabled(enabled: bool) -> None:
        save_button.configure(state="normal" if enabled else "disabled")

    def _load_or_compute(draw_args):
        if draw_args.from_csv:
            return load_curves_from_csv(Path(draw_args.from_csv), draw_args)
        return compute_curves(draw_args)

    def _start_compute(draw_args, reason: str, generation: int) -> None:
        _set_save_enabled(False)
        _set_status(reason)

        def _worker() -> None:
            try:
                curves = _load_or_compute(draw_args)
            except Exception as exc:
                root.after(
                    0, lambda exc=exc, generation=generation: _finish_compute_error(generation, exc)
                )
                return
            root.after(
                0,
                lambda curves=curves, draw_args=draw_args, generation=generation: (
                    _finish_compute_success(generation, curves, draw_args)
                ),
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_compute_success(generation: int, curves: pd.DataFrame, draw_args) -> None:
        nonlocal current_args, current_curves
        should_draw = compute_gate.is_current(generation)
        next_compute = compute_gate.finish_live_compute(generation)
        if should_draw:
            current_args = draw_args
            current_curves = curves
            _draw_curves(curves, draw_args)
        if next_compute is not None:
            next_generation, next_args = next_compute
            _start_compute(next_args, "Computing latest slider value...", next_generation)
            return
        if should_draw:
            if _stitch_curve_visible() and _stitch_curve_missing(curves):
                _schedule_live_recompute()
                return
            _set_save_enabled(True)
            _set_status("Curves updated.")

    def _finish_compute_error(generation: int, exc: Exception) -> None:
        should_report = compute_gate.is_current(generation)
        next_compute = compute_gate.finish_live_compute(generation)
        if next_compute is not None:
            next_generation, next_args = next_compute
            _start_compute(next_args, "Computing latest slider value...", next_generation)
            return
        _set_save_enabled(True)
        if should_report:
            _set_status(f"Compute failed: {exc}")

    def _schedule_live_recompute() -> None:
        _capture_state_from_vars()
        updated = state.updated_args(current_args)
        updated.L_min, updated.L_max = validated_l_limits(l_min_var.get(), l_max_var.get())
        updated.from_csv = None
        updated.stitch = _stitch_curve_visible()
        next_compute = compute_gate.request_live_compute(updated)
        if next_compute is None:
            _set_status("Parameter changed. Latest slider value queued...")
            return
        generation, draw_args = next_compute
        _start_compute(draw_args, "Computing curves with updated parameters...", generation)

    def _reset_defaults() -> None:
        default_args = parse_args([])
        default_state = Fig2GuiParameterState.from_args(default_args)
        for attr, value in vars(default_state).items():
            setattr(state, attr, value)
        _sync_parameter_vars_from_state()
        _set_vars_silently(
            (
                (l_min_var, default_args.L_min),
                (l_max_var, default_args.L_max),
                (y_min_log_var, log10_or_default(default_args.y_min, DEFAULT_Y_MIN_LOG10)),
                (y_max_log_var, log10_or_default(default_args.y_max, DEFAULT_Y_MAX_LOG10)),
            )
        )
        current_args.y_min = default_args.y_min
        current_args.y_max = default_args.y_max
        y_log_var.set(True)
        for label, variable in curve_visible_vars.items():
            variable.set(DEFAULT_GUI_CURVE_VISIBILITY[label])
        _refresh_l_readouts()
        _refresh_y_readouts()
        _schedule_live_recompute()

    def _save_current_figure() -> None:
        out_dir = Path(current_args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / current_args.out_name
        fig.savefig(out_path, dpi=current_args.dpi)
        if current_curves is not None:
            csv_path = out_dir / (Path(current_args.out_name).stem + "_curves.csv")
            current_curves.to_csv(csv_path, index=False)
            _set_status(f"Saved figure: {out_path}; curves: {csv_path}")
        else:
            _set_status(f"Saved figure: {out_path}")

    def _add_linked_material_controls(
        *,
        parent,
        row: int,
        material_kind: str,
        qc_name: str,
        qc_label: str,
        qc_var,
        qc_bounds: tuple[float, float],
        density_name: str,
        density_label: str,
        density_var,
        density_bounds: tuple[float, float],
        qc_attr: str,
        density_attr: str,
    ) -> None:
        _add_slider(
            parent=parent,
            row=row,
            name=qc_name,
            label=qc_label,
            variable=qc_var,
            from_=qc_bounds[0],
            to=qc_bounds[1],
            digits=4,
            command=_on_qc_changed(
                material_kind,
                qc_attr,
                density_attr,
                qc_var,
                density_var,
            ),
            entry_command=lambda name, variable, digits: _apply_qc_entry(
                name,
                variable,
                digits,
                material_kind,
                qc_attr,
                density_attr,
                density_var,
            ),
        )
        _add_slider(
            parent=parent,
            row=row + 1,
            name=density_name,
            label=density_label,
            variable=density_var,
            from_=density_bounds[0],
            to=density_bounds[1],
            digits=4,
            command=_on_density_changed(
                material_kind,
                qc_attr,
                density_attr,
                qc_var,
                density_var,
            ),
            entry_command=lambda name, variable, digits: _apply_density_entry(
                name,
                variable,
                digits,
                material_kind,
                qc_attr,
                density_attr,
                qc_var,
            ),
        )

    ttk.Label(view_axes_section, text="Visible range").grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    ttk.Checkbutton(
        view_axes_section,
        text="Log Y axis",
        variable=y_log_var,
        command=_on_y_scale_changed,
    ).grid(row=0, column=2, sticky="e")
    _add_slider(
        parent=view_axes_section,
        row=1,
        name="l_min",
        label="L min",
        variable=l_min_var,
        from_=0.0,
        to=0.2,
        digits=3,
        command=_on_l_limits_changed,
        entry_command=_apply_l_entry,
    )
    _add_slider(
        parent=view_axes_section,
        row=2,
        name="l_max",
        label="L max",
        variable=l_max_var,
        from_=0.02,
        to=1.0,
        digits=3,
        command=_on_l_limits_changed,
        entry_command=_apply_l_entry,
    )
    _add_slider(
        parent=view_axes_section,
        row=3,
        name="y_min_log",
        label="Y min (log10)",
        variable=y_min_log_var,
        from_=-12.0,
        to=30.0,
        digits=2,
        command=_on_y_limits_changed,
        entry_command=_apply_y_entry,
    )
    _add_slider(
        parent=view_axes_section,
        row=4,
        name="y_max_log",
        label="Y max (log10)",
        variable=y_max_log_var,
        from_=-12.0,
        to=30.0,
        digits=2,
        command=_on_y_limits_changed,
        entry_command=_apply_y_entry,
    )

    ttk.Label(curve_visibility_section, text="Curve visibility").grid(
        row=0, column=0, columnspan=3, sticky="w"
    )
    curve_button_frame = ttk.Frame(curve_visibility_section)
    curve_button_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
    for column in range(2):
        curve_button_frame.columnconfigure(column, weight=1)
    for index, label in enumerate(GUI_CURVE_ORDER):
        ttk.Checkbutton(
            curve_button_frame,
            text=label,
            variable=curve_visible_vars[label],
            command=_on_curve_visibility_changed,
            style="Toolbutton",
        ).grid(
            row=index // 2,
            column=index % 2,
            sticky="ew",
            padx=(0, 4) if index % 2 == 0 else (4, 0),
            pady=(0, 4),
        )

    _add_slider(
        parent=curve_scale_section,
        row=0,
        name="ht_over_q2_scale",
        label="HT / Qz² scale",
        variable=ht_over_q2_scale_var,
        from_=0.0,
        to=10.0,
        digits=3,
        command=_on_plain_parameter_changed("ht_over_q2_scale", ht_over_q2_scale_var, 3),
        entry_command=lambda name, variable, digits: _apply_plain_entry(
            name, variable, digits, expand_scale=True
        ),
    )
    ttk.Button(
        curve_scale_section,
        text="Auto-fit HT/Qz²",
        command=_auto_fit_ht_over_q2_scale,
    ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    _add_slider(
        parent=beam_bandwidth_section,
        row=0,
        name="bandwidth_percent",
        label="Bandwidth FWHM (%)",
        variable=bandwidth_percent_var,
        from_=0.0,
        to=20.0,
        digits=2,
        command=_on_plain_parameter_changed("bandwidth_percent", bandwidth_percent_var, 2),
        entry_command=_apply_plain_entry,
    )
    _add_slider(
        parent=beam_bandwidth_section,
        row=1,
        name="n_wavelength_samples",
        label="Wavelength samples",
        variable=n_wavelength_samples_var,
        from_=1.0,
        to=501.0,
        digits=0,
        command=_on_plain_parameter_changed("n_wavelength_samples", n_wavelength_samples_var, 0),
        entry_command=lambda name, variable, digits: _apply_plain_entry(
            name, variable, digits, expand_scale=True
        ),
    )
    _add_slider(
        parent=beam_divergence_section,
        row=0,
        name="divergence_fwhm_deg",
        label="Divergence FWHM (deg)",
        variable=divergence_fwhm_deg_var,
        from_=0.0,
        to=2.0,
        digits=3,
        command=_on_plain_parameter_changed("divergence_fwhm_deg", divergence_fwhm_deg_var, 3),
        entry_command=_apply_plain_entry,
    )
    _add_slider(
        parent=beam_divergence_section,
        row=1,
        name="n_divergence_samples",
        label="Divergence samples",
        variable=n_divergence_samples_var,
        from_=1.0,
        to=501.0,
        digits=0,
        command=_on_plain_parameter_changed("n_divergence_samples", n_divergence_samples_var, 0),
        entry_command=lambda name, variable, digits: _apply_plain_entry(
            name, variable, digits, expand_scale=True
        ),
    )

    _add_linked_material_controls(
        parent=sample_material_section,
        row=0,
        material_kind="film",
        qc_name="film_qc",
        qc_label="Film Qc (Å⁻¹)",
        qc_var=film_qc_var,
        qc_bounds=(0.02, 0.08),
        density_name="film_density",
        density_label="Film ρ (e/Å³)",
        density_var=film_density_var,
        density_bounds=(0.2, 5.0),
        qc_attr="film_qc_inv_angstrom",
        density_attr="film_density_e_per_a3",
    )
    _add_linked_material_controls(
        parent=sample_material_section,
        row=2,
        material_kind="substrate",
        qc_name="substrate_qc",
        qc_label="Substrate Qc (Å⁻¹)",
        qc_var=substrate_qc_var,
        qc_bounds=(0.005, 0.06),
        density_name="substrate_density",
        density_label="Substrate ρ (e/Å³)",
        density_var=substrate_density_var,
        density_bounds=(0.05, 2.5),
        qc_attr="substrate_qc_inv_angstrom",
        density_attr="substrate_density_e_per_a3",
    )

    _add_slider(
        parent=sample_geometry_section,
        row=0,
        name="thickness",
        label="d (Å)",
        variable=thickness_var,
        from_=50.0,
        to=2000.0,
        digits=1,
        command=_on_plain_parameter_changed("thickness", thickness_var, 1),
        entry_command=lambda name, variable, digits: _apply_plain_entry(
            name, variable, digits, expand_scale=True
        ),
    )
    _add_slider(
        parent=sample_geometry_section,
        row=1,
        name="top_roughness",
        label="σt (Å)",
        variable=top_roughness_var,
        from_=0.0,
        to=50.0,
        digits=1,
        command=_on_plain_parameter_changed("top_roughness", top_roughness_var, 1),
        entry_command=_apply_plain_entry,
    )
    _add_slider(
        parent=sample_geometry_section,
        row=2,
        name="bottom_roughness",
        label="σb (Å)",
        variable=bottom_roughness_var,
        from_=0.0,
        to=50.0,
        digits=1,
        command=_on_plain_parameter_changed("bottom_roughness", bottom_roughness_var, 1),
        entry_command=_apply_plain_entry,
    )

    button_row = ttk.Frame(controls)
    button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    button_row.columnconfigure(0, weight=1)
    button_row.columnconfigure(1, weight=1)
    reset_button = ttk.Button(button_row, text="Reset defaults", command=_reset_defaults)
    save_button = ttk.Button(button_row, text="Save figure", command=_save_current_figure)
    reset_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    save_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    _refresh_l_readouts()
    _refresh_y_readouts()
    _refresh_parameter_readouts()
    initial_compute = compute_gate.request_live_compute(args)
    if initial_compute is not None:
        initial_generation, initial_args = initial_compute
        _start_compute(initial_args, "Computing initial curves...", initial_generation)
    root.mainloop()


def main(argv=None):
    args = parse_args(argv)

    if args.gui:
        run_fig2_gui(args)
        return

    if args.from_csv:
        curves = load_curves_from_csv(Path(args.from_csv), args)
    else:
        curves = compute_curves(args)

    plot_fig2(curves, args)


if __name__ == "__main__":
    main()
