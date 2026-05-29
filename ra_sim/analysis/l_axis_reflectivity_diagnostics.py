"""L-axis reflectivity diagnostic helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import patches

matplotlib.use("Agg", force=False)
from matplotlib import pyplot as plt

from ra_sim.analysis.parratt import (
    ParrattLayer,
    born_fresnel_asymptote,
    fresnel_reflectivity_single_interface,
    miceli_correction_factor,
    parratt_reflectivity,
)
from ra_sim.utils.stacking_fault import ht_Iinf_dict


DEFAULT_QZ_FLOOR_FOR_BORN_SCALING = 1.0e-8


@dataclass(frozen=True)
class LAxisDiagnosticMaterial:
    name: str
    cif_path: str
    a_angstrom: float
    c_angstrom: float
    qc_inv_angstrom: float
    layers: list[ParrattLayer]
    phase_delta_expression: str | None = None
    phi_l_divisor: float = 1.0
    stack_layers: int = 100


MODEL_CURVE_FIELDNAMES = [
    "material",
    "curve_name",
    "L",
    "Qz_Ainv",
    "intensity",
    "source",
    "p_user",
    "stack_layers",
    "thickness_nm",
    "bandwidth_fwhm",
    "bandwidth_mode",
    "substrate",
    "n_wavelength_samples",
    "lambda0_angstrom",
    "qz_floor_for_born_scaling",
]

MARKER_FIELDNAMES = ["material", "feature", "Qz_Ainv", "L", "source"]


def l_to_qz(L, c_angstrom: float):
    return 2.0 * np.pi * np.asarray(L, dtype=float) / float(c_angstrom)


def qz_to_l(qz, c_angstrom: float):
    return np.asarray(qz, dtype=float) * float(c_angstrom) / (2.0 * np.pi)


def add_l_qz_top_axis(ax, c_angstrom: float):
    ax.set_xlabel("L")
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda values: l_to_qz(values, c_angstrom),
            lambda values: qz_to_l(values, c_angstrom),
        ),
    )
    secax.set_xlabel("Q_z (A^-1)")
    return secax


def get_hk_curve(curves, h=0, k=0):
    if (h, k) in curves:
        return curves[(h, k)]
    if f"{h},{k}" in curves:
        return curves[f"{h},{k}"]
    if f"({h}, {k})" in curves:
        return curves[f"({h}, {k})"]
    raise KeyError(f"No curve for hk=({h},{k}); keys={list(curves)}")


def stable_zero_reference(L, I):
    L_arr = np.asarray(L)
    I_arr = np.asarray(I)
    idx = np.nanargmin(np.abs(L_arr))
    ref = I_arr[idx]
    if not np.isfinite(ref) or ref <= 0:
        near = np.abs(L_arr) < 0.02
        ref = np.nanmax(I_arr[near]) if np.any(near) else np.nanmax(I_arr)
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("Could not normalize HT curve.")
    return ref


def simulate_ht_p0_dense(
    material: LAxisDiagnosticMaterial,
    *,
    lambda_angstrom: float = 1.5418,
    L_step_dense: float = 0.002,
    L_max_padded: float = 9.5,
) -> tuple[np.ndarray, np.ndarray]:
    curves = ht_Iinf_dict(
        cif_path=str(material.cif_path),
        hk_list=[(0, 0)],
        p=0.0,
        L_step=L_step_dense,
        L_max=L_max_padded,
        lambda_=lambda_angstrom,
        a_lattice=material.a_angstrom,
        c_lattice=material.c_angstrom,
        phase_delta_expression=material.phase_delta_expression,
        phi_l_divisor=material.phi_l_divisor,
        finite_stack=True,
        stack_layers=material.stack_layers,
    )
    curve = get_hk_curve(curves, h=0, k=0)
    return np.asarray(curve["L"], dtype=float), np.asarray(curve["I"], dtype=float)


def build_unaveraged_model_curves(
    material: LAxisDiagnosticMaterial,
    L,
    ht_p0_intensity,
    *,
    qz_floor_for_born_scaling: float = DEFAULT_QZ_FLOOR_FOR_BORN_SCALING,
    wavelength_angstrom: float = 1.5418,
    sf_only_intensity=None,
) -> dict[str, np.ndarray]:
    del sf_only_intensity
    L_arr = np.asarray(L, dtype=float)
    I_arr = np.asarray(ht_p0_intensity, dtype=float)
    ref = stable_zero_reference(L_arr, I_arr)
    ht_norm = I_arr / ref
    qz = l_to_qz(L_arr, material.c_angstrom)
    qz_safe = np.maximum(qz, float(qz_floor_for_born_scaling))

    born = born_fresnel_asymptote(qz_safe, material.qc_inv_angstrom) * ht_norm
    fresnel = fresnel_reflectivity_single_interface(qz_safe, material.qc_inv_angstrom) * ht_norm
    parratt = (
        parratt_reflectivity(qz_safe, material.layers, wavelength_angstrom=wavelength_angstrom)
        * ht_norm
    )

    return {
        "HT_p0_normalized": ht_norm,
        "Born_scaled_HT_p0": born,
        "Fresnel_corrected_HT_p0": fresnel,
        "Parratt_envelope_HT_p0": parratt,
        "Parratt_reflectivity": parratt_reflectivity(
            qz_safe,
            material.layers,
            wavelength_angstrom=wavelength_angstrom,
        ),
        "Miceli_correction_factor": miceli_correction_factor(
            qz_safe,
            material.qc_inv_angstrom,
        ),
    }


def wavelength_samples(lambda0: float, fwhm: float = 0.05, n: int = 241):
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    if int(n) == 1:
        return np.array([float(lambda0)], dtype=float), np.array([1.0], dtype=float)
    sigma = float(fwhm) / 2.354820045
    eps = np.linspace(-4.0 * sigma, 4.0 * sigma, int(n))
    w = np.exp(-0.5 * (eps / sigma) ** 2)
    w /= w.sum()
    return float(lambda0) * (1.0 + eps), w


def wavelength_average_parratt_on_L(
    L_nominal,
    c_angstrom,
    layers,
    lambda0=1.5418,
    bandwidth_fwhm=0.05,
    n_samples=241,
):
    L_nominal_arr = np.asarray(L_nominal, dtype=float)
    lambdas, weights = wavelength_samples(lambda0, bandwidth_fwhm, n_samples)
    out = np.zeros_like(L_nominal_arr, dtype=float)

    for lam_j, w_j in zip(lambdas, weights):
        L_true = L_nominal_arr * float(lambda0) / float(lam_j)
        qz_true = l_to_qz(L_true, c_angstrom)
        out += float(w_j) * parratt_reflectivity(
            qz_true,
            layers,
            wavelength_angstrom=float(lam_j),
        )

    return out


def wavelength_average_fresnel_born_on_qz(
    qz_nominal,
    qc_inv_angstrom: float,
    lambda0=1.5418,
    bandwidth_fwhm=0.05,
    n_samples=241,
    qz_floor_for_born_scaling: float = DEFAULT_QZ_FLOOR_FOR_BORN_SCALING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qz_nominal_arr = np.asarray(qz_nominal, dtype=float)
    lambdas, weights = wavelength_samples(lambda0, bandwidth_fwhm, n_samples)
    fresnel_avg = np.zeros_like(qz_nominal_arr, dtype=float)
    born_avg = np.zeros_like(qz_nominal_arr, dtype=float)

    for lam_j, w_j in zip(lambdas, weights):
        qz_true = qz_nominal_arr * float(lambda0) / float(lam_j)
        qz_safe = np.maximum(qz_true, float(qz_floor_for_born_scaling))
        fresnel_avg += float(w_j) * fresnel_reflectivity_single_interface(
            qz_safe,
            qc_inv_angstrom,
        )
        born_avg += float(w_j) * born_fresnel_asymptote(qz_safe, qc_inv_angstrom)

    return fresnel_avg, born_avg, fresnel_avg / born_avg


def wavelength_average_ht_p0_on_L(
    L_nominal,
    material: LAxisDiagnosticMaterial,
    lambda0=1.5418,
    bandwidth_fwhm=0.05,
    n_samples=241,
    ht_bandwidth_mode="resimulate",
):
    L_nominal_arr = np.asarray(L_nominal, dtype=float)
    lambdas, weights = wavelength_samples(lambda0, bandwidth_fwhm, n_samples)

    born_avg = np.zeros_like(L_nominal_arr, dtype=float)
    fresnel_avg = np.zeros_like(L_nominal_arr, dtype=float)
    parratt_env_avg = np.zeros_like(L_nominal_arr, dtype=float)

    cached_ht = None
    if ht_bandwidth_mode == "interpolate-only":
        cached_ht = simulate_ht_p0_dense(material, lambda_angstrom=float(lambda0))
    elif ht_bandwidth_mode != "resimulate":
        raise ValueError(
            "ht_bandwidth_mode must be 'resimulate' or 'interpolate-only'"
        )

    for lam_j, w_j in zip(lambdas, weights):
        L_true = L_nominal_arr * float(lambda0) / float(lam_j)
        qz_true = l_to_qz(L_true, material.c_angstrom)
        qz_safe = np.maximum(qz_true, DEFAULT_QZ_FLOOR_FOR_BORN_SCALING)

        if cached_ht is None:
            L_dense, I_ht_dense = simulate_ht_p0_dense(
                material,
                lambda_angstrom=float(lam_j),
            )
        else:
            L_dense, I_ht_dense = cached_ht

        I_ht_true = np.interp(L_true, L_dense, I_ht_dense)
        S = I_ht_true / stable_zero_reference(L_dense, I_ht_dense)

        born_avg += float(w_j) * born_fresnel_asymptote(
            qz_safe,
            material.qc_inv_angstrom,
        ) * S
        fresnel_avg += float(w_j) * fresnel_reflectivity_single_interface(
            qz_safe,
            material.qc_inv_angstrom,
        ) * S
        parratt_env_avg += float(w_j) * parratt_reflectivity(
            qz_safe,
            material.layers,
            wavelength_angstrom=float(lam_j),
        ) * S

    return born_avg, fresnel_avg, parratt_env_avg


def _finite_film_thickness_nm(material: LAxisDiagnosticMaterial) -> float:
    for layer in material.layers[1:-1]:
        if layer.thickness_angstrom is not None:
            return float(layer.thickness_angstrom) / 10.0
    return 0.0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def stack_caption(material: LAxisDiagnosticMaterial) -> str:
    names = []
    for layer in material.layers:
        if layer.thickness_angstrom is None:
            names.append(layer.name)
        else:
            thickness_nm = float(layer.thickness_angstrom) / 10.0
            names.append(f"{layer.name}({_format_number(thickness_nm)} nm)")
    return " / ".join(names)


def _substrate_name(material: LAxisDiagnosticMaterial) -> str:
    return material.layers[-1].name if material.layers else ""


def _ht_normalized_wavelength_average(
    L_nominal,
    material: LAxisDiagnosticMaterial,
    *,
    lambda0_angstrom: float,
    bandwidth_fwhm: float,
    n_wavelength_samples: int,
    ht_bandwidth_mode: str,
) -> np.ndarray:
    L_nominal_arr = np.asarray(L_nominal, dtype=float)
    lambdas, weights = wavelength_samples(
        lambda0_angstrom,
        bandwidth_fwhm,
        n_wavelength_samples,
    )
    out = np.zeros_like(L_nominal_arr, dtype=float)

    cached_ht = None
    if ht_bandwidth_mode == "interpolate-only":
        cached_ht = simulate_ht_p0_dense(material, lambda_angstrom=float(lambda0_angstrom))
    elif ht_bandwidth_mode != "resimulate":
        raise ValueError(
            "ht_bandwidth_mode must be 'resimulate' or 'interpolate-only'"
        )

    for lam_j, w_j in zip(lambdas, weights):
        L_true = L_nominal_arr * float(lambda0_angstrom) / float(lam_j)
        if cached_ht is None:
            L_dense, I_ht_dense = simulate_ht_p0_dense(
                material,
                lambda_angstrom=float(lam_j),
            )
        else:
            L_dense, I_ht_dense = cached_ht
        I_ht_true = np.interp(L_true, L_dense, I_ht_dense)
        out += float(w_j) * (I_ht_true / stable_zero_reference(L_dense, I_ht_dense))
    return out


def build_wavelength_averaged_model_curves(
    material: LAxisDiagnosticMaterial,
    L_values,
    *,
    lambda0_angstrom: float = 1.5418,
    bandwidth_fwhm: float = 0.05,
    n_wavelength_samples: int = 241,
    ht_bandwidth_mode: str = "resimulate",
    qz_floor_for_born_scaling: float = DEFAULT_QZ_FLOOR_FOR_BORN_SCALING,
) -> dict[str, np.ndarray]:
    L_arr = np.asarray(L_values, dtype=float)
    qz = l_to_qz(L_arr, material.c_angstrom)
    qz_safe = np.maximum(qz, float(qz_floor_for_born_scaling))

    born, fresnel, parratt_env = wavelength_average_ht_p0_on_L(
        L_arr,
        material,
        lambda0=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_samples=n_wavelength_samples,
        ht_bandwidth_mode=ht_bandwidth_mode,
    )
    ht_norm = _ht_normalized_wavelength_average(
        L_arr,
        material,
        lambda0_angstrom=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_wavelength_samples=n_wavelength_samples,
        ht_bandwidth_mode=ht_bandwidth_mode,
    )
    parratt_only = wavelength_average_parratt_on_L(
        L_arr,
        material.c_angstrom,
        material.layers,
        lambda0=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_samples=n_wavelength_samples,
    )

    fresnel_ref, born_ref, correction = wavelength_average_fresnel_born_on_qz(
        qz_safe,
        material.qc_inv_angstrom,
        lambda0=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_samples=n_wavelength_samples,
        qz_floor_for_born_scaling=qz_floor_for_born_scaling,
    )
    del fresnel_ref, born_ref

    return {
        "HT_p0_normalized": ht_norm,
        "Born_scaled_HT_p0": born,
        "Fresnel_corrected_HT_p0": fresnel,
        "Parratt_envelope_HT_p0": parratt_env,
        "Parratt_reflectivity": parratt_only,
        "Miceli_correction_factor": correction,
    }


def build_marker_rows(material: LAxisDiagnosticMaterial) -> list[dict[str, object]]:
    qc = float(material.qc_inv_angstrom)
    c = float(material.c_angstrom)
    markers = [
        ("Qc", qc, qz_to_l(qc, c), "critical wavevector"),
        ("near-beam feature", 0.0, 0.0, "axis origin"),
        ("fitted sub-003 feature", l_to_qz(2.8, c), 2.8, "diagnostic placeholder"),
        ("003 reference", l_to_qz(3.0, c), 3.0, "integer L reference"),
    ]
    return [
        {
            "material": material.name,
            "feature": feature,
            "Qz_Ainv": float(qz),
            "L": float(L_value),
            "source": source,
        }
        for feature, qz, L_value, source in markers
    ]


def build_model_curve_rows(
    material: LAxisDiagnosticMaterial,
    L_values,
    curves: dict[str, np.ndarray],
    *,
    lambda0_angstrom: float,
    bandwidth_fwhm: float,
    n_wavelength_samples: int,
    ht_bandwidth_mode: str,
    qz_floor_for_born_scaling: float = DEFAULT_QZ_FLOOR_FOR_BORN_SCALING,
) -> list[dict[str, object]]:
    L_arr = np.asarray(L_values, dtype=float)
    qz = l_to_qz(L_arr, material.c_angstrom)
    thickness_nm = _finite_film_thickness_nm(material)
    substrate = _substrate_name(material)
    source_by_curve = {
        "HT_p0_normalized": "ht_Iinf_dict p=0.0 normalized at L=0",
        "Born_scaled_HT_p0": "Born asymptote times ht_Iinf_dict p=0.0",
        "Fresnel_corrected_HT_p0": "single-interface Fresnel times ht_Iinf_dict p=0.0",
        "Parratt_envelope_HT_p0": "Parratt optical envelope times ht_Iinf_dict p=0.0",
        "Parratt_reflectivity": "Parratt recursion reflectivity",
        "Miceli_correction_factor": "Fresnel reflectivity divided by Born asymptote",
    }
    rows: list[dict[str, object]] = []
    for curve_name, intensity in curves.items():
        intensity_arr = np.asarray(intensity, dtype=float)
        for L_value, qz_value, value in zip(L_arr, qz, intensity_arr):
            rows.append(
                {
                    "material": material.name,
                    "curve_name": curve_name,
                    "L": float(L_value),
                    "Qz_Ainv": float(qz_value),
                    "intensity": float(value),
                    "source": source_by_curve.get(curve_name, ""),
                    "p_user": 0.0 if curve_name.endswith("_HT_p0") or curve_name == "HT_p0_normalized" else "",
                    "stack_layers": int(material.stack_layers),
                    "thickness_nm": float(thickness_nm),
                    "bandwidth_fwhm": float(bandwidth_fwhm),
                    "bandwidth_mode": ht_bandwidth_mode,
                    "substrate": substrate,
                    "n_wavelength_samples": int(n_wavelength_samples),
                    "lambda0_angstrom": float(lambda0_angstrom),
                    "qz_floor_for_born_scaling": float(qz_floor_for_born_scaling),
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _positive_for_log(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite_positive = arr[np.isfinite(arr) & (arr > 0.0)]
    floor = float(np.nanmin(finite_positive)) * 1.0e-3 if finite_positive.size else 1.0e-300
    return np.where(np.isfinite(arr) & (arr > 0.0), arr, floor)


def _draw_l_axis_curves(
    path: Path,
    L_values,
    curves: dict[str, np.ndarray],
    material: LAxisDiagnosticMaterial,
    curve_names: list[str],
    labels: dict[str, str],
    *,
    title: str,
    xlim: tuple[float, float] = (0.0, 9.0),
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    L_arr = np.asarray(L_values, dtype=float)
    for curve_name in curve_names:
        linewidth = 1.0 if curve_name == "Parratt_reflectivity" else 1.8
        alpha = 0.65 if curve_name == "Parratt_reflectivity" else 1.0
        ax.plot(
            L_arr,
            _positive_for_log(curves[curve_name]),
            label=labels[curve_name],
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.set_xlim(*xlim)
    ax.set_yscale("log")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    add_l_qz_top_axis(ax, material.c_angstrom)
    marker_rows = build_marker_rows(material)
    for row in marker_rows:
        L_marker = float(row["L"])
        if xlim[0] <= L_marker <= xlim[1]:
            ax.axvline(L_marker, color="0.65", linewidth=0.8, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_miceli_figures(
    output_dir: Path,
    material: LAxisDiagnosticMaterial,
    *,
    lambda0_angstrom: float,
    bandwidth_fwhm: float,
    n_wavelength_samples: int,
    qz_floor_for_born_scaling: float,
) -> None:
    x = np.geomspace(1.0001, 100.0, 600)
    qz = x * float(material.qc_inv_angstrom)
    rf, born, correction = wavelength_average_fresnel_born_on_qz(
        qz,
        material.qc_inv_angstrom,
        lambda0=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_samples=n_wavelength_samples,
        qz_floor_for_born_scaling=qz_floor_for_born_scaling,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.loglog(x, rf, label="R_F(Qz)", linewidth=1.8)
    ax.loglog(x, born, label="(Qc/2Qz)^4", linewidth=1.8)
    for marker in (1, 2, 5, 10):
        ax.axvline(marker, color="0.7", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Qz/Qc")
    ax.set_ylabel("Reflectivity")
    ax.set_title("Miceli replacement: R_F versus Born asymptote")
    ax.legend(loc="best")
    fig.savefig(output_dir / "fig_miceli_RF_vs_Born_asymptote.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), constrained_layout=True)
    axes[0].loglog(x, rf, label="R_F(Qz)", linewidth=1.8)
    axes[0].loglog(x, born, label="(Qc/2Qz)^4", linewidth=1.8)
    axes[0].set_ylabel("Reflectivity")
    axes[0].legend(loc="best")
    axes[1].semilogx(x, correction, label="C(Qz)", linewidth=1.8)
    axes[1].axhline(1.0, color="0.35", linewidth=0.9)
    axes[1].set_xlabel("Qz/Qc")
    axes[1].set_ylabel("Correction factor")
    axes[1].legend(loc="best")
    for ax in axes:
        for marker in (1, 2, 5, 10):
            ax.axvline(marker, color="0.7", linewidth=0.8, linestyle="--")
    fig.suptitle("At large Qz/Qc, exact Fresnel reflectivity approaches the Born asymptote")
    fig.savefig(output_dir / "fig_miceli_correction_factor.png", dpi=180)
    fig.savefig(output_dir / "fig_L0_L9_miceli_correction_factor.png", dpi=180)
    plt.close(fig)


def _draw_detector_roi_overlay(
    output_dir: Path,
    *,
    detector_image_path: str | None,
    detector_roi_csv: str | None,
    allow_missing_rois: bool,
) -> None:
    out_path = output_dir / "fig_detector_roi_overlay.png"
    if not detector_image_path and not detector_roi_csv:
        if not allow_missing_rois:
            raise ValueError("ROI input is required; pass --allow-missing-rois to write a placeholder.")
        fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
        ax.text(
            0.5,
            0.5,
            "Detector ROI input was not supplied.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    if detector_image_path:
        ax.imshow(plt.imread(detector_image_path), cmap="gray", origin="upper")
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)

    if detector_roi_csv:
        with Path(detector_roi_csv).open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                row_min = float(row["row_min"])
                row_max = float(row["row_max"])
                col_min = float(row["col_min"])
                col_max = float(row["col_max"])
                rect = patches.Rectangle(
                    (col_min, row_min),
                    col_max - col_min,
                    row_max - row_min,
                    fill=False,
                    linewidth=1.3,
                    edgecolor="tab:orange",
                )
                ax.add_patch(rect)
                ax.text(col_min, row_min, row.get("feature", "ROI"), color="tab:orange")
    ax.set_title("Detector ROI Overlay")
    ax.set_xlabel("Detector column")
    ax.set_ylabel("Detector row")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    material: LAxisDiagnosticMaterial,
    *,
    ht_bandwidth_mode: str,
    bandwidth_fwhm: float,
    n_wavelength_samples: int,
    lambda0_angstrom: float,
    qz_floor_for_born_scaling: float,
) -> None:
    text = f"""# L-Axis Reflectivity Diagnostics

Material: {material.name}
Stack: {stack_caption(material)}
Bandwidth mode: {ht_bandwidth_mode}
Bandwidth FWHM: {bandwidth_fwhm}
Wavelength samples: {n_wavelength_samples}
Lambda0: {lambda0_angstrom} A
Born-scaling Qz floor: {qz_floor_for_born_scaling}

The plotted HT-derived comparisons are:

- Born-scaled HT p=0: (Qc/2Qz)^4 * I_HT0/I_HT0(0)
- Fresnel-corrected HT p=0: R_F(Qz) * I_HT0/I_HT0(0)
- Parratt-envelope HT p=0: R_Parratt(Qz) * I_HT0/I_HT0(0)

Parratt-envelope HT p=0 is an optical-envelope comparison for the finite stack.
It does not prove total external reflection; it only shows whether a feature lies
on the surface-optical scale.

The Miceli-style replacement is:

(Qc/2Qz)^4 -> R_F(Qz)

or, for a finite stack:

(Qc/2Qz)^4 -> R_Parratt(Qz).

At large Qz/Qc, the exact Fresnel reflectivity approaches the Born asymptote.
Near Qc, the Born asymptote underestimates the bounded optical reflectivity.
"""
    path.write_text(text, encoding="utf-8")


def run_l_axis_reflectivity_diagnostics(
    *,
    output_dir,
    material: LAxisDiagnosticMaterial,
    L_values=None,
    lambda0_angstrom: float = 1.5418,
    bandwidth_fwhm: float = 0.05,
    n_wavelength_samples: int = 241,
    ht_bandwidth_mode: str = "resimulate",
    qz_floor_for_born_scaling: float = DEFAULT_QZ_FLOOR_FOR_BORN_SCALING,
    detector_image_path: str | None = None,
    detector_roi_csv: str | None = None,
    allow_missing_rois: bool = False,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _draw_detector_roi_overlay(
        output_path,
        detector_image_path=detector_image_path,
        detector_roi_csv=detector_roi_csv,
        allow_missing_rois=allow_missing_rois,
    )

    if L_values is None:
        L_values = np.linspace(0.0, 9.0, 1801)
    L_arr = np.asarray(L_values, dtype=float)
    curves = build_wavelength_averaged_model_curves(
        material,
        L_arr,
        lambda0_angstrom=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_wavelength_samples=n_wavelength_samples,
        ht_bandwidth_mode=ht_bandwidth_mode,
        qz_floor_for_born_scaling=qz_floor_for_born_scaling,
    )

    labels = {
        "Born_scaled_HT_p0": "Born-scaled HT p=0: (Qc/2Qz)^4 * I_HT0/I_HT0(0)",
        "Fresnel_corrected_HT_p0": "Fresnel-corrected HT p=0: R_F(Qz) * I_HT0/I_HT0(0)",
        "Parratt_envelope_HT_p0": "Parratt-envelope HT p=0: R_Parratt(Qz) * I_HT0/I_HT0(0)",
        "Parratt_reflectivity": "Parratt reflectivity",
        "HT_p0_normalized": "HT p=0 normalized",
    }
    _draw_l_axis_curves(
        output_path / "fig_L0_L9_born_vs_fresnel_corrected_HT0.png",
        L_arr,
        curves,
        material,
        ["Born_scaled_HT_p0", "Fresnel_corrected_HT_p0"],
        labels,
        title="Born-scaled HT p=0 versus Fresnel-corrected HT p=0",
    )
    _draw_l_axis_curves(
        output_path / "fig_L0_L9_born_vs_fresnel_vs_parratt_envelope_HT0.png",
        L_arr,
        curves,
        material,
        [
            "Born_scaled_HT_p0",
            "Fresnel_corrected_HT_p0",
            "Parratt_envelope_HT_p0",
            "Parratt_reflectivity",
        ],
        labels,
        title="Born, Fresnel, and Parratt-envelope HT p=0",
    )
    _draw_l_axis_curves(
        output_path / "fig_L0_L9_parratt_vs_born_ht0.png",
        L_arr,
        curves,
        material,
        ["Born_scaled_HT_p0", "Parratt_envelope_HT_p0", "Parratt_reflectivity"],
        labels,
        title="Born-scaled HT p=0 and Parratt-envelope HT p=0",
    )
    Lc = float(qz_to_l(material.qc_inv_angstrom, material.c_angstrom))
    _draw_l_axis_curves(
        output_path / "fig_near_Qc_parratt_vs_born_ht0.png",
        L_arr,
        curves,
        material,
        ["Born_scaled_HT_p0", "Fresnel_corrected_HT_p0", "Parratt_envelope_HT_p0"],
        labels,
        title="Near-Qc optical scale",
        xlim=(0.0, max(1.0, 4.0 * Lc)),
    )
    _draw_l_axis_curves(
        output_path / "fig_Qz0_parratt_flat_born_diverges.png",
        L_arr,
        curves,
        material,
        ["Born_scaled_HT_p0", "Fresnel_corrected_HT_p0", "Parratt_envelope_HT_p0"],
        labels,
        title="Qz=0: Born scaling uses a plotting floor",
        xlim=(0.0, max(0.5, 2.0 * Lc)),
    )
    _draw_l_axis_curves(
        output_path / "fig_HT_p0_vs_SF_only.png",
        L_arr,
        curves,
        material,
        ["HT_p0_normalized"],
        labels,
        title="HT p=0 normalized structure term",
    )
    _draw_miceli_figures(
        output_path,
        material,
        lambda0_angstrom=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_wavelength_samples=n_wavelength_samples,
        qz_floor_for_born_scaling=qz_floor_for_born_scaling,
    )

    model_rows = build_model_curve_rows(
        material,
        L_arr,
        curves,
        lambda0_angstrom=lambda0_angstrom,
        bandwidth_fwhm=bandwidth_fwhm,
        n_wavelength_samples=n_wavelength_samples,
        ht_bandwidth_mode=ht_bandwidth_mode,
        qz_floor_for_born_scaling=qz_floor_for_born_scaling,
    )
    marker_rows = build_marker_rows(material)
    _write_csv(output_path / "model_curves_L_Qz.csv", MODEL_CURVE_FIELDNAMES, model_rows)
    _write_csv(output_path / "marker_table_L_Qz.csv", MARKER_FIELDNAMES, marker_rows)
    _write_report(
        output_path / "report.md",
        material,
        ht_bandwidth_mode=ht_bandwidth_mode,
        bandwidth_fwhm=bandwidth_fwhm,
        n_wavelength_samples=n_wavelength_samples,
        lambda0_angstrom=lambda0_angstrom,
        qz_floor_for_born_scaling=qz_floor_for_born_scaling,
    )

    return {path.name: path for path in output_path.iterdir()}
