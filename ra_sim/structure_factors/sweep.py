"""One-knob structure-factor sweep helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

import numpy as np

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_f_debug_payload


def _get(row: Any, name: str):
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _sim_rows(
    reference_rows: Iterable[Any],
    cif_path: str,
    wavelength_angstrom: float,
    options: StructureFactorOptions,
) -> list[dict[str, float | int | None]]:
    payloads = []
    for row in reference_rows:
        payload = compute_raw_f_debug_payload(
            cif_path,
            (int(_get(row, "h")), int(_get(row, "k")), int(_get(row, "l"))),
            wavelength_angstrom,
            options,
        )
        payloads.append(
            {
                "h": payload["h"],
                "k": payload["k"],
                "l": payload["l"],
                "d": payload["d"],
                "two_theta": payload["two_theta"],
                "f_real": payload["f_real"],
                "f_imag": payload["f_imag"],
                "f_abs": payload["f_abs"],
            }
        )
    return payloads


def _summarize(reference_rows: Iterable[Any], sim_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    joined = []
    sim_by_hkl = {(row["h"], row["k"], row["l"]): row for row in sim_rows}
    for ref in reference_rows:
        sim = sim_by_hkl[(int(_get(ref, "h")), int(_get(ref, "k")), int(_get(ref, "l")))]
        ref_abs = float(_get(ref, "f_abs"))
        joined.append(
            {
                "hkl": [int(_get(ref, "h")), int(_get(ref, "k")), int(_get(ref, "l"))],
                "abs_error": abs(float(sim["f_abs"]) - ref_abs),
                "rel_error": abs(float(sim["f_abs"]) - ref_abs) / ref_abs,
                "real_error": abs(float(sim["f_real"]) - float(_get(ref, "f_real"))),
                "imag_error": abs(float(sim["f_imag"]) - float(_get(ref, "f_imag"))),
                "d_error": abs(float(sim["d"]) - float(_get(ref, "d"))),
                "two_theta_error": None
                if _get(ref, "two_theta") is None or sim["two_theta"] is None
                else abs(float(sim["two_theta"]) - float(_get(ref, "two_theta"))),
            }
        )
    abs_errors = np.array([row["abs_error"] for row in joined], dtype=float)
    rel_errors = np.array([row["rel_error"] for row in joined], dtype=float)
    real_errors = np.array([row["real_error"] for row in joined], dtype=float)
    imag_errors = np.array([row["imag_error"] for row in joined], dtype=float)
    d_errors = np.array([row["d_error"] for row in joined], dtype=float)
    two_theta_errors = np.array(
        [row["two_theta_error"] for row in joined if row["two_theta_error"] is not None],
        dtype=float,
    )
    worst = max(joined, key=lambda row: row["rel_error"])
    return {
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_rel_error": float(np.median(rel_errors)),
        "mean_rel_error": float(np.mean(rel_errors)),
        "max_rel_error": float(np.max(rel_errors)),
        "mean_f_real_abs_error": float(np.mean(real_errors)),
        "mean_f_imag_abs_error": float(np.mean(imag_errors)),
        "finite_two_theta_count": int(len(two_theta_errors)),
        "mean_two_theta_abs_error": float(np.mean(two_theta_errors))
        if len(two_theta_errors)
        else None,
        "max_two_theta_abs_error": float(np.max(two_theta_errors))
        if len(two_theta_errors)
        else None,
        "max_d_abs_error": float(np.max(d_errors)) if len(d_errors) else None,
        "worst_hkl": worst["hkl"],
    }


def build_switch_sweep(
    reference_rows: Iterable[Any],
    cif_path: str,
    wavelength_angstrom: float,
    baseline_options: StructureFactorOptions | None = None,
) -> list[dict[str, Any]]:
    rows = list(reference_rows)
    baseline = baseline_options or StructureFactorOptions.vesta_cu_ka1()
    variants = [
        ("baseline", baseline, wavelength_angstrom, []),
        (
            "wavelength_ui_1p5409",
            baseline,
            1.5409,
            ["wavelength_angstrom"],
        ),
        (
            "anomalous_off",
            replace(baseline, anomalous_mode="off"),
            wavelength_angstrom,
            ["anomalous_mode"],
        ),
        (
            "anomalous_xraydb",
            replace(baseline, anomalous_mode="xraydb"),
            wavelength_angstrom,
            ["anomalous_mode"],
        ),
        (
            "debye_waller_off",
            replace(baseline, debye_waller_mode="off"),
            wavelength_angstrom,
            ["debye_waller_mode"],
        ),
        (
            "unit_occupancy",
            replace(baseline, occupancy_mode="unit"),
            wavelength_angstrom,
            ["occupancy_mode"],
        ),
        (
            "itc_table",
            replace(baseline, scattering_table="itc"),
            wavelength_angstrom,
            ["scattering_table"],
        ),
    ]
    output = []
    for name, options, wavelength, changed in variants:
        sim = _sim_rows(rows, str(cif_path), wavelength, options)
        output.append(
            {
                "name": name,
                "changed_options": changed,
                "wavelength_angstrom": wavelength,
                "options": options.to_dict(),
                "summary": _summarize(rows, sim),
            }
        )
    return output
