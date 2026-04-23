"""Helpers for VESTA structure-factor TXT parity fixtures."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPECTED_BI2SE3_CIF_SHA256 = "2a02dfceb6b302810229076479c54515382b3adcd558e3d9f874192dcd6b539f"
EXPECTED_BI2SE3_TXT_SHA256 = "b2f7e2de27db6bfee7497ad2c8e0ebf622d5db14c5da4240b7f334d6de89d379"
EXPECTED_BI2SE3_ROW_COUNT = 206
EXPECTED_BI2SE3_FINITE_TWOTHETA_COUNT = 161
EXPECTED_BI2SE3_NAN_TWOTHETA_COUNT = 45
EXPECTED_BI2SE3_WAVELENGTH_ANGSTROM = 1.5405929254021151
EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM = 1.540592865402115
HKL_COMPARISON_CSV_FIELDS = [
    "h",
    "k",
    "l",
    "vesta_d",
    "sim_d",
    "d_abs_error",
    "vesta_f_real",
    "sim_f_real",
    "f_real_abs_error",
    "vesta_f_imag",
    "sim_f_imag",
    "f_imag_abs_error",
    "vesta_f_abs",
    "sim_f_abs",
    "f_abs_abs_error",
    "f_abs_rel_error",
    "vesta_two_theta",
    "sim_two_theta",
    "two_theta_abs_error",
]


@dataclass(frozen=True)
class VestaStructureFactorRow:
    h: int
    k: int
    l: int
    d: float
    f_real: float
    f_imag: float
    f_abs: float
    two_theta: float | None
    intensity: float | None
    multiplicity: int

    @property
    def hkl(self) -> tuple[int, int, int]:
        return (self.h, self.k, self.l)


@dataclass(frozen=True)
class HklComparison:
    h: int
    k: int
    l: int
    vesta_d: float
    sim_d: float | None
    d_abs_error: float | None
    vesta_f_real: float
    sim_f_real: float
    f_real_abs_error: float
    vesta_f_imag: float
    sim_f_imag: float
    f_imag_abs_error: float
    vesta_f_abs: float
    sim_f_abs: float
    f_abs_abs_error: float
    f_abs_rel_error: float
    vesta_two_theta: float | None
    sim_two_theta: float | None
    two_theta_abs_error: float | None


def _parse_float(token: str) -> float | None:
    text = token.strip().lower()
    if text in {"nan", "+nan", "-nan", "nan(ind)", "+nan(ind)", "-nan(ind)"}:
        return None
    return float(token)


def parse_vesta_structure_factor_txt(path: str | Path) -> list[VestaStructureFactorRow]:
    """Parse VESTA text-area structure-factor rows."""

    rows: list[VestaStructureFactorRow] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            h, k, l = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue

        rows.append(
            VestaStructureFactorRow(
                h=h,
                k=k,
                l=l,
                d=float(parts[3]),
                f_real=float(parts[4]),
                f_imag=float(parts[5]),
                f_abs=float(parts[6]),
                two_theta=_parse_float(parts[7]),
                intensity=_parse_float(parts[8]),
                multiplicity=int(parts[9]),
            )
        )
    return rows


def load_fixture_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def infer_lambda_from_vesta_rows(rows: list[VestaStructureFactorRow]) -> float:
    values = [
        2.0 * row.d * math.sin(math.radians(row.two_theta / 2.0))
        for row in rows
        if row.two_theta is not None
    ]
    if not values:
        raise ValueError("No finite two_theta rows available for wavelength inference.")
    return float(np.median(values))


def _field(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source[key]
    return getattr(source, key)


def _optional_field(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def compare_hkl_tables(
    vesta_rows: list[VestaStructureFactorRow],
    sim_rows: list[Mapping[str, Any] | Any],
) -> list[HklComparison]:
    sim_by_hkl = {
        (int(_field(row, "h")), int(_field(row, "k")), int(_field(row, "l"))): row
        for row in sim_rows
    }
    comparisons: list[HklComparison] = []
    for vesta in vesta_rows:
        sim = sim_by_hkl[vesta.hkl]
        sim_real = float(_field(sim, "f_real"))
        sim_imag = float(_field(sim, "f_imag"))
        sim_abs = float(_field(sim, "f_abs"))
        sim_d = _optional_field(sim, "d")
        sim_two_theta = _optional_field(sim, "two_theta")
        d_error = None if sim_d is None else abs(float(sim_d) - vesta.d)
        two_theta_error = (
            None
            if vesta.two_theta is None or sim_two_theta is None
            else abs(float(sim_two_theta) - vesta.two_theta)
        )
        abs_error = abs(sim_abs - vesta.f_abs)
        comparisons.append(
            HklComparison(
                h=vesta.h,
                k=vesta.k,
                l=vesta.l,
                vesta_d=vesta.d,
                sim_d=None if sim_d is None else float(sim_d),
                d_abs_error=d_error,
                vesta_f_real=vesta.f_real,
                sim_f_real=sim_real,
                f_real_abs_error=abs(sim_real - vesta.f_real),
                vesta_f_imag=vesta.f_imag,
                sim_f_imag=sim_imag,
                f_imag_abs_error=abs(sim_imag - vesta.f_imag),
                vesta_f_abs=vesta.f_abs,
                sim_f_abs=sim_abs,
                f_abs_abs_error=abs_error,
                f_abs_rel_error=abs_error / vesta.f_abs,
                vesta_two_theta=vesta.two_theta,
                sim_two_theta=None if sim_two_theta is None else float(sim_two_theta),
                two_theta_abs_error=two_theta_error,
            )
        )
    return comparisons


def summarize_f_errors(comparison: list[HklComparison]) -> dict[str, Any]:
    if not comparison:
        raise ValueError("No HKL comparison rows to summarize.")

    abs_errors = np.array([row.f_abs_abs_error for row in comparison], dtype=float)
    rel_errors = np.array([row.f_abs_rel_error for row in comparison], dtype=float)
    real_errors = np.array([row.f_real_abs_error for row in comparison], dtype=float)
    imag_errors = np.array([row.f_imag_abs_error for row in comparison], dtype=float)
    sim_abs = np.array([row.sim_f_abs for row in comparison], dtype=float)
    vesta_abs = np.array([row.vesta_f_abs for row in comparison], dtype=float)
    worst = max(comparison, key=lambda row: row.f_abs_rel_error)

    return {
        "count": len(comparison),
        "mean_abs_error": float(np.mean(abs_errors)),
        "median_abs_error": float(np.median(abs_errors)),
        "max_abs_error": float(np.max(abs_errors)),
        "mean_rel_error": float(np.mean(rel_errors)),
        "median_rel_error": float(np.median(rel_errors)),
        "max_rel_error": float(np.max(rel_errors)),
        "p95_rel_error": float(np.percentile(rel_errors, 95)),
        "mean_f_real_abs_error": float(np.mean(real_errors)),
        "mean_f_imag_abs_error": float(np.mean(imag_errors)),
        "correlation": float(np.corrcoef(vesta_abs, sim_abs)[0, 1]),
        "worst_hkl": [worst.h, worst.k, worst.l],
    }


def summarize_geometry_errors(comparison: list[HklComparison]) -> dict[str, Any]:
    if not comparison:
        raise ValueError("No HKL comparison rows to summarize.")
    d_errors = np.array(
        [row.d_abs_error for row in comparison if row.d_abs_error is not None],
        dtype=float,
    )
    two_theta_errors = np.array(
        [row.two_theta_abs_error for row in comparison if row.two_theta_abs_error is not None],
        dtype=float,
    )
    return {
        "finite_two_theta_count": int(len(two_theta_errors)),
        "mean_two_theta_abs_error": float(np.mean(two_theta_errors))
        if len(two_theta_errors)
        else None,
        "max_two_theta_abs_error": float(np.max(two_theta_errors))
        if len(two_theta_errors)
        else None,
        "max_d_abs_error": float(np.max(d_errors)) if len(d_errors) else None,
    }


def residuals_binned_by_s(
    comparison: list[HklComparison],
) -> dict[str, dict[str, float | int]]:
    ordered = sorted(comparison, key=lambda row: 1.0 / (2.0 * row.vesta_d))
    splits = np.array_split(np.array(ordered, dtype=object), 3)
    names = ("low_s", "mid_s", "high_s")
    result: dict[str, dict[str, float | int]] = {}
    for name, group in zip(names, splits, strict=True):
        rows = list(group)
        rel = np.array([row.f_abs_rel_error for row in rows], dtype=float)
        s_values = np.array([1.0 / (2.0 * row.vesta_d) for row in rows], dtype=float)
        result[name] = {
            "count": len(rows),
            "mean_s": float(np.mean(s_values)),
            "mean_rel_error": float(np.mean(rel)),
        }
    return result


def write_comparison_csv(path: str | Path, comparison: list[HklComparison]) -> None:
    rows = [asdict(row) for row in comparison]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HKL_COMPARISON_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
