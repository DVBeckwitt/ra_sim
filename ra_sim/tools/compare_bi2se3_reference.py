"""Compare Bi2Se3 VESTA parity TXT against raw structure factors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_f_debug_payload
from ra_sim.validation.vesta_reference import (
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    compare_hkl_tables,
    infer_lambda_from_vesta_rows,
    parse_vesta_structure_factor_txt,
    summarize_f_errors,
    summarize_geometry_errors,
    write_comparison_csv,
)


DEFAULT_REFERENCE_PATH = Path("tests/fixtures/Bi2Se3_vesta_cu_ka1_dmin_0p7.txt")
DEFAULT_CIF_PATH = Path("tests/fixtures/Bi2Se3.cif")
VESTA_FACTOR_CONVENTION = "VESTA Cu Kalpha1 legacy mode"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _site_occupancies(payload: dict[str, object]) -> dict[str, float]:
    occupancies: dict[str, float] = {}
    for site in payload["expanded_atom_list"]:  # type: ignore[index]
        label = str(site["label"])
        occupancies.setdefault(label, float(site["occupancy"]))
    return occupancies


def compare_bi2se3_reference(
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    cif_path: str | Path = DEFAULT_CIF_PATH,
    *,
    mode: str = "vesta",
    wavelength_mode: str = "two_theta_balanced",
) -> tuple[dict[str, object], list[object]]:
    rows = parse_vesta_structure_factor_txt(reference_path)
    inferred_wavelength = infer_lambda_from_vesta_rows(rows)
    if wavelength_mode == "two_theta_balanced":
        wavelength = EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
    elif wavelength_mode == "inferred":
        wavelength = inferred_wavelength
    else:
        raise ValueError(f"Unsupported wavelength_mode: {wavelength_mode}")
    options = (
        StructureFactorOptions.vesta_cu_ka1()
        if mode == "vesta"
        else StructureFactorOptions.package_default()
    )
    sim_rows = [compute_raw_f_debug_payload(cif_path, row.hkl, wavelength, options) for row in rows]
    comparison = compare_hkl_tables(rows, sim_rows)
    summary = summarize_f_errors(comparison)
    summary.update(summarize_geometry_errors(comparison))
    site_occupancies = _site_occupancies(sim_rows[0]) if sim_rows else {}
    finite_two_theta_count = sum(row.two_theta is not None for row in rows)
    summary["row_count"] = len(rows)
    summary["finite_two_theta_count"] = finite_two_theta_count
    summary["nan_two_theta_count"] = len(rows) - finite_two_theta_count
    summary["cif_sha256"] = _sha256(cif_path)
    summary["txt_sha256"] = _sha256(reference_path)
    summary["wavelength_angstrom"] = wavelength
    summary["inferred_wavelength_angstrom"] = inferred_wavelength
    summary["wavelength_mode"] = wavelength_mode
    summary["mode"] = mode
    summary["factor_convention"] = VESTA_FACTOR_CONVENTION if mode == "vesta" else "package default"
    summary["site_occupancy"] = site_occupancies
    summary["se1_occupancy"] = site_occupancies.get("Se1")
    summary["options"] = options.to_dict()
    return summary, comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Bi2Se3 VESTA TXT F rows to raw CIF structure factors."
    )
    parser.add_argument("--reference", default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--cif", default=DEFAULT_CIF_PATH)
    parser.add_argument("--mode", choices=["vesta", "package"], default="vesta")
    parser.add_argument(
        "--wavelength-mode",
        choices=["two_theta_balanced", "inferred"],
        default="two_theta_balanced",
    )
    parser.add_argument("--csv-output")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary, comparison = compare_bi2se3_reference(
        args.reference,
        args.cif,
        mode=args.mode,
        wavelength_mode=args.wavelength_mode,
    )
    if args.csv_output:
        write_comparison_csv(args.csv_output, comparison)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"mode: {summary['mode']}")
        print(f"rows: {summary['count']}")
        print(f"wavelength_mode: {summary['wavelength_mode']}")
        print(f"wavelength_angstrom: {summary['wavelength_angstrom']:.12f}")
        print(f"mean_rel_error: {summary['mean_rel_error']:.8g}")
        print(f"median_rel_error: {summary['median_rel_error']:.8g}")
        print(f"max_rel_error: {summary['max_rel_error']:.8g}")
        print(f"max_two_theta_abs_error: {summary['max_two_theta_abs_error']:.8g}")
        print(f"worst_hkl: {summary['worst_hkl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
