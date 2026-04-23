from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import compute_raw_f_debug_payload
from ra_sim.validation.vesta_reference import (
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    infer_lambda_from_vesta_rows,
    parse_vesta_structure_factor_txt,
)


def _options(name: str) -> StructureFactorOptions:
    if name == "vesta":
        return StructureFactorOptions.vesta_cu_ka1()
    if name == "package":
        return StructureFactorOptions.package_default()
    raise ValueError(f"Unsupported mode: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", default="tests/fixtures/Bi2Se3.cif")
    parser.add_argument("--reference", default="tests/fixtures/Bi2Se3_vesta_cu_ka1_dmin_0p7.txt")
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--l", type=int, required=True)
    parser.add_argument("--mode", choices=["vesta", "package"], default="vesta")
    parser.add_argument(
        "--wavelength-mode",
        choices=["two_theta_balanced", "inferred"],
        default="two_theta_balanced",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = parse_vesta_structure_factor_txt(args.reference)
    inferred_wavelength = infer_lambda_from_vesta_rows(rows)
    wavelength = (
        EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM
        if args.wavelength_mode == "two_theta_balanced"
        else inferred_wavelength
    )
    reference = next((row for row in rows if row.hkl == (args.h, args.k, args.l)), None)
    if reference is None:
        parser.error(f"HKL {(args.h, args.k, args.l)} not found in reference.")
    payload = compute_raw_f_debug_payload(
        args.cif,
        reference.hkl,
        wavelength,
        _options(args.mode),
    )
    result = {
        "wavelength_mode": args.wavelength_mode,
        "wavelength_angstrom": wavelength,
        "inferred_wavelength_angstrom": inferred_wavelength,
        "reference": {
            "h": reference.h,
            "k": reference.k,
            "l": reference.l,
            "d": reference.d,
            "f_real": reference.f_real,
            "f_imag": reference.f_imag,
            "f_abs": reference.f_abs,
            "two_theta": reference.two_theta,
        },
        "simulation": payload,
        "residual": {
            "f_real": payload["f_real"] - reference.f_real,
            "f_imag": payload["f_imag"] - reference.f_imag,
            "f_abs": payload["f_abs"] - reference.f_abs,
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
