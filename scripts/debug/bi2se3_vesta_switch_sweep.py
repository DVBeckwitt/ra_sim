from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.sweep import build_switch_sweep
from ra_sim.validation.vesta_reference import (
    EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
    parse_vesta_structure_factor_txt,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", default="tests/fixtures/Bi2Se3.cif")
    parser.add_argument("--reference", default="tests/fixtures/Bi2Se3_vesta_cu_ka1_dmin_0p7.txt")
    args = parser.parse_args()

    rows = parse_vesta_structure_factor_txt(args.reference)
    sweep = build_switch_sweep(
        rows,
        args.cif,
        EXPECTED_BI2SE3_TWOTHETA_WAVELENGTH_ANGSTROM,
        StructureFactorOptions.vesta_cu_ka1(),
    )
    print(json.dumps(sweep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
