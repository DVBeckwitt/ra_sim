"""Generate L-axis reflectivity diagnostic plots and CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.analysis.l_axis_reflectivity_diagnostics import (
    LAxisDiagnosticMaterial,
    run_l_axis_reflectivity_diagnostics,
)
from ra_sim.analysis.parratt import make_bi_chalcogenide_stack


_DEFAULT_LATTICE = {
    "Bi2Se3": (4.143, 28.64),
    "Bi2Te3": (4.386, 30.497),
}


def _default_cif_path(material: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if material == "Bi2Se3":
        return repo_root / "tests" / "fixtures" / "Bi2Se3.cif"
    return repo_root / "tests" / "fixtures" / "Bi2Se3.cif"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Miceli-style L-axis reflectivity diagnostics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "l_axis_reflectivity_diagnostics",
    )
    parser.add_argument("--material", choices=sorted(_DEFAULT_LATTICE), default="Bi2Se3")
    parser.add_argument("--cif-path", type=Path, default=None)
    parser.add_argument("--a-angstrom", type=float, default=None)
    parser.add_argument("--c-angstrom", type=float, default=None)
    parser.add_argument("--thickness-nm", type=float, default=100.0)
    parser.add_argument("--stack-layers", type=int, default=100)
    parser.add_argument("--substrate", choices=["SiO2", "air"], default="SiO2")
    parser.add_argument("--lambda0-angstrom", type=float, default=1.5418)
    parser.add_argument("--bandwidth-fwhm", type=float, default=0.05)
    parser.add_argument("--n-wavelength-samples", type=int, default=241)
    parser.add_argument(
        "--ht-bandwidth-mode",
        choices=["resimulate", "interpolate-only"],
        default="resimulate",
    )
    parser.add_argument("--l-points", type=int, default=1801)
    parser.add_argument("--detector-image", type=str, default=None)
    parser.add_argument("--detector-roi-csv", type=str, default=None)
    parser.add_argument("--allow-missing-rois", action="store_true")
    return parser


def material_from_args(args: argparse.Namespace) -> LAxisDiagnosticMaterial:
    default_a, default_c = _DEFAULT_LATTICE[args.material]
    layers = make_bi_chalcogenide_stack(
        args.material,
        thickness_nm=float(args.thickness_nm),
        substrate=args.substrate,
    )
    return LAxisDiagnosticMaterial(
        name=args.material,
        cif_path=str(args.cif_path or _default_cif_path(args.material)),
        a_angstrom=float(args.a_angstrom if args.a_angstrom is not None else default_a),
        c_angstrom=float(args.c_angstrom if args.c_angstrom is not None else default_c),
        qc_inv_angstrom=float(layers[1].qc_inv_angstrom),
        layers=layers,
        stack_layers=max(1, int(args.stack_layers)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    L_values = np.linspace(0.0, 9.0, max(2, int(args.l_points)))
    run_l_axis_reflectivity_diagnostics(
        output_dir=args.output_dir,
        material=material_from_args(args),
        L_values=L_values,
        lambda0_angstrom=float(args.lambda0_angstrom),
        bandwidth_fwhm=float(args.bandwidth_fwhm),
        n_wavelength_samples=int(args.n_wavelength_samples),
        ht_bandwidth_mode=str(args.ht_bandwidth_mode),
        detector_image_path=args.detector_image,
        detector_roi_csv=args.detector_roi_csv,
        allow_missing_rois=bool(args.allow_missing_rois),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
