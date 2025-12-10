"""Simple CLI to run the diffraction simulation headlessly and save an image.

Usage examples:

- Use defaults from config and write `output.png`:
    python -m ra_sim --out output.png

- Override samples and image size:
    python -m ra_sim --out out.png --samples 2000 --image-size 3000

This CLI intentionally mirrors the defaults used by the GUI by reading
instrument and file paths from `config/` via `ra_sim.path_config`.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict

import numpy as np
from PIL import Image

from ra_sim.path_config import get_instrument_config, get_path
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.utils.stacking_fault import (
    ht_Iinf_dict,
    ht_dict_to_qr_dict,
)
from ra_sim.utils.tools import (
    detector_two_theta_max,
    DEFAULT_PIXEL_SIZE_M,
)
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import process_qr_rods_parallel
from ra_sim.utils.calculations import IndexofRefraction


def _parse_cif_cell_a_c(cif_file: str) -> tuple[float, float]:
    """Return (a, c) from a CIF file using PyCifRW, following main.py pattern.

    Multiplies `c` by 3 to align with existing GUI assumptions.
    """
    import CifFile
    import re

    cf = CifFile.ReadCif(cif_file)
    blk = cf[list(cf.keys())[0]]

    def _parse_num(txt: str) -> float:
        m = re.match(r"[-+0-9\.Ee]+", txt)
        if not m:
            raise ValueError(f"Can't parse '{txt}' as a number from CIF")
        return float(m.group(0))

    a_text = blk.get("_cell_length_a")
    c_text = blk.get("_cell_length_c")
    if a_text is None or c_text is None:
        raise ValueError("CIF is missing _cell_length_a/_c fields")

    a = _parse_num(a_text)
    c = _parse_num(c_text) * 3.0
    return a, c


def _combine_qr_dicts(caches: list[Dict], weights: np.ndarray) -> Dict:
    """Combine multiple qr_dicts with linear weights (shape consistent with GUI).

    Each cache item must be a dict with key "qr" as produced by ht_dict_to_qr_dict.
    """
    out: Dict = {}
    for cache, w in zip(caches, weights):
        qr = cache["qr"]
        for m, data in qr.items():
            if m not in out:
                out[m] = {
                    "L": data["L"].copy(),
                    "I": w * data["I"].copy(),
                    "hk": data["hk"],
                    "deg": data.get("deg", 1),
                }
            else:
                entry = out[m]
                if entry["L"].shape != data["L"].shape or not np.allclose(entry["L"], data["L"]):
                    union_L = np.union1d(entry["L"], data["L"])
                    entry_I = np.interp(union_L, entry["L"], entry["I"], left=0.0, right=0.0)
                    add_I = w * np.interp(union_L, data["L"], data["I"], left=0.0, right=0.0)
                    entry["L"] = union_L
                    entry["I"] = entry_I + add_I
                else:
                    entry["I"] += w * data["I"]
                entry["deg"] += int(data.get("deg", 1))
    return out


def run_headless_simulation(
    out_path: str,
    image_size: int | None = None,
    samples: int | None = None,
    vmax: float | None = None,
) -> str:
    """Run a headless simulation using defaults from config and save an image.

    Returns the absolute path to the written image.
    """

    # Load instrument + file paths
    inst = get_instrument_config().get("instrument", {})
    det_cfg = inst.get("detector", {})
    geom_cfg = inst.get("geometry_defaults", {})
    beam_cfg = inst.get("beam", {})
    sample_cfg = inst.get("sample_orientation", {})
    debye_cfg = inst.get("debye_waller", {})
    ht_cfg = inst.get("hendricks_teller", {})

    if image_size is None:
        image_size = int(det_cfg.get("image_size", 3000))
    if samples is None:
        samples = int(det_cfg.get("monte_carlo_samples", 1000))
    if vmax is None:
        vmax = float(det_cfg.get("vmax", 1000))

    cif_file = get_path("cif_file")
    poni_file = get_path("geometry_poni")

    # Geometry/beam parameters (mirror main.py defaults)
    poni = parse_poni_file(poni_file)
    D = float(poni.get("Dist", geom_cfg.get("distance_m", 0.075)))
    Gamma_initial = float(poni.get("Rot1", geom_cfg.get("rot1", 0.0)))
    gamma_initial = float(poni.get("Rot2", geom_cfg.get("rot2", 0.0)))
    poni1 = float(poni.get("Poni1", geom_cfg.get("poni1_m", 0.0)))
    poni2 = float(poni.get("Poni2", geom_cfg.get("poni2_m", 0.0)))
    wave_m = float(poni.get("Wavelength", geom_cfg.get("wavelength_m", 1e-10)))
    lambda_from_poni = wave_m * 1e10

    lambda_override = beam_cfg.get("wavelength_angstrom")
    lambda_ang = float(lambda_override if lambda_override is not None else lambda_from_poni)

    pixel_size_m = float(det_cfg.get("pixel_size_m", DEFAULT_PIXEL_SIZE_M))

    # Beam center: follow GUI default mapping from PONI to pixels
    center = np.array([
        (poni2 / pixel_size_m),
        image_size - (poni1 / pixel_size_m),
    ], dtype=np.float64)

    two_theta_max = detector_two_theta_max(
        image_size,
        center,
        D,
        pixel_size=pixel_size_m,
    )

    # HT inputs
    mx = int(ht_cfg.get("max_miller_index", 19))
    p_defaults = list(ht_cfg.get("default_p", [0.01, 0.99, 0.5]))
    w_defaults = np.array(ht_cfg.get("default_w", [50.0, 50.0, 0.0]), dtype=float)
    w_norm = w_defaults / (w_defaults.sum() if w_defaults.sum() else 1.0)

    # Occupancies
    occ = inst.get("occupancies", {}).get("default", [1.0, 1.0, 1.0])

    # Lattice constants (mirrors GUI default of tripling c)
    av, cv = _parse_cif_cell_a_c(cif_file)

    # Build HT curves and rods for the three p values
    finite_stack_flag = bool(ht_cfg.get("finite_stack", True))
    stack_layers_count = int(
        max(1, float(ht_cfg.get("stack_layers", 50)))
    )

    def build_ht_cache(p_val: float):
        curves = ht_Iinf_dict(
            cif_path=cif_file,
            mx=mx,
            occ=occ,
            p=float(p_val),
            L_step=0.01,
            two_theta_max=float(two_theta_max),
            lambda_=lambda_ang,
            c_lattice=cv,
            finite_stack=finite_stack_flag,
            stack_layers=stack_layers_count,
        )
        qr = ht_dict_to_qr_dict(curves)
        return {"p": float(p_val), "qr": qr}

    caches = [build_ht_cache(p) for p in p_defaults]
    qr_combined = _combine_qr_dicts(caches, w_norm)

    # Mosaic/beam sampling
    fwhm2sigma = 1 / (2 * math.sqrt(2 * math.log(2)))
    divergence_fwhm = float(beam_cfg.get("divergence_fwhm_deg", 0.05))
    divergence_sigma = math.radians(divergence_fwhm * fwhm2sigma)
    bw_sigma = float(beam_cfg.get("bandwidth_sigma_fraction", 0.05e-3)) * fwhm2sigma
    bandwidth = float(beam_cfg.get("bandwidth_percent", 0.7)) / 100.0

    (beam_x_array,
     beam_y_array,
     theta_array,
     phi_array,
     wavelength_array) = generate_random_profiles(samples, divergence_sigma, bw_sigma, lambda_ang, bandwidth)

    # Film/optical parameters
    theta_initial = float(sample_cfg.get("theta_initial_deg", 6.0))
    chi = float(sample_cfg.get("chi_deg", 0.0))
    psi = float(sample_cfg.get("psi_deg", 0.0))
    zb = float(sample_cfg.get("zb", 0.0))
    zs = float(sample_cfg.get("zs", 0.0))
    debye_x = float(inst.get("debye_waller", {}).get("x", 0.0))
    debye_y = float(inst.get("debye_waller", {}).get("y", 0.0))

    # Index of refraction for active material
    n2 = IndexofRefraction()

    # Fixed axes like GUI
    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])

    # Run simulation on rods
    image = np.zeros((image_size, image_size), dtype=np.float64)
    # Mosaic pseudo-Voigt parameters (sigma, gamma are passed as sigma in degrees)
    sigma_mosaic_deg = float(beam_cfg.get("sigma_mosaic_fwhm_deg", 0.8)) * fwhm2sigma
    gamma_mosaic_deg = float(beam_cfg.get("gamma_mosaic_fwhm_deg", 0.7)) * fwhm2sigma

    (sim_image,
     _hit_tables,
     _q_data,
     _q_count,
     _all_status,
     _miss_tables,
     _deg) = process_qr_rods_parallel(
        qr_combined,
        image_size,
        av,
        cv,
        lambda_ang,
        image,
        D,
        gamma_initial,
        Gamma_initial,
        chi,
        psi,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_mosaic_deg,
        gamma_mosaic_deg,
        float(beam_cfg.get("eta", 0.0)),
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial,
        unit_x,
        n_detector,
        save_flag=0,
        record_status=False,
        thickness=0.0,
    )

    # Save as 16-bit PNG scaled to `vmax` for reasonable visualization
    vmax = float(vmax)
    if vmax <= 0:
        vmax = float(np.nanmax(sim_image) or 1.0)
    sim_clip = np.clip(sim_image, 0, vmax)
    sim_u16 = np.round((sim_clip / vmax) * 65535.0).astype(np.uint16)
    img = Image.fromarray(sim_u16, mode="I;16")
    img.save(out_path)
    return str(out_path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run RA-SIM headless and save an image.")
    ap.add_argument("--out", required=True, help="Output image path (e.g., output.png)")
    ap.add_argument("--image-size", type=int, default=None, help="Simulation image size (pixels)")
    ap.add_argument("--samples", type=int, default=None, help="Monte Carlo samples")
    ap.add_argument("--vmax", type=float, default=None, help="Max intensity for scaling (default from config)")
    args = ap.parse_args(argv)

    out_path = run_headless_simulation(
        out_path=args.out,
        image_size=args.image_size,
        samples=args.samples,
        vmax=args.vmax,
    )
    print(f"Wrote simulated image to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
