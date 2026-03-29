"""Simple CLI to run the diffraction simulation headlessly, launch the GUI, and invoke tools.

Usage examples:

- Choose startup mode interactively:
    python -m ra_sim

- Use defaults from config and write `output.png`:
    python -m ra_sim simulate --out output.png

- Override samples and image size:
    python -m ra_sim simulate --out out.png --samples 2000 --image-size 3000

- Run the hBN ellipse fitting workflow:
    python -m ra_sim hbn-fit --osc /path/to/calibrant.osc --dark /path/to/dark.osc

- Launch the new calibrant fitter GUI:
    python -m ra_sim calibrant --bundle /path/to/hbn_bundle.npz

This CLI intentionally mirrors the defaults used by the GUI by reading
instrument and file paths from `config/` via `ra_sim.config`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
from typing import Dict

import numpy as np
from PIL import Image

from ra_sim import launcher
from ra_sim.config import get_instrument_config, get_path
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.utils.stacking_fault import (
    DEFAULT_PHASE_DELTA_EXPRESSION,
    DEFAULT_PHI_L_DIVISOR,
    ht_Iinf_dict,
    ht_dict_to_qr_dict,
    normalize_phi_l_divisor,
    normalize_phase_delta_expression,
    validate_phase_delta_expression,
)
from ra_sim.hbn import load_tilt_hint, run_hbn_fit
from ra_sim.utils.tools import (
    detector_two_theta_max,
    DEFAULT_PIXEL_SIZE_M,
)
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import (
    DEFAULT_SOLVE_Q_MODE,
    SOLVE_Q_MODE_ADAPTIVE,
    SOLVE_Q_MODE_UNIFORM,
)
from ra_sim.simulation.engine import simulate_qr_rods
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)
from ra_sim.utils.calculations import (
    resolve_index_of_refraction,
    resolve_index_of_refraction_array,
)


@dataclass(frozen=True)
class HeadlessSimulationDefaults:
    """Resolved defaults and typed parameter objects for headless simulation."""

    out_path: str
    image_size: int
    samples: int
    vmax: float
    cif_file: str
    geometry: DetectorGeometry
    mosaic: MosaicParams
    debye_waller: DebyeWallerParams
    occ: tuple[float, ...]
    p_values: tuple[float, ...]
    weights: np.ndarray
    two_theta_max: float
    ht_max_miller_index: int
    ht_phase_delta_expression: str
    ht_phi_l_divisor: float
    ht_finite_stack: bool
    ht_stack_layers: int
    divergence_sigma_rad: float
    bandwidth_sigma: float
    bandwidth_fraction: float
    sample_depth_m: float


@dataclass(frozen=True)
class HeadlessSimulationPlan:
    """Executable headless simulation inputs built from config and CLI overrides."""

    defaults: HeadlessSimulationDefaults
    qr_dict: Dict
    request: SimulationRequest


def _parse_cif_cell_a_c(cif_file: str) -> tuple[float, float]:
    """Return (a, c) from a CIF file using PyCifRW."""
    import CifFile
    import re

    cf = CifFile.ReadCif(cif_file)
    blk = cf[list(cf.keys())[0]]

    def _parse_num(txt: str) -> float:
        if isinstance(txt, (int, float)):
            return float(txt)
        m = re.match(r"[-+0-9\.Ee]+", str(txt).strip())
        if not m:
            raise ValueError(f"Can't parse '{txt}' as a number from CIF")
        return float(m.group(0))

    a_text = blk.get("_cell_length_a")
    c_text = blk.get("_cell_length_c")
    if a_text is None or c_text is None:
        raise ValueError("CIF is missing _cell_length_a/_c fields")

    a = _parse_num(a_text)
    c = _parse_num(c_text)
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


def _coerce_finite_float(value: object) -> float | None:
    """Return a finite float, or ``None`` when coercion fails."""

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_float):
        return None
    return value_float


def _resolve_solve_q_mode(mode_raw: object) -> int:
    """Normalize CLI/config solve-q mode values to engine constants."""

    if isinstance(mode_raw, (int, np.integer, float, np.floating)):
        return (
            SOLVE_Q_MODE_UNIFORM
            if int(round(float(mode_raw))) == 0
            else SOLVE_Q_MODE_ADAPTIVE
        )

    mode_txt = str(mode_raw).strip().lower()
    if mode_txt in {"uniform", "fast", "0"}:
        return SOLVE_Q_MODE_UNIFORM
    if mode_txt in {"adaptive", "robust", "1"}:
        return SOLVE_Q_MODE_ADAPTIVE
    return DEFAULT_SOLVE_Q_MODE


def _apply_headless_tilt_hint(
    *,
    gamma_initial: float,
    Gamma_initial: float,
    distance_m: float,
) -> tuple[float, float, float]:
    """Apply optional hBN tilt-hint defaults to geometry values."""

    tilt_hint = load_tilt_hint()
    if not tilt_hint:
        return gamma_initial, Gamma_initial, distance_m

    hinted_gamma = _coerce_finite_float(tilt_hint.get("gamma_deg"))
    hinted_Gamma = _coerce_finite_float(tilt_hint.get("Gamma_deg"))
    hinted_distance = _coerce_finite_float(tilt_hint.get("distance_m"))

    if hinted_gamma is not None:
        gamma_initial = hinted_gamma
    if hinted_Gamma is not None:
        Gamma_initial = hinted_Gamma
    if hinted_distance is not None:
        distance_m = hinted_distance

    print(
        "Using detector tilt defaults from hBN fit profile: "
        f"sim γ={gamma_initial:.4f} deg, sim Γ={Gamma_initial:.4f} deg"
    )
    if hinted_distance is not None:
        print(
            "Using detector distance default from hBN fit profile: "
            f"Dist={hinted_distance:.4f} m"
        )

    return gamma_initial, Gamma_initial, distance_m


def _cmd_gui(args: argparse.Namespace) -> None:
    """Launch the Tkinter GUI through the packaged launcher."""

    write_excel_flag = None if not args.no_excel else False
    launcher.launch_simulation_gui(write_excel_flag=write_excel_flag)


def _cmd_calibrant(args: argparse.Namespace) -> None:
    """Launch the hBN calibrant fitter GUI through the packaged launcher."""

    launcher.launch_calibrant_gui(bundle=args.bundle)


def _prompt_startup_mode() -> str | None:
    """Prompt for startup mode when launched with no CLI args."""

    if not sys.stdin.isatty():
        return None

    print("Select startup mode:")
    print("  1) Fit calibrant (hBN fitter)")
    print("  2) Run simulation GUI")
    while True:
        try:
            choice = input("Enter choice [2]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("")
            return None

        if choice in {"", "2", "sim", "simulate", "simulation", "s"}:
            return "simulation"
        if choice in {"1", "cal", "calibrant", "fit", "f"}:
            return "calibrant"
        print("Please enter 1 or 2.")


def build_headless_simulation_defaults(
    out_path: str,
    image_size: int | None = None,
    samples: int | None = None,
    vmax: float | None = None,
) -> HeadlessSimulationDefaults:
    """Resolve config-driven defaults and typed parameter objects for the CLI."""

    inst = get_instrument_config().get("instrument", {})
    det_cfg = inst.get("detector", {})
    geom_cfg = inst.get("geometry_defaults", {})
    beam_cfg = inst.get("beam", {})
    sample_cfg = inst.get("sample_orientation", {})
    debye_cfg = inst.get("debye_waller", {})
    ht_cfg = inst.get("hendricks_teller", {})

    resolved_image_size = (
        int(det_cfg.get("image_size", 3000))
        if image_size is None
        else int(image_size)
    )
    resolved_samples = (
        int(det_cfg.get("monte_carlo_samples", 1000))
        if samples is None
        else int(samples)
    )
    resolved_vmax = (
        float(det_cfg.get("vmax", 1000))
        if vmax is None
        else float(vmax)
    )

    cif_file = get_path("cif_file")
    poni = parse_poni_file(get_path("geometry_poni"))

    distance_m = float(poni.get("Dist", geom_cfg.get("distance_m", 0.075)))
    Gamma_initial = float(poni.get("Rot1", geom_cfg.get("rot1", 0.0)))
    gamma_initial = float(poni.get("Rot2", geom_cfg.get("rot2", 0.0)))
    poni1 = float(poni.get("Poni1", geom_cfg.get("poni1_m", 0.0)))
    poni2 = float(poni.get("Poni2", geom_cfg.get("poni2_m", 0.0)))
    wave_m = float(poni.get("Wavelength", geom_cfg.get("wavelength_m", 1e-10)))
    lambda_from_poni = wave_m * 1e10

    lambda_override = beam_cfg.get("wavelength_angstrom")
    lambda_ang = float(
        lambda_override if lambda_override is not None else lambda_from_poni
    )
    pixel_size_m = float(det_cfg.get("pixel_size_m", DEFAULT_PIXEL_SIZE_M))

    gamma_initial, Gamma_initial, distance_m = _apply_headless_tilt_hint(
        gamma_initial=gamma_initial,
        Gamma_initial=Gamma_initial,
        distance_m=distance_m,
    )

    center = np.array(
        [
            (poni2 / pixel_size_m),
            resolved_image_size - (poni1 / pixel_size_m),
        ],
        dtype=np.float64,
    )
    two_theta_max = detector_two_theta_max(
        resolved_image_size,
        center,
        distance_m,
        pixel_size=pixel_size_m,
    )

    occ = tuple(inst.get("occupancies", {}).get("default", [1.0, 1.0, 1.0]))
    av, cv = _parse_cif_cell_a_c(cif_file)

    p_values = tuple(ht_cfg.get("default_p", [0.01, 0.99, 0.5]))
    w_defaults = np.asarray(
        ht_cfg.get("default_w", [50.0, 50.0, 0.0]),
        dtype=np.float64,
    )
    weights = w_defaults / (w_defaults.sum() if w_defaults.sum() else 1.0)

    finite_stack_flag = bool(ht_cfg.get("finite_stack", True))
    stack_layers_count = int(max(1, float(ht_cfg.get("stack_layers", 50))))
    phase_delta_expression = validate_phase_delta_expression(
        normalize_phase_delta_expression(
            ht_cfg.get(
                "phase_delta_expression",
                DEFAULT_PHASE_DELTA_EXPRESSION,
            ),
            fallback=DEFAULT_PHASE_DELTA_EXPRESSION,
        )
    )
    phi_l_divisor = normalize_phi_l_divisor(
        ht_cfg.get("phi_l_divisor", DEFAULT_PHI_L_DIVISOR),
        fallback=DEFAULT_PHI_L_DIVISOR,
    )

    fwhm2sigma = 1 / (2 * math.sqrt(2 * math.log(2)))
    divergence_fwhm = float(beam_cfg.get("divergence_fwhm_deg", 0.05))
    divergence_sigma = math.radians(divergence_fwhm * fwhm2sigma)
    bw_sigma = (
        float(beam_cfg.get("bandwidth_sigma_fraction", 0.05e-3)) * fwhm2sigma
    )
    bandwidth = float(beam_cfg.get("bandwidth_percent", 0.7)) / 100.0

    try:
        solve_q_steps = int(round(float(beam_cfg.get("solve_q_steps", 1000))))
    except (TypeError, ValueError):
        solve_q_steps = 1000
    solve_q_steps = int(np.clip(solve_q_steps, 32, 8192))

    try:
        solve_q_rel_tol = float(beam_cfg.get("solve_q_rel_tol", 5.0e-4))
    except (TypeError, ValueError):
        solve_q_rel_tol = 5.0e-4
    solve_q_rel_tol = float(np.clip(solve_q_rel_tol, 1.0e-6, 5.0e-2))

    theta_initial = float(sample_cfg.get("theta_initial_deg", 6.0))
    cor_angle = float(sample_cfg.get("cor_deg", 0.0))
    chi = float(sample_cfg.get("chi_deg", 0.0))
    psi = float(sample_cfg.get("psi_deg", 0.0))
    psi_z = float(sample_cfg.get("psi_z_deg", 0.0))
    zb = float(sample_cfg.get("zb", 0.0))
    zs = float(sample_cfg.get("zs", 0.0))
    sample_width_m = float(sample_cfg.get("width_m", 0.0))
    sample_length_m = float(sample_cfg.get("length_m", 0.0))
    sample_depth_m = float(sample_cfg.get("depth_m", 0.0))

    geometry = DetectorGeometry(
        image_size=resolved_image_size,
        av=av,
        cv=cv,
        lambda_angstrom=lambda_ang,
        distance_m=distance_m,
        gamma_deg=gamma_initial,
        Gamma_deg=Gamma_initial,
        chi_deg=chi,
        psi_deg=psi,
        psi_z_deg=psi_z,
        zs=zs,
        zb=zb,
        center=np.asarray(center, dtype=np.float64),
        theta_initial_deg=theta_initial,
        cor_angle_deg=cor_angle,
        unit_x=np.array([1.0, 0.0, 0.0]),
        n_detector=np.array([0.0, 1.0, 0.0]),
        pixel_size_m=pixel_size_m,
        sample_width_m=sample_width_m,
        sample_length_m=sample_length_m,
    )
    mosaic = MosaicParams(
        sigma_mosaic_deg=float(beam_cfg.get("sigma_mosaic_fwhm_deg", 0.8))
        * fwhm2sigma,
        gamma_mosaic_deg=float(beam_cfg.get("gamma_mosaic_fwhm_deg", 0.7))
        * fwhm2sigma,
        eta=float(beam_cfg.get("eta", 0.0)),
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=_resolve_solve_q_mode(beam_cfg.get("solve_q_mode", "uniform")),
    )
    debye_waller = DebyeWallerParams(
        x=float(debye_cfg.get("x", 0.0)),
        y=float(debye_cfg.get("y", 0.0)),
    )

    return HeadlessSimulationDefaults(
        out_path=str(out_path),
        image_size=resolved_image_size,
        samples=resolved_samples,
        vmax=resolved_vmax,
        cif_file=str(cif_file),
        geometry=geometry,
        mosaic=mosaic,
        debye_waller=debye_waller,
        occ=occ,
        p_values=p_values,
        weights=weights,
        two_theta_max=float(two_theta_max),
        ht_max_miller_index=int(ht_cfg.get("max_miller_index", 19)),
        ht_phase_delta_expression=phase_delta_expression,
        ht_phi_l_divisor=phi_l_divisor,
        ht_finite_stack=finite_stack_flag,
        ht_stack_layers=stack_layers_count,
        divergence_sigma_rad=divergence_sigma,
        bandwidth_sigma=bw_sigma,
        bandwidth_fraction=bandwidth,
        sample_depth_m=sample_depth_m,
    )


def build_headless_qr_dict(defaults: HeadlessSimulationDefaults) -> Dict:
    """Build the combined HT rod dictionary used by the headless CLI."""

    caches = []
    for p_value in defaults.p_values:
        curves = ht_Iinf_dict(
            cif_path=defaults.cif_file,
            mx=defaults.ht_max_miller_index,
            occ=list(defaults.occ),
            p=float(p_value),
            L_step=0.01,
            two_theta_max=float(defaults.two_theta_max),
            lambda_=defaults.geometry.lambda_angstrom,
            c_lattice=defaults.geometry.cv,
            phase_z_divisor=defaults.ht_phi_l_divisor,
            phase_delta_expression=defaults.ht_phase_delta_expression,
            phi_l_divisor=defaults.ht_phi_l_divisor,
            finite_stack=defaults.ht_finite_stack,
            stack_layers=defaults.ht_stack_layers,
        )
        caches.append(
            {
                "p": float(p_value),
                "qr": ht_dict_to_qr_dict(curves),
            }
        )

    return _combine_qr_dicts(caches, defaults.weights)


def build_headless_beam_samples(
    defaults: HeadlessSimulationDefaults,
) -> BeamSamples:
    """Build Monte Carlo beam samples and wavelength-dependent optical constants."""

    (
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        wavelength_array,
    ) = generate_random_profiles(
        defaults.samples,
        defaults.divergence_sigma_rad,
        defaults.bandwidth_sigma,
        defaults.geometry.lambda_angstrom,
        defaults.bandwidth_fraction,
    )

    n2_sample_array = resolve_index_of_refraction_array(
        np.asarray(wavelength_array, dtype=np.float64) * 1.0e-10,
        cif_path=defaults.cif_file,
    )
    return BeamSamples(
        beam_x_array=np.asarray(beam_x_array, dtype=np.float64),
        beam_y_array=np.asarray(beam_y_array, dtype=np.float64),
        theta_array=np.asarray(theta_array, dtype=np.float64),
        phi_array=np.asarray(phi_array, dtype=np.float64),
        wavelength_array=np.asarray(wavelength_array, dtype=np.float64),
        n2_sample_array=np.asarray(n2_sample_array, dtype=np.complex128),
    )


def build_headless_simulation_request(
    defaults: HeadlessSimulationDefaults,
    beam_samples: BeamSamples | None = None,
) -> SimulationRequest:
    """Build the typed simulation request consumed by the engine."""

    beam = (
        build_headless_beam_samples(defaults)
        if beam_samples is None
        else beam_samples
    )
    n2 = resolve_index_of_refraction(
        defaults.geometry.lambda_angstrom * 1.0e-10,
        cif_path=defaults.cif_file,
    )
    return SimulationRequest(
        miller=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty(0, dtype=np.float64),
        geometry=defaults.geometry,
        beam=beam,
        mosaic=defaults.mosaic,
        debye_waller=defaults.debye_waller,
        n2=n2,
        image_buffer=np.zeros(
            (defaults.image_size, defaults.image_size),
            dtype=np.float64,
        ),
        save_flag=0,
        record_status=False,
        thickness=defaults.sample_depth_m,
        collect_hit_tables=False,
    )


def build_headless_simulation_plan(
    out_path: str,
    image_size: int | None = None,
    samples: int | None = None,
    vmax: float | None = None,
) -> HeadlessSimulationPlan:
    """Build the reusable config/request/render inputs for one CLI simulation."""

    defaults = build_headless_simulation_defaults(
        out_path=out_path,
        image_size=image_size,
        samples=samples,
        vmax=vmax,
    )
    return HeadlessSimulationPlan(
        defaults=defaults,
        qr_dict=build_headless_qr_dict(defaults),
        request=build_headless_simulation_request(defaults),
    )


def run_headless_simulation_plan(plan: HeadlessSimulationPlan) -> np.ndarray:
    """Execute a prepared headless simulation plan and return the image array."""

    return simulate_qr_rods(plan.qr_dict, plan.request).image


def write_headless_simulation_image(
    image: np.ndarray,
    *,
    out_path: str,
    vmax: float,
) -> str:
    """Write a simulated image to disk as a scaled 16-bit PNG."""

    render_vmax = float(vmax)
    if render_vmax <= 0:
        render_vmax = float(np.nanmax(image) or 1.0)
    sim_clip = np.clip(image, 0, render_vmax)
    sim_u16 = np.round((sim_clip / render_vmax) * 65535.0).astype(np.uint16)
    Image.fromarray(sim_u16, mode="I;16").save(out_path)
    return str(out_path)


def run_headless_simulation(
    out_path: str,
    image_size: int | None = None,
    samples: int | None = None,
    vmax: float | None = None,
) -> str:
    """Run the headless CLI simulation via the builder/run/render pipeline."""

    plan = build_headless_simulation_plan(
        out_path=out_path,
        image_size=image_size,
        samples=samples,
        vmax=vmax,
    )
    sim_image = run_headless_simulation_plan(plan)
    return write_headless_simulation_image(
        sim_image,
        out_path=plan.defaults.out_path,
        vmax=plan.defaults.vmax,
    )


def _cmd_simulate(args: argparse.Namespace) -> None:
    out_path = run_headless_simulation(
        out_path=args.out,
        image_size=args.image_size,
        samples=args.samples,
        vmax=args.vmax,
    )
    print(f"Wrote simulated image to {out_path}")


def _cmd_hbn_fit(args: argparse.Namespace) -> None:
    results = run_hbn_fit(
        osc_path=args.osc,
        dark_path=args.dark,
        output_dir=args.output_dir,
        load_bundle=args.load_bundle,
        load_bundle_requested=args.load_bundle is not None,
        highres_refine=args.highres_refine,
        reclick=args.reclick,
        reuse_profile=args.reuse_profile,
        paths_file=args.paths_file,
        prompt_save_bundle=getattr(args, "prompt_save_bundle", False),
        load_clicks=args.load_clicks,
        save_clicks=args.save_clicks,
        clicks_only=args.clicks_only,
        beam_center=(args.beam_center_x, args.beam_center_y)
        if args.beam_center_x is not None and args.beam_center_y is not None
        else None,
    )

    if results.get("aborted"):
        reason = results.get("abort_reason") or "early termination"
        print(f"hBN ellipse fitting did not complete: {reason}")
        return

    print("Completed hBN ellipse fitting. Outputs written to:")
    for key in [
        "background_subtracted",
        "overlay",
        "click_profile",
        "bundle",
    ]:
        value = results.get(key, "n/a")
        print(f"  {key.replace('_', ' ').title()}: {value}")
    if results.get("manual_bundle"):
        print(f"  Manual Bundle: {results['manual_bundle']}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run RA-SIM tools.")
    subparsers = ap.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the RA-SIM Tkinter GUI (same behavior as running main.py directly).",
    )
    gui_parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Do not write the initial intensity Excel file on startup (matches main.py option).",
    )
    gui_parser.set_defaults(func=_cmd_gui)

    calibrant_parser = subparsers.add_parser(
        "calibrant",
        aliases=["calibrant-fit"],
        help="Launch the hBN calibrant fitter GUI from hbn_fitter/fitter.py.",
    )
    calibrant_parser.add_argument(
        "--bundle",
        default=None,
        help="Optional NPZ bundle to load at startup in the calibrant fitter.",
    )
    calibrant_parser.set_defaults(func=_cmd_calibrant)

    sim_parser = subparsers.add_parser(
        "simulate",
        help="Run the diffraction simulation headlessly and save an image.",
    )
    sim_parser.add_argument("--out", required=True, help="Output image path (e.g., output.png)")
    sim_parser.add_argument("--image-size", type=int, default=None, help="Simulation image size (pixels)")
    sim_parser.add_argument("--samples", type=int, default=None, help="Monte Carlo samples")
    sim_parser.add_argument(
        "--vmax", type=float, default=None, help="Max intensity for scaling (default from config)"
    )
    sim_parser.set_defaults(func=_cmd_simulate)

    hbn_parser = subparsers.add_parser(
        "hbn-fit", help="Run the hBN ellipse fitting workflow without the GUI."
    )
    hbn_parser.add_argument("--osc", help="Path to the hBN OSC image")
    hbn_parser.add_argument("--dark", help="Path to the dark frame OSC image")
    hbn_parser.add_argument(
        "--output-dir",
        help=(
            "Directory to write hBN outputs (defaults to ~/Downloads or the bundle directory when using --load-bundle)."
        ),
    )
    hbn_parser.add_argument(
        "--load-bundle",
        nargs="?",
        const="",
        help=(
            "Existing NPZ bundle created by the hBN workflow to reload or refine. "
            "Omit the path to let the CLI pull the bundle location from a paths file "
            "(defaults to config/hbn_paths.yaml)."
        ),
    )
    hbn_parser.add_argument(
        "--highres-refine",
        action="store_true",
        help="When loading a bundle, recompute a full resolution background subtraction and refine ellipses on it.",
    )
    hbn_parser.add_argument(
        "--reclick",
        action="store_true",
        help=(
            "Force a new interactive click session even when loading a bundle (requires --osc/--dark to rebuild the "
            "background before collecting 5 points per ellipse)."
        ),
    )
    hbn_parser.add_argument(
        "--reuse-profile",
        action="store_true",
        help="Reuse an existing click profile JSON in the output directory if present.",
    )
    hbn_parser.add_argument(
        "--prompt-save-bundle",
        action="store_true",
        help=(
            "After a successful fit, open a file-save dialog to choose where to write an hBN NPZ bundle."
        ),
    )
    hbn_parser.add_argument(
        "--paths-file",
        help=(
            "Optional YAML/JSON file containing calibrant, dark, and artifact paths "
            "(keys: calibrant/osc, dark/dark_file, bundle/npz, click_profile/profile, "
            "fit_profile/fit). If omitted, the CLI falls back to "
            "config/hbn_paths.yaml when available."
        ),
    )
    hbn_parser.add_argument(
        "--beam-center-x",
        type=float,
        default=None,
        help=(
            "Beam center x-position in pixels (origin at image top-left). When provided "
            "with --beam-center-y, guides will radiate from this point during clicking."
        ),
    )
    hbn_parser.add_argument(
        "--beam-center-y",
        type=float,
        default=None,
        help=(
            "Beam center y-position in pixels (origin at image top-left). When provided "
            "with --beam-center-x, guides will radiate from this point during clicking."
        ),
    )
    hbn_parser.add_argument(
        "--load-clicks",
        help=(
            "Optional JSON click profile to load instead of interactively collecting points "
            "(keys: image_shape, points)."
        ),
    )
    hbn_parser.add_argument(
        "--save-clicks",
        nargs="?",
        const="",
        help=(
            "Write the clicked points to a JSON profile after selection (defaults to "
            "hbn_click_profile.json in the output directory when omitted)."
        ),
    )
    hbn_parser.add_argument(
        "--clicks-only",
        action="store_true",
        help=(
            "Stop after collecting points (and saving them if requested) without fitting ellipses or "
            "writing the full bundle."
        ),
    )
    hbn_parser.set_defaults(func=_cmd_hbn_fit)

    return ap


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = _build_parser()

    known_commands = {
        "gui",
        "simulate",
        "hbn-fit",
        "calibrant",
        "calibrant-fit",
        "-h",
        "--help",
    }

    if not argv:
        startup_mode = _prompt_startup_mode()
        if startup_mode == "calibrant":
            argv = ["calibrant"]
        elif startup_mode == "simulation":
            argv = ["gui"]

    if not argv:
        ap.print_help()
        return

    if argv[0] not in known_commands:
        argv = ["simulate"] + argv

    args = ap.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        ap.print_help()
        return

    handler(args)


if __name__ == "__main__":  # pragma: no cover
    main()
