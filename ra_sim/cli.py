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
import copy
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import numpy as np
from PIL import Image

from ra_sim import launcher
from ra_sim.config import get_instrument_config, get_path
from ra_sim.fitting.optimization import (
    build_geometry_fit_central_mosaic_params,
    fit_geometry_parameters,
    simulate_and_compare_hkl,
)
from ra_sim.gui import background as gui_background
from ra_sim.gui import background_theta as gui_background_theta
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import geometry_overlay as gui_geometry_overlay
from ra_sim.gui import geometry_q_group_manager as gui_geometry_q_group_manager
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.gui import structure_model as gui_structure_model
from ra_sim.gui.state import AtomSiteOverrideState, SimulationRuntimeState
from ra_sim.hbn import load_tilt_hint, run_hbn_fit
from ra_sim.io.data_loading import load_gui_state_file, save_gui_state_file
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.io.osc_reader import read_osc
from ra_sim.utils.stacking_fault import (
    DEFAULT_PHASE_DELTA_EXPRESSION,
    DEFAULT_PHI_L_DIVISOR,
    ht_Iinf_dict,
    ht_dict_to_qr_dict,
    normalize_phi_l_divisor,
    normalize_phase_delta_expression,
    validate_phase_delta_expression,
)
from ra_sim.utils.diffraction_tools import (
    detector_two_theta_max,
    DEFAULT_PIXEL_SIZE_M,
)
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import (
    DEFAULT_SOLVE_Q_MODE,
    DEFAULT_SOLVE_Q_REL_TOL,
    DEFAULT_SOLVE_Q_STEPS,
    OPTICS_MODE_EXACT,
    OPTICS_MODE_FAST,
    SOLVE_Q_MODE_ADAPTIVE,
    SOLVE_Q_MODE_UNIFORM,
    hit_tables_to_max_positions,
    process_peaks_parallel,
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
from ra_sim.utils.tools import (
    build_intensity_dataframes,
    inject_fractional_reflections,
    miller_generator,
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


HEADLESS_GEOMETRY_BACKGROUND_DISPLAY_ROTATE_K = -1
HEADLESS_GEOMETRY_SIM_DISPLAY_ROTATE_K = 0
HEADLESS_GEOMETRY_CUSTOM_SAMPLING_OPTION = "Custom"
HEADLESS_GEOMETRY_SAMPLE_COUNTS: Dict[str, int] = {
    "Low": 25,
    "Medium": 250,
    "High": 500,
}


class _HeadlessVar:
    """Minimal Tk-variable stand-in for headless CLI geometry fitting."""

    def __init__(self, value: object) -> None:
        self._value = value

    def get(self) -> object:
        return self._value

    def set(self, value: object) -> None:
        self._value = value


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


def _saved_state_section(
    state: Mapping[str, object] | None,
    key: str,
) -> dict[str, object]:
    """Return one mapping section from a saved GUI-state snapshot."""

    if not isinstance(state, Mapping):
        return {}
    section = state.get(key)
    return dict(section) if isinstance(section, Mapping) else {}


def _saved_state_var_value(
    saved_variables: Mapping[str, object],
    name: str,
    fallback: object,
) -> object:
    """Return one saved GUI variable value or a fallback."""

    value = saved_variables.get(name, fallback)
    return fallback if value is None else value


def _saved_state_float(
    saved_variables: Mapping[str, object],
    name: str,
    fallback: object,
) -> float:
    """Return one saved GUI variable as a finite float."""

    fallback_value = float(fallback)
    value = _coerce_finite_float(saved_variables.get(name, fallback))
    return fallback_value if value is None else float(value)


def _saved_state_int(
    saved_variables: Mapping[str, object],
    name: str,
    fallback: object,
) -> int:
    """Return one saved GUI variable as an integer."""

    try:
        fallback_value = int(round(float(fallback)))
    except (TypeError, ValueError):
        fallback_value = 0
    try:
        return int(round(float(_saved_state_var_value(saved_variables, name, fallback))))
    except (TypeError, ValueError):
        return int(fallback_value)


def _saved_state_bool(
    saved_variables: Mapping[str, object],
    name: str,
    fallback: bool,
) -> bool:
    """Return one saved GUI variable as a boolean."""

    value = _saved_state_var_value(saved_variables, name, fallback)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return bool(int(round(numeric))) if np.isfinite(numeric) else bool(fallback)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(fallback)


def _saved_state_text(
    saved_variables: Mapping[str, object],
    name: str,
    fallback: object,
) -> str:
    """Return one saved GUI variable as text."""

    value = _saved_state_var_value(saved_variables, name, fallback)
    return str(value) if value is not None else str(fallback)


def _normalize_headless_optics_mode_flag(value: object) -> int:
    """Normalize saved optics-mode values to simulation engine flags."""

    if isinstance(value, (int, np.integer)):
        return OPTICS_MODE_EXACT if int(value) == OPTICS_MODE_EXACT else OPTICS_MODE_FAST
    if isinstance(value, (float, np.floating)):
        return (
            OPTICS_MODE_EXACT
            if int(round(float(value))) == OPTICS_MODE_EXACT
            else OPTICS_MODE_FAST
        )

    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = " ".join(text.split())
    if text in {
        "1",
        "true",
        "yes",
        "on",
        "exact",
        "precise",
        "slow",
        "complex_k_dwba_slab",
        "complex-k dwba slab optics",
        "phase-matched complex-k multilayer dwba",
    }:
        return OPTICS_MODE_EXACT
    return OPTICS_MODE_FAST


def _load_cif_snapshot(path: str) -> tuple[object, object]:
    """Load one CIF file and return ``(cf, first_block)``."""

    import CifFile

    cf = CifFile.ReadCif(str(path))
    keys = list(cf.keys())
    if not keys:
        raise ValueError(f"No CIF data blocks found in {path}")
    return cf, cf[keys[0]]


def _headless_geometry_fit_sample_count(
    saved_variables: Mapping[str, object],
) -> int:
    """Resolve the Monte Carlo sample count used by headless geometry fitting."""

    return gui_controllers.resolve_sampling_count(
        _saved_state_text(saved_variables, "resolution_var", "Low"),
        custom_option=HEADLESS_GEOMETRY_CUSTOM_SAMPLING_OPTION,
        custom_value=_saved_state_var_value(saved_variables, "custom_samples_var", 0),
        preset_counts=HEADLESS_GEOMETRY_SAMPLE_COUNTS,
        fallback_resolution="Low",
        fallback_count=HEADLESS_GEOMETRY_SAMPLE_COUNTS["Low"],
    )


def _build_headless_geometry_mosaic_params(
    *,
    saved_variables: Mapping[str, object],
    beam_config: Mapping[str, object],
    lambda_angstrom: float,
    active_cif_path: str,
) -> tuple[dict[str, object], int]:
    """Build the deterministic runtime mosaic payload used by geometry fitting."""

    fwhm2sigma = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sample_count = _headless_geometry_fit_sample_count(saved_variables)
    divergence_sigma = math.radians(
        float(beam_config.get("divergence_fwhm_deg", 0.05)) * fwhm2sigma
    )
    bandwidth_sigma = (
        float(beam_config.get("bandwidth_sigma_fraction", 5.0e-5)) * fwhm2sigma
    )
    bandwidth_percent = float(
        np.clip(
            _saved_state_float(
                saved_variables,
                "bandwidth_percent_var",
                beam_config.get("bandwidth_percent", 0.7),
            ),
            0.0,
            10.0,
        )
    )
    solve_q_steps = int(
        np.clip(
            _saved_state_int(
                saved_variables,
                "solve_q_steps_var",
                beam_config.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
            ),
            32,
            8192,
        )
    )
    solve_q_rel_tol = float(
        np.clip(
            _saved_state_float(
                saved_variables,
                "solve_q_rel_tol_var",
                beam_config.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL),
            ),
            1.0e-6,
            5.0e-2,
        )
    )
    solve_q_mode = _resolve_solve_q_mode(
        _saved_state_var_value(
            saved_variables,
            "solve_q_mode_var",
            beam_config.get("solve_q_mode", DEFAULT_SOLVE_Q_MODE),
        )
    )

    (
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        wavelength_array,
    ) = generate_random_profiles(
        int(sample_count),
        float(divergence_sigma),
        float(bandwidth_sigma),
        float(lambda_angstrom),
        float(bandwidth_percent) / 100.0,
    )
    n2_sample_array = resolve_index_of_refraction_array(
        np.asarray(wavelength_array, dtype=np.float64) * 1.0e-10,
        cif_path=str(active_cif_path),
    )

    return (
        {
            "beam_x_array": np.asarray(beam_x_array, dtype=np.float64),
            "beam_y_array": np.asarray(beam_y_array, dtype=np.float64),
            "theta_array": np.asarray(theta_array, dtype=np.float64),
            "phi_array": np.asarray(phi_array, dtype=np.float64),
            "wavelength_array": np.asarray(wavelength_array, dtype=np.float64),
            "n2_sample_array": np.asarray(n2_sample_array, dtype=np.complex128),
            "sigma_mosaic_deg": _saved_state_float(saved_variables, "sigma_mosaic_var", 0.0),
            "gamma_mosaic_deg": _saved_state_float(saved_variables, "gamma_mosaic_var", 0.0),
            "eta": _saved_state_float(saved_variables, "eta_var", beam_config.get("eta", 0.0)),
            "solve_q_steps": int(solve_q_steps),
            "solve_q_rel_tol": float(solve_q_rel_tol),
            "solve_q_mode": int(solve_q_mode),
        },
        int(sample_count),
    )


def run_headless_geometry_fit(
    payload: Mapping[str, object],
    *,
    source_path: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run geometry fitting directly from one saved GUI-state payload."""

    state = payload.get("state") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("GUI state payload is missing a valid 'state' object.")
    updated_state = copy.deepcopy(dict(state))
    saved_variables = _saved_state_section(updated_state, "variables")
    saved_lists = _saved_state_section(updated_state, "dynamic_lists")
    files_state = _saved_state_section(updated_state, "files")
    flags_state = _saved_state_section(updated_state, "flags")
    geometry_state = _saved_state_section(updated_state, "geometry")

    raw_background_files = files_state.get("background_files", [])
    osc_files = [
        str(Path(str(path)).expanduser())
        for path in (raw_background_files if isinstance(raw_background_files, list) else [])
        if path is not None
    ]
    if not osc_files:
        raise ValueError("Saved GUI state does not contain any background files.")
    try:
        current_background_index = int(files_state.get("current_background_index", 0))
    except Exception:
        current_background_index = 0
    current_background_index = max(0, min(current_background_index, len(osc_files) - 1))

    simulation_defaults = build_headless_simulation_defaults(out_path=str(source_path))
    inst = get_instrument_config().get("instrument", {})
    detector_cfg = inst.get("detector", {})
    beam_cfg = inst.get("beam", {})
    sample_cfg = inst.get("sample_orientation", {})
    debye_cfg = inst.get("debye_waller", {})
    ht_cfg = inst.get("hendricks_teller", {})
    fit_cfg = inst.get("fit", {})
    fit_geometry_cfg = fit_cfg.get("geometry", {}) if isinstance(fit_cfg, Mapping) else {}
    fit_geometry_cfg = dict(fit_geometry_cfg) if isinstance(fit_geometry_cfg, Mapping) else {}

    image_size = int(detector_cfg.get("image_size", simulation_defaults.image_size))
    pixel_size_m = float(detector_cfg.get("pixel_size_m", simulation_defaults.geometry.pixel_size_m))
    lambda_angstrom = float(simulation_defaults.geometry.lambda_angstrom)
    theta_initial_default = float(sample_cfg.get("theta_initial_deg", simulation_defaults.geometry.theta_initial_deg))
    cor_angle_default = float(sample_cfg.get("cor_deg", simulation_defaults.geometry.cor_angle_deg))
    chi_default = float(sample_cfg.get("chi_deg", simulation_defaults.geometry.chi_deg))
    psi_default = float(sample_cfg.get("psi_deg", 0.0))
    psi_z_default = float(sample_cfg.get("psi_z_deg", simulation_defaults.geometry.psi_z_deg))
    zs_default = float(sample_cfg.get("zs", simulation_defaults.geometry.zs))
    zb_default = float(sample_cfg.get("zb", simulation_defaults.geometry.zb))
    sample_width_default = float(sample_cfg.get("width_m", simulation_defaults.geometry.sample_width_m))
    sample_length_default = float(sample_cfg.get("length_m", simulation_defaults.geometry.sample_length_m))
    sample_depth_default = float(sample_cfg.get("depth_m", simulation_defaults.sample_depth_m))
    gamma_default = float(simulation_defaults.geometry.Gamma_deg)
    Gamma_default = float(simulation_defaults.geometry.gamma_deg)
    distance_default = float(simulation_defaults.geometry.distance_m)
    center_x_default = float(simulation_defaults.geometry.center[0])
    center_y_default = float(simulation_defaults.geometry.center[1])
    debye_x_default = float(debye_cfg.get("x", simulation_defaults.debye_waller.x))
    debye_y_default = float(debye_cfg.get("y", simulation_defaults.debye_waller.y))

    primary_cif_path = str(Path(str(files_state.get("primary_cif_path") or get_path("cif_file"))).expanduser())
    if not Path(primary_cif_path).is_file():
        raise FileNotFoundError(f"Primary CIF file not found: {primary_cif_path}")
    secondary_cif_path = None
    if files_state.get("secondary_cif_path"):
        secondary_candidate = Path(str(files_state.get("secondary_cif_path"))).expanduser()
        if secondary_candidate.is_file():
            secondary_cif_path = str(secondary_candidate)

    primary_cf, primary_blk = _load_cif_snapshot(primary_cif_path)
    primary_a = float(gui_structure_model.parse_cif_num(primary_blk.get("_cell_length_a")))
    primary_c = float(gui_structure_model.parse_cif_num(primary_blk.get("_cell_length_c")))
    secondary_a = secondary_c = None
    if secondary_cif_path:
        _secondary_cf, secondary_blk = _load_cif_snapshot(secondary_cif_path)
        secondary_a = float(gui_structure_model.parse_cif_num(secondary_blk.get("_cell_length_a")))
        secondary_c = float(gui_structure_model.parse_cif_num(secondary_blk.get("_cell_length_c")))

    occupancy_labels, occupancy_expanded_map = gui_structure_model.extract_occupancy_site_metadata(primary_blk, primary_cif_path)
    atom_site_fractional_metadata = gui_structure_model.extract_atom_site_fractional_metadata(primary_blk)
    fallback_occ = list(inst.get("occupancies", {}).get("default", [1.0, 1.0, 1.0]))
    raw_occ_values = saved_lists.get("occupancy_values", [])
    if not isinstance(raw_occ_values, list):
        raw_occ_values = fallback_occ
    occ_values = gui_controllers.clamp_site_occupancy_values(raw_occ_values, fallback_values=fallback_occ)
    if occupancy_labels and len(occ_values) < len(occupancy_labels):
        occ_values.extend([occ_values[-1] if occ_values else 1.0] * (len(occupancy_labels) - len(occ_values)))

    raw_atom_site_values = saved_lists.get("atom_site_fractional_values", [])
    atom_site_values: list[tuple[float, float, float]] = []
    if isinstance(raw_atom_site_values, list):
        for idx, row in enumerate(raw_atom_site_values):
            if not isinstance(row, Mapping):
                continue
            fallback_row = atom_site_fractional_metadata[idx] if idx < len(atom_site_fractional_metadata) else {"x": 0.0, "y": 0.0, "z": 0.0}
            atom_site_values.append(
                (
                    _coerce_finite_float(row.get("x")) if _coerce_finite_float(row.get("x")) is not None else float(fallback_row.get("x", 0.0)),
                    _coerce_finite_float(row.get("y")) if _coerce_finite_float(row.get("y")) is not None else float(fallback_row.get("y", 0.0)),
                    _coerce_finite_float(row.get("z")) if _coerce_finite_float(row.get("z")) is not None else float(fallback_row.get("z", 0.0)),
                )
            )
    while len(atom_site_values) < len(atom_site_fractional_metadata):
        row = atom_site_fractional_metadata[len(atom_site_values)]
        atom_site_values.append((float(row["x"]), float(row["y"]), float(row["z"])))

    p_defaults = list(ht_cfg.get("default_p", [0.01, 0.99, 0.5]))
    while len(p_defaults) < 3:
        p_defaults.append(p_defaults[-1] if p_defaults else 0.5)
    w_defaults = list(ht_cfg.get("default_w", [50.0, 50.0, 0.0]))
    while len(w_defaults) < 3:
        w_defaults.append(0.0)

    theta_initial_var = _HeadlessVar(_saved_state_float(saved_variables, "theta_initial_var", theta_initial_default))
    cor_angle_var = _HeadlessVar(_saved_state_float(saved_variables, "cor_angle_var", cor_angle_default))
    chi_var = _HeadlessVar(_saved_state_float(saved_variables, "chi_var", chi_default))
    psi_z_var = _HeadlessVar(_saved_state_float(saved_variables, "psi_z_var", psi_z_default))
    zs_var = _HeadlessVar(_saved_state_float(saved_variables, "zs_var", zs_default))
    zb_var = _HeadlessVar(_saved_state_float(saved_variables, "zb_var", zb_default))
    gamma_var = _HeadlessVar(_saved_state_float(saved_variables, "gamma_var", gamma_default))
    Gamma_var = _HeadlessVar(_saved_state_float(saved_variables, "Gamma_var", Gamma_default))
    corto_detector_var = _HeadlessVar(_saved_state_float(saved_variables, "corto_detector_var", distance_default))
    a_var = _HeadlessVar(_saved_state_float(saved_variables, "a_var", primary_a))
    c_var = _HeadlessVar(_saved_state_float(saved_variables, "c_var", primary_c))
    center_x_var = _HeadlessVar(_saved_state_float(saved_variables, "center_x_var", center_x_default))
    center_y_var = _HeadlessVar(_saved_state_float(saved_variables, "center_y_var", center_y_default))
    sample_width_var = _HeadlessVar(_saved_state_float(saved_variables, "sample_width_var", sample_width_default))
    sample_length_var = _HeadlessVar(_saved_state_float(saved_variables, "sample_length_var", sample_length_default))
    sample_depth_var = _HeadlessVar(_saved_state_float(saved_variables, "sample_depth_var", sample_depth_default))
    debye_x_var = _HeadlessVar(_saved_state_float(saved_variables, "debye_x_var", debye_x_default))
    debye_y_var = _HeadlessVar(_saved_state_float(saved_variables, "debye_y_var", debye_y_default))
    geometry_theta_offset_var = _HeadlessVar(_saved_state_text(saved_variables, "geometry_theta_offset_var", "0.0"))
    background_theta_list_var = _HeadlessVar(
        _saved_state_text(
            saved_variables,
            "background_theta_list_var",
            gui_background_theta.format_background_theta_values([float(theta_initial_var.get())] * len(osc_files)),
        )
    )
    geometry_fit_background_selection_var = _HeadlessVar(
        _saved_state_text(
            saved_variables,
            "geometry_fit_background_selection_var",
            gui_background_theta.default_geometry_fit_background_selection(osc_files=osc_files),
        )
    )

    fit_zb_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_zb_var", True))
    fit_zs_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_zs_var", True))
    fit_theta_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_theta_var", True))
    fit_psi_z_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_psi_z_var", False))
    fit_chi_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_chi_var", True))
    fit_cor_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_cor_var", False))
    fit_gamma_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_gamma_var", True))
    fit_Gamma_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_Gamma_var", True))
    fit_dist_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_dist_var", True))
    fit_a_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_a_var", False))
    fit_c_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_c_var", False))
    fit_center_x_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_center_x_var", True))
    fit_center_y_var = _HeadlessVar(_saved_state_bool(saved_variables, "fit_center_y_var", False))

    structure_defaults = {
        "a": float(a_var.get()),
        "c": float(c_var.get()),
        "p0": _saved_state_float(saved_variables, "p0_var", p_defaults[0]),
        "p1": _saved_state_float(saved_variables, "p1_var", p_defaults[1]),
        "p2": _saved_state_float(saved_variables, "p2_var", p_defaults[2]),
        "w0": _saved_state_float(saved_variables, "w0_var", w_defaults[0]),
        "w1": _saved_state_float(saved_variables, "w1_var", w_defaults[1]),
        "w2": _saved_state_float(saved_variables, "w2_var", w_defaults[2]),
        "iodine_z": _saved_state_float(saved_variables, "iodine_z_var", 0.0),
        "phi_l_divisor": _saved_state_float(saved_variables, "phi_l_divisor_var", ht_cfg.get("phi_l_divisor", DEFAULT_PHI_L_DIVISOR)),
        "finite_stack": _saved_state_bool(saved_variables, "finite_stack_var", bool(ht_cfg.get("finite_stack", True))),
        "stack_layers": max(1, _saved_state_int(saved_variables, "stack_layers_var", ht_cfg.get("stack_layers", 50))),
        "phase_delta_expression": str(_saved_state_var_value(saved_variables, "phase_delta_expr_var", ht_cfg.get("phase_delta_expression", DEFAULT_PHASE_DELTA_EXPRESSION))),
    }

    center = np.array([float(center_x_var.get()), float(center_y_var.get())], dtype=np.float64)
    two_theta_max = detector_two_theta_max(int(image_size), center, float(corto_detector_var.get()), pixel_size=pixel_size_m)
    structure_model_state = gui_structure_model.build_initial_structure_model_state(
        cif_file=primary_cif_path,
        cf=primary_cf,
        blk=primary_blk,
        cif_file2=secondary_cif_path,
        occupancy_site_labels=occupancy_labels,
        occupancy_site_expanded_map=occupancy_expanded_map,
        occ=occ_values,
        atom_site_fractional_metadata=atom_site_fractional_metadata,
        av=primary_a,
        bv=primary_a,
        cv=primary_c,
        av2=secondary_a,
        cv2=secondary_c,
        defaults=structure_defaults,
        mx=int(ht_cfg.get("max_miller_index", 19)),
        lambda_angstrom=lambda_angstrom,
        intensity_threshold=float(detector_cfg.get("intensity_threshold", 1.0)),
        two_theta_range=(0.0, float(two_theta_max)),
        include_rods_flag=bool(ht_cfg.get("include_rods", False)),
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        miller_generator=miller_generator,
        inject_fractional_reflections=inject_fractional_reflections,
    )
    structure_model_state.occ_vars = [_HeadlessVar(value) for value in occ_values]
    structure_model_state.atom_site_fract_vars = [
        {"x": _HeadlessVar(x_val), "y": _HeadlessVar(y_val), "z": _HeadlessVar(z_val)}
        for x_val, y_val, z_val in atom_site_values
    ]

    atom_site_override_state = AtomSiteOverrideState()
    active_cif_path = gui_structure_model.active_primary_cif_path(
        structure_model_state,
        atom_site_override_state,
        atom_site_values=atom_site_values,
    )
    iodine_z_current = gui_structure_model.current_iodine_z(
        structure_model_state,
        atom_site_override_state,
        active_cif_path=active_cif_path,
        atom_site_values=atom_site_values,
    )

    gui_structure_model.rebuild_diffraction_inputs(
        structure_model_state,
        new_occ=occ_values,
        p_vals=[
            _saved_state_float(saved_variables, "p0_var", p_defaults[0]),
            _saved_state_float(saved_variables, "p1_var", p_defaults[1]),
            _saved_state_float(saved_variables, "p2_var", p_defaults[2]),
        ],
        weights=gui_controllers.normalize_stacking_weight_values(
            [
                _saved_state_float(saved_variables, "w0_var", w_defaults[0]),
                _saved_state_float(saved_variables, "w1_var", w_defaults[1]),
                _saved_state_float(saved_variables, "w2_var", w_defaults[2]),
            ]
        ),
        a_axis=float(a_var.get()),
        c_axis=float(c_var.get()),
        finite_stack_flag=_saved_state_bool(saved_variables, "finite_stack_var", bool(ht_cfg.get("finite_stack", True))),
        layers=max(1, _saved_state_int(saved_variables, "stack_layers_var", ht_cfg.get("stack_layers", 50))),
        phase_delta_expression_current=str(_saved_state_var_value(saved_variables, "phase_delta_expr_var", ht_cfg.get("phase_delta_expression", DEFAULT_PHASE_DELTA_EXPRESSION))),
        phi_l_divisor_current=_saved_state_float(saved_variables, "phi_l_divisor_var", ht_cfg.get("phi_l_divisor", DEFAULT_PHI_L_DIVISOR)),
        atom_site_values=atom_site_values,
        iodine_z_current=float(iodine_z_current),
        atom_site_override_state=atom_site_override_state,
        simulation_runtime_state=SimulationRuntimeState(),
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        build_intensity_dataframes=build_intensity_dataframes,
        apply_bragg_qr_filters=lambda *args, **kwargs: None,
        schedule_update=lambda: None,
        weight1=_saved_state_float(saved_variables, "weight1_var", 0.5 if secondary_cif_path else 1.0),
        weight2=_saved_state_float(saved_variables, "weight2_var", 0.5 if secondary_cif_path else 0.0),
        force=True,
        trigger_update=False,
    )

    theta_defaults = {"theta_initial": theta_initial_default}
    background_runtime_state = SimpleNamespace(current_background_index=int(current_background_index))

    def _current_geometry_fit_background_indices(*, strict: bool = False) -> list[int]:
        return gui_background_theta.current_geometry_fit_background_indices(
            osc_files=osc_files,
            current_background_index=int(background_runtime_state.current_background_index),
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            strict=bool(strict),
        )

    def _geometry_fit_uses_shared_theta_offset(selected_indices: list[int] | None = None) -> bool:
        return gui_background_theta.geometry_fit_uses_shared_theta_offset(
            selected_indices,
            osc_files=osc_files,
            current_background_index=int(background_runtime_state.current_background_index),
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        )

    def _current_geometry_theta_offset(*, strict: bool = False) -> float:
        return gui_background_theta.current_geometry_theta_offset(
            geometry_theta_offset_var=geometry_theta_offset_var,
            strict=bool(strict),
        )

    def _current_background_theta_values(*, strict_count: bool = False) -> list[float]:
        return gui_background_theta.current_background_theta_values(
            osc_files=osc_files,
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=theta_initial_default,
            background_theta_list_var=background_theta_list_var,
            strict_count=bool(strict_count),
        )

    def _background_theta_for_index(index: int, *, strict_count: bool = False) -> float:
        return gui_background_theta.background_theta_for_index(
            index,
            osc_files=osc_files,
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=theta_initial_default,
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            current_background_index=int(background_runtime_state.current_background_index),
            strict_count=bool(strict_count),
        )

    def _apply_background_theta_metadata(*, trigger_update: bool = False, sync_live_theta: bool = True) -> bool:
        return gui_background_theta.apply_background_theta_metadata(
            osc_files=osc_files,
            current_background_index=int(background_runtime_state.current_background_index),
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=theta_initial_default,
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=None,
            schedule_update=None,
            progress_label=None,
            trigger_update=bool(trigger_update),
            sync_live_theta=bool(sync_live_theta),
        )

    def _apply_geometry_fit_background_selection(*, trigger_update: bool = False, sync_live_theta: bool = True) -> bool:
        return gui_background_theta.apply_geometry_fit_background_selection(
            osc_files=osc_files,
            current_background_index=int(background_runtime_state.current_background_index),
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=theta_initial_default,
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            fit_theta_checkbutton=None,
            theta_controls={},
            set_background_file_status_text=None,
            schedule_update=None,
            progress_label_geometry=None,
            trigger_update=bool(trigger_update),
            sync_live_theta=bool(sync_live_theta),
        )

    if not _apply_background_theta_metadata(trigger_update=False, sync_live_theta=True):
        raise ValueError("Saved GUI state contains invalid background theta settings.")
    if not _apply_geometry_fit_background_selection(trigger_update=False, sync_live_theta=True):
        raise ValueError("Saved GUI state contains an invalid geometry-fit background selection.")

    n2_value = resolve_index_of_refraction(float(lambda_angstrom) * 1.0e-10, cif_path=str(active_cif_path))
    mosaic_params, sample_count = _build_headless_geometry_mosaic_params(
        saved_variables=saved_variables,
        beam_config=beam_cfg,
        lambda_angstrom=lambda_angstrom,
        active_cif_path=str(active_cif_path),
    )
    simulation_runtime_state = SimulationRuntimeState(
        num_samples=int(sample_count),
        profile_cache=copy.deepcopy(mosaic_params),
    )
    value_callbacks = gui_geometry_fit.build_runtime_geometry_fit_value_callbacks(
        gui_geometry_fit.GeometryFitRuntimeValueBindings(
            fit_zb_var=fit_zb_var,
            fit_zs_var=fit_zs_var,
            fit_theta_var=fit_theta_var,
            fit_psi_z_var=fit_psi_z_var,
            fit_chi_var=fit_chi_var,
            fit_cor_var=fit_cor_var,
            fit_gamma_var=fit_gamma_var,
            fit_Gamma_var=fit_Gamma_var,
            fit_dist_var=fit_dist_var,
            fit_a_var=fit_a_var,
            fit_c_var=fit_c_var,
            fit_center_x_var=fit_center_x_var,
            fit_center_y_var=fit_center_y_var,
            zb_var=zb_var,
            zs_var=zs_var,
            theta_initial_var=theta_initial_var,
            psi_z_var=psi_z_var,
            chi_var=chi_var,
            cor_angle_var=cor_angle_var,
            sample_width_var=sample_width_var,
            sample_length_var=sample_length_var,
            sample_depth_var=sample_depth_var,
            gamma_var=gamma_var,
            Gamma_var=Gamma_var,
            corto_detector_var=corto_detector_var,
            a_var=a_var,
            c_var=c_var,
            center_x_var=center_x_var,
            center_y_var=center_y_var,
            debye_x_var=debye_x_var,
            debye_y_var=debye_y_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            current_background_index=lambda: int(background_runtime_state.current_background_index),
            geometry_fit_uses_shared_theta_offset=lambda: _geometry_fit_uses_shared_theta_offset(),
            current_geometry_theta_offset=_current_geometry_theta_offset,
            background_theta_for_index=_background_theta_for_index,
            build_mosaic_params=lambda: mosaic_params,
            current_optics_mode_flag=lambda: _normalize_headless_optics_mode_flag(_saved_state_var_value(saved_variables, "optics_mode_var", "fast")),
            lambda_value=float(lambda_angstrom),
            psi=float(psi_default),
            n2=lambda: n2_value,
            pixel_size_value=float(pixel_size_m),
        )
    )

    pairs_by_background: dict[int, list[dict[str, object]]] = {}

    def _replace_pairs_by_background(new_map: dict[int, list[dict[str, object]]]) -> None:
        pairs_by_background.clear()
        pairs_by_background.update({int(idx): [dict(entry) for entry in entries] for idx, entries in new_map.items()})

    gui_manual_geometry.apply_geometry_manual_pairs_rows(
        geometry_state.get("manual_pairs", []),
        osc_files=osc_files,
        pairs_for_index=lambda idx: gui_manual_geometry.geometry_manual_pairs_for_index(idx, pairs_by_background=pairs_by_background),
        replace_pairs_by_background=_replace_pairs_by_background,
        clear_preview_artists=lambda **kwargs: None,
        cancel_pick_session=lambda **kwargs: None,
        invalidate_pick_cache=lambda: None,
        clear_manual_undo_stack=lambda: None,
        clear_geometry_fit_undo_stack=lambda: None,
        render_current_pairs=lambda **kwargs: None,
        update_button_label=lambda: None,
        refresh_status=lambda: None,
        replace_existing=True,
    )

    background_cache = {
        "background_images": [None] * len(osc_files),
        "background_images_native": [None] * len(osc_files),
        "background_images_display": [None] * len(osc_files),
        "current_background_image": None,
        "current_background_display": None,
    }

    def _load_background_by_index(index: int) -> tuple[np.ndarray, np.ndarray]:
        updated = gui_background.load_background_image_by_index(
            int(index),
            osc_files=osc_files,
            background_images=background_cache["background_images"],
            background_images_native=background_cache["background_images_native"],
            background_images_display=background_cache["background_images_display"],
            display_rotate_k=HEADLESS_GEOMETRY_BACKGROUND_DISPLAY_ROTATE_K,
            read_osc=read_osc,
        )
        background_cache["background_images"] = list(updated["background_images"])
        background_cache["background_images_native"] = list(updated["background_images_native"])
        background_cache["background_images_display"] = list(updated["background_images_display"])
        if int(index) == int(background_runtime_state.current_background_index):
            background_cache["current_background_image"] = np.asarray(updated["background_image"])
            background_cache["current_background_display"] = np.asarray(updated["background_display"])
        return np.asarray(updated["background_image"]), np.asarray(updated["background_display"])

    def _current_background_native() -> np.ndarray:
        native = background_cache["current_background_image"]
        if native is None:
            native, _display = _load_background_by_index(int(background_runtime_state.current_background_index))
        return np.asarray(native)

    def _current_background_display() -> np.ndarray:
        display = background_cache["current_background_display"]
        if display is None:
            _native, display = _load_background_by_index(int(background_runtime_state.current_background_index))
        return np.asarray(display)

    _load_background_by_index(int(background_runtime_state.current_background_index))
    backend_rotation_k = int(flags_state.get("background_backend_rotation_k", 0) or 0)
    backend_flip_x = _saved_state_bool(flags_state, "background_backend_flip_x", False)
    backend_flip_y = _saved_state_bool(flags_state, "background_backend_flip_y", False)

    simulation_callbacks = gui_geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        build_geometry_fit_central_mosaic_params=build_geometry_fit_central_mosaic_params,
        process_peaks_parallel=process_peaks_parallel,
        hit_tables_to_max_positions=hit_tables_to_max_positions,
        native_sim_to_display_coords=lambda col, row, image_shape: gui_geometry_overlay.native_sim_to_display_coords(col, row, image_shape, sim_display_rotate_k=HEADLESS_GEOMETRY_SIM_DISPLAY_ROTATE_K),
        peak_table_lattice_factory=None,
        primary_a_factory=lambda: float(a_var.get()),
        primary_c_factory=lambda: float(c_var.get()),
        default_source_label="primary",
        round_pixel_centers=False,
        default_solve_q_steps=int(mosaic_params["solve_q_steps"]),
        default_solve_q_rel_tol=float(mosaic_params["solve_q_rel_tol"]),
        default_solve_q_mode=int(mosaic_params["solve_q_mode"]),
    )

    projection_callbacks = gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=False,
        last_caked_background_image_unscaled=None,
        last_caked_radial_values=None,
        last_caked_azimuth_values=None,
        current_background_display=_current_background_display,
        current_background_native=_current_background_native,
        center=lambda: [float(center_x_var.get()), float(center_y_var.get())],
        detector_distance=lambda: float(corto_detector_var.get()),
        pixel_size=float(pixel_size_m),
        rotate_point_for_display=gui_geometry_overlay.rotate_point_for_display,
        display_rotate_k=HEADLESS_GEOMETRY_BACKGROUND_DISPLAY_ROTATE_K,
        current_geometry_fit_params=value_callbacks.current_params,
        simulate_preview_style_peaks_for_fit=simulation_callbacks.simulate_preview_style_peaks,
        miller=lambda: structure_model_state.miller,
        intensities=lambda: structure_model_state.intensities,
        image_size=int(image_size),
        display_to_native_sim_coords=lambda col, row, image_shape: gui_geometry_overlay.display_to_native_sim_coords(col, row, image_shape, sim_display_rotate_k=HEADLESS_GEOMETRY_SIM_DISPLAY_ROTATE_K),
    )

    manual_dataset_bindings = gui_geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=osc_files,
        current_background_index=int(background_runtime_state.current_background_index),
        image_size=int(image_size),
        display_rotate_k=HEADLESS_GEOMETRY_BACKGROUND_DISPLAY_ROTATE_K,
        geometry_manual_pairs_for_index=lambda idx: gui_manual_geometry.geometry_manual_pairs_for_index(idx, pairs_by_background=pairs_by_background),
        load_background_by_index=_load_background_by_index,
        apply_background_backend_orientation=lambda image: gui_background.apply_background_backend_orientation(image, flip_x=backend_flip_x, flip_y=backend_flip_y, rotation_k=backend_rotation_k),
        geometry_manual_simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        geometry_manual_simulated_lookup=projection_callbacks.simulated_lookup,
        geometry_manual_entry_display_coords=projection_callbacks.entry_display_coords,
        unrotate_display_peaks=gui_geometry_overlay.unrotate_display_peaks,
        display_to_native_sim_coords=lambda col, row, image_shape: gui_geometry_overlay.display_to_native_sim_coords(col, row, image_shape, sim_display_rotate_k=HEADLESS_GEOMETRY_SIM_DISPLAY_ROTATE_K),
        select_fit_orientation=gui_geometry_overlay.select_fit_orientation,
        apply_orientation_to_entries=gui_geometry_overlay.apply_orientation_to_entries,
        orient_image_for_fit=gui_geometry_overlay.orient_image_for_fit,
    )

    current_var_names = value_callbacks.current_var_names()
    preserve_live_theta = "theta_initial" not in current_var_names and "theta_offset" not in current_var_names
    prepare_result = gui_geometry_fit.prepare_runtime_geometry_fit_run(
        params=value_callbacks.current_params(),
        var_names=current_var_names,
        preserve_live_theta=preserve_live_theta,
        bindings=gui_geometry_fit.GeometryFitRuntimePreparationBindings(
            fit_config=fit_cfg,
            theta_initial=theta_initial_var.get(),
            apply_geometry_fit_background_selection=_apply_geometry_fit_background_selection,
            current_geometry_fit_background_indices=_current_geometry_fit_background_indices,
            geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
            apply_background_theta_metadata=_apply_background_theta_metadata,
            current_background_theta_values=_current_background_theta_values,
            current_geometry_theta_offset=_current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=lambda: None,
            manual_dataset_bindings=manual_dataset_bindings,
            build_runtime_config=lambda _fit_params: copy.deepcopy(fit_geometry_cfg),
        ),
    )
    if prepare_result.prepared_run is None:
        raise RuntimeError(prepare_result.error_text or "Geometry fit could not be prepared.")

    progress_state = {"text": ""}
    overlay_state = {"payload": None}
    dataset_cache_state = {"payload": None}
    command_log: list[str] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    downloads_dir = Path(output_dir or Path(source_path).resolve().parent)
    execution_result = gui_geometry_fit.execute_runtime_geometry_fit(
        prepared_run=prepare_result.prepared_run,
        var_names=current_var_names,
        preserve_live_theta=preserve_live_theta,
        solve_fit=fit_geometry_parameters,
        setup=gui_geometry_fit.build_runtime_geometry_fit_execution_setup(
            prepared_run=prepare_result.prepared_run,
            mosaic_params=mosaic_params,
            stamp=stamp,
            downloads_dir=downloads_dir,
            simulation_runtime_state=simulation_runtime_state,
            background_runtime_state=background_runtime_state,
            theta_initial_var=theta_initial_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            current_ui_params=value_callbacks.current_ui_params,
            var_map=value_callbacks.var_map,
            background_theta_for_index=_background_theta_for_index,
            refresh_status=lambda: None,
            update_manual_pick_button_label=lambda: None,
            capture_undo_state=lambda: {},
            push_undo_state=lambda _value: None,
            replace_dataset_cache=lambda payload_value: dataset_cache_state.update({"payload": copy.deepcopy(payload_value)}),
            request_preview_skip_once=lambda: None,
            schedule_update=lambda: None,
            draw_overlay_records=lambda records, marker_limit: None,
            draw_initial_pairs_overlay=lambda pairs, marker_limit: None,
            set_last_overlay_state=lambda payload_value: overlay_state.update({"payload": copy.deepcopy(payload_value)}),
            set_progress_text=lambda text: progress_state.update({"text": str(text)}),
            cmd_line=lambda text: command_log.append(str(text)),
            solver_inputs=gui_geometry_fit.GeometryFitRuntimeSolverInputs(
                miller=structure_model_state.miller,
                intensities=structure_model_state.intensities,
                image_size=int(image_size),
            ),
            sim_display_rotate_k=HEADLESS_GEOMETRY_SIM_DISPLAY_ROTATE_K,
            background_display_rotate_k=HEADLESS_GEOMETRY_BACKGROUND_DISPLAY_ROTATE_K,
            simulate_and_compare_hkl=simulate_and_compare_hkl,
            aggregate_match_centers=gui_geometry_overlay.aggregate_match_centers,
            build_overlay_records=gui_geometry_overlay.build_geometry_fit_overlay_records,
            compute_frame_diagnostics=gui_geometry_overlay.compute_geometry_overlay_frame_diagnostics,
        ),
    )
    if execution_result.error_text:
        raise RuntimeError(execution_result.error_text)
    apply_result = execution_result.apply_result
    if apply_result is None:
        raise RuntimeError("Geometry fit did not return an apply result.")
    if not bool(apply_result.accepted):
        rejection_reason = str(apply_result.rejection_reason or "geometry fit solution was rejected")
        if execution_result.log_path is not None:
            rejection_reason += f" (log: {execution_result.log_path})"
        raise RuntimeError(rejection_reason)

    updated_variables = _saved_state_section(updated_state, "variables")
    updated_variables.update(
        {
            "fit_zb_var": fit_zb_var.get(),
            "fit_zs_var": fit_zs_var.get(),
            "fit_theta_var": fit_theta_var.get(),
            "fit_psi_z_var": fit_psi_z_var.get(),
            "fit_chi_var": fit_chi_var.get(),
            "fit_cor_var": fit_cor_var.get(),
            "fit_gamma_var": fit_gamma_var.get(),
            "fit_Gamma_var": fit_Gamma_var.get(),
            "fit_dist_var": fit_dist_var.get(),
            "fit_a_var": fit_a_var.get(),
            "fit_c_var": fit_c_var.get(),
            "fit_center_x_var": fit_center_x_var.get(),
            "fit_center_y_var": fit_center_y_var.get(),
            "theta_initial_var": theta_initial_var.get(),
            "cor_angle_var": cor_angle_var.get(),
            "chi_var": chi_var.get(),
            "psi_z_var": psi_z_var.get(),
            "zs_var": zs_var.get(),
            "zb_var": zb_var.get(),
            "gamma_var": gamma_var.get(),
            "Gamma_var": Gamma_var.get(),
            "corto_detector_var": corto_detector_var.get(),
            "a_var": a_var.get(),
            "c_var": c_var.get(),
            "center_x_var": center_x_var.get(),
            "center_y_var": center_y_var.get(),
            "sample_width_var": sample_width_var.get(),
            "sample_length_var": sample_length_var.get(),
            "sample_depth_var": sample_depth_var.get(),
            "debye_x_var": debye_x_var.get(),
            "debye_y_var": debye_y_var.get(),
            "background_theta_list_var": background_theta_list_var.get(),
            "geometry_theta_offset_var": geometry_theta_offset_var.get(),
            "geometry_fit_background_selection_var": geometry_fit_background_selection_var.get(),
        }
    )
    updated_state["variables"] = updated_variables
    updated_state["dynamic_lists"] = {
        **_saved_state_section(updated_state, "dynamic_lists"),
        "occupancy_values": [float(var.get()) for var in structure_model_state.occ_vars],
        "atom_site_fractional_values": [
            {"x": float(row["x"].get()), "y": float(row["y"].get()), "z": float(row["z"].get())}
            for row in structure_model_state.atom_site_fract_vars
            if isinstance(row, Mapping)
        ],
    }
    updated_state["files"] = {
        **_saved_state_section(updated_state, "files"),
        "primary_cif_path": str(primary_cif_path),
        "secondary_cif_path": str(secondary_cif_path) if secondary_cif_path else None,
        "background_files": list(osc_files),
        "current_background_index": int(background_runtime_state.current_background_index),
    }
    return (
        updated_state,
        {
            "accepted": True,
            "log_path": str(execution_result.log_path) if execution_result.log_path is not None else None,
            "matched_peaks_path": str(apply_result.postprocess.save_path) if apply_result.postprocess is not None else None,
            "progress_text": progress_state["text"],
            "command_log": list(command_log),
            "overlay_state": overlay_state["payload"],
            "dataset_cache": dataset_cache_state["payload"],
        },
    )


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


def _resolve_fit_geometry_input_path(args: argparse.Namespace) -> Path:
    """Return the saved GUI-state path requested for headless geometry fitting."""

    for attr_name in ("state", "input_state", "gui_state", "source_state"):
        raw_value = getattr(args, attr_name, None)
        if raw_value:
            return Path(str(raw_value)).expanduser()
    raise ValueError("Provide a saved GUI state JSON path.")


def _resolve_fit_geometry_output_path(
    args: argparse.Namespace,
    *,
    input_path: Path,
) -> Path:
    """Return the output GUI-state path for headless geometry fitting."""

    if bool(getattr(args, "in_place", False)):
        return input_path

    for attr_name in ("out_state", "output_state", "output"):
        raw_value = getattr(args, attr_name, None)
        if raw_value:
            return Path(str(raw_value)).expanduser()

    suffix = input_path.suffix or ".json"
    return input_path.with_name(f"{input_path.stem}_fit{suffix}")


def _extract_fit_geometry_state_result(
    result: object,
) -> tuple[dict[str, object], dict[str, object]]:
    """Normalize one headless geometry-fit return payload."""

    report: dict[str, object] = {}
    state_value = result
    if isinstance(result, tuple) and result:
        state_value = result[0]
        if len(result) > 1 and isinstance(result[1], Mapping):
            report = dict(result[1])

    if isinstance(state_value, Mapping) and isinstance(state_value.get("state"), Mapping):
        return dict(state_value["state"]), report
    if isinstance(state_value, Mapping):
        return dict(state_value), report
    raise ValueError("Headless geometry fit did not return a GUI-state mapping.")


def _cmd_fit_geometry(args: argparse.Namespace) -> None:
    """Run geometry fitting from a saved GUI state without launching the GUI."""

    input_path = _resolve_fit_geometry_input_path(args)
    output_path = _resolve_fit_geometry_output_path(args, input_path=input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = load_gui_state_file(input_path)
        state_result, report = _extract_fit_geometry_state_result(
            run_headless_geometry_fit(
                payload,
                source_path=input_path,
                output_dir=output_path.parent,
            )
        )
        save_gui_state_file(output_path, state_result)
        print(f"Wrote fitted GUI state to {output_path}")
        log_path = report.get("log_path")
        matched_peaks_path = report.get("matched_peaks_path")
        if log_path:
            print(f"Geometry fit log: {log_path}")
        if matched_peaks_path:
            print(f"Matched peaks: {matched_peaks_path}")
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run RA-SIM tools.")
    subparsers = ap.add_subparsers(dest="command")

    gui_parser = subparsers.add_parser(
        "gui",
        help="Launch the RA-SIM Tkinter GUI.",
    )
    gui_parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Do not write the initial intensity Excel file on startup.",
    )
    gui_parser.set_defaults(func=_cmd_gui)

    calibrant_parser = subparsers.add_parser(
        "calibrant",
        aliases=["calibrant-fit"],
        help="Launch the packaged hBN calibrant fitter GUI.",
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

    fit_geometry_parser = subparsers.add_parser(
        "fit-geometry",
        help="Run geometry fitting from a saved GUI state without launching the GUI.",
    )
    fit_geometry_parser.add_argument(
        "state",
        help="Path to a saved GUI state JSON file.",
    )
    fit_geometry_parser.add_argument(
        "--out-state",
        dest="out_state",
        default=None,
        help="Optional output GUI state path. Defaults to '<input>_fit.json'.",
    )
    fit_geometry_parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input GUI state instead of writing a new file.",
    )
    fit_geometry_parser.set_defaults(func=_cmd_fit_geometry)

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
        "fit-geometry",
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
