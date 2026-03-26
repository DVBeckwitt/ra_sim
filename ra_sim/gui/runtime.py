#!/usr/bin/env python3

"""Main application entry point for running the Tk based GUI."""

import os
import sys

from ra_sim.gui import bootstrap as gui_bootstrap

write_excel = False


gui_bootstrap.early_main_bootstrap(__name__)

import math
import json
import copy
import re
from collections import defaultdict, namedtuple
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Sequence
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from OSC_Reader import read_osc
import CifFile

from ra_sim.utils.stacking_fault import (
    DEFAULT_PHASE_DELTA_EXPRESSION,
    DEFAULT_PHI_L_DIVISOR,
    _infer_iodine_z_like_diffuse,
    ht_Iinf_dict,
    ht_dict_to_arrays,
    ht_dict_to_qr_dict,
    normalize_phi_l_divisor,
    normalize_phase_delta_expression,
    validate_phase_delta_expression,
)

from ra_sim.utils.calculations import IndexofRefraction
from ra_sim.io.file_parsing import parse_poni_file, Open_ASC
from ra_sim.utils.tools import (
    miller_generator,
    view_azimuthal_radial,
    detect_blobs,
    inject_fractional_reflections,
    build_intensity_dataframes,
    detector_two_theta_max,
    DEFAULT_PIXEL_SIZE_M,
)
from ra_sim.io.data_loading import (
    load_geometry_placements_file,
    load_and_format_reference_profiles,
    save_all_parameters,
    save_geometry_placements_file,
    load_parameters,
    load_gui_state_file,
    save_gui_state_file,
)
from ra_sim.fitting.optimization import (
    build_geometry_fit_central_mosaic_params,
    build_measured_dict,
    fit_geometry_parameters,
    fit_mosaic_widths_separable,
    simulate_and_compare_hkl,
)
from ra_sim.fitting.background_peak_matching import (
    build_background_peak_context,
    match_simulated_peaks_to_peak_context,
)
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import (
    DEFAULT_SOLVE_Q_REL_TOL,
    DEFAULT_SOLVE_Q_MODE,
    DEFAULT_SOLVE_Q_STEPS,
    MAX_SOLVE_Q_REL_TOL,
    MAX_SOLVE_Q_STEPS,
    MIN_SOLVE_Q_REL_TOL,
    MIN_SOLVE_Q_STEPS,
    SOLVE_Q_MODE_ADAPTIVE,
    SOLVE_Q_MODE_UNIFORM,
    hit_tables_to_max_positions,
    OPTICS_MODE_EXACT,
    OPTICS_MODE_FAST,
    process_peaks_parallel,
    process_qr_rods_parallel,
)
from ra_sim.simulation.intersection_analysis import (
    BeamSamples as IntersectionBeamSamples,
    IntersectionGeometry,
    MosaicParams as IntersectionMosaicParams,
    analyze_reflection_intersection,
    plot_intersection_analysis,
    project_qr_cylinder_to_detector,
)
from ra_sim.simulation.diffraction_debug import (
    process_peaks_parallel_debug,
    process_qr_rods_parallel_debug,
    dump_debug_log,
)
from ra_sim.simulation.simulation import simulate_diffraction
from ra_sim.gui import background as gui_background
from ra_sim.gui import background_manager as gui_background_manager
from ra_sim.gui import background_theta as gui_background_theta
from ra_sim.gui import bragg_qr_manager as gui_bragg_qr_manager
from ra_sim.gui import geometry_q_group_manager as gui_geometry_q_group_manager
from ra_sim.gui import geometry_overlay as gui_geometry_overlay
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.gui import views as gui_views
from ra_sim.gui import structure_model as gui_structure_model
from ra_sim.gui.geometry_overlay import (
    build_geometry_fit_overlay_records,
    compute_geometry_overlay_frame_diagnostics,
)
from ra_sim.gui import overlays as gui_overlays
from ra_sim.gui import peak_selection as gui_peak_selection
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui import state as gui_state
from ra_sim.gui import state_io as gui_state_io
from ra_sim.gui.qr_cylinder_overlay import interpolate_trace_to_caked_coords
from ra_sim.gui.sliders import create_slider
from ra_sim.gui.diffuse_cif_toggle import (
    open_diffuse_cif_toggle_algebraic,
    export_algebraic_ht_txt,
)
from ra_sim.debug_utils import debug_print, is_debug_enabled
from ra_sim.hbn import (
    build_hbn_geometry_debug_trace,
    convert_hbn_bundle_geometry_to_simulation,
    format_hbn_geometry_debug_trace,
    load_bundle_npz,
    load_tilt_hint,
)
from ra_sim.gui.collapsible import CollapsibleFrame


turbo = matplotlib.colormaps.get_cmap('turbo').resampled(256)
turbo_rgba = turbo(np.linspace(0, 1, 256))
turbo_rgba[0] = [1.0, 1.0, 1.0, 1.0]       # make the 0-bin white
turbo_white0 = ListedColormap(turbo_rgba, name='turbo_white0')
turbo_white0.set_bad('white')              # NaNs will also show white


# Force TkAgg backend to ensure GUI usage
matplotlib.use('TkAgg')
# Default to non-debug mode; set RA_SIM_DEBUG=1 to enable diagnostics.
os.environ.setdefault("RA_SIM_DEBUG", "0")
# Enable extra diagnostics when the RA_SIM_DEBUG environment variable is set.
DEBUG_ENABLED = is_debug_enabled()
if DEBUG_ENABLED:
    print("Debug mode active (RA_SIM_DEBUG=1)")
    from ra_sim.debug_utils import enable_numba_logging
    enable_numba_logging()

# Toggle creation of backend orientation controls (kept off while automated
# diagnostics try permutations internally).
BACKEND_ORIENTATION_UI_ENABLED = False
BACKGROUND_BACKEND_DEBUG_UI_ENABLED = False
HBN_GEOMETRY_DEBUG_ENABLED = False


###############################################################################
#                          DATA & PARAMETER SETUP
###############################################################################
from ra_sim.path_config import (
    get_path,
    get_path_first,
    get_dir,
)


def _ensure_triplet(values, fallback):
    """Return a 3-element list combining *values* with *fallback*."""

    if not isinstance(values, (list, tuple)):
        return list(fallback)
    merged = list(fallback)
    for idx, val in enumerate(values[:3]):
        merged[idx] = val
    return merged


def _ensure_numeric_vector(values, fallback, length):
    """Return a float vector of ``length`` using ``values`` with fallback fill."""

    try:
        target_len = int(length)
    except (TypeError, ValueError):
        target_len = 1
    target_len = max(1, target_len)

    out = []
    if isinstance(values, (list, tuple, np.ndarray)):
        for raw in values:
            try:
                out.append(float(raw))
            except (TypeError, ValueError):
                continue

    if not out:
        if isinstance(fallback, (list, tuple, np.ndarray)):
            for raw in fallback:
                try:
                    out.append(float(raw))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                out = [float(fallback)]
            except (TypeError, ValueError):
                out = [1.0]

    if not out:
        out = [1.0]

    if len(out) < target_len:
        out.extend([out[-1]] * (target_len - len(out)))
    elif len(out) > target_len:
        out = out[:target_len]
    return out


app_state = gui_controllers.build_initial_state()
background_runtime_state = app_state.background_runtime
peak_selection_state = app_state.peak_selection
atom_site_override_state = app_state.atom_site_override
geometry_runtime_state = app_state.geometry_runtime
simulation_runtime_state = app_state.simulation_runtime
bragg_qr_manager_state = app_state.bragg_qr_manager

instrument_config = app_state.instrument_config.get("instrument", {})
detector_config = instrument_config.get("detector", {})
geometry_config = instrument_config.get("geometry_defaults", {})
beam_config = instrument_config.get("beam", {})
sample_config = instrument_config.get("sample_orientation", {})
debye_config = instrument_config.get("debye_waller", {})
occupancy_config = instrument_config.get("occupancies", {})
hendricks_config = instrument_config.get("hendricks_teller", {})
output_config = instrument_config.get("output", {})
fit_config = instrument_config.get("fit", {})

background_runtime_state.osc_files = get_path_first("simulation_background_osc_files", "osc_files")
if isinstance(background_runtime_state.osc_files, str):
    background_runtime_state.osc_files = [background_runtime_state.osc_files]
if not background_runtime_state.osc_files:
    raise ValueError(
        "No oscillation images configured in simulation_background_osc_files/osc_files"
    )
app_state.file_paths["simulation_background_osc_files"] = list(background_runtime_state.osc_files)
background_runtime_state.osc_files = list(background_runtime_state.osc_files)

# Background and simulated overlays can use different display orientations.
# ``k`` is the np.rot90 factor; -1 is 90° clockwise, 0 keeps native orientation.
DISPLAY_ROTATE_K = -1
SIM_DISPLAY_ROTATE_K = 0
# hBN fitter bundles are in native OSC orientation (no rot90).
HBN_FITTER_ROTATE_K = 0
# Simulation geometry runs in native simulation pixels; convert hBN bundle
# geometry so that the displayed simulation (after SIM_DISPLAY_ROTATE_K) lands
# in the same frame as the displayed background (after DISPLAY_ROTATE_K).
SIMULATION_GEOMETRY_ROTATE_K = DISPLAY_ROTATE_K - SIM_DISPLAY_ROTATE_K

# Preserve native-orientation copies for fitting/analysis. Load only the first
# background at startup and lazy-load the rest on demand to improve first paint.
_initial_background_state = gui_background.initialize_background_cache(
    str(background_runtime_state.osc_files[0]),
    total_count=len(background_runtime_state.osc_files),
    display_rotate_k=DISPLAY_ROTATE_K,
    read_osc=read_osc,
)
background_runtime_state.background_images = list(_initial_background_state["background_images"])
background_runtime_state.background_images_native = list(_initial_background_state["background_images_native"])
background_runtime_state.background_images_display = list(_initial_background_state["background_images_display"])
background_runtime_state.background_images = list(background_runtime_state.background_images)
background_runtime_state.background_images_native = list(background_runtime_state.background_images_native)
background_runtime_state.background_images_display = list(background_runtime_state.background_images_display)

# Parse geometry
poni_file_path = get_path("geometry_poni")
app_state.file_paths["geometry_poni"] = str(poni_file_path)
parameters = parse_poni_file(poni_file_path)

Distance_CoR_to_Detector = parameters.get(
    "Dist", geometry_config.get("distance_m", 0.075)
)
Gamma_initial = parameters.get("Rot1", geometry_config.get("rot1", 0.0))
gamma_initial = parameters.get("Rot2", geometry_config.get("rot2", 0.0))
poni1 = parameters.get("Poni1", geometry_config.get("poni1_m", 0.0))
poni2 = parameters.get("Poni2", geometry_config.get("poni2_m", 0.0))
wave_m = parameters.get("Wavelength", geometry_config.get("wavelength_m", 1e-10))
lambda_from_poni = wave_m * 1e10  # Convert m -> Å

tilt_hint = load_tilt_hint()
hinted_center_row = None
hinted_center_col = None
if tilt_hint:
    hinted_gamma = tilt_hint.get("gamma_deg")
    hinted_Gamma = tilt_hint.get("Gamma_deg")
    hinted_center_row = tilt_hint.get("center_row")
    hinted_center_col = tilt_hint.get("center_col")
    try:
        hinted_gamma = float(hinted_gamma)
        hinted_Gamma = float(hinted_Gamma)
    except (TypeError, ValueError):
        hinted_gamma = None
        hinted_Gamma = None
    try:
        hinted_center_row = float(hinted_center_row)
        hinted_center_col = float(hinted_center_col)
    except (TypeError, ValueError):
        hinted_center_row = None
        hinted_center_col = None
    if hinted_center_row is not None and hinted_center_col is not None:
        if not (np.isfinite(hinted_center_row) and np.isfinite(hinted_center_col)):
            hinted_center_row = None
            hinted_center_col = None
    if hinted_gamma is not None and np.isfinite(hinted_gamma):
        # Historical naming here is swapped: defaults['gamma'] reads Gamma_initial.
        Gamma_initial = hinted_gamma
    if hinted_Gamma is not None and np.isfinite(hinted_Gamma):
        # Historical naming here is swapped: defaults['Gamma'] reads gamma_initial.
        gamma_initial = hinted_Gamma
    hinted_distance = tilt_hint.get("distance_m")
    try:
        hinted_distance = float(hinted_distance)
    except (TypeError, ValueError):
        hinted_distance = None
    if hinted_distance is not None and np.isfinite(hinted_distance):
        Distance_CoR_to_Detector = hinted_distance
    center_text = ""
    if hinted_center_row is not None and hinted_center_col is not None:
        center_text = (
            f", center(row={hinted_center_row:.3f}, col={hinted_center_col:.3f})"
        )
    print(
        "Initialized detector tilt from last hBN fit profile "
        f"(sim γ={Gamma_initial:.4f} deg, sim Γ={gamma_initial:.4f} deg{center_text})."
    )
    if hinted_distance is not None and np.isfinite(hinted_distance):
        print(
            "Initialized detector distance from last hBN fit profile "
            f"(Dist={hinted_distance:.4f} m)."
        )

image_size = int(app_state.image_size)
pixel_size_m = float(detector_config.get("pixel_size_m", DEFAULT_PIXEL_SIZE_M))
resolution_sample_counts = {
    "Low": 25,
    "Medium": 250,
    "High": 500,
}
CUSTOM_SAMPLING_OPTION = "Custom"
simulation_runtime_state.num_samples = resolution_sample_counts["Low"]
write_excel = output_config.get("write_excel", write_excel)
intensity_threshold = detector_config.get("intensity_threshold", 1.0)
vmax_default = detector_config.get("vmax", 1000)
vmax_slider_max = detector_config.get("vmax_slider_max", 3000)

# Approximate beam center
center_default = [
    (poni2 / pixel_size_m),
    image_size - (poni1 / pixel_size_m)
]
if hinted_center_row is not None and hinted_center_col is not None:
    center_default = [hinted_center_row, hinted_center_col]

two_theta_max = detector_two_theta_max(
    image_size,
    center_default,
    Distance_CoR_to_Detector,
    pixel_size=pixel_size_m,
)
two_theta_range = (0.0, two_theta_max)

mx = hendricks_config.get("max_miller_index", 19)

fwhm2sigma = 1 / (2 * math.sqrt(2 * math.log(2)))
divergence_fwhm = beam_config.get("divergence_fwhm_deg", 0.05)
divergence_sigma = math.radians(divergence_fwhm * fwhm2sigma)

sigma_mosaic = math.radians(
    beam_config.get("sigma_mosaic_fwhm_deg", 0.8) * fwhm2sigma
)
gamma_mosaic = math.radians(
    beam_config.get("gamma_mosaic_fwhm_deg", 0.7) * fwhm2sigma
)
eta = beam_config.get("eta", 0.0)

theta_initial = sample_config.get("theta_initial_deg", 6.0)
cor_angle = sample_config.get("cor_deg", 0.0)
chi = sample_config.get("chi_deg", 0.0)
psi = sample_config.get("psi_deg", 0.0)
psi_z = sample_config.get("psi_z_deg", 0.0)
zb = sample_config.get("zb", 0.0)
bw_sigma = beam_config.get("bandwidth_sigma_fraction", 0.05e-3) * fwhm2sigma
zs = sample_config.get("zs", 0.0)
debye_x = debye_config.get("x", 0.0)
debye_y = debye_config.get("y", 0.0)

# Print the computed complex index of refraction on startup and exit
#print("Computed complex index of refraction n2:", n2)
#sys.exit(0)

bandwidth_percent_default = float(beam_config.get("bandwidth_percent", 0.7))
bandwidth = bandwidth_percent_default / 100.0

# NOTE: Occupancy defaults are sized to the GUI's unique atom-site controls
# after loading the CIF below.
occupancy_default_values = occupancy_config.get("default", [1.0, 1.0, 1.0])

# When enabled, additional fractional reflections ("rods")
# are injected between integer L values.
include_rods_flag = hendricks_config.get("include_rods", False)
sf_prune_bias_default = float(
    np.clip(hendricks_config.get("sf_prune_bias", 0.0), -1.0, 1.0)
)
try:
    solve_q_steps_default = int(
        round(float(beam_config.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS)))
    )
except (TypeError, ValueError):
    solve_q_steps_default = int(DEFAULT_SOLVE_Q_STEPS)
solve_q_steps_default = int(
    np.clip(solve_q_steps_default, MIN_SOLVE_Q_STEPS, MAX_SOLVE_Q_STEPS)
)
try:
    solve_q_rel_tol_default = float(
        beam_config.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL)
    )
except (TypeError, ValueError):
    solve_q_rel_tol_default = float(DEFAULT_SOLVE_Q_REL_TOL)
solve_q_rel_tol_default = float(
    np.clip(solve_q_rel_tol_default, MIN_SOLVE_Q_REL_TOL, MAX_SOLVE_Q_REL_TOL)
)
solve_q_mode_raw = beam_config.get("solve_q_mode", "uniform")
if isinstance(solve_q_mode_raw, (int, np.integer, float, np.floating)):
    solve_q_mode_default = (
        SOLVE_Q_MODE_UNIFORM
        if int(round(float(solve_q_mode_raw))) == SOLVE_Q_MODE_UNIFORM
        else SOLVE_Q_MODE_ADAPTIVE
    )
else:
    solve_q_mode_text = str(solve_q_mode_raw).strip().lower()
    if solve_q_mode_text in {"uniform", "fast", "0"}:
        solve_q_mode_default = SOLVE_Q_MODE_UNIFORM
    else:
        solve_q_mode_default = SOLVE_Q_MODE_UNIFORM

lambda_override = beam_config.get("wavelength_angstrom")
lambda_ = lambda_override if lambda_override is not None else lambda_from_poni
n2 = IndexofRefraction(float(lambda_) * 1.0e-10)

# Parameters and file paths.
cif_file = get_path("cif_file")
try:
    cif_file2 = get_path("cif_file2")
except KeyError:
    cif_file2 = None

# read with PyCifRW
cf    = CifFile.ReadCif(cif_file)
blk   = cf[list(cf.keys())[0]]


def _normalize_occupancy_label(raw_label, fallback_idx):
    return gui_structure_model.normalize_occupancy_label(raw_label, fallback_idx)


def _extract_occupancy_site_metadata(cif_block, cif_path):
    """Return (unique labels, expanded-site -> unique-label index mapping)."""

    return gui_structure_model.extract_occupancy_site_metadata(cif_block, cif_path)


occupancy_site_labels, occupancy_site_expanded_map = _extract_occupancy_site_metadata(blk, cif_file)
if occupancy_site_labels:
    occupancy_site_count = len(occupancy_site_labels)
else:
    occupancy_site_count = max(
        1,
        len(occupancy_default_values)
        if isinstance(occupancy_default_values, (list, tuple, np.ndarray))
        else 1,
    )
occ = _ensure_numeric_vector(
    occupancy_default_values,
    [1.0],
    occupancy_site_count,
)
occ = [min(1.0, max(0.0, float(v))) for v in occ]


def _expand_occupancy_values_for_generated_sites(occ_values):
    """Map unique-site occupancies to generated per-site occupancies for HT."""

    return gui_structure_model.expand_occupancy_values_for_generated_sites(
        occ_values,
        occupancy_site_labels=occupancy_site_labels,
        occupancy_site_expanded_map=occupancy_site_expanded_map,
    )


def _extract_atom_site_fractional_metadata(cif_block):
    """Return atom-site fractional coordinates from raw CIF loop rows."""

    return gui_structure_model.extract_atom_site_fractional_metadata(cif_block)


atom_site_fractional_metadata = _extract_atom_site_fractional_metadata(blk)
atom_site_fract_vars = []


def _atom_site_fractional_default_values():
    return gui_structure_model.atom_site_fractional_default_values(
        structure_model_state
    )


def _current_atom_site_fractional_values():
    """Return current atom-site fractional coordinates from GUI state."""

    return gui_structure_model.current_atom_site_fractional_values(
        structure_model_state,
        tcl_error_types=(tk.TclError,),
    )


def _atom_site_fractional_signature(values):
    return gui_structure_model.atom_site_fractional_signature(values)


def _atom_site_fractional_defaults_signature():
    return _atom_site_fractional_signature(_atom_site_fractional_default_values())


def _atom_site_fractional_values_are_default(values):
    return gui_structure_model.atom_site_fractional_values_are_default(
        structure_model_state,
        values,
    )


def _active_primary_cif_path(atom_site_values=None):
    """Return primary CIF path with optional atom-site coordinate overrides."""

    return gui_structure_model.active_primary_cif_path(
        structure_model_state,
        atom_site_override_state,
        atom_site_values=atom_site_values,
        tcl_error_types=(tk.TclError,),
    )


def _reset_atom_site_override_cache():
    gui_structure_model.reset_atom_site_override_cache(atom_site_override_state)

# pull the raw text
a_text = blk.get("_cell_length_a")
b_text = blk.get("_cell_length_b")
c_text = blk.get("_cell_length_c")

# strip the '(uncertainty)' and cast
def parse_cif_num(txt):
    return gui_structure_model.parse_cif_num(txt)

if a_text is None or b_text is None or c_text is None:
    raise ValueError("CIF is missing one or more required cell-length fields (_a/_b/_c).")

av = parse_cif_num(a_text)
bv = parse_cif_num(b_text)
cv = parse_cif_num(c_text)

if cif_file2:
    cf2  = CifFile.ReadCif(cif_file2)
    blk2 = cf2[list(cf2.keys())[0]]
    a2_text = blk2.get("_cell_length_a")
    c2_text = blk2.get("_cell_length_c")
    av2 = parse_cif_num(a2_text) if a2_text is not None else av
    cv2 = parse_cif_num(c2_text) if c2_text is not None else cv
else:
    av2 = None
    cv2 = None

energy = 6.62607e-34 * 2.99792458e8 / (lambda_*1e-10) / (1.602176634e-19)    # keV

p_defaults = _ensure_triplet(
    hendricks_config.get("default_p"), [0.01, 0.99, 0.5]
)
w_defaults = _ensure_triplet(
    hendricks_config.get("default_w"), [50.0, 50.0, 0.0]
)
phase_delta_expression_default = normalize_phase_delta_expression(
    hendricks_config.get(
        "phase_delta_expression",
        DEFAULT_PHASE_DELTA_EXPRESSION,
    ),
    fallback=DEFAULT_PHASE_DELTA_EXPRESSION,
)
try:
    phase_delta_expression_default = validate_phase_delta_expression(
        phase_delta_expression_default
    )
except ValueError:
    phase_delta_expression_default = DEFAULT_PHASE_DELTA_EXPRESSION
phi_l_divisor_default = normalize_phi_l_divisor(
    hendricks_config.get(
        "phi_l_divisor",
        DEFAULT_PHI_L_DIVISOR,
    ),
    fallback=DEFAULT_PHI_L_DIVISOR,
)
finite_stack_default = bool(hendricks_config.get("finite_stack", True))
stack_layers_default = int(
    max(1, float(hendricks_config.get("stack_layers", 50)))
)
try:
    iodine_z_default = _infer_iodine_z_like_diffuse(str(cif_file))
except Exception:
    iodine_z_default = None
if iodine_z_default is None or not np.isfinite(float(iodine_z_default)):
    iodine_z_default = 0.0
iodine_z_default = float(np.clip(float(iodine_z_default), 0.0, 1.0))

# ---------------------------------------------------------------------------
# Default GUI/fit parameter values. These must be defined before any calls
# that reference them (e.g. ``ht_Iinf_dict`` below).
# ---------------------------------------------------------------------------
defaults = {
    'theta_initial': theta_initial,
    'cor_angle': cor_angle,
    'gamma': Gamma_initial,
    'Gamma': gamma_initial,
    'chi': chi,
    'psi_z': psi_z,
    'zs': zs,
    'zb': zb,
    'debye_x': debye_x,
    'debye_y': debye_y,
    'corto_detector': Distance_CoR_to_Detector,
    'sigma_mosaic_deg': np.degrees(sigma_mosaic),
    'gamma_mosaic_deg': np.degrees(gamma_mosaic),
    'eta': eta,
    'a': av,
    'c': cv,
    'vmax': vmax_default,
    'p0': p_defaults[0],
    'p1': p_defaults[1],
    'p2': p_defaults[2],
    'w0': w_defaults[0],
    'w1': w_defaults[1],
    'w2': w_defaults[2],
    'iodine_z': iodine_z_default,
    'phase_delta_expression': phase_delta_expression_default,
    'phi_l_divisor': float(phi_l_divisor_default),
    'center_x': center_default[0],
    'center_y': center_default[1],
    'sampling_resolution': 'Low',
    'bandwidth_percent': float(np.clip(bandwidth_percent_default, 0.0, 10.0)),
    'sf_prune_bias': sf_prune_bias_default,
    'solve_q_steps': solve_q_steps_default,
    'solve_q_rel_tol': solve_q_rel_tol_default,
    'solve_q_mode': solve_q_mode_default,
    'finite_stack': finite_stack_default,
    'stack_layers': stack_layers_default,
    'optics_mode': 'fast',
}

# ---------------------------------------------------------------------------
# Replace the old miller_generator call with the new Hendricks–Teller helper.
# ---------------------------------------------------------------------------
structure_model_state = gui_structure_model.build_initial_structure_model_state(
    cif_file=cif_file,
    cf=cf,
    blk=blk,
    cif_file2=cif_file2,
    occupancy_site_labels=occupancy_site_labels,
    occupancy_site_expanded_map=occupancy_site_expanded_map,
    occ=occ,
    atom_site_fractional_metadata=atom_site_fractional_metadata,
    av=av,
    bv=bv,
    cv=cv,
    av2=av2,
    cv2=cv2,
    defaults=defaults,
    mx=mx,
    lambda_angstrom=lambda_,
    intensity_threshold=intensity_threshold,
    two_theta_range=two_theta_range,
    include_rods_flag=include_rods_flag,
    combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
    miller_generator=miller_generator,
    inject_fractional_reflections=inject_fractional_reflections,
    debug_print=debug_print,
)


def _sync_structure_model_aliases() -> None:
    global cif_file, cf, blk, cf2, blk2
    global occupancy_site_labels, occupancy_site_count, occupancy_site_expanded_map
    global occ, occ_vars
    global atom_site_fractional_metadata, atom_site_fract_vars
    global av, cv, av2, cv2
    global defaults
    global ht_cache_multi, ht_curves_cache
    global miller, intensities, degeneracy, details
    global df_summary, df_details
    global intensities_cif1, intensities_cif2
    global _last_occ_for_ht, _last_p_triplet, _last_weights
    global _last_a_for_ht, _last_c_for_ht, _last_iodine_z_for_ht
    global _last_phi_l_divisor, _last_phase_delta_expression
    global _last_finite_stack, _last_stack_layers
    global _last_atom_site_fractional_signature

    cif_file = structure_model_state.cif_file
    cf = structure_model_state.cf
    blk = structure_model_state.blk
    cf2 = structure_model_state.cf2
    blk2 = structure_model_state.blk2
    occupancy_site_labels = list(structure_model_state.occupancy_site_labels)
    occupancy_site_count = int(structure_model_state.occupancy_site_count)
    occupancy_site_expanded_map = list(structure_model_state.occupancy_site_expanded_map)
    occ = list(structure_model_state.occ)
    occ_vars = list(structure_model_state.occ_vars)
    atom_site_fractional_metadata = [
        dict(row) for row in structure_model_state.atom_site_fractional_metadata
    ]
    atom_site_fract_vars = list(structure_model_state.atom_site_fract_vars)
    av = float(structure_model_state.av)
    cv = float(structure_model_state.cv)
    av2 = structure_model_state.av2
    cv2 = structure_model_state.cv2
    defaults = structure_model_state.defaults
    ht_cache_multi = structure_model_state.ht_cache_multi
    ht_curves_cache = structure_model_state.ht_curves_cache
    miller = structure_model_state.miller
    intensities = structure_model_state.intensities
    degeneracy = structure_model_state.degeneracy
    details = structure_model_state.details
    df_summary = structure_model_state.df_summary
    df_details = structure_model_state.df_details
    intensities_cif1 = structure_model_state.intensities_cif1
    intensities_cif2 = structure_model_state.intensities_cif2
    _last_occ_for_ht = list(structure_model_state.last_occ_for_ht)
    _last_p_triplet = list(structure_model_state.last_p_triplet)
    _last_weights = list(structure_model_state.last_weights)
    _last_a_for_ht = float(structure_model_state.last_a_for_ht)
    _last_c_for_ht = float(structure_model_state.last_c_for_ht)
    _last_iodine_z_for_ht = float(structure_model_state.last_iodine_z_for_ht)
    _last_phi_l_divisor = float(structure_model_state.last_phi_l_divisor)
    _last_phase_delta_expression = str(structure_model_state.last_phase_delta_expression)
    _last_finite_stack = bool(structure_model_state.last_finite_stack)
    _last_stack_layers = int(structure_model_state.last_stack_layers)
    _last_atom_site_fractional_signature = tuple(
        structure_model_state.last_atom_site_fractional_signature
    )


_sync_structure_model_aliases()

primary_miller, primary_intens, _, _ = structure_model_state.ht_curves_cache["arrays"]
if DEBUG_ENABLED:
    from ra_sim.debug_utils import check_ht_arrays

    check_ht_arrays(primary_miller, primary_intens)
    debug_print("miller1 dtype:", primary_miller.dtype, "shape:", primary_miller.shape)
    debug_print("L range:", primary_miller[:, 2].min(), primary_miller[:, 2].max())
    debug_print(
        "intens1 dtype:",
        primary_intens.dtype,
        "min:",
        primary_intens.min(),
        "max:",
        primary_intens.max(),
    )
    debug_print("miller1 contiguous:", primary_miller.flags["C_CONTIGUOUS"])
    debug_print("intens1 contiguous:", primary_intens.flags["C_CONTIGUOUS"])

has_second_cif = structure_model_state.has_second_cif
weight1 = 0.5 if has_second_cif else 1.0
weight2 = 0.5 if has_second_cif else 0.0
if has_second_cif:
    debug_print("combined miller count:", structure_model_state.miller.shape[0])
else:
    debug_print("single CIF miller count:", structure_model_state.miller.shape[0])

simulation_runtime_state.sim_miller1_all = structure_model_state.sim_miller1_all.copy()
simulation_runtime_state.sim_intens1_all = structure_model_state.sim_intens1_all.copy()
simulation_runtime_state.sim_miller2_all = structure_model_state.sim_miller2_all.copy()
simulation_runtime_state.sim_intens2_all = structure_model_state.sim_intens2_all.copy()
simulation_runtime_state.sim_primary_qr_all = dict(structure_model_state.sim_primary_qr_all)

simulation_runtime_state.sim_miller1 = simulation_runtime_state.sim_miller1_all.copy()
simulation_runtime_state.sim_intens1 = simulation_runtime_state.sim_intens1_all.copy()
simulation_runtime_state.sim_miller2 = simulation_runtime_state.sim_miller2_all.copy()
simulation_runtime_state.sim_intens2 = simulation_runtime_state.sim_intens2_all.copy()
simulation_runtime_state.sim_primary_qr = {}

BRAGG_QR_L_KEY_SCALE = gui_controllers.BRAGG_QR_L_KEY_SCALE
BRAGG_QR_L_INVALID_KEY = gui_controllers.BRAGG_QR_L_INVALID_KEY

SF_PRUNE_BIAS_MIN = gui_controllers.SF_PRUNE_BIAS_MIN
SF_PRUNE_BIAS_MAX = gui_controllers.SF_PRUNE_BIAS_MAX

simulation_runtime_state.sf_prune_stats = {
    "qr_total": 0,
    "qr_kept": 0,
    "hkl_primary_total": 0,
    "hkl_primary_kept": 0,
    "hkl_secondary_total": 0,
    "hkl_secondary_kept": 0,
}


def _clip_sf_prune_bias(value) -> float:
    return gui_controllers.clip_structure_factor_prune_bias(
        value,
        fallback=defaults.get("sf_prune_bias", 0.0),
        minimum=SF_PRUNE_BIAS_MIN,
        maximum=SF_PRUNE_BIAS_MAX,
    )


def _current_sf_prune_bias() -> float:
    view_state = globals().get("structure_factor_pruning_controls_view_state")
    var = getattr(view_state, "sf_prune_bias_var", None)
    if var is None:
        return _clip_sf_prune_bias(defaults.get("sf_prune_bias", 0.0))
    try:
        return _clip_sf_prune_bias(var.get())
    except Exception:
        return _clip_sf_prune_bias(defaults.get("sf_prune_bias", 0.0))


def _sf_prune_profile_from_bias(bias: float) -> tuple[float, float, int, int]:
    return gui_controllers.structure_factor_prune_profile_from_bias(bias)


def _update_sf_prune_status_label() -> None:
    view_state = globals().get("structure_factor_pruning_controls_view_state")
    if getattr(view_state, "sf_prune_status_var", None) is None:
        return
    gui_views.set_structure_factor_pruning_status_text(
        view_state,
        gui_controllers.format_structure_factor_pruning_status(
            simulation_runtime_state.sf_prune_stats,
            prune_bias=_current_sf_prune_bias(),
        ),
    )


def _normalize_bragg_qr_source_label(source_label: str | None) -> str:
    return gui_controllers.normalize_bragg_qr_source_label(source_label)


def _l_value_to_key(l_value: float) -> int:
    return gui_controllers.bragg_qr_l_value_to_key(l_value)


def _l_key_to_value(l_key: int) -> float:
    return gui_controllers.bragg_qr_l_key_to_value(l_key)


def _l_keys_from_l_array(l_vals: np.ndarray) -> np.ndarray:
    return gui_controllers.bragg_qr_l_keys_from_l_array(l_vals)


def _source_all_miller_for_label(source_label: str | None) -> np.ndarray:
    label = _normalize_bragg_qr_source_label(source_label)
    if label == "secondary":
        return np.asarray(simulation_runtime_state.sim_miller2_all, dtype=np.float64)
    return np.asarray(simulation_runtime_state.sim_miller1_all, dtype=np.float64)


def _m_indices_from_miller_array(miller_arr: np.ndarray, *, unique: bool = False) -> np.ndarray:
    return gui_controllers.bragg_qr_m_indices_from_miller_array(
        miller_arr,
        unique=unique,
    )


def _apply_bragg_qr_filters(*, trigger_update: bool = True) -> None:
    gui_controllers.apply_bragg_qr_filters(
        simulation_runtime_state,
        bragg_qr_manager_state,
        prune_bias=_current_sf_prune_bias(),
    )
    _update_sf_prune_status_label()

    simulation_runtime_state.last_simulation_signature = None
    simulation_runtime_state.stored_max_positions_local = None
    simulation_runtime_state.stored_sim_image = None
    simulation_runtime_state.stored_peak_table_lattice = None
    simulation_runtime_state.selected_peak_record = None

    refresh_window_fn = globals().get("_refresh_bragg_qr_toggle_window")
    if callable(refresh_window_fn):
        refresh_window_fn()

    if trigger_update:
        schedule_update_fn = globals().get("schedule_update")
        if callable(schedule_update_fn):
            schedule_update_fn()


_apply_bragg_qr_filters(trigger_update=False)

# Build summary and details dataframes using the helper.
df_summary, df_details = build_intensity_dataframes(
    miller, intensities, degeneracy, details
)
structure_model_state.df_summary = df_summary
structure_model_state.df_details = df_details
_sync_structure_model_aliases()

def export_initial_excel():
    """Write the initial intensity tables to Excel when enabled."""

    if not write_excel:
        return

    import pandas as pd

    download_dir = get_dir("downloads")
    excel_path = download_dir / "miller_intensities.xlsx"

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_details.to_excel(writer, sheet_name='Details', index=False)

        workbook  = writer.book
        summary_sheet = writer.sheets['Summary']
        details_sheet = writer.sheets['Details']

        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'vcenter',
            'align': 'center',
            'fg_color': '#4F81BD',
            'font_color': '#FFFFFF',
            'border': 1
        })
        for col_num, col_name in enumerate(df_summary.columns):
            summary_sheet.write(0, col_num, col_name, header_format)
            summary_sheet.set_column(col_num, col_num, 18)

        header_format_details = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'vcenter',
            'align': 'center',
            'fg_color': '#4BACC6',
            'font_color': '#FFFFFF',
            'border': 1
        })
        for col_num, col_name in enumerate(df_details.columns):
            details_sheet.write(0, col_num, col_name, header_format_details)
            details_sheet.set_column(col_num, col_num, 18)

        last_row = len(df_summary) + 1
        summary_sheet.conditional_format(f'D2:D{last_row}', {
            'type': '3_color_scale',
            'min_color': '#FFFFFF',
            'mid_color': '#FFEB84',
            'max_color': '#FF0000'
        })

        zebra_format = workbook.add_format({'bg_color': '#F2F2F2'})
        for row in range(1, len(df_details) + 1):
            if row % 2 == 1:
                details_sheet.set_row(row, cell_format=zebra_format)

    print(f"Excel file saved at {excel_path}")

background_runtime_state.current_background_image = _initial_background_state["current_background_image"]
background_runtime_state.current_background_display = _initial_background_state["current_background_display"]
background_runtime_state.current_background_index = int(_initial_background_state["current_background_index"])
background_runtime_state.visible = True
background_runtime_state.backend_rotation_k = 3
background_runtime_state.backend_flip_x = False
background_runtime_state.backend_flip_y = False
app_shell_view_state = app_state.app_shell_view
status_panel_view_state = app_state.status_panel_view
background_theta_controls_view_state = app_state.background_theta_controls_view
workspace_panels_view_state = app_state.workspace_panels_view
background_backend_debug_view_state = app_state.background_backend_debug_view
background_theta_list_var = None
geometry_theta_offset_var = None
geometry_fit_background_selection_var = None
fit_theta_checkbutton = None


def _sync_background_runtime_state() -> None:
    """Normalize shared background-runtime state after one update path mutates it."""

    background_runtime_state.background_images = list(
        background_runtime_state.background_images
    )
    background_runtime_state.background_images_native = list(
        background_runtime_state.background_images_native
    )
    background_runtime_state.background_images_display = list(
        background_runtime_state.background_images_display
    )
    background_runtime_state.current_background_index = int(background_runtime_state.current_background_index)
    background_runtime_state.visible = bool(background_runtime_state.visible)
    background_runtime_state.backend_rotation_k = int(background_runtime_state.backend_rotation_k)
    background_runtime_state.backend_flip_x = bool(background_runtime_state.backend_flip_x)
    background_runtime_state.backend_flip_y = bool(background_runtime_state.backend_flip_y)
    app_state.file_paths["simulation_background_osc_files"] = list(background_runtime_state.osc_files)


_sync_background_runtime_state()


def _background_theta_default_value() -> float:
    """Return the fallback theta value used when no per-background list exists."""
    return gui_background_theta.background_theta_default_value(
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
    )


def _format_background_theta_values(values: Sequence[object]) -> str:
    """Format per-background theta values for the GUI entry."""
    return gui_background_theta.format_background_theta_values(values)


def _parse_background_theta_values(
    raw_text: object,
    *,
    expected_count: int | None = None,
) -> list[float]:
    """Parse comma/space/semicolon separated background theta values."""
    return gui_background_theta.parse_background_theta_values(
        raw_text,
        expected_count=expected_count,
    )


def _default_geometry_fit_background_selection() -> str:
    """Return the default geometry-fit background selector text."""
    return gui_background_theta.default_geometry_fit_background_selection(
        osc_files=background_runtime_state.osc_files,
    )


def _format_geometry_fit_background_indices(indices: Sequence[object]) -> str:
    """Format a background-index list using 1-based labels for the GUI entry."""
    return gui_background_theta.format_geometry_fit_background_indices(indices)


def _parse_geometry_fit_background_indices(
    raw_text: object,
    *,
    total_count: int,
    current_index: int = 0,
) -> list[int]:
    """Parse geometry-fit background selection text into 0-based indices."""
    return gui_background_theta.parse_geometry_fit_background_indices(
        raw_text,
        total_count=total_count,
        current_index=current_index,
    )


def _current_geometry_fit_background_indices(*, strict: bool = False) -> list[int]:
    """Return the background indices currently selected for geometry fitting."""
    return gui_background_theta.current_geometry_fit_background_indices(
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        strict=strict,
    )


def _geometry_fit_uses_shared_theta_offset(
    selected_indices: Sequence[int] | None = None,
) -> bool:
    """Return whether geometry fitting should use a shared theta offset."""
    return gui_background_theta.geometry_fit_uses_shared_theta_offset(
        selected_indices,
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
    )


def _current_geometry_theta_offset(*, strict: bool = False) -> float:
    """Return the shared theta offset used by multi-background fitting."""
    return gui_background_theta.current_geometry_theta_offset(
        geometry_theta_offset_var=geometry_theta_offset_var,
        strict=strict,
    )


def _current_background_theta_values(*, strict_count: bool = False) -> list[float]:
    """Return one configured theta value per loaded background image."""
    return gui_background_theta.current_background_theta_values(
        osc_files=background_runtime_state.osc_files,
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        strict_count=strict_count,
    )


def _background_theta_for_index(
    index: int,
    *,
    strict_count: bool = False,
) -> float:
    """Return the effective theta used for one background index."""
    return gui_background_theta.background_theta_for_index(
        index,
        osc_files=background_runtime_state.osc_files,
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        current_background_index=background_runtime_state.current_background_index,
        strict_count=strict_count,
    )


def _refresh_geometry_fit_theta_checkbox_label() -> None:
    """Update the theta fit toggle label for single vs multi-background mode."""
    gui_background_theta.refresh_geometry_fit_theta_checkbox_label(
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=geometry_fit_constraints_view_state.controls,
        shared_theta=_geometry_fit_uses_shared_theta_offset(),
    )


def _sync_background_theta_controls(
    *,
    preserve_existing: bool = True,
    trigger_update: bool = False,
) -> None:
    """Keep the theta list entry aligned with the currently loaded backgrounds."""
    gui_background_theta.sync_background_theta_controls(
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=geometry_fit_constraints_view_state.controls,
        set_background_file_status_text=_set_background_file_status_text,
        schedule_update=schedule_update,
        preserve_existing=preserve_existing,
        trigger_update=trigger_update,
    )


def _apply_background_theta_metadata(
    *,
    trigger_update: bool = True,
    sync_live_theta: bool = True,
) -> bool:
    """Validate the theta list/offset entries and optionally refresh the display."""
    return gui_background_theta.apply_background_theta_metadata(
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=geometry_fit_constraints_view_state.controls,
        set_background_file_status_text=_set_background_file_status_text,
        schedule_update=schedule_update,
        progress_label=globals().get("progress_label"),
        trigger_update=trigger_update,
        sync_live_theta=sync_live_theta,
    )


def _apply_geometry_fit_background_selection(
    *,
    trigger_update: bool = False,
    sync_live_theta: bool = True,
) -> bool:
    """Validate the geometry-fit background selection entry."""
    return gui_background_theta.apply_geometry_fit_background_selection(
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=geometry_fit_constraints_view_state.controls,
        set_background_file_status_text=_set_background_file_status_text,
        schedule_update=schedule_update,
        progress_label_geometry=globals().get("progress_label_geometry"),
        trigger_update=trigger_update,
        sync_live_theta=sync_live_theta,
    )


def _sync_geometry_fit_background_selection(*, preserve_existing: bool = True) -> None:
    """Keep the fit-background selector valid when the background list changes."""
    gui_background_theta.sync_geometry_fit_background_selection(
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=geometry_fit_constraints_view_state.controls,
        set_background_file_status_text=_set_background_file_status_text,
        schedule_update=schedule_update,
        progress_label_geometry=globals().get("progress_label_geometry"),
        preserve_existing=preserve_existing,
    )


def _load_background_image_by_index(index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached background arrays for *index*, loading from disk if needed."""


    updated = gui_background.load_background_image_by_index(
        int(index),
        osc_files=background_runtime_state.osc_files,
        background_images=background_runtime_state.background_images,
        background_images_native=background_runtime_state.background_images_native,
        background_images_display=background_runtime_state.background_images_display,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
    )
    background_runtime_state.background_images = list(updated["background_images"])
    background_runtime_state.background_images_native = list(updated["background_images_native"])
    background_runtime_state.background_images_display = list(updated["background_images_display"])
    _sync_background_runtime_state()
    return (
        np.asarray(updated["background_image"]),
        np.asarray(updated["background_display"]),
    )


def _get_current_background_native() -> np.ndarray:
    """Return the unrotated background image corresponding to the current index."""


    updated = gui_background.get_current_background_native(
        osc_files=background_runtime_state.osc_files,
        background_images=background_runtime_state.background_images,
        background_images_native=background_runtime_state.background_images_native,
        background_images_display=background_runtime_state.background_images_display,
        current_background_index=background_runtime_state.current_background_index,
        current_background_image=background_runtime_state.current_background_image,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
    )
    background_runtime_state.background_images = list(updated["background_images"])
    background_runtime_state.background_images_native = list(updated["background_images_native"])
    background_runtime_state.background_images_display = list(updated["background_images_display"])
    _sync_background_runtime_state()
    return np.asarray(updated["background_image"])


def _get_current_background_display() -> np.ndarray:
    """Return the rotated background image used for GUI display."""


    updated = gui_background.get_current_background_display(
        osc_files=background_runtime_state.osc_files,
        background_images=background_runtime_state.background_images,
        background_images_native=background_runtime_state.background_images_native,
        background_images_display=background_runtime_state.background_images_display,
        current_background_index=background_runtime_state.current_background_index,
        current_background_image=background_runtime_state.current_background_image,
        current_background_display=background_runtime_state.current_background_display,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
    )
    background_runtime_state.background_images = list(updated["background_images"])
    background_runtime_state.background_images_native = list(updated["background_images_native"])
    background_runtime_state.background_images_display = list(updated["background_images_display"])
    _sync_background_runtime_state()
    return np.asarray(updated["background_display"])


def _apply_background_backend_orientation(image: np.ndarray | None) -> np.ndarray | None:
    """Apply debug-only backend rotations/flips to the background array."""

    return gui_background.apply_background_backend_orientation(
        image,
        flip_x=background_runtime_state.backend_flip_x,
        flip_y=background_runtime_state.backend_flip_y,
        rotation_k=background_runtime_state.backend_rotation_k,
    )


def _get_current_background_backend() -> np.ndarray | None:
    """Return the background array used for backend comparisons (debug)."""

    return _apply_background_backend_orientation(_get_current_background_native())


def _rotate_point_for_display(col: float, row: float, shape: tuple[int, ...], k: int):
    """Rotate a single (col, row) pair by ``k`` using the same rule as ``np.rot90``.

    The transformation mirrors what ``np.rot90`` does to the underlying image so
    point overlays stay aligned with whichever orientation we render.
    """
    return gui_geometry_overlay.rotate_point_for_display(col, row, shape, k)


def _rotate_measured_peaks_for_display(measured, rotated_shape):
    """Rotate measured-peak coordinates to match the displayed background."""
    return gui_geometry_overlay.rotate_measured_peaks_for_display(
        measured,
        rotated_shape,
        display_rotate_k=DISPLAY_ROTATE_K,
    )


def _unrotate_display_peaks(measured, rotated_shape, *, k=None):
    """Undo a display rotation on peak coordinates.

    Pass ``k`` to match the rotation applied for display (e.g. ``DISPLAY_ROTATE_K``
    or ``SIM_DISPLAY_ROTATE_K``), so the returned points land back in that image's
    native orientation.
    """

    return gui_geometry_overlay.unrotate_display_peaks(
        measured,
        rotated_shape,
        k=k,
        default_display_rotate_k=DISPLAY_ROTATE_K,
    )


def _apply_indexing_mode_to_entries(
    measured,
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
):
    """Swap x/y coordinates when using alternate indexing modes."""
    return gui_geometry_overlay.apply_indexing_mode_to_entries(
        measured,
        shape,
        indexing_mode=indexing_mode,
    )


def _apply_orientation_to_entries(
    measured,
    rotated_shape,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Apply backend-only rotations/flips to measured peak entries."""
    return gui_geometry_overlay.apply_orientation_to_entries(
        measured,
        rotated_shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _orient_image_for_fit(
    image: np.ndarray | None,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Return a rotated/flipped copy of ``image`` for backend fitting only."""
    return gui_geometry_overlay.orient_image_for_fit(
        image,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _native_sim_to_display_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Rotate native simulation coordinates into the displayed frame."""
    return gui_geometry_overlay.native_sim_to_display_coords(
        col,
        row,
        image_shape,
        sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
    )


def _display_to_native_sim_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Map displayed simulation coordinates back into native simulation frame."""
    return gui_geometry_overlay.display_to_native_sim_coords(
        col,
        row,
        image_shape,
        sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
    )


def _transform_points_orientation(
    points: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
) -> list[tuple[float, float]]:
    """Apply flips/rotations to a list of (col, row) points for diagnostics."""
    return gui_geometry_overlay.transform_points_orientation(
        points,
        shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _best_orientation_alignment(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
):
    """Search over 90° rotations and axis flips to minimize RMS distance."""
    return gui_geometry_overlay.best_orientation_alignment(
        sim_coords,
        meas_coords,
        shape,
    )


def _orientation_metrics(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str,
    k: int,
    flip_x: bool,
    flip_y: bool,
    flip_order: str,
):
    """Return RMS/mean/max distance after transforming measured coordinates."""
    return gui_geometry_overlay.orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _select_fit_orientation(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    cfg: dict[str, object] | None = None,
):
    """Choose a measured-peak orientation transform that best aligns to simulation."""
    return gui_geometry_overlay.select_fit_orientation(
        sim_coords,
        meas_coords,
        shape,
        cfg=cfg,
    )


def _aggregate_match_centers(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    sim_millers: list[tuple[int, int, int]],
    meas_millers: list[tuple[int, int, int]],
):
    """Collapse matched peaks by HKL and return centroid pairs."""
    return gui_geometry_overlay.aggregate_match_centers(
        sim_coords,
        meas_coords,
        sim_millers,
        meas_millers,
    )


def _normalize_hkl_key(
    value: object,
) -> tuple[int, int, int] | None:
    """Return a rounded integer HKL tuple when *value* looks like one."""
    from ra_sim.gui.geometry_overlay import normalize_hkl_key

    return normalize_hkl_key(value)


def _aggregate_initial_geometry_display_pairs(
    initial_pairs_display: Sequence[dict[str, object]] | None,
) -> dict[tuple[int, int, int], dict[str, tuple[float, float]]]:
    """Aggregate initial display-frame picks by HKL."""
    return gui_geometry_overlay.aggregate_initial_geometry_display_pairs(
        initial_pairs_display
    )


def _iter_orientation_transform_candidates():
    """Yield all discrete orientation transform candidates."""
    return gui_geometry_overlay.iter_orientation_transform_candidates()


def _inverse_orientation_transform(
    shape: tuple[int, int],
    orientation_choice: dict[str, object] | None,
) -> dict[str, object]:
    """Return the inverse transform for an orientation-choice dict."""
    return gui_geometry_overlay.inverse_orientation_transform(
        shape,
        orientation_choice,
    )


def _draw_geometry_fit_overlay(
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw one fixed-background/fitted-simulation overlay record per match."""

    gui_overlays.draw_geometry_fit_overlay(
        ax,
        overlay_records,
        geometry_pick_artists=geometry_runtime_state.pick_artists,
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        draw_idle=canvas.draw_idle,
        max_display_markers=max_display_markers,
        show_caked_2d=bool(analysis_view_controls_view_state.show_caked_2d_var.get()),
        native_detector_coords_to_caked_display_coords=(
            _native_detector_coords_to_caked_display_coords
        ),
    )


def _draw_initial_geometry_pairs_overlay(
    initial_pairs_display: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw only the initially selected simulation/background peak pairs."""

    gui_overlays.draw_initial_geometry_pairs_overlay(
        ax,
        initial_pairs_display,
        geometry_pick_artists=geometry_runtime_state.pick_artists,
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        draw_idle=canvas.draw_idle,
        max_display_markers=max_display_markers,
    )


def _build_geometry_manual_initial_pairs_display(
    background_index: int,
    *,
    param_set: dict[str, object] | None = None,
    prefer_cache: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build overlay-ready manual geometry pairs for one background image."""
    return gui_manual_geometry.build_geometry_manual_initial_pairs_display(
        background_index,
        param_set=param_set,
        current_background_index=background_runtime_state.current_background_index,
        prefer_cache=prefer_cache,
        pairs_for_index=_geometry_manual_pairs_for_index,
        current_geometry_fit_params=_current_geometry_fit_params,
        get_cache_data=_get_geometry_manual_pick_cache,
        simulated_peaks_for_params=_geometry_manual_simulated_peaks_for_params,
        build_simulated_lookup=_geometry_manual_simulated_lookup,
        entry_display_coords=_geometry_manual_entry_display_coords,
    )


def _render_current_geometry_manual_pairs(*, update_status: bool = False) -> bool:
    """Redraw the saved manual geometry-pair overlay for the current background."""
    return gui_manual_geometry.render_current_geometry_manual_pairs(
        background_visible=background_runtime_state.visible,
        current_background_index=int(background_runtime_state.current_background_index),
        current_background_image=_current_geometry_manual_pick_background_image(),
        pick_session=geometry_manual_state.pick_session,
        build_initial_pairs_display=_build_geometry_manual_initial_pairs_display,
        session_initial_pairs_display=_geometry_manual_session_initial_pairs_display,
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        draw_initial_geometry_pairs_overlay=_draw_initial_geometry_pairs_overlay,
        update_button_label_fn=_update_geometry_manual_pick_button_label,
        set_background_file_status_text_fn=_set_background_file_status_text,
        pair_group_count=_geometry_manual_pair_group_count,
        set_status_text=lambda text: progress_label_geometry.config(text=text),
        update_status=update_status,
    )


def _toggle_geometry_manual_selection_at(col: float, row: float) -> bool:
    """Select one manual Qr/Qz group and arm background-point placement."""


    def _set_pick_session(session: dict[str, object]) -> None:
        _set_geometry_manual_pick_session(session)

    handled, next_session, suppress_drag = gui_manual_geometry.geometry_manual_toggle_selection_at(
        float(col),
        float(row),
        pick_session=geometry_manual_state.pick_session,
        current_background_index=int(background_runtime_state.current_background_index),
        display_background=_current_geometry_manual_pick_background_image(),
        get_cache_data=lambda **kwargs: _get_geometry_manual_pick_cache(
            param_set=_current_geometry_fit_params(),
            prefer_cache=True,
            **kwargs,
        ),
        pairs_for_index=_geometry_manual_pairs_for_index,
        set_pairs_for_index_fn=_set_geometry_manual_pairs_for_index,
        set_pick_session_fn=_set_pick_session,
        restore_view_fn=_restore_geometry_manual_pick_view,
        clear_preview_artists_fn=_clear_geometry_manual_preview_artists,
        render_current_pairs_fn=_render_current_geometry_manual_pairs,
        update_button_label_fn=_update_geometry_manual_pick_button_label,
        set_status_text=lambda text: progress_label_geometry.config(text=text),
        push_undo_state_fn=_push_geometry_manual_undo_state,
        listed_q_group_entries=_listed_geometry_q_group_entries,
        format_q_group_line=_format_geometry_q_group_line,
        use_caked_space=_geometry_manual_pick_uses_caked_space(),
        pick_search_window_px=float(GEOMETRY_MANUAL_PICK_SEARCH_WINDOW_PX),
        caked_search_tth_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_TTH_DEG),
        caked_search_phi_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_PHI_DEG),
    )
    _set_geometry_manual_pick_session(next_session)
    peak_selection_state.suppress_drag_press_once = bool(suppress_drag)
    _sync_peak_selection_state()
    return bool(handled)


def _place_geometry_manual_selection_at(col: float, row: float) -> bool:
    """Record the next manual background point for the active Qr/Qz pick session."""

    def _set_pick_session(session: dict[str, object]) -> None:
        _set_geometry_manual_pick_session(session)

    handled, next_session = gui_manual_geometry.geometry_manual_place_selection_at(
        float(col),
        float(row),
        pick_session=geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        display_background=_current_geometry_manual_pick_background_image(),
        get_cache_data=lambda **kwargs: _get_geometry_manual_pick_cache(
            param_set=_current_geometry_fit_params(),
            prefer_cache=True,
            **kwargs,
        ),
        refine_preview_point=_geometry_manual_refine_preview_point,
        set_pairs_for_index_fn=_set_geometry_manual_pairs_for_index,
        set_pick_session_fn=_set_pick_session,
        clear_preview_artists_fn=_clear_geometry_manual_preview_artists,
        restore_view_fn=_restore_geometry_manual_pick_view,
        render_current_pairs_fn=_render_current_geometry_manual_pairs,
        update_button_label_fn=_update_geometry_manual_pick_button_label,
        set_status_text=lambda text: progress_label_geometry.config(text=text),
        push_undo_state_fn=_push_geometry_manual_undo_state,
        use_caked_space=_geometry_manual_pick_uses_caked_space(),
        caked_angles_to_background_display_coords=_caked_angles_to_background_display_coords,
    )
    _set_geometry_manual_pick_session(next_session)
    return bool(handled)


def _geometry_overlay_frame_diagnostics(
    overlay_records: Sequence[dict[str, object]] | None,
) -> tuple[dict[str, float], str]:
    """Compare per-match fitted overlay alignment in display space."""

    return compute_geometry_overlay_frame_diagnostics(overlay_records)

# Measured peaks are collected interactively in the current GUI workflow.
# Keep this list for compatibility, but avoid loading a large file at startup.
measured_peaks = []

###############################################################################
#                                  TK SETUP
###############################################################################
root = gui_views.create_root_window("RA-SIM Simulation")
root.minsize(1200, 760)
gui_views.create_app_shell(root=root, view_state=app_shell_view_state)
if (
    app_shell_view_state.workspace_body is None
    or app_shell_view_state.fit_body is None
    or app_shell_view_state.analysis_views_frame is None
    or app_shell_view_state.analysis_exports_frame is None
    or app_shell_view_state.status_frame is None
    or app_shell_view_state.fig_frame is None
    or app_shell_view_state.canvas_frame is None
    or app_shell_view_state.left_col is None
    or app_shell_view_state.right_col is None
    or app_shell_view_state.plot_frame_1d is None
):
    raise RuntimeError("Top-level GUI shell was not created.")

gui_views.create_workspace_panels(
    parent=app_shell_view_state.workspace_body,
    view_state=workspace_panels_view_state,
)
if (
    workspace_panels_view_state.workspace_actions_frame is None
    or workspace_panels_view_state.workspace_backgrounds_frame is None
    or workspace_panels_view_state.workspace_session_frame is None
):
    raise RuntimeError("Workspace panels were not created.")


def _shutdown_gui():
    """Close all application windows and end the Tk event loop."""

    try:
        if root.winfo_exists():
            root.destroy()
    except tk.TclError:
        # Window is already gone or cannot be destroyed cleanly; proceed to
        # shut down the rest of the application.
        pass

    try:
        root.quit()
    except tk.TclError:
        pass


root.protocol("WM_DELETE_WINDOW", _shutdown_gui)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("auto")
canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
    fig,
    master=app_shell_view_state.canvas_frame,
)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

global_image_buffer = np.zeros((image_size, image_size), dtype=np.float64)
simulation_runtime_state.unscaled_image = None

# ── replace the original imshow call ────────────────────────────
image_display = ax.imshow(
    global_image_buffer,
    cmap=turbo_white0,
    alpha=0.5,
    zorder=1,
    origin='upper'
)


background_display = ax.imshow(
    background_runtime_state.current_background_display,
    cmap='turbo',
    zorder=0,
    origin='upper'
)

highlight_cmap = ListedColormap(
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0, 0.35),
    ]
)
integration_region_overlay = ax.imshow(
    np.zeros_like(global_image_buffer),
    cmap=highlight_cmap,
    vmin=0,
    vmax=1,
    origin='upper',
    zorder=4,
    interpolation='nearest'
)
integration_region_overlay.set_visible(False)

integration_region_rect = Rectangle(
    (0.0, 0.0),
    0.0,
    0.0,
    linewidth=2.0,
    edgecolor='cyan',
    facecolor='none',
    linestyle='--',
    zorder=5,
)
integration_region_rect.set_visible(False)
ax.add_patch(integration_region_rect)
# ---------------------------------------------------------------------------
#  helper – returns a fully populated, *consistent* mosaic_params dict
# ---------------------------------------------------------------------------
def build_mosaic_params():
    return {
        "beam_x_array":       simulation_runtime_state.profile_cache["beam_x_array"],
        "beam_y_array":       simulation_runtime_state.profile_cache["beam_y_array"],
        "theta_array":        simulation_runtime_state.profile_cache["theta_array"],
        "phi_array":          simulation_runtime_state.profile_cache["phi_array"],
        "wavelength_array":   simulation_runtime_state.profile_cache["wavelength_array"],   #  <<< name fixed
        "sigma_mosaic_deg":   sigma_mosaic_var.get(),
        "gamma_mosaic_deg":   gamma_mosaic_var.get(),
        "eta":                eta_var.get(),
        "solve_q_steps":      _current_solve_q_steps(),
        "solve_q_rel_tol":    _current_solve_q_rel_tol(),
        "solve_q_mode":       _current_solve_q_mode_flag(),
    }


def _normalize_optics_mode_label(value) -> str:
    """Normalize optics mode to UI labels: ``'fast'`` or ``'exact'``."""

    if value is None:
        return "fast"
    if isinstance(value, (int, np.integer)):
        return "exact" if int(value) == OPTICS_MODE_EXACT else "fast"
    if isinstance(value, (float, np.floating)):
        return "exact" if int(round(float(value))) == OPTICS_MODE_EXACT else "fast"

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
        return "exact"
    if text in {
        "0",
        "false",
        "no",
        "off",
        "fast",
        "approx",
        "fresnel_ctr_damping",
        "fresnel-weighted kinematic ctr absorption correction",
        "uncoupled fresnel + ctr damping (ufd)",
        "fast dwba-lite (fresnel + depth-sum attenuation)",
        "ufd",
        "dwba-lite",
    }:
        return "fast"

    if "complex-k dwba" in text or "complex_k_dwba" in text:
        return "exact"
    if "fresnel" in text and "ctr" in text:
        return "fast"
    return "fast"


def _current_optics_mode_flag() -> int:
    mode_var = globals().get("optics_mode_var")
    mode_label = _normalize_optics_mode_label(mode_var.get() if mode_var is not None else "fast")
    if mode_label == "exact":
        return OPTICS_MODE_EXACT
    return OPTICS_MODE_FAST

colorbar_main = fig.colorbar(image_display, ax=ax, label='Intensity', shrink=0.6, pad=0.02)

# Additional colorbar axis for caked 2D (not used in basic 1D, but reserved)
caked_cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
caked_cbar_ax.set_visible(False)
caked_colorbar = fig.colorbar(image_display, cax=caked_cbar_ax)
caked_colorbar.set_label('Intensity (binned)')
caked_colorbar.ax.set_visible(False)

center_marker, = ax.plot(
    center_default[1],
    center_default[0],
    'ro',
    markersize=5,
    zorder=2
)

ax.set_xlim(0, image_size)
ax.set_ylim(image_size, 0)
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
ax.set_title('Simulated Diffraction Pattern')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
canvas.draw()

# -----------------------------------------------------------
# 1)  Highlight‑marker that we can move each click
# -----------------------------------------------------------
selected_peak_marker, = ax.plot([], [], 'ys',  # yellow square outline
                               markersize=8, markerfacecolor='none',
                               linewidth=1.5, zorder=6)
selected_peak_marker.set_visible(False)

# Geometry click markers (sim vs real)
geometry_runtime_state.pick_artists = []
geometry_runtime_state.preview_artists = []
geometry_runtime_state.manual_preview_artists = []
geometry_runtime_state.qr_cylinder_overlay_artists = []
geometry_runtime_state.qr_cylinder_overlay_cache = {
    "signature": None,
    "paths": [],
}
geometry_preview_state = app_state.geometry_preview
geometry_q_group_view_state = app_state.geometry_q_group_view
geometry_q_group_state = app_state.geometry_q_groups
geometry_manual_state = app_state.manual_geometry
geometry_runtime_state.manual_pick_armed = False
geometry_runtime_state.manual_pick_cache_signature = None
geometry_runtime_state.manual_pick_cache_data = {}
geometry_fit_history_state = app_state.geometry_fit_history
geometry_fit_parameter_controls_view_state = (
    app_state.geometry_fit_parameter_controls_view
)
geometry_fit_constraints_view_state = app_state.geometry_fit_constraints_view
geometry_tool_actions_view_state = app_state.geometry_tool_actions_view
GEOMETRY_MANUAL_UNDO_LIMIT = 100
GEOMETRY_FIT_UNDO_LIMIT = 16
GEOMETRY_PREVIEW_TOGGLE_MAX_DISTANCE_PX = 14.0
GEOMETRY_MANUAL_PICK_SEARCH_WINDOW_PX = 50.0
GEOMETRY_MANUAL_PICK_ZOOM_WINDOW_PX = 100.0
GEOMETRY_MANUAL_CAKED_SEARCH_TTH_DEG = 1.5
GEOMETRY_MANUAL_CAKED_SEARCH_PHI_DEG = 10.0
GEOMETRY_MANUAL_CAKED_ZOOM_TTH_DEG = 4.0
GEOMETRY_MANUAL_CAKED_ZOOM_PHI_DEG = 24.0
GEOMETRY_MANUAL_PREVIEW_MIN_INTERVAL_S = 0.03
GEOMETRY_MANUAL_PREVIEW_MIN_MOVE_PX = 0.8
GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX = 0.75


def _geometry_manual_position_error_px(
    raw_col: float,
    raw_row: float,
    refined_col: float,
    refined_row: float,
) -> float:
    """Return the click-to-refined placement error in display pixels."""
    return gui_manual_geometry.geometry_manual_position_error_px(
        raw_col,
        raw_row,
        refined_col,
        refined_row,
    )


def _geometry_manual_position_sigma_px(
    placement_error_px: object,
    *,
    floor_px: float | None = None,
) -> float:
    """Convert a manual click-placement error into a fit sigma in pixels."""
    if floor_px is None:
        floor_px = float(globals().get("GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX", 0.75))
    return gui_manual_geometry.geometry_manual_position_sigma_px(
        placement_error_px,
        floor_px=float(floor_px),
    )


def _normalize_geometry_manual_pair_entry(
    entry: dict[str, object] | None,
) -> dict[str, object] | None:
    """Normalize one saved manual geometry-pair entry."""
    return gui_manual_geometry.normalize_geometry_manual_pair_entry(
        entry,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _geometry_manual_pairs_for_index(index: int) -> list[dict[str, object]]:
    """Return normalized saved manual geometry pairs for one background index."""
    return gui_manual_geometry.geometry_manual_pairs_for_index(
        index,
        pairs_by_background=geometry_manual_state.pairs_by_background,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _set_geometry_manual_pairs_for_index(
    index: int,
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Replace one background's saved manual geometry-pair list."""
    return gui_manual_geometry.set_geometry_manual_pairs_for_index(
        index,
        entries,
        pairs_by_background=geometry_manual_state.pairs_by_background,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _geometry_manual_pair_group_count(index: int) -> int:
    """Return how many distinct Qr/Qz groups are saved for one background."""
    return gui_manual_geometry.geometry_manual_pair_group_count(
        index,
        pairs_by_background=geometry_manual_state.pairs_by_background,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _replace_geometry_manual_pairs_by_background(
    pairs_by_background: dict[int, list[dict[str, object]]] | None,
) -> dict[int, list[dict[str, object]]]:
    """Replace the stored manual-geometry pair map in the shared state container."""
    return gui_controllers.replace_manual_geometry_pairs_by_background(
        geometry_manual_state,
        pairs_by_background,
    )


def _set_geometry_manual_pick_session(
    pick_session: dict[str, object] | None,
) -> dict[str, object]:
    """Replace the active manual-geometry pick session in the shared state container."""
    return gui_controllers.replace_manual_geometry_pick_session(
        geometry_manual_state,
        pick_session,
    )


def _clear_geometry_manual_undo_stack() -> None:
    """Discard the manual-placement undo history."""
    gui_controllers.clear_manual_geometry_undo_stack(geometry_manual_state)


def _geometry_manual_undo_snapshot() -> gui_state.ManualGeometryUndoSnapshot:
    """Return a deep copy of the manual-placement state for undo."""
    return gui_controllers.build_manual_geometry_undo_snapshot(geometry_manual_state)


def _push_geometry_manual_undo_state() -> None:
    """Push the current manual-placement state onto the undo stack."""
    gui_controllers.push_manual_geometry_undo_state(
        geometry_manual_state,
        limit=int(GEOMETRY_MANUAL_UNDO_LIMIT),
    )


def _undo_last_geometry_manual_placement() -> None:
    """Restore the most recent manual-placement state."""

    if not geometry_manual_state.undo_stack:
        progress_label_geometry.config(text="No manual placement changes are available to undo.")
        return

    gui_controllers.restore_last_manual_geometry_undo_state(geometry_manual_state)
    _clear_geometry_manual_preview_artists(redraw=False)
    _render_current_geometry_manual_pairs(update_status=True)
    _update_geometry_manual_pick_button_label()
    _set_background_file_status_text()
    progress_label_geometry.config(text="Undid the last manual placement change.")


def _copy_geometry_fit_state_value(value):
    """Deep-copy simple geometry-fit GUI state."""

    return gui_geometry_fit.copy_geometry_fit_state_value(value)


def _geometry_fit_last_overlay_state() -> dict[str, object] | None:
    """Return the remembered geometry-fit overlay state."""

    return geometry_fit_history_state.last_overlay_state


def _set_geometry_fit_last_overlay_state(
    overlay_state: dict[str, object] | None,
) -> dict[str, object] | None:
    """Replace the remembered geometry-fit overlay state."""

    return gui_controllers.replace_geometry_fit_last_overlay_state(
        geometry_fit_history_state,
        overlay_state,
        copy_state_value=_copy_geometry_fit_state_value,
    )


def _current_geometry_fit_ui_params() -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    theta_offset = None
    if geometry_theta_offset_var is not None:
        theta_offset = float(_current_geometry_theta_offset(strict=False))
    return gui_geometry_fit.current_geometry_fit_ui_params(
        zb=float(zb_var.get()),
        zs=float(zs_var.get()),
        theta_initial=float(theta_initial_var.get()),
        psi_z=float(psi_z_var.get()),
        chi=float(chi_var.get()),
        cor_angle=float(cor_angle_var.get()),
        gamma=float(gamma_var.get()),
        Gamma=float(Gamma_var.get()),
        corto_detector=float(corto_detector_var.get()),
        a=float(a_var.get()),
        c=float(c_var.get()),
        center_x=float(center_x_var.get()),
        center_y=float(center_y_var.get()),
        theta_offset=theta_offset,
    )


def _update_geometry_fit_undo_button_state() -> None:
    """Enable fit history buttons only when the relevant history exists."""

    gui_views.set_geometry_fit_history_button_state(
        geometry_tool_actions_view_state,
        can_undo=bool(geometry_fit_history_state.undo_stack),
        can_redo=bool(geometry_fit_history_state.redo_stack),
    )


def _clear_geometry_fit_undo_stack() -> None:
    """Discard the geometry-fit undo/redo history."""

    gui_controllers.clear_geometry_fit_history(geometry_fit_history_state)
    _update_geometry_fit_undo_button_state()


def _capture_geometry_fit_undo_state() -> dict[str, object]:
    """Capture the current geometry-fit state for undo."""

    overlay_state = _copy_geometry_fit_state_value(_geometry_fit_last_overlay_state())
    if not (
        isinstance(overlay_state, dict)
        and (
            overlay_state.get("overlay_records")
            or overlay_state.get("initial_pairs_display")
        )
    ):
        try:
            _, initial_pairs_display = _build_geometry_manual_initial_pairs_display(
                int(background_runtime_state.current_background_index),
                param_set=_current_geometry_fit_params(),
                prefer_cache=True,
            )
            pending_pairs_display = _geometry_manual_session_initial_pairs_display()
            combined_pairs_display = list(initial_pairs_display) + list(pending_pairs_display)
            if combined_pairs_display:
                overlay_state = {
                    "overlay_records": [],
                    "initial_pairs_display": _copy_geometry_fit_state_value(
                        combined_pairs_display
                    ),
                    "max_display_markers": max(1, len(combined_pairs_display)),
                }
        except Exception:
            pass

    return {
        "ui_params": _copy_geometry_fit_state_value(_current_geometry_fit_ui_params()),
        "profile_cache": _copy_geometry_fit_state_value(simulation_runtime_state.profile_cache),
        "overlay_state": overlay_state,
    }


def _push_geometry_fit_undo_state(state: dict[str, object] | None) -> None:
    """Push one pre-fit state onto the geometry-fit undo stack."""
    gui_controllers.push_geometry_fit_undo_state(
        geometry_fit_history_state,
        state,
        copy_state_value=_copy_geometry_fit_state_value,
        limit=int(GEOMETRY_FIT_UNDO_LIMIT),
    )
    _update_geometry_fit_undo_button_state()


def _push_geometry_fit_redo_state(state: dict[str, object] | None) -> None:
    """Push one state onto the geometry-fit redo stack."""
    gui_controllers.push_geometry_fit_redo_state(
        geometry_fit_history_state,
        state,
        copy_state_value=_copy_geometry_fit_state_value,
        limit=int(GEOMETRY_FIT_UNDO_LIMIT),
    )
    _update_geometry_fit_undo_button_state()


def _restore_geometry_fit_undo_state(state: dict[str, object]) -> None:
    """Restore a previously captured geometry-fit state."""


    if not isinstance(state, dict):
        return

    restored = gui_geometry_fit.apply_geometry_fit_undo_state(
        state,
        var_map={
            "zb": zb_var,
            "zs": zs_var,
            "theta_initial": theta_initial_var,
            "psi_z": psi_z_var,
            "chi": chi_var,
            "cor_angle": cor_angle_var,
            "gamma": gamma_var,
            "Gamma": Gamma_var,
            "corto_detector": corto_detector_var,
            "a": a_var,
            "c": c_var,
            "center_x": center_x_var,
            "center_y": center_y_var,
        },
        geometry_theta_offset_var=geometry_theta_offset_var,
    )
    simulation_runtime_state.profile_cache = restored["profile_cache"]
    _set_geometry_fit_last_overlay_state(restored["overlay_state"])
    simulation_runtime_state.last_simulation_signature = None

    if simulation_runtime_state.update_pending is not None:
        try:
            root.after_cancel(simulation_runtime_state.update_pending)
        except Exception:
            pass
        simulation_runtime_state.update_pending = None

    do_update()

    overlay_state = _geometry_fit_last_overlay_state() or {}
    overlay_records = overlay_state.get("overlay_records", []) or []
    initial_pairs_display = overlay_state.get("initial_pairs_display", []) or []
    max_display_markers = int(overlay_state.get("max_display_markers", 120))
    max_display_markers = max(1, max_display_markers)
    if overlay_records:
        _draw_geometry_fit_overlay(
            overlay_records,
            max_display_markers=max_display_markers,
        )
    elif initial_pairs_display:
        _draw_initial_geometry_pairs_overlay(
            initial_pairs_display,
            max_display_markers=max_display_markers,
        )

    _set_background_file_status_text()
    _update_geometry_manual_pick_button_label()


def _undo_last_geometry_fit() -> None:
    """Restore the most recent geometry-fit state."""

    if not geometry_fit_history_state.undo_stack:
        progress_label_geometry.config(text="No geometry fit history available to undo.")
        return

    current_state = _capture_geometry_fit_undo_state()
    state = gui_controllers.peek_last_geometry_fit_undo_state(
        geometry_fit_history_state,
        copy_state_value=_copy_geometry_fit_state_value,
    )
    if not isinstance(state, dict):
        progress_label_geometry.config(text="No geometry fit history available to undo.")
        return
    try:
        _restore_geometry_fit_undo_state(state)
    except Exception as exc:
        progress_label_geometry.config(text=f"Failed to undo geometry fit: {exc}")
        return

    gui_controllers.commit_geometry_fit_undo(
        geometry_fit_history_state,
        current_state,
        copy_state_value=_copy_geometry_fit_state_value,
        limit=int(GEOMETRY_FIT_UNDO_LIMIT),
    )
    _update_geometry_fit_undo_button_state()
    progress_label_geometry.config(text="Restored the previous geometry-fit state.")


def _redo_last_geometry_fit() -> None:
    """Reapply the most recently undone geometry-fit state."""

    if not geometry_fit_history_state.redo_stack:
        progress_label_geometry.config(text="No geometry fit history available to redo.")
        return

    current_state = _capture_geometry_fit_undo_state()
    state = gui_controllers.peek_last_geometry_fit_redo_state(
        geometry_fit_history_state,
        copy_state_value=_copy_geometry_fit_state_value,
    )
    if not isinstance(state, dict):
        progress_label_geometry.config(text="No geometry fit history available to redo.")
        return
    try:
        _restore_geometry_fit_undo_state(state)
    except Exception as exc:
        progress_label_geometry.config(text=f"Failed to redo geometry fit: {exc}")
        return

    gui_controllers.commit_geometry_fit_redo(
        geometry_fit_history_state,
        current_state,
        copy_state_value=_copy_geometry_fit_state_value,
        limit=int(GEOMETRY_FIT_UNDO_LIMIT),
    )
    _update_geometry_fit_undo_button_state()
    progress_label_geometry.config(text="Reapplied the next geometry-fit state.")


def _geometry_manual_pair_entry_to_jsonable(
    entry: dict[str, object] | None,
) -> dict[str, object] | None:
    """Convert one saved manual pair entry into a JSON-safe dictionary."""
    return gui_manual_geometry.geometry_manual_pair_entry_to_jsonable(
        entry,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _geometry_manual_pair_entry_from_jsonable(
    row: dict[str, object] | None,
) -> dict[str, object] | None:
    """Rebuild one saved manual pair entry from imported JSON data."""
    return gui_manual_geometry.geometry_manual_pair_entry_from_jsonable(
        row,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _normalized_background_path_for_compare(raw_path: object) -> str | None:
    """Return a normalized path string suitable for background matching."""

    try:
        candidate = Path(str(raw_path)).expanduser()
    except Exception:
        return None
    if not str(candidate):
        return None
    return os.path.normcase(os.path.normpath(str(candidate)))


def _geometry_manual_pairs_export_rows() -> list[dict[str, object]]:
    """Return the saved manual geometry pairs as JSON-safe background rows."""

    background_indices: set[int] = set()
    for raw_idx in geometry_manual_state.pairs_by_background.keys():
        try:
            background_indices.add(int(raw_idx))
        except Exception:
            continue
    rows: list[dict[str, object]] = []
    for background_idx in sorted(background_indices):
        entries = [
            serialized
            for serialized in (
                _geometry_manual_pair_entry_to_jsonable(entry)
                for entry in _geometry_manual_pairs_for_index(background_idx)
            )
            if serialized is not None
        ]
        if not entries:
            continue
        background_path = (
            str(Path(str(background_runtime_state.osc_files[background_idx])).expanduser())
            if 0 <= int(background_idx) < len(background_runtime_state.osc_files)
            else None
        )
        background_name = (
            Path(str(background_runtime_state.osc_files[background_idx])).name
            if 0 <= int(background_idx) < len(background_runtime_state.osc_files)
            else f"background_{int(background_idx) + 1}"
        )
        rows.append(
            {
                "background_index": int(background_idx),
                "background_path": background_path,
                "background_name": background_name,
                "entries": entries,
            }
        )
    return rows


def _collect_geometry_manual_pairs_snapshot() -> dict[str, object]:
    """Return a portable snapshot of all saved manual geometry placements."""

    return {
        "background_files": [str(Path(str(path)).expanduser()) for path in background_runtime_state.osc_files],
        "current_background_index": int(background_runtime_state.current_background_index),
        "manual_pairs": _geometry_manual_pairs_export_rows(),
    }


def _apply_geometry_manual_pairs_rows(
    rows: Sequence[object] | None,
    *,
    replace_existing: bool = True,
) -> tuple[int, int, list[str]]:
    """Import saved manual geometry pairs onto the currently loaded backgrounds."""

    exact_path_lookup: dict[str, int] = {}
    name_lookup: defaultdict[str, list[int]] = defaultdict(list)
    for idx, raw_path in enumerate(background_runtime_state.osc_files):
        normalized_path = _normalized_background_path_for_compare(raw_path)
        if normalized_path is not None:
            exact_path_lookup[normalized_path] = int(idx)
        name_lookup[Path(str(raw_path)).name].append(int(idx))

    imported_map: dict[int, list[dict[str, object]]]
    if replace_existing:
        imported_map = {}
    else:
        imported_map = {
            int(idx): _geometry_manual_pairs_for_index(idx)
            for idx in range(len(background_runtime_state.osc_files))
            if _geometry_manual_pairs_for_index(idx)
        }

    warnings: list[str] = []
    matched_backgrounds: set[int] = set()
    pair_count = 0
    for raw_row in rows or []:
        if not isinstance(raw_row, dict):
            continue

        target_index = None
        normalized_path = _normalized_background_path_for_compare(raw_row.get("background_path"))
        if normalized_path is not None:
            target_index = exact_path_lookup.get(normalized_path)
        if target_index is None:
            background_name = raw_row.get("background_name")
            if background_name is not None:
                matches = name_lookup.get(Path(str(background_name)).name, [])
                if len(matches) == 1:
                    target_index = int(matches[0])
        if target_index is None:
            try:
                fallback_index = int(raw_row.get("background_index"))
            except Exception:
                fallback_index = None
            if fallback_index is not None and 0 <= fallback_index < len(background_runtime_state.osc_files):
                target_index = int(fallback_index)

        if target_index is None:
            warnings.append(
                f"Skipped placements for '{raw_row.get('background_name', 'unknown background')}'."
            )
            continue

        imported_entries = [
            restored
            for restored in (
                _geometry_manual_pair_entry_from_jsonable(entry)
                for entry in raw_row.get("entries", [])
            )
            if restored is not None
        ]
        imported_map[int(target_index)] = imported_entries
        if imported_entries:
            matched_backgrounds.add(int(target_index))
            pair_count += len(imported_entries)

    _replace_geometry_manual_pairs_by_background({
        int(idx): list(entries)
        for idx, entries in imported_map.items()
        if entries
    })
    _clear_geometry_manual_preview_artists(redraw=False)
    _cancel_geometry_manual_pick_session(restore_view=True, redraw=False)
    _invalidate_geometry_manual_pick_cache()
    _clear_geometry_manual_undo_stack()
    _clear_geometry_fit_undo_stack()
    _render_current_geometry_manual_pairs(update_status=False)
    _update_geometry_manual_pick_button_label()
    _set_background_file_status_text()
    return int(len(matched_backgrounds)), int(pair_count), warnings


def _apply_geometry_manual_pairs_snapshot(
    snapshot: dict[str, object],
    *,
    allow_background_reload: bool = True,
) -> str:
    """Restore saved manual geometry placements from a snapshot dictionary."""

    warnings: list[str] = []

    if allow_background_reload:
        raw_background_paths = snapshot.get("background_files", [])
        background_paths: list[str] = []
        if isinstance(raw_background_paths, list):
            for raw_path in raw_background_paths:
                if raw_path is None:
                    continue
                background_paths.append(str(Path(str(raw_path)).expanduser()))
        if background_paths:
            saved_paths_norm = [
                path_norm
                for path_norm in (
                    _normalized_background_path_for_compare(path) for path in background_paths
                )
                if path_norm is not None
            ]
            current_paths_norm = [
                path_norm
                for path_norm in (
                    _normalized_background_path_for_compare(path) for path in background_runtime_state.osc_files
                )
                if path_norm is not None
            ]
            if saved_paths_norm != current_paths_norm:
                missing_paths = [
                    path for path in background_paths if not Path(path).is_file()
                ]
                if not missing_paths:
                    try:
                        _load_background_files(
                            background_paths,
                            select_index=int(snapshot.get("current_background_index", 0)),
                        )
                    except Exception as exc:
                        warnings.append(f"background reload: {exc}")
                else:
                    warnings.append(
                        "saved background files are missing; placements were mapped onto the "
                        "currently loaded backgrounds where possible"
                    )

    imported_backgrounds, imported_pairs, import_warnings = _apply_geometry_manual_pairs_rows(
        snapshot.get("manual_pairs", []),
        replace_existing=True,
    )
    warnings.extend(import_warnings)
    schedule_update()

    message = (
        f"Imported {imported_pairs} manual placement(s) across {imported_backgrounds} background(s)."
    )
    if warnings:
        message += " Warnings: " + "; ".join(warnings[:4])
        if len(warnings) > 4:
            message += f"; +{len(warnings) - 4} more"
    return message


def _invalidate_geometry_manual_pick_cache() -> None:
    """Drop cached manual-pick simulation/background state."""


    geometry_runtime_state.manual_pick_cache_signature = None
    geometry_runtime_state.manual_pick_cache_data = {}


def _current_geometry_manual_match_config() -> dict[str, object]:
    """Return the refined background-peak matcher config for manual picking."""
    return gui_manual_geometry.current_geometry_manual_match_config(fit_config)


def _geometry_manual_pick_cache_signature(
    *,
    background_index: int | None = None,
    background_image: object | None = None,
) -> tuple[object, ...]:
    """Return a cache signature for reusable manual-pick state."""
    return gui_manual_geometry.geometry_manual_pick_cache_signature(
        last_simulation_signature=simulation_runtime_state.last_simulation_signature,
        background_index=int(background_runtime_state.current_background_index if background_index is None else background_index),
        background_image=(
            _current_geometry_manual_pick_background_image()
            if background_image is None
            else background_image
        ),
        use_caked_space=bool(_geometry_manual_pick_uses_caked_space()),
        geometry_preview_excluded_q_groups=geometry_preview_state.excluded_q_groups,
        geometry_q_group_cached_entries=geometry_q_group_state.cached_entries,
        stored_max_positions_local=simulation_runtime_state.stored_max_positions_local,
        stored_peak_table_lattice=simulation_runtime_state.stored_peak_table_lattice,
    )


def _get_geometry_manual_pick_cache(
    *,
    param_set: dict[str, object] | None = None,
    prefer_cache: bool = True,
    background_index: int | None = None,
    background_image: object | None = None,
) -> dict[str, object]:
    """Build or reuse the current manual-pick simulation/background cache."""


    bg_index = int(background_runtime_state.current_background_index if background_index is None else background_index)
    background_local = (
        _current_geometry_manual_pick_background_image() if background_image is None else background_image
    )
    cache_data, geometry_runtime_state.manual_pick_cache_signature, geometry_manual_pick_cache_data = (
        gui_manual_geometry.build_geometry_manual_pick_cache(
            param_set=param_set,
            prefer_cache=prefer_cache,
            background_index=bg_index,
            current_background_index=int(background_runtime_state.current_background_index),
            background_image=background_local,
            existing_cache_signature=geometry_runtime_state.manual_pick_cache_signature,
            existing_cache_data=geometry_runtime_state.manual_pick_cache_data,
            cache_signature_fn=_geometry_manual_pick_cache_signature,
            simulated_peaks_for_params=_geometry_manual_simulated_peaks_for_params,
            build_grouped_candidates=_geometry_manual_pick_candidates,
            build_simulated_lookup=_geometry_manual_simulated_lookup,
            current_match_config=_current_geometry_manual_match_config,
            auto_match_background_context=_auto_match_background_context,
        )
    )
    return cache_data


def _geometry_manual_candidate_source_key(
    entry: dict[str, object] | None,
) -> tuple[object, ...] | None:
    """Return a stable lookup key for one manual-pick candidate or match."""
    return gui_manual_geometry.geometry_manual_candidate_source_key(
        entry,
        normalize_hkl_key=_normalize_hkl_key,
    )


def _geometry_manual_choose_group_at(
    grouped_candidates: dict[tuple[object, ...], list[dict[str, object]]] | None,
    col: float,
    row: float,
    *,
    window_size_px: float,
) -> tuple[tuple[object, ...] | None, list[dict[str, object]], float]:
    """Return the nearest clickable Qr/Qz group inside a local click window."""
    return gui_manual_geometry.geometry_manual_choose_group_at(
        grouped_candidates,
        col,
        row,
        window_size_px=window_size_px,
    )


def _geometry_manual_zoom_bounds(
    col: float,
    row: float,
    image_shape: Sequence[int] | None,
    *,
    window_size_px: float = 100.0,
) -> tuple[float, float, float, float]:
    """Return clamped image-space bounds for a square manual-pick zoom window."""
    return gui_manual_geometry.geometry_manual_zoom_bounds(
        col,
        row,
        image_shape,
        window_size_px=window_size_px,
    )


def _geometry_manual_anchor_axis_limits(
    value: float,
    span: float,
    anchor_fraction: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Return clamped axis limits that keep *value* at a fixed screen fraction."""
    return gui_manual_geometry.geometry_manual_anchor_axis_limits(
        value,
        span,
        anchor_fraction,
        lower_bound,
        upper_bound,
    )


def _geometry_manual_group_target_count(
    group_key: tuple[object, ...] | None,
    group_entries: Sequence[dict[str, object]] | None,
) -> int:
    """Return how many manual background peaks a selected group should collect."""
    return gui_manual_geometry.geometry_manual_group_target_count(
        group_key,
        group_entries,
        normalize_hkl_key=_normalize_hkl_key,
    )


def _geometry_manual_pick_session_active(*, require_current_background: bool = True) -> bool:
    """Return whether a manual background-placement session is in progress."""
    return gui_manual_geometry.geometry_manual_pick_session_active(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        require_current_background=require_current_background,
    )


def _geometry_manual_unassigned_group_candidates() -> list[dict[str, object]]:
    """Return manual-pick group candidates that do not yet have a BG assignment."""
    return gui_manual_geometry.geometry_manual_unassigned_group_candidates(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        candidate_source_key=_geometry_manual_candidate_source_key,
    )


def _geometry_manual_current_pending_candidate() -> dict[str, object] | None:
    """Return one remaining simulated peak awaiting a manual background click."""
    return gui_manual_geometry.geometry_manual_current_pending_candidate(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        candidate_source_key=_geometry_manual_candidate_source_key,
    )


def _geometry_manual_nearest_candidate_to_point(
    col: float,
    row: float,
    candidate_entries: Sequence[dict[str, object]] | None,
) -> tuple[dict[str, object] | None, float]:
    """Return the nearest simulated candidate to one display-space point."""
    return gui_manual_geometry.geometry_manual_nearest_candidate_to_point(
        col,
        row,
        candidate_entries,
    )


def _geometry_manual_pair_entry_from_candidate(
    candidate: dict[str, object] | None,
    peak_col: float,
    peak_row: float,
    *,
    group_key: tuple[object, ...] | None,
    raw_col: float | None = None,
    raw_row: float | None = None,
    caked_col: float | None = None,
    caked_row: float | None = None,
    raw_caked_col: float | None = None,
    raw_caked_row: float | None = None,
    placement_error_px: float | None = None,
    sigma_px: float | None = None,
) -> dict[str, object] | None:
    """Build one saved manual pair entry from a candidate + measured background point."""
    return gui_manual_geometry.geometry_manual_pair_entry_from_candidate(
        candidate,
        peak_col,
        peak_row,
        group_key=group_key,
        raw_col=raw_col,
        raw_row=raw_row,
        caked_col=caked_col,
        caked_row=caked_row,
        raw_caked_col=raw_caked_col,
        raw_caked_row=raw_caked_row,
        placement_error_px=placement_error_px,
        sigma_px=sigma_px,
        normalize_hkl_key=_normalize_hkl_key,
    )


def _clear_geometry_manual_preview_artists(*, redraw: bool = True) -> None:
    """Remove manual-placement preview markers from the plot."""


    for artist in geometry_runtime_state.manual_preview_artists:
        try:
            artist.remove()
        except ValueError:
            pass
    geometry_runtime_state.manual_preview_artists.clear()
    if redraw:
        canvas.draw_idle()


def _show_geometry_manual_preview(
    raw_col: float,
    raw_row: float,
    refined_col: float | None = None,
    refined_row: float | None = None,
) -> None:
    """Draw raw and refined manual-placement preview markers."""


    _clear_geometry_manual_preview_artists(redraw=False)
    raw_artist, = ax.plot(
        [float(raw_col)],
        [float(raw_row)],
        marker="D",
        markerfacecolor="none",
        markeredgecolor="yellow",
        markersize=8,
        markeredgewidth=1.6,
        linestyle="none",
        zorder=11,
    )
    geometry_runtime_state.manual_preview_artists.append(raw_artist)

    if (
        refined_col is not None
        and refined_row is not None
        and np.isfinite(float(refined_col))
        and np.isfinite(float(refined_row))
    ):
        refined_artist, = ax.plot(
            [float(refined_col)],
            [float(refined_row)],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="cyan",
            markersize=9,
            markeredgewidth=1.8,
            linestyle="none",
            zorder=12,
        )
        link_artist, = ax.plot(
            [float(raw_col), float(refined_col)],
            [float(raw_row), float(refined_row)],
            "-",
            color="cyan",
            linewidth=1.1,
            alpha=0.75,
            zorder=11,
        )
        geometry_runtime_state.manual_preview_artists.extend([refined_artist, link_artist])

    canvas.draw_idle()


def _geometry_manual_preview_due(col: float, row: float) -> bool:
    """Throttle manual-placement preview updates during mouse motion."""
    return gui_manual_geometry.geometry_manual_preview_due(
        col,
        row,
        pick_session=geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        min_interval_s=float(GEOMETRY_MANUAL_PREVIEW_MIN_INTERVAL_S),
        min_move_px=float(GEOMETRY_MANUAL_PREVIEW_MIN_MOVE_PX),
        perf_counter_fn=perf_counter,
    )


def _geometry_manual_refine_preview_point(
    candidate: dict[str, object] | None,
    raw_col: float,
    raw_row: float,
    *,
    display_background: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
) -> tuple[float, float]:
    """Refine one manual raw click/release position to the best background peak."""
    background_local = (
        _current_geometry_manual_pick_background_image()
        if display_background is None
        else display_background
    )
    return gui_manual_geometry.geometry_manual_refine_preview_point(
        candidate,
        raw_col,
        raw_row,
        display_background=background_local,
        cache_data=cache_data,
        build_cache_data=lambda: _get_geometry_manual_pick_cache(
            param_set=_current_geometry_fit_params(),
            prefer_cache=True,
            background_image=background_local,
        ),
        use_caked_space=_geometry_manual_pick_uses_caked_space(),
        radial_axis=np.asarray(simulation_runtime_state.last_caked_radial_values, dtype=float),
        azimuth_axis=np.asarray(simulation_runtime_state.last_caked_azimuth_values, dtype=float),
        match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
        peak_maximum_near_in_image_fn=_peak_maximum_near_in_image,
        caked_axis_to_image_index_fn=_caked_axis_to_image_index,
        caked_image_index_to_axis_fn=_caked_image_index_to_axis,
        refine_caked_peak_center_fn=_refine_caked_peak_center,
    )


def _restore_geometry_manual_pick_view(*, redraw: bool = True) -> None:
    """Restore the pre-zoom axis view for manual background placement."""
    gui_manual_geometry.restore_geometry_manual_pick_view(
        geometry_manual_state.pick_session,
        axis=ax,
        canvas=canvas,
        redraw=redraw,
    )


def _apply_geometry_manual_pick_zoom(
    col: float,
    row: float,
    *,
    anchor_fraction_x: float = 0.5,
    anchor_fraction_y: float = 0.5,
) -> None:
    """Zoom to a fixed local window while the user is placing manual points."""
    gui_manual_geometry.apply_geometry_manual_pick_zoom(
        geometry_manual_state.pick_session,
        col,
        row,
        display_background=_current_geometry_manual_pick_background_image(),
        axis=ax,
        canvas=canvas,
        use_caked_space=_geometry_manual_pick_uses_caked_space(),
        last_caked_extent=simulation_runtime_state.last_caked_extent,
        caked_zoom_tth_deg=float(GEOMETRY_MANUAL_CAKED_ZOOM_TTH_DEG),
        caked_zoom_phi_deg=float(GEOMETRY_MANUAL_CAKED_ZOOM_PHI_DEG),
        pick_zoom_window_px=float(GEOMETRY_MANUAL_PICK_ZOOM_WINDOW_PX),
        anchor_fraction_x=anchor_fraction_x,
        anchor_fraction_y=anchor_fraction_y,
        anchor_axis_limits_fn=_geometry_manual_anchor_axis_limits,
    )


def _update_geometry_manual_pick_preview(
    raw_col: float,
    raw_row: float,
    *,
    force: bool = False,
) -> None:
    """Refresh the manual raw/refined placement preview at one cursor position."""
    display_background = _current_geometry_manual_pick_background_image()
    preview_state = gui_manual_geometry.geometry_manual_pick_preview_state(
        float(raw_col),
        float(raw_row),
        pick_session=geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        force=force,
        remaining_candidates=_geometry_manual_unassigned_group_candidates(),
        display_background=display_background,
        build_cache_data=lambda: _get_geometry_manual_pick_cache(
            param_set=_current_geometry_fit_params(),
            prefer_cache=True,
            background_image=display_background,
        ),
        refine_preview_point=_geometry_manual_refine_preview_point,
        preview_due=_geometry_manual_preview_due,
        nearest_candidate_to_point=_geometry_manual_nearest_candidate_to_point,
        position_error_px=_geometry_manual_position_error_px,
        position_sigma_px=_geometry_manual_position_sigma_px,
        use_caked_space=_geometry_manual_pick_uses_caked_space(),
        caked_angles_to_background_display_coords=_caked_angles_to_background_display_coords,
    )
    if preview_state is None:
        return
    _show_geometry_manual_preview(
        float(preview_state["raw_col"]),
        float(preview_state["raw_row"]),
        float(preview_state["refined_col"]),
        float(preview_state["refined_row"]),
    )
    progress_label_geometry.config(text=str(preview_state["message"]))


def _geometry_manual_session_initial_pairs_display() -> list[dict[str, object]]:
    """Return overlay-ready display entries for the in-progress manual pick session."""
    return gui_manual_geometry.geometry_manual_session_initial_pairs_display(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        candidate_source_key=_geometry_manual_candidate_source_key,
        entry_display_coords=_geometry_manual_entry_display_coords,
    )


def _cancel_geometry_manual_pick_session(
    *,
    restore_view: bool = True,
    redraw: bool = True,
    message: str | None = None,
) -> None:
    """Discard any in-progress manual Qr/Qz placement state."""
    _set_geometry_manual_pick_session(
        gui_manual_geometry.cancel_geometry_manual_pick_session(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        restore_view_fn=_restore_geometry_manual_pick_view,
        clear_preview_artists_fn=_clear_geometry_manual_preview_artists,
        render_current_pairs_fn=_render_current_geometry_manual_pairs,
        update_button_label_fn=_update_geometry_manual_pick_button_label,
        set_status_text=lambda text: progress_label_geometry.config(text=text),
        restore_view=restore_view,
        redraw=redraw,
        message=message,
        )
    )


def _match_geometry_manual_group_to_background(
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    background_image: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
) -> dict[tuple[object, ...], tuple[float, float]]:
    """Return refined measured peak centers for one clicked symmetric Qr/Qz group."""
    background_local = (
        _get_current_background_display() if background_image is None else background_image
    )
    return gui_manual_geometry.match_geometry_manual_group_to_background(
        candidate_entries,
        background_image=background_local,
        cache_data=cache_data,
        build_cache_data=lambda: _get_geometry_manual_pick_cache(
            prefer_cache=True,
            background_image=background_local,
        ),
        auto_match_background_context=_auto_match_background_context,
        match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
        candidate_source_key=_geometry_manual_candidate_source_key,
    )


def _update_geometry_manual_pick_button_label() -> None:
    label = gui_manual_geometry.geometry_manual_pick_button_label(
        armed=geometry_runtime_state.manual_pick_armed,
        current_background_index=background_runtime_state.current_background_index,
        pick_session=geometry_manual_state.pick_session,
        pairs_for_index=_geometry_manual_pairs_for_index,
        pair_group_count=_geometry_manual_pair_group_count,
    )
    gui_views.set_geometry_tool_action_texts(
        geometry_tool_actions_view_state,
        manual_pick_text=label,
    )


def _set_geometry_manual_pick_mode(enabled: bool, *, message: str | None = None) -> None:
    """Arm or disarm manual Qr-set selection on the detector image."""


    geometry_runtime_state.manual_pick_armed = bool(enabled)
    if geometry_runtime_state.manual_pick_armed:
        if not bool(analysis_view_controls_view_state.show_caked_2d_var.get()):
            analysis_view_controls_view_state.show_caked_2d_var.set(True)
            toggle_caked_2d()
        _set_hkl_pick_mode(False)
        _set_geometry_preview_exclude_mode(False)
    else:
        _cancel_geometry_manual_pick_session(restore_view=True, redraw=True)
    _update_geometry_manual_pick_button_label()
    try:
        canvas.get_tk_widget().configure(cursor="crosshair" if geometry_runtime_state.manual_pick_armed else "")
    except Exception:
        pass
    if message:
        progress_label_geometry.config(text=message)


def _ensure_geometry_fit_caked_view(*, force_refresh: bool = False) -> None:
    """Switch geometry fitting/import into the 2D caked integration view now."""

    simulation_runtime_state.update_pending, integration_update_pending = (
        gui_manual_geometry.ensure_geometry_fit_caked_view(
            show_caked_2d_var=analysis_view_controls_view_state.show_caked_2d_var,
            pick_uses_caked_space=_geometry_manual_pick_uses_caked_space,
            toggle_caked_2d=toggle_caked_2d,
            do_update=do_update,
            schedule_update=schedule_update,
            root=root,
            update_pending=simulation_runtime_state.update_pending,
            integration_update_pending=simulation_runtime_state.integration_update_pending,
            update_running=bool(globals().get("update_running", False)),
            force_refresh=force_refresh,
        )
    )


def _toggle_geometry_manual_pick_mode() -> None:
    """Toggle manual geometry Qr-set selection on the image."""

    _set_geometry_manual_pick_mode(
        not geometry_runtime_state.manual_pick_armed,
        message=(
            (
                "Manual geometry picking armed in 2D caked phi-vs-2theta view. "
                "Click a Qr/Qz set once, then click the matching background peaks "
                "for each simulated member of that set."
            )
            if not geometry_runtime_state.manual_pick_armed
            else "Manual geometry picking disabled."
        ),
    )


def _clear_current_geometry_manual_pairs() -> None:
    """Clear saved manual geometry pairs for the current background image."""

    if _geometry_manual_pairs_for_index(background_runtime_state.current_background_index) or _geometry_manual_pick_session_active():
        _push_geometry_manual_undo_state()
    _cancel_geometry_manual_pick_session(restore_view=True, redraw=False)
    _set_geometry_manual_pairs_for_index(background_runtime_state.current_background_index, [])
    _clear_geometry_pick_artists()
    _update_geometry_manual_pick_button_label()
    _set_background_file_status_text()
    progress_label_geometry.config(text="Cleared saved geometry pairs for the current background image.")


def _peak_maximum_near_in_image(
    image: np.ndarray | None,
    col: float,
    row: float,
    *,
    search_radius: int = 5,
) -> tuple[float, float]:
    """Return the brightest local pixel near ``(col, row)`` in display coordinates."""
    return gui_manual_geometry.peak_maximum_near_in_image(
        image,
        col,
        row,
        search_radius=search_radius,
    )


def _detector_pixel_to_scattering_angles(
    col: float,
    row: float,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
) -> tuple[float | None, float | None]:
    """Convert one detector pixel position to ``(2theta, phi)`` in degrees."""

    if center is None or len(center) < 2:
        return None, None
    if not (np.isfinite(detector_distance) and detector_distance > 0.0):
        return None, None
    if not (np.isfinite(pixel_size) and pixel_size > 0.0):
        return None, None
    try:
        center_row = float(center[0])
        center_col = float(center[1])
    except Exception:
        return None, None
    if not (np.isfinite(col) and np.isfinite(row) and np.isfinite(center_row) and np.isfinite(center_col)):
        return None, None

    dx = (float(col) - center_col) * float(pixel_size)
    dy = (center_row - float(row)) * float(pixel_size)
    radius = float(np.hypot(dx, dy))
    two_theta = float(np.degrees(np.arctan2(radius, float(detector_distance))))
    phi = float(np.degrees(np.arctan2(dx, dy)))
    return two_theta, _wrap_phi_range(phi)


def _scattering_angles_to_detector_pixel(
    two_theta_deg: float,
    phi_deg: float,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
) -> tuple[float | None, float | None]:
    """Convert ``(2theta, phi)`` in degrees to one detector pixel position."""

    if center is None or len(center) < 2:
        return None, None
    if not (np.isfinite(detector_distance) and detector_distance > 0.0):
        return None, None
    if not (np.isfinite(pixel_size) and pixel_size > 0.0):
        return None, None
    try:
        center_row = float(center[0])
        center_col = float(center[1])
    except Exception:
        return None, None
    if not (np.isfinite(two_theta_deg) and np.isfinite(phi_deg) and np.isfinite(center_row) and np.isfinite(center_col)):
        return None, None

    radius = float(detector_distance) * float(np.tan(np.deg2rad(float(two_theta_deg))))
    phi_rad = float(np.deg2rad(float(phi_deg)))
    dx = radius * float(np.sin(phi_rad))
    dy = radius * float(np.cos(phi_rad))
    col = center_col + dx / float(pixel_size)
    row = center_row - dy / float(pixel_size)
    return float(col), float(row)


def _caked_axis_to_image_index(
    value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one caked-axis coordinate in degrees to a floating image index."""
    return gui_manual_geometry.caked_axis_to_image_index(value, axis_values)


def _caked_image_index_to_axis(
    index_value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one floating caked image index back to axis-space degrees."""
    return gui_manual_geometry.caked_image_index_to_axis(index_value, axis_values)


def _refine_profile_peak_index(
    profile: Sequence[float] | None,
    seed_index: float,
) -> float:
    """Return one subpixel 1D peak center focused on the top of a local profile."""
    return gui_manual_geometry.refine_profile_peak_index(profile, seed_index)


def _refine_caked_peak_center(
    image: np.ndarray | None,
    radial_axis: Sequence[float] | None,
    azimuth_axis: Sequence[float] | None,
    two_theta_deg: float,
    phi_deg: float,
    *,
    tth_window_deg: float | None = None,
    phi_window_deg: float | None = None,
) -> tuple[float, float]:
    """Refine one caked click to the crest of the local 2theta/phi ridge."""
    return gui_manual_geometry.refine_caked_peak_center(
        image,
        radial_axis,
        azimuth_axis,
        two_theta_deg,
        phi_deg,
        tth_window_deg=tth_window_deg,
        phi_window_deg=phi_window_deg,
        default_tth_window_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_TTH_DEG),
        default_phi_window_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_PHI_DEG),
    )


def _geometry_manual_pick_uses_caked_space() -> bool:
    """Return whether manual geometry picking is currently operating in caked space."""

    if not bool(analysis_view_controls_view_state.show_caked_2d_var.get()):
        return False
    return (
        isinstance(simulation_runtime_state.last_caked_background_image_unscaled, np.ndarray)
        and simulation_runtime_state.last_caked_background_image_unscaled.ndim == 2
        and simulation_runtime_state.last_caked_background_image_unscaled.size > 0
        and np.asarray(simulation_runtime_state.last_caked_radial_values, dtype=float).size > 1
        and np.asarray(simulation_runtime_state.last_caked_azimuth_values, dtype=float).size > 1
    )


def _current_geometry_manual_pick_background_image():
    """Return the background image currently used by manual geometry picking."""

    if _geometry_manual_pick_uses_caked_space():
        return simulation_runtime_state.last_caked_background_image_unscaled
    return _get_current_background_display()


def _geometry_manual_entry_display_coords(
    entry: dict[str, object] | None,
) -> tuple[float, float] | None:
    """Return one saved/manual entry's coordinates in the current displayed view."""

    if not isinstance(entry, dict):
        return None
    key_x = "caked_x" if _geometry_manual_pick_uses_caked_space() else "x"
    key_y = "caked_y" if _geometry_manual_pick_uses_caked_space() else "y"
    try:
        col = float(entry.get(key_x, np.nan))
        row = float(entry.get(key_y, np.nan))
    except Exception:
        col = float("nan")
        row = float("nan")
    if _geometry_manual_pick_uses_caked_space() and not (np.isfinite(col) and np.isfinite(row)):
        try:
            raw_col = float(entry.get("x", np.nan))
            raw_row = float(entry.get("y", np.nan))
        except Exception:
            raw_col = float("nan")
            raw_row = float("nan")
        angles = _display_to_detector_angles(raw_col, raw_row, simulation_runtime_state.ai_cache.get("ai"))
        if angles is not None:
            col = float(angles[0])
            row = float(_wrap_phi_range(float(angles[1])))
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _caked_angles_to_background_display_coords(
    two_theta_deg: float,
    phi_deg: float,
) -> tuple[float | None, float | None]:
    """Back-project one caked-space point to the displayed detector background."""
    center = None
    try:
        center = [float(center_x_var.get()), float(center_y_var.get())]
    except Exception:
        center = None
    return gui_manual_geometry.caked_angles_to_background_display_coords(
        two_theta_deg,
        phi_deg,
        ai=simulation_runtime_state.ai_cache.get("ai"),
        native_background=_get_current_background_native(),
        get_detector_angular_maps=_get_detector_angular_maps,
        scattering_angles_to_detector_pixel=_scattering_angles_to_detector_pixel,
        center=center,
        detector_distance=float(corto_detector_var.get()),
        pixel_size=float(pixel_size_m),
        rotate_point_for_display=_rotate_point_for_display,
        display_rotate_k=DISPLAY_ROTATE_K,
    )


def _native_detector_coords_to_caked_display_coords(
    col: float,
    row: float,
) -> tuple[float, float] | None:
    """Project one native detector pixel into the active caked display axes."""
    center = None
    try:
        center = [float(center_x_var.get()), float(center_y_var.get())]
    except Exception:
        center = None
    return gui_manual_geometry.native_detector_coords_to_caked_display_coords(
        col,
        row,
        ai=simulation_runtime_state.ai_cache.get("ai"),
        get_detector_angular_maps=_get_detector_angular_maps,
        detector_pixel_to_scattering_angles=_detector_pixel_to_scattering_angles,
        center=center,
        detector_distance=float(corto_detector_var.get()),
        pixel_size=float(pixel_size_m),
        wrap_phi_range=_wrap_phi_range,
    )


def _project_geometry_manual_peaks_to_current_view(
    simulated_peaks: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Project simulated manual-pick candidates into the currently displayed view."""

    projected: list[dict[str, object]] = []
    use_caked = _geometry_manual_pick_uses_caked_space()
    ai = simulation_runtime_state.ai_cache.get("ai")
    sim_shape = (int(image_size), int(image_size))
    radial_axis = np.asarray(simulation_runtime_state.last_caked_radial_values, dtype=float) if use_caked else np.array([])
    azimuth_axis = np.asarray(simulation_runtime_state.last_caked_azimuth_values, dtype=float) if use_caked else np.array([])

    for raw_entry in simulated_peaks or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        try:
            raw_col = float(entry.get("sim_col", np.nan))
            raw_row = float(entry.get("sim_row", np.nan))
        except Exception:
            raw_col = float("nan")
            raw_row = float("nan")
        entry["sim_col_raw"] = float(raw_col)
        entry["sim_row_raw"] = float(raw_row)

        if np.isfinite(raw_col) and np.isfinite(raw_row):
            angles = _display_to_detector_angles(raw_col, raw_row, ai)
            if angles is None:
                native_col, native_row = _display_to_native_sim_coords(raw_col, raw_row, sim_shape)
                angles = _detector_pixel_to_scattering_angles(
                    float(native_col),
                    float(native_row),
                    [float(center_x_var.get()), float(center_y_var.get())],
                    float(corto_detector_var.get()),
                    float(pixel_size_m),
                )
            if angles is not None and angles[0] is not None and angles[1] is not None:
                entry["caked_x"] = float(angles[0])
                entry["caked_y"] = float(_wrap_phi_range(float(angles[1])))

        if use_caked:
            try:
                caked_col = float(entry.get("caked_x", np.nan))
                caked_row = float(entry.get("caked_y", np.nan))
            except Exception:
                caked_col = float("nan")
                caked_row = float("nan")
            if np.isfinite(caked_col) and np.isfinite(caked_row):
                entry["sim_col"] = float(caked_col)
                entry["sim_row"] = float(caked_row)
                entry["sim_col_global"] = float(caked_col)
                entry["sim_row_global"] = float(caked_row)
                entry["sim_col_local"] = float(_caked_axis_to_image_index(caked_col, radial_axis))
                entry["sim_row_local"] = float(_caked_axis_to_image_index(caked_row, azimuth_axis))

        projected.append(entry)
    return projected


def _geometry_manual_simulated_peaks_for_params(
    param_set: dict[str, object] | None = None,
    *,
    prefer_cache: bool = True,
) -> list[dict[str, object]]:
    """Return preview-style simulated peaks for manual geometry pair selection."""

    try:
        params_local = dict(param_set) if isinstance(param_set, dict) else _current_geometry_fit_params()
        miller_array = np.asarray(miller, dtype=np.float64)
        intensity_array = np.asarray(intensities, dtype=np.float64)
        if miller_array.ndim != 2 or miller_array.shape[1] != 3 or miller_array.size == 0:
            return []
        if intensity_array.shape[0] != miller_array.shape[0]:
            return []
        raw_peaks = _simulate_preview_style_peaks_for_fit(
            miller_array,
            intensity_array,
            image_size,
            params_local,
        )
        return _project_geometry_manual_peaks_to_current_view(raw_peaks)
    except Exception:
        return []


def _geometry_manual_pick_candidates(
    simulated_peaks: Sequence[dict[str, object]] | None,
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    """Collapse simulated peaks into clickable manual-pick Qr/Qz groups."""

    filtered, _, _ = _filter_geometry_fit_simulated_peaks(simulated_peaks)
    collapsed, _ = _collapse_geometry_fit_simulated_peaks(filtered, merge_radius_px=6.0)
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for raw_entry in collapsed:
        if not isinstance(raw_entry, dict):
            continue
        group_key = raw_entry.get("q_group_key")
        if not isinstance(group_key, tuple):
            continue
        grouped[group_key].append(dict(raw_entry))
    for entry_list in grouped.values():
        entry_list.sort(
            key=lambda entry: (
                float(entry.get("sim_col", np.nan))
                if np.isfinite(float(entry.get("sim_col", np.nan)))
                else float("inf"),
                float(entry.get("sim_row", np.nan))
                if np.isfinite(float(entry.get("sim_row", np.nan)))
                else float("inf"),
            )
        )
    return dict(grouped)


def _geometry_manual_simulated_lookup(
    simulated_peaks: Sequence[dict[str, object]] | None,
) -> dict[tuple[int, int], dict[str, object]]:
    """Map preview-style simulated peaks back to their reflection/source rows."""

    lookup: dict[tuple[int, int], dict[str, object]] = {}
    for raw_entry in simulated_peaks or []:
        if not isinstance(raw_entry, dict):
            continue
        try:
            key = (
                int(raw_entry.get("source_table_index")),
                int(raw_entry.get("source_row_index")),
            )
        except Exception:
            continue
        lookup[key] = dict(raw_entry)
    return lookup


def _clear_geometry_pick_artists(*, redraw: bool = True):
    """Remove geometry fit markers from the plot and reset the cache."""


    gui_overlays.clear_artists(
        geometry_runtime_state.pick_artists,
        draw_idle=canvas.draw_idle,
        redraw=redraw,
    )


def _clear_geometry_preview_artists(*, redraw: bool = True):
    """Remove live geometry preview markers from the plot and reset the cache."""


    gui_overlays.clear_artists(
        geometry_runtime_state.preview_artists,
        draw_idle=canvas.draw_idle,
        redraw=redraw,
    )


def _clear_qr_cylinder_overlay_artists(*, redraw: bool = True):
    """Remove analytic Qr-cylinder detector traces from the plot."""


    gui_overlays.clear_artists(
        geometry_runtime_state.qr_cylinder_overlay_artists,
        draw_idle=canvas.draw_idle,
        redraw=redraw,
    )


def _active_qr_cylinder_overlay_entries() -> list[dict[str, object]]:
    """Return active Qr groups from the current simulation state."""

    def _m_values(arr_like: object) -> set[int]:
        arr = np.asarray(arr_like, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            return set()
        finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        if not np.any(finite):
            return set()
        rows = arr[finite, :2]
        h_vals = np.rint(rows[:, 0]).astype(np.int64, copy=False)
        k_vals = np.rint(rows[:, 1]).astype(np.int64, copy=False)
        return {int(v) for v in (h_vals * h_vals + h_vals * k_vals + k_vals * k_vals)}

    try:
        primary_a = float(a_var.get())
    except Exception:
        primary_a = float(av)

    secondary_a = primary_a
    try:
        if av2 is not None:
            secondary_candidate = float(av2)
            if np.isfinite(secondary_candidate) and secondary_candidate > 0.0:
                secondary_a = secondary_candidate
    except Exception:
        secondary_a = primary_a

    primary_m = set()
    if isinstance(simulation_runtime_state.sim_primary_qr, dict):
        for m_raw in simulation_runtime_state.sim_primary_qr.keys():
            try:
                primary_m.add(int(m_raw))
            except (TypeError, ValueError):
                continue
    primary_m.update(_m_values(globals().get("SIM_MILLER1")))

    secondary_m = _m_values(globals().get("SIM_MILLER2"))

    entries: list[dict[str, object]] = []
    for source_label, lattice_a, m_values in (
        ("primary", primary_a, primary_m),
        ("secondary", secondary_a, secondary_m),
    ):
        for m_idx in sorted(m_values):
            qr_val = _qr_value_for_m(int(m_idx), lattice_a)
            if not np.isfinite(qr_val) or qr_val < 0.0:
                continue
            entries.append(
                {
                    "key": (str(source_label), int(m_idx)),
                    "source": str(source_label),
                    "m": int(m_idx),
                    "qr": float(qr_val),
                }
            )
    return entries


def _qr_cylinder_overlay_signature(
    entries: Sequence[dict[str, object]],
) -> tuple[object, ...]:
    """Return a cache signature for the analytic detector-trace overlay."""

    qr_keys = tuple(
        (str(entry["source"]), int(entry["m"]), round(float(entry["qr"]), 10))
        for entry in entries
    )
    return (
        tuple(qr_keys),
        bool(analysis_view_controls_view_state.show_caked_2d_var.get()),
        int(image_size),
        int(SIM_DISPLAY_ROTATE_K),
        float(center_x_var.get()),
        float(center_y_var.get()),
        float(corto_detector_var.get()),
        float(gamma_var.get()),
        float(Gamma_var.get()),
        float(chi_var.get()),
        float(psi),
        float(psi_z_var.get()),
        float(zs_var.get()),
        float(zb_var.get()),
        float(theta_initial_var.get()),
        float(cor_angle_var.get()),
        float(pixel_size_m),
        float(lambda_),
        round(float(np.real(n2)), 12),
        round(float(np.imag(n2)), 12),
    )


def _refresh_qr_cylinder_overlay(*, redraw: bool = True, update_status: bool = False):
    """Draw analytic Ewald/constant-Qr traces in the current detector or caked view."""


    overlay_var = geometry_overlay_actions_view_state.show_qr_cylinder_overlay_var
    if overlay_var is None or not bool(overlay_var.get()):
        _clear_qr_cylinder_overlay_artists(redraw=redraw)
        return

    entries = _active_qr_cylinder_overlay_entries()
    if not entries:
        geometry_runtime_state.qr_cylinder_overlay_cache = {"signature": None, "paths": []}
        _clear_qr_cylinder_overlay_artists(redraw=redraw)
        if update_status and "progress_label_positions" in globals():
            progress_label_positions.config(
                text="Qr cylinder overlay unavailable: no active Bragg Qr groups."
            )
        return

    signature = _qr_cylinder_overlay_signature(entries)
    cached_sig = geometry_runtime_state.qr_cylinder_overlay_cache.get("signature")
    if cached_sig != signature:
        render_in_caked_space = bool(analysis_view_controls_view_state.show_caked_2d_var.get())
        ai = simulation_runtime_state.ai_cache.get("ai")
        two_theta_map = None
        phi_map = None
        if render_in_caked_space:
            two_theta_map, phi_map = _get_detector_angular_maps(ai)
        geometry = IntersectionGeometry(
            image_size=int(image_size),
            center_col=float(center_y_var.get()),
            center_row=float(center_x_var.get()),
            distance_cor_to_detector=float(corto_detector_var.get()),
            gamma_deg=float(gamma_var.get()),
            Gamma_deg=float(Gamma_var.get()),
            chi_deg=float(chi_var.get()),
            psi_deg=float(psi),
            psi_z_deg=float(psi_z_var.get()),
            zs=float(zs_var.get()),
            zb=float(zb_var.get()),
            theta_initial_deg=float(theta_initial_var.get()),
            cor_angle_deg=float(cor_angle_var.get()),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            pixel_size_m=float(pixel_size_m),
        )
        native_shape = (int(image_size), int(image_size))
        paths: list[dict[str, object]] = []
        for entry in entries:
            try:
                traces = project_qr_cylinder_to_detector(
                    qr_value=float(entry["qr"]),
                    geometry=geometry,
                    wavelength=float(lambda_),
                    n2=n2,
                    phi_samples=721,
                )
            except Exception:
                continue
            for trace in traces:
                if render_in_caked_space:
                    if two_theta_map is None or phi_map is None:
                        continue
                    display_cols, display_rows = interpolate_trace_to_caked_coords(
                        detector_cols=trace.detector_col,
                        detector_rows=trace.detector_row,
                        valid_mask=trace.valid_mask,
                        two_theta_map=two_theta_map,
                        phi_map_deg=phi_map,
                        two_theta_limits=(0.0, 90.0),
                    )
                else:
                    display_cols = np.full_like(trace.detector_col, np.nan, dtype=np.float64)
                    display_rows = np.full_like(trace.detector_row, np.nan, dtype=np.float64)
                    valid_idx = np.nonzero(trace.valid_mask)[0]
                    for idx in valid_idx:
                        dcol, drow = _native_sim_to_display_coords(
                            float(trace.detector_col[idx]),
                            float(trace.detector_row[idx]),
                            native_shape,
                        )
                        display_cols[idx] = dcol
                        display_rows[idx] = drow
                valid_count = int(
                    np.count_nonzero(np.isfinite(display_cols) & np.isfinite(display_rows))
                )
                if valid_count < 2:
                    continue
                paths.append(
                    {
                        "source": str(entry["source"]),
                        "qr": float(entry["qr"]),
                        "cols": display_cols,
                        "rows": display_rows,
                    }
                )
        geometry_runtime_state.qr_cylinder_overlay_cache = {"signature": signature, "paths": paths}

    paths = geometry_runtime_state.qr_cylinder_overlay_cache.get("paths", [])
    if not paths:
        _clear_qr_cylinder_overlay_artists(redraw=False)
        if redraw:
            canvas.draw_idle()
        if update_status and "progress_label_positions" in globals():
            progress_label_positions.config(
                text="Qr cylinder overlay found no visible traces in the current view."
            )
        return
    gui_overlays.draw_qr_cylinder_overlay_paths(
        ax,
        paths,
        qr_cylinder_overlay_artists=geometry_runtime_state.qr_cylinder_overlay_artists,
        clear_qr_cylinder_overlay_artists=_clear_qr_cylinder_overlay_artists,
        draw_idle=canvas.draw_idle,
        redraw=redraw,
    )
    if update_status and "progress_label_positions" in globals():
        progress_label_positions.config(
            text=f"Showing analytic Qr-cylinder traces for {len(entries)} active Qr groups."
        )


def _geometry_q_group_window_open() -> bool:
    """Return whether the Qr/Qz geometry-fit selector window is open."""

    return gui_views.geometry_q_group_window_open(geometry_q_group_view_state)


def _clear_all_geometry_overlay_artists():
    """Clear fitted markers and live preview overlays together."""

    _clear_geometry_pick_artists(redraw=False)
    _clear_geometry_preview_artists(redraw=False)
    canvas.draw_idle()
    try:
        progress_label_geometry.config(
            text="Geometry overlays cleared. Run Fit Geometry again to redraw them."
        )
    except Exception:
        pass


def _live_geometry_preview_signature() -> tuple[object, ...]:
    """Return a lightweight signature for the current live preview context."""

    return (
        simulation_runtime_state.last_simulation_signature,
        int(background_runtime_state.current_background_index),
        id(background_runtime_state.current_background_display),
        bool(background_runtime_state.visible),
        bool(analysis_view_controls_view_state.show_caked_2d_var.get())
        if analysis_view_controls_view_state.show_caked_2d_var is not None
        else False,
    )


def _live_preview_match_key(entry: dict[str, object] | None) -> tuple[object, ...] | None:
    """Build a stable exclusion key for a preview match entry."""

    if not isinstance(entry, dict):
        return None
    hkl_key = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
    source_label = entry.get("source_label")
    source_table_index = entry.get("source_table_index")
    source_row_index = entry.get("source_row_index")
    if (
        hkl_key is not None
        and source_label is not None
        and source_table_index is not None
        and source_row_index is not None
    ):
        try:
            return (
                "peak",
                str(source_label),
                int(source_table_index),
                int(source_row_index),
                int(hkl_key[0]),
                int(hkl_key[1]),
                int(hkl_key[2]),
            )
        except Exception:
            pass

    source_peak_index = entry.get("source_peak_index")
    if hkl_key is not None and source_peak_index is not None:
        try:
            return (
                "peak_index",
                int(source_peak_index),
                int(hkl_key[0]),
                int(hkl_key[1]),
                int(hkl_key[2]),
            )
        except Exception:
            pass

    if hkl_key is None:
        return None
    try:
        sim_col = float(entry.get("sim_x", np.nan))
        sim_row = float(entry.get("sim_y", np.nan))
    except Exception:
        sim_col = float("nan")
        sim_row = float("nan")
    return (
        "hkl_coord",
        int(hkl_key[0]),
        int(hkl_key[1]),
        int(hkl_key[2]),
        round(sim_col, 1),
        round(sim_row, 1),
    )


def _live_preview_match_hkl(entry: dict[str, object] | None) -> tuple[int, int, int] | None:
    """Return the normalized HKL for a preview match entry."""

    if not isinstance(entry, dict):
        return None
    return _normalize_hkl_key(entry.get("hkl", entry.get("label")))


def _live_preview_match_is_excluded(entry: dict[str, object] | None) -> bool:
    """Return whether a preview match entry is currently excluded."""

    key = _live_preview_match_key(entry)
    return key in geometry_preview_state.excluded_keys if key is not None else False


def _filter_live_preview_matches(
    matched_pairs: Sequence[dict[str, object]] | None,
) -> tuple[list[dict[str, object]], int]:
    """Return auto-match pairs after applying user exclusions."""

    filtered: list[dict[str, object]] = []
    excluded_count = 0
    excluded_hkls: set[tuple[int, int, int]] = set()
    for entry in geometry_preview_state.overlay.pairs:
        if not isinstance(entry, dict):
            continue
        if not _live_preview_match_is_excluded(entry):
            continue
        if _live_preview_match_key(entry) is not None:
            continue
        hkl_key = _live_preview_match_hkl(entry)
        if hkl_key is not None:
            excluded_hkls.add(hkl_key)

    for raw_entry in matched_pairs or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        key = _live_preview_match_key(entry)
        excluded = key in geometry_preview_state.excluded_keys if key is not None else False
        if not excluded:
            hkl_key = _live_preview_match_hkl(entry)
            if hkl_key is not None and hkl_key in excluded_hkls:
                excluded = True
        if excluded:
            excluded_count += 1
            continue
        filtered.append(entry)
    return filtered, excluded_count


def _apply_live_preview_match_exclusions(
    matched_pairs: Sequence[dict[str, object]] | None,
    match_stats: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object], int]:
    """Apply preview exclusions and refresh fit-facing summary metrics."""

    filtered_pairs, excluded_count = _filter_live_preview_matches(matched_pairs)
    stats = dict(match_stats) if isinstance(match_stats, dict) else {}
    stats["excluded_count"] = int(excluded_count)
    stats["matched_count"] = int(len(filtered_pairs))
    stats["matched_after_exclusions"] = int(len(filtered_pairs))

    match_dists = np.asarray(
        [float(entry.get("distance_px", np.nan)) for entry in filtered_pairs],
        dtype=float,
    )
    match_dists = match_dists[np.isfinite(match_dists)]
    match_conf = np.asarray(
        [float(entry.get("confidence", np.nan)) for entry in filtered_pairs],
        dtype=float,
    )
    match_conf = match_conf[np.isfinite(match_conf)]

    stats["mean_match_distance_px"] = (
        float(np.mean(match_dists)) if match_dists.size else float("nan")
    )
    stats["p90_match_distance_px"] = (
        float(np.percentile(match_dists, 90.0)) if match_dists.size else float("nan")
    )
    stats["median_match_confidence"] = (
        float(np.median(match_conf)) if match_conf.size else float("nan")
    )
    return filtered_pairs, stats, int(excluded_count)


def _geometry_reference_hit_rows(table: object) -> list[np.ndarray]:
    """Return all propagated hit rows recorded for the selected beam sample."""

    tbl_arr = np.asarray(table, dtype=float)
    if tbl_arr.ndim != 2 or tbl_arr.shape[0] == 0 or tbl_arr.shape[1] < 7:
        return []
    rows: list[np.ndarray] = []
    for row in tbl_arr:
        if row.shape[0] < 7:
            continue
        if not (
            np.isfinite(row[0])
            and np.isfinite(row[1])
            and np.isfinite(row[2])
            and np.isfinite(row[4])
            and np.isfinite(row[5])
            and np.isfinite(row[6])
        ):
            continue
        rows.append(np.asarray(row[:7], dtype=float))
    return rows


def _build_live_preview_simulated_peaks_from_cache() -> list[dict[str, object]]:
    """Build simulated peaks from all detector solutions of the selected beam sample."""

    simulated_peaks: list[dict[str, object]] = []
    max_positions_local = simulation_runtime_state.stored_max_positions_local
    if max_positions_local is None:
        return simulated_peaks
    peak_table_lattice_local = simulation_runtime_state.stored_peak_table_lattice
    if (
        not peak_table_lattice_local
        or len(peak_table_lattice_local) != len(max_positions_local)
    ):
        peak_table_lattice_local = [
            (float(a_var.get()), float(c_var.get()), "primary")
            for _ in max_positions_local
        ]

    image_shape = (
        tuple(int(v) for v in simulation_runtime_state.stored_sim_image.shape[:2])
        if simulation_runtime_state.stored_sim_image is not None
        else (int(image_size), int(image_size))
    )

    for table_idx, tbl in enumerate(max_positions_local):
        rows = _geometry_reference_hit_rows(tbl)
        if not rows:
            continue
        av_used, cv_used, source_label = peak_table_lattice_local[table_idx]
        for row_idx, row in enumerate(rows):
            I, xpix, ypix, _phi, H, K, L = row[:7]
            cx = int(round(float(xpix)))
            cy = int(round(float(ypix)))
            disp_cx, disp_cy = _native_sim_to_display_coords(cx, cy, image_shape)
            hkl_key = tuple(int(np.rint(v)) for v in (H, K, L))
            hkl_raw = (float(H), float(K), float(L))
            q_group_key, qr_val, qz_val = _reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
            )
            if q_group_key is None:
                continue
            simulated_peaks.append(
                {
                    "hkl": hkl_key,
                    "label": f"{hkl_key[0]},{hkl_key[1]},{hkl_key[2]}",
                    "sim_col": float(disp_cx),
                    "sim_row": float(disp_cy),
                    "weight": max(0.0, float(I)),
                    "source_peak_index": int(len(simulated_peaks)),
                    "source_label": str(source_label),
                    "source_table_index": int(table_idx),
                    "source_row_index": int(row_idx),
                    "hkl_raw": hkl_raw,
                    "av": float(av_used),
                    "cv": float(cv_used),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "q_group_key": q_group_key,
                }
            )
    return simulated_peaks


def _q_group_key_component(value: float) -> int | float:
    """Normalize a floating Q-group component into a stable hashable value."""
    return gui_manual_geometry.q_group_key_component(value)


def _integer_gz_index(value: object, *, tol: float = 1e-6) -> int | None:
    """Return the integer Gz/L index when the value is close enough to an integer."""
    return gui_manual_geometry.integer_gz_index(value, tol=tol)


def _reflection_q_group_metadata(
    hkl_value: object,
    *,
    source_label: object = "primary",
    a_value: object = np.nan,
    c_value: object = np.nan,
    qr_value: object = np.nan,
) -> tuple[tuple[object, ...] | None, float, float]:
    """Return stable Qr/Qz grouping metadata for a simulated reflection."""

    if isinstance(hkl_value, (list, tuple, np.ndarray)) and len(hkl_value) >= 3:
        try:
            h_raw = float(hkl_value[0])
            k_raw = float(hkl_value[1])
            l_raw = float(hkl_value[2])
        except Exception:
            return None, float("nan"), float("nan")
    else:
        return None, float("nan"), float("nan")

    m_val = h_raw * h_raw + h_raw * k_raw + k_raw * k_raw
    l_int = _integer_gz_index(l_raw)
    if l_int is None:
        return None, float("nan"), float("nan")

    try:
        qr_val = float(qr_value)
    except Exception:
        qr_val = float("nan")
    try:
        a_used = float(a_value)
    except Exception:
        a_used = float("nan")
    try:
        c_used = float(c_value)
    except Exception:
        c_used = float("nan")

    if not np.isfinite(qr_val):
        if np.isfinite(a_used) and a_used > 0.0 and m_val >= 0.0:
            qr_val = (2.0 * np.pi / a_used) * np.sqrt((4.0 / 3.0) * m_val)
        else:
            qr_val = float("nan")
    qz_val = (
        (2.0 * np.pi / c_used) * float(l_int)
        if np.isfinite(c_used) and c_used > 0.0
        else float("nan")
    )
    key = (
        "q_group",
        _normalize_bragg_qr_source_label(str(source_label) if source_label is not None else "primary"),
        _q_group_key_component(m_val),
        int(l_int),
    )
    return key, float(qr_val), float(qz_val)


def _geometry_q_group_key_from_entry(entry: dict[str, object] | None) -> tuple[object, ...] | None:
    """Return the stable Qr/Qz group key for a simulated peak entry."""

    if not isinstance(entry, dict):
        return None
    key, _, _ = _reflection_q_group_metadata(
        entry.get("hkl_raw", entry.get("hkl")),
        source_label=entry.get("source_label", "primary"),
        a_value=entry.get("av", np.nan),
        c_value=entry.get("cv", np.nan),
        qr_value=entry.get("qr", np.nan),
    )
    return key


def _filter_geometry_fit_simulated_peaks(
    simulated_peaks: Sequence[dict[str, object]] | None,
) -> tuple[list[dict[str, object]], int, int]:
    """Apply the Qr/Qz selection state to simulated peaks."""

    filtered: list[dict[str, object]] = []
    excluded_count = 0
    available_keys: set[tuple[object, ...]] = set()
    listed_keys = _listed_geometry_q_group_keys()
    restrict_to_listed = bool(listed_keys)
    if restrict_to_listed:
        available_keys = set(listed_keys)

    for raw_entry in simulated_peaks or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        group_key = _geometry_q_group_key_from_entry(entry)
        if group_key is None:
            excluded_count += 1
            continue
        entry["q_group_key"] = group_key
        if restrict_to_listed and group_key not in listed_keys:
            excluded_count += 1
            continue
        if not restrict_to_listed:
            available_keys.add(group_key)
        if group_key in geometry_preview_state.excluded_q_groups:
            excluded_count += 1
            continue
        filtered.append(entry)

    return filtered, int(excluded_count), int(len(available_keys))


def _geometry_fit_seed_representative_sort_key(
    entry: dict[str, object] | None,
) -> tuple[object, ...]:
    """Return a stable ordering key for degenerate geometry-fit seeds."""

    if not isinstance(entry, dict):
        return (2, "")
    try:
        return (0, int(entry.get("source_peak_index")))
    except Exception:
        pass
    hkl_key = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
    if hkl_key is not None:
        return (1, int(hkl_key[0]), int(hkl_key[1]), int(hkl_key[2]))
    return (2, str(entry.get("label", "")))


def _geometry_fit_seed_sim_point(
    entry: dict[str, object] | None,
) -> tuple[float, float] | None:
    """Return the simulated detector point for a geometry-fit seed entry."""

    if not isinstance(entry, dict):
        return None
    try:
        sim_col = float(entry.get("sim_col", np.nan))
        sim_row = float(entry.get("sim_row", np.nan))
    except Exception:
        return None
    if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
        return None
    return float(sim_col), float(sim_row)


def _geometry_fit_seed_cluster_anchor(
    entries: Sequence[dict[str, object]],
) -> tuple[float, float] | None:
    """Return the mean simulated point for a cluster of seed entries."""

    cols: list[float] = []
    rows: list[float] = []
    for entry in entries:
        point = _geometry_fit_seed_sim_point(entry)
        if point is None:
            continue
        cols.append(float(point[0]))
        rows.append(float(point[1]))
    if not cols or not rows:
        return None
    return float(np.mean(np.asarray(cols, dtype=float))), float(
        np.mean(np.asarray(rows, dtype=float))
    )


def _collapse_geometry_fit_simulated_peaks(
    simulated_peaks: Sequence[dict[str, object]] | None,
    *,
    merge_radius_px: float = 6.0,
) -> tuple[list[dict[str, object]], int]:
    """Collapse only overlapping degenerate seeds within each Qr/Qz group."""

    grouped_entries: dict[object, list[dict[str, object]]] = {}
    ordered_keys: list[object] = []
    ungrouped_index = 0
    merge_radius = max(0.0, float(merge_radius_px))

    for raw_entry in simulated_peaks or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        group_key = entry.get("q_group_key")
        if group_key is None:
            group_key = ("ungrouped", ungrouped_index)
            ungrouped_index += 1
        if group_key not in grouped_entries:
            grouped_entries[group_key] = []
            ordered_keys.append(group_key)
        grouped_entries[group_key].append(entry)

    collapsed: list[dict[str, object]] = []
    collapsed_degenerate_count = 0

    for group_key in ordered_keys:
        entries = grouped_entries.get(group_key, [])
        if not entries:
            continue

        clusters: list[list[dict[str, object]]] = []
        cluster_anchors: list[tuple[float, float] | None] = []
        for entry in entries:
            point = _geometry_fit_seed_sim_point(entry)
            chosen_cluster_idx = None
            chosen_cluster_dist = float("inf")
            if point is not None and merge_radius > 0.0:
                for cluster_idx, anchor in enumerate(cluster_anchors):
                    if anchor is None:
                        continue
                    dist = float(math.hypot(point[0] - anchor[0], point[1] - anchor[1]))
                    if dist <= merge_radius and dist < chosen_cluster_dist:
                        chosen_cluster_idx = cluster_idx
                        chosen_cluster_dist = dist
            if chosen_cluster_idx is None:
                clusters.append([entry])
                cluster_anchors.append(point)
                continue
            clusters[chosen_cluster_idx].append(entry)
            cluster_anchors[chosen_cluster_idx] = _geometry_fit_seed_cluster_anchor(
                clusters[chosen_cluster_idx]
            )

        for cluster_entries in clusters:
            representative = min(
                cluster_entries,
                key=_geometry_fit_seed_representative_sort_key,
            )
            merged = dict(representative)
            degenerate_hkls: list[tuple[int, int, int]] = []
            seen_hkls: set[tuple[int, int, int]] = set()
            total_weight = 0.0

            for entry in cluster_entries:
                hkl_key = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
                if hkl_key is not None and hkl_key not in seen_hkls:
                    seen_hkls.add(hkl_key)
                    degenerate_hkls.append(hkl_key)
                try:
                    weight = float(entry.get("weight", 0.0))
                except Exception:
                    weight = 0.0
                if np.isfinite(weight) and weight > 0.0:
                    total_weight += float(weight)
                else:
                    total_weight += 1.0

            merged["weight"] = float(total_weight)
            merged["degenerate_count"] = int(len(cluster_entries))
            merged["degenerate_hkls"] = list(degenerate_hkls)
            collapsed_degenerate_count += max(0, len(cluster_entries) - 1)
            collapsed.append(merged)

    return collapsed, int(collapsed_degenerate_count)


def _build_geometry_q_group_entries() -> list[dict[str, object]]:
    """Aggregate current simulated hit tables into unique Qr/Qz selector rows."""

    entries_by_key: dict[tuple[object, ...], dict[str, object]] = {}

    max_positions_local = simulation_runtime_state.stored_max_positions_local
    peak_table_lattice_local = simulation_runtime_state.stored_peak_table_lattice
    if max_positions_local is None:
        return []
    if (
        not peak_table_lattice_local
        or len(peak_table_lattice_local) != len(max_positions_local)
    ):
        peak_table_lattice_local = [
            (float(a_var.get()), float(c_var.get()), "primary")
            for _ in max_positions_local
        ]

    for table_idx, tbl in enumerate(max_positions_local):
        rows = _geometry_reference_hit_rows(tbl)
        if not rows:
            continue
        av_used, cv_used, source_label = peak_table_lattice_local[table_idx]
        for row in rows:
            I, _xpix, _ypix, _phi, H, K, L = row[:7]
            if not np.isfinite(I):
                continue
            group_key, qr_val, qz_val = _reflection_q_group_metadata(
                (H, K, L),
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
            )
            if group_key is None:
                continue
            hkl_key = tuple(int(np.rint(v)) for v in (H, K, L))
            entry = entries_by_key.get(group_key)
            if entry is None:
                entry = {
                    "key": group_key,
                    "source_label": str(source_label),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "gz_index": int(group_key[3]),
                    "total_intensity": 0.0,
                    "peak_count": 0,
                    "hkl_preview": [],
                }
                entries_by_key[group_key] = entry
            entry["total_intensity"] = float(entry["total_intensity"]) + float(abs(I))
            entry["peak_count"] = int(entry["peak_count"]) + 1
            if len(entry["hkl_preview"]) < 4 and hkl_key not in entry["hkl_preview"]:
                entry["hkl_preview"].append(hkl_key)

    entries = list(entries_by_key.values())
    entries.sort(
        key=lambda entry: (
            str(entry.get("source_label", "")),
            float(entry.get("qr", np.nan)) if np.isfinite(float(entry.get("qr", np.nan))) else float("inf"),
            float(entry.get("qz", np.nan)) if np.isfinite(float(entry.get("qz", np.nan))) else float("inf"),
        )
    )
    return entries


def _clone_geometry_q_group_entries(
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Return detached copies of Qr/Qz selector entries."""
    return gui_controllers.clone_geometry_q_group_entries(entries)


def _listed_geometry_q_group_entries() -> list[dict[str, object]]:
    """Return the manually refreshed Qr/Qz rows currently shown in the selector."""
    return gui_controllers.listed_geometry_q_group_entries(geometry_q_group_state)


def _listed_geometry_q_group_keys(
    entries: Sequence[dict[str, object]] | None = None,
) -> set[tuple[object, ...]]:
    """Return the stable keys for the currently listed Qr/Qz rows."""
    return gui_controllers.listed_geometry_q_group_keys(
        geometry_q_group_state,
        entries,
    )


def _set_geometry_q_group_entries_snapshot(
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Replace the listed Qr/Qz snapshot with the provided entries."""
    gui_controllers.replace_geometry_q_group_cached_entries(
        geometry_q_group_state,
        entries,
    )
    listed_keys = _listed_geometry_q_group_keys()
    gui_controllers.retain_geometry_preview_excluded_q_groups(
        geometry_preview_state,
        listed_keys,
    )
    _invalidate_geometry_manual_pick_cache()
    _update_geometry_preview_exclude_button_label()
    return _listed_geometry_q_group_entries()


def _capture_geometry_q_group_entries_snapshot() -> list[dict[str, object]]:
    """Rebuild and store the manual Qr/Qz selector snapshot from the current simulation."""

    entries = _set_geometry_q_group_entries_snapshot(_build_geometry_q_group_entries())
    if _geometry_q_group_window_open():
        _refresh_geometry_q_group_window()
    return entries


def _geometry_q_group_key_from_jsonable(value: object) -> tuple[object, ...] | None:
    """Rebuild one stable Qr/Qz group key from JSON-loaded data."""
    return gui_geometry_q_group_manager.geometry_q_group_key_from_jsonable(value)


def _geometry_q_group_export_rows(
    entries: Sequence[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Build JSON export rows for the current Qr/Qz selector listing."""
    return gui_geometry_q_group_manager.build_geometry_q_group_export_rows(
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        entries=entries,
    )


def _save_geometry_q_group_selection() -> None:
    """Export the currently listed Qr/Qz rows and checkbox state to JSON."""
    gui_geometry_q_group_manager.save_geometry_q_group_selection_with_dialog(
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        file_dialog_dir=get_dir("file_dialog_dir"),
        asksaveasfilename=filedialog.asksaveasfilename,
        set_status_text=lambda text: progress_label_geometry.config(text=text),
    )


def _load_geometry_q_group_selection() -> None:
    """Import a saved Qr/Qz selector list and apply it to the current rows."""
    gui_geometry_q_group_manager.load_geometry_q_group_selection_with_dialog(
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        file_dialog_dir=get_dir("file_dialog_dir"),
        askopenfilename=filedialog.askopenfilename,
        update_geometry_preview_exclude_button_label=_update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=_refresh_geometry_q_group_window,
        live_geometry_preview_enabled=_live_geometry_preview_enabled,
        refresh_live_geometry_preview=(
            lambda: _refresh_live_geometry_preview(update_status=False)
        ),
        set_status_text=lambda text: progress_label_geometry.config(text=text),
    )


def _background_backend_status() -> str:
    return (
        f"k={int(background_runtime_state.backend_rotation_k) % 4} "
        f"flip_x={bool(background_runtime_state.backend_flip_x)} "
        f"flip_y={bool(background_runtime_state.backend_flip_y)}"
    )


def _update_background_backend_status():
    gui_views.set_background_backend_status_text(
        background_backend_debug_view_state,
        _background_backend_status(),
    )


def _rotate_background_backend(delta_k: int):
    background_runtime_state.backend_rotation_k = (int(background_runtime_state.backend_rotation_k) + int(delta_k)) % 4
    _sync_background_runtime_state()
    _update_background_backend_status()
    _mark_chi_square_dirty()
    _update_chi_square_display(force=True)
    schedule_update()


def _toggle_background_backend_flip(axis: str):
    axis = (axis or "").lower()
    if axis == "x":
        background_runtime_state.backend_flip_x = not background_runtime_state.backend_flip_x
    elif axis == "y":
        background_runtime_state.backend_flip_y = not background_runtime_state.backend_flip_y
    _sync_background_runtime_state()
    _update_background_backend_status()
    _mark_chi_square_dirty()
    _update_chi_square_display(force=True)
    schedule_update()


def _reset_background_backend_orientation():
    background_runtime_state.backend_rotation_k = 0
    background_runtime_state.backend_flip_x = False
    background_runtime_state.backend_flip_y = False
    _sync_background_runtime_state()
    _update_background_backend_status()
    _mark_chi_square_dirty()
    _update_chi_square_display(force=True)
    schedule_update()

# -----------------------------------------------------------
# 2)  Mouse‑click handler
# -----------------------------------------------------------
hkl_lookup_view_state = app_state.hkl_lookup_view
bragg_qr_manager_view_state = app_state.bragg_qr_manager_view
hbn_geometry_debug_view_state = app_state.hbn_geometry_debug_view
geometry_overlay_actions_view_state = app_state.geometry_overlay_actions_view
analysis_view_controls_view_state = app_state.analysis_view_controls_view
analysis_export_controls_view_state = app_state.analysis_export_controls_view
integration_range_controls_view_state = app_state.integration_range_controls_view
display_controls_state = app_state.display_controls_state
display_controls_view_state = app_state.display_controls_view
primary_cif_controls_view_state = app_state.primary_cif_controls_view
cif_weight_controls_view_state = app_state.cif_weight_controls_view
structure_factor_pruning_controls_view_state = (
    app_state.structure_factor_pruning_controls_view
)
beam_mosaic_parameter_sliders_view_state = (
    app_state.beam_mosaic_parameter_sliders_view
)
sampling_optics_controls_view_state = app_state.sampling_optics_controls_view
finite_stack_controls_view_state = app_state.finite_stack_controls_view
stacking_parameter_controls_view_state = app_state.stacking_parameter_controls_view


def _sync_peak_selection_state() -> None:
    """Normalize peak/HKL interaction flags stored in shared app state."""

    peak_selection_state.hkl_pick_armed = bool(peak_selection_state.hkl_pick_armed)
    peak_selection_state.suppress_drag_press_once = bool(peak_selection_state.suppress_drag_press_once)


_sync_peak_selection_state()


def _format_geometry_q_group_line(entry: dict[str, object]) -> str:
    """Return a compact label for one Qr/Qz selector row."""
    return gui_geometry_q_group_manager.format_geometry_q_group_line(entry)


def _current_geometry_auto_match_min_matches() -> int:
    """Return the current minimum match count required by the geometry auto-match gate."""
    return gui_geometry_q_group_manager.current_geometry_auto_match_min_matches(
        fit_config,
        _current_geometry_fit_var_names(),
    )


def _geometry_q_group_excluded_count(
    entries: Sequence[dict[str, object]] | None = None,
) -> int:
    """Return how many listed Qr/Qz rows are currently excluded."""
    return gui_geometry_q_group_manager.geometry_q_group_excluded_count(
        geometry_preview_state,
        geometry_q_group_state,
        entries,
    )


def _build_geometry_q_group_window_status_text(
    entries: Sequence[dict[str, object]] | None = None,
) -> str:
    """Build the summary text shown above the Qr/Qz selector rows."""
    return gui_geometry_q_group_manager.build_geometry_q_group_window_status_text(
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names=_current_geometry_fit_var_names(),
        entries=entries,
    )


def _update_geometry_q_group_window_status(entries: Sequence[dict[str, object]] | None = None) -> None:
    """Refresh the summary text at the top of the Qr/Qz selector window."""
    gui_geometry_q_group_manager.update_geometry_q_group_window_status(
        view_state=geometry_q_group_view_state,
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names=_current_geometry_fit_var_names(),
        entries=entries,
    )


def _refresh_geometry_q_group_window() -> None:
    """Redraw the Qr/Qz selector window from the stored manual snapshot."""
    gui_geometry_q_group_manager.refresh_geometry_q_group_window(
        view_state=geometry_q_group_view_state,
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names=_current_geometry_fit_var_names(),
        on_toggle=_on_geometry_q_group_checkbox_changed,
    )


def _on_geometry_q_group_checkbox_changed(
    group_key: tuple[object, ...] | None,
    row_var: tk.BooleanVar,
) -> None:
    """Apply one Qr/Qz include/exclude toggle from the selector window."""
    gui_geometry_q_group_manager.apply_geometry_q_group_checkbox_change_with_side_effects(
        preview_state=geometry_preview_state,
        group_key=group_key,
        row_var=row_var,
        invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=_update_geometry_preview_exclude_button_label,
        update_geometry_q_group_window_status=_update_geometry_q_group_window_status,
        live_geometry_preview_enabled=_live_geometry_preview_enabled,
        refresh_live_geometry_preview=(
            lambda: _refresh_live_geometry_preview(update_status=True)
        ),
        set_status_text=lambda text: progress_label_geometry.config(text=text),
    )


def _set_all_geometry_q_groups_enabled(enabled: bool) -> None:
    """Enable or disable every currently listed Qr/Qz group."""
    gui_geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        enabled=enabled,
        invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
        update_geometry_preview_exclude_button_label=_update_geometry_preview_exclude_button_label,
        refresh_geometry_q_group_window=_refresh_geometry_q_group_window,
        live_geometry_preview_enabled=_live_geometry_preview_enabled,
        refresh_live_geometry_preview=(
            lambda: _refresh_live_geometry_preview(update_status=True)
        ),
        set_status_text=lambda text: progress_label_geometry.config(text=text),
    )


def _request_geometry_q_group_window_update() -> None:
    """Rebuild the listed Qr/Qz rows from the current simulation on demand."""
    gui_geometry_q_group_manager.request_geometry_q_group_window_update_with_side_effects(
        q_group_state=geometry_q_group_state,
        clear_last_simulation_signature=(
            lambda: setattr(
                simulation_runtime_state,
                "last_simulation_signature",
                None,
            )
        ),
        invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
        set_status_text=lambda text: progress_label_geometry.config(text=text),
        schedule_update=schedule_update,
    )


def _close_geometry_q_group_window() -> None:
    """Destroy the Qr/Qz selector window and clear its widget references."""
    gui_geometry_q_group_manager.close_geometry_q_group_window(
        geometry_q_group_view_state,
        geometry_q_group_state,
    )


def _open_geometry_q_group_window() -> None:
    """Open a scrollable Qr/Qz selector for geometry-fit peak inclusion."""
    gui_geometry_q_group_manager.open_geometry_q_group_window(
        root=root,
        view_state=geometry_q_group_view_state,
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names=_current_geometry_fit_var_names(),
        on_toggle=_on_geometry_q_group_checkbox_changed,
        on_include_all=lambda: _set_all_geometry_q_groups_enabled(True),
        on_exclude_all=lambda: _set_all_geometry_q_groups_enabled(False),
        on_update_listed_peaks=_request_geometry_q_group_window_update,
        on_save=_save_geometry_q_group_selection,
        on_load=_load_geometry_q_group_selection,
        on_close=_close_geometry_q_group_window,
    )


def _update_geometry_preview_exclude_button_label():
    label = "Select Qr/Qz Peaks"
    excluded_count = _geometry_q_group_excluded_count()
    if excluded_count > 0:
        label += f" ({excluded_count} off)"
    gui_views.set_geometry_tool_action_texts(
        geometry_tool_actions_view_state,
        preview_exclude_text=label,
    )


def _update_hkl_pick_button_label():
    gui_views.set_hkl_pick_button_text(
        hkl_lookup_view_state,
        gui_peak_selection.hkl_pick_button_text(peak_selection_state.hkl_pick_armed),
    )


def _set_hkl_pick_mode(enabled: bool, *, message: str | None = None):
    peak_selection_state.hkl_pick_armed = bool(enabled)
    _sync_peak_selection_state()
    if peak_selection_state.hkl_pick_armed:
        _set_geometry_preview_exclude_mode(False)
    _update_hkl_pick_button_label()
    if message:
        progress_label_positions.config(text=message)


def _set_geometry_preview_exclude_mode(enabled: bool, *, message: str | None = None):
    """Arm or disarm live-preview exclusion editing."""

    if gui_controllers.set_geometry_preview_exclude_mode(
        geometry_preview_state,
        enabled,
    ):
        _set_hkl_pick_mode(False)
    _update_geometry_preview_exclude_button_label()
    if message:
        progress_label_geometry.config(text=message)


def _toggle_geometry_preview_exclude_mode():
    _open_geometry_q_group_window()
    progress_label_geometry.config(
        text=(
            "Opened the Qr/Qz selector for manual geometry picking. "
            "Unchecked rows are unavailable when selecting Qr sets on the image."
        )
    )


def _distance_point_to_segment_sq(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    """Return squared distance from point P to segment AB in display pixels."""

    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return (float(px) - float(x0)) ** 2 + (float(py) - float(y0)) ** 2

    t = (
        ((float(px) - float(x0)) * dx + (float(py) - float(y0)) * dy)
        / (dx * dx + dy * dy)
    )
    t = min(1.0, max(0.0, float(t)))
    cx = float(x0) + t * dx
    cy = float(y0) + t * dy
    return (float(px) - cx) ** 2 + (float(py) - cy) ** 2


def _clear_live_geometry_preview_exclusions():
    """Clear all user-excluded live preview peaks and redraw the preview."""

    gui_controllers.clear_geometry_preview_excluded_keys(geometry_preview_state)
    gui_controllers.clear_geometry_preview_excluded_q_groups(
        geometry_preview_state,
    )
    _invalidate_geometry_manual_pick_cache()
    _update_geometry_preview_exclude_button_label()
    _refresh_geometry_q_group_window()
    if _live_geometry_preview_enabled():
        _refresh_live_geometry_preview(update_status=True)
    else:
        progress_label_geometry.config(text="Reset all Qr/Qz geometry-fit selections.")


def _toggle_live_geometry_preview_exclusion_at(col: float, row: float) -> bool:
    """Toggle exclusion for the nearest live preview pair near the click point."""

    pairs = geometry_preview_state.overlay.pairs
    if not pairs:
        progress_label_geometry.config(
            text="No live preview pairs are available to exclude."
        )
        return False

    best_entry: dict[str, object] | None = None
    best_d2 = float("inf")
    for raw_entry in pairs:
        if not isinstance(raw_entry, dict):
            continue
        try:
            sim_col = float(raw_entry["sim_x"])
            sim_row = float(raw_entry["sim_y"])
            bg_col = float(raw_entry["x"])
            bg_row = float(raw_entry["y"])
        except Exception:
            continue
        d2 = min(
            (float(col) - sim_col) ** 2 + (float(row) - sim_row) ** 2,
            (float(col) - bg_col) ** 2 + (float(row) - bg_row) ** 2,
            _distance_point_to_segment_sq(
                float(col),
                float(row),
                sim_col,
                sim_row,
                bg_col,
                bg_row,
            ),
        )
        if d2 < best_d2:
            best_d2 = d2
            best_entry = raw_entry

    if best_entry is None or best_d2 > float(GEOMETRY_PREVIEW_TOGGLE_MAX_DISTANCE_PX) ** 2:
        progress_label_geometry.config(
            text=(
                f"No preview pair within {GEOMETRY_PREVIEW_TOGGLE_MAX_DISTANCE_PX:.0f}px "
                "to toggle."
            )
        )
        return False

    key = _live_preview_match_key(best_entry)
    hkl_key = _live_preview_match_hkl(best_entry)
    if key is None or hkl_key is None:
        progress_label_geometry.config(
            text="The selected preview pair cannot be excluded."
        )
        return False

    if key in geometry_preview_state.excluded_keys:
        gui_controllers.set_geometry_preview_match_included(
            geometry_preview_state,
            key,
            included=True,
        )
        action = "Included"
    else:
        gui_controllers.set_geometry_preview_match_included(
            geometry_preview_state,
            key,
            included=False,
        )
        action = "Excluded"

    _render_live_geometry_preview_state(update_status=True)
    progress_label_geometry.config(
        text=f"{action} live preview peak HKL={hkl_key} from geometry fit."
    )
    return True


def _toggle_hkl_pick_mode():
    if peak_selection_state.hkl_pick_armed:
        _set_hkl_pick_mode(False, message="HKL image-pick canceled.")
        return

    if analysis_view_controls_view_state.show_caked_2d_var.get():
        progress_label_positions.config(
            text="Switch off 2D caked view before picking HKL in the image."
        )
        return

    if simulation_runtime_state.unscaled_image is None:
        progress_label_positions.config(text="Run a simulation first.")
        return

    if not _ensure_peak_overlay_data(force=False) or not simulation_runtime_state.peak_positions:
        _set_hkl_pick_mode(
            True,
            message=(
                "Preparing simulated peak map for HKL picking... "
                "wait for the next update."
            ),
        )
        schedule_update()
        return

    _set_hkl_pick_mode(
        True,
        message="HKL image-pick armed: click near a Bragg peak in raw camera view.",
    )


def _nearest_integer_hkl(h: float, k: float, l: float) -> tuple[int, int, int]:
    return gui_peak_selection.nearest_integer_hkl(h, k, l)


def _format_hkl_triplet(h: int, k: int, l: int) -> str:
    return gui_peak_selection.format_hkl_triplet(h, k, l)


def _source_miller_for_label(source_label: str | None) -> np.ndarray:
    return gui_peak_selection.source_miller_for_label(
        simulation_runtime_state,
        source_label,
    )


def _degenerate_hkls_for_qr(
    h: int,
    k: int,
    l: int,
    *,
    source_label: str | None,
) -> list[tuple[int, int, int]]:
    return gui_peak_selection.degenerate_hkls_for_qr(
        simulation_runtime_state,
        h,
        k,
        l,
        source_label=source_label,
    )


def _selected_peak_qr_and_degenerates(
    H: float,
    K: float,
    L: float,
    selected_peak: dict | None,
) -> tuple[float, list[tuple[int, int, int]]]:
    return gui_peak_selection.selected_peak_qr_and_degenerates(
        simulation_runtime_state,
        H,
        K,
        L,
        selected_peak,
        primary_a=float(av),
    )


def _select_peak_by_index(
    idx: int,
    *,
    prefix: str = "Selected peak",
    sync_hkl_vars: bool = True,
    clicked_display: tuple[float, float] | None = None,
    clicked_native: tuple[float, float] | None = None,
    selected_display: tuple[float, float] | None = None,
    selected_native: tuple[float, float] | None = None,
):
    return gui_peak_selection.select_peak_by_index(
        simulation_runtime_state,
        peak_selection_state,
        hkl_lookup_view_state,
        selected_peak_marker,
        idx,
        primary_a=float(av),
        sync_peak_selection_state=_sync_peak_selection_state,
        set_status_text=lambda text: progress_label_positions.config(text=text),
        draw_idle=canvas.draw_idle,
        prefix=prefix,
        sync_hkl_vars=sync_hkl_vars,
        clicked_display=clicked_display,
        clicked_native=clicked_native,
        selected_display=selected_display,
        selected_native=selected_native,
    )


def _select_peak_by_hkl(
    h: int,
    k: int,
    l: int,
    *,
    sync_hkl_vars: bool = True,
    silent_if_missing: bool = False,
):
    return gui_peak_selection.select_peak_by_hkl(
        simulation_runtime_state,
        peak_selection_state,
        hkl_lookup_view_state,
        selected_peak_marker,
        h,
        k,
        l,
        primary_a=float(av),
        ensure_peak_overlay_data=_ensure_peak_overlay_data,
        schedule_update=schedule_update,
        sync_peak_selection_state=_sync_peak_selection_state,
        set_status_text=lambda text: progress_label_positions.config(text=text),
        draw_idle=canvas.draw_idle,
        sync_hkl_vars=sync_hkl_vars,
        silent_if_missing=silent_if_missing,
    )


def _select_peak_from_hkl_controls():
    gui_peak_selection.select_peak_from_hkl_controls(
        simulation_runtime_state,
        peak_selection_state,
        hkl_lookup_view_state,
        selected_peak_marker,
        primary_a=float(av),
        ensure_peak_overlay_data=_ensure_peak_overlay_data,
        schedule_update=schedule_update,
        sync_peak_selection_state=_sync_peak_selection_state,
        set_status_text=lambda text: progress_label_positions.config(text=text),
        draw_idle=canvas.draw_idle,
        tcl_error_types=(tk.TclError,),
    )


def _clear_selected_peak():
    gui_peak_selection.clear_selected_peak(
        simulation_runtime_state,
        peak_selection_state,
        selected_peak_marker,
        sync_peak_selection_state=_sync_peak_selection_state,
        set_status_text=lambda text: progress_label_positions.config(text=text),
        draw_idle=canvas.draw_idle,
    )


def _open_selected_peak_intersection_figure():
    """Open a Bragg/Ewald intersection analysis plot for the selected peak."""

    if simulation_runtime_state.selected_peak_record is None:
        progress_label_positions.config(
            text="Select a Bragg peak first (arm Pick HKL on Image or use HKL controls)."
        )
        return

    try:
        h, k, l = tuple(int(v) for v in simulation_runtime_state.selected_peak_record["hkl"])
        native_col = float(
            simulation_runtime_state.selected_peak_record.get(
                "selected_native_col",
                simulation_runtime_state.selected_peak_record.get("native_col"),
            )
        )
        native_row = float(
            simulation_runtime_state.selected_peak_record.get(
                "selected_native_row",
                simulation_runtime_state.selected_peak_record.get("native_row"),
            )
        )
        lattice_a = float(simulation_runtime_state.selected_peak_record["av"])
        lattice_c = float(simulation_runtime_state.selected_peak_record["cv"])

        geometry = IntersectionGeometry(
            image_size=int(image_size),
            center_col=float(center_y_var.get()),
            center_row=float(center_x_var.get()),
            distance_cor_to_detector=float(corto_detector_var.get()),
            gamma_deg=float(gamma_var.get()),
            Gamma_deg=float(Gamma_var.get()),
            chi_deg=float(chi_var.get()),
            psi_deg=float(psi),
            psi_z_deg=float(psi_z_var.get()),
            zs=float(zs_var.get()),
            zb=float(zb_var.get()),
            theta_initial_deg=float(theta_initial_var.get()),
            cor_angle_deg=float(cor_angle_var.get()),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        beam = IntersectionBeamSamples(
            beam_x_array=np.asarray(simulation_runtime_state.profile_cache["beam_x_array"], dtype=np.float64),
            beam_y_array=np.asarray(simulation_runtime_state.profile_cache["beam_y_array"], dtype=np.float64),
            theta_array=np.asarray(simulation_runtime_state.profile_cache["theta_array"], dtype=np.float64),
            phi_array=np.asarray(simulation_runtime_state.profile_cache["phi_array"], dtype=np.float64),
            wavelength_array=np.asarray(simulation_runtime_state.profile_cache["wavelength_array"], dtype=np.float64),
        )
        mosaic = IntersectionMosaicParams(
            sigma_mosaic_deg=float(sigma_mosaic_var.get()),
            gamma_mosaic_deg=float(gamma_mosaic_var.get()),
            eta=float(eta_var.get()),
            solve_q_steps=_current_solve_q_steps(),
            solve_q_rel_tol=_current_solve_q_rel_tol(),
            solve_q_mode=_current_solve_q_mode_flag(),
        )

        analysis = analyze_reflection_intersection(
            h=h,
            k=k,
            l=l,
            lattice_a=lattice_a,
            lattice_c=lattice_c,
            selected_native_col=native_col,
            selected_native_row=native_row,
            geometry=geometry,
            beam=beam,
            mosaic=mosaic,
            n2=n2,
        )
        fig_analysis = plot_intersection_analysis(analysis)
        manager = getattr(fig_analysis.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(f"Bragg/Ewald HKL=({h},{k},{l})")
            manager.show()
        else:
            fig_analysis.show()
        progress_label_positions.config(
            text=(
                f"Opened Bragg/Ewald analysis for HKL=({h} {k} {l}) "
                f"from source={simulation_runtime_state.selected_peak_record.get('source_label', 'unknown')}."
            )
        )
    except Exception as exc:
        progress_label_positions.config(
            text=f"Intersection analysis failed for selected peak: {exc}"
        )


def _qr_value_for_m(m_idx: int, lattice_a: float) -> float:
    return gui_controllers.qr_value_for_m(m_idx, lattice_a)


def _hk_pairs_grouped_by_m(source_label: str) -> dict[int, list[tuple[int, int]]]:
    return gui_controllers.hk_pairs_grouped_by_m(
        _source_all_miller_for_label(source_label)
    )


def _build_bragg_qr_entries() -> list[dict[str, object]]:
    try:
        primary_a = float(a_var.get())
    except Exception:
        primary_a = float(av)

    secondary_a = primary_a
    try:
        if av2 is not None:
            secondary_candidate = float(av2)
            if np.isfinite(secondary_candidate) and secondary_candidate > 0.0:
                secondary_a = secondary_candidate
    except Exception:
        secondary_a = primary_a

    return gui_controllers.build_bragg_qr_entries(
        simulation_runtime_state,
        primary_a=primary_a,
        secondary_a=secondary_a,
    )


def _l_value_map_for_qr(source_label: str, m_idx: int) -> dict[int, float]:
    return gui_controllers.build_bragg_qr_l_value_map(
        simulation_runtime_state,
        source_label,
        int(m_idx),
    )


def _on_bragg_qr_selection_changed(_event=None) -> None:
    gui_bragg_qr_manager.on_bragg_qr_selection_changed(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        l_listbox=bragg_qr_manager_view_state.l_listbox,
        build_l_value_map=_l_value_map_for_qr,
        tcl_error_types=(tk.TclError,),
    )


def _refresh_bragg_qr_toggle_window() -> None:
    if gui_bragg_qr_manager.refresh_bragg_qr_toggle_window(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        entries=_build_bragg_qr_entries(),
        tcl_error_types=(tk.TclError,),
    ):
        _on_bragg_qr_selection_changed()


def _disable_selected_bragg_qr_groups() -> None:
    gui_bragg_qr_manager.disable_selected_bragg_qr_groups(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
        tcl_error_types=(tk.TclError,),
    )


def _enable_selected_bragg_qr_groups() -> None:
    gui_bragg_qr_manager.enable_selected_bragg_qr_groups(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
        tcl_error_types=(tk.TclError,),
    )


def _toggle_selected_bragg_qr_groups(_event=None) -> None:
    gui_bragg_qr_manager.toggle_selected_bragg_qr_groups(
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        tcl_error_types=(tk.TclError,),
    )


def _disable_selected_bragg_qr_l_values() -> None:
    gui_bragg_qr_manager.disable_selected_bragg_qr_l_values(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        l_listbox=bragg_qr_manager_view_state.l_listbox,
        invalid_key=BRAGG_QR_L_INVALID_KEY,
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
        tcl_error_types=(tk.TclError,),
    )


def _enable_selected_bragg_qr_l_values() -> None:
    gui_bragg_qr_manager.enable_selected_bragg_qr_l_values(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        l_listbox=bragg_qr_manager_view_state.l_listbox,
        invalid_key=BRAGG_QR_L_INVALID_KEY,
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
        tcl_error_types=(tk.TclError,),
    )


def _toggle_selected_bragg_qr_l_values(_event=None) -> None:
    gui_bragg_qr_manager.toggle_selected_bragg_qr_l_values(
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        l_listbox=bragg_qr_manager_view_state.l_listbox,
        invalid_key=BRAGG_QR_L_INVALID_KEY,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        tcl_error_types=(tk.TclError,),
    )


def _disable_all_bragg_qr_groups() -> None:
    gui_bragg_qr_manager.disable_all_bragg_qr_groups(
        manager_state=bragg_qr_manager_state,
        entries=_build_bragg_qr_entries(),
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
    )


def _enable_all_bragg_qr_groups() -> None:
    gui_bragg_qr_manager.enable_all_bragg_qr_groups(
        manager_state=bragg_qr_manager_state,
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
    )


def _disable_all_bragg_qr_l_values_for_selected_qr() -> None:
    gui_bragg_qr_manager.disable_all_bragg_qr_l_values_for_selected_qr(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        build_l_value_map=_l_value_map_for_qr,
        invalid_key=BRAGG_QR_L_INVALID_KEY,
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
        tcl_error_types=(tk.TclError,),
    )


def _enable_all_bragg_qr_l_values_for_selected_qr() -> None:
    gui_bragg_qr_manager.enable_all_bragg_qr_l_values_for_selected_qr(
        view_state=bragg_qr_manager_view_state,
        manager_state=bragg_qr_manager_state,
        qr_listbox=bragg_qr_manager_view_state.qr_listbox,
        refresh_window=_refresh_bragg_qr_toggle_window,
        apply_filters=lambda: _apply_bragg_qr_filters(trigger_update=True),
        set_progress_text=(
            (lambda text: progress_label_positions.config(text=text))
            if "progress_label_positions" in globals()
            else None
        ),
        tcl_error_types=(tk.TclError,),
    )


def _close_bragg_qr_toggle_window() -> None:
    gui_bragg_qr_manager.close_bragg_qr_toggle_window(
        bragg_qr_manager_view_state,
        bragg_qr_manager_state,
    )


def _open_bragg_qr_toggle_window() -> None:
    gui_bragg_qr_manager.open_bragg_qr_toggle_window(
        root=root,
        view_state=bragg_qr_manager_view_state,
        on_qr_selection_changed=_on_bragg_qr_selection_changed,
        on_toggle_qr=_toggle_selected_bragg_qr_groups,
        on_toggle_l=_toggle_selected_bragg_qr_l_values,
        on_enable_selected_qr=_enable_selected_bragg_qr_groups,
        on_disable_selected_qr=_disable_selected_bragg_qr_groups,
        on_enable_all_qr=_enable_all_bragg_qr_groups,
        on_disable_all_qr=_disable_all_bragg_qr_groups,
        on_enable_selected_l=_enable_selected_bragg_qr_l_values,
        on_disable_selected_l=_disable_selected_bragg_qr_l_values,
        on_enable_all_l=_enable_all_bragg_qr_l_values_for_selected_qr,
        on_disable_all_l=_disable_all_bragg_qr_l_values_for_selected_qr,
        on_refresh=_refresh_bragg_qr_toggle_window,
        on_close=_close_bragg_qr_toggle_window,
    )


def _brightest_hit_native_from_table(hit_table) -> tuple[float, float] | None:
    """Return (native_col, native_row) for the strongest valid hit row."""

    try:
        arr = np.asarray(hit_table, dtype=float)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None

    finite_mask = (
        np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    )
    if not np.any(finite_mask):
        return None

    valid = arr[finite_mask]
    best_idx = int(np.argmax(valid[:, 0]))
    return float(valid[best_idx, 1]), float(valid[best_idx, 2])


def _simulate_ideal_hkl_native_center(
    h: float,
    k: float,
    l: float,
    av: float,
    cv: float,
) -> tuple[float, float] | None:
    """Simulate a single HKL (integer or fractional) and return its ideal center."""

    optics_mode_flag = _current_optics_mode_flag()

    def _run_single(
        beam_x: np.ndarray,
        beam_y: np.ndarray,
        theta_arr: np.ndarray,
        phi_arr: np.ndarray,
        wavelength_arr: np.ndarray,
        *,
        forced_sample_indices: np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        image_buf = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
        miller_arr = np.array([[float(h), float(k), float(l)]], dtype=np.float64)
        intens_arr = np.array([1.0], dtype=np.float64)
        try:
            _image, hit_tables, *_ = process_peaks_parallel(
                miller_arr,
                intens_arr,
                int(image_size),
                float(av),
                float(cv),
                float(lambda_),
                image_buf,
                float(corto_detector_var.get()),
                float(gamma_var.get()),
                float(Gamma_var.get()),
                float(chi_var.get()),
                float(psi),
                float(psi_z_var.get()),
                float(zs_var.get()),
                float(zb_var.get()),
                n2,
                beam_x,
                beam_y,
                theta_arr,
                phi_arr,
                1e-6,  # near-zero mosaic spread (avoid singular zero width)
                1e-6,  # near-zero mosaic spread (avoid singular zero width)
                0.0,  # pure Gaussian in the zero-width limit
                wavelength_arr,
                float(debye_x_var.get()),
                float(debye_y_var.get()),
                [float(center_x_var.get()), float(center_y_var.get())],
                float(theta_initial_var.get()),
                float(cor_angle_var.get()),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                save_flag=0,
                record_status=False,
                optics_mode=optics_mode_flag,
                solve_q_steps=_current_solve_q_steps(),
                solve_q_rel_tol=_current_solve_q_rel_tol(),
                solve_q_mode=_current_solve_q_mode_flag(),
                single_sample_indices=forced_sample_indices,
            )
        except Exception:
            return None

        if hit_tables is None or len(hit_tables) == 0:
            return None
        return _brightest_hit_native_from_table(hit_tables[0])

    # First try the strict no-divergence/no-offset beam sample.
    strict_center = _run_single(
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([float(lambda_)], dtype=np.float64),
    )
    if strict_center is not None:
        return strict_center

    # Fallback: force the least-divergent sampled beam state from the profile.
    beam_x = np.asarray(simulation_runtime_state.profile_cache.get("beam_x_array", []), dtype=np.float64).ravel()
    beam_y = np.asarray(simulation_runtime_state.profile_cache.get("beam_y_array", []), dtype=np.float64).ravel()
    theta_arr = np.asarray(simulation_runtime_state.profile_cache.get("theta_array", []), dtype=np.float64).ravel()
    phi_arr = np.asarray(simulation_runtime_state.profile_cache.get("phi_array", []), dtype=np.float64).ravel()
    wavelength_arr = np.asarray(
        simulation_runtime_state.profile_cache.get("wavelength_array", []), dtype=np.float64
    ).ravel()

    n_samp = beam_x.size
    if (
        n_samp == 0
        or beam_y.size != n_samp
        or theta_arr.size != n_samp
        or phi_arr.size != n_samp
        or wavelength_arr.size != n_samp
    ):
        return None

    metric = theta_arr * theta_arr + phi_arr * phi_arr
    valid_metric = np.where(np.isfinite(metric), metric, np.inf)
    best_idx = int(np.argmin(valid_metric))
    if not np.isfinite(valid_metric[best_idx]):
        return None

    forced = np.array([best_idx], dtype=np.int64)
    return _run_single(
        beam_x,
        beam_y,
        theta_arr,
        phi_arr,
        wavelength_arr,
        forced_sample_indices=forced,
    )


def on_canvas_click(event):

    if event.button == 3 and peak_selection_state.hkl_pick_armed:
        _set_hkl_pick_mode(False, message="HKL image-pick canceled.")
        return
    if event.button == 3 and geometry_runtime_state.manual_pick_armed:
        _set_geometry_manual_pick_mode(False, message="Manual geometry picking canceled.")
        return
    if event.button == 3 and geometry_preview_state.exclude_armed:
        _set_geometry_preview_exclude_mode(
            False,
            message="Preview exclusion mode canceled.",
        )
        return

    if event.button != 1:
        return
    if geometry_runtime_state.manual_pick_armed:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        if _geometry_manual_pick_session_active():
            return
        _toggle_geometry_manual_selection_at(
            float(event.xdata),
            float(event.ydata),
        )
        return
    if analysis_view_controls_view_state.show_caked_2d_var.get():
        return
    if geometry_preview_state.exclude_armed:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        _toggle_live_geometry_preview_exclusion_at(
            float(event.xdata),
            float(event.ydata),
        )
        return
    if not peak_selection_state.hkl_pick_armed:
        return
    if event.inaxes is not ax or event.xdata is None or event.ydata is None:
        return                               # click was outside the image
    _ensure_peak_overlay_data(force=False)
    if not simulation_runtime_state.peak_positions:                   # no simulation yet
        schedule_update()
        progress_label_positions.config(
            text="Preparing simulated peak map... click again after update."
        )
        return

    click_col = float(event.xdata)
    click_row = float(event.ydata)

    # Find nearest stored peak
    best_i, best_d2 = -1, float("inf")
    best_i_val = float("-inf")
    for i, (px, py) in enumerate(simulation_runtime_state.peak_positions):
        if px < 0:              # invalid entries kept as (-1,-1)
            continue
        d2 = (px - click_col) ** 2 + (py - click_row) ** 2
        val = simulation_runtime_state.peak_intensities[i]
        score_val = float(val) if np.isfinite(val) else float("-inf")
        if d2 < best_d2 - 1e-9 or (abs(d2 - best_d2) <= 1e-9 and score_val > best_i_val):
            best_i, best_d2 = i, d2
            best_i_val = score_val

    if best_i == -1:
        progress_label_positions.config(text="No peaks on screen.")
        return
    if best_d2 > float(HKL_PICK_MAX_DISTANCE_PX) ** 2:
        progress_label_positions.config(
            text=(
                f"No simulated peak within {HKL_PICK_MAX_DISTANCE_PX:.0f}px "
                f"(nearest is {best_d2**0.5:.1f}px away)."
            )
        )
        return

    cx, cy = int(round(click_col)), int(round(click_row))
    sim_shape = global_image_buffer.shape if global_image_buffer.size else (image_size, image_size)
    clicked_native_col, clicked_native_row = _display_to_native_sim_coords(cx, cy, sim_shape)
    selected_display = None
    selected_native = None

    if best_i < len(simulation_runtime_state.peak_records) and isinstance(simulation_runtime_state.peak_records[best_i], dict):
        peak_record = simulation_runtime_state.peak_records[best_i]
        try:
            raw_hkl = peak_record.get("hkl_raw")
            if isinstance(raw_hkl, (list, tuple, np.ndarray)) and len(raw_hkl) >= 3:
                rec_h = float(raw_hkl[0])
                rec_k = float(raw_hkl[1])
                rec_l = float(raw_hkl[2])
            else:
                rec_h, rec_k, rec_l = tuple(float(v) for v in simulation_runtime_state.peak_millers[best_i])
            rec_av = float(peak_record.get("av", float(a_var.get())))
            rec_cv = float(peak_record.get("cv", float(c_var.get())))
            ideal_native = _simulate_ideal_hkl_native_center(
                rec_h, rec_k, rec_l, rec_av, rec_cv
            )
            if ideal_native is not None:
                ideal_display = _native_sim_to_display_coords(
                    ideal_native[0],
                    ideal_native[1],
                    sim_shape,
                )
                base_display = simulation_runtime_state.peak_positions[best_i]
                snap_delta = math.hypot(
                    float(ideal_display[0]) - float(base_display[0]),
                    float(ideal_display[1]) - float(base_display[1]),
                )
                snap_limit = max(4.0, float(HKL_PICK_MIN_SEPARATION_PX) * 2.0)
                if snap_delta <= snap_limit:
                    selected_native = ideal_native
                    selected_display = ideal_display
        except Exception:
            selected_display = None
            selected_native = None

    if selected_native is None and best_i < len(simulation_runtime_state.peak_records) and isinstance(simulation_runtime_state.peak_records[best_i], dict):
        selected_native = (
            float(simulation_runtime_state.peak_records[best_i].get("native_col", clicked_native_col)),
            float(simulation_runtime_state.peak_records[best_i].get("native_row", clicked_native_row)),
        )

    prefix = f"Nearest peak (Δ={best_d2**0.5:.1f}px)"
    if selected_display is not None:
        snapped_dist = math.hypot(
            float(selected_display[0]) - click_col,
            float(selected_display[1]) - click_row,
        )
        prefix = f"Ideal HKL center (click Δ={snapped_dist:.1f}px)"

    picked = _select_peak_by_index(
        best_i,
        prefix=prefix,
        sync_hkl_vars=True,
        clicked_display=(click_col, click_row),
        clicked_native=(float(clicked_native_col), float(clicked_native_row)),
        selected_display=selected_display,
        selected_native=selected_native,
    )
    if picked:
        _set_hkl_pick_mode(False)
        peak_selection_state.suppress_drag_press_once = True
        _sync_peak_selection_state()


drag_select_rect = Rectangle(
    (0.0, 0.0),
    0.0,
    0.0,
    linewidth=1.8,
    edgecolor='yellow',
    facecolor='none',
    linestyle='-',
    zorder=6,
)
drag_select_rect.set_visible(False)
ax.add_patch(drag_select_rect)

_drag_select_state = {
    "active": False,
    "mode": None,
    "x0": None,
    "y0": None,
    "x1": None,
    "y1": None,
    "tth0": None,
    "phi0": None,
    "tth1": None,
    "phi1": None,
}


def _clamp_to_axis_view(x, y):
    x_lo, x_hi = sorted(ax.get_xlim())
    y_lo, y_hi = sorted(ax.get_ylim())
    clamped_x = min(max(float(x), x_lo), x_hi)
    clamped_y = min(max(float(y), y_lo), y_hi)
    return clamped_x, clamped_y


def _update_drag_rect(x0, y0, x1, y1):
    x_min, x_max = sorted((float(x0), float(x1)))
    y_min, y_max = sorted((float(y0), float(y1)))
    drag_select_rect.set_xy((x_min, y_min))
    drag_select_rect.set_width(x_max - x_min)
    drag_select_rect.set_height(y_max - y_min)
    drag_select_rect.set_visible(True)
    canvas.draw_idle()


def _display_to_detector_angles(col: float, row: float, ai):
    two_theta, phi_vals = _get_detector_angular_maps(ai)
    if two_theta is None or phi_vals is None:
        return None

    height, width = two_theta.shape[:2]
    if height <= 0 or width <= 0:
        return None

    col_idx = min(max(int(round(float(col))), 0), width - 1)
    row_idx = min(max(int(round(float(row))), 0), height - 1)

    tth = float(two_theta[row_idx, col_idx])
    phi = float(phi_vals[row_idx, col_idx])
    if not (np.isfinite(tth) and np.isfinite(phi)):
        return None
    return tth, phi


def _update_raw_drag_preview(ai):
    tth0 = _drag_select_state["tth0"]
    phi0 = _drag_select_state["phi0"]
    tth1 = _drag_select_state["tth1"]
    phi1 = _drag_select_state["phi1"]
    if None in (tth0, phi0, tth1, phi1):
        return

    two_theta, phi_vals = _get_detector_angular_maps(ai)
    if two_theta is None or phi_vals is None:
        return

    tth_min, tth_max = sorted((float(tth0), float(tth1)))
    phi_min, phi_max = sorted((float(phi0), float(phi1)))
    mask = (
        (two_theta >= tth_min)
        & (two_theta <= tth_max)
        & (phi_vals >= phi_min)
        & (phi_vals <= phi_max)
    )

    drag_select_rect.set_visible(False)
    integration_region_rect.set_visible(False)
    if np.any(mask):
        integration_region_overlay.set_data(mask.astype(float))
        integration_region_overlay.set_extent(image_display.get_extent())
        integration_region_overlay.set_visible(True)
    else:
        integration_region_overlay.set_visible(False)
    canvas.draw_idle()


def _set_range_from_drag(x0, y0, x1, y1):
    tth_min, tth_max = sorted((float(x0), float(x1)))
    phi_min, phi_max = sorted((float(y0), float(y1)))

    tth_lo = float(tth_min_slider.cget("from"))
    tth_hi = float(tth_min_slider.cget("to"))
    phi_lo = float(phi_min_slider.cget("from"))
    phi_hi = float(phi_min_slider.cget("to"))
    tth_min = min(max(tth_min, min(tth_lo, tth_hi)), max(tth_lo, tth_hi))
    tth_max = min(max(tth_max, min(tth_lo, tth_hi)), max(tth_lo, tth_hi))
    phi_min = min(max(phi_min, min(phi_lo, phi_hi)), max(phi_lo, phi_hi))
    phi_max = min(max(phi_max, min(phi_lo, phi_hi)), max(phi_lo, phi_hi))

    if tth_max <= tth_min:
        eps = max(abs(tth_min) * 1e-6, 1e-3)
        tth_max = min(tth_min + eps, max(tth_lo, tth_hi))
    if phi_max <= phi_min:
        eps = max(abs(phi_min) * 1e-6, 1e-3)
        phi_max = min(phi_min + eps, max(phi_lo, phi_hi))

    tth_min_var.set(tth_min)
    tth_max_var.set(tth_max)
    phi_min_var.set(phi_min)
    phi_max_var.set(phi_max)

    schedule_range_update()
    progress_label_positions.config(
        text=(
            "Integration region set: "
            f"2θ=[{tth_min:.2f}, {tth_max:.2f}]°, "
            f"φ=[{phi_min:.2f}, {phi_max:.2f}]°"
        )
    )


def on_canvas_press(event):

    if peak_selection_state.suppress_drag_press_once:
        peak_selection_state.suppress_drag_press_once = False
        _sync_peak_selection_state()
        return

    if event.button != 1:
        return
    if event.inaxes is not ax or event.xdata is None or event.ydata is None:
        return
    if geometry_runtime_state.manual_pick_armed and _geometry_manual_pick_session_active():
        x0, y0 = _clamp_to_axis_view(event.xdata, event.ydata)
        anchor_fraction_x = 0.5
        anchor_fraction_y = 0.5
        try:
            bbox = getattr(ax, "bbox", None)
            if (
                bbox is not None
                and np.isfinite(float(bbox.width))
                and np.isfinite(float(bbox.height))
                and float(bbox.width) > 0.0
                and float(bbox.height) > 0.0
            ):
                anchor_fraction_x = float(
                    np.clip(
                        (float(event.x) - float(bbox.x0)) / float(bbox.width),
                        0.0,
                        1.0,
                    )
                )
                anchor_fraction_y = float(
                    np.clip(
                        (float(event.y) - float(bbox.y0)) / float(bbox.height),
                        0.0,
                        1.0,
                    )
                )
        except Exception:
            anchor_fraction_x = 0.5
            anchor_fraction_y = 0.5
        _apply_geometry_manual_pick_zoom(
            x0,
            y0,
            anchor_fraction_x=anchor_fraction_x,
            anchor_fraction_y=anchor_fraction_y,
        )
        _update_geometry_manual_pick_preview(x0, y0, force=True)
        return
    if simulation_runtime_state.unscaled_image is None:
        progress_label_positions.config(text="Run a simulation first.")
        return

    if analysis_view_controls_view_state.show_caked_2d_var.get():
        x0, y0 = _clamp_to_axis_view(event.xdata, event.ydata)
        _drag_select_state["active"] = True
        _drag_select_state["mode"] = "caked"
        _drag_select_state["x0"] = x0
        _drag_select_state["y0"] = y0
        _drag_select_state["x1"] = x0
        _drag_select_state["y1"] = y0
        _drag_select_state["tth0"] = None
        _drag_select_state["phi0"] = None
        _drag_select_state["tth1"] = None
        _drag_select_state["phi1"] = None
        _update_drag_rect(x0, y0, x0, y0)
        return

    if peak_selection_state.hkl_pick_armed:
        return

    ai = simulation_runtime_state.ai_cache.get("ai")
    if ai is None:
        return

    x0, y0 = _clamp_to_axis_view(event.xdata, event.ydata)
    angles = _display_to_detector_angles(x0, y0, ai)
    if angles is None:
        return
    tth0, phi0 = angles

    _drag_select_state["active"] = True
    _drag_select_state["mode"] = "raw"
    _drag_select_state["x0"] = x0
    _drag_select_state["y0"] = y0
    _drag_select_state["x1"] = x0
    _drag_select_state["y1"] = y0
    _drag_select_state["tth0"] = tth0
    _drag_select_state["phi0"] = phi0
    _drag_select_state["tth1"] = tth0
    _drag_select_state["phi1"] = phi0
    _update_raw_drag_preview(ai)


def on_canvas_motion(event):
    if (
        geometry_runtime_state.manual_pick_armed
        and _geometry_manual_pick_session_active()
        and bool(geometry_manual_state.pick_session.get("zoom_active", False))
    ):
        if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            x1, y1 = _clamp_to_axis_view(event.xdata, event.ydata)
            _update_geometry_manual_pick_preview(x1, y1)
        return

    if not _drag_select_state["active"]:
        return

    mode = _drag_select_state["mode"]
    if mode == "caked":
        if not analysis_view_controls_view_state.show_caked_2d_var.get():
            return

        if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            x1, y1 = _clamp_to_axis_view(event.xdata, event.ydata)
            _drag_select_state["x1"] = x1
            _drag_select_state["y1"] = y1
        else:
            x1 = _drag_select_state["x1"]
            y1 = _drag_select_state["y1"]
            if x1 is None or y1 is None:
                return

        _update_drag_rect(
            _drag_select_state["x0"],
            _drag_select_state["y0"],
            _drag_select_state["x1"],
            _drag_select_state["y1"],
        )
        return

    if mode != "raw" or analysis_view_controls_view_state.show_caked_2d_var.get():
        return

    ai = simulation_runtime_state.ai_cache.get("ai")
    if ai is None:
        return

    if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
        x1, y1 = _clamp_to_axis_view(event.xdata, event.ydata)
        angles = _display_to_detector_angles(x1, y1, ai)
        if angles is not None:
            tth1, phi1 = angles
            _drag_select_state["x1"] = x1
            _drag_select_state["y1"] = y1
            _drag_select_state["tth1"] = tth1
            _drag_select_state["phi1"] = phi1

    _update_raw_drag_preview(ai)


def on_canvas_release(event):
    if event.button != 1:
        return
    if (
        geometry_runtime_state.manual_pick_armed
        and _geometry_manual_pick_session_active()
        and bool(geometry_manual_state.pick_session.get("zoom_active", False))
    ):
        if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            x_sel, y_sel = _clamp_to_axis_view(event.xdata, event.ydata)
            _place_geometry_manual_selection_at(float(x_sel), float(y_sel))
        else:
            _clear_geometry_manual_preview_artists(redraw=False)
            _restore_geometry_manual_pick_view(redraw=False)
            _render_current_geometry_manual_pairs(update_status=False)
            progress_label_geometry.config(text="Manual point placement canceled: release inside the image.")
            canvas.draw_idle()
        return
    if not _drag_select_state["active"]:
        return

    mode = _drag_select_state["mode"]
    _drag_select_state["active"] = False

    if mode == "caked":
        if analysis_view_controls_view_state.show_caked_2d_var.get() and event.inaxes is ax and event.xdata is not None and event.ydata is not None:
            x1, y1 = _clamp_to_axis_view(event.xdata, event.ydata)
            _drag_select_state["x1"] = x1
            _drag_select_state["y1"] = y1

        x0 = _drag_select_state["x0"]
        y0 = _drag_select_state["y0"]
        x1 = _drag_select_state["x1"]
        y1 = _drag_select_state["y1"]
        if None not in (x0, y0, x1, y1):
            _set_range_from_drag(x0, y0, x1, y1)
        _drag_select_state["mode"] = None
        drag_select_rect.set_visible(False)
        canvas.draw_idle()
        return

    if mode == "raw":
        ai = simulation_runtime_state.ai_cache.get("ai")
        if (
            not analysis_view_controls_view_state.show_caked_2d_var.get()
            and ai is not None
            and event.inaxes is ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            x1, y1 = _clamp_to_axis_view(event.xdata, event.ydata)
            angles = _display_to_detector_angles(x1, y1, ai)
            if angles is not None:
                tth1, phi1 = angles
                _drag_select_state["x1"] = x1
                _drag_select_state["y1"] = y1
                _drag_select_state["tth1"] = tth1
                _drag_select_state["phi1"] = phi1

        tth0 = _drag_select_state["tth0"]
        phi0 = _drag_select_state["phi0"]
        tth1 = _drag_select_state["tth1"]
        phi1 = _drag_select_state["phi1"]
        if None not in (tth0, phi0, tth1, phi1):
            _set_range_from_drag(tth0, phi0, tth1, phi1)
        else:
            update_integration_region_visuals(ai, simulation_runtime_state.last_res2_sim)
            canvas.draw_idle()

    _drag_select_state["mode"] = None
    drag_select_rect.set_visible(False)
    canvas.draw_idle()

# -----------------------------------------------------------
# 3)  Bind the handler
# -----------------------------------------------------------
canvas.mpl_connect('button_press_event', on_canvas_click)
canvas.mpl_connect('button_press_event', on_canvas_press)
canvas.mpl_connect('motion_notify_event', on_canvas_motion)
canvas.mpl_connect('button_release_event', on_canvas_release)


# ---------------------------------------------------------------------------
# Display controls for background and simulation intensity scaling
# ---------------------------------------------------------------------------


def _finite_percentile(array, percentile, fallback):
    if array is None:
        return fallback
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    return float(np.nanpercentile(finite, percentile))


def _ensure_valid_range(min_val, max_val):
    return gui_controllers.ensure_display_intensity_range(min_val, max_val)


def _apply_background_transparency():
    background_transparency_var = display_controls_view_state.background_transparency_var
    if background_transparency_var is None:
        return
    transparency = max(0.0, min(1.0, background_transparency_var.get()))
    background_display.set_alpha(1.0 - transparency)


def _apply_background_limits():
    background_min_var = display_controls_view_state.background_min_var
    background_max_var = display_controls_view_state.background_max_var
    if background_min_var is None or background_max_var is None:
        return
    min_val = background_min_var.get()
    max_val = background_max_var.get()
    if min_val >= max_val:
        adjustment = max(abs(max_val) * 1e-6, 1e-6)
        display_controls_state.suppress_background_limit_callback = True
        background_min_var.set(max_val - adjustment)
        display_controls_state.suppress_background_limit_callback = False
        return
    display_controls_state.background_limits_user_override = True
    background_display.set_clim(min_val, max_val)
    _apply_background_transparency()
    canvas.draw_idle()


def _apply_simulation_limits():
    simulation_min_var = display_controls_view_state.simulation_min_var
    simulation_max_var = display_controls_view_state.simulation_max_var
    if simulation_min_var is None or simulation_max_var is None:
        return
    min_val = simulation_min_var.get()
    max_val = simulation_max_var.get()
    if min_val >= max_val:
        adjustment = max(abs(max_val) * 1e-6, 1e-6)
        display_controls_state.suppress_simulation_limit_callback = True
        simulation_min_var.set(max_val - adjustment)
        display_controls_state.suppress_simulation_limit_callback = False
        return
    display_controls_state.simulation_limits_user_override = True
    apply_scale_factor_to_existing_results()


background_min_candidate = _finite_percentile(background_runtime_state.current_background_display, 1, 0.0)
background_vmin_default = 0.0
_, background_vmax_default = _ensure_valid_range(
    background_vmin_default,
    _finite_percentile(background_runtime_state.current_background_display, 99, 1.0),
)

background_slider_min = min(background_min_candidate, 0.0)
background_slider_max = max(background_vmax_default * 5.0, background_slider_min + 1.0)
background_slider_step = max((background_slider_max - background_slider_min) / 500.0, 0.01)


simulation_slider_min = 0.0
simulation_slider_max = max(background_slider_max, defaults['vmax'] * 5.0)
# Ensure fine-grained control so the intensity sliders support at least 1e-4 precision.
simulation_slider_step = min(
    max((simulation_slider_max - simulation_slider_min) / 500.0, 1e-6),
    1e-4,
)

scale_factor_slider_min = 0.0
scale_factor_slider_max = 2.0
scale_factor_step = 0.0001

gui_views.create_display_controls(
    parent=app_shell_view_state.fig_frame,
    view_state=display_controls_view_state,
    background_range=(background_slider_min, background_slider_max),
    background_defaults=(background_vmin_default, background_vmax_default),
    background_step=background_slider_step,
    background_transparency=0.0,
    simulation_range=(simulation_slider_min, simulation_slider_max),
    simulation_defaults=(0.0, background_vmax_default),
    simulation_step=simulation_slider_step,
    scale_factor_range=(scale_factor_slider_min, scale_factor_slider_max),
    scale_factor_value=1.0,
    scale_factor_step=scale_factor_step,
    on_apply_background_limits=_apply_background_limits,
    on_apply_simulation_limits=_apply_simulation_limits,
)

background_display.set_clim(background_vmin_default, background_vmax_default)
_apply_background_transparency()


def _get_scale_factor_value(default=1.0):
    simulation_scale_factor_var = display_controls_view_state.simulation_scale_factor_var
    if simulation_scale_factor_var is None:
        return default
    try:
        scale = float(simulation_scale_factor_var.get())
    except (tk.TclError, ValueError):
        return default
    if not np.isfinite(scale):
        return default
    return gui_controllers.normalize_display_scale_factor(scale, fallback=1.0)


def _on_scale_factor_change(*args):
    if display_controls_state.suppress_scale_factor_callback:
        return
    if _get_scale_factor_value(default=None) is None:
        return
    display_controls_state.scale_factor_user_override = True
    apply_scale_factor_to_existing_results()


if display_controls_view_state.simulation_scale_factor_var is not None:
    display_controls_view_state.simulation_scale_factor_var.trace_add(
        "write", _on_scale_factor_change
    )


def _update_background_slider_defaults(image, reset_override=False):
    background_max_var = display_controls_view_state.background_max_var
    background_min_var = display_controls_view_state.background_min_var
    background_min_slider = display_controls_view_state.background_min_slider
    background_max_slider = display_controls_view_state.background_max_slider
    if image is None:
        return
    min_candidate = _finite_percentile(image, 1, 0.0)
    max_candidate = _finite_percentile(image, 99, background_max_var.get())
    min_candidate, max_candidate = _ensure_valid_range(min_candidate, max_candidate)
    slider_from = min(float(background_min_slider.cget("from")), min_candidate, 0.0)
    slider_to = max(float(background_min_slider.cget("to")), max_candidate, 1.0)
    background_min_slider.configure(from_=slider_from, to=slider_to)
    background_max_slider.configure(from_=slider_from, to=slider_to)
    display_controls_state.suppress_background_limit_callback = True
    if reset_override or not display_controls_state.background_limits_user_override:
        min_value = 0.0
        max_value = max_candidate
    else:
        min_value = float(background_min_var.get())
        max_value = float(background_max_var.get())
        min_value = min(max(min_value, slider_from), slider_to)
        max_value = min(max(max_value, slider_from), slider_to)
    min_value, max_value = _ensure_valid_range(min_value, max_value)
    background_min_var.set(min_value)
    background_max_var.set(max_value)
    display_controls_state.suppress_background_limit_callback = False
    background_display.set_clim(min_value, max_value)
    if reset_override:
        display_controls_state.background_limits_user_override = False


def _update_simulation_sliders_from_image(image, reset_override=False):
    simulation_min_var = display_controls_view_state.simulation_min_var
    simulation_max_var = display_controls_view_state.simulation_max_var
    simulation_min_slider = display_controls_view_state.simulation_min_slider
    simulation_max_slider = display_controls_view_state.simulation_max_slider
    if image is None or image.size == 0:
        return
    finite_pixels = np.asarray(image, dtype=float)
    finite_pixels = finite_pixels[np.isfinite(finite_pixels)]
    if finite_pixels.size == 0:
        return
    sim_min = float(np.min(finite_pixels))
    sim_max = float(np.max(finite_pixels))
    sim_min, sim_max = _ensure_valid_range(sim_min, sim_max)
    margin = 0.05 * max(abs(sim_max), 1.0)
    lower_bound = 0.0
    upper_bound = max(sim_max + margin, 1.0)
    slider_to = max(float(simulation_max_slider.cget("to")), upper_bound)
    simulation_min_slider.configure(from_=lower_bound, to=slider_to)
    simulation_max_slider.configure(from_=lower_bound, to=slider_to)
    slider_from = lower_bound
    display_controls_state.suppress_simulation_limit_callback = True
    if reset_override or not display_controls_state.simulation_limits_user_override:
        min_value = 0.0
        max_value = upper_bound
    else:
        min_value = float(simulation_min_var.get())
        max_value = float(simulation_max_var.get())
        min_value = min(max(min_value, slider_from), slider_to)
        max_value = min(max(max_value, slider_from), slider_to)
    min_value, max_value = _ensure_valid_range(min_value, max_value)
    simulation_min_var.set(min_value)
    simulation_max_var.set(max_value)
    display_controls_state.suppress_simulation_limit_callback = False
    if reset_override:
        display_controls_state.simulation_limits_user_override = False


def _set_scale_factor_value(value, adjust_range=True, reset_override=False):
    simulation_scale_factor_var = display_controls_view_state.simulation_scale_factor_var
    scale_factor_slider = display_controls_view_state.scale_factor_slider
    if simulation_scale_factor_var is None or scale_factor_slider is None:
        return
    value = gui_controllers.normalize_display_scale_factor(value, fallback=1.0)

    slider_min = float(scale_factor_slider.cget("from"))
    slider_max = float(scale_factor_slider.cget("to"))
    if adjust_range:
        new_min = slider_min
        new_max = slider_max
        if value < slider_min:
            new_min = value
        if value > slider_max:
            new_max = value
        if new_min != slider_min or new_max != slider_max:
            scale_factor_slider.configure(from_=new_min, to=new_max)
    else:
        value = min(max(value, slider_min), slider_max)

    display_controls_state.suppress_scale_factor_callback = True
    simulation_scale_factor_var.set(value)
    display_controls_state.suppress_scale_factor_callback = False
    if reset_override:
        display_controls_state.scale_factor_user_override = False
    else:
        display_controls_state.scale_factor_user_override = True


def _install_scale_factor_entry_bindings():
    scale_entry = display_controls_view_state.scale_factor_entry
    if scale_entry is None:
        return

    def _apply_scale_entry(_event=None):
        raw = scale_entry.get().strip()
        if not raw:
            current = _get_scale_factor_value(default=1.0)
            scale_entry.delete(0, tk.END)
            scale_entry.insert(0, f"{current:.6g}")
            return
        try:
            value = float(raw)
        except ValueError:
            current = _get_scale_factor_value(default=1.0)
            scale_entry.delete(0, tk.END)
            scale_entry.insert(0, f"{current:.6g}")
            return
        _set_scale_factor_value(value, adjust_range=True, reset_override=False)
        apply_scale_factor_to_existing_results(update_limits=False)

    scale_entry.unbind("<FocusOut>")
    scale_entry.unbind("<Return>")
    scale_entry.bind("<FocusOut>", _apply_scale_entry)
    scale_entry.bind("<Return>", _apply_scale_entry)


_install_scale_factor_entry_bindings()


def _suggest_scale_factor(sim_image, bg_image):
    sim_pixels = None if sim_image is None else np.asarray(sim_image, dtype=float)
    bg_pixels = None if bg_image is None else np.asarray(bg_image, dtype=float)
    if sim_pixels is None or bg_pixels is None:
        return 1.0

    # Percentiles over every pixel are expensive on full detector images.
    # Subsample finite values to keep this nearly constant-time.
    sim_pixels = sim_pixels[np.isfinite(sim_pixels)]
    bg_pixels = bg_pixels[np.isfinite(bg_pixels)]
    max_samples = 200_000
    if sim_pixels.size > max_samples:
        sim_step = max(1, sim_pixels.size // max_samples)
        sim_pixels = sim_pixels[::sim_step]
    if bg_pixels.size > max_samples:
        bg_step = max(1, bg_pixels.size // max_samples)
        bg_pixels = bg_pixels[::bg_step]

    if sim_pixels.size == 0 or bg_pixels.size == 0:
        return 1.0
    sim_reference_pixels = sim_pixels[sim_pixels > 0]
    if sim_reference_pixels.size == 0:
        sim_reference_pixels = np.abs(sim_pixels)
    bg_reference_pixels = bg_pixels[bg_pixels > 0]
    if bg_reference_pixels.size == 0:
        bg_reference_pixels = np.abs(bg_pixels)
    sim_ref = float(np.nanpercentile(sim_reference_pixels, 99))
    bg_ref = float(np.nanpercentile(bg_reference_pixels, 99))
    if not np.isfinite(sim_ref) or abs(sim_ref) < 1e-12:
        return 1.0
    if not np.isfinite(bg_ref) or abs(bg_ref) < 1e-12:
        return 1.0
    return bg_ref / sim_ref


def _mark_chi_square_dirty():
    simulation_runtime_state.chi_square_update_token = int(globals().get("chi_square_update_token", 0)) + 1


def _auto_match_scale_factor_to_radial_peak():
    sim_curve = simulation_runtime_state.last_1d_integration_data.get("intensities_2theta_sim")
    bg_curve = simulation_runtime_state.last_1d_integration_data.get("intensities_2theta_bg")

    if sim_curve is None or bg_curve is None:
        ai = simulation_runtime_state.ai_cache.get("ai")
        sim_img = simulation_runtime_state.last_1d_integration_data.get("simulated_2d_image")
        bg_img = _get_current_background_backend()
        if ai is not None and sim_img is not None and bg_img is not None:
            try:
                tth_min, tth_max = sorted((float(tth_min_var.get()), float(tth_max_var.get())))
                phi_min, phi_max = sorted((float(phi_min_var.get()), float(phi_max_var.get())))
                sim_res2 = caking(sim_img, ai)
                i2t_sim, _, _, _ = caked_up(
                    sim_res2,
                    tth_min,
                    tth_max,
                    phi_min,
                    phi_max,
                )
                bg_res2 = caking(bg_img, ai)
                i2t_bg, _, _, _ = caked_up(
                    bg_res2,
                    tth_min,
                    tth_max,
                    phi_min,
                    phi_max,
                )
                sim_curve = i2t_sim
                bg_curve = i2t_bg
                simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] = i2t_sim
                simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = i2t_bg
            except Exception:
                sim_curve = None
                bg_curve = None
        if sim_curve is None or bg_curve is None:
            progress_label_positions.config(
                text=(
                    "Auto-match requires background + simulation radial curves. "
                    "Enable background and run/update simulation once."
                )
            )
            return

    sim_vals = np.asarray(sim_curve, dtype=float)
    bg_vals = np.asarray(bg_curve, dtype=float)
    sim_vals = sim_vals[np.isfinite(sim_vals)]
    bg_vals = bg_vals[np.isfinite(bg_vals)]
    if sim_vals.size == 0 or bg_vals.size == 0:
        progress_label_positions.config(text="Auto-match failed: radial curves are empty.")
        return

    sim_peak = float(np.max(sim_vals))
    bg_peak = float(np.max(bg_vals))
    if not np.isfinite(sim_peak) or sim_peak <= 0.0:
        progress_label_positions.config(
            text="Auto-match failed: simulated radial peak is non-positive."
        )
        return
    if not np.isfinite(bg_peak) or bg_peak < 0.0:
        progress_label_positions.config(
            text="Auto-match failed: background radial peak is invalid."
        )
        return

    target_scale = bg_peak / sim_peak
    _set_scale_factor_value(target_scale, adjust_range=True, reset_override=False)
    apply_scale_factor_to_existing_results(update_limits=False)
    progress_label_positions.config(
        text=(
            f"Auto-matched scale factor to radial peak: {target_scale:.6g} "
            f"(sim max={sim_peak:.6g}, bg max={bg_peak:.6g})."
        )
    )


ttk.Button(
    display_controls_view_state.simulation_controls_frame,
    text="Auto-Match Scale (Radial Peak)",
    command=_auto_match_scale_factor_to_radial_peak,
).pack(anchor=tk.W, padx=5, pady=(0, 6))


def _update_chi_square_display(force=False):
    state = globals().setdefault(
        "chi_square_state",
        {
            "last_ts": 0.0,
            "last_token": -1,
            "last_text": "Chi-Squared: N/A",
        },
    )
    now = perf_counter()
    # Throttle to avoid repeatedly scanning full detector arrays during slider
    # drags; keep the last displayed value between refreshes.
    if (not force) and (
        (now - float(state.get("last_ts", 0.0))) < CHI_SQUARE_UPDATE_INTERVAL_S
    ):
        last_text = str(state.get("last_text", "Chi-Squared: N/A"))
        chi_square_label.config(text=last_text)
        return

    token = int(globals().get("chi_square_update_token", 0))
    if (not force) and token == int(state.get("last_token", -1)):
        last_text = str(state.get("last_text", "Chi-Squared: N/A"))
        chi_square_label.config(text=last_text)
        state["last_ts"] = now
        return

    try:
        native_background = _get_current_background_backend()
        text = "Chi-Squared: N/A"
        if (
            background_runtime_state.visible
            and native_background is not None
            and global_image_buffer.size
        ):
            sim_vals = np.asarray(global_image_buffer, dtype=float)
            bg_vals = np.asarray(native_background, dtype=float)
            if (
                sim_vals.size
                and bg_vals.size
                and np.max(sim_vals) > 0
                and np.max(bg_vals) > 0
                and sim_vals.shape == bg_vals.shape
            ):
                # Subsample large arrays for responsive UI updates.
                max_points = 250_000
                if sim_vals.ndim == 2 and sim_vals.size > max_points:
                    stride = int(max(1, np.sqrt(sim_vals.size / float(max_points))))
                    sim_vals = sim_vals[::stride, ::stride]
                    bg_vals = bg_vals[::stride, ::stride]
                norm_sim = sim_vals / np.max(sim_vals)
                norm_bg = bg_vals / np.max(bg_vals)
                chi_sq_val = float(np.mean(np.square(norm_bg - norm_sim)) * norm_sim.size)
                text = f"Chi-Squared: {chi_sq_val:.2e}"
    except Exception as exc:
        text = f"Chi-Squared Error: {exc}"

    state["last_ts"] = now
    state["last_token"] = token
    state["last_text"] = text
    chi_square_label.config(text=text)


def apply_scale_factor_to_existing_results(update_limits=False):
    chi_state = globals().setdefault(
        "chi_square_state",
        {
            "last_ts": 0.0,
            "last_token": -1,
            "last_text": "Chi-Squared: N/A",
        },
    )
    if simulation_runtime_state.unscaled_image is None:
        background_min_var = display_controls_view_state.background_min_var
        background_max_var = display_controls_view_state.background_max_var
        chi_square_sig = (
            None,
            bool(background_runtime_state.visible),
            int(background_runtime_state.current_background_index),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
        )
        if chi_state.get("buffer_sig") != chi_square_sig:
            chi_state["buffer_sig"] = chi_square_sig
            _mark_chi_square_dirty()
        background_display.set_clim(
            background_min_var.get(),
            background_max_var.get(),
        )
        if not analysis_view_controls_view_state.show_caked_2d_var.get():
            if background_runtime_state.visible and background_runtime_state.current_background_display is not None:
                background_display.set_data(background_runtime_state.current_background_display)
                background_display.set_visible(True)
            else:
                background_display.set_visible(False)
        canvas.draw_idle()
        _update_chi_square_display()
        return

    scale = _get_scale_factor_value(default=1.0)
    if abs(float(scale) - 1.0) <= 1e-12:
        np.copyto(global_image_buffer, simulation_runtime_state.unscaled_image, casting="unsafe")
    else:
        np.multiply(simulation_runtime_state.unscaled_image, float(scale), out=global_image_buffer)
    scaled_image = global_image_buffer
    base_unscaled_sig = globals().get("last_unscaled_image_signature")
    chi_square_sig = (
        base_unscaled_sig,
        round(float(scale), 9),
        bool(background_runtime_state.visible),
        int(background_runtime_state.current_background_index),
        int(background_runtime_state.backend_rotation_k) % 4,
        bool(background_runtime_state.backend_flip_x),
        bool(background_runtime_state.backend_flip_y),
    )
    if chi_state.get("buffer_sig") != chi_square_sig:
        chi_state["buffer_sig"] = chi_square_sig
        _mark_chi_square_dirty()

    # Keep display limits stable during reruns/parameter changes unless an
    # explicit reset path requests recomputing limits.
    if update_limits:
        _update_simulation_sliders_from_image(
            scaled_image, reset_override=update_limits
        )

    if not analysis_view_controls_view_state.show_caked_2d_var.get():
        _set_image_origin(image_display, 'upper')
        image_display.set_extent([0, image_size, image_size, 0])
        image_display.set_data(global_image_buffer)
        image_display.set_clim(
            display_controls_view_state.simulation_min_var.get(),
            display_controls_view_state.simulation_max_var.get(),
        )
        colorbar_main.update_normal(image_display)
    elif simulation_runtime_state.last_caked_image_unscaled is not None and simulation_runtime_state.last_caked_extent is not None:
        _set_image_origin(image_display, 'lower')
        scaled_caked = simulation_runtime_state.last_caked_image_unscaled * scale
        if not display_controls_state.simulation_limits_user_override:
            _update_simulation_sliders_from_image(scaled_caked, reset_override=True)
        image_display.set_data(scaled_caked)
        image_display.set_extent(simulation_runtime_state.last_caked_extent)

    if analysis_view_controls_view_state.show_caked_2d_var.get():
        caked_min = float(display_controls_view_state.simulation_min_var.get())
        caked_max = float(display_controls_view_state.simulation_max_var.get())
        caked_min, caked_max = _ensure_valid_range(caked_min, caked_max)
        image_display.set_clim(caked_min, caked_max)
        if not simulation_runtime_state.caked_limits_user_override:
            vmin_caked_var.set(caked_min)
            vmax_caked_var.set(caked_max)

    if (
        analysis_view_controls_view_state.show_1d_var.get()
        and "line_1d_rad" in globals()
        and "line_1d_az" in globals()
    ):
        if (
            simulation_runtime_state.last_1d_integration_data["radials_sim"] is not None
            and simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] is not None
        ):
            line_1d_rad.set_data(
                simulation_runtime_state.last_1d_integration_data["radials_sim"],
                simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] * scale,
            )
        if (
            simulation_runtime_state.last_1d_integration_data["azimuths_sim"] is not None
            and simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"] is not None
        ):
            line_1d_az.set_data(
                simulation_runtime_state.last_1d_integration_data["azimuths_sim"],
                simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"] * scale,
            )
        if "canvas_1d" in globals():
            canvas_1d.draw_idle()

    background_display.set_clim(
        display_controls_view_state.background_min_var.get(),
        display_controls_view_state.background_max_var.get(),
    )

    if not analysis_view_controls_view_state.show_caked_2d_var.get():
        if background_runtime_state.visible and background_runtime_state.current_background_display is not None:
            background_display.set_data(background_runtime_state.current_background_display)
            background_display.set_visible(True)
        else:
            background_display.set_visible(False)

    canvas.draw_idle()
    _update_chi_square_display()
_update_background_slider_defaults(background_runtime_state.current_background_display, reset_override=True)


# Track caked intensity limits without exposing separate sliders in the UI.
simulation_runtime_state.caked_limits_user_override = False

vmin_caked_var = tk.DoubleVar(value=0.0)
vmax_caked_var = tk.DoubleVar(value=2000.0)

fig_1d, (ax_1d_radial, ax_1d_azim) = plt.subplots(2, 1, figsize=(5, 8))
canvas_1d = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
    fig_1d,
    master=app_shell_view_state.plot_frame_1d,
)
canvas_1d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

line_1d_rad, = ax_1d_radial.plot([], [], 'b-', label='Simulated (2θ)')
line_1d_rad_bg, = ax_1d_radial.plot([], [], 'r--', label='Background (2θ)')
ax_1d_radial.legend()
ax_1d_radial.set_xlabel('2θ (degrees)')
ax_1d_radial.set_ylabel('Intensity')
ax_1d_radial.set_title('Radial Integration (2θ)')

line_1d_az, = ax_1d_azim.plot([], [], 'b-', label='Simulated (φ)')
line_1d_az_bg, = ax_1d_azim.plot([], [], 'r--', label='Background (φ)')
ax_1d_azim.legend()
ax_1d_azim.set_xlabel('Azimuth (degrees)')
ax_1d_azim.set_ylabel('Intensity')
ax_1d_azim.set_title('Azimuthal Integration (φ)')

canvas_1d.draw()

def _sync_range_text_vars():
    range_view = integration_range_controls_view_state
    specs = (
        (range_view.tth_min_var, range_view.tth_min_label_var, range_view.tth_min_entry_var),
        (range_view.tth_max_var, range_view.tth_max_label_var, range_view.tth_max_entry_var),
        (range_view.phi_min_var, range_view.phi_min_label_var, range_view.phi_min_entry_var),
        (range_view.phi_max_var, range_view.phi_max_label_var, range_view.phi_max_entry_var),
    )
    if any(
        value_var is None or label_var is None or entry_var is None
        for value_var, label_var, entry_var in specs
    ):
        return
    for value_var, label_var, entry_var in specs:
        label_var.set(f"{value_var.get():.1f}")
        entry_var.set(f"{value_var.get():.4f}")


def _apply_range_entry(entry_var, value_var, slider):
    try:
        entered = float(entry_var.get().strip())
    except (ValueError, tk.TclError, AttributeError):
        _sync_range_text_vars()
        return

    lo = float(slider.cget("from"))
    hi = float(slider.cget("to"))
    clamped = min(max(entered, min(lo, hi)), max(lo, hi))
    value_var.set(clamped)
    _sync_range_text_vars()
    schedule_range_update()


def _on_range_var_write(*_args):
    _sync_range_text_vars()


def tth_min_slider_command(val):
    value_var = integration_range_controls_view_state.tth_min_var
    if value_var is None:
        return
    val_f = float(val)
    value_var.set(val_f)
    _sync_range_text_vars()
    schedule_range_update()

def tth_max_slider_command(val):
    value_var = integration_range_controls_view_state.tth_max_var
    if value_var is None:
        return
    val_f = float(val)
    value_var.set(val_f)
    _sync_range_text_vars()
    schedule_range_update()

def phi_min_slider_command(val):
    value_var = integration_range_controls_view_state.phi_min_var
    if value_var is None:
        return
    val_f = float(val)
    value_var.set(val_f)
    _sync_range_text_vars()
    schedule_range_update()

def phi_max_slider_command(val):
    value_var = integration_range_controls_view_state.phi_max_var
    if value_var is None:
        return
    val_f = float(val)
    value_var.set(val_f)
    _sync_range_text_vars()
    schedule_range_update()

gui_views.create_integration_range_controls(
    parent=app_shell_view_state.plot_frame_1d,
    view_state=integration_range_controls_view_state,
    tth_min=0.0,
    tth_max=60.0,
    phi_min=-15.0,
    phi_max=15.0,
    on_tth_min_changed=tth_min_slider_command,
    on_tth_max_changed=tth_max_slider_command,
    on_phi_min_changed=phi_min_slider_command,
    on_phi_max_changed=phi_max_slider_command,
    on_apply_entry=_apply_range_entry,
)
tth_min_var = integration_range_controls_view_state.tth_min_var
tth_max_var = integration_range_controls_view_state.tth_max_var
phi_min_var = integration_range_controls_view_state.phi_min_var
phi_max_var = integration_range_controls_view_state.phi_max_var
tth_min_slider = integration_range_controls_view_state.tth_min_slider
tth_max_slider = integration_range_controls_view_state.tth_max_slider
phi_min_slider = integration_range_controls_view_state.phi_min_slider
phi_max_slider = integration_range_controls_view_state.phi_max_slider
if any(
    ref is None
    for ref in (
        tth_min_var,
        tth_max_var,
        phi_min_var,
        phi_max_var,
        tth_min_slider,
        tth_max_slider,
        phi_min_slider,
        phi_max_slider,
    )
):
    raise RuntimeError("Integration-range controls did not create the expected widgets.")

for _var in (tth_min_var, tth_max_var, phi_min_var, phi_max_var):
    _var.trace_add("write", _on_range_var_write)

_sync_range_text_vars()

PHI_ZERO_OFFSET_DEGREES = -90.0


def _adjust_phi_zero(phi_values):
    """Center azimuths at ``PHI_ZERO_OFFSET_DEGREES`` and mirror about the x-axis."""

    return PHI_ZERO_OFFSET_DEGREES - np.asarray(phi_values)


def _wrap_phi_range(phi_values):
    """Wrap azimuthal values into the ``[-180, 180)`` interval."""

    wrapped = ((np.asarray(phi_values) + 180.0) % 360.0) - 180.0
    return wrapped


def caking(data, ai):
    return ai.integrate2d(
        data,
        npt_rad=1000,
        npt_azim=720,
        correctSolidAngle=True,
        method="lut",
        unit="2th_deg"
    )


def _auto_caked_limits(image):
    """Return sensible display limits for a caked image."""

    if image is None:
        return 0.0, 1.0

    finite_mask = np.isfinite(image)
    if not np.any(finite_mask):
        return 0.0, 1.0

    finite_vals = image[finite_mask]
    vmin = float(np.nanmin(finite_vals))
    vmax = float(np.nanmax(finite_vals))

    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return 0.0, 1.0

    if math.isclose(vmin, vmax):
        if vmin == 0.0:
            vmax = 1.0
        else:
            spread = abs(vmax) * 1e-3 or 1.0
            vmin -= spread
            vmax += spread

    return vmin, vmax


def _set_image_origin(image_display, origin):
    """Set the origin for an AxesImage while tolerating older Matplotlib APIs."""

    setter = getattr(image_display, "set_origin", None)
    if callable(setter):
        try:
            setter(origin)
            return
        except AttributeError:
            # Older Matplotlib builds may expose ``set_origin`` but still raise
            # AttributeError when called; fall back to setting the attribute
            # directly below.
            pass

    # Some Matplotlib releases exposed the origin as a simple attribute.
    try:
        image_display.origin = origin
    except AttributeError:
        # If the fallback attribute assignment also fails, there's not much
        # else we can do; let the caller continue without crashing.
        return


def caked_up(res2, tth_min, tth_max, phi_min, phi_max):
    intensity = res2.intensity
    radial_2theta = res2.radial
    azimuth_vals = _adjust_phi_zero(res2.azimuthal)

    tth_min, tth_max = sorted((float(tth_min), float(tth_max)))
    phi_min, phi_max = sorted((float(phi_min), float(phi_max)))

    mask_rad = (radial_2theta >= tth_min) & (radial_2theta <= tth_max)
    radial_filtered = radial_2theta[mask_rad]

    mask_az = (azimuth_vals >= phi_min) & (azimuth_vals <= phi_max)
    azimuth_sub = azimuth_vals[mask_az]

    intensity_sub = intensity[np.ix_(mask_az, mask_rad)]
    intensity_vs_2theta = np.sum(intensity_sub, axis=0)
    intensity_vs_phi = np.sum(intensity_sub, axis=1)

    return intensity_vs_2theta, intensity_vs_phi, azimuth_sub, radial_filtered


def _get_detector_angular_maps(ai):
    if ai is None:
        return None, None

    detector_shape = getattr(global_image_buffer, "shape", None)
    if detector_shape is None or len(detector_shape) < 2:
        return None, None
    if detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None, None

    if simulation_runtime_state.ai_cache.get("detector_shape") != detector_shape:
        try:
            two_theta = ai.twoThetaArray(shape=detector_shape, unit="2th_deg")
        except TypeError:
            two_theta = np.rad2deg(ai.twoThetaArray(shape=detector_shape))

        try:
            phi_vals = ai.chiArray(shape=detector_shape, unit="deg")
        except TypeError:
            phi_vals = np.rad2deg(ai.chiArray(shape=detector_shape))

        simulation_runtime_state.ai_cache["detector_shape"] = detector_shape
        simulation_runtime_state.ai_cache["detector_two_theta"] = two_theta
        simulation_runtime_state.ai_cache["detector_phi"] = phi_vals

    two_theta = simulation_runtime_state.ai_cache.get("detector_two_theta")
    phi_vals = simulation_runtime_state.ai_cache.get("detector_phi")
    if two_theta is None or phi_vals is None:
        return None, None

    return two_theta, _adjust_phi_zero(phi_vals)


def update_integration_region_visuals(ai, sim_res2):
    show_region = analysis_view_controls_view_state.show_1d_var.get() and simulation_runtime_state.unscaled_image is not None

    if not show_region:
        integration_region_overlay.set_visible(False)
        integration_region_rect.set_visible(False)
        return

    tth_values = sorted((tth_min_var.get(), tth_max_var.get()))
    phi_values = sorted((phi_min_var.get(), phi_max_var.get()))
    tth_min, tth_max = tth_values
    phi_min, phi_max = phi_values

    if analysis_view_controls_view_state.show_caked_2d_var.get() and sim_res2 is not None:
        integration_region_overlay.set_visible(False)
        integration_region_rect.set_xy((tth_min, phi_min))
        integration_region_rect.set_width(tth_max - tth_min)
        integration_region_rect.set_height(phi_max - phi_min)
        integration_region_rect.set_visible(True)
        return

    integration_region_rect.set_visible(False)

    if ai is None:
        integration_region_overlay.set_visible(False)
        return

    two_theta, phi_vals = _get_detector_angular_maps(ai)
    if two_theta is None or phi_vals is None:
        integration_region_overlay.set_visible(False)
        return

    mask = (
        (two_theta >= tth_min)
        & (two_theta <= tth_max)
        & (phi_vals >= phi_min)
        & (phi_vals <= phi_max)
    )

    if not np.any(mask):
        integration_region_overlay.set_visible(False)
        return

    integration_region_overlay.set_data(mask.astype(float))
    integration_region_overlay.set_extent(image_display.get_extent())
    integration_region_overlay.set_visible(True)


simulation_runtime_state.profile_cache = {}
simulation_runtime_state.last_1d_integration_data = {
    "radials_sim": None,
    "intensities_2theta_sim": None,
    "azimuths_sim": None,
    "intensities_azimuth_sim": None,
    "radials_bg": None,
    "intensities_2theta_bg": None,
    "azimuths_bg": None,
    "intensities_azimuth_bg": None,
    "simulated_2d_image": None
}

simulation_runtime_state.last_caked_image_unscaled = None
simulation_runtime_state.last_caked_extent = None
simulation_runtime_state.last_caked_background_image_unscaled = None
simulation_runtime_state.last_caked_radial_values = None
simulation_runtime_state.last_caked_azimuth_values = None

simulation_runtime_state.last_res2_background = None
simulation_runtime_state.last_res2_sim = None
simulation_runtime_state.ai_cache = {}


def _clear_1d_plot_cache_and_lines():
    line_1d_rad.set_data([], [])
    line_1d_az.set_data([], [])
    line_1d_rad_bg.set_data([], [])
    line_1d_az_bg.set_data([], [])
    canvas_1d.draw_idle()
    simulation_runtime_state.last_1d_integration_data["radials_sim"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] = None
    simulation_runtime_state.last_1d_integration_data["azimuths_sim"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"] = None
    simulation_runtime_state.last_1d_integration_data["radials_bg"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = None
    simulation_runtime_state.last_1d_integration_data["azimuths_bg"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_azimuth_bg"] = None


def _update_1d_plots_from_caked(sim_res2, bg_res2):
    i2t_sim, i_phi_sim, az_sim, rad_sim = caked_up(
        sim_res2,
        tth_min_var.get(),
        tth_max_var.get(),
        phi_min_var.get(),
        phi_max_var.get(),
    )
    simulation_runtime_state.last_1d_integration_data["radials_sim"] = rad_sim
    simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] = i2t_sim
    simulation_runtime_state.last_1d_integration_data["azimuths_sim"] = az_sim
    simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"] = i_phi_sim

    scale = _get_scale_factor_value(default=1.0)
    line_1d_rad.set_data(rad_sim, i2t_sim * scale)
    line_1d_az.set_data(az_sim, i_phi_sim * scale)

    if bg_res2 is not None:
        i2t_bg, i_phi_bg, az_bg, rad_bg = caked_up(
            bg_res2,
            tth_min_var.get(),
            tth_max_var.get(),
            phi_min_var.get(),
            phi_max_var.get(),
        )
        simulation_runtime_state.last_1d_integration_data["radials_bg"] = rad_bg
        simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = i2t_bg
        simulation_runtime_state.last_1d_integration_data["azimuths_bg"] = az_bg
        simulation_runtime_state.last_1d_integration_data["intensities_azimuth_bg"] = i_phi_bg
        line_1d_rad_bg.set_data(rad_bg, i2t_bg)
        line_1d_az_bg.set_data(az_bg, i_phi_bg)
    else:
        line_1d_rad_bg.set_data([], [])
        line_1d_az_bg.set_data([], [])
        simulation_runtime_state.last_1d_integration_data["radials_bg"] = None
        simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = None
        simulation_runtime_state.last_1d_integration_data["azimuths_bg"] = None
        simulation_runtime_state.last_1d_integration_data["intensities_azimuth_bg"] = None

    ax_1d_radial.set_yscale('log' if analysis_view_controls_view_state.log_radial_var.get() else 'linear')
    ax_1d_azim.set_yscale('log' if analysis_view_controls_view_state.log_azimuth_var.get() else 'linear')
    ax_1d_radial.relim()
    ax_1d_radial.autoscale_view()
    ax_1d_azim.relim()
    ax_1d_azim.autoscale_view()
    canvas_1d.draw_idle()


def _refresh_integration_from_cached_results():

    ai = simulation_runtime_state.ai_cache.get("ai")
    if not analysis_view_controls_view_state.show_1d_var.get():
        _clear_1d_plot_cache_and_lines()
        update_integration_region_visuals(ai, simulation_runtime_state.last_res2_sim)
        canvas.draw_idle()
        return True

    if simulation_runtime_state.unscaled_image is None:
        return False

    if ai is None:
        return False

    if simulation_runtime_state.last_res2_sim is None:
        simulation_runtime_state.last_res2_sim = caking(simulation_runtime_state.unscaled_image, ai)

    bg_res2 = None
    native_background = _get_current_background_backend()
    if background_runtime_state.visible and native_background is not None:
        if simulation_runtime_state.last_res2_background is None:
            simulation_runtime_state.last_res2_background = caking(native_background, ai)
        bg_res2 = simulation_runtime_state.last_res2_background
    else:
        simulation_runtime_state.last_res2_background = None

    _update_1d_plots_from_caked(simulation_runtime_state.last_res2_sim, bg_res2)
    update_integration_region_visuals(ai, simulation_runtime_state.last_res2_sim)
    canvas.draw_idle()
    return True


def _clip_bandwidth_percent(value) -> float:
    fallback = float(defaults.get("bandwidth_percent", bandwidth * 100.0))
    try:
        bw_percent = float(value)
    except (TypeError, ValueError, tk.TclError):
        bw_percent = fallback
    if not np.isfinite(bw_percent):
        bw_percent = fallback
    return float(np.clip(bw_percent, 0.0, 10.0))


def _current_bandwidth_fraction() -> float:
    var = globals().get("bandwidth_percent_var")
    if var is None:
        return float(max(0.0, bandwidth))
    try:
        bw_percent = var.get()
    except Exception:
        bw_percent = defaults.get("bandwidth_percent", bandwidth * 100.0)
    return _clip_bandwidth_percent(bw_percent) / 100.0


def _clip_solve_q_steps(value) -> int:
    return gui_controllers.clip_solve_q_steps(
        value,
        fallback=defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
        minimum=MIN_SOLVE_Q_STEPS,
        maximum=MAX_SOLVE_Q_STEPS,
    )


def _current_solve_q_steps() -> int:
    view_state = globals().get("structure_factor_pruning_controls_view_state")
    var = getattr(view_state, "solve_q_steps_var", None)
    if var is None:
        return _clip_solve_q_steps(defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS))
    try:
        raw = var.get()
    except Exception:
        raw = defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS)
    return _clip_solve_q_steps(raw)


def _clip_solve_q_rel_tol(value) -> float:
    return gui_controllers.clip_solve_q_rel_tol(
        value,
        fallback=defaults.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL),
        minimum=MIN_SOLVE_Q_REL_TOL,
        maximum=MAX_SOLVE_Q_REL_TOL,
    )


def _current_solve_q_rel_tol() -> float:
    view_state = globals().get("structure_factor_pruning_controls_view_state")
    var = getattr(view_state, "solve_q_rel_tol_var", None)
    if var is None:
        return _clip_solve_q_rel_tol(defaults.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL))
    try:
        raw = var.get()
    except Exception:
        raw = defaults.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL)
    return _clip_solve_q_rel_tol(raw)


def _normalize_solve_q_mode_label(value) -> str:
    return gui_controllers.normalize_solve_q_mode_label(value)


def _solve_q_mode_flag_from_label(label: str) -> int:
    return gui_controllers.solve_q_mode_flag_from_label(
        label,
        uniform_flag=SOLVE_Q_MODE_UNIFORM,
        adaptive_flag=SOLVE_Q_MODE_ADAPTIVE,
    )


def _current_solve_q_mode_flag() -> int:
    view_state = globals().get("structure_factor_pruning_controls_view_state")
    var = getattr(view_state, "solve_q_mode_var", None)
    if var is None:
        return _solve_q_mode_flag_from_label(defaults.get("solve_q_mode", SOLVE_Q_MODE_UNIFORM))
    try:
        raw = var.get()
    except Exception:
        raw = defaults.get("solve_q_mode", SOLVE_Q_MODE_UNIFORM)
    return _solve_q_mode_flag_from_label(raw)


def update_mosaic_cache():
    """
    Keep the current random beam/mosaic samples unless sampling inputs changed.

    This preserves the same sampled beam positions/divergence across normal
    simulation updates so changing unrelated sliders does not re-randomize the
    detector pattern.
    """
    active_bandwidth = _current_bandwidth_fraction()
    sampling_signature = (
        int(simulation_runtime_state.num_samples),
        float(divergence_sigma),
        float(bw_sigma),
        float(lambda_),
        float(active_bandwidth),
    )

    beam_x_cached = np.asarray(simulation_runtime_state.profile_cache.get("beam_x_array", []), dtype=np.float64).ravel()
    beam_y_cached = np.asarray(simulation_runtime_state.profile_cache.get("beam_y_array", []), dtype=np.float64).ravel()
    theta_cached = np.asarray(simulation_runtime_state.profile_cache.get("theta_array", []), dtype=np.float64).ravel()
    phi_cached = np.asarray(simulation_runtime_state.profile_cache.get("phi_array", []), dtype=np.float64).ravel()
    wavelength_cached = np.asarray(simulation_runtime_state.profile_cache.get("wavelength_array", []), dtype=np.float64).ravel()

    has_cached_samples = (
        beam_x_cached.size > 0
        and beam_x_cached.size == int(simulation_runtime_state.num_samples)
        and beam_y_cached.size == beam_x_cached.size
        and theta_cached.size == beam_x_cached.size
        and phi_cached.size == beam_x_cached.size
        and wavelength_cached.size == beam_x_cached.size
    )

    cached_signature = simulation_runtime_state.profile_cache.get("_sampling_signature")
    should_resample = not has_cached_samples
    if has_cached_samples:
        if cached_signature is None:
            # Existing externally-provided samples: keep them and adopt the
            # current signature so future explicit sampling-input changes can
            # trigger a re-sample.
            simulation_runtime_state.profile_cache["_sampling_signature"] = sampling_signature
        else:
            should_resample = tuple(cached_signature) != sampling_signature

    if should_resample:
        (beam_x_array,
         beam_y_array,
         theta_array,
         phi_array,
         wavelength_array) = generate_random_profiles(
             num_samples=simulation_runtime_state.num_samples,
             divergence_sigma=divergence_sigma,
             bw_sigma=bw_sigma,
             lambda0=lambda_,
             bandwidth=active_bandwidth
         )
        simulation_runtime_state.profile_cache = {
            "beam_x_array": beam_x_array,
            "beam_y_array": beam_y_array,
            "theta_array": theta_array,
            "phi_array": phi_array,
            "wavelength_array": wavelength_array,
            "_sampling_signature": sampling_signature,
        }

    simulation_runtime_state.profile_cache.update(
        {
            "sigma_mosaic_deg": sigma_mosaic_var.get(),
            "gamma_mosaic_deg": gamma_mosaic_var.get(),
            "eta": eta_var.get(),
            "solve_q_steps": _current_solve_q_steps(),
            "solve_q_rel_tol": _current_solve_q_rel_tol(),
            "solve_q_mode": _current_solve_q_mode_flag(),
            "bandwidth_percent": active_bandwidth * 100.0,
        }
    )

def on_mosaic_slider_change(*args):
    update_mosaic_cache()
    schedule_update()


def _current_phase_delta_expression() -> str:
    """Return the active HT delta-phase expression from GUI state."""

    fallback = normalize_phase_delta_expression(
        defaults.get("phase_delta_expression", DEFAULT_PHASE_DELTA_EXPRESSION),
        fallback=DEFAULT_PHASE_DELTA_EXPRESSION,
    )
    var = finite_stack_controls_view_state.phase_delta_expr_var
    if var is None:
        return fallback

    try:
        value = str(var.get())
    except Exception:
        value = fallback

    candidate = normalize_phase_delta_expression(value, fallback=fallback)
    try:
        return validate_phase_delta_expression(candidate)
    except ValueError:
        return fallback


def _current_phi_l_divisor() -> float:
    """Return the active HT L-divisor used in ``phi = delta + 2*pi*L/div``."""

    fallback = normalize_phi_l_divisor(
        defaults.get("phi_l_divisor", DEFAULT_PHI_L_DIVISOR),
        fallback=DEFAULT_PHI_L_DIVISOR,
    )
    var = finite_stack_controls_view_state.phi_l_divisor_var
    if var is None:
        return fallback

    try:
        value = var.get()
    except Exception:
        value = fallback
    return normalize_phi_l_divisor(value, fallback=fallback)


def _current_iodine_z(active_cif_path=None) -> float:
    """Return iodine z inferred from the active CIF (clamped to [0, 1])."""

    return gui_structure_model.current_iodine_z(
        structure_model_state,
        atom_site_override_state,
        active_cif_path=active_cif_path,
        tcl_error_types=(tk.TclError,),
    )


line_rmin, = ax.plot([], [], color='white', linestyle='-', linewidth=2, zorder=5)
line_rmax, = ax.plot([], [], color='white', linestyle='-', linewidth=2, zorder=5)
line_amin, = ax.plot([], [], color='cyan', linestyle='-', linewidth=2, zorder=5)
line_amax, = ax.plot([], [], color='cyan', linestyle='-', linewidth=2, zorder=5)

UPDATE_DEBOUNCE_MS = 120
RANGE_UPDATE_DEBOUNCE_MS = 120
CHI_SQUARE_UPDATE_INTERVAL_S = 0.5

simulation_runtime_state.update_pending = None
simulation_runtime_state.integration_update_pending = None
simulation_runtime_state.update_running = False

def schedule_update():
    """Queue a throttled simulation/redraw update."""
    if simulation_runtime_state.integration_update_pending is not None:
        root.after_cancel(simulation_runtime_state.integration_update_pending)
        simulation_runtime_state.integration_update_pending = None
    if simulation_runtime_state.update_pending is not None:
        root.after_cancel(simulation_runtime_state.update_pending)
    simulation_runtime_state.update_pending = root.after(UPDATE_DEBOUNCE_MS, do_update)


def _run_scheduled_range_update():
    simulation_runtime_state.integration_update_pending = None

    if simulation_runtime_state.update_running:
        schedule_range_update(delay_ms=RANGE_UPDATE_DEBOUNCE_MS)
        return

    if not _refresh_integration_from_cached_results():
        schedule_update()


def schedule_range_update(delay_ms=RANGE_UPDATE_DEBOUNCE_MS):
    """Queue throttled redraw-only integration updates."""
    if simulation_runtime_state.integration_update_pending is not None:
        root.after_cancel(simulation_runtime_state.integration_update_pending)
    delay = max(RANGE_UPDATE_DEBOUNCE_MS, int(delay_ms))
    simulation_runtime_state.integration_update_pending = root.after(delay, _run_scheduled_range_update)


def _should_collect_hit_tables_for_update() -> bool:
    """Return whether the next redraw needs per-hit detector tables."""
    return gui_manual_geometry.should_collect_hit_tables_for_update(
        background_visible=bool(background_runtime_state.visible),
        current_background_index=background_runtime_state.current_background_index,
        hkl_pick_armed=bool(peak_selection_state.hkl_pick_armed),
        selected_hkl_target=peak_selection_state.selected_hkl_target,
        selected_peak_record=simulation_runtime_state.selected_peak_record,
        geometry_q_group_refresh_requested=bool(geometry_q_group_state.refresh_requested),
        live_geometry_preview_enabled=_live_geometry_preview_enabled,
        current_manual_pick_background_image=_current_geometry_manual_pick_background_image,
        geometry_manual_pairs_for_index=_geometry_manual_pairs_for_index,
        geometry_manual_pick_session_active=_geometry_manual_pick_session_active,
    )

simulation_runtime_state.peak_positions = []
simulation_runtime_state.peak_millers = []
simulation_runtime_state.peak_intensities = []
simulation_runtime_state.peak_records = []
simulation_runtime_state.selected_peak_record = None
# 0 disables the cap and keeps all distinct per-reflection hit candidates.
HKL_PICK_MAX_HITS_PER_REFLECTION = 0
HKL_PICK_MIN_SEPARATION_PX = 2.0
HKL_PICK_MAX_DISTANCE_PX = 12.0

simulation_runtime_state.prev_background_visible = True
simulation_runtime_state.last_bg_signature = None
simulation_runtime_state.last_sim_signature = None
simulation_runtime_state.last_simulation_signature = None
simulation_runtime_state.stored_max_positions_local = None
simulation_runtime_state.stored_sim_image = None
simulation_runtime_state.stored_peak_table_lattice = None
simulation_runtime_state.stored_primary_sim_image = None
simulation_runtime_state.stored_secondary_sim_image = None
simulation_runtime_state.stored_primary_max_positions = None
simulation_runtime_state.stored_secondary_max_positions = None
simulation_runtime_state.stored_primary_peak_table_lattice = None
simulation_runtime_state.stored_secondary_peak_table_lattice = None
simulation_runtime_state.last_unscaled_image_signature = None
simulation_runtime_state.normalization_scale_cache = {"sig": None, "value": 1.0}
simulation_runtime_state.peak_overlay_cache = {
    "sig": None,
    "positions": [],
    "millers": [],
    "intensities": [],
    "records": [],
}
simulation_runtime_state.caking_cache = {
    "sim_sig": None,
    "sim_res2": None,
    "bg_sig": None,
    "bg_res2": None,
}
simulation_runtime_state.chi_square_update_token = 0
simulation_runtime_state.chi_square_state = {
    "last_ts": 0.0,
    "last_token": -1,
    "last_text": "Chi-Squared: N/A",
}


def _ensure_peak_overlay_data(*, force: bool = False) -> bool:
    """Ensure peak overlay lists are available for the current simulation cache."""


    if simulation_runtime_state.stored_max_positions_local is None or simulation_runtime_state.stored_sim_image is None:
        simulation_runtime_state.peak_positions.clear()
        simulation_runtime_state.peak_millers.clear()
        simulation_runtime_state.peak_intensities.clear()
        simulation_runtime_state.peak_records.clear()
        return False

    max_positions_local = simulation_runtime_state.stored_max_positions_local
    updated_image = simulation_runtime_state.stored_sim_image
    peak_table_lattice_local = simulation_runtime_state.stored_peak_table_lattice

    if (
        not peak_table_lattice_local
        or len(peak_table_lattice_local) != len(max_positions_local)
    ):
        peak_table_lattice_local = [
            (float(a_var.get()), float(c_var.get()), "primary")
            for _ in max_positions_local
        ]

    peak_sig = (
        simulation_runtime_state.last_simulation_signature,
        id(max_positions_local),
        len(max_positions_local),
        tuple(updated_image.shape),
        int(HKL_PICK_MAX_HITS_PER_REFLECTION),
        float(HKL_PICK_MIN_SEPARATION_PX),
    )
    peak_cached = (not force) and simulation_runtime_state.peak_overlay_cache.get("sig") == peak_sig

    simulation_runtime_state.peak_positions.clear()
    simulation_runtime_state.peak_millers.clear()
    simulation_runtime_state.peak_intensities.clear()
    simulation_runtime_state.peak_records.clear()

    if peak_cached:
        simulation_runtime_state.peak_positions.extend(list(simulation_runtime_state.peak_overlay_cache.get("positions", ())))
        simulation_runtime_state.peak_millers.extend(list(simulation_runtime_state.peak_overlay_cache.get("millers", ())))
        simulation_runtime_state.peak_intensities.extend(list(simulation_runtime_state.peak_overlay_cache.get("intensities", ())))
        simulation_runtime_state.peak_records.extend(dict(rec) for rec in simulation_runtime_state.peak_overlay_cache.get("records", ()))
        return True

    try:
        max_hits_raw = int(HKL_PICK_MAX_HITS_PER_REFLECTION)
    except Exception:
        max_hits_raw = 0
    max_hits_per_reflection = max_hits_raw if max_hits_raw > 0 else None
    min_sep_sq = float(HKL_PICK_MIN_SEPARATION_PX) ** 2

    for table_idx, tbl in enumerate(max_positions_local):
        tbl_arr = np.asarray(tbl, dtype=float)
        if tbl_arr.ndim != 2 or tbl_arr.shape[0] == 0:
            continue
        av_used, cv_used, source_label = peak_table_lattice_local[table_idx]
        row_order = np.argsort(tbl_arr[:, 0])[::-1]
        chosen_rows: list[tuple[int, np.ndarray, float, float, float, float]] = []

        for row_idx in row_order:
            row = tbl_arr[row_idx]
            I, xpix, ypix, _phi, H, K, L = row
            if not (np.isfinite(I) and np.isfinite(xpix) and np.isfinite(ypix)):
                continue
            cx = float(xpix)
            cy = float(ypix)
            disp_cx, disp_cy = _native_sim_to_display_coords(cx, cy, updated_image.shape)
            too_close = False
            for _, _, _, _, prev_col, prev_row in chosen_rows:
                d2 = (disp_cx - prev_col) ** 2 + (disp_cy - prev_row) ** 2
                if d2 < min_sep_sq:
                    too_close = True
                    break
            if too_close:
                continue
            chosen_rows.append((int(row_idx), row, cx, cy, disp_cx, disp_cy))
            if (
                max_hits_per_reflection is not None
                and len(chosen_rows) >= max_hits_per_reflection
            ):
                break

        for row_idx, row, cx, cy, disp_cx, disp_cy in chosen_rows:
            I, _xpix, _ypix, _phi, H, K, L = row
            simulation_runtime_state.peak_positions.append((disp_cx, disp_cy))
            simulation_runtime_state.peak_intensities.append(float(I))
            hkl = tuple(int(np.rint(val)) for val in (H, K, L))
            hkl_raw = (float(H), float(K), float(L))
            m_val = float(H * H + H * K + K * K)
            qr_val = (
                (2.0 * np.pi / float(av_used)) * np.sqrt((4.0 / 3.0) * m_val)
                if float(av_used) > 0.0 and np.isfinite(float(av_used)) and m_val >= 0.0
                else float("nan")
            )
            q_group_key, _, qz_val = _reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
                qr_value=qr_val,
            )
            simulation_runtime_state.peak_millers.append(hkl)
            simulation_runtime_state.peak_records.append(
                {
                    "display_col": float(disp_cx),
                    "display_row": float(disp_cy),
                    "native_col": float(cx),
                    "native_row": float(cy),
                    "hkl": hkl,
                    "hkl_raw": hkl_raw,
                    "intensity": float(I),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "q_group_key": q_group_key,
                    "phi": float(_phi),
                    "source_table_index": int(table_idx),
                    "source_row_index": int(row_idx),
                    "source_label": str(source_label),
                    "av": float(av_used),
                    "cv": float(cv_used),
                }
            )

    simulation_runtime_state.peak_overlay_cache.update(
        {
            "sig": peak_sig,
            "positions": list(simulation_runtime_state.peak_positions),
            "millers": list(simulation_runtime_state.peak_millers),
            "intensities": list(simulation_runtime_state.peak_intensities),
            "records": [dict(rec) for rec in simulation_runtime_state.peak_records],
        }
    )
    return True

###############################################################################
#                              MAIN UPDATE
###############################################################################
def do_update():
    global av2, cv2

    if simulation_runtime_state.update_running:
        # another update is in progress; try again shortly
        simulation_runtime_state.update_pending = root.after(UPDATE_DEBOUNCE_MS, do_update)
        return

    simulation_runtime_state.update_pending = None
    simulation_runtime_state.update_running = True
    update_start_time = perf_counter()
    image_generation_elapsed_ms = 0.0
    image_generation_cached = True

    gamma_updated      = float(gamma_var.get())
    Gamma_updated      = float(Gamma_var.get())
    chi_updated        = float(chi_var.get())
    psi_z_updated      = float(psi_z_var.get())
    zs_updated         = float(zs_var.get())
    zb_updated         = float(zb_var.get())
    cor_angle_updated  = float(cor_angle_var.get())
    a_updated          = float(a_var.get())
    c_updated          = float(c_var.get())
    theta_init_up      = float(theta_initial_var.get())
    debye_x_updated    = float(debye_x_var.get())
    debye_y_updated    = float(debye_y_var.get())
    corto_det_up       = float(corto_detector_var.get())
    center_x_up        = float(center_x_var.get())
    center_y_up        = float(center_y_var.get())

    new_two_theta_max = detector_two_theta_max(
        image_size,
        [center_x_up, center_y_up],
        corto_det_up,
        pixel_size=pixel_size_m,
    )

    global two_theta_range, _last_a_for_ht, _last_c_for_ht, _last_iodine_z_for_ht
    global _last_phi_l_divisor, _last_phase_delta_expression
    global _last_atom_site_fractional_signature
    phase_delta_expression_current = _current_phase_delta_expression()
    phi_l_divisor_current = _current_phi_l_divisor()
    iodine_z_current = _current_iodine_z()
    atom_site_signature = _atom_site_fractional_signature(_current_atom_site_fractional_values())
    need_rebuild = False
    if not math.isclose(new_two_theta_max, two_theta_range[1], rel_tol=1e-6, abs_tol=1e-6):
        two_theta_range = (0.0, new_two_theta_max)
        need_rebuild = True
    if not math.isclose(c_updated, _last_c_for_ht, rel_tol=1e-9, abs_tol=1e-9):
        need_rebuild = True
    if not math.isclose(a_updated, _last_a_for_ht, rel_tol=1e-9, abs_tol=1e-9):
        need_rebuild = True
    if not math.isclose(iodine_z_current, _last_iodine_z_for_ht, rel_tol=1e-9, abs_tol=1e-9):
        need_rebuild = True
    if not math.isclose(
        float(phi_l_divisor_current),
        float(_last_phi_l_divisor),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        need_rebuild = True
    if phase_delta_expression_current != _last_phase_delta_expression:
        need_rebuild = True
    if atom_site_signature != _last_atom_site_fractional_signature:
        need_rebuild = True

    if need_rebuild:
        current_occ = [occ_var.get() for occ_var in occ_vars]
        current_p = [p0_var.get(), p1_var.get(), p2_var.get()]
        weight_values = [w0_var.get(), w1_var.get(), w2_var.get()]
        normalized_weights = gui_controllers.normalize_stacking_weight_values(
            weight_values
        )
        _rebuild_diffraction_inputs(
            current_occ,
            current_p,
            normalized_weights,
            a_updated,
            c_updated,
            force=True,
            trigger_update=False,
        )

    center_marker.set_xdata([center_y_up])
    center_marker.set_ydata([center_x_up])
    center_marker.set_visible(False)

    mosaic_params = build_mosaic_params()
    optics_mode_flag = _current_optics_mode_flag()
    collect_hit_tables_requested = _should_collect_hit_tables_for_update()


    def get_sim_signature():
        return (
            round(gamma_updated, 6),
            round(Gamma_updated, 6),
            round(chi_updated, 6),
            round(psi_z_updated, 6),
            round(zs_updated, 9),
            round(zb_updated, 9),
            round(debye_x_updated, 6),
            round(debye_y_updated, 6),
            round(a_updated, 6),
            round(c_updated, 6),
            round(theta_init_up, 6),
            round(cor_angle_updated, 6),
            round(center_x_up, 3),
            round(center_y_up, 3),
            round(mosaic_params["sigma_mosaic_deg"], 6),
            round(mosaic_params["gamma_mosaic_deg"], 6),
            round(mosaic_params["eta"], 6),
            int(mosaic_params["solve_q_steps"]),
            round(float(mosaic_params["solve_q_rel_tol"]), 8),
            int(mosaic_params["solve_q_mode"]),
            round(_current_bandwidth_fraction(), 8),
            round(_current_sf_prune_bias(), 3),
            int(simulation_runtime_state.sf_prune_stats.get("qr_kept", 0)),
            int(simulation_runtime_state.sf_prune_stats.get("hkl_primary_kept", 0)),
            int(optics_mode_flag),
            int(simulation_runtime_state.num_samples),
            int(np.size(mosaic_params["beam_x_array"])),
            int(np.size(mosaic_params["theta_array"])),
            int(collect_hit_tables_requested),
        )

    new_sim_sig = get_sim_signature()
    if new_sim_sig != simulation_runtime_state.last_simulation_signature:
        _invalidate_geometry_manual_pick_cache()
        simulation_runtime_state.last_simulation_signature = new_sim_sig
        simulation_runtime_state.peak_positions.clear()
        simulation_runtime_state.peak_millers.clear()
        simulation_runtime_state.peak_intensities.clear()
        simulation_runtime_state.peak_records.clear()
        simulation_runtime_state.selected_peak_record = None
        image_generation_cached = False
        image_generation_start_time = perf_counter()

        def run_one(data, intens_arr, a_val, c_val):
            buf = np.zeros((image_size, image_size), dtype=np.float64)
            if isinstance(data, dict):
                if len(data) == 0:
                    return buf, [], None, None, None, None, None
                if DEBUG_ENABLED:
                    n_pts = sum(len(v["L"]) for v in data.values())
                    debug_print("process_qr_rods_parallel with", n_pts, "points")
                return process_qr_rods_parallel(
                    data,
                    image_size,
                    a_val,
                    c_val,
                    lambda_,
                    buf,
                    corto_det_up,
                    gamma_updated,
                    Gamma_updated,
                    chi_updated,
                    psi,
                    psi_z_updated,
                    zs_updated,
                    zb_updated,
                    n2,
                    mosaic_params["beam_x_array"],
                    mosaic_params["beam_y_array"],
                    mosaic_params["theta_array"],
                    mosaic_params["phi_array"],
                    mosaic_params["sigma_mosaic_deg"],
                    mosaic_params["gamma_mosaic_deg"],
                    mosaic_params["eta"],
                    mosaic_params["wavelength_array"],
                    debye_x_updated,
                    debye_y_updated,
                    [center_x_up, center_y_up],
                    theta_init_up,
                    cor_angle_updated,
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    save_flag=0,
                    optics_mode=optics_mode_flag,
                    solve_q_steps=int(mosaic_params["solve_q_steps"]),
                    solve_q_rel_tol=float(mosaic_params["solve_q_rel_tol"]),
                    solve_q_mode=int(mosaic_params["solve_q_mode"]),
                    collect_hit_tables=collect_hit_tables_requested,
                )
            else:
                miller_arr = np.asarray(data, dtype=np.float64)
                intens_vals = np.asarray(intens_arr, dtype=np.float64).reshape(-1)
                if miller_arr.ndim != 2 or miller_arr.shape[1] < 3:
                    return buf, [], None, None, None, None, None
                row_count = min(miller_arr.shape[0], intens_vals.shape[0])
                if row_count <= 0:
                    return buf, [], None, None, None, None, None
                miller_arr = miller_arr[:row_count, :]
                intens_vals = intens_vals[:row_count]
                if DEBUG_ENABLED:
                    debug_print("process_peaks_parallel with", miller_arr.shape[0], "reflections")
                    if not np.all(np.isfinite(miller_arr)):
                        debug_print("Non-finite miller indices detected")
                    if not np.all(np.isfinite(intens_vals)):
                        debug_print("Non-finite intensities detected")
                return process_peaks_parallel(
                    miller_arr,
                    intens_vals,
                    image_size,
                    a_val,
                    c_val,
                    lambda_,
                    buf,
                    corto_det_up,
                    gamma_updated,
                    Gamma_updated,
                    chi_updated,
                    psi,
                    psi_z_updated,
                    zs_updated,
                    zb_updated,
                    n2,
                    mosaic_params["beam_x_array"],
                    mosaic_params["beam_y_array"],
                    mosaic_params["theta_array"],
                    mosaic_params["phi_array"],
                    mosaic_params["sigma_mosaic_deg"],
                    mosaic_params["gamma_mosaic_deg"],
                    mosaic_params["eta"],
                    mosaic_params["wavelength_array"],
                    debye_x_updated,
                    debye_y_updated,
                    [center_x_up, center_y_up],
                    theta_init_up,
                    cor_angle_updated,
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    save_flag=0,
                    optics_mode=optics_mode_flag,
                    solve_q_steps=int(mosaic_params["solve_q_steps"]),
                    solve_q_rel_tol=float(mosaic_params["solve_q_rel_tol"]),
                    solve_q_mode=int(mosaic_params["solve_q_mode"]),
                    collect_hit_tables=collect_hit_tables_requested,
                ) + (None,)

        w1 = float(weight1_var.get())
        w2 = float(weight2_var.get())
        run_primary = abs(w1) > 1e-12
        run_secondary = bool(simulation_runtime_state.sim_miller2.size > 0 and abs(w2) > 1e-12)

        img1 = np.zeros((image_size, image_size), dtype=np.float64)
        img2 = np.zeros((image_size, image_size), dtype=np.float64)
        maxpos1 = []
        maxpos2 = []

        if run_primary:
            primary_data = (
                simulation_runtime_state.sim_primary_qr
                if isinstance(simulation_runtime_state.sim_primary_qr, dict) and len(simulation_runtime_state.sim_primary_qr) > 0
                else simulation_runtime_state.sim_miller1
            )
            img1, maxpos1, _, _, _, _, _ = run_one(
                primary_data,
                simulation_runtime_state.sim_intens1,
                a_updated,
                c_updated,
            )

        if run_secondary:
            img2, maxpos2, _, _, _, _, _ = run_one(
                simulation_runtime_state.sim_miller2,
                simulation_runtime_state.sim_intens2,
                av2,
                cv2,
            )

        primary_max_positions_local = list(maxpos1)
        secondary_max_positions_local = list(maxpos2)
        primary_peak_table_lattice_local = [
            (float(a_updated), float(c_updated), "primary")
            for _ in maxpos1
        ]
        secondary_peak_table_lattice_local = []
        if secondary_max_positions_local:
            sec_a = float(av2) if av2 is not None else float(a_updated)
            sec_c = float(cv2) if cv2 is not None else float(c_updated)
            secondary_peak_table_lattice_local = [
                (sec_a, sec_c, "secondary")
                for _ in secondary_max_positions_local
            ]

        simulation_runtime_state.stored_primary_sim_image = img1
        simulation_runtime_state.stored_secondary_sim_image = img2
        simulation_runtime_state.stored_primary_max_positions = primary_max_positions_local
        simulation_runtime_state.stored_secondary_max_positions = secondary_max_positions_local
        simulation_runtime_state.stored_primary_peak_table_lattice = primary_peak_table_lattice_local
        simulation_runtime_state.stored_secondary_peak_table_lattice = secondary_peak_table_lattice_local

        updated_image = w1 * simulation_runtime_state.stored_primary_sim_image + w2 * simulation_runtime_state.stored_secondary_sim_image
        max_positions_local = []
        peak_table_lattice_local = []
        if run_primary:
            max_positions_local.extend(simulation_runtime_state.stored_primary_max_positions)
            peak_table_lattice_local.extend(simulation_runtime_state.stored_primary_peak_table_lattice)
        if run_secondary:
            max_positions_local.extend(simulation_runtime_state.stored_secondary_max_positions)
            peak_table_lattice_local.extend(simulation_runtime_state.stored_secondary_peak_table_lattice)

        simulation_runtime_state.stored_max_positions_local = list(max_positions_local)
        simulation_runtime_state.stored_peak_table_lattice = list(peak_table_lattice_local)
        simulation_runtime_state.stored_sim_image = updated_image
        image_generation_elapsed_ms = (
            perf_counter() - image_generation_start_time
        ) * 1e3
    else:
        # fall back to the cached arrays
        if simulation_runtime_state.stored_primary_sim_image is None and simulation_runtime_state.stored_secondary_sim_image is None:
            # first run after programme start – force a simulation
            simulation_runtime_state.last_simulation_signature = None
            simulation_runtime_state.update_running = False
            return do_update()          # re-enter with computation path

        w1 = float(weight1_var.get())
        w2 = float(weight2_var.get())
        run_primary = bool(simulation_runtime_state.stored_primary_sim_image is not None and abs(w1) > 1e-12)
        run_secondary = bool(simulation_runtime_state.stored_secondary_sim_image is not None and abs(w2) > 1e-12)

        img1 = (
            simulation_runtime_state.stored_primary_sim_image
            if simulation_runtime_state.stored_primary_sim_image is not None
            else np.zeros((image_size, image_size), dtype=np.float64)
        )
        img2 = (
            simulation_runtime_state.stored_secondary_sim_image
            if simulation_runtime_state.stored_secondary_sim_image is not None
            else np.zeros((image_size, image_size), dtype=np.float64)
        )

        updated_image = w1 * img1 + w2 * img2
        max_positions_local = []
        peak_table_lattice_local = []
        if run_primary and simulation_runtime_state.stored_primary_max_positions is not None:
            max_positions_local.extend(simulation_runtime_state.stored_primary_max_positions)
            if simulation_runtime_state.stored_primary_peak_table_lattice is not None:
                peak_table_lattice_local.extend(simulation_runtime_state.stored_primary_peak_table_lattice)
        if run_secondary and simulation_runtime_state.stored_secondary_max_positions is not None:
            max_positions_local.extend(simulation_runtime_state.stored_secondary_max_positions)
            if simulation_runtime_state.stored_secondary_peak_table_lattice is not None:
                peak_table_lattice_local.extend(simulation_runtime_state.stored_secondary_peak_table_lattice)

        simulation_runtime_state.stored_max_positions_local = list(max_positions_local)
        simulation_runtime_state.stored_peak_table_lattice = list(peak_table_lattice_local)
        simulation_runtime_state.stored_sim_image = updated_image

    if not peak_table_lattice_local or len(peak_table_lattice_local) != len(max_positions_local):
        peak_table_lattice_local = [
            (float(a_updated), float(c_updated), "primary")
            for _ in max_positions_local
        ]

    if gui_controllers.consume_geometry_q_group_refresh_request(geometry_q_group_state):
        listed_entries = _capture_geometry_q_group_entries_snapshot()
        progress_label_geometry.config(
            text=(
                f"Updated listed Qr/Qz peaks: {len(listed_entries)} groups, "
                f"{sum(int(entry.get('peak_count', 0)) for entry in listed_entries)} peaks."
            )
        )

    redraw_update_start_time = perf_counter()
    display_image = np.rot90(updated_image, SIM_DISPLAY_ROTATE_K)
    need_peak_overlay = bool(
        peak_selection_state.hkl_pick_armed
        or peak_selection_state.selected_hkl_target is not None
        or simulation_runtime_state.selected_peak_record is not None
        or _live_geometry_preview_enabled()
    )
    if need_peak_overlay:
        _ensure_peak_overlay_data(force=False)
    else:
        simulation_runtime_state.peak_positions.clear()
        simulation_runtime_state.peak_millers.clear()
        simulation_runtime_state.peak_intensities.clear()
        simulation_runtime_state.peak_records.clear()

    if peak_selection_state.selected_hkl_target is not None:
        _select_peak_by_hkl(
            peak_selection_state.selected_hkl_target[0],
            peak_selection_state.selected_hkl_target[1],
            peak_selection_state.selected_hkl_target[2],
            sync_hkl_vars=False,
            silent_if_missing=True,
        )
    normalization_scale = 1.0
    native_background = _get_current_background_backend()
    if native_background is not None and display_image is not None:
        normalization_sig = (
            new_sim_sig,
            int(background_runtime_state.current_background_index),
            id(background_runtime_state.current_background_image),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
            tuple(display_image.shape),
            tuple(np.asarray(native_background).shape),
        )
        if simulation_runtime_state.normalization_scale_cache.get("sig") == normalization_sig:
            normalization_scale = float(simulation_runtime_state.normalization_scale_cache.get("value", 1.0))
        else:
            normalization_scale = _suggest_scale_factor(
                display_image, native_background
            )
            simulation_runtime_state.normalization_scale_cache["sig"] = normalization_sig
            simulation_runtime_state.normalization_scale_cache["value"] = float(normalization_scale)
        if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
            normalization_scale = 1.0

    simulation_runtime_state.unscaled_image = None
    simulation_runtime_state.last_unscaled_image_signature = None
    if display_image is not None:
        if abs(float(normalization_scale) - 1.0) <= 1e-12:
            simulation_runtime_state.unscaled_image = display_image
        else:
            simulation_runtime_state.unscaled_image = display_image * normalization_scale
        simulation_runtime_state.last_unscaled_image_signature = (
            new_sim_sig,
            round(float(normalization_scale), 9),
            tuple(display_image.shape),
            int(background_runtime_state.current_background_index),
            id(background_runtime_state.current_background_image),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
        )
        if simulation_runtime_state.peak_intensities and normalization_scale != 1.0:
            simulation_runtime_state.peak_intensities[:] = [
                intensity * normalization_scale for intensity in simulation_runtime_state.peak_intensities
            ]
            for rec in simulation_runtime_state.peak_records:
                rec["intensity"] = float(rec.get("intensity", 0.0)) * normalization_scale

    simulation_runtime_state.last_1d_integration_data["simulated_2d_image"] = simulation_runtime_state.unscaled_image

    if simulation_runtime_state.unscaled_image is not None:
        if display_controls_state.scale_factor_user_override:
            _set_scale_factor_value(
                _get_scale_factor_value(default=1.0),
                adjust_range=False,
                reset_override=False,
            )
        else:
            _set_scale_factor_value(
                1.0,
                adjust_range=False,
                reset_override=True,
            )
    # ---------------------------------------------------------------
    # pyFAI integrator setup is relatively expensive. Cache the
    # AzimuthalIntegrator instance and only recreate it when any of the
    # geometry parameters actually change. This significantly reduces
    # overhead when repeatedly redrawing the live simulation with
    # unchanged geometry settings.
    # ---------------------------------------------------------------
    sig = (
        corto_det_up,
        center_x_up,
        center_y_up,
        Gamma_updated,
        gamma_updated,
        wave_m,
    )
    if simulation_runtime_state.ai_cache.get("sig") != sig:
        simulation_runtime_state.ai_cache = {
            "sig": sig,
            "ai": AzimuthalIntegrator(
                dist=corto_det_up,
                # Keep the legacy row/col mapping aligned with simulation pixels.
                poni1=center_x_up * pixel_size_m,
                poni2=center_y_up * pixel_size_m,
                rot1=0.0,
                rot2=0.0,
                rot3=0.0,
                wavelength=wave_m,
                pixel1=pixel_size_m,
                pixel2=pixel_size_m,
            ),
        }
    ai = simulation_runtime_state.ai_cache["ai"]
    sim_caking_sig = (
        new_sim_sig,
        sig,
        round(float(normalization_scale), 9),
    )
    bg_caking_sig = None
    if native_background is not None:
        bg_caking_sig = (
            sig,
            int(background_runtime_state.current_background_index),
            id(background_runtime_state.current_background_image),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
            tuple(np.asarray(native_background).shape),
        )

    # Caked 2D or normal 2D?
    sim_res2 = None
    bg_res2 = None
    if analysis_view_controls_view_state.show_caked_2d_var.get() and simulation_runtime_state.unscaled_image is not None:
        if (
            simulation_runtime_state.caking_cache.get("sim_sig") == sim_caking_sig
            and simulation_runtime_state.caking_cache.get("sim_res2") is not None
        ):
            sim_res2 = simulation_runtime_state.caking_cache["sim_res2"]
        else:
            sim_res2 = caking(simulation_runtime_state.unscaled_image, ai)
            simulation_runtime_state.caking_cache["sim_sig"] = sim_caking_sig
            simulation_runtime_state.caking_cache["sim_res2"] = sim_res2
        caked_img = sim_res2.intensity
        radial_vals = np.asarray(sim_res2.radial, dtype=float)
        azimuth_vals = _wrap_phi_range(_adjust_phi_zero(sim_res2.azimuthal))

        if azimuth_vals.size:
            azimuth_order = np.argsort(azimuth_vals)
            azimuth_vals = azimuth_vals[azimuth_order]
            caked_img = caked_img[azimuth_order, :]

        radial_mask = (radial_vals >= 0.0) & (radial_vals <= 90.0)
        if np.any(radial_mask):
            radial_vals = radial_vals[radial_mask]
            caked_img = caked_img[:, radial_mask]

        simulation_runtime_state.last_caked_image_unscaled = caked_img
        simulation_runtime_state.last_caked_radial_values = np.asarray(radial_vals, dtype=float)
        simulation_runtime_state.last_caked_azimuth_values = np.asarray(azimuth_vals, dtype=float)

        current_scale = _get_scale_factor_value(default=1.0)
        scaled_caked_for_limits = caked_img * current_scale
        auto_vmin, auto_vmax = _auto_caked_limits(scaled_caked_for_limits)

        if not display_controls_state.simulation_limits_user_override:
            _update_simulation_sliders_from_image(
                scaled_caked_for_limits, reset_override=True
            )

        if not simulation_runtime_state.caked_limits_user_override:
            vmin_caked_var.set(auto_vmin)
            vmax_caked_var.set(auto_vmax)

        vmin_val = float(display_controls_view_state.simulation_min_var.get())
        vmax_val = float(display_controls_view_state.simulation_max_var.get())
        global_sim_max = vmax_val

        if not math.isfinite(vmin_val):
            vmin_val = auto_vmin
        if not math.isfinite(vmax_val):
            vmax_val = auto_vmax
        vmin_val, vmax_val = _ensure_valid_range(vmin_val, vmax_val)
        if not math.isfinite(global_sim_max) or global_sim_max <= vmin_val:
            global_sim_max = auto_vmax

        display_vmax = min(vmax_val, global_sim_max)
        if not math.isfinite(display_vmax):
            display_vmax = auto_vmax
        if display_vmax <= vmin_val:
            fallback_vmax = max(global_sim_max, auto_vmax, vmax_val)
            if math.isfinite(fallback_vmax) and fallback_vmax > vmin_val:
                display_vmax = fallback_vmax
            else:
                display_vmax = vmin_val + max(abs(vmin_val) * 1e-3, 1e-3)

        background_caked_available = False
        simulation_runtime_state.last_caked_background_image_unscaled = None
        if background_runtime_state.visible and native_background is not None:
            if (
                simulation_runtime_state.caking_cache.get("bg_sig") == bg_caking_sig
                and simulation_runtime_state.caking_cache.get("bg_res2") is not None
            ):
                bg_res2 = simulation_runtime_state.caking_cache["bg_res2"]
            else:
                bg_res2 = caking(native_background, ai)
                simulation_runtime_state.caking_cache["bg_sig"] = bg_caking_sig
                simulation_runtime_state.caking_cache["bg_res2"] = bg_res2
            bg_caked = bg_res2.intensity
            bg_radial = np.asarray(bg_res2.radial, dtype=float)
            bg_azimuth = _wrap_phi_range(_adjust_phi_zero(bg_res2.azimuthal))

            if bg_azimuth.size:
                bg_order = np.argsort(bg_azimuth)
                bg_azimuth = bg_azimuth[bg_order]
                bg_caked = bg_caked[bg_order, :]

            bg_radial_mask = (bg_radial >= 0.0) & (bg_radial <= 90.0)
            if np.any(bg_radial_mask):
                bg_radial = bg_radial[bg_radial_mask]
                bg_caked = bg_caked[:, bg_radial_mask]

            simulation_runtime_state.last_caked_background_image_unscaled = np.asarray(bg_caked, dtype=float)
            _set_image_origin(background_display, 'lower')
            background_display.set_data(bg_caked)
            bg_display_vmax = vmax_val
            if not math.isfinite(bg_display_vmax):
                bg_display_vmax = auto_vmax
            if not math.isfinite(bg_display_vmax):
                bg_display_vmax = display_vmax
            if bg_display_vmax <= vmin_val:
                fallback_vmax = None
                for candidate in (auto_vmax, display_vmax, vmax_val):
                    if math.isfinite(candidate) and candidate > vmin_val:
                        fallback_vmax = candidate
                        break
                if fallback_vmax is None:
                    fallback_vmax = vmin_val + max(abs(vmin_val) * 1e-3, 1e-3)
                bg_display_vmax = fallback_vmax
            background_display.set_clim(vmin_val, bg_display_vmax)
            background_display.set_visible(True)
            background_caked_available = True
        else:
            background_display.set_visible(False)

        if radial_vals.size:
            radial_min = float(np.min(radial_vals))
            radial_max = float(np.max(radial_vals))
        else:
            radial_min, radial_max = 0.0, 90.0

        if azimuth_vals.size:
            azimuth_min = float(np.min(azimuth_vals))
            azimuth_max = float(np.max(azimuth_vals))
        else:
            azimuth_min, azimuth_max = -180.0, 180.0

        simulation_runtime_state.last_caked_extent = [
            radial_min,
            radial_max,
            azimuth_min,
            azimuth_max,
        ]
        if background_caked_available:
            background_display.set_extent([
                radial_min,
                radial_max,
                azimuth_min,
                azimuth_max,
            ])
        else:
            background_display.set_visible(False)
        if not (math.isfinite(radial_min) and math.isfinite(radial_max) and radial_max > radial_min):
            radial_min, radial_max = 0.0, 90.0
        if not (
            math.isfinite(azimuth_min)
            and math.isfinite(azimuth_max)
            and azimuth_max > azimuth_min
        ):
            azimuth_min, azimuth_max = -180.0, 180.0

        ax.set_xlim(radial_min, radial_max)
        ax.set_ylim(azimuth_min, azimuth_max)
        ax.set_aspect("auto")
        ax.set_xlabel('2θ (degrees)')
        ax.set_ylabel('φ (degrees)')
        ax.set_title('2D Caked Integration')
    else:
        simulation_runtime_state.last_caked_image_unscaled = None
        simulation_runtime_state.last_caked_extent = None
        simulation_runtime_state.last_caked_background_image_unscaled = None
        simulation_runtime_state.last_caked_radial_values = None
        simulation_runtime_state.last_caked_azimuth_values = None
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)
        ax.set_aspect("auto")
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Simulated Diffraction Pattern')

        _set_image_origin(background_display, 'upper')
        background_display.set_extent([0, image_size, image_size, 0])
        if background_runtime_state.visible and background_runtime_state.current_background_display is not None:
            background_display.set_data(background_runtime_state.current_background_display)
            background_display.set_clim(
                display_controls_view_state.background_min_var.get(),
                display_controls_view_state.background_max_var.get(),
            )
            background_display.set_visible(True)
        else:
            background_display.set_visible(False)
        
    # 1D integration
    if analysis_view_controls_view_state.show_1d_var.get() and simulation_runtime_state.unscaled_image is not None:
        if sim_res2 is None:
            if (
                simulation_runtime_state.caking_cache.get("sim_sig") == sim_caking_sig
                and simulation_runtime_state.caking_cache.get("sim_res2") is not None
            ):
                sim_res2 = simulation_runtime_state.caking_cache["sim_res2"]
            else:
                sim_res2 = caking(simulation_runtime_state.unscaled_image, ai)
                simulation_runtime_state.caking_cache["sim_sig"] = sim_caking_sig
                simulation_runtime_state.caking_cache["sim_res2"] = sim_res2

        if background_runtime_state.visible and native_background is not None:
            if (
                simulation_runtime_state.caking_cache.get("bg_sig") == bg_caking_sig
                and simulation_runtime_state.caking_cache.get("bg_res2") is not None
            ):
                bg_res2 = simulation_runtime_state.caking_cache["bg_res2"]
            else:
                bg_res2 = caking(native_background, ai)
                simulation_runtime_state.caking_cache["bg_sig"] = bg_caking_sig
                simulation_runtime_state.caking_cache["bg_res2"] = bg_res2
        else:
            bg_res2 = None
        _update_1d_plots_from_caked(sim_res2, bg_res2)
    else:
        _clear_1d_plot_cache_and_lines()

    simulation_runtime_state.last_res2_sim = sim_res2
    simulation_runtime_state.last_res2_background = bg_res2

    # Keep simulation display limits sticky across regenerated simulations.
    # Users can still change limits manually or reset defaults explicitly.
    apply_scale_factor_to_existing_results(update_limits=False)

    update_integration_region_visuals(ai, sim_res2)
    _refresh_qr_cylinder_overlay(redraw=True, update_status=False)
    if _live_geometry_preview_enabled():
        if gui_controllers.consume_geometry_preview_skip_once(geometry_preview_state):
            _clear_geometry_preview_artists()
        else:
            _refresh_live_geometry_preview(update_status=True)
    else:
        gui_controllers.clear_geometry_preview_skip_once(geometry_preview_state)
        _clear_geometry_preview_artists()
    _render_current_geometry_manual_pairs(update_status=False)

    redraw_update_elapsed_ms = (perf_counter() - redraw_update_start_time) * 1e3
    total_update_elapsed_ms = (perf_counter() - update_start_time) * 1e3
    image_generation_text = (
        "cached" if image_generation_cached else f"{image_generation_elapsed_ms:.1f} ms"
    )
    if "update_timing_label" in globals():
        update_timing_label.config(
            text=(
                "Timing | image generation: "
                f"{image_generation_text} | redraw/update: "
                f"{redraw_update_elapsed_ms:.1f} ms | total: "
                f"{total_update_elapsed_ms:.1f} ms"
            )
        )

    # mark update completion so future updates can run
    simulation_runtime_state.update_running = False

# ── after you’ve updated background_runtime_state.visible in toggle_background() ──
def toggle_background():
    background_runtime_state.visible = not background_runtime_state.visible
    _sync_background_runtime_state()
    # ↓ force opaque if the background is hidden, 0.5 otherwise
    image_display.set_alpha(0.5 if background_runtime_state.visible else 1.0)
    schedule_update()


def _set_background_file_status_text():
    """Refresh the background-file status line shown in the GUI."""
    gui_background_manager.set_background_file_status_from_state(
        view_state=workspace_panels_view_state,
        background_state=background_runtime_state,
        current_background_theta_values=(
            lambda: _current_background_theta_values(strict_count=False)
        ),
        background_theta_for_index=(
            lambda idx: _background_theta_for_index(idx, strict_count=False)
        ),
        geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
        geometry_manual_pairs_for_index=_geometry_manual_pairs_for_index,
        geometry_manual_pair_group_count=_geometry_manual_pair_group_count,
        current_geometry_fit_background_indices=(
            lambda: _current_geometry_fit_background_indices(strict=False)
        ),
    )


def _load_background_files(file_paths, *, select_index=0):
    """Replace background images from selected OSC file paths."""
    gui_background_manager.load_background_files_with_side_effects(
        background_runtime_state,
        file_paths,
        image_size=image_size,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
        sync_background_runtime_state=_sync_background_runtime_state,
        replace_geometry_manual_pairs_by_background=_replace_geometry_manual_pairs_by_background,
        invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
        clear_geometry_manual_undo_stack=_clear_geometry_manual_undo_stack,
        clear_geometry_fit_undo_stack=_clear_geometry_fit_undo_stack,
        set_geometry_manual_pick_mode=_set_geometry_manual_pick_mode,
        set_background_display_data=background_display.set_data,
        update_background_slider_defaults=(
            lambda image: _update_background_slider_defaults(
                image,
                reset_override=True,
            )
        ),
        sync_background_theta_controls=(
            lambda: _sync_background_theta_controls(
                preserve_existing=True,
                trigger_update=False,
            )
        ),
        sync_geometry_fit_background_selection=(
            lambda: _sync_geometry_fit_background_selection(
                preserve_existing=True,
            )
        ),
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        refresh_background_file_status=_set_background_file_status_text,
        schedule_update=schedule_update,
        select_index=select_index,
    )


def _browse_background_files():
    """Choose one or more OSC backgrounds and load them into the GUI."""

    initial_dir = gui_background_manager.background_file_dialog_initial_dir(
        background_runtime_state.osc_files,
        background_runtime_state.current_background_index,
        get_dir("file_dialog_dir"),
    )

    selected = filedialog.askopenfilenames(
        title="Select Background OSC Files",
        initialdir=initial_dir,
        filetypes=[("OSC files", "*.osc *.OSC"), ("All files", "*.*")],
    )
    if not selected:
        return

    try:
        _load_background_files(selected, select_index=0)
        progress_label.config(text=f"Loaded {len(background_runtime_state.osc_files)} background file(s).")
    except Exception as exc:
        progress_label.config(text=f"Failed to load background files: {exc}")


def switch_background():
    if not background_runtime_state.osc_files:
        progress_label.config(text="No background images loaded.")
        return

    try:
        gui_background_manager.switch_background_with_side_effects(
            background_runtime_state,
            display_rotate_k=DISPLAY_ROTATE_K,
            read_osc=read_osc,
            sync_background_runtime_state=_sync_background_runtime_state,
            invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
            clear_geometry_manual_undo_stack=_clear_geometry_manual_undo_stack,
            clear_geometry_fit_undo_stack=_clear_geometry_fit_undo_stack,
            sync_theta_initial_to_background=(
                (
                    lambda idx: theta_initial_var.set(
                        _background_theta_for_index(
                            idx,
                            strict_count=False,
                        )
                    )
                )
                if "theta_initial_var" in globals() and theta_initial_var is not None
                else None
            ),
            set_background_display_data=background_display.set_data,
            update_background_slider_defaults=(
                lambda image: _update_background_slider_defaults(
                    image,
                    reset_override=True,
                )
            ),
            refresh_background_file_status=_set_background_file_status_text,
            render_current_geometry_manual_pairs=(
                lambda: _render_current_geometry_manual_pairs(update_status=False)
            ),
            schedule_update=schedule_update,
        )
    except Exception as exc:
        progress_label.config(text=f"Failed to switch background: {exc}")
        return

def reset_to_defaults():
    _clear_geometry_fit_undo_stack()
    theta_initial_var.set(defaults['theta_initial'])
    if geometry_theta_offset_var is not None:
        geometry_theta_offset_var.set("0.0")
    cor_angle_var.set(defaults['cor_angle'])
    gamma_var.set(defaults['gamma'])
    Gamma_var.set(defaults['Gamma'])
    chi_var.set(defaults['chi'])
    psi_z_var.set(defaults['psi_z'])
    zs_var.set(defaults['zs'])
    zb_var.set(defaults['zb'])
    debye_x_var.set(defaults['debye_x'])
    debye_y_var.set(defaults['debye_y'])
    corto_detector_var.set(defaults['corto_detector'])
    sigma_mosaic_var.set(defaults['sigma_mosaic_deg'])
    gamma_mosaic_var.set(defaults['gamma_mosaic_deg'])
    eta_var.set(defaults['eta'])
    bandwidth_percent_var.set(_clip_bandwidth_percent(defaults.get('bandwidth_percent', bandwidth * 100.0)))
    a_var.set(defaults['a'])
    c_var.set(defaults['c'])
    default_resolution = defaults['sampling_resolution']
    resolution_var.set(default_resolution)
    custom_samples_var.set(
        str(
            int(
                max(
                    1,
                    resolution_sample_counts.get(
                        default_resolution, resolution_sample_counts['Low']
                    ),
                )
            )
        )
    )
    optics_mode_var.set(_normalize_optics_mode_label(defaults.get('optics_mode', 'fast')))
    sf_prune_bias_var.set(_clip_sf_prune_bias(defaults.get('sf_prune_bias', 0.0)))
    solve_q_mode_var.set(_normalize_solve_q_mode_label(defaults.get('solve_q_mode', SOLVE_Q_MODE_UNIFORM)))
    solve_q_steps_var.set(float(_clip_solve_q_steps(defaults.get('solve_q_steps', DEFAULT_SOLVE_Q_STEPS))))
    solve_q_rel_tol_var.set(
        float(_clip_solve_q_rel_tol(defaults.get('solve_q_rel_tol', DEFAULT_SOLVE_Q_REL_TOL)))
    )
    center_x_var.set(defaults['center_x'])
    center_y_var.set(defaults['center_y'])
    _sync_background_theta_controls(preserve_existing=False, trigger_update=False)
    if geometry_fit_background_selection_var is not None:
        geometry_fit_background_selection_var.set(_default_geometry_fit_background_selection())
        _apply_geometry_fit_background_selection(trigger_update=False)
    tth_min_var.set(0.0)
    tth_max_var.set(80.0)
    phi_min_var.set(-15.0)
    phi_max_var.set(15.0)
    analysis_view_controls_view_state.show_1d_var.set(False)
    analysis_view_controls_view_state.show_caked_2d_var.set(False)
    vmin_caked_var.set(0.0)
    vmax_caked_var.set(2000.0)
    simulation_runtime_state.caked_limits_user_override = False

    display_controls_state.background_limits_user_override = False
    display_controls_state.simulation_limits_user_override = False
    display_controls_state.scale_factor_user_override = False

    _update_background_slider_defaults(background_runtime_state.current_background_display, reset_override=True)

    display_controls_state.suppress_simulation_limit_callback = True
    display_controls_view_state.simulation_min_var.set(0.0)
    display_controls_view_state.simulation_max_var.set(background_vmax_default)
    display_controls_state.suppress_simulation_limit_callback = False

    _set_scale_factor_value(1.0, adjust_range=False, reset_override=True)

    # ALSO reset occupancies to defaults for all configured unique atom sites.
    for idx, occ_var in enumerate(occ_vars):
        default_occ = occ[idx] if idx < len(occ) else occ[-1]
        occ_var.set(default_occ)
    for idx, axis_vars in enumerate(atom_site_fract_vars):
        if idx >= len(atom_site_fractional_metadata):
            break
        row = atom_site_fractional_metadata[idx]
        axis_vars["x"].set(float(row["x"]))
        axis_vars["y"].set(float(row["y"]))
        axis_vars["z"].set(float(row["z"]))
    _reset_atom_site_override_cache()
    p0_var.set(defaults['p0'])
    p1_var.set(defaults['p1'])
    p2_var.set(defaults['p2'])
    w0_var.set(defaults['w0'])
    w1_var.set(defaults['w1'])
    w2_var.set(defaults['w2'])
    finite_stack_var.set(defaults['finite_stack'])
    stack_layers_var.set(int(defaults['stack_layers']))
    phase_delta_expr_var.set(str(defaults['phase_delta_expression']))
    phi_l_divisor_var.set(float(defaults['phi_l_divisor']))
    if finite_stack_controls_view_state.phase_delta_entry_var is not None:
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            _current_phase_delta_expression(),
        )
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
        gui_views.set_finite_stack_phi_l_divisor_entry_text(
            finite_stack_controls_view_state,
            gui_controllers.format_finite_stack_phi_l_divisor(_current_phi_l_divisor()),
        )
    _sync_finite_controls()

    update_mosaic_cache()
    simulation_runtime_state.last_simulation_signature = None
    schedule_update()

gui_views.populate_stacked_button_group(
    workspace_panels_view_state.workspace_actions_frame,
    [
        ("Toggle Background", toggle_background),
        ("Switch Background", switch_background),
    ],
)

gui_views.create_background_file_controls(
    parent=workspace_panels_view_state.workspace_backgrounds_frame,
    view_state=workspace_panels_view_state,
    on_load_backgrounds=_browse_background_files,
    status_text="",
)
_set_background_file_status_text()

gui_views.create_background_theta_controls(
    parent=workspace_panels_view_state.workspace_backgrounds_frame,
    view_state=background_theta_controls_view_state,
    background_theta_values_text=_format_background_theta_values(
        [_background_theta_default_value()] * len(background_runtime_state.osc_files)
    ),
    geometry_theta_offset_text="0.0",
    on_apply=lambda: _apply_background_theta_metadata(trigger_update=True),
)
background_theta_list_var = background_theta_controls_view_state.background_theta_list_var
geometry_theta_offset_var = background_theta_controls_view_state.geometry_theta_offset_var
_sync_background_theta_controls(preserve_existing=True, trigger_update=False)

gui_views.create_geometry_fit_background_controls(
    parent=app_shell_view_state.fit_body,
    view_state=background_theta_controls_view_state,
    selection_text=_default_geometry_fit_background_selection(),
    on_apply=lambda: _apply_geometry_fit_background_selection(trigger_update=True),
)
geometry_fit_background_selection_var = (
    background_theta_controls_view_state.geometry_fit_background_selection_var
)
_sync_geometry_fit_background_selection(preserve_existing=False)

gui_views.populate_stacked_button_group(
    workspace_panels_view_state.workspace_actions_frame,
    [
        ("Reset to Defaults", reset_to_defaults),
        (
            "Azim vs Radial Plot Demo",
            lambda: view_azimuthal_radial(
                simulate_diffraction(
                    theta_initial=theta_initial_var.get(),
                    cor_angle=cor_angle_var.get(),
                    gamma=gamma_var.get(),
                    Gamma=Gamma_var.get(),
                    chi=chi_var.get(),
                    psi_z=psi_z_var.get(),
                    zs=zs_var.get(),
                    zb=zb_var.get(),
                    debye_x_value=debye_x_var.get(),
                    debye_y_value=debye_y_var.get(),
                    corto_detector_value=corto_detector_var.get(),
                    miller=miller,
                    intensities=intensities,
                    image_size=image_size,
                    av=a_var.get(),
                    cv=c_var.get(),
                    lambda_=lambda_,
                    psi=psi,
                    n2=n2,
                    center=[center_x_var.get(), center_y_var.get()],
                    num_samples=simulation_runtime_state.num_samples,
                    divergence_sigma=divergence_sigma,
                    bw_sigma=bw_sigma,
                    sigma_mosaic_var=sigma_mosaic_var,
                    gamma_mosaic_var=gamma_mosaic_var,
                    eta_var=eta_var,
                    bandwidth=_current_bandwidth_fraction(),
                    optics_mode=_current_optics_mode_flag(),
                    solve_q_steps=_current_solve_q_steps(),
                    solve_q_rel_tol=_current_solve_q_rel_tol(),
                    solve_q_mode=_current_solve_q_mode_flag(),
                ),
                [center_x_var.get(), center_y_var.get()],
                {
                    'pixel_size': pixel_size_m,
                    'poni1': (center_x_var.get()) * pixel_size_m,
                    'poni2': (center_y_var.get()) * pixel_size_m,
                    'dist': corto_detector_var.get(),
                    'rot1': 0.0,
                    'rot2': 0.0,
                    'rot3': 0.0,
                    'wavelength': wave_m
                }
            ),
        ),
    ],
)

gui_views.create_status_panel(
    parent=app_shell_view_state.status_frame,
    view_state=status_panel_view_state,
)
progress_label_positions = status_panel_view_state.progress_label_positions
progress_label_geometry = status_panel_view_state.progress_label_geometry
mosaic_progressbar = status_panel_view_state.mosaic_progressbar
progress_label_mosaic = status_panel_view_state.progress_label_mosaic
progress_label = status_panel_view_state.progress_label
update_timing_label = status_panel_view_state.update_timing_label
chi_square_label = status_panel_view_state.chi_square_label
if (
    progress_label_positions is None
    or progress_label_geometry is None
    or mosaic_progressbar is None
    or progress_label_mosaic is None
    or progress_label is None
    or update_timing_label is None
    or chi_square_label is None
):
    raise RuntimeError("Status panel was not created.")

hbn_geometry_debug_view_state.report_text = (
    "No hBN geometry debug report yet.\n"
    "Import an hBN bundle to generate one."
)


def _close_hbn_geometry_debug_window() -> None:
    gui_views.close_hbn_geometry_debug_window(hbn_geometry_debug_view_state)


def show_last_hbn_geometry_debug():
    """Display the most recent hBN->simulation geometry debug report."""

    gui_views.open_hbn_geometry_debug_window(
        root=root,
        view_state=hbn_geometry_debug_view_state,
        text=str(hbn_geometry_debug_view_state.report_text),
        on_close=_close_hbn_geometry_debug_window,
    )


def import_hbn_tilt_from_bundle():
    """Load an hBN ellipse bundle NPZ and apply its tilt hint to the GUI sliders."""

    bundle_path = filedialog.askopenfilename(
        title="Select hBN bundle (.npz)",
        filetypes=[("hBN bundle", "*.npz"), ("All files", "*.*")],
    )
    if not bundle_path:
        return

    mean_dist = None
    tilt_x_deg = None
    tilt_y_deg = None
    source_rotate_k = int(HBN_FITTER_ROTATE_K)
    gamma_sign_from_tilt_x = 1
    gamma_sign_from_tilt_y = 1
    imported_center = None

    def _normalize_sign(value, default=1):
        try:
            iv = int(value)
        except Exception:
            iv = int(default)
        if iv < 0:
            return -1
        if iv > 0:
            return 1
        return -1 if int(default) < 0 else 1

    try:
        _, _, _, _, distance_info, tilt_correction, tilt_hint, _, imported_center = load_bundle_npz(
            bundle_path
        )
    except Exception as exc:  # pragma: no cover - GUI interaction
        progress_label.config(text=f"Failed to load bundle: {exc}")
        return

    if isinstance(tilt_correction, dict):
        try:
            tx = float(tilt_correction.get("tilt_x_deg"))
            ty = float(tilt_correction.get("tilt_y_deg"))
            if np.isfinite(tx) and np.isfinite(ty):
                tilt_x_deg = tx
                tilt_y_deg = ty
        except (TypeError, ValueError):
            pass
        source_rotate_k = int(
            tilt_correction.get("sim_background_rotate_k", source_rotate_k)
        )
        gamma_sign_from_tilt_x = _normalize_sign(
            tilt_correction.get("simulation_gamma_sign_from_tilt_x", gamma_sign_from_tilt_x),
            gamma_sign_from_tilt_x,
        )
        gamma_sign_from_tilt_y = _normalize_sign(
            tilt_correction.get("simulation_Gamma_sign_from_tilt_y", gamma_sign_from_tilt_y),
            gamma_sign_from_tilt_y,
        )

    if (tilt_x_deg is None or tilt_y_deg is None) and isinstance(tilt_hint, dict):
        try:
            rx = float(tilt_hint.get("rot1_rad"))
            ry = float(tilt_hint.get("rot2_rad"))
            if np.isfinite(rx) and np.isfinite(ry):
                tilt_x_deg = float(np.degrees(rx))
                tilt_y_deg = float(np.degrees(ry))
        except (TypeError, ValueError):
            pass
        source_rotate_k = int(tilt_hint.get("sim_background_rotate_k", source_rotate_k))
        gamma_sign_from_tilt_x = _normalize_sign(
            tilt_hint.get("simulation_gamma_sign_from_tilt_x", gamma_sign_from_tilt_x),
            gamma_sign_from_tilt_x,
        )
        gamma_sign_from_tilt_y = _normalize_sign(
            tilt_hint.get("simulation_Gamma_sign_from_tilt_y", gamma_sign_from_tilt_y),
            gamma_sign_from_tilt_y,
        )

    if tilt_x_deg is None or tilt_y_deg is None:
        progress_label.config(text="Bundle loaded, but no tilt information was found.")
        return

    gamma_sign_from_tilt_x = -_normalize_sign(gamma_sign_from_tilt_x, 1)

    converted = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=float(tilt_x_deg),
        tilt_y_deg=float(tilt_y_deg),
        center_xy=imported_center,
        source_rotate_k=int(source_rotate_k),
        target_rotate_k=int(SIMULATION_GEOMETRY_ROTATE_K),
        image_size=(int(image_size), int(image_size)),
        simulation_gamma_sign_from_tilt_x=int(gamma_sign_from_tilt_x),
        simulation_Gamma_sign_from_tilt_y=int(gamma_sign_from_tilt_y),
    )

    def _ensure_slider_includes(slider_widget, value, pad=0.1):
        try:
            lo = float(slider_widget.cget("from"))
            hi = float(slider_widget.cget("to"))
        except Exception:
            return False
        if lo > hi:
            lo, hi = hi, lo
        val = float(value)
        new_lo = lo
        new_hi = hi
        if val < lo:
            new_lo = val - float(pad)
        if val > hi:
            new_hi = val + float(pad)
        if math.isclose(new_lo, lo, rel_tol=0.0, abs_tol=1e-12) and math.isclose(
            new_hi, hi, rel_tol=0.0, abs_tol=1e-12
        ):
            return False
        slider_widget.configure(from_=new_lo, to=new_hi)
        return True

    gamma_sim_deg = float(converted["gamma_deg"])
    Gamma_sim_deg = float(converted["Gamma_deg"])
    angle_range_expanded = False
    angle_range_expanded |= _ensure_slider_includes(gamma_scale, gamma_sim_deg, pad=0.1)
    angle_range_expanded |= _ensure_slider_includes(Gamma_scale, Gamma_sim_deg, pad=0.1)
    gamma_var.set(gamma_sim_deg)
    Gamma_var.set(Gamma_sim_deg)
    if distance_info and isinstance(distance_info, dict):
        mean_dist = distance_info.get("mean_m")
        if mean_dist is not None:
            try:
                _ensure_slider_includes(corto_detector_scale, float(mean_dist), pad=0.001)
                corto_detector_var.set(float(mean_dist))
            except Exception:
                pass
    center_text = ""
    center_row = converted.get("center_row")
    center_col = converted.get("center_col")
    if center_row is not None and center_col is not None:
        try:
            center_row = float(center_row)
            center_col = float(center_col)
            if np.isfinite(center_row) and np.isfinite(center_col):
                _ensure_slider_includes(center_x_scale, center_row, pad=5.0)
                _ensure_slider_includes(center_y_scale, center_col, pad=5.0)
                center_x_var.set(center_row)
                center_y_var.set(center_col)
                center_text = (
                    f" and center (row={center_row:.2f}, col={center_col:.2f}) px"
                )
        except Exception:
            pass
    applied_center_row = None
    applied_center_col = None
    try:
        applied_center_row = float(center_x_var.get())
        applied_center_col = float(center_y_var.get())
    except Exception:
        applied_center_row = None
        applied_center_col = None

    debug_text = ""
    if HBN_GEOMETRY_DEBUG_ENABLED:
        debug_trace = build_hbn_geometry_debug_trace(
            npz_center_xy=imported_center,
            source_rotate_k=int(source_rotate_k),
            target_rotate_k=int(SIMULATION_GEOMETRY_ROTATE_K),
            image_size=(int(image_size), int(image_size)),
            tilt_x_deg=float(tilt_x_deg),
            tilt_y_deg=float(tilt_y_deg),
            simulation_gamma_sign_from_tilt_x=int(gamma_sign_from_tilt_x),
            simulation_Gamma_sign_from_tilt_y=int(gamma_sign_from_tilt_y),
            simulation_center_row=applied_center_row,
            simulation_center_col=applied_center_col,
        )
        hbn_geometry_debug_view_state.report_text = format_hbn_geometry_debug_trace(debug_trace)
        gui_views.set_hbn_geometry_debug_text(
            hbn_geometry_debug_view_state,
            str(hbn_geometry_debug_view_state.report_text),
        )
        print(hbn_geometry_debug_view_state.report_text)
        debug_text = " [debug report updated]"

    schedule_update()
    progress_label.config(
        text=(
            "Applied hBN bundle tilt hint "
            f"(γ={gamma_sim_deg:.3f}°, Γ={Gamma_sim_deg:.3f}°)"
            + (" [range expanded]" if angle_range_expanded else "")
            + (
                f" and distance {mean_dist:.4f} m"  # type: ignore[arg-type]
                if distance_info and mean_dist is not None
                else ""
            )
            + center_text
            + debug_text
            + f" from {bundle_path}"
        )
    )


def _gui_state_variable_items() -> dict[str, object]:
    """Return Tk variables that participate in full GUI-state snapshots."""

    items = dict(globals())
    items.update(
        {
            "background_min_var": display_controls_view_state.background_min_var,
            "background_max_var": display_controls_view_state.background_max_var,
            "background_transparency_var": (
                display_controls_view_state.background_transparency_var
            ),
            "simulation_min_var": display_controls_view_state.simulation_min_var,
            "simulation_max_var": display_controls_view_state.simulation_max_var,
            "simulation_scale_factor_var": (
                display_controls_view_state.simulation_scale_factor_var
            ),
        }
    )
    return items


def _collect_full_gui_state_snapshot() -> dict[str, object]:
    return gui_state_io.collect_full_gui_state_snapshot(
        global_items=_gui_state_variable_items(),
        tk_variable_type=tk.Variable,
        occ_vars=globals().get("occ_vars", []),
        atom_site_fract_vars=globals().get("atom_site_fract_vars", []),
        geometry_q_group_rows=_geometry_q_group_export_rows(),
        geometry_manual_pairs=_geometry_manual_pairs_export_rows(),
        selected_hkl_target=peak_selection_state.selected_hkl_target,
        primary_cif_path=cif_file,
        secondary_cif_path=cif_file2,
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        background_visible=background_runtime_state.visible,
        background_backend_rotation_k=background_runtime_state.backend_rotation_k,
        background_backend_flip_x=background_runtime_state.backend_flip_x,
        background_backend_flip_y=background_runtime_state.backend_flip_y,
        background_limits_user_override=(
            display_controls_state.background_limits_user_override
        ),
        simulation_limits_user_override=(
            display_controls_state.simulation_limits_user_override
        ),
        scale_factor_user_override=display_controls_state.scale_factor_user_override,
    )


def _apply_full_gui_state_snapshot(snapshot: dict[str, object]) -> str:

    if not isinstance(snapshot, dict):
        snapshot = {}

    warnings: list[str] = []

    warnings.extend(
        gui_state_io.apply_gui_state_files(
            snapshot.get("files", {}),
            apply_primary_cif_path=_apply_primary_cif_path,
            load_background_files=(
                lambda file_paths, select_index: _load_background_files(
                    file_paths,
                    select_index=select_index,
                )
            ),
        )
    )

    variables = snapshot.get("variables", {})
    warnings.extend(
        gui_state_io.apply_gui_state_variables(
            variables,
            global_items=_gui_state_variable_items(),
            tk_variable_type=tk.Variable,
        )
    )

    gui_state_io.apply_gui_state_background_theta_compatibility(
        variables,
        osc_files=background_runtime_state.osc_files,
        theta_initial_var=theta_initial_var,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        format_background_theta_values=_format_background_theta_values,
        default_geometry_fit_background_selection=_default_geometry_fit_background_selection,
    )

    gui_state_io.apply_dynamic_gui_state_lists(
        snapshot.get("dynamic_lists", {}),
        occ_vars=globals().get("occ_vars", []),
        atom_site_fract_vars=globals().get("atom_site_fract_vars", []),
    )

    flag_state = gui_state_io.apply_gui_state_flags(
        snapshot.get("flags", {}),
        current_flags={
            "background_visible": background_runtime_state.visible,
            "background_backend_rotation_k": background_runtime_state.backend_rotation_k,
            "background_backend_flip_x": background_runtime_state.backend_flip_x,
            "background_backend_flip_y": background_runtime_state.backend_flip_y,
            "background_limits_user_override": (
                display_controls_state.background_limits_user_override
            ),
            "simulation_limits_user_override": (
                display_controls_state.simulation_limits_user_override
            ),
            "scale_factor_user_override": (
                display_controls_state.scale_factor_user_override
            ),
        },
        toggle_background=toggle_background,
    )
    background_runtime_state.visible = bool(flag_state["background_visible"])
    background_runtime_state.backend_rotation_k = int(flag_state["background_backend_rotation_k"])
    background_runtime_state.backend_flip_x = bool(flag_state["background_backend_flip_x"])
    background_runtime_state.backend_flip_y = bool(flag_state["background_backend_flip_y"])
    _sync_background_runtime_state()
    display_controls_state.background_limits_user_override = bool(
        flag_state["background_limits_user_override"]
    )
    display_controls_state.simulation_limits_user_override = bool(
        flag_state["simulation_limits_user_override"]
    )
    display_controls_state.scale_factor_user_override = bool(
        flag_state["scale_factor_user_override"]
    )

    geometry_state = gui_state_io.apply_gui_state_geometry(
        snapshot.get("geometry", {}),
        geometry_preview_excluded_q_groups=geometry_preview_state.excluded_q_groups,
        geometry_q_group_key_from_jsonable=_geometry_q_group_key_from_jsonable,
        invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
        apply_geometry_manual_pairs_snapshot=_apply_geometry_manual_pairs_snapshot,
        current_background_index=background_runtime_state.current_background_index,
        selected_hkl_target=peak_selection_state.selected_hkl_target,
    )
    peak_selection_state.selected_hkl_target = geometry_state["selected_hkl_target"]
    _sync_peak_selection_state()
    warnings.extend(list(geometry_state["warnings"]))

    if finite_stack_controls_view_state.phase_delta_entry_var is not None:
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            _current_phase_delta_expression(),
        )
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
        gui_views.set_finite_stack_phi_l_divisor_entry_text(
            finite_stack_controls_view_state,
            gui_controllers.format_finite_stack_phi_l_divisor(_current_phi_l_divisor()),
        )
    _sync_finite_controls()
    _apply_geometry_fit_background_selection(trigger_update=False)
    _set_background_file_status_text()
    _update_geometry_preview_exclude_button_label()
    _refresh_geometry_q_group_window()
    _update_hkl_pick_button_label()
    ensure_valid_resolution_choice()
    toggle_1d_plots()
    toggle_caked_2d()
    toggle_log_radial()
    toggle_log_azimuth()
    _update_background_backend_status()
    _mark_chi_square_dirty()
    _update_chi_square_display(force=True)
    _refresh_qr_cylinder_overlay(redraw=True, update_status=False)
    _refresh_live_geometry_preview(update_status=False)
    schedule_update()

    return gui_state_io.build_gui_state_import_summary(warnings)


def _export_full_gui_state() -> None:
    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    file_path = filedialog.asksaveasfilename(
        title="Export Full GUI State",
        initialdir=initial_dir,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=f"ra_sim_gui_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    if not file_path:
        progress_label.config(text="GUI state export canceled.")
        return
    try:
        save_gui_state_file(
            file_path,
            _collect_full_gui_state_snapshot(),
            metadata={"entrypoint": "main.py"},
        )
    except Exception as exc:
        progress_label.config(text=f"Failed to export GUI state: {exc}")
        return
    progress_label.config(text=f"Saved GUI state to {file_path}")


def _import_full_gui_state() -> None:
    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    file_path = filedialog.askopenfilename(
        title="Import Full GUI State",
        initialdir=initial_dir,
        filetypes=[
            ("RA-SIM GUI state", "*.json"),
            ("Legacy parameter profiles", "*.npy"),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        progress_label.config(text="GUI state import canceled.")
        return

    suffix = Path(file_path).suffix.lower()
    if suffix == ".npy":
        message = load_parameters(
            file_path,
            theta_initial_var,
            cor_angle_var,
            gamma_var,
            Gamma_var,
            chi_var,
            zs_var,
            zb_var,
            debye_x_var,
            debye_y_var,
            corto_detector_var,
            sigma_mosaic_var,
            gamma_mosaic_var,
            eta_var,
            a_var,
            c_var,
            center_x_var,
            center_y_var,
            resolution_var,
            custom_samples_var,
            bandwidth_percent_var=bandwidth_percent_var,
            optics_mode_var=optics_mode_var,
            phase_delta_expr_var=phase_delta_expr_var,
            phi_l_divisor_var=phi_l_divisor_var,
            sf_prune_bias_var=sf_prune_bias_var,
            solve_q_steps_var=solve_q_steps_var,
            solve_q_rel_tol_var=solve_q_rel_tol_var,
            solve_q_mode_var=solve_q_mode_var,
        )
        if finite_stack_controls_view_state.phase_delta_entry_var is not None:
            gui_views.set_finite_stack_phase_delta_entry_text(
                finite_stack_controls_view_state,
                _current_phase_delta_expression(),
            )
        if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
            gui_views.set_finite_stack_phi_l_divisor_entry_text(
                finite_stack_controls_view_state,
                gui_controllers.format_finite_stack_phi_l_divisor(
                    _current_phi_l_divisor()
                ),
            )
        _sync_finite_controls()
        ensure_valid_resolution_choice()
        schedule_update()
        progress_label.config(text=message)
        return

    try:
        payload = load_gui_state_file(file_path)
        message = _apply_full_gui_state_snapshot(payload.get("state", {}))
    except Exception as exc:
        progress_label.config(text=f"Failed to import GUI state: {exc}")
        return
    progress_label.config(text=message)


def _export_geometry_manual_pairs() -> None:
    """Write the saved manual geometry placements to a JSON file."""

    if not any(_geometry_manual_pairs_for_index(idx) for idx in range(len(background_runtime_state.osc_files))):
        progress_label_geometry.config(text="No saved manual placements are available to export.")
        return

    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    file_path = filedialog.asksaveasfilename(
        title="Export Geometry Placements",
        initialdir=initial_dir,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=f"ra_sim_geometry_placements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    if not file_path:
        progress_label_geometry.config(text="Geometry placement export canceled.")
        return

    try:
        save_geometry_placements_file(
            file_path,
            _collect_geometry_manual_pairs_snapshot(),
            metadata={"entrypoint": "main.py"},
        )
    except Exception as exc:
        progress_label_geometry.config(text=f"Failed to export geometry placements: {exc}")
        return
    progress_label_geometry.config(text=f"Saved manual geometry placements to {file_path}")


def _import_geometry_manual_pairs() -> None:
    """Load manual geometry placements from a JSON file."""

    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    file_path = filedialog.askopenfilename(
        title="Import Geometry Placements",
        initialdir=initial_dir,
        filetypes=[("RA-SIM geometry placements", "*.json"), ("All files", "*.*")],
    )
    if not file_path:
        progress_label_geometry.config(text="Geometry placement import canceled.")
        return

    try:
        payload = load_geometry_placements_file(file_path)
        message = _apply_geometry_manual_pairs_snapshot(
            payload.get("state", {}),
            allow_background_reload=True,
        )
    except Exception as exc:
        progress_label_geometry.config(text=f"Failed to import geometry placements: {exc}")
        return
    try:
        _ensure_geometry_fit_caked_view(force_refresh=True)
    except Exception as exc:
        message += f" Warning: imported placements but could not switch to 2D caked view ({exc})."
    progress_label_geometry.config(text=message)


session_button_specs = [
    ("Export GUI State...", _export_full_gui_state),
    ("Import GUI State...", _import_full_gui_state),
    ("Import hBN Bundle Tilt", import_hbn_tilt_from_bundle),
]
if HBN_GEOMETRY_DEBUG_ENABLED:
    session_button_specs.append(
        ("Show hBN Geometry Debug", show_last_hbn_geometry_debug)
    )
gui_views.populate_stacked_button_group(
    workspace_panels_view_state.workspace_session_frame,
    session_button_specs,
)

# Frame for selecting which geometry params to fit
gui_views.create_geometry_fit_parameter_controls(
    parent=app_shell_view_state.fit_body,
    view_state=geometry_fit_parameter_controls_view_state,
    initial_values={
        "zb": True,
        "zs": True,
        "theta_initial": True,
        "psi_z": True,
        "chi": True,
        "cor_angle": True,
        "gamma": True,
        "Gamma": True,
        "corto_detector": True,
        "a": False,
        "c": False,
        "center_x": False,
        "center_y": False,
    },
)
fit_frame = geometry_fit_parameter_controls_view_state.frame
fit_zb_var = geometry_fit_parameter_controls_view_state.fit_zb_var
fit_zs_var = geometry_fit_parameter_controls_view_state.fit_zs_var
fit_theta_var = geometry_fit_parameter_controls_view_state.fit_theta_var
fit_psi_z_var = geometry_fit_parameter_controls_view_state.fit_psi_z_var
fit_chi_var = geometry_fit_parameter_controls_view_state.fit_chi_var
fit_cor_var = geometry_fit_parameter_controls_view_state.fit_cor_var
fit_gamma_var = geometry_fit_parameter_controls_view_state.fit_gamma_var
fit_Gamma_var = geometry_fit_parameter_controls_view_state.fit_Gamma_var
fit_dist_var = geometry_fit_parameter_controls_view_state.fit_dist_var
fit_a_var = geometry_fit_parameter_controls_view_state.fit_a_var
fit_c_var = geometry_fit_parameter_controls_view_state.fit_c_var
fit_center_x_var = geometry_fit_parameter_controls_view_state.fit_center_x_var
fit_center_y_var = geometry_fit_parameter_controls_view_state.fit_center_y_var
fit_theta_checkbutton = geometry_fit_parameter_controls_view_state.fit_theta_checkbutton
if fit_frame is None:
    raise RuntimeError("Geometry-fit parameter controls did not create the frame.")
_refresh_geometry_fit_theta_checkbox_label()

GEOMETRY_FIT_PARAM_ORDER = [
    *gui_geometry_fit.GEOMETRY_FIT_PARAM_ORDER,
]
geometry_fit_toggle_vars = dict(geometry_fit_parameter_controls_view_state.toggle_vars)
geometry_fit_parameter_specs = {}


def _sync_geometry_fit_constraint_rows(*_args) -> None:
    controls = geometry_fit_constraints_view_state.controls
    for name in GEOMETRY_FIT_PARAM_ORDER:
        control = controls.get(name)
        if not isinstance(control, dict):
            continue
        row = control.get("row")
        if row is None:
            continue
        toggle_var = geometry_fit_toggle_vars.get(name)
        enabled = bool(toggle_var.get()) if toggle_var is not None else True
        mapped = bool(control.get("_mapped", False))
        if enabled and not mapped:
            row.pack(fill=tk.X, padx=4, pady=4)
            control["_mapped"] = True
        elif not enabled and mapped:
            row.pack_forget()
            control["_mapped"] = False

if BACKGROUND_BACKEND_DEBUG_UI_ENABLED:
    gui_views.create_background_backend_debug_controls(
        parent=app_shell_view_state.fit_body,
        view_state=background_backend_debug_view_state,
        status_text=_background_backend_status(),
        on_rotate_minus_90=lambda: _rotate_background_backend(-1),
        on_rotate_plus_90=lambda: _rotate_background_backend(1),
        on_flip_x=lambda: _toggle_background_backend_flip("x"),
        on_flip_y=lambda: _toggle_background_backend_flip("y"),
        on_reset=_reset_background_backend_orientation,
    )

if DEBUG_ENABLED and BACKEND_ORIENTATION_UI_ENABLED:
    gui_views.create_backend_orientation_debug_controls(
        parent=app_shell_view_state.fit_body,
        view_state=background_backend_debug_view_state,
    )


def _center_from_maxpos_entry(entry: Sequence[float]) -> tuple[float, float] | None:
    """Return the strongest finite max position stored in ``entry``."""

    if entry is None or len(entry) < 6:
        return None
    i0, x0, y0, i1, x1, y1 = entry
    primary_valid = np.isfinite(x0) and np.isfinite(y0)
    secondary_valid = np.isfinite(x1) and np.isfinite(y1)

    if primary_valid and (not secondary_valid or not np.isfinite(i1) or float(i0) >= float(i1)):
        return float(x0), float(y0)
    if secondary_valid:
        return float(x1), float(y1)
    if primary_valid:
        return float(x0), float(y0)
    return None


def _aggregate_simulated_peaks_from_max_positions(
    max_positions: Sequence[Sequence[float]],
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
) -> list[dict[str, object]]:
    """Aggregate one simulated peak center per integer HKL from cached maxima."""

    miller_arr = np.asarray(miller_array, dtype=float)
    intensity_arr = np.asarray(intensity_array, dtype=float).reshape(-1)
    if miller_arr.ndim != 2 or miller_arr.shape[1] < 3:
        return []

    row_count = min(len(max_positions), miller_arr.shape[0], intensity_arr.shape[0])
    if row_count <= 0:
        return []

    centers_by_hkl: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    weights_by_hkl: dict[tuple[int, int, int], float] = {}
    for idx in range(row_count):
        H, K, L = miller_arr[idx, :3]
        key = (int(round(H)), int(round(K)), int(round(L)))
        center = _center_from_maxpos_entry(max_positions[idx])
        if center is None:
            continue
        centers_by_hkl.setdefault(key, []).append(center)
        weights_by_hkl[key] = weights_by_hkl.get(key, 0.0) + float(
            abs(intensity_arr[idx])
        )

    simulated_peaks: list[dict[str, object]] = []
    for key, center_list in centers_by_hkl.items():
        arr = np.asarray(center_list, dtype=float)
        simulated_peaks.append(
            {
                "hkl": key,
                "label": f"{key[0]},{key[1]},{key[2]}",
                "sim_col": float(arr[:, 0].mean()),
                "sim_row": float(arr[:, 1].mean()),
                "weight": float(weights_by_hkl.get(key, 0.0)),
            }
        )
    return simulated_peaks


def _build_simulated_peaks_from_hit_tables(
    hit_tables: Sequence[object],
    image_shape: tuple[int, int],
    peak_table_lattice: Sequence[Sequence[object]] | None = None,
) -> list[dict[str, object]]:
    """Build simulated peaks from all detector solutions of the selected beam sample."""

    simulated_peaks: list[dict[str, object]] = []

    for table_idx, tbl in enumerate(hit_tables):
        rows = _geometry_reference_hit_rows(tbl)
        if not rows:
            continue

        source_label = f"table_{table_idx}"
        av_used = float(a_var.get())
        cv_used = float(c_var.get())
        if (
            peak_table_lattice is not None
            and table_idx < len(peak_table_lattice)
            and len(peak_table_lattice[table_idx]) >= 3
        ):
            try:
                av_used = float(peak_table_lattice[table_idx][0])
                cv_used = float(peak_table_lattice[table_idx][1])
                source_label = str(peak_table_lattice[table_idx][2])
            except Exception:
                source_label = f"table_{table_idx}"

        for row_idx, row in enumerate(rows):
            I, xpix, ypix, _phi, H, K, L = row[:7]
            if not (np.isfinite(I) and np.isfinite(xpix) and np.isfinite(ypix)):
                continue

            cx = float(xpix)
            cy = float(ypix)
            disp_cx, disp_cy = _native_sim_to_display_coords(cx, cy, image_shape)
            hkl = tuple(int(np.rint(val)) for val in (H, K, L))
            hkl_raw = (float(H), float(K), float(L))
            q_group_key, qr_val, qz_val = _reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
            )
            if q_group_key is None:
                continue
            simulated_peaks.append(
                {
                    "hkl": hkl,
                    "label": f"{hkl[0]},{hkl[1]},{hkl[2]}",
                    "sim_col": float(disp_cx),
                    "sim_row": float(disp_cy),
                    "weight": float(abs(I)),
                    "source_peak_index": int(len(simulated_peaks)),
                    "source_label": str(source_label),
                    "source_table_index": int(table_idx),
                    "source_row_index": int(row_idx),
                    "hkl_raw": hkl_raw,
                    "av": float(av_used),
                    "cv": float(cv_used),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "q_group_key": q_group_key,
                }
            )

    return simulated_peaks


def _simulate_hit_tables_for_fit(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: dict[str, object],
) -> list[object]:
    """Simulate once and return raw hit tables for geometry fitting."""

    params_local = dict(param_set)
    params_local["mosaic_params"] = build_geometry_fit_central_mosaic_params(params_local)
    mosaic = dict(params_local.get("mosaic_params", {}))
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        wavelength_array = np.full(
            int(np.size(mosaic.get("beam_x_array", []))),
            float(params_local.get("lambda", 1.0)),
            dtype=np.float64,
        )

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    _, hit_tables, *_ = process_peaks_parallel(
        miller_array,
        intensity_array,
        image_size,
        float(params_local["a"]),
        float(params_local["c"]),
        wavelength_array,
        sim_buffer,
        float(params_local["corto_detector"]),
        float(params_local["gamma"]),
        float(params_local["Gamma"]),
        float(params_local["chi"]),
        float(params_local.get("psi", 0.0)),
        float(params_local.get("psi_z", 0.0)),
        float(params_local["zs"]),
        float(params_local["zb"]),
        params_local["n2"],
        np.asarray(mosaic["beam_x_array"], dtype=np.float64),
        np.asarray(mosaic["beam_y_array"], dtype=np.float64),
        np.asarray(mosaic["theta_array"], dtype=np.float64),
        np.asarray(mosaic["phi_array"], dtype=np.float64),
        float(mosaic["sigma_mosaic_deg"]),
        float(mosaic["gamma_mosaic_deg"]),
        float(mosaic["eta"]),
        np.asarray(wavelength_array, dtype=np.float64),
        float(params_local["debye_x"]),
        float(params_local["debye_y"]),
        [float(params_local["center"][0]), float(params_local["center"][1])],
        float(params_local["theta_initial"]),
        float(params_local.get("cor_angle", 0.0)),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0,
        optics_mode=int(params_local.get("optics_mode", 0)),
        solve_q_steps=int(mosaic.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS)),
        solve_q_rel_tol=float(mosaic.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL)),
        solve_q_mode=int(mosaic.get("solve_q_mode", DEFAULT_SOLVE_Q_MODE)),
    )
    return hit_tables


def _simulate_hkl_peak_centers_for_fit(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: dict[str, object],
) -> list[dict[str, object]]:
    """Simulate once and return one aggregated peak center per integer HKL."""

    hit_tables = _simulate_hit_tables_for_fit(
        miller_array,
        intensity_array,
        image_size,
        param_set,
    )
    maxpos = hit_tables_to_max_positions(hit_tables)

    return _aggregate_simulated_peaks_from_max_positions(
        maxpos,
        miller_array,
        intensity_array,
    )


def _simulate_preview_style_peaks_for_fit(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: dict[str, object],
) -> list[dict[str, object]]:
    """Simulate per-hit peaks using the same representation as the live preview."""

    hit_tables = _simulate_hit_tables_for_fit(
        miller_array,
        intensity_array,
        image_size,
        param_set,
    )
    peak_table_lattice = (
        simulation_runtime_state.stored_peak_table_lattice if isinstance(simulation_runtime_state.stored_peak_table_lattice, list) else None
    )
    return _build_simulated_peaks_from_hit_tables(
        hit_tables,
        (int(image_size), int(image_size)),
        peak_table_lattice=peak_table_lattice,
    )


def _auto_match_console(cfg: dict[str, object] | None, text: str) -> None:
    """Emit one console progress line for geometry auto-match when enabled."""

    if not isinstance(cfg, dict) or not bool(cfg.get("console_progress", False)):
        return
    try:
        print(f"[geometry-fit] auto-match: {text}", flush=True)
    except Exception:
        pass


def _auto_match_refine_peak_center(
    peakness: np.ndarray,
    fine_image: np.ndarray,
    row_idx: int,
    col_idx: int,
) -> tuple[float, float]:
    """Refine one summit center to a local weighted centroid."""

    if peakness.ndim != 2:
        return float(col_idx), float(row_idx)
    height, width = peakness.shape
    r0 = max(0, int(row_idx) - 1)
    r1 = min(height, int(row_idx) + 2)
    c0 = max(0, int(col_idx) - 1)
    c1 = min(width, int(col_idx) + 2)
    peak_patch = peakness[r0:r1, c0:c1]
    weight_patch = np.clip(peak_patch, 0.0, None)
    if not np.any(weight_patch > 0.0) and fine_image.ndim == 2:
        fine_patch = fine_image[r0:r1, c0:c1]
        weight_patch = np.clip(fine_patch - np.min(fine_patch), 0.0, None)
    total_weight = float(np.sum(weight_patch))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return float(col_idx), float(row_idx)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    center_col = float(np.sum(weight_patch * cc) / total_weight)
    center_row = float(np.sum(weight_patch * rr) / total_weight)
    return center_col, center_row


def _auto_match_background_context(
    background_image: object,
    cfg: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return cached background peakness data for seed-local auto-matching."""

    config = cfg if isinstance(cfg, dict) else {}
    local_max_size = int(config.get("local_max_size_px", 5))
    local_max_size = max(3, local_max_size)
    if local_max_size % 2 == 0:
        local_max_size += 1
    smooth_sigma = max(0.0, float(config.get("smooth_sigma_px", 3.0)))
    climb_sigma = max(
        0.0,
        float(
            config.get(
                "climb_sigma_px",
                min(1.0, 0.5 * smooth_sigma) if smooth_sigma > 0.0 else 0.8,
            )
        ),
    )
    min_prominence_sigma = float(config.get("min_prominence_sigma", 2.0))
    fallback_percentile = float(config.get("fallback_percentile", 99.5))
    fallback_percentile = min(100.0, max(50.0, fallback_percentile))

    raw_arr = np.asarray(background_image)
    bg_ptr = 0
    try:
        bg_ptr = int(raw_arr.__array_interface__["data"][0])
    except Exception:
        bg_ptr = int(id(raw_arr))
    bg_key = (
        bg_ptr,
        tuple(int(v) for v in raw_arr.shape),
        tuple(int(v) for v in raw_arr.strides),
        str(raw_arr.dtype),
    )
    cache_key = (
        bg_key,
        int(local_max_size),
        round(float(smooth_sigma), 6),
        round(float(climb_sigma), 6),
        round(float(min_prominence_sigma), 6),
        round(float(fallback_percentile), 6),
    )
    cached_context = gui_controllers.get_geometry_auto_match_background_cache(
        geometry_preview_state,
        cache_key,
    )
    if cached_context is not None:
        _auto_match_console(config, f"context cache hit shape={bg_key[1]}")
        return dict(config), cached_context

    cache_data = build_background_peak_context(
        background_image,
        config,
        logger=lambda text: _auto_match_console(config, text),
    )
    _auto_match_console(
        config,
        "context build done "
        f"summits={len(cache_data.get('summit_records', []))} "
        f"sigma_est={float(cache_data.get('sigma_est', np.nan)):.4f}",
    )
    gui_controllers.replace_geometry_auto_match_background_cache(
        geometry_preview_state,
        cache_key,
        cache_data,
    )
    return dict(config), cache_data


def _auto_match_background_peaks(
    simulated_peaks: list[dict[str, object]],
    background_image: np.ndarray,
    cfg: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Match simulated seeds to nearby background peaks using local assignment."""

    config = dict(cfg) if isinstance(cfg, dict) else {}
    _auto_match_console(config, f"seed-local match start seeds={len(simulated_peaks)}")
    context_offset_col = 0.0
    context_offset_row = 0.0
    context_simulated: list[dict[str, object]] = []
    raw_background = np.asarray(background_image)
    context_margin_px = float(config.get("context_margin_px", np.inf))
    if not np.isfinite(context_margin_px) or context_margin_px <= 0.0:
        context_margin_px = np.inf
    if raw_background.ndim == 2 and simulated_peaks and np.isfinite(context_margin_px):
        smooth_sigma = max(0.0, float(config.get("smooth_sigma_px", 3.0)))
        climb_sigma = max(
            0.0,
            float(
                config.get(
                    "climb_sigma_px",
                    min(1.0, 0.5 * smooth_sigma) if smooth_sigma > 0.0 else 0.8,
                )
            ),
        )
        broad_sigma = smooth_sigma if smooth_sigma > 0.0 else max(2.0, climb_sigma + 1.5)
        roi_pad = max(8.0, 3.0 * broad_sigma + 4.0)
        seed_cols = np.asarray(
            [float(entry.get("sim_col", np.nan)) for entry in simulated_peaks],
            dtype=float,
        )
        seed_rows = np.asarray(
            [float(entry.get("sim_row", np.nan)) for entry in simulated_peaks],
            dtype=float,
        )
        finite_seed = np.isfinite(seed_cols) & np.isfinite(seed_rows)
        if np.any(finite_seed):
            seed_cols = seed_cols[finite_seed]
            seed_rows = seed_rows[finite_seed]
            col0 = max(
                0,
                int(np.floor(float(np.min(seed_cols) - context_margin_px - roi_pad))),
            )
            col1 = min(
                int(raw_background.shape[1]),
                int(np.ceil(float(np.max(seed_cols) + context_margin_px + roi_pad))) + 1,
            )
            row0 = max(
                0,
                int(np.floor(float(np.min(seed_rows) - context_margin_px - roi_pad))),
            )
            row1 = min(
                int(raw_background.shape[0]),
                int(np.ceil(float(np.max(seed_rows) + context_margin_px + roi_pad))) + 1,
            )
            if row1 > row0 and col1 > col0:
                context_offset_col = float(col0)
                context_offset_row = float(row0)
                raw_background = raw_background[row0:row1, col0:col1]
    if np.asarray(raw_background).ndim == 2:
        _auto_match_console(
            config,
            "roi "
            f"shape={tuple(int(v) for v in np.asarray(raw_background).shape)} "
            f"offset=({int(round(context_offset_col))},{int(round(context_offset_row))})",
        )
    for entry in simulated_peaks:
        local_entry = dict(entry)
        sim_col_global = float(entry.get("sim_col", np.nan))
        sim_row_global = float(entry.get("sim_row", np.nan))
        local_entry["sim_col_global"] = sim_col_global
        local_entry["sim_row_global"] = sim_row_global
        local_entry["sim_col_local"] = sim_col_global - context_offset_col
        local_entry["sim_row_local"] = sim_row_global - context_offset_row
        context_simulated.append(local_entry)

    config, bg_ctx = _auto_match_background_context(raw_background, config)
    if not bool(bg_ctx.get("img_valid", False)):
        return [], {"simulated_count": float(len(simulated_peaks))}
    matches, stats = match_simulated_peaks_to_peak_context(
        context_simulated,
        bg_ctx,
        config,
        context_offset_col=context_offset_col,
        context_offset_row=context_offset_row,
        logger=lambda text: _auto_match_console(config, text),
    )
    _auto_match_console(
        config,
        f"done matches={len(matches)} pre_clip={int(stats.get('matched_pre_clip_count', 0.0))} "
        f"conflicts={int(stats.get('conflicted_match_count', 0.0))}",
    )
    return matches, stats


def _auto_match_background_peaks_with_relaxation(
    simulated_peaks: list[dict[str, object]],
    background_image: np.ndarray,
    cfg: dict[str, object] | None = None,
    *,
    min_matches: int = 1,
) -> tuple[
    list[dict[str, object]],
    dict[str, float],
    dict[str, object],
    list[dict[str, float | str]],
]:
    """Retry auto-match with progressively looser thresholds when needed."""

    base_cfg = dict(cfg) if isinstance(cfg, dict) else {}
    target_min_matches = max(1, min(int(min_matches), int(len(simulated_peaks))))
    attempts: list[dict[str, float | str]] = []

    def _run_attempt(
        attempt_cfg: dict[str, object], label: str
    ) -> tuple[list[dict[str, object]], dict[str, float]]:
        _auto_match_console(
            attempt_cfg,
            "attempt "
            f"{label} start min_prom="
            f"{float(attempt_cfg.get('min_match_prominence_sigma', attempt_cfg.get('min_prominence_sigma', np.nan))):.2f} "
            f"max_candidates={int(attempt_cfg.get('max_candidate_peaks', 0))}",
        )
        matches_i, stats_i = _auto_match_background_peaks(
            simulated_peaks,
            background_image,
            attempt_cfg,
        )
        attempts.append(
            {
                "label": label,
                "matches": float(len(matches_i)),
                "search_radius_px": float(attempt_cfg.get("search_radius_px", np.nan)),
                "min_match_prominence_sigma": float(
                    attempt_cfg.get(
                        "min_match_prominence_sigma",
                        attempt_cfg.get("min_prominence_sigma", np.nan),
                    )
                ),
                "max_candidate_peaks": float(attempt_cfg.get("max_candidate_peaks", np.nan)),
                "ambiguity_margin_px": float(attempt_cfg.get("ambiguity_margin_px", np.nan)),
                "ambiguity_ratio_min": float(attempt_cfg.get("ambiguity_ratio_min", np.nan)),
                "qualified_summit_count": float(
                    stats_i.get("qualified_summit_count", np.nan)
                ),
                "claimed_summit_count": float(
                    stats_i.get("claimed_summit_count", np.nan)
                ),
                "matched_pre_clip_count": float(
                    stats_i.get("matched_pre_clip_count", np.nan)
                ),
                "clipped_count": float(stats_i.get("clipped_count", np.nan)),
            }
        )
        _auto_match_console(
            attempt_cfg,
            "attempt "
            f"{label} result matches={len(matches_i)} "
            f"qualified={int(stats_i.get('qualified_summit_count', 0.0))} "
            f"claimed={int(stats_i.get('claimed_summit_count', 0.0))} "
            f"pre_clip={int(stats_i.get('matched_pre_clip_count', 0.0))} "
            f"clipped={int(stats_i.get('clipped_count', 0.0))}",
        )
        return matches_i, stats_i

    best_matches, best_stats = _run_attempt(base_cfg, "base")
    best_cfg = dict(base_cfg)

    if len(best_matches) >= target_min_matches:
        _auto_match_console(
            base_cfg,
            f"target reached on base attempt ({len(best_matches)}/{target_min_matches})",
        )
        return best_matches, best_stats, best_cfg, attempts

    if not bool(base_cfg.get("relax_on_low_matches", True)):
        _auto_match_console(
            base_cfg,
            f"relaxation disabled at {len(best_matches)}/{target_min_matches} matches",
        )
        return best_matches, best_stats, best_cfg, attempts

    base_radius = max(1.0, float(base_cfg.get("search_radius_px", 24.0)))
    base_match_prom = float(
        base_cfg.get(
            "min_match_prominence_sigma",
            base_cfg.get("min_prominence_sigma", 2.0),
        )
    )
    base_max_candidates = max(50, int(base_cfg.get("max_candidate_peaks", 1200)))
    base_ambiguity_margin = max(0.0, float(base_cfg.get("ambiguity_margin_px", 2.0)))
    base_ambiguity_ratio = max(1.0, float(base_cfg.get("ambiguity_ratio_min", 1.15)))
    base_distance_sigma_clip = max(0.0, float(base_cfg.get("distance_sigma_clip", 3.5)))

    relax_steps = max(1, min(6, int(base_cfg.get("relax_steps", 3))))
    radius_growth = max(0.1, float(base_cfg.get("relax_radius_growth", 0.65)))
    prominence_step = max(0.0, float(base_cfg.get("relax_prominence_step", 0.4)))
    candidate_growth = max(0.0, float(base_cfg.get("relax_candidate_growth", 0.35)))
    relax_ownership_guards = bool(base_cfg.get("relax_ownership_guards", False))

    for step in range(1, relax_steps + 1):
        relax_cfg = dict(base_cfg)
        relax_cfg["search_radius_px"] = float(
            min(120.0, max(base_radius, base_radius * (1.0 + radius_growth * step)))
        )
        relax_cfg["min_match_prominence_sigma"] = float(
            max(0.0, base_match_prom - prominence_step * step)
        )
        relax_cfg["max_candidate_peaks"] = int(
            min(
                5000,
                max(
                    base_max_candidates,
                    round(base_max_candidates * (1.0 + candidate_growth * step)),
                ),
            )
        )
        if relax_ownership_guards:
            relax_cfg["ambiguity_margin_px"] = float(
                max(0.0, base_ambiguity_margin - 0.5 * step)
            )
            relax_cfg["ambiguity_ratio_min"] = float(
                max(1.0, base_ambiguity_ratio - 0.05 * step)
            )
        else:
            relax_cfg["ambiguity_margin_px"] = float(base_ambiguity_margin)
            relax_cfg["ambiguity_ratio_min"] = float(base_ambiguity_ratio)
        relax_cfg["distance_sigma_clip"] = float(
            max(base_distance_sigma_clip, 3.5 + 0.5 * step)
        )

        matches_i, stats_i = _run_attempt(relax_cfg, f"relax_{step}")
        best_is_improved = len(matches_i) > len(best_matches)
        if len(matches_i) == len(best_matches):
            best_conf = float(best_stats.get("median_match_confidence", -np.inf))
            cand_conf = float(stats_i.get("median_match_confidence", -np.inf))
            best_is_improved = cand_conf > best_conf
        if best_is_improved:
            best_matches = matches_i
            best_stats = stats_i
            best_cfg = relax_cfg

        if len(best_matches) >= target_min_matches:
            _auto_match_console(
                relax_cfg,
                f"target reached on relax_{step} ({len(best_matches)}/{target_min_matches})",
            )
            break

    _auto_match_console(
        best_cfg,
        "best attempt "
        f"matches={len(best_matches)}/{target_min_matches} "
        f"qualified={int(best_stats.get('qualified_summit_count', 0.0))} "
        f"claimed={int(best_stats.get('claimed_summit_count', 0.0))}",
    )
    return best_matches, best_stats, best_cfg, attempts


def _current_geometry_fit_var_names() -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    return gui_geometry_fit.current_geometry_fit_var_names(
        fit_zb=bool(fit_zb_var.get()),
        fit_zs=bool(fit_zs_var.get()),
        fit_theta=bool(fit_theta_var.get()),
        fit_psi_z=bool(fit_psi_z_var.get()),
        fit_chi=bool(fit_chi_var.get()),
        fit_cor=bool(fit_cor_var.get()),
        fit_gamma=bool(fit_gamma_var.get()),
        fit_Gamma=bool(fit_Gamma_var.get()),
        fit_dist=bool(fit_dist_var.get()),
        fit_a=bool(fit_a_var.get()),
        fit_c=bool(fit_c_var.get()),
        fit_center_x=bool(fit_center_x_var.get()),
        fit_center_y=bool(fit_center_y_var.get()),
        use_shared_theta_offset=_geometry_fit_uses_shared_theta_offset(),
    )


def _current_geometry_fit_params() -> dict[str, object]:
    """Assemble the current geometry-fit parameter dictionary."""

    use_theta_offset = _geometry_fit_uses_shared_theta_offset()
    theta_offset_current = (
        _current_geometry_theta_offset(strict=False) if use_theta_offset else 0.0
    )
    theta_current = (
        _background_theta_for_index(background_runtime_state.current_background_index, strict_count=False)
        if use_theta_offset
        else theta_initial_var.get()
    )
    return {
        "a": a_var.get(),
        "c": c_var.get(),
        "lambda": lambda_,
        "psi": psi,
        "psi_z": psi_z_var.get(),
        "zs": zs_var.get(),
        "zb": zb_var.get(),
        "chi": chi_var.get(),
        "n2": n2,
        "mosaic_params": build_mosaic_params(),
        "debye_x": debye_x_var.get(),
        "debye_y": debye_y_var.get(),
        "center": [center_x_var.get(), center_y_var.get()],
        "center_x": center_x_var.get(),
        "center_y": center_y_var.get(),
        "theta_initial": theta_current,
        "theta_offset": theta_offset_current,
        "uv1": np.array([1.0, 0.0, 0.0]),
        "uv2": np.array([0.0, 1.0, 0.0]),
        "corto_detector": corto_detector_var.get(),
        "gamma": gamma_var.get(),
        "Gamma": Gamma_var.get(),
        "cor_angle": cor_angle_var.get(),
        "optics_mode": _current_optics_mode_flag(),
    }


def _geometry_fit_constraint_source_name(name: str) -> str:
    """Map fitted variable names back to the UI constraint control names."""

    if name == "theta_offset":
        return "theta_initial"
    return str(name)


def _geometry_fit_constraint_parameter_name(name: str) -> str:
    """Map UI constraint rows to the active fitted parameter names."""

    if name == "theta_initial" and _geometry_fit_uses_shared_theta_offset():
        return "theta_offset"
    return str(name)


def _current_geometry_fit_constraint_state(
    names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    selected_names = (
        list(names)
        if names is not None
        else list(geometry_fit_constraints_view_state.controls)
    )
    state: dict[str, dict[str, float]] = {}
    for name in selected_names:
        control = geometry_fit_constraints_view_state.controls.get(
            _geometry_fit_constraint_source_name(name)
        )
        if not isinstance(control, dict):
            continue
        try:
            window = float(control["window_var"].get())
        except Exception:
            window = float("nan")
        try:
            pull = float(control["pull_var"].get())
        except Exception:
            pull = 0.0
        if not np.isfinite(window):
            continue
        window = max(0.0, float(window))
        if not np.isfinite(pull):
            pull = 0.0
        pull = min(max(float(pull), 0.0), 1.0)
        state[str(name)] = {
            "window": float(window),
            "pull": float(pull),
        }
    return state


def _current_geometry_fit_parameter_domains(
    names: Sequence[str] | None = None,
) -> dict[str, tuple[float, float]]:
    selected_names = (
        list(names) if names is not None else list(geometry_fit_parameter_specs)
    )
    domains: dict[str, tuple[float, float]] = {}
    fit_geometry_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(fit_geometry_cfg, dict):
        fit_geometry_cfg = {}
    bounds_cfg = fit_geometry_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}

    for name in selected_names:
        parameter_name = _geometry_fit_constraint_parameter_name(str(name))
        control_name = _geometry_fit_constraint_source_name(parameter_name)

        if parameter_name == "center_x" or parameter_name == "center_y":
            domains[str(name)] = (0.0, max(float(image_size) - 1.0, 0.0))
            continue

        if parameter_name == "theta_offset":
            entry = bounds_cfg.get("theta_offset")
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                try:
                    lo = float(entry[0])
                    hi = float(entry[1])
                except Exception:
                    lo = float("nan")
                    hi = float("nan")
                if np.isfinite(lo) and np.isfinite(hi):
                    if lo > hi:
                        lo, hi = hi, lo
                    domains[str(name)] = (float(lo), float(hi))
                    continue
            elif isinstance(entry, dict):
                try:
                    lo = float(entry.get("min"))
                    hi = float(entry.get("max"))
                except Exception:
                    lo = float("nan")
                    hi = float("nan")
                if np.isfinite(lo) and np.isfinite(hi):
                    if lo > hi:
                        lo, hi = hi, lo
                    domains[str(name)] = (float(lo), float(hi))
                    continue

        spec = geometry_fit_parameter_specs.get(control_name)
        if not isinstance(spec, dict):
            continue
        slider_widget = spec.get("value_slider")
        if slider_widget is None:
            continue
        try:
            lo = float(slider_widget.cget("from"))
            hi = float(slider_widget.cget("to"))
        except Exception:
            continue
        if parameter_name == "theta_offset":
            span = max(abs(lo), abs(hi), 1.0)
            domains[str(name)] = (-float(span), float(span))
            continue
        if lo > hi:
            lo, hi = hi, lo
        domains[str(name)] = (float(lo), float(hi))
    return domains


def _build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
):
    return gui_geometry_fit.build_geometry_fit_runtime_config(
        base_config,
        current_params,
        control_settings,
        parameter_domains,
    )


def _live_geometry_preview_enabled() -> bool:
    """Return whether live auto-match preview is enabled."""

    try:
        return bool(live_geometry_preview_var.get())
    except Exception:
        return False


def _draw_live_geometry_preview_overlay(
    matched_pairs: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw the current live auto-match preview without disturbing fit markers."""
    gui_overlays.draw_live_geometry_preview_overlay(
        ax,
        matched_pairs,
        geometry_preview_artists=geometry_runtime_state.preview_artists,
        clear_geometry_preview_artists=_clear_geometry_preview_artists,
        draw_idle=canvas.draw_idle,
        normalize_hkl_key=_normalize_hkl_key,
        live_preview_match_is_excluded=_live_preview_match_is_excluded,
        max_display_markers=max_display_markers,
    )


def _render_live_geometry_preview_state(*, update_status: bool = True) -> bool:
    """Redraw the live preview from cached state, applying exclusions."""

    preview_overlay_state = geometry_preview_state.overlay
    pairs = preview_overlay_state.pairs
    max_display_markers = int(preview_overlay_state.max_display_markers)
    _draw_live_geometry_preview_overlay(
        pairs,
        max_display_markers=max_display_markers,
    )

    if not update_status:
        return bool(pairs)

    simulated_count = int(preview_overlay_state.simulated_count)
    min_matches = int(preview_overlay_state.min_matches)
    best_radius = float(preview_overlay_state.best_radius)
    mean_dist = float(preview_overlay_state.mean_dist)
    p90_dist = float(preview_overlay_state.p90_dist)
    quality_fail = bool(preview_overlay_state.quality_fail)
    q_group_total = int(preview_overlay_state.q_group_total)
    q_group_excluded = int(preview_overlay_state.q_group_excluded)
    collapsed_deg = int(preview_overlay_state.collapsed_degenerate_peaks)
    active_pairs, excluded_count = _filter_live_preview_matches(pairs)
    shown_count = min(len(pairs), max(1, int(max_display_markers)))
    summary = (
        "Live auto-match preview: "
        f"{len(active_pairs)}/{simulated_count} active peaks "
        f"(need {min_matches}, local-peak match"
    )
    if np.isfinite(best_radius):
        summary += f", limit={best_radius:.1f}px"
    if np.isfinite(mean_dist):
        summary += f", mean={mean_dist:.1f}px"
    if np.isfinite(p90_dist):
        summary += f", p90={p90_dist:.1f}px"
    summary += ")."
    if excluded_count > 0:
        summary += f" Excluded={excluded_count}."
    if q_group_total > 0:
        summary += f" Qr/Qz groups on={max(0, q_group_total - q_group_excluded)}/{q_group_total}."
    if collapsed_deg > 0:
        summary += f" Degenerate collapsed={collapsed_deg}."
    if len(active_pairs) < min_matches:
        summary += " Geometry fit would stop on the minimum-match gate."
    elif quality_fail:
        summary += " Geometry fit would stop on the quality gate."
    else:
        summary += " Geometry fit gates pass."
    if shown_count < len(pairs):
        summary += f" Showing {shown_count}/{len(pairs)} overlays."
    progress_label_geometry.config(text=summary)
    return bool(active_pairs)


def _refresh_live_geometry_preview(*, update_status: bool = True) -> bool:
    """Recompute and redraw the live auto-match overlay from the current state."""

    if not _live_geometry_preview_enabled():
        _clear_geometry_preview_artists()
        return False

    if analysis_view_controls_view_state.show_caked_2d_var.get():
        _clear_geometry_preview_artists()
        if update_status:
            progress_label_geometry.config(
                text="Live auto-match preview unavailable in 2D caked view."
            )
        return False

    display_background = _get_current_background_display()
    if not background_runtime_state.visible or display_background is None:
        _clear_geometry_preview_artists()
        if update_status:
            progress_label_geometry.config(
                text="Live auto-match preview unavailable: background image is hidden."
            )
        return False

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(geometry_refine_cfg, dict):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, dict):
        auto_match_cfg = {}
    preview_auto_match_cfg = dict(auto_match_cfg)
    preview_auto_match_cfg["relax_on_low_matches"] = False
    preview_auto_match_cfg.setdefault(
        "context_margin_px",
        max(192.0, 8.0 * float(auto_match_cfg.get("search_radius_px", 24.0))),
    )

    var_names = _current_geometry_fit_var_names()
    default_min_matches = max(6, len(var_names) + 2)
    min_matches = int(auto_match_cfg.get("min_matches", default_min_matches))
    min_matches = max(1, min_matches)

    simulated_peaks = _build_live_preview_simulated_peaks_from_cache()

    if not simulated_peaks and simulation_runtime_state.stored_max_positions_local:
        try:
            simulated_peaks = _simulate_preview_style_peaks_for_fit(
                np.asarray(miller, dtype=np.float64),
                np.asarray(intensities, dtype=np.float64),
                image_size,
                _current_geometry_fit_params(),
            )
        except Exception as exc:
            _clear_geometry_preview_artists()
            if update_status:
                progress_label_geometry.config(
                    text=f"Live auto-match preview unavailable: failed to simulate peaks ({exc})."
                )
            return False

    if not simulated_peaks:
        _clear_geometry_preview_artists()
        if update_status:
            progress_label_geometry.config(
                text="Live auto-match preview unavailable: no simulated peaks are available."
            )
            return False

    simulated_peaks, excluded_q_peaks, q_group_total = _filter_geometry_fit_simulated_peaks(
        simulated_peaks
    )
    if not simulated_peaks:
        _clear_geometry_preview_artists()
        if update_status:
            progress_label_geometry.config(
                text="Live auto-match preview unavailable: no Qr/Qz groups are selected."
            )
        gui_controllers.replace_geometry_preview_overlay_state(
            geometry_preview_state,
            {
                "signature": _live_geometry_preview_signature(),
                "pairs": [],
                "simulated_count": 0,
                "min_matches": int(min_matches),
                "best_radius": float("nan"),
                "mean_dist": float("nan"),
                "p90_dist": float("nan"),
                "quality_fail": False,
                "max_display_markers": int(preview_auto_match_cfg.get("max_display_markers", 120)),
                "auto_match_attempts": [],
                "q_group_total": int(q_group_total),
                "q_group_excluded": int(_geometry_q_group_excluded_count()),
                "excluded_q_peaks": int(excluded_q_peaks),
                "collapsed_degenerate_peaks": 0,
            },
        )
        return False
    simulated_peaks, collapsed_deg_preview = _collapse_geometry_fit_simulated_peaks(
        simulated_peaks,
        merge_radius_px=float(
            preview_auto_match_cfg.get(
                "degenerate_merge_radius_px",
                min(
                    6.0,
                    0.33 * float(preview_auto_match_cfg.get("search_radius_px", 24.0)),
                ),
            )
        ),
    )
    if not simulated_peaks:
        _clear_geometry_preview_artists()
        if update_status:
            progress_label_geometry.config(
                text="Live auto-match preview unavailable: no geometry-fit seeds remain after Qr/Qz collapse."
            )
        return False

    matched_pairs, match_stats, effective_auto_match_cfg, auto_match_attempts = (
        _auto_match_background_peaks_with_relaxation(
            simulated_peaks,
            display_background,
            preview_auto_match_cfg,
            min_matches=min_matches,
        )
    )

    max_display_markers = int(preview_auto_match_cfg.get("max_display_markers", 120))

    simulated_count = int(match_stats.get("simulated_count", len(simulated_peaks)))
    best_radius = float(match_stats.get("search_radius_px", np.nan))
    max_auto_p90 = float(preview_auto_match_cfg.get("max_p90_distance_px", 35.0))
    max_auto_mean = float(preview_auto_match_cfg.get("max_mean_distance_px", 22.0))
    p90_dist = float(match_stats.get("p90_match_distance_px", np.nan))
    mean_dist = float(match_stats.get("mean_match_distance_px", np.nan))
    quality_fail = (
        (np.isfinite(max_auto_p90) and np.isfinite(p90_dist) and p90_dist > max_auto_p90)
        or (np.isfinite(max_auto_mean) and np.isfinite(mean_dist) and mean_dist > max_auto_mean)
    )
    gui_controllers.replace_geometry_preview_overlay_state(
        geometry_preview_state,
        {
            "signature": _live_geometry_preview_signature(),
            "pairs": [dict(entry) for entry in matched_pairs],
            "simulated_count": int(simulated_count),
            "min_matches": int(min_matches),
            "best_radius": float(best_radius),
            "mean_dist": float(mean_dist),
            "p90_dist": float(p90_dist),
            "quality_fail": bool(quality_fail),
            "max_display_markers": int(max_display_markers),
            "auto_match_attempts": list(auto_match_attempts),
            "q_group_total": int(q_group_total),
            "q_group_excluded": int(_geometry_q_group_excluded_count()),
            "excluded_q_peaks": int(excluded_q_peaks),
            "collapsed_degenerate_peaks": int(collapsed_deg_preview),
        },
    )

    return _render_live_geometry_preview_state(update_status=update_status)


def _on_live_geometry_preview_toggle():
    """Enable or disable the live geometry auto-match preview."""

    if not _live_geometry_preview_enabled():
        _set_geometry_preview_exclude_mode(False)
        _clear_geometry_preview_artists()
        progress_label_geometry.config(text="Live auto-match preview disabled.")
        return

    _open_geometry_q_group_window()

    if simulation_runtime_state.update_running or simulation_runtime_state.stored_max_positions_local is None:
        schedule_update()
        return

    if not _refresh_live_geometry_preview(update_status=True):
        schedule_update()


def _legacy_auto_match_on_fit_geometry_click():
    _clear_geometry_pick_artists()
    _clear_geometry_preview_artists()

    def _cmd_line(text: str) -> None:
        try:
            print(f"[geometry-fit] {text}", flush=True)
        except Exception:
            pass

    params = _current_geometry_fit_params()
    try:
        mosaic_params = build_mosaic_params()
    except Exception:
        mosaic_params = {}
    var_names = _current_geometry_fit_var_names()
    if not var_names:
        _cmd_line("aborted: no geometry parameters selected")
        progress_label_geometry.config(text="No parameters selected!")
        return
    _cmd_line(f"start: vars={','.join(var_names)}")
    preserve_live_theta = (
        "theta_initial" not in var_names and "theta_offset" not in var_names
    )

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(geometry_refine_cfg, dict):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, dict):
        auto_match_cfg = {}
    orientation_cfg = geometry_refine_cfg.get("orientation", {}) or {}
    if not isinstance(orientation_cfg, dict):
        orientation_cfg = {}

    if not _apply_geometry_fit_background_selection(
        trigger_update=False,
        sync_live_theta=not preserve_live_theta,
    ):
        return

    try:
        selected_background_indices = _current_geometry_fit_background_indices(strict=True)
    except Exception as exc:
        _cmd_line(f"aborted: invalid fit background selection ({exc})")
        progress_label_geometry.config(
            text=f"Geometry fit unavailable: invalid fit background selection ({exc})."
        )
        return
    if int(background_runtime_state.current_background_index) not in {
        int(idx) for idx in selected_background_indices
    }:
        _cmd_line("aborted: active background missing from fit selection")
        progress_label_geometry.config(
            text=(
                "Geometry fit unavailable: the active background must be part of the fit "
                "selection so the overlay can be drawn on the current image."
            )
        )
        return

    joint_background_mode = False
    background_theta_values: list[float] = []
    shared_theta_offset_seed = 0.0
    if _geometry_fit_uses_shared_theta_offset(selected_background_indices):
        if not _apply_background_theta_metadata(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        ):
            _cmd_line("aborted: invalid background theta_i/shared offset settings")
            progress_label_geometry.config(
                text="Geometry fit unavailable: invalid background theta_i/shared offset settings."
            )
            return
        try:
            background_theta_values = _current_background_theta_values(strict_count=True)
            shared_theta_offset_seed = _current_geometry_theta_offset(strict=True)
        except Exception as exc:
            _cmd_line(f"aborted: failed to parse background theta settings ({exc})")
            progress_label_geometry.config(
                text=f"Geometry fit unavailable: failed to parse background theta settings ({exc})."
            )
            return
        joint_background_mode = len(selected_background_indices) > 1
        params["theta_offset"] = float(shared_theta_offset_seed)
        if background_theta_values:
            try:
                params["theta_initial"] = float(
                    background_theta_values[int(background_runtime_state.current_background_index)] + shared_theta_offset_seed
                )
            except Exception:
                pass
    else:
        params["theta_offset"] = 0.0

    native_background = _get_current_background_native()
    backend_background = _get_current_background_backend()
    display_background = background_runtime_state.current_background_display
    if display_background is None and native_background is not None:
        display_background = np.rot90(native_background, DISPLAY_ROTATE_K)
    if native_background is None or display_background is None:
        _cmd_line("aborted: no background image loaded")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no background image is loaded."
        )
        return
    if backend_background is None:
        backend_background = native_background

    miller_array = np.asarray(miller, dtype=np.float64)
    intensity_array = np.asarray(intensities, dtype=np.float64)
    if miller_array.ndim != 2 or miller_array.shape[1] != 3 or miller_array.size == 0:
        _cmd_line("aborted: no simulated reflections available")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no simulated reflections are available."
        )
        return
    if intensity_array.shape[0] != miller_array.shape[0]:
        _cmd_line("aborted: intensity array does not match HKLs")
        progress_label_geometry.config(
            text="Geometry fit unavailable: intensity array does not match HKLs."
        )
        return

    try:
        simulated_peaks = _simulate_preview_style_peaks_for_fit(
            miller_array,
            intensity_array,
            image_size,
            params,
        )
    except Exception as exc:
        _cmd_line(f"aborted: failed to simulate peak centers ({exc})")
        progress_label_geometry.config(
            text=f"Geometry fit unavailable: failed to simulate peak centers ({exc})."
        )
        return

    if not simulated_peaks:
        _cmd_line("aborted: no simulated Bragg peak centers were found")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no simulated Bragg peak centers were found."
        )
        return

    simulated_peaks, excluded_q_peaks, q_group_total = _filter_geometry_fit_simulated_peaks(
        simulated_peaks
    )
    if not simulated_peaks:
        _cmd_line("aborted: no Qr/Qz groups selected for fitting")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no Qr/Qz groups are selected for fitting."
        )
        return
    default_min_matches = max(6, len(var_names) + 2)
    min_matches = int(auto_match_cfg.get("min_matches", default_min_matches))
    min_matches = max(1, min_matches)
    fit_auto_match_cfg = dict(auto_match_cfg)
    fit_auto_match_cfg["console_progress"] = True
    fit_auto_match_cfg.setdefault("console_progress_every", 10)
    fit_auto_match_cfg.setdefault(
        "context_margin_px",
        max(256.0, 10.0 * float(auto_match_cfg.get("search_radius_px", 24.0))),
    )
    raw_simulated_peak_count = len(simulated_peaks)
    simulated_peaks, collapsed_deg_fit = _collapse_geometry_fit_simulated_peaks(
        simulated_peaks,
        merge_radius_px=float(
            fit_auto_match_cfg.get(
                "degenerate_merge_radius_px",
                min(
                    6.0,
                    0.33 * float(fit_auto_match_cfg.get("search_radius_px", 24.0)),
                ),
            )
        ),
    )
    if not simulated_peaks:
        _cmd_line("aborted: no geometry-fit seeds remain after Qr/Qz collapse")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no geometry-fit seeds remain after collapsing degenerate Qr/Qz peaks."
        )
        return
    _cmd_line(
        "prepared: "
        f"simulated_peaks={len(simulated_peaks)} "
        f"(collapsed={collapsed_deg_fit} from {raw_simulated_peak_count}) "
        f"q_groups_on={max(0, q_group_total - _geometry_q_group_excluded_count())}/{q_group_total}"
    )

    _cmd_line(f"auto-match: local-peak match start (min_matches={min_matches})")

    matched_pairs, match_stats, effective_auto_match_cfg, auto_match_attempts = (
        _auto_match_background_peaks_with_relaxation(
            simulated_peaks,
            display_background,
            fit_auto_match_cfg,
            min_matches=min_matches,
        )
    )
    matched_pairs, match_stats, excluded_preview_count = _apply_live_preview_match_exclusions(
        matched_pairs,
        match_stats,
    )
    _cmd_line(
        "auto-match: "
        f"matched={len(matched_pairs)}/"
        f"{int(match_stats.get('simulated_count', len(simulated_peaks)))} "
        f"qualified_summits={int(match_stats.get('qualified_summit_count', 0))} "
        f"claimed_summits={int(match_stats.get('claimed_summit_count', 0))} "
        f"mean={float(match_stats.get('mean_match_distance_px', np.nan)):.2f}px "
        f"p90={float(match_stats.get('p90_match_distance_px', np.nan)):.2f}px "
        f"excluded_preview={excluded_preview_count}"
    )

    if len(matched_pairs) < min_matches:
        _cmd_line(
            f"cancelled: minimum-match gate failed ({len(matched_pairs)} < {min_matches})"
        )
        simulated_count = int(match_stats.get("simulated_count", len(simulated_peaks)))
        progress_label_geometry.config(
            text=(
                "Geometry fit cancelled: local-peak auto-match found "
                f"{len(matched_pairs)}/{simulated_count} active peaks "
                f"(need at least {min_matches})."
                + (
                    f" Qr/Qz groups off={_geometry_q_group_excluded_count()}/{q_group_total}."
                    if q_group_total > 0
                    else ""
                )
                + (
                    f" Excluded in preview={excluded_preview_count}."
                    if excluded_preview_count > 0
                    else ""
                )
                + " Try Auto-Match Scale, then retry."
            )
        )
        return

    max_auto_p90 = float(auto_match_cfg.get("max_p90_distance_px", 35.0))
    max_auto_mean = float(auto_match_cfg.get("max_mean_distance_px", 22.0))
    p90_dist = float(match_stats.get("p90_match_distance_px", np.nan))
    mean_dist = float(match_stats.get("mean_match_distance_px", np.nan))
    quality_fail = (
        (np.isfinite(max_auto_p90) and np.isfinite(p90_dist) and p90_dist > max_auto_p90)
        or (np.isfinite(max_auto_mean) and np.isfinite(mean_dist) and mean_dist > max_auto_mean)
    )
    if quality_fail:
        _cmd_line(
            "cancelled: quality gate failed "
            f"(mean={mean_dist:.2f}px, p90={p90_dist:.2f}px, "
            f"limits={max_auto_mean:.2f}/{max_auto_p90:.2f}px)"
        )
        progress_label_geometry.config(
            text=(
                "Geometry fit cancelled: auto-match quality gate failed "
                f"(mean={mean_dist:.1f}px, p90={p90_dist:.1f}px; "
                f"limits mean<={max_auto_mean:.1f}px, p90<={max_auto_p90:.1f}px). "
                + (
                    f"Qr/Qz groups off={_geometry_q_group_excluded_count()}/{q_group_total}. "
                    if q_group_total > 0
                    else ""
                )
                + (
                    f"Excluded in preview={excluded_preview_count}. "
                    if excluded_preview_count > 0
                    else ""
                )
                + "Tighten auto-match or use manual peak picks."
            )
        )
        return

    def _attach_overlay_match_indices(
        pairs: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        normalized_pairs: list[dict[str, object]] = []
        for pair_idx, raw_entry in enumerate(pairs or []):
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            try:
                overlay_idx = int(entry.get("overlay_match_index", pair_idx))
            except Exception:
                overlay_idx = int(pair_idx)
            if overlay_idx < 0:
                overlay_idx = int(pair_idx)
            entry["overlay_match_index"] = int(overlay_idx)
            normalized_pairs.append(entry)
        return normalized_pairs

    matched_pairs = _attach_overlay_match_indices(matched_pairs)

    initial_auto_pairs_display: list[dict[str, object]] = []
    for entry in matched_pairs:
        hkl_key = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl_key is None:
            continue
        try:
            sim_col = float(entry["sim_x"])
            sim_row = float(entry["sim_y"])
            bg_col = float(entry["x"])
            bg_row = float(entry["y"])
        except Exception:
            continue
        if not all(np.isfinite(v) for v in (sim_col, sim_row, bg_col, bg_row)):
            continue
        sim_display = _rotate_point_for_display(
            sim_col,
            sim_row,
            (image_size, image_size),
            SIM_DISPLAY_ROTATE_K,
        )
        initial_auto_pairs_display.append(
            {
                "overlay_match_index": int(entry.get("overlay_match_index", len(initial_auto_pairs_display))),
                "hkl": hkl_key,
                "sim_display": (float(sim_display[0]), float(sim_display[1])),
                "bg_display": (bg_col, bg_row),
            }
        )
    preview_marker_limit = int(auto_match_cfg.get("max_display_markers", 120))
    preview_marker_limit = max(1, preview_marker_limit)
    _draw_initial_geometry_pairs_overlay(
        initial_auto_pairs_display,
        max_display_markers=preview_marker_limit,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = get_dir("downloads") / f"geometry_fit_log_{stamp}.txt"
    log_file = log_path.open("w", encoding="utf-8")
    _cmd_line(f"log: {log_path}")

    def _log_line(text: str = ""):
        try:
            log_file.write(text + "\n")
            log_file.flush()
        except Exception:
            pass

    def _log_section(title: str, lines: list[str]):
        _log_line(title)
        for line in lines:
            _log_line(f"  {line}")
        _log_line()

    def _log_format_hkl(entry: object) -> str:
        if not isinstance(entry, dict):
            return "n/a"
        hkl_key = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
        return str(hkl_key) if hkl_key is not None else str(entry.get("label", "n/a"))

    def _log_format_point(entry: object, x_key: str = "x", y_key: str = "y") -> str:
        if not isinstance(entry, dict):
            return "n/a"
        try:
            x_val = float(entry.get(x_key, np.nan))
            y_val = float(entry.get(y_key, np.nan))
        except Exception:
            return "n/a"
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            return "n/a"
        return f"({x_val:.3f},{y_val:.3f})"

    def _log_radius_to_center(
        entry: object,
        *,
        center_pair: tuple[float, float] | None = None,
    ) -> str:
        if center_pair is None or not isinstance(entry, dict):
            return "n/a"
        try:
            x_val = float(entry.get("x", np.nan))
            y_val = float(entry.get("y", np.nan))
            center_row = float(center_pair[0])
            center_col = float(center_pair[1])
        except Exception:
            return "n/a"
        if not (
            np.isfinite(x_val)
            and np.isfinite(y_val)
            and np.isfinite(center_row)
            and np.isfinite(center_col)
        ):
            return "n/a"
        return f"{math.hypot(x_val - center_col, y_val - center_row):.3f}px"

    def _log_source_descriptor(entry: object) -> str:
        if not isinstance(entry, dict):
            return "src=n/a"
        parts: list[str] = []
        if entry.get("source_label") is not None:
            parts.append(f"label={entry.get('source_label')}")
        if entry.get("source_table_index") is not None:
            parts.append(f"table={entry.get('source_table_index')}")
        if entry.get("source_row_index") is not None:
            parts.append(f"row={entry.get('source_row_index')}")
        if entry.get("source_peak_index") is not None:
            parts.append(f"peak={entry.get('source_peak_index')}")
        return "src=" + (",".join(parts) if parts else "n/a")

    def _log_fixed_peak_frames(
        display_entries: Sequence[dict[str, object]] | None,
        native_entries: Sequence[dict[str, object]] | None,
        fit_entries: Sequence[dict[str, object]] | None,
        *,
        center_pair: tuple[float, float] | None = None,
    ) -> list[str]:
        lines: list[str] = []
        count = max(
            len(display_entries or []),
            len(native_entries or []),
            len(fit_entries or []),
        )
        for idx in range(count):
            display_entry = (
                dict(display_entries[idx])
                if display_entries is not None and idx < len(display_entries)
                else None
            )
            native_entry = (
                dict(native_entries[idx])
                if native_entries is not None and idx < len(native_entries)
                else None
            )
            fit_entry = (
                dict(fit_entries[idx])
                if fit_entries is not None and idx < len(fit_entries)
                else None
            )
            base_entry = display_entry or native_entry or fit_entry or {}
            try:
                sim_x = float(base_entry.get("sim_x", np.nan))
                sim_y = float(base_entry.get("sim_y", np.nan))
                sim_text = (
                    f"({sim_x:.3f},{sim_y:.3f})"
                    if np.isfinite(sim_x) and np.isfinite(sim_y)
                    else "n/a"
                )
            except Exception:
                sim_text = "n/a"
            overlay_match_index = base_entry.get("overlay_match_index")
            lines.append(
                f"idx={idx} overlay={overlay_match_index if overlay_match_index is not None else 'n/a'} "
                f"HKL={_log_format_hkl(base_entry)} {_log_source_descriptor(base_entry)} "
                f"sim_display={sim_text} "
                f"meas_display={_log_format_point(display_entry)} "
                f"meas_native={_log_format_point(native_entry)} "
                f"meas_fit={_log_format_point(fit_entry)} "
                f"fit_r={_log_radius_to_center(fit_entry, center_pair=center_pair)}"
            )
        return lines or ["<none>"]

    def _log_point_match_diagnostics(entries: object) -> list[str]:
        if not isinstance(entries, list) or not entries:
            return ["<none>"]
        lines: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            hkl_key = entry.get("hkl", "n/a")
            src_text = (
                f"table={entry.get('source_table_index', 'n/a')},"
                f"row={entry.get('source_row_index', 'n/a')}"
            )
            try:
                meas_x = float(entry.get("measured_x", np.nan))
                meas_y = float(entry.get("measured_y", np.nan))
                sim_x = float(entry.get("simulated_x", np.nan))
                sim_y = float(entry.get("simulated_y", np.nan))
                dist_px = float(entry.get("distance_px", np.nan))
                meas_r = float(entry.get("measured_radius_px", np.nan))
                sim_r = float(entry.get("simulated_radius_px", np.nan))
            except Exception:
                meas_x = meas_y = sim_x = sim_y = dist_px = meas_r = sim_r = float("nan")
            lines.append(
                f"idx={entry.get('match_input_index', 'n/a')} "
                f"dataset={entry.get('dataset_index', 'n/a')}:{entry.get('dataset_label', 'n/a')} "
                f"overlay={entry.get('overlay_match_index', 'n/a')} HKL={hkl_key} "
                f"label={entry.get('label', 'n/a')} src={src_text} "
                f"resolution={entry.get('resolution_kind', 'n/a')}/"
                f"{entry.get('resolution_reason', 'n/a')} "
                f"status={entry.get('match_status', 'n/a')} "
                f"kind={entry.get('match_kind', 'n/a')} "
                f"meas=({meas_x:.3f},{meas_y:.3f}) "
                f"sim=({sim_x:.3f},{sim_y:.3f}) "
                f"d={dist_px:.3f}px meas_r={meas_r:.3f}px sim_r={sim_r:.3f}px"
            )
        return lines or ["<none>"]

    progress_label_geometry.config(text="Running geometry fit (auto-matched peaks)…")
    root.update_idletasks()
    _cmd_line(
        f"fit: running optimizer with {len(matched_pairs)} matched peaks and orientation auto-selection"
    )

    try:
        measured_from_display = []
        for entry in matched_pairs:
            try:
                x_val = float(entry["x"])
                y_val = float(entry["y"])
            except Exception:
                continue
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue
            normalized_hkl = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
            measured_entry = dict(entry)
            measured_entry["label"] = str(entry["label"])
            measured_entry["x"] = x_val
            measured_entry["y"] = y_val
            measured_entry["overlay_match_index"] = int(
                entry.get("overlay_match_index", len(measured_from_display))
            )
            if normalized_hkl is not None:
                measured_entry["hkl"] = normalized_hkl
            measured_from_display.append(measured_entry)

        _log_line(f"Geometry fit started: {stamp}")
        _log_line()
        _log_section(
            "Auto-match requested configuration:",
            [
                f"{key}={value}" for key, value in sorted(auto_match_cfg.items())
            ] or ["<defaults>"],
        )
        _log_section(
            "Auto-match effective configuration:",
            [
                f"{key}={value}"
                for key, value in sorted(effective_auto_match_cfg.items())
            ] or ["<defaults>"],
        )
        _log_section(
            "Auto-match attempts:",
            [
                (
                    f"{entry.get('label', 'attempt')}: "
                    f"matches={int(float(entry.get('matches', 0.0)))} "
                    f"radius={float(entry.get('search_radius_px', np.nan)):.2f}px "
                    f"min_prom={float(entry.get('min_match_prominence_sigma', np.nan)):.2f}σ "
                    f"cands={int(float(entry.get('max_candidate_peaks', 0.0)))} "
                    f"amb_margin={float(entry.get('ambiguity_margin_px', np.nan)):.2f}px "
                    f"amb_ratio={float(entry.get('ambiguity_ratio_min', np.nan)):.2f} "
                    f"qualified={int(float(entry.get('qualified_summit_count', 0.0)))} "
                    f"claimed={int(float(entry.get('claimed_summit_count', 0.0)))} "
                    f"pre_clip={int(float(entry.get('matched_pre_clip_count', 0.0)))} "
                    f"clipped={int(float(entry.get('clipped_count', 0.0)))}"
                )
                for entry in auto_match_attempts
            ]
            or ["<none>"],
        )
        _log_section(
            "Auto-match summary:",
            [
                f"simulated_peaks={int(match_stats.get('simulated_count', len(simulated_peaks)))}",
                f"candidate_summits={int(match_stats.get('candidate_count', 0))}",
                f"qualified_summits={int(match_stats.get('qualified_summit_count', 0))}",
                f"within_radius_summits={int(match_stats.get('within_radius_count', 0))}",
                f"unambiguous_summits={int(match_stats.get('unambiguous_count', 0))}",
                f"ownership_filtered_summits={int(match_stats.get('ownership_filtered_count', 0))}",
                f"claimed_summits={int(match_stats.get('claimed_summit_count', 0))}",
                f"conflicted_matches={int(match_stats.get('conflicted_match_count', 0))}",
                f"matched_pre_clip={int(match_stats.get('matched_pre_clip_count', 0))}",
                f"clipped_matches={int(match_stats.get('clipped_count', 0))}",
                f"matched_peaks={len(matched_pairs)}",
                f"excluded_live_preview_peaks={int(match_stats.get('excluded_count', 0))}",
                f"excluded_q_group_peaks={int(excluded_q_peaks)}",
                f"collapsed_degenerate_peaks={int(collapsed_deg_fit)}",
                f"excluded_q_groups={_geometry_q_group_excluded_count()}",
                f"available_q_groups={int(q_group_total)}",
                f"prominence_sigma_est={float(match_stats.get('sigma_est', np.nan)):.6f}",
                f"mean_walk_steps={float(match_stats.get('mean_walk_steps', np.nan)):.3f}",
                f"mean_net_ascent_sigma={float(match_stats.get('mean_net_ascent_sigma', np.nan)):.3f}",
                f"distance_clip_limit_px={float(match_stats.get('distance_clip_limit_px', np.nan)):.3f}",
                f"mean_match_distance_px={float(match_stats.get('mean_match_distance_px', np.nan)):.3f}",
                f"p90_match_distance_px={float(match_stats.get('p90_match_distance_px', np.nan)):.3f}",
                f"median_match_confidence={float(match_stats.get('median_match_confidence', np.nan)):.3f}",
            ],
        )
        _log_section(
            "Auto-matched pairs (display frame):",
            [
                (
                    f"HKL=({entry['hkl'][0]},{entry['hkl'][1]},{entry['hkl'][2]}) "
                    f"sim=({entry['sim_x']:.3f},{entry['sim_y']:.3f}) "
                    f"meas=({entry['x']:.3f},{entry['y']:.3f}) "
                    f"Ibg={float(entry.get('background_intensity', np.nan)):.3f} "
                    f"d={entry['distance_px']:.3f}px "
                    f"prom={entry['prominence_sigma']:.2f}σ "
                    f"conf={entry['confidence']:.3f}"
                )
                for entry in matched_pairs
            ],
        )

        measured_native = _unrotate_display_peaks(
            measured_from_display,
            display_background.shape,
            k=DISPLAY_ROTATE_K,
        )

        sim_orientation_points: list[tuple[float, float]] = []
        meas_orientation_points: list[tuple[float, float]] = []
        for pair_entry, measured_entry in zip(matched_pairs, measured_native):
            if not isinstance(measured_entry, dict):
                continue
            try:
                sx = float(pair_entry["sim_x"])
                sy = float(pair_entry["sim_y"])
                mx = float(measured_entry["x"])
                my = float(measured_entry["y"])
            except Exception:
                continue
            if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(mx) and np.isfinite(my)):
                continue
            sim_orientation_points.append((sx, sy))
            meas_orientation_points.append((mx, my))

        orientation_choice, orientation_diag = _select_fit_orientation(
            sim_orientation_points,
            meas_orientation_points,
            tuple(int(v) for v in native_background.shape[:2]),
            cfg=orientation_cfg,
        )
        _cmd_line(
            "orientation: "
            f"chosen={orientation_choice.get('label', 'identity')} "
            f"identity_rms={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}px "
            f"best_rms={float(orientation_diag.get('best_rms_px', np.nan)):.4f}px "
            f"reason={orientation_diag.get('reason', 'n/a')}"
        )
        _log_section(
            "Orientation diagnostics:",
            [
                f"pairs={orientation_diag.get('pairs', 0)}",
                f"enabled={orientation_diag.get('enabled', True)}",
                f"display_unrotate_k={DISPLAY_ROTATE_K}",
                f"sim_display_k={SIM_DISPLAY_ROTATE_K}",
                f"identity_rms_px={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}",
                f"best_label={orientation_diag.get('best_label', 'identity')}",
                f"best_rms_px={float(orientation_diag.get('best_rms_px', np.nan)):.4f}",
                f"improvement_px={float(orientation_diag.get('improvement_px', np.nan)):.4f}",
                f"chosen={orientation_choice.get('label', 'identity')}",
                f"reason={orientation_diag.get('reason', 'n/a')}",
            ],
        )

        measured_for_fit = _apply_orientation_to_entries(
            measured_native,
            native_background.shape,
            indexing_mode=orientation_choice["indexing_mode"],
            k=orientation_choice["k"],
            flip_x=orientation_choice["flip_x"],
            flip_y=orientation_choice["flip_y"],
            flip_order=orientation_choice["flip_order"],
        )
        experimental_image_for_fit = _orient_image_for_fit(
            backend_background,
            indexing_mode=orientation_choice["indexing_mode"],
            k=orientation_choice["k"],
            flip_x=orientation_choice["flip_x"],
            flip_y=orientation_choice["flip_y"],
            flip_order=orientation_choice["flip_order"],
        )
        _log_section(
            "Fixed peak frame transforms:",
            _log_fixed_peak_frames(
                measured_from_display,
                measured_native,
                measured_for_fit,
                center_pair=(
                    float(params.get("center_x", params["center"][0])),
                    float(params.get("center_y", params["center"][1])),
                ),
            ),
        )

        def _pairs_to_measured_display_entries(
            pairs: Sequence[dict[str, object]] | None,
        ) -> list[dict[str, object]]:
            measured_entries: list[dict[str, object]] = []
            for entry in pairs or []:
                try:
                    x_val = float(entry["x"])
                    y_val = float(entry["y"])
                except Exception:
                    continue
                if not (np.isfinite(x_val) and np.isfinite(y_val)):
                    continue
                normalized_hkl = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
                measured_entry = dict(entry)
                measured_entry["label"] = str(entry["label"])
                measured_entry["x"] = x_val
                measured_entry["y"] = y_val
                measured_entry["overlay_match_index"] = int(
                    entry.get("overlay_match_index", len(measured_entries))
                )
                if normalized_hkl is not None:
                    measured_entry["hkl"] = normalized_hkl
                measured_entries.append(measured_entry)
            return measured_entries

        def _build_joint_background_dataset_specs(
            base_fit_params: dict[str, object],
        ) -> tuple[list[dict[str, object]], list[str]]:
            dataset_specs_local: list[dict[str, object]] = []
            dataset_summary_lines: list[str] = []
            if not joint_background_mode:
                return dataset_specs_local, dataset_summary_lines

            current_theta_base = float(background_theta_values[int(background_runtime_state.current_background_index)])
            dataset_specs_local.append(
                {
                    "dataset_index": int(background_runtime_state.current_background_index),
                    "label": Path(str(background_runtime_state.osc_files[background_runtime_state.current_background_index])).name,
                    "theta_initial": float(current_theta_base),
                    "measured_peaks": list(measured_for_fit),
                    "experimental_image": experimental_image_for_fit,
                }
            )
            dataset_summary_lines.append(
                "current[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
                "matches={matches} excluded={excluded}"
                .format(
                    idx=int(background_runtime_state.current_background_index),
                    name=Path(str(background_runtime_state.osc_files[background_runtime_state.current_background_index])).name,
                    theta_base=float(current_theta_base),
                    theta_eff=float(current_theta_base + float(base_fit_params.get("theta_offset", 0.0))),
                    matches=len(matched_pairs),
                    excluded=int(match_stats.get("excluded_count", excluded_preview_count)),
                )
            )

            for bg_idx in selected_background_indices:
                if int(bg_idx) == int(background_runtime_state.current_background_index):
                    continue
                theta_base = float(background_theta_values[int(bg_idx)])

                native_bg_i, display_bg_i = _load_background_image_by_index(int(bg_idx))
                backend_bg_i = _apply_background_backend_orientation(native_bg_i)
                if backend_bg_i is None:
                    backend_bg_i = native_bg_i

                params_i = dict(base_fit_params)
                params_i["theta_initial"] = float(theta_base + float(base_fit_params.get("theta_offset", 0.0)))

                simulated_i = _simulate_preview_style_peaks_for_fit(
                    miller_array,
                    intensity_array,
                    image_size,
                    params_i,
                )
                if not simulated_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) has no simulated peaks"
                    )
                simulated_i, excluded_q_i, q_group_total_i = _filter_geometry_fit_simulated_peaks(
                    simulated_i
                )
                if not simulated_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) has no selected Qr/Qz groups"
                    )
                simulated_i, collapsed_deg_i = _collapse_geometry_fit_simulated_peaks(
                    simulated_i,
                    merge_radius_px=float(
                        fit_auto_match_cfg.get(
                            "degenerate_merge_radius_px",
                            min(
                                6.0,
                                0.33 * float(fit_auto_match_cfg.get("search_radius_px", 24.0)),
                            ),
                        )
                    ),
                )
                if not simulated_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) has no seeds after Qr/Qz collapse"
                    )

                matched_i, stats_i, _, _ = _auto_match_background_peaks_with_relaxation(
                    simulated_i,
                    display_bg_i,
                    fit_auto_match_cfg,
                    min_matches=min_matches,
                )
                matched_i, stats_i, excluded_i = _apply_live_preview_match_exclusions(
                    matched_i,
                    stats_i,
                )
                matched_i = _attach_overlay_match_indices(matched_i)

                if len(matched_i) < min_matches:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) matched "
                        f"{len(matched_i)}/{int(stats_i.get('simulated_count', len(simulated_i)))} peaks "
                        f"(need {min_matches})"
                    )

                mean_i = float(stats_i.get("mean_match_distance_px", np.nan))
                p90_i = float(stats_i.get("p90_match_distance_px", np.nan))
                quality_fail_i = (
                    (np.isfinite(max_auto_p90) and np.isfinite(p90_i) and p90_i > max_auto_p90)
                    or (np.isfinite(max_auto_mean) and np.isfinite(mean_i) and mean_i > max_auto_mean)
                )
                if quality_fail_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) failed the quality gate "
                        f"(mean={mean_i:.2f}px, p90={p90_i:.2f}px)"
                    )

                measured_i_display = _pairs_to_measured_display_entries(matched_i)
                measured_i_native = _unrotate_display_peaks(
                    measured_i_display,
                    display_bg_i.shape,
                    k=DISPLAY_ROTATE_K,
                )
                measured_i_for_fit = _apply_orientation_to_entries(
                    measured_i_native,
                    native_bg_i.shape,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                experimental_i_for_fit = _orient_image_for_fit(
                    backend_bg_i,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                dataset_specs_local.append(
                    {
                        "dataset_index": int(bg_idx),
                        "label": Path(str(background_runtime_state.osc_files[bg_idx])).name,
                        "theta_initial": float(theta_base),
                        "measured_peaks": measured_i_for_fit,
                        "experimental_image": experimental_i_for_fit,
                    }
                )
                dataset_summary_lines.append(
                    "bg[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
                    "matches={matches} mean={mean:.3f}px p90={p90:.3f}px "
                    "excluded={excluded} q_groups_on={q_on}/{q_total} collapsed={collapsed} excluded_q={excluded_q}"
                    .format(
                        idx=int(bg_idx),
                        name=Path(str(background_runtime_state.osc_files[bg_idx])).name,
                        theta_base=float(theta_base),
                        theta_eff=float(theta_base + float(base_fit_params.get("theta_offset", 0.0))),
                        matches=len(matched_i),
                        mean=float(mean_i),
                        p90=float(p90_i),
                        excluded=int(excluded_i),
                        q_on=max(0, int(q_group_total_i) - _geometry_q_group_excluded_count()),
                        q_total=int(q_group_total_i),
                        collapsed=int(collapsed_deg_i),
                        excluded_q=int(excluded_q_i),
                    )
                )

            return dataset_specs_local, dataset_summary_lines

        _log_section(
            "Fitting variables (start values):",
            [
                (
                    f"{name}: <missing>"
                    if params.get(name) is None
                    else f"{name}: {float(params.get(name)):.6f}"
                )
                for name in var_names
            ],
        )

        fit_iterations = int(auto_match_cfg.get("fit_iterations", 3))
        fit_iterations = max(1, min(8, fit_iterations))
        rematch_between_iterations = bool(
            auto_match_cfg.get("rematch_between_iterations", False)
        )
        if joint_background_mode and rematch_between_iterations:
            _cmd_line(
                "auto-match: multi-background rematch is not yet supported; "
                "using the initial joint matches only"
            )
            rematch_between_iterations = False
        if not rematch_between_iterations and fit_iterations > 1:
            _cmd_line(
                "auto-match: background peaks fixed after initial selection; "
                "skipping iterative rematch"
            )
            fit_iterations = 1

        current_dataset_specs, joint_dataset_summary_lines = _build_joint_background_dataset_specs(
            params
        )
        if joint_dataset_summary_lines:
            _log_section(
                "Joint background datasets:",
                joint_dataset_summary_lines,
            )

        result = None
        iteration_logs: list[str] = []
        current_fit_params = dict(params)
        current_matched_pairs = list(matched_pairs)
        current_measured_for_fit = list(measured_for_fit)

        def _set_fit_param(target: dict[str, object], name: str, value: float) -> None:
            val = float(value)
            target[name] = val
            if name == "center_x" or name == "center_y":
                center_pair = list(target.get("center", [center_x_var.get(), center_y_var.get()]))
                if len(center_pair) < 2:
                    center_pair = [center_x_var.get(), center_y_var.get()]
                if name == "center_x":
                    center_pair[0] = val
                else:
                    center_pair[1] = val
                target["center"] = [float(center_pair[0]), float(center_pair[1])]
                target["center_x"] = float(center_pair[0])
                target["center_y"] = float(center_pair[1])

        for iter_idx in range(fit_iterations):
            _cmd_line(
                f"iter {iter_idx + 1}/{fit_iterations}: optimize start "
                f"(matches={len(current_matched_pairs)})"
            )
            progress_label_geometry.config(
                text=(
                    f"Geometry fit iteration {iter_idx + 1}/{fit_iterations} "
                    f"(matches={len(current_matched_pairs)})…"
                )
            )
            root.update_idletasks()
            geometry_runtime_cfg = _build_geometry_fit_runtime_config(
                geometry_refine_cfg,
                {
                    name: current_fit_params.get(name)
                    for name in var_names
                },
                _current_geometry_fit_constraint_state(var_names),
                _current_geometry_fit_parameter_domains(var_names),
            )

            result = fit_geometry_parameters(
                miller,
                intensities,
                image_size,
                current_fit_params,
                current_measured_for_fit,
                var_names,
                pixel_tol=float('inf'),
                experimental_image=experimental_image_for_fit,
                dataset_specs=current_dataset_specs if joint_background_mode else None,
                refinement_config=geometry_runtime_cfg,
            )

            if getattr(result, "x", None) is None or len(result.x) != len(var_names):
                _cmd_line(f"iter {iter_idx + 1}/{fit_iterations}: optimizer returned no parameter vector")
                iteration_logs.append(
                    f"iter={iter_idx + 1}: optimizer returned no parameter vector"
                )
                break

            for name, val in zip(var_names, result.x):
                _set_fit_param(current_fit_params, name, float(val))

            iter_rms = (
                float(getattr(result, "rms_px"))
                if np.isfinite(float(getattr(result, "rms_px", np.nan)))
                else (
                    float(np.sqrt(np.mean(result.fun ** 2)))
                    if getattr(result, "fun", None) is not None and result.fun.size
                    else float("nan")
                )
            )
            iteration_logs.append(
                (
                    f"iter={iter_idx + 1}: matches={len(current_matched_pairs)}, "
                    f"cost={float(getattr(result, 'cost', np.nan)):.6f}, "
                    f"RMS={iter_rms:.4f}px, "
                    f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}, "
                    f"orientation={orientation_choice.get('label', 'identity')}"
                )
            )
            _cmd_line(
                f"iter {iter_idx + 1}/{fit_iterations}: "
                f"cost={float(getattr(result, 'cost', np.nan)):.6f} "
                f"rms={iter_rms:.4f}px "
                f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}"
            )

            if iter_idx + 1 >= fit_iterations:
                break

            try:
                _cmd_line(f"iter {iter_idx + 1}/{fit_iterations}: rematch start")
                sim_iter = _simulate_preview_style_peaks_for_fit(
                    miller_array,
                    intensity_array,
                    image_size,
                    current_fit_params,
                )
                if not sim_iter:
                    _cmd_line(f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped (no simulated peaks)")
                    iteration_logs.append(
                        f"iter={iter_idx + 1}: rematch skipped (no simulated peaks)"
                    )
                    break
                sim_iter, excluded_q_iter, q_group_total_iter = _filter_geometry_fit_simulated_peaks(
                    sim_iter
                )
                if not sim_iter:
                    _cmd_line(
                        f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped (no Qr/Qz groups selected)"
                    )
                    iteration_logs.append(
                        f"iter={iter_idx + 1}: rematch skipped (no Qr/Qz groups selected)"
                    )
                    break
                sim_iter, collapsed_deg_iter = _collapse_geometry_fit_simulated_peaks(
                    sim_iter,
                    merge_radius_px=float(
                        fit_auto_match_cfg.get(
                            "degenerate_merge_radius_px",
                            min(
                                6.0,
                                0.33
                                * float(fit_auto_match_cfg.get("search_radius_px", 24.0)),
                            ),
                        )
                    ),
                )
                if not sim_iter:
                    _cmd_line(
                        f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped (no seeds remain after Qr/Qz collapse)"
                    )
                    iteration_logs.append(
                        f"iter={iter_idx + 1}: rematch skipped (no seeds remain after Qr/Qz collapse)"
                    )
                    break
                iteration_logs.append(
                    f"iter={iter_idx + 1}: rematch seeds={len(sim_iter)} (collapsed={collapsed_deg_iter})"
                )

                matched_iter, stats_iter, _, rematch_attempts = (
                    _auto_match_background_peaks_with_relaxation(
                        sim_iter,
                        display_background,
                        fit_auto_match_cfg,
                        min_matches=min_matches,
                    )
                )
                matched_iter, stats_iter, excluded_iter_count = _apply_live_preview_match_exclusions(
                    matched_iter,
                    stats_iter,
                )
                matched_iter = _attach_overlay_match_indices(matched_iter)
                if len(matched_iter) < min_matches:
                    best_attempt = (
                        max(
                            rematch_attempts,
                            key=lambda entry: float(entry.get("matches", 0.0)),
                        )
                        if rematch_attempts
                        else None
                    )
                    best_attempt_note = ""
                    if best_attempt is not None:
                        best_attempt_note = (
                            "; best_attempt="
                            f"{best_attempt.get('label', 'n/a')}"
                            f"({int(float(best_attempt.get('matches', 0.0)))} matches, "
                            f"qualified={int(float(best_attempt.get('qualified_summit_count', 0.0)))}, "
                            f"claimed={int(float(best_attempt.get('claimed_summit_count', 0.0)))})"
                        )
                    iteration_logs.append(
                        (
                            f"iter={iter_idx + 1}: rematch skipped "
                            f"({len(matched_iter)} < min_matches={min_matches}"
                            + (
                                f", q_groups_off={_geometry_q_group_excluded_count()}/{q_group_total_iter}"
                                if q_group_total_iter > 0
                                else ""
                            )
                            + (
                                f", excluded={excluded_iter_count}"
                                if excluded_iter_count > 0
                                else ""
                            )
                            + f"{best_attempt_note})"
                        )
                    )
                    _cmd_line(
                        f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped "
                        f"({len(matched_iter)} < {min_matches})"
                    )
                    break

                measured_iter_display = []
                for entry in matched_iter:
                    try:
                        x_val = float(entry["x"])
                        y_val = float(entry["y"])
                    except Exception:
                        continue
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                    normalized_hkl = _normalize_hkl_key(
                        entry.get("hkl", entry.get("label"))
                    )
                    measured_entry = dict(entry)
                    measured_entry["label"] = str(entry["label"])
                    measured_entry["x"] = x_val
                    measured_entry["y"] = y_val
                    measured_entry["overlay_match_index"] = int(
                        entry.get("overlay_match_index", len(measured_iter_display))
                    )
                    if normalized_hkl is not None:
                        measured_entry["hkl"] = normalized_hkl
                    measured_iter_display.append(measured_entry)
                measured_iter_native = _unrotate_display_peaks(
                    measured_iter_display,
                    display_background.shape,
                    k=DISPLAY_ROTATE_K,
                )
                sim_iter_points: list[tuple[float, float]] = []
                meas_iter_points: list[tuple[float, float]] = []
                for pair_entry, measured_entry in zip(matched_iter, measured_iter_native):
                    if not isinstance(measured_entry, dict):
                        continue
                    try:
                        sx = float(pair_entry["sim_x"])
                        sy = float(pair_entry["sim_y"])
                        mx = float(measured_entry["x"])
                        my = float(measured_entry["y"])
                    except Exception:
                        continue
                    if not (
                        np.isfinite(sx)
                        and np.isfinite(sy)
                        and np.isfinite(mx)
                        and np.isfinite(my)
                    ):
                        continue
                    sim_iter_points.append((sx, sy))
                    meas_iter_points.append((mx, my))

                orientation_choice, orientation_diag_iter = _select_fit_orientation(
                    sim_iter_points,
                    meas_iter_points,
                    tuple(int(v) for v in native_background.shape[:2]),
                    cfg=orientation_cfg,
                )
                _cmd_line(
                    f"iter {iter_idx + 1}/{fit_iterations}: rematch orientation="
                    f"{orientation_choice.get('label', 'identity')} "
                    f"best_rms={float(orientation_diag_iter.get('best_rms_px', np.nan)):.4f}px"
                )
                iteration_logs.append(
                    (
                        f"iter={iter_idx + 1}: rematch orientation={orientation_choice.get('label', 'identity')} "
                        f"(identity_rms={float(orientation_diag_iter.get('identity_rms_px', np.nan)):.4f}px, "
                        f"best_rms={float(orientation_diag_iter.get('best_rms_px', np.nan)):.4f}px, "
                        f"reason={orientation_diag_iter.get('reason', 'n/a')})"
                    )
                )
                measured_iter_for_fit = _apply_orientation_to_entries(
                    measured_iter_native,
                    native_background.shape,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                experimental_image_for_fit = _orient_image_for_fit(
                    backend_background,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )

                current_matched_pairs = matched_iter
                current_measured_for_fit = measured_iter_for_fit
                match_stats = stats_iter
                progress_label_geometry.config(
                    text=(
                        f"Geometry fit iteration {iter_idx + 1}/{fit_iterations} "
                        f"rematch complete (matches={len(current_matched_pairs)}"
                        + (
                            f", q_groups_off={_geometry_q_group_excluded_count()}/{q_group_total_iter}"
                            if q_group_total_iter > 0
                            else ""
                        )
                        + (
                            f", excluded={excluded_iter_count}"
                            if excluded_iter_count > 0
                            else ""
                        )
                        + ")."
                    )
                )
                root.update_idletasks()
                _cmd_line(
                    f"iter {iter_idx + 1}/{fit_iterations}: rematch complete "
                    f"(matches={len(current_matched_pairs)})"
                )
            except Exception as rematch_exc:
                _cmd_line(f"iter {iter_idx + 1}/{fit_iterations}: rematch failed ({rematch_exc})")
                iteration_logs.append(
                    f"iter={iter_idx + 1}: rematch failed ({rematch_exc})"
                )
                break

        if result is None:
            raise RuntimeError("Geometry optimizer did not run.")

        matched_pairs = current_matched_pairs
        measured_for_fit = current_measured_for_fit
        excluded_preview_count = int(match_stats.get("excluded_count", excluded_preview_count))

        _log_section(
            "Optimizer diagnostics:",
            [
                f"iterations={len(iteration_logs)}",
                *iteration_logs,
                f"final_orientation={orientation_choice.get('label', 'identity')}",
                f"success={getattr(result, 'success', False)}",
                f"status={getattr(result, 'status', '')}",
                f"message={(getattr(result, 'message', '') or '').strip()}",
                f"nfev={getattr(result, 'nfev', '<unknown>')}",
                f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
                f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
                f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
                f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
                f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
                f"active_mask={list(getattr(result, 'active_mask', []))}",
                *[
                    "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                        idx=int(entry.get("restart", -1)),
                        cost=float(entry.get("cost", np.nan)),
                        success=bool(entry.get("success", False)),
                        msg=str(entry.get("message", "")).strip(),
                    )
                    for entry in (getattr(result, "restart_history", []) or [])
                ],
            ],
        )

        for name in var_names:
            val = float(current_fit_params.get(name, params.get(name, 0.0)))
            if name == 'zb':
                zb_var.set(val)
            elif name == 'zs':
                zs_var.set(val)
            elif name == 'theta_initial':
                theta_initial_var.set(val)
            elif name == 'theta_offset':
                if geometry_theta_offset_var is not None:
                    geometry_theta_offset_var.set(f"{val:.6g}")
            elif name == 'psi_z':
                psi_z_var.set(val)
            elif name == 'chi':
                chi_var.set(val)
            elif name == 'cor_angle':
                cor_angle_var.set(val)
            elif name == 'gamma':
                gamma_var.set(val)
            elif name == 'Gamma':
                Gamma_var.set(val)
            elif name == 'corto_detector':
                corto_detector_var.set(val)
            elif name == 'a':
                a_var.set(val)
            elif name == 'c':
                c_var.set(val)
            elif name == 'center_x':
                center_x_var.set(val)
            elif name == 'center_y':
                center_y_var.set(val)

        if joint_background_mode and not preserve_live_theta:
            theta_initial_var.set(
                _background_theta_for_index(background_runtime_state.current_background_index, strict_count=False)
            )
            _set_background_file_status_text()

        simulation_runtime_state.profile_cache = dict(simulation_runtime_state.profile_cache)
        simulation_runtime_state.profile_cache.update(mosaic_params)
        simulation_runtime_state.profile_cache.update(
            {
                "theta_initial": theta_initial_var.get(),
                "theta_offset": _current_geometry_theta_offset(strict=False),
                "cor_angle": cor_angle_var.get(),
                "chi": chi_var.get(),
                "zs": zs_var.get(),
                "zb": zb_var.get(),
                "gamma": gamma_var.get(),
                "Gamma": Gamma_var.get(),
                "corto_detector": corto_detector_var.get(),
                "a": a_var.get(),
                "c": c_var.get(),
                "center_x": center_x_var.get(),
                "center_y": center_y_var.get(),
            }
        )

        gui_controllers.request_geometry_preview_skip_once(geometry_preview_state)
        simulation_runtime_state.last_simulation_signature = None
        schedule_update()

        rms = (
            float(getattr(result, "rms_px"))
            if np.isfinite(float(getattr(result, "rms_px", np.nan)))
            else (
                np.sqrt(np.mean(result.fun ** 2))
                if getattr(result, "fun", None) is not None and result.fun.size
                else 0.0
            )
        )
        _log_section(
            "Optimization result:",
            [f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)]
            + [f"RMS residual = {rms:.6f} px"],
        )

        fitted_params = dict(params)
        fitted_params.update(
            {
                'zb': zb_var.get(),
                'zs': zs_var.get(),
                'theta_initial': theta_initial_var.get(),
                'theta_offset': _current_geometry_theta_offset(strict=False),
                'chi': chi_var.get(),
                'cor_angle': cor_angle_var.get(),
                'psi_z': psi_z_var.get(),
                'gamma': gamma_var.get(),
                'Gamma': Gamma_var.get(),
                'corto_detector': corto_detector_var.get(),
                'a': a_var.get(),
                'c': c_var.get(),
                'center': [center_x_var.get(), center_y_var.get()],
                'center_x': center_x_var.get(),
                'center_y': center_y_var.get(),
            }
        )
        point_match_summary = getattr(result, "point_match_summary", None)
        if isinstance(point_match_summary, dict):
            _log_section(
                "Final point-match summary:",
                [
                    f"{key}={value}"
                    for key, value in sorted(point_match_summary.items())
                ]
                or ["<none>"],
            )
        _log_section(
            "Final point-match diagnostics:",
            _log_point_match_diagnostics(
                getattr(result, "point_match_diagnostics", None)
            ),
        )
        _log_section(
            "Fixed peak frame transforms after fit:",
            _log_fixed_peak_frames(
                measured_from_display,
                measured_native,
                measured_for_fit,
                center_pair=(
                    float(fitted_params.get("center_x", fitted_params["center"][0])),
                    float(fitted_params.get("center_y", fitted_params["center"][1])),
                ),
            ),
        )

        (
            _,
            sim_coords,
            meas_coords,
            sim_millers,
            meas_millers,
        ) = simulate_and_compare_hkl(
            miller,
            intensities,
            image_size,
            fitted_params,
            measured_for_fit,
            pixel_tol=float('inf'),
        )

        (
            agg_sim_coords,
            agg_meas_coords,
            agg_millers,
        ) = _aggregate_match_centers(
            sim_coords,
            meas_coords,
            sim_millers,
            meas_millers,
        )

        pixel_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
        max_display_markers = int(auto_match_cfg.get("max_display_markers", 120))
        max_display_markers = max(1, max_display_markers)

        for hkl_key, sim_center, meas_center in zip(
            agg_millers, agg_sim_coords, agg_meas_coords
        ):
            dx = sim_center[0] - meas_center[0]
            dy = sim_center[1] - meas_center[1]
            dist = math.hypot(dx, dy)
            pixel_offsets.append((hkl_key, dx, dy, dist))

        native_overlay_shape = tuple(int(v) for v in native_background.shape[:2])
        overlay_point_match_diagnostics = getattr(result, "point_match_diagnostics", None)
        if joint_background_mode and isinstance(overlay_point_match_diagnostics, list):
            overlay_point_match_diagnostics = [
                dict(entry)
                for entry in overlay_point_match_diagnostics
                if isinstance(entry, dict)
                and int(entry.get("dataset_index", -1)) == int(background_runtime_state.current_background_index)
            ]
        overlay_records = build_geometry_fit_overlay_records(
            initial_auto_pairs_display,
            overlay_point_match_diagnostics,
            native_shape=native_overlay_shape,
            orientation_choice=orientation_choice,
            sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
            background_display_rotate_k=DISPLAY_ROTATE_K,
        )
        frame_diag, frame_warning = _geometry_overlay_frame_diagnostics(
            overlay_records,
        )
        _log_section(
            "Overlay frame diagnostics:",
            [
                "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
                f"overlay_records={len(overlay_records)}",
                f"paired_records={int(frame_diag.get('paired_records', 0))}",
                f"sim_display_med_px={float(frame_diag.get('sim_display_med_px', np.nan)):.3f}",
                f"bg_display_med_px={float(frame_diag.get('bg_display_med_px', np.nan)):.3f}",
                f"sim_display_p90_px={float(frame_diag.get('sim_display_p90_px', np.nan)):.3f}",
                f"bg_display_p90_px={float(frame_diag.get('bg_display_p90_px', np.nan)):.3f}",
            ],
        )
        if frame_warning:
            _log_line(f"WARNING: {frame_warning}")
            _log_line()

        if overlay_records:
            _draw_geometry_fit_overlay(
                overlay_records,
                max_display_markers=max_display_markers,
            )
        else:
            _draw_initial_geometry_pairs_overlay(
                initial_auto_pairs_display,
                max_display_markers=max_display_markers,
            )

        _log_section(
            "Pixel offsets (native frame):",
            [
                f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px"
                for hkl, dx, dy, dist in pixel_offsets
            ] or ["No matched peaks"],
        )

        export_recs = []
        for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_sim_coords, pixel_offsets):
            export_recs.append(
                {
                    'source': 'sim',
                    'hkl': tuple(int(v) for v in hkl),
                    'x': int(x),
                    'y': int(y),
                    'dist_px': float(dist),
                }
            )
        for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_meas_coords, pixel_offsets):
            export_recs.append(
                {
                    'source': 'meas',
                    'hkl': tuple(int(v) for v in hkl),
                    'x': int(x),
                    'y': int(y),
                    'dist_px': float(dist),
                }
            )

        save_path = get_dir("downloads") / f"matched_peaks_{stamp}.npy"
        np.save(save_path, np.array(export_recs, dtype=object), allow_pickle=True)

        _log_section(
            "Fit summary:",
            [
                f"auto_matched_peaks={len(matched_pairs)}",
                f"overlay_records={len(overlay_records)}",
                f"excluded_live_preview_peaks={excluded_preview_count}",
                f"excluded_q_groups={_geometry_q_group_excluded_count()}",
                f"available_q_groups={int(q_group_total)}",
                f"excluded_q_group_peaks={int(excluded_q_peaks)}",
                f"auto_simulated_peaks={int(match_stats.get('simulated_count', len(simulated_peaks)))}",
                f"auto_distance_p90_px={float(match_stats.get('p90_match_distance_px', np.nan)):.3f}",
                f"auto_distance_clip_limit_px={float(match_stats.get('distance_clip_limit_px', np.nan)):.3f}",
                f"orientation={orientation_choice.get('label', 'identity')}",
                *[f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)],
                f"RMS residual = {rms:.6f} px",
                f"Matched peaks saved to: {save_path}",
            ],
        )

        base_summary = (
            "Auto geometry fit complete:\n"
            + "\n".join(f"{name} = {val:.4f}" for name, val in zip(var_names, result.x))
            + f"\nRMS residual = {rms:.2f} px"
            + f"\nOrientation = {orientation_choice.get('label', 'identity')}"
        )
        overlay_hint = (
            "Overlay: blue squares=initial simulated picks, amber triangles=fixed background picks, "
            "green circles=fitted simulated peaks, dashed arrows=initial->fitted sim shifts."
        )
        frame_warning_line = f"{frame_warning}\n" if frame_warning else ""
        progress_label_geometry.config(
            text=(
                f"{base_summary}\n"
                f"Auto-matched peaks: {len(matched_pairs)}/"
                f"{int(match_stats.get('simulated_count', len(simulated_peaks)))}"
                + (
                    f" | Qr/Qz groups on {max(0, q_group_total - _geometry_q_group_excluded_count())}/{q_group_total}"
                    if q_group_total > 0
                    else ""
                )
                + (
                    f" (excluded {excluded_preview_count})"
                    if excluded_preview_count > 0
                    else ""
                )
                + "\n"
                + f"{overlay_hint}\n"
                + frame_warning_line
                + f"Saved {len(export_recs)} peak records → {save_path}\n"
                + f"Fit log → {log_path}"
            )
        )
        _cmd_line(
            "done: "
            f"matched={len(matched_pairs)} "
            f"orientation={orientation_choice.get('label', 'identity')} "
            f"rms={float(rms):.4f}px "
            f"matched_peaks_file={save_path} "
            f"log_file={log_path}"
        )
        return
    except Exception as exc:
        _cmd_line(f"failed: {exc}")
        _log_line(f"Geometry fit failed: {exc}")
        progress_label_geometry.config(text=f"Geometry fit failed: {exc}")
        return
    finally:
        try:
            log_file.close()
        except Exception:
            pass

    if not _ensure_peak_overlay_data(force=False) or not simulation_runtime_state.peak_positions:
        progress_label_geometry.config(text="No simulated peaks available to pick.")
        return

    def _nearest_simulated_peak(col: float, row: float):
        """Return (index, distance^2) of the nearest simulated peak, or (None, inf)."""

        best_idx, best_d2 = None, float("inf")
        for idx, (px, py) in enumerate(simulation_runtime_state.peak_positions):
            if px < 0 or py < 0:
                continue
            d2 = (px - col) ** 2 + (py - row) ** 2
            if d2 < best_d2:
                best_idx, best_d2 = idx, d2
        return best_idx, best_d2

    def _peak_maximum_near(col: float, row: float, search_radius: int = 5):
        """Return the (col,row) of the brightest pixel near ``(col,row)``."""

        r = int(round(row))
        c = int(round(col))
        r0 = max(0, r - search_radius)
        r1 = min(background_runtime_state.current_background_display.shape[0], r + search_radius + 1)
        c0 = max(0, c - search_radius)
        c1 = min(background_runtime_state.current_background_display.shape[1], c + search_radius + 1)

        window = np.asarray(background_runtime_state.current_background_display[r0:r1, c0:c1], dtype=float)
        if window.size == 0 or not np.isfinite(window).any():
            return float(col), float(row)

        max_idx = np.nanargmax(window)
        win_r, win_c = np.unravel_index(max_idx, window.shape)
        return float(c0 + win_c), float(r0 + win_r)

    def _mark_pick(col: float, row: float, label: str, color: str, marker: str):
        point, = ax.plot([col], [row], marker, color=color, markersize=8,
                         markerfacecolor='none', zorder=7, linestyle='None')
        text = ax.text(
            col,
            row,
            label,
            color=color,
            fontsize=8,
            ha='left',
            va='bottom',
            zorder=8,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.0),
        )
        geometry_runtime_state.pick_artists.extend([point, text])
        canvas.draw_idle()

    picked_pairs = []  # list[(h,k,l), (x_real, y_real)]
    initial_manual_pairs_display: list[dict[str, object]] = []
    selection_state = {
        "expecting": "sim",
        "pending_hkl": None,
        "pending_sim_display": None,
    }
    canvas_widget = canvas.get_tk_widget()

    progress_label_geometry.config(
        text="Click a simulated peak, then the matching real peak (right click to finish)."
    )
    canvas_widget.configure(cursor="crosshair")

    click_cid = None

    def _finish_pair_collection():
        nonlocal click_cid
        if click_cid is not None:
            canvas.mpl_disconnect(click_cid)
            click_cid = None
        canvas_widget.configure(cursor="")

    def _on_geometry_pick(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 3 or getattr(event, "dblclick", False):
            _finish_pair_collection()
            if not picked_pairs:
                progress_label_geometry.config(text="No peak pairs selected; fit cancelled.")
                return

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = get_dir("downloads") / f"geometry_fit_log_{stamp}.txt"
            log_file = log_path.open("w", encoding="utf-8")

            def _log_line(text: str = ""):
                try:
                    log_file.write(text + "\n")
                    log_file.flush()
                except Exception:
                    pass

            def _log_section(title: str, lines: list[str]):
                _log_line(title)
                for line in lines:
                    _log_line(f"  {line}")
                _log_line()

            progress_label_geometry.config(text="Running geometry fit…")
            root.update_idletasks()

            measured_from_clicks = [
                {
                    "label": f"{h},{k},{l}",
                    "x": float(x),
                    "y": float(y),
                    "overlay_match_index": int(pair_idx),
                }
                for pair_idx, ((h, k, l), (x, y)) in enumerate(picked_pairs)
            ]
            _log_line(f"Geometry fit started: {stamp}")
            _log_line()
            _log_section(
                "Picked pairs (display frame):",
                [
                    f"HKL=({h},{k},{l}) display_px=({x:.3f}, {y:.3f})"
                    for (h, k, l), (x, y) in picked_pairs
                ],
            )
            native_background = _get_current_background_native()
            backend_background = _get_current_background_backend()
            if backend_background is None:
                backend_background = native_background
            measured_native = _unrotate_display_peaks(
                measured_from_clicks,
                background_runtime_state.current_background_display.shape,
                k=DISPLAY_ROTATE_K,
            )
            picked_frames = [
                {
                    "label": entry_disp.get("label"),
                    "display": (float(entry_disp.get("x")), float(entry_disp.get("y"))),
                    "native": (float(entry_nat.get("x")), float(entry_nat.get("y"))),
                }
                for entry_disp, entry_nat in zip(measured_from_clicks, measured_native)
            ]

            _log_section(
                "Unrotated measured peaks (fit frame):",
                [
                    (
                        "label="
                        f"{entry.get('label')} fit_px=({entry.get('x'):.3f}, {entry.get('y'):.3f})"
                    )
                    for entry in measured_native
                ],
            )

            sim_orientation_points: list[tuple[float, float]] = []
            meas_orientation_points: list[tuple[float, float]] = []
            try:
                simulated_for_orientation = _simulate_hkl_peak_centers_for_fit(
                    miller_array,
                    intensity_array,
                    image_size,
                    params,
                )
                simulated_lookup = {
                    str(entry.get("label")): (
                        float(entry.get("sim_col")),
                        float(entry.get("sim_row")),
                    )
                    for entry in simulated_for_orientation
                }
            except Exception:
                simulated_lookup = {}

            for entry in measured_native:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("label", ""))
                sim_pt = simulated_lookup.get(label)
                if sim_pt is None:
                    continue
                try:
                    mx = float(entry.get("x"))
                    my = float(entry.get("y"))
                    sx = float(sim_pt[0])
                    sy = float(sim_pt[1])
                except Exception:
                    continue
                if not (np.isfinite(mx) and np.isfinite(my) and np.isfinite(sx) and np.isfinite(sy)):
                    continue
                sim_orientation_points.append((sx, sy))
                meas_orientation_points.append((mx, my))

            orientation_choice, orientation_diag = _select_fit_orientation(
                sim_orientation_points,
                meas_orientation_points,
                tuple(int(v) for v in native_background.shape[:2]),
                cfg=orientation_cfg,
            )
            _log_section(
                "Orientation diagnostics:",
                [
                    f"pairs={orientation_diag.get('pairs', 0)}",
                    f"enabled={orientation_diag.get('enabled', True)}",
                    f"display_unrotate_k={DISPLAY_ROTATE_K}",
                    f"sim_display_k={SIM_DISPLAY_ROTATE_K}",
                    f"identity_rms_px={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}",
                    f"best_label={orientation_diag.get('best_label', 'identity')}",
                    f"best_rms_px={float(orientation_diag.get('best_rms_px', np.nan)):.4f}",
                    f"improvement_px={float(orientation_diag.get('improvement_px', np.nan)):.4f}",
                    f"chosen={orientation_choice.get('label', 'identity')}",
                    f"reason={orientation_diag.get('reason', 'n/a')}",
                ],
            )

            try:
                measured_for_fit = _apply_orientation_to_entries(
                    measured_native,
                    native_background.shape,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                for frame_entry, entry_fit in zip(picked_frames, measured_for_fit):
                    if isinstance(entry_fit, dict):
                        frame_entry["fit"] = (
                            float(entry_fit.get("x")), float(entry_fit.get("y"))
                        )
                    elif isinstance(entry_fit, (list, tuple)) and len(entry_fit) >= 5:
                        frame_entry["fit"] = (float(entry_fit[3]), float(entry_fit[4]))
                _log_section(
                    "Measured peaks used for fitting (after orientation):",
                    [
                        (
                            "label="
                            f"{entry.get('label')} fit_px=({entry.get('x'):.3f}, {entry.get('y'):.3f})"
                        )
                        for entry in measured_for_fit
                    ],
                )
                experimental_image_for_fit = _orient_image_for_fit(
                    backend_background,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )

                def _log_assignment_snapshot(title: str, param_set: dict[str, float]):
                    try:
                        mosaic = param_set["mosaic_params"]
                        wavelength_array = mosaic.get("wavelength_array")
                        if wavelength_array is None:
                            wavelength_array = mosaic.get("wavelength_i_array")

                        sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
                        _, hit_tables, *_ = process_peaks_parallel(
                            miller,
                            intensities,
                            image_size,
                            param_set["a"],
                            param_set["c"],
                            wavelength_array,
                            sim_buffer,
                            param_set["corto_detector"],
                            param_set["gamma"],
                            param_set["Gamma"],
                            param_set["chi"],
                            param_set.get("psi", 0.0),
                            param_set.get("psi_z", 0.0),
                            param_set["zs"],
                            param_set["zb"],
                            param_set["n2"],
                            mosaic["beam_x_array"],
                            mosaic["beam_y_array"],
                            mosaic["theta_array"],
                            mosaic["phi_array"],
                            mosaic["sigma_mosaic_deg"],
                            mosaic["gamma_mosaic_deg"],
                            mosaic["eta"],
                            wavelength_array,
                            param_set["debye_x"],
                            param_set["debye_y"],
                            param_set["center"],
                            param_set["theta_initial"],
                            param_set.get("cor_angle", 0.0),
                            np.array([1.0, 0.0, 0.0]),
                            np.array([0.0, 1.0, 0.0]),
                            save_flag=0,
                            optics_mode=_current_optics_mode_flag(),
                            solve_q_steps=int(mosaic.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS)),
                            solve_q_rel_tol=float(mosaic.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL)),
                            solve_q_mode=int(mosaic.get("solve_q_mode", DEFAULT_SOLVE_Q_MODE)),
                        )

                        maxpos = hit_tables_to_max_positions(hit_tables)
                        measured_dict = build_measured_dict(measured_for_fit)

                        rows: list[str] = []
                        for idx, (H, K, L) in enumerate(miller):
                            key = (int(round(H)), int(round(K)), int(round(L)))
                            measured_list = measured_dict.get(key)
                            if not measured_list:
                                continue

                            I0, x0, y0, I1, x1, y1 = maxpos[idx]
                            sim_candidates = [
                                (float(x0), float(y0)) if np.isfinite(x0) and np.isfinite(y0) else None,
                                (float(x1), float(y1)) if np.isfinite(x1) and np.isfinite(y1) else None,
                            ]
                            sim_candidates = [p for p in sim_candidates if p is not None]
                            if not sim_candidates:
                                continue

                            for mx, my in measured_list:
                                best = None
                                for sx, sy in sim_candidates:
                                    dx = sx - float(mx)
                                    dy = sy - float(my)
                                    dist = math.hypot(dx, dy)
                                    if best is None or dist < best[0]:
                                        best = (dist, dx, dy, sx, sy)

                                if best is None:
                                    continue

                                dist, dx, dy, sx, sy = best

                                frame_entry = next(
                                    (
                                        fr
                                        for fr in picked_frames
                                        if math.isclose(fr.get("fit", (mx, my))[0], mx, abs_tol=1e-9)
                                        and math.isclose(fr.get("fit", (mx, my))[1], my, abs_tol=1e-9)
                                        and fr.get("label") == f"{key[0]},{key[1]},{key[2]}"
                                    ),
                                    None,
                                )
                                disp_part = (
                                    f"display=({frame_entry['display'][0]:.3f}, {frame_entry['display'][1]:.3f}), "
                                    f"native=({frame_entry['native'][0]:.3f}, {frame_entry['native'][1]:.3f}), "
                                    f"fit=({mx:.3f}, {my:.3f})"
                                    if frame_entry
                                    else f"fit=({mx:.3f}, {my:.3f})"
                                )

                                rows.append(
                                    "HKL=({},{},{}) {} -> sim=({:.3f}, {:.3f}) dx={:.3f} dy={:.3f} |Δ|={:.3f}".format(
                                        key[0], key[1], key[2], disp_part, sx, sy, dx, dy, dist
                                    )
                                )

                        _log_section(title, rows or ["No measured peaks matched"],)
                    except Exception as exc:  # pragma: no cover - debug path
                        _log_section(title, [f"Failed to record assignments: {exc}"])

                def _log_pixel_match_snapshot(title: str, param_set: dict[str, float]):
                    mosaic = param_set["mosaic_params"]
                    wavelength_array = mosaic.get("wavelength_array")
                    if wavelength_array is None:
                        wavelength_array = mosaic.get("wavelength_i_array")

                    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
                    _, hit_tables, *_ = process_peaks_parallel(
                        miller,
                        intensities,
                        image_size,
                        param_set["a"],
                        param_set["c"],
                        wavelength_array,
                        sim_buffer,
                        param_set["corto_detector"],
                        param_set["gamma"],
                        param_set["Gamma"],
                        param_set["chi"],
                        param_set.get("psi", 0.0),
                        param_set.get("psi_z", 0.0),
                        param_set["zs"],
                        param_set["zb"],
                        param_set["n2"],
                        mosaic["beam_x_array"],
                        mosaic["beam_y_array"],
                        mosaic["theta_array"],
                        mosaic["phi_array"],
                        mosaic["sigma_mosaic_deg"],
                        mosaic["gamma_mosaic_deg"],
                        mosaic["eta"],
                        wavelength_array,
                        param_set["debye_x"],
                        param_set["debye_y"],
                        param_set["center"],
                        param_set["theta_initial"],
                        param_set.get("cor_angle", 0.0),
                        np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]),
                        save_flag=0,
                        optics_mode=_current_optics_mode_flag(),
                        solve_q_steps=int(mosaic.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS)),
                        solve_q_rel_tol=float(mosaic.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL)),
                        solve_q_mode=int(mosaic.get("solve_q_mode", DEFAULT_SOLVE_Q_MODE)),
                    )

                    maxpos = hit_tables_to_max_positions(hit_tables)
                    measured_dict = build_measured_dict(measured_for_fit)

                    rows: list[str] = []
                    per_residual: list[float] = []

                    def _center_from_maxpos(entry: Sequence[float]) -> tuple[float, float] | None:
                        _, x0, y0, _, x1, y1 = entry
                        candidates = [
                            (float(x0), float(y0)) if np.isfinite(x0) and np.isfinite(y0) else None,
                            (float(x1), float(y1)) if np.isfinite(x1) and np.isfinite(y1) else None,
                        ]
                        candidates = [p for p in candidates if p is not None]
                        if not candidates:
                            return None
                        cols, rows_local = zip(*candidates)
                        return float(np.mean(cols)), float(np.mean(rows_local))

                    simulated_by_hkl: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
                    for idx, (H, K, L) in enumerate(miller):
                        key = (int(round(H)), int(round(K)), int(round(L)))
                        if key not in measured_dict:
                            continue
                        center = _center_from_maxpos(maxpos[idx])
                        if center is not None:
                            simulated_by_hkl.setdefault(key, []).append(center)

                    for hkl_key, measured_list in measured_dict.items():
                        sim_list = simulated_by_hkl.get(hkl_key)
                        if not sim_list:
                            continue

                        sim_arr = np.asarray(sim_list, dtype=float)
                        sim_center = (
                            float(sim_arr[:, 0].mean()),
                            float(sim_arr[:, 1].mean()),
                        )

                        meas_arr = np.asarray(measured_list, dtype=float)
                        meas_center = (
                            float(meas_arr[:, 0].mean()),
                            float(meas_arr[:, 1].mean()),
                        )

                        dx = sim_center[0] - meas_center[0]
                        dy = sim_center[1] - meas_center[1]
                        dist = math.hypot(dx, dy)
                        per_residual.append(dist)
                        rows.append(
                            "HKL=({},{},{}) sim=({:.3f}, {:.3f}) meas=({:.3f}, {:.3f}) "
                            "dx={:.3f} dy={:.3f} |Δ|={:.3f}".format(
                                hkl_key[0],
                                hkl_key[1],
                                hkl_key[2],
                                sim_center[0],
                                sim_center[1],
                                meas_center[0],
                                meas_center[1],
                                dx,
                                dy,
                                dist,
                            )
                        )

                    rms = math.sqrt(float(np.mean(np.square(per_residual)))) if per_residual else 0.0
                    max_dist = max(per_residual) if per_residual else 0.0
                    _log_section(
                        title,
                        [
                            f"matches={len(per_residual)}, RMS={rms:.3f} px, max={max_dist:.3f} px",
                            *rows,
                        ],
                    )

                def _log_matches_snapshot(title: str, param_set: dict[str, float]):
                    try:
                        (
                            _,
                            pre_sim_coords,
                            pre_meas_coords,
                            pre_sim_millers,
                            pre_meas_millers,
                        ) = simulate_and_compare_hkl(
                            miller,
                            intensities,
                            image_size,
                            param_set,
                            measured_for_fit,
                            pixel_tol=float('inf'),
                        )
                        (
                            pre_sim_centers,
                            pre_meas_centers,
                            pre_hkls,
                        ) = _aggregate_match_centers(
                            pre_sim_coords,
                            pre_meas_coords,
                            pre_sim_millers,
                            pre_meas_millers,
                        )
                    except Exception as exc:  # pragma: no cover - debug path
                        _log_section(title, [f"Failed to collect matches: {exc}"])
                        return None

                    if not pre_hkls:
                        _log_section(title, ["No matched peaks found; residuals would be empty."])
                        return []

                    rows: list[str] = []
                    distances: list[float] = []
                    for hkl_key, sim_ctr, meas_ctr in zip(
                        pre_hkls, pre_sim_centers, pre_meas_centers
                    ):
                        dx = sim_ctr[0] - meas_ctr[0]
                        dy = sim_ctr[1] - meas_ctr[1]
                        dist = math.hypot(dx, dy)
                        distances.append(dist)
                        rows.append(
                            "HKL="
                            f"{hkl_key}: sim=({sim_ctr[0]:.3f}, {sim_ctr[1]:.3f}), "
                            f"meas=({meas_ctr[0]:.3f}, {meas_ctr[1]:.3f}), "
                            f"dx={dx:.3f}, dy={dy:.3f}, |Δ|={dist:.3f}"
                        )

                    rms = math.sqrt(float(np.mean(np.square(distances)))) if distances else 0.0
                    max_dist = max(distances) if distances else 0.0
                    summary = [
                        f"Matches={len(rows)}, RMS={rms:.3f} px, max={max_dist:.3f} px",
                        *rows,
                    ]
                    _log_section(title, summary)
                    return distances

                _log_section(
                    "Fitting variables (start values):",
                    [
                        (
                            f"{name}: <missing>"
                            if params.get(name) is None
                            else f"{name}: {float(params.get(name)):.6f}"
                        )
                        for name in var_names
                    ],
                )
                _log_matches_snapshot("Matches before fit (native frame):", params)
                _log_pixel_match_snapshot(
                    "Pixel matches before fit (native frame):", params
                )
                _log_assignment_snapshot(
                    "Match assignments before fit (native frame):", params
                )
                geometry_runtime_cfg = _build_geometry_fit_runtime_config(
                    geometry_refine_cfg,
                    {
                        name: params.get(name)
                        for name in var_names
                    },
                    _current_geometry_fit_constraint_state(var_names),
                    _current_geometry_fit_parameter_domains(var_names),
                )

                result = fit_geometry_parameters(
                    miller,
                    intensities,
                    image_size,
                    params,
                    measured_for_fit,
                    var_names,
                    pixel_tol=float('inf'),
                    experimental_image=experimental_image_for_fit,
                    refinement_config=geometry_runtime_cfg,
                )

                _log_section(
                    "Optimizer diagnostics:",
                    [
                        f"orientation={orientation_choice.get('label', 'identity')}",
                        f"success={getattr(result, 'success', False)}",
                        f"status={getattr(result, 'status', '')}",
                        f"message={(getattr(result, 'message', '') or '').strip()}",
                        f"nfev={getattr(result, 'nfev', '<unknown>')}",
                        f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
                        f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
                        f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
                        f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
                        f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
                        f"active_mask={list(getattr(result, 'active_mask', []))}",
                        *[
                            "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                                idx=int(entry.get("restart", -1)),
                                cost=float(entry.get("cost", np.nan)),
                                success=bool(entry.get("success", False)),
                                msg=str(entry.get("message", "")).strip(),
                            )
                            for entry in (getattr(result, "restart_history", []) or [])
                        ],
                    ],
                )

                for name, val in zip(var_names, result.x):
                    if name == 'zb':               zb_var.set(val)
                    elif name == 'zs':             zs_var.set(val)
                    elif name == 'theta_initial':  theta_initial_var.set(val)
                    elif name == 'theta_offset':
                        if geometry_theta_offset_var is not None:
                            geometry_theta_offset_var.set(f"{float(val):.6g}")
                    elif name == 'psi_z':          psi_z_var.set(val)
                    elif name == 'chi':            chi_var.set(val)
                    elif name == 'cor_angle':      cor_angle_var.set(val)
                    elif name == 'gamma':          gamma_var.set(val)
                    elif name == 'Gamma':          Gamma_var.set(val)
                    elif name == 'corto_detector': corto_detector_var.set(val)
                    elif name == 'a':              a_var.set(val)
                    elif name == 'c':              c_var.set(val)
                    elif name == 'center_x':       center_x_var.set(val)
                    elif name == 'center_y':       center_y_var.set(val)

                if _geometry_fit_uses_shared_theta_offset() and not preserve_live_theta:
                    theta_initial_var.set(
                        _background_theta_for_index(background_runtime_state.current_background_index, strict_count=False)
                    )
                    _set_background_file_status_text()

                # Keep the cached profile in sync with the fitted geometry so the
                # next simulation uses the updated parameters even when diagnostics
                # are disabled.
                simulation_runtime_state.profile_cache = dict(simulation_runtime_state.profile_cache)
                simulation_runtime_state.profile_cache.update(mosaic_params)
                simulation_runtime_state.profile_cache.update(
                    {
                        "theta_initial": theta_initial_var.get(),
                        "theta_offset": _current_geometry_theta_offset(strict=False),
                        "cor_angle": cor_angle_var.get(),
                        "chi": chi_var.get(),
                        "zs": zs_var.get(),
                        "zb": zb_var.get(),
                        "gamma": gamma_var.get(),
                        "Gamma": Gamma_var.get(),
                        "corto_detector": corto_detector_var.get(),
                        "a": a_var.get(),
                        "c": c_var.get(),
                        "center_x": center_x_var.get(),
                        "center_y": center_y_var.get(),
                    }
                )

                # Force a fresh simulation with the fitted values.
                gui_controllers.request_geometry_preview_skip_once(
                    geometry_preview_state
                )
                simulation_runtime_state.last_simulation_signature = None
                schedule_update()

                rms = (
                    float(getattr(result, "rms_px"))
                    if np.isfinite(float(getattr(result, "rms_px", np.nan)))
                    else (
                        np.sqrt(np.mean(result.fun**2))
                        if getattr(result, "fun", None) is not None and result.fun.size
                        else 0.0
                    )
                )
                _log_section(
                    "Optimization result:",
                    [f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)]
                    + [f"RMS residual = {rms:.6f} px"],
                )
                base_summary = (
                    "Fit complete:\n"
                    + "\n".join(
                        f"{name} = {val:.4f}" for name, val in zip(var_names, result.x)
                    )
                    + f"\nRMS residual = {rms:.2f} px"
                )

                fitted_params = dict(params)
                fitted_params.update({
                    'zb': zb_var.get(),
                    'zs': zs_var.get(),
                    'theta_initial': theta_initial_var.get(),
                    'theta_offset': _current_geometry_theta_offset(strict=False),
                    'chi': chi_var.get(),
                    'cor_angle': cor_angle_var.get(),
                    'psi_z': psi_z_var.get(),
                    'gamma': gamma_var.get(),
                    'Gamma': Gamma_var.get(),
                    'corto_detector': corto_detector_var.get(),
                    'a': a_var.get(),
                    'c': c_var.get(),
                    'center': [center_x_var.get(), center_y_var.get()],
                    'center_x': center_x_var.get(),
                    'center_y': center_y_var.get(),
                })

                _log_matches_snapshot("Matches after fit (native frame):", fitted_params)
                _log_pixel_match_snapshot(
                    "Pixel matches after fit (native frame):", fitted_params
                )
                _log_assignment_snapshot(
                    "Match assignments after fit (native frame):", fitted_params
                )

                (
                    _,
                    sim_coords,
                    meas_coords,
                    sim_millers,
                    meas_millers,
                ) = simulate_and_compare_hkl(
                    miller,
                    intensities,
                    image_size,
                    fitted_params,
                    measured_for_fit,
                    pixel_tol=float('inf'),
                )
            except Exception as exc:
                _log_line(f"Geometry fit failed: {exc}")
                try:
                    log_file.close()
                except Exception:
                    pass
                progress_label_geometry.config(
                    text=f"Geometry fit failed: {exc}"
                )
                return

            (
                agg_sim_coords,
                agg_meas_coords,
                agg_millers,
            ) = _aggregate_match_centers(
                sim_coords, meas_coords, sim_millers, meas_millers
            )

            pixel_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
            max_display_markers = int(auto_match_cfg.get("max_display_markers", 120))
            max_display_markers = max(1, max_display_markers)

            for hkl_key, sim_center, meas_center in zip(
                agg_millers, agg_sim_coords, agg_meas_coords
            ):
                dx = sim_center[0] - meas_center[0]
                dy = sim_center[1] - meas_center[1]
                dist = math.hypot(dx, dy)
                pixel_offsets.append((hkl_key, dx, dy, dist))

            native_overlay_shape = tuple(int(v) for v in native_background.shape[:2])
            overlay_records = build_geometry_fit_overlay_records(
                initial_manual_pairs_display,
                getattr(result, "point_match_diagnostics", None),
                native_shape=native_overlay_shape,
                orientation_choice=orientation_choice,
                sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
                background_display_rotate_k=DISPLAY_ROTATE_K,
            )
            frame_diag, frame_warning = _geometry_overlay_frame_diagnostics(
                overlay_records,
            )
            _log_section(
                "Overlay frame diagnostics:",
                [
                    "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
                    f"overlay_records={len(overlay_records)}",
                    f"paired_records={int(frame_diag.get('paired_records', 0))}",
                    f"sim_display_med_px={float(frame_diag.get('sim_display_med_px', np.nan)):.3f}",
                    f"bg_display_med_px={float(frame_diag.get('bg_display_med_px', np.nan)):.3f}",
                    f"sim_display_p90_px={float(frame_diag.get('sim_display_p90_px', np.nan)):.3f}",
                    f"bg_display_p90_px={float(frame_diag.get('bg_display_p90_px', np.nan)):.3f}",
                ],
            )
            if frame_warning:
                _log_line(f"WARNING: {frame_warning}")
                _log_line()

            if overlay_records:
                _draw_geometry_fit_overlay(
                    overlay_records,
                    max_display_markers=max_display_markers,
                )
            else:
                _draw_initial_geometry_pairs_overlay(
                    initial_manual_pairs_display,
                    max_display_markers=max_display_markers,
                )

            _log_section(
                "Pixel offsets (native frame):",
                [
                    f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px"
                    for hkl, dx, dy, dist in pixel_offsets
                ]
                or ["No matched peaks"],
            )

            try:
                log_file.close()
            except Exception:
                pass

            export_recs = []
            for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_sim_coords, pixel_offsets):
                export_recs.append({
                    'source': 'sim',
                    'hkl': tuple(int(v) for v in hkl),
                    'x': int(x),
                    'y': int(y),
                    'dist_px': float(dist),
                })
            for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_meas_coords, pixel_offsets):
                export_recs.append({
                    'source': 'meas',
                    'hkl': tuple(int(v) for v in hkl),
                    'x': int(x),
                    'y': int(y),
                    'dist_px': float(dist),
                })

            download_dir = get_dir("downloads")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = download_dir / f"matched_peaks_{stamp}.npy"
            np.save(save_path, np.array(export_recs, dtype=object), allow_pickle=True)

            _log_section(
                "Fit summary:",
                [
                    "Parameters:" if var_names else "Parameters: <none>",
                    *[
                        f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)
                    ],
                    f"overlay_records={len(overlay_records)}",
                    f"RMS residual = {rms:.6f} px",
                    f"Applied orientation: {orientation_choice.get('label', 'identity')}",
                    f"Matched peaks saved to: {save_path}",
                ],
            )

            if pixel_offsets:
                dist_lines = [
                    f"HKL={hkl}: |Δ|={dist:.2f}px (dx={dx:.2f}, dy={dy:.2f})"
                    for hkl, dx, dy, dist in pixel_offsets
                ]
                dist_report = "\n".join(dist_lines)
            else:
                dist_report = "No matched peaks to report distances."

            orientation_report = (
                f"Applied orientation: {orientation_choice.get('label', 'identity')}"
            )
            overlay_hint = (
                    "Overlay: blue squares=initial simulated picks, amber triangles=fixed background picks, "
                    "green circles=fitted simulated peaks, dashed arrows=initial->fitted sim shifts."
                )

            if DEBUG_ENABLED:
                final_text = (
                    f"{base_summary}\n{orientation_report}\n{overlay_hint}\n"
                    + (f"{frame_warning}\n" if frame_warning else "")
                    + f"Fit log → {log_path}"
                )
            else:
                final_text = (
                    f"{base_summary}\n\nSaved {len(export_recs)} peak records →\n{save_path}"
                    + f"\n\n{overlay_hint}\n"
                    + (f"{frame_warning}\n" if frame_warning else "")
                    + f"\nPixel offsets:\n{dist_report}\nFit log → {log_path}"
                )

            progress_label_geometry.config(text=final_text)
            return

        col, row = float(event.xdata), float(event.ydata)

        if selection_state["expecting"] == "sim":
            idx, _ = _nearest_simulated_peak(col, row)
            if idx is None:
                progress_label_geometry.config(text="No simulated peaks available to pick.")
                return
            selection_state["pending_hkl"] = simulation_runtime_state.peak_millers[idx]
            sim_col, sim_row = simulation_runtime_state.peak_positions[idx]
            selection_state["pending_sim_display"] = (float(sim_col), float(sim_row))
            _mark_pick(sim_col, sim_row, f"{selection_state['pending_hkl']} sim", '#00b894', 'o')
            progress_label_geometry.config(
                text=(
                    f"Selected simulated peak HKL={selection_state['pending_hkl']} "
                    "→ click matching real peak (right click to finish)."
                )
            )
            selection_state["expecting"] = "real"
            return

        pending = selection_state.get("pending_hkl")
        if pending is None:
            selection_state["expecting"] = "sim"
            selection_state["pending_sim_display"] = None
            progress_label_geometry.config(text="Pick a simulated peak first.")
            return

        peak_col, peak_row = _peak_maximum_near(col, row, search_radius=6)
        _mark_pick(peak_col, peak_row, f"{pending} real", '#e17055', 'x')

        pending_sim_display = selection_state.get("pending_sim_display")
        picked_pairs.append((pending, (peak_col, peak_row)))
        if pending_sim_display is not None:
            initial_manual_pairs_display.append(
                {
                    "overlay_match_index": len(initial_manual_pairs_display),
                    "hkl": tuple(int(v) for v in pending),
                    "sim_display": (
                        float(pending_sim_display[0]),
                        float(pending_sim_display[1]),
                    ),
                    "bg_display": (float(peak_col), float(peak_row)),
                }
            )
        progress_label_geometry.config(
            text=(
                f"Recorded pair for HKL={pending} at real px=({peak_col:.1f},{peak_row:.1f}). "
                "Select another simulated peak or right click to fit."
            )
        )
        selection_state["expecting"] = "sim"
        selection_state["pending_sim_display"] = None

    click_cid = canvas.mpl_connect('button_press_event', _on_geometry_pick)
    return


def on_fit_mosaic_click():
    """Run the separable mosaic-width optimizer and apply the results."""


    miller_array = np.asarray(miller, dtype=np.float64)
    if miller_array.ndim != 2 or miller_array.shape[1] != 3 or miller_array.size == 0:
        progress_label_mosaic.config(
            text="Mosaic fit unavailable: no simulated reflections loaded."
        )
        return

    intensity_array = np.asarray(intensities, dtype=np.float64)
    if intensity_array.shape[0] != miller_array.shape[0]:
        progress_label_mosaic.config(
            text="Mosaic fit unavailable: intensity array is not aligned with HKLs."
        )
        return

    mosaic_params = build_mosaic_params()
    required_keys = (
        "beam_x_array",
        "beam_y_array",
        "theta_array",
        "phi_array",
        "wavelength_array",
    )
    missing = [key for key in required_keys if not np.asarray(mosaic_params.get(key)).size]
    if missing:
        progress_label_mosaic.config(
            text="Mosaic fit unavailable: run a simulation to populate mosaic samples first."
        )
        return

    experimental_image = np.asarray(_get_current_background_backend(), dtype=np.float64)
    if experimental_image.shape != (image_size, image_size):
        progress_label_mosaic.config(
            text=(
                "Mosaic fit unavailable: experimental image has shape "
                f"{experimental_image.shape}, expected {(image_size, image_size)}."
            )
        )
        return

    params = {
        'a':             a_var.get(),
        'c':             c_var.get(),
        'lambda':        lambda_,
        'psi':           psi,
        'psi_z':         psi_z_var.get(),
        'zs':            zs_var.get(),
        'zb':            zb_var.get(),
        'chi':           chi_var.get(),
        'n2':            n2,
        'mosaic_params': mosaic_params,
        'debye_x':       debye_x_var.get(),
        'debye_y':       debye_y_var.get(),
        'center':        [center_x_var.get(), center_y_var.get()],
        'theta_initial': theta_initial_var.get(),
        'uv1':           np.array([1.0, 0.0, 0.0]),
        'uv2':           np.array([0.0, 1.0, 0.0]),
        'corto_detector': corto_detector_var.get(),
        'gamma':          gamma_var.get(),
        'Gamma':          Gamma_var.get(),
        'optics_mode':    _current_optics_mode_flag(),
    }

    progress_label_mosaic.config(text="Running mosaic optimization…")
    mosaic_progressbar.start(10)
    root.update_idletasks()

    result = None
    try:
        result = fit_mosaic_widths_separable(
            experimental_image,
            miller_array,
            intensity_array,
            image_size,
            params,
            stratify="twotheta",
        )
    except Exception as exc:  # pragma: no cover - GUI feedback path
        progress_label_mosaic.config(text=f"Mosaic fit failed: {exc}")
        return
    finally:
        mosaic_progressbar.stop()
        mosaic_progressbar["value"] = 0
        root.update_idletasks()

    if result.x is None or not np.all(np.isfinite(result.x)):
        progress_label_mosaic.config(
            text="Mosaic fit failed: optimizer returned invalid parameters."
        )
        return

    sigma_deg, gamma_deg, eta_val = map(float, result.x[:3])

    sigma_mosaic_var.set(sigma_deg)
    gamma_mosaic_var.set(gamma_deg)
    eta_var.set(eta_val)

    best_params = getattr(result, "best_params", None)
    if best_params and "mosaic_params" in best_params:
        simulation_runtime_state.profile_cache = dict(best_params["mosaic_params"])
    else:
        simulation_runtime_state.profile_cache = dict(mosaic_params)
        simulation_runtime_state.profile_cache.update(
            {
                "sigma_mosaic_deg": sigma_deg,
                "gamma_mosaic_deg": gamma_deg,
                "eta": eta_val,
            }
        )

    simulation_runtime_state.last_simulation_signature = None
    schedule_update()

    residual_norm = 0.0
    if getattr(result, "fun", None) is not None and result.fun.size:
        residual_norm = float(np.linalg.norm(result.fun))
    selected_rois = list(getattr(result, "selected_rois", []) or [])
    roi_count = len(selected_rois)
    status = "converged" if bool(getattr(result, "success", False)) else "finished"
    message = (getattr(result, "message", "") or "").strip()
    if message:
        status_text = f"{status} ({message})"
    else:
        status_text = status.capitalize()

    peaks_summary = ""
    if selected_rois:
        formatted = []
        for roi in selected_rois:
            try:
                hkl = tuple(int(round(val)) for val in roi.hkl)
            except Exception:  # pragma: no cover - defensive formatting
                hkl = tuple(roi.hkl)
            formatted.append(f"{hkl}")
        max_display = 10
        display = ", ".join(formatted[:max_display])
        remaining = len(formatted) - max_display
        if remaining > 0:
            display += f", +{remaining} more"
        peaks_summary = f"\nPeaks used: {display}"

    progress_label_mosaic.config(
        text=(
            f"Mosaic fit {status_text}\n"
            f"σ={sigma_deg:.3f}°, γ={gamma_deg:.3f}°, η={eta_val:.3f}\n"
            f"Residual norm={residual_norm:.2f} using {roi_count} ROIs"
            f"{peaks_summary}"
        )
    )


def _build_geometry_manual_fit_dataset(
    background_index: int,
    *,
    theta_base: float,
    base_fit_params: dict[str, object],
    orientation_cfg: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one saved-manual-pair geometry dataset for the optimizer."""

    background_idx = int(background_index)
    selected_entries = _geometry_manual_pairs_for_index(background_idx)
    if not selected_entries:
        raise RuntimeError(
            f"background {background_idx + 1} has no saved manual geometry pairs"
        )

    native_background, display_background = _load_background_image_by_index(background_idx)
    backend_background = _apply_background_backend_orientation(native_background)
    if backend_background is None:
        backend_background = native_background

    params_i = dict(base_fit_params)
    theta_offset = float(base_fit_params.get("theta_offset", 0.0))
    params_i["theta_initial"] = float(theta_base + theta_offset)
    simulated_peaks = _geometry_manual_simulated_peaks_for_params(
        params_i,
        prefer_cache=(background_idx == int(background_runtime_state.current_background_index)),
    )
    simulated_lookup = _geometry_manual_simulated_lookup(simulated_peaks)

    measured_display: list[dict[str, object]] = []
    initial_pairs_display: list[dict[str, object]] = []
    for pair_idx, entry in enumerate(selected_entries):
        measured_entry = dict(entry)
        measured_entry["overlay_match_index"] = int(pair_idx)
        measured_display.append(measured_entry)

        initial_entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "hkl": entry.get("hkl", entry.get("label")),
        }
        bg_coords = _geometry_manual_entry_display_coords(entry)
        if bg_coords is not None:
            initial_entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
        try:
            source_key = (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            )
        except Exception:
            source_key = None
        if source_key is not None:
            sim_entry = simulated_lookup.get(source_key)
            if isinstance(sim_entry, dict):
                try:
                    sim_col = float(sim_entry.get("sim_col"))
                    sim_row = float(sim_entry.get("sim_row"))
                except Exception:
                    sim_col = float("nan")
                    sim_row = float("nan")
                if np.isfinite(sim_col) and np.isfinite(sim_row):
                    initial_entry["sim_display"] = (float(sim_col), float(sim_row))
        initial_pairs_display.append(initial_entry)

    measured_native = _unrotate_display_peaks(
        measured_display,
        display_background.shape,
        k=DISPLAY_ROTATE_K,
    )

    sim_orientation_points: list[tuple[float, float]] = []
    meas_orientation_points: list[tuple[float, float]] = []
    sim_native_shape = (int(image_size), int(image_size))
    for initial_entry, measured_entry in zip(initial_pairs_display, measured_native):
        if not isinstance(measured_entry, dict):
            continue
        sim_display = initial_entry.get("sim_display")
        if (
            not isinstance(sim_display, (list, tuple, np.ndarray))
            or len(sim_display) < 2
        ):
            continue
        try:
            sim_native = _display_to_native_sim_coords(
                float(sim_display[0]),
                float(sim_display[1]),
                sim_native_shape,
            )
            mx = float(measured_entry.get("x"))
            my = float(measured_entry.get("y"))
        except Exception:
            continue
        if not (
            np.isfinite(sim_native[0])
            and np.isfinite(sim_native[1])
            and np.isfinite(mx)
            and np.isfinite(my)
        ):
            continue
        sim_orientation_points.append((float(sim_native[0]), float(sim_native[1])))
        meas_orientation_points.append((float(mx), float(my)))

    orientation_choice, orientation_diag = _select_fit_orientation(
        sim_orientation_points,
        meas_orientation_points,
        tuple(int(v) for v in native_background.shape[:2]),
        cfg=orientation_cfg or {},
    )
    measured_for_fit = _apply_orientation_to_entries(
        measured_native,
        native_background.shape,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )
    experimental_image_for_fit = _orient_image_for_fit(
        backend_background,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )

    label = (
        Path(str(background_runtime_state.osc_files[background_idx])).name
        if 0 <= background_idx < len(background_runtime_state.osc_files)
        else f"background_{background_idx}"
    )
    group_count = len(
        {
            entry.get("q_group_key")
            for entry in selected_entries
            if entry.get("q_group_key") is not None
        }
    )

    return {
        "dataset_index": int(background_idx),
        "label": label,
        "theta_base": float(theta_base),
        "theta_effective": float(theta_base + theta_offset),
        "group_count": int(group_count),
        "pair_count": int(len(measured_display)),
        "measured_display": measured_display,
        "measured_native": measured_native,
        "measured_for_fit": measured_for_fit,
        "initial_pairs_display": initial_pairs_display,
        "experimental_image_for_fit": experimental_image_for_fit,
        "native_background": native_background,
        "orientation_choice": orientation_choice,
        "orientation_diag": orientation_diag,
        "summary_line": (
            "bg[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
            "groups={groups} points={points} orientation={orientation}"
        ).format(
            idx=background_idx,
            name=label,
            theta_base=float(theta_base),
            theta_eff=float(theta_base + theta_offset),
            groups=int(group_count),
            points=int(len(measured_display)),
            orientation=orientation_choice.get("label", "identity"),
        ),
        "spec": {
            "dataset_index": int(background_idx),
            "label": label,
            "theta_initial": float(theta_base),
            "measured_peaks": measured_for_fit,
            "experimental_image": experimental_image_for_fit,
        },
    }


def on_fit_geometry_click():
    """Fit geometry from the saved manual Qr/Qz pair selections."""


    _clear_geometry_preview_artists()
    _clear_geometry_pick_artists()

    def _cmd_line(text: str) -> None:
        try:
            print(f"[geometry-fit] {text}", flush=True)
        except Exception:
            pass

    params = _current_geometry_fit_params()
    mosaic_params = dict(params.get("mosaic_params", {}))
    var_names = _current_geometry_fit_var_names()
    if not var_names:
        progress_label_geometry.config(text="No geometry parameters are selected for fitting.")
        return
    preserve_live_theta = (
        "theta_initial" not in var_names and "theta_offset" not in var_names
    )

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(geometry_refine_cfg, dict):
        geometry_refine_cfg = {}
    orientation_cfg = geometry_refine_cfg.get("orientation", {}) or {}
    if not isinstance(orientation_cfg, dict):
        orientation_cfg = {}
    overlay_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(overlay_cfg, dict):
        overlay_cfg = {}
    max_display_markers = max(1, int(overlay_cfg.get("max_display_markers", 120)))

    if not background_runtime_state.osc_files:
        progress_label_geometry.config(text="Geometry fit unavailable: no background image is loaded.")
        return

    if not _apply_geometry_fit_background_selection(
        trigger_update=False,
        sync_live_theta=not preserve_live_theta,
    ):
        return

    try:
        selected_background_indices = _current_geometry_fit_background_indices(strict=True)
    except Exception as exc:
        progress_label_geometry.config(
            text=f"Geometry fit unavailable: invalid fit background selection ({exc})."
        )
        return
    if int(background_runtime_state.current_background_index) not in {
        int(idx) for idx in selected_background_indices
    }:
        progress_label_geometry.config(
            text=(
                "Geometry fit unavailable: the active background must be part of the fit "
                "selection so the overlay can be drawn on the current image."
            )
        )
        return

    joint_background_mode = False
    background_theta_values: list[float] = []
    if _geometry_fit_uses_shared_theta_offset(selected_background_indices):
        if not _apply_background_theta_metadata(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        ):
            progress_label_geometry.config(
                text="Geometry fit unavailable: invalid background theta_i/shared offset settings."
            )
            return
        try:
            background_theta_values = _current_background_theta_values(strict_count=True)
            params["theta_offset"] = float(_current_geometry_theta_offset(strict=True))
        except Exception as exc:
            progress_label_geometry.config(
                text=f"Geometry fit unavailable: failed to parse background theta settings ({exc})."
            )
            return
        joint_background_mode = len(selected_background_indices) > 1
        if background_theta_values:
            params["theta_initial"] = float(
                background_theta_values[int(background_runtime_state.current_background_index)]
                + float(params.get("theta_offset", 0.0))
            )
    else:
        params["theta_offset"] = 0.0
        background_theta_values = [float(params.get("theta_initial", theta_initial_var.get()))]

    required_indices = (
        list(selected_background_indices)
        if joint_background_mode
        else [int(background_runtime_state.current_background_index)]
    )
    missing_indices = [
        idx
        for idx in required_indices
        if not _geometry_manual_pairs_for_index(idx)
    ]
    if missing_indices:
        missing_names = [
            Path(str(background_runtime_state.osc_files[idx])).name
            for idx in missing_indices
            if 0 <= idx < len(background_runtime_state.osc_files)
        ]
        progress_label_geometry.config(
            text=(
                "Geometry fit unavailable: save manual Qr/Qz pairs first for "
                + ", ".join(missing_names or [f"background {idx + 1}" for idx in missing_indices])
                + "."
            )
        )
        return

    try:
        _ensure_geometry_fit_caked_view()
    except Exception as exc:
        progress_label_geometry.config(
            text=f"Geometry fit unavailable: failed to prepare the 2D caked view ({exc})."
        )
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = get_dir("downloads") / f"geometry_fit_log_{stamp}.txt"
    log_file = None

    def _log_line(text: str = "") -> None:
        if log_file is None:
            return
        try:
            log_file.write(text + "\n")
            log_file.flush()
        except Exception:
            pass

    def _log_section(title: str, lines: Sequence[str]) -> None:
        _log_line(title)
        for line in lines:
            _log_line(f"  {line}")
        _log_line()

    try:
        log_file = log_path.open("w", encoding="utf-8")

        dataset_infos: list[dict[str, object]] = []
        current_theta_base = (
            float(background_theta_values[int(background_runtime_state.current_background_index)])
            if joint_background_mode
            else float(params.get("theta_initial", theta_initial_var.get()))
        )
        current_dataset = _build_geometry_manual_fit_dataset(
            int(background_runtime_state.current_background_index),
            theta_base=float(current_theta_base),
            base_fit_params=params,
            orientation_cfg=orientation_cfg,
        )
        dataset_infos.append(current_dataset)
        if joint_background_mode:
            for bg_idx in selected_background_indices:
                if int(bg_idx) == int(background_runtime_state.current_background_index):
                    continue
                theta_base = float(background_theta_values[int(bg_idx)])
                dataset_infos.append(
                    _build_geometry_manual_fit_dataset(
                        int(bg_idx),
                        theta_base=float(theta_base),
                        base_fit_params=params,
                        orientation_cfg=orientation_cfg,
                    )
                )

        _cmd_line(
            "start: "
            f"vars={','.join(var_names)} "
            f"datasets={len(dataset_infos)} "
            f"current_groups={current_dataset['group_count']} "
            f"current_points={current_dataset['pair_count']}"
        )
        _log_line(f"Geometry fit started: {stamp}")
        _log_line()
        _log_section(
            "Fitting variables (start values):",
            [
                f"{name}={float(params.get(name, np.nan)):.6f}"
                for name in var_names
            ],
        )
        _log_section(
            "Manual geometry datasets:",
            [str(info.get("summary_line", "")) for info in dataset_infos] or ["<none>"],
        )
        _log_section(
            "Current orientation diagnostics:",
            [
                f"pairs={current_dataset['orientation_diag'].get('pairs', 0)}",
                f"chosen={current_dataset['orientation_choice'].get('label', 'identity')}",
                f"identity_rms_px={float(current_dataset['orientation_diag'].get('identity_rms_px', np.nan)):.4f}",
                f"best_rms_px={float(current_dataset['orientation_diag'].get('best_rms_px', np.nan)):.4f}",
                f"reason={current_dataset['orientation_diag'].get('reason', 'n/a')}",
            ],
        )

        progress_label_geometry.config(text="Running geometry fit from saved manual Qr/Qz pairs…")
        root.update_idletasks()

        dataset_specs = [dict(info["spec"]) for info in dataset_infos]
        geometry_runtime_cfg = _build_geometry_fit_runtime_config(
            geometry_refine_cfg,
            {
                name: params.get(name)
                for name in var_names
            },
            _current_geometry_fit_constraint_state(var_names),
            _current_geometry_fit_parameter_domains(var_names),
        )
        result = fit_geometry_parameters(
            miller,
            intensities,
            image_size,
            params,
            current_dataset["measured_for_fit"],
            var_names,
            pixel_tol=float("inf"),
            experimental_image=current_dataset["experimental_image_for_fit"],
            dataset_specs=dataset_specs if joint_background_mode else None,
            refinement_config=geometry_runtime_cfg,
        )

        _log_section(
            "Optimizer diagnostics:",
            [
                f"success={getattr(result, 'success', False)}",
                f"status={getattr(result, 'status', '')}",
                f"message={(getattr(result, 'message', '') or '').strip()}",
                f"nfev={getattr(result, 'nfev', '<unknown>')}",
                f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
                f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
                f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
                f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
                f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
                f"active_mask={list(getattr(result, 'active_mask', []))}",
            ]
            + [
                "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                    idx=int(entry.get("restart", -1)),
                    cost=float(entry.get("cost", np.nan)),
                    success=bool(entry.get("success", False)),
                    msg=str(entry.get("message", "")).strip(),
                )
                for entry in (getattr(result, "restart_history", []) or [])
            ],
        )

        undo_state = _capture_geometry_fit_undo_state()
        for name, value in zip(var_names, getattr(result, "x", [])):
            val = float(value)
            if name == "zb":
                zb_var.set(val)
            elif name == "zs":
                zs_var.set(val)
            elif name == "theta_initial":
                theta_initial_var.set(val)
            elif name == "theta_offset":
                if geometry_theta_offset_var is not None:
                    geometry_theta_offset_var.set(f"{val:.6g}")
            elif name == "psi_z":
                psi_z_var.set(val)
            elif name == "chi":
                chi_var.set(val)
            elif name == "cor_angle":
                cor_angle_var.set(val)
            elif name == "gamma":
                gamma_var.set(val)
            elif name == "Gamma":
                Gamma_var.set(val)
            elif name == "corto_detector":
                corto_detector_var.set(val)
            elif name == "a":
                a_var.set(val)
            elif name == "c":
                c_var.set(val)
            elif name == "center_x":
                center_x_var.set(val)
            elif name == "center_y":
                center_y_var.set(val)

        if joint_background_mode and not preserve_live_theta:
            theta_initial_var.set(
                _background_theta_for_index(background_runtime_state.current_background_index, strict_count=False)
            )
        _set_background_file_status_text()
        _update_geometry_manual_pick_button_label()

        simulation_runtime_state.profile_cache = dict(simulation_runtime_state.profile_cache)
        simulation_runtime_state.profile_cache.update(mosaic_params)
        simulation_runtime_state.profile_cache.update(
            {
                "theta_initial": theta_initial_var.get(),
                "theta_offset": _current_geometry_theta_offset(strict=False),
                "cor_angle": cor_angle_var.get(),
                "chi": chi_var.get(),
                "zs": zs_var.get(),
                "zb": zb_var.get(),
                "gamma": gamma_var.get(),
                "Gamma": Gamma_var.get(),
                "corto_detector": corto_detector_var.get(),
                "a": a_var.get(),
                "c": c_var.get(),
                "center_x": center_x_var.get(),
                "center_y": center_y_var.get(),
            }
        )
        _push_geometry_fit_undo_state(undo_state)

        gui_controllers.request_geometry_preview_skip_once(geometry_preview_state)
        simulation_runtime_state.last_simulation_signature = None
        schedule_update()

        rms = (
            float(getattr(result, "rms_px"))
            if np.isfinite(float(getattr(result, "rms_px", np.nan)))
            else (
                np.sqrt(np.mean(result.fun ** 2))
                if getattr(result, "fun", None) is not None and result.fun.size
                else 0.0
            )
        )
        _log_section(
            "Optimization result:",
            [f"{name} = {float(val):.6f}" for name, val in zip(var_names, getattr(result, "x", []))]
            + [f"RMS residual = {float(rms):.6f} px"],
        )

        fitted_params = dict(params)
        fitted_params.update(
            {
                "zb": zb_var.get(),
                "zs": zs_var.get(),
                "theta_initial": theta_initial_var.get(),
                "theta_offset": _current_geometry_theta_offset(strict=False),
                "chi": chi_var.get(),
                "cor_angle": cor_angle_var.get(),
                "psi_z": psi_z_var.get(),
                "gamma": gamma_var.get(),
                "Gamma": Gamma_var.get(),
                "corto_detector": corto_detector_var.get(),
                "a": a_var.get(),
                "c": c_var.get(),
                "center": [center_x_var.get(), center_y_var.get()],
                "center_x": center_x_var.get(),
                "center_y": center_y_var.get(),
            }
        )

        point_match_summary = getattr(result, "point_match_summary", None)
        if isinstance(point_match_summary, dict):
            _log_section(
                "Point-match summary:",
                [f"{key}={value}" for key, value in sorted(point_match_summary.items())],
            )

        (
            _,
            sim_coords,
            meas_coords,
            sim_millers,
            meas_millers,
        ) = simulate_and_compare_hkl(
            miller,
            intensities,
            image_size,
            fitted_params,
            current_dataset["measured_for_fit"],
            pixel_tol=float("inf"),
        )
        (
            agg_sim_coords,
            agg_meas_coords,
            agg_millers,
        ) = _aggregate_match_centers(
            sim_coords,
            meas_coords,
            sim_millers,
            meas_millers,
        )

        pixel_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
        for hkl_key, sim_center, meas_center in zip(
            agg_millers, agg_sim_coords, agg_meas_coords
        ):
            dx = float(sim_center[0] - meas_center[0])
            dy = float(sim_center[1] - meas_center[1])
            pixel_offsets.append((hkl_key, dx, dy, float(math.hypot(dx, dy))))

        overlay_point_match_diagnostics = getattr(result, "point_match_diagnostics", None)
        if joint_background_mode and isinstance(overlay_point_match_diagnostics, list):
            overlay_point_match_diagnostics = [
                dict(entry)
                for entry in overlay_point_match_diagnostics
                if isinstance(entry, dict)
                and int(entry.get("dataset_index", -1)) == int(background_runtime_state.current_background_index)
            ]
        overlay_records = build_geometry_fit_overlay_records(
            current_dataset["initial_pairs_display"],
            overlay_point_match_diagnostics,
            native_shape=tuple(int(v) for v in current_dataset["native_background"].shape[:2]),
            orientation_choice=current_dataset["orientation_choice"],
            sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
            background_display_rotate_k=DISPLAY_ROTATE_K,
        )
        frame_diag, frame_warning = _geometry_overlay_frame_diagnostics(overlay_records)
        _log_section(
            "Overlay frame diagnostics:",
            [
                "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
                f"overlay_records={len(overlay_records)}",
                f"paired_records={int(frame_diag.get('paired_records', 0))}",
                f"sim_display_med_px={float(frame_diag.get('sim_display_med_px', np.nan)):.3f}",
                f"bg_display_med_px={float(frame_diag.get('bg_display_med_px', np.nan)):.3f}",
                f"sim_display_p90_px={float(frame_diag.get('sim_display_p90_px', np.nan)):.3f}",
                f"bg_display_p90_px={float(frame_diag.get('bg_display_p90_px', np.nan)):.3f}",
            ],
        )

        if overlay_records:
            _draw_geometry_fit_overlay(
                overlay_records,
                max_display_markers=max_display_markers,
            )
        else:
            _draw_initial_geometry_pairs_overlay(
                current_dataset["initial_pairs_display"],
                max_display_markers=max_display_markers,
            )
        _set_geometry_fit_last_overlay_state({
            "overlay_records": _copy_geometry_fit_state_value(overlay_records),
            "initial_pairs_display": _copy_geometry_fit_state_value(
                current_dataset["initial_pairs_display"]
            ),
            "max_display_markers": int(max_display_markers),
        })

        export_recs = []
        for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_sim_coords, pixel_offsets):
            export_recs.append(
                {
                    "source": "sim",
                    "hkl": tuple(int(v) for v in hkl),
                    "x": int(x),
                    "y": int(y),
                    "dist_px": float(dist),
                }
            )
        for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_meas_coords, pixel_offsets):
            export_recs.append(
                {
                    "source": "meas",
                    "hkl": tuple(int(v) for v in hkl),
                    "x": int(x),
                    "y": int(y),
                    "dist_px": float(dist),
                }
            )
        save_path = get_dir("downloads") / f"matched_peaks_{stamp}.npy"
        np.save(save_path, np.array(export_recs, dtype=object), allow_pickle=True)

        _log_section(
            "Pixel offsets (native frame):",
            [
                f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px"
                for hkl, dx, dy, dist in pixel_offsets
            ]
            or ["No matched peaks"],
        )
        _log_section(
            "Fit summary:",
            [
                f"manual_groups={current_dataset['group_count']}",
                f"manual_points={current_dataset['pair_count']}",
                f"overlay_records={len(overlay_records)}",
                f"orientation={current_dataset['orientation_choice'].get('label', 'identity')}",
                *[f"{name} = {float(val):.6f}" for name, val in zip(var_names, getattr(result, "x", []))],
                f"RMS residual = {float(rms):.6f} px",
                f"Matched peaks saved to: {save_path}",
            ],
        )

        base_summary = (
            "Manual geometry fit complete:\n"
            + "\n".join(f"{name} = {float(val):.4f}" for name, val in zip(var_names, getattr(result, "x", [])))
            + f"\nRMS residual = {float(rms):.2f} px"
            + f"\nOrientation = {current_dataset['orientation_choice'].get('label', 'identity')}"
        )
        overlay_hint = (
            "Overlay: blue squares=selected simulated points, amber triangles=saved background points, "
            "green circles=fitted simulated peaks, dashed arrows=initial->fitted sim shifts."
        )
        dist_report = (
            "\n".join(
                f"HKL={hkl}: |Δ|={dist:.2f}px (dx={dx:.2f}, dy={dy:.2f})"
                for hkl, dx, dy, dist in pixel_offsets
            )
            if pixel_offsets
            else "No matched peaks to report distances."
        )
        progress_label_geometry.config(
            text=(
                f"{base_summary}\n"
                f"Manual pairs: {current_dataset['pair_count']} points across {current_dataset['group_count']} groups"
                + (
                    f" | joint backgrounds={len(dataset_infos)}"
                    if joint_background_mode
                    else ""
                )
                + "\n"
                + overlay_hint
                + ("\n" + frame_warning if frame_warning else "")
                + f"\nSaved {len(export_recs)} peak records → {save_path}"
                + f"\nPixel offsets:\n{dist_report}"
                + f"\nFit log → {log_path}"
            )
        )
        _cmd_line(
            "done: "
            f"datasets={len(dataset_infos)} "
            f"groups={current_dataset['group_count']} "
            f"points={current_dataset['pair_count']} "
            f"rms={float(rms):.4f}px"
        )
    except Exception as exc:
        _cmd_line(f"failed: {exc}")
        _log_line(f"Geometry fit failed: {exc}")
        progress_label_geometry.config(text=f"Geometry fit failed: {exc}")
        return
    finally:
        try:
            if log_file is not None:
                log_file.close()
        except Exception:
            pass


fit_button_geometry = ttk.Button(
    app_shell_view_state.fit_actions_frame,
    text="Fit Positions & Geometry",
    command=on_fit_geometry_click
)
fit_button_geometry.pack(side=tk.TOP, padx=5, pady=2)
fit_button_geometry.config(text="Fit Geometry (LSQ)", command=on_fit_geometry_click)

live_geometry_preview_var = tk.BooleanVar(value=False)
gui_views.create_geometry_tool_action_controls(
    parent=app_shell_view_state.fit_actions_frame,
    view_state=geometry_tool_actions_view_state,
    on_undo_fit=_undo_last_geometry_fit,
    on_redo_fit=_redo_last_geometry_fit,
    on_toggle_manual_pick=_toggle_geometry_manual_pick_mode,
    on_undo_manual_placement=_undo_last_geometry_manual_placement,
    on_export_manual_pairs=_export_geometry_manual_pairs,
    on_import_manual_pairs=_import_geometry_manual_pairs,
    on_toggle_preview_exclude=_toggle_geometry_preview_exclude_mode,
    on_clear_manual_pairs=_clear_current_geometry_manual_pairs,
)
_update_geometry_fit_undo_button_state()
_update_geometry_manual_pick_button_label()
_update_geometry_preview_exclude_button_label()

gui_views.create_hkl_lookup_controls(
    parent=app_shell_view_state.fit_actions_frame,
    view_state=hkl_lookup_view_state,
    on_select_hkl=_select_peak_from_hkl_controls,
    on_toggle_hkl_pick=_toggle_hkl_pick_mode,
    on_clear_selected_peak=_clear_selected_peak,
    on_show_bragg_ewald=_open_selected_peak_intersection_figure,
    on_open_bragg_qr_groups=_open_bragg_qr_toggle_window,
)
_update_hkl_pick_button_label()

def _on_qr_cylinder_overlay_toggle():
    overlay_var = geometry_overlay_actions_view_state.show_qr_cylinder_overlay_var
    if overlay_var is not None and overlay_var.get():
        _refresh_qr_cylinder_overlay(redraw=True, update_status=True)
    else:
        _clear_qr_cylinder_overlay_artists(redraw=True)
        if "progress_label_positions" in globals():
            progress_label_positions.config(text="Qr cylinder overlay hidden.")


gui_views.create_geometry_overlay_action_controls(
    parent=app_shell_view_state.fit_actions_frame,
    view_state=geometry_overlay_actions_view_state,
    on_toggle_qr_cylinder_overlay=_on_qr_cylinder_overlay_toggle,
    on_clear_geometry_overlays=_clear_all_geometry_overlay_artists,
    on_fit_mosaic=on_fit_mosaic_click,
)


def toggle_1d_plots():
    schedule_range_update()


def toggle_caked_2d():
    show_caked_2d_var = analysis_view_controls_view_state.show_caked_2d_var
    if show_caked_2d_var is None or not show_caked_2d_var.get():
        simulation_runtime_state.caked_limits_user_override = False
    else:
        # Entering caked view should start from auto-scaled simulation limits.
        display_controls_state.simulation_limits_user_override = False
        _set_hkl_pick_mode(False)
    _drag_select_state["active"] = False
    _drag_select_state["mode"] = None
    drag_select_rect.set_visible(False)
    canvas.draw_idle()
    schedule_update()


def toggle_log_radial():
    schedule_range_update()


def toggle_log_azimuth():
    schedule_range_update()


gui_views.create_analysis_view_controls(
    parent=app_shell_view_state.analysis_views_frame,
    view_state=analysis_view_controls_view_state,
    on_toggle_1d_plots=toggle_1d_plots,
    on_toggle_caked_2d=toggle_caked_2d,
    on_toggle_log_radial=toggle_log_radial,
    on_toggle_log_azimuth=toggle_log_azimuth,
)

# Option to add fractional rods between integer L values. This can be enabled via
# configuration; the GUI control has been removed to reduce interface clutter.

def save_1d_snapshot():
    """
    Save only the final 2D simulated image as a .npy file.
    """
    file_path = filedialog.asksaveasfilename(
        initialdir=get_dir("file_dialog_dir"),
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )
    if not file_path:
        progress_label.config(text="No file path selected.")
        return
    
    if not file_path.lower().endswith(".npy"):
        file_path += ".npy"
    
    # Grab the currently displayed simulated image. ``global_image_buffer`` holds
    # the scaled image that is shown in the GUI so copying it ensures we save
    # exactly what the user sees.  ``simulation_runtime_state.last_1d_integration_data`` may be empty if
    # the simulation hasn't run yet so rely directly on the buffer instead of
    # that cache.
    sim_img = np.asarray(global_image_buffer, dtype=np.float64).copy()
    if sim_img.size == 0:
        progress_label.config(text="No simulated image available to save!")
        return
    try:
        np.save(file_path, sim_img, allow_pickle=False)
        progress_label.config(text=f"Saved simulated image to {file_path}")
    except Exception as e:
        progress_label.config(text=f"Error saving simulated image: {e}")

def save_q_space_representation():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        title="Save Q-Space Snapshot"
    )
    if not file_path:
        return

    param_dict = {
        "theta_initial": theta_initial_var.get(),
        "cor_angle": cor_angle_var.get(),
        "gamma": gamma_var.get(),
        "Gamma": Gamma_var.get(),
        "chi": chi_var.get(),
        "psi_z": psi_z_var.get(),
        "zs": zs_var.get(),
        "zb": zb_var.get(),
        "debye_x": debye_x_var.get(),
        "debye_y": debye_y_var.get(),
        "corto_detector": corto_detector_var.get(),
        "sigma_mosaic_deg": sigma_mosaic_var.get(),
        "gamma_mosaic_deg": gamma_mosaic_var.get(),
        "eta": eta_var.get(),
        "solve_q_steps": _current_solve_q_steps(),
        "solve_q_rel_tol": _current_solve_q_rel_tol(),
        "solve_q_mode": _current_solve_q_mode_flag(),
        "a": a_var.get(),
        "c": c_var.get(),
        "center_x": center_x_var.get(),
        "center_y": center_y_var.get(),
    }

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    
    mosaic_params = {
        "beam_x_array": simulation_runtime_state.profile_cache.get("beam_x_array", []),
        "beam_y_array": simulation_runtime_state.profile_cache.get("beam_y_array", []),
        "theta_array":  simulation_runtime_state.profile_cache.get("theta_array", []),
        "phi_array":    simulation_runtime_state.profile_cache.get("phi_array", []),
        "wavelength_array": simulation_runtime_state.profile_cache.get("wavelength_array", []),
        "sigma_mosaic_deg": simulation_runtime_state.profile_cache.get("sigma_mosaic_deg", 0.0),
        "gamma_mosaic_deg": simulation_runtime_state.profile_cache.get("gamma_mosaic_deg", 0.0),
        "eta": simulation_runtime_state.profile_cache.get("eta", 0.0),
        "solve_q_steps": simulation_runtime_state.profile_cache.get("solve_q_steps", _current_solve_q_steps()),
        "solve_q_rel_tol": simulation_runtime_state.profile_cache.get("solve_q_rel_tol", _current_solve_q_rel_tol()),
        "solve_q_mode": simulation_runtime_state.profile_cache.get("solve_q_mode", _current_solve_q_mode_flag()),
    }

    image_result, hit_tables, q_data, q_count, _, _ = process_peaks_parallel(
        miller,
        intensities,
        image_size,
        a_var.get(),
        c_var.get(),
        lambda_,
        sim_buffer,
        corto_detector_var.get(),
        gamma_var.get(),
        Gamma_var.get(),
        chi_var.get(),
        psi,
        psi_z_var.get(),
        zs_var.get(),
        zb_var.get(),
        n2,
        mosaic_params["beam_x_array"],
        mosaic_params["beam_y_array"],
        mosaic_params["theta_array"],
        mosaic_params["phi_array"],
        mosaic_params["sigma_mosaic_deg"],
        mosaic_params["gamma_mosaic_deg"],
        mosaic_params["eta"],
        mosaic_params["wavelength_array"],
        debye_x_var.get(),
        debye_y_var.get(),
        [center_x_var.get(), center_y_var.get()],
        theta_initial_var.get(),
        cor_angle_var.get(),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=1,
        optics_mode=_current_optics_mode_flag(),
        solve_q_steps=int(mosaic_params["solve_q_steps"]),
        solve_q_rel_tol=float(mosaic_params["solve_q_rel_tol"]),
        solve_q_mode=int(mosaic_params["solve_q_mode"]),
    )

    max_positions_local = hit_tables_to_max_positions(hit_tables)

    current_2d_display = global_image_buffer.copy()

    data_dict = {
        "parameters": param_dict,
        "q_data": q_data,
        "q_count": q_count,
        "image_2d": current_2d_display
    }
    np.save(file_path, data_dict, allow_pickle=True)
    progress_label.config(text=f"Saved Q-Space representation to {file_path}")

def save_1d_permutations():
    pass

gui_views.create_analysis_export_controls(
    parent=app_shell_view_state.analysis_exports_frame,
    view_state=analysis_export_controls_view_state,
    on_save_snapshot=save_1d_snapshot,
    on_save_q_space=save_q_space_representation,
    on_save_1d_grid=save_1d_permutations,
)

def run_debug_simulation():

    gamma_val = float(gamma_var.get())
    Gamma_val = float(Gamma_var.get())
    chi_val   = float(chi_var.get())
    zs_val    = float(zs_var.get())
    zb_val    = float(zb_var.get())
    a_val     = float(a_var.get())
    c_val     = float(c_var.get())
    theta_val = float(theta_initial_var.get())
    dx_val    = float(debye_x_var.get())
    dy_val    = float(debye_y_var.get())
    corto_val = float(corto_detector_var.get())
    cx_val    = float(center_x_var.get())
    cy_val    = float(center_y_var.get())

    mosaic_params = {
        "beam_x_array": simulation_runtime_state.profile_cache.get("beam_x_array", []),
        "beam_y_array": simulation_runtime_state.profile_cache.get("beam_y_array", []),
        "theta_array":  simulation_runtime_state.profile_cache.get("theta_array", []),
        "phi_array":    simulation_runtime_state.profile_cache.get("phi_array", []),
        "wavelength_array": simulation_runtime_state.profile_cache.get("wavelength_array", []),
        "sigma_mosaic_deg": simulation_runtime_state.profile_cache.get("sigma_mosaic_deg", 0.0),
        "gamma_mosaic_deg": simulation_runtime_state.profile_cache.get("gamma_mosaic_deg", 0.0),
        "eta": simulation_runtime_state.profile_cache.get("eta", 0.0),
        "solve_q_steps": simulation_runtime_state.profile_cache.get("solve_q_steps", _current_solve_q_steps()),
        "solve_q_rel_tol": simulation_runtime_state.profile_cache.get("solve_q_rel_tol", _current_solve_q_rel_tol()),
        "solve_q_mode": simulation_runtime_state.profile_cache.get("solve_q_mode", _current_solve_q_mode_flag()),
    }

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    image_out, maxpos, qdata, qcount = process_peaks_parallel_debug(
        miller,
        intensities,
        image_size,
        a_val,
        c_val,
        lambda_,
        sim_buffer,
        corto_val,
        gamma_val,
        Gamma_val,
        chi_val,
        psi,
        psi_z_var.get(),
        zs_val,
        zb_val,
        n2,
        mosaic_params["beam_x_array"],
        mosaic_params["beam_y_array"],
        mosaic_params["theta_array"],
        mosaic_params["phi_array"],
        mosaic_params["sigma_mosaic_deg"],
        mosaic_params["gamma_mosaic_deg"],
        mosaic_params["eta"],
        mosaic_params["wavelength_array"],
        dx_val,
        dy_val,
        [cx_val, cy_val],
        theta_val,
        cor_angle_var.get(),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=1
    )

    dump_debug_log()
    progress_label.config(text="Debug simulation complete. Log saved.")

gui_views.populate_stacked_button_group(
    workspace_panels_view_state.workspace_actions_frame,
    [
        ("Run Debug Simulation", run_debug_simulation),
        ("Force Update", lambda: update_occupancies()),
    ],
)

# Group related sliders in collapsible sections so the interface remains
# manageable as more controls are added.
geo_frame = CollapsibleFrame(
    app_shell_view_state.left_col,
    text='Geometry',
    expanded=True,
)
geo_frame.pack(fill=tk.X, padx=5, pady=5)

debye_frame = CollapsibleFrame(app_shell_view_state.left_col, text='Debye Parameters')
debye_frame.pack(fill=tk.X, padx=5, pady=5)

detector_frame = CollapsibleFrame(app_shell_view_state.right_col, text='Detector')
detector_frame.pack(fill=tk.X, padx=5, pady=5)

lattice_frame = CollapsibleFrame(
    app_shell_view_state.right_col,
    text='Lattice Parameters',
    expanded=True,
)
lattice_frame.pack(fill=tk.X, padx=5, pady=5)

mosaic_frame = CollapsibleFrame(
    app_shell_view_state.right_col,
    text='Mosaic Broadening',
)
mosaic_frame.pack(fill=tk.X, padx=5, pady=5)

initial_resolution = defaults.get('sampling_resolution', 'Low')
resolution_options = [*resolution_sample_counts.keys(), CUSTOM_SAMPLING_OPTION]
if initial_resolution not in resolution_options:
    initial_resolution = 'Low'

def _parse_sample_count(raw_value, fallback):
    return gui_controllers.parse_sampling_count(raw_value, fallback)


def _current_custom_sample_count(default=None):
    fallback = int(max(1, default if default is not None else simulation_runtime_state.num_samples))
    custom_var = sampling_optics_controls_view_state.custom_samples_var
    raw_value = custom_var.get() if custom_var is not None else fallback
    return _parse_sample_count(raw_value, fallback)


def _set_custom_sample_controls_state():
    resolution_var = sampling_optics_controls_view_state.resolution_var
    enabled = bool(
        resolution_var is not None and resolution_var.get() == CUSTOM_SAMPLING_OPTION
    )
    gui_views.set_sampling_custom_controls_enabled(
        sampling_optics_controls_view_state,
        enabled=enabled,
    )

def _refresh_resolution_display():
    resolution_var = sampling_optics_controls_view_state.resolution_var
    if resolution_var is None:
        return
    summary_text = gui_controllers.format_sampling_resolution_summary(
        resolution_var.get(),
        custom_option=CUSTOM_SAMPLING_OPTION,
        custom_value=(
            sampling_optics_controls_view_state.custom_samples_var.get()
            if sampling_optics_controls_view_state.custom_samples_var is not None
            else max(1, simulation_runtime_state.num_samples)
        ),
        preset_counts=resolution_sample_counts,
        fallback_resolution=defaults.get('sampling_resolution', 'Low'),
        fallback_count=max(1, simulation_runtime_state.num_samples),
    )
    gui_views.set_sampling_resolution_summary_text(
        sampling_optics_controls_view_state,
        summary_text,
    )


def _apply_resolution_selection(trigger_update=True):

    resolution_var = sampling_optics_controls_view_state.resolution_var
    custom_samples_var = sampling_optics_controls_view_state.custom_samples_var
    if resolution_var is None:
        return
    previous_num_samples = int(simulation_runtime_state.num_samples)
    selected_count = gui_controllers.resolve_sampling_count(
        resolution_var.get(),
        custom_option=CUSTOM_SAMPLING_OPTION,
        custom_value=custom_samples_var.get() if custom_samples_var is not None else simulation_runtime_state.num_samples,
        preset_counts=resolution_sample_counts,
        fallback_resolution=defaults.get('sampling_resolution', 'Low'),
        fallback_count=max(1, simulation_runtime_state.num_samples),
    )
    if resolution_var.get() == CUSTOM_SAMPLING_OPTION and custom_samples_var is not None:
        custom_samples_var.set(str(selected_count))

    simulation_runtime_state.num_samples = selected_count
    _refresh_resolution_display()
    if trigger_update and simulation_runtime_state.num_samples != previous_num_samples:
        update_mosaic_cache()
        schedule_update()


def _apply_custom_sample_count(_event=None):
    resolution_var = sampling_optics_controls_view_state.resolution_var
    custom_samples_var = sampling_optics_controls_view_state.custom_samples_var
    parsed_value = _current_custom_sample_count(default=max(1, simulation_runtime_state.num_samples))
    if custom_samples_var is not None:
        custom_samples_var.set(str(parsed_value))
    if resolution_var is not None and resolution_var.get() != CUSTOM_SAMPLING_OPTION:
        resolution_var.set(CUSTOM_SAMPLING_OPTION)
        return
    _apply_resolution_selection(trigger_update=True)


def ensure_valid_resolution_choice():
    resolution_var = sampling_optics_controls_view_state.resolution_var
    if resolution_var is None:
        return
    normalized = gui_controllers.normalize_sampling_resolution_choice(
        resolution_var.get(),
        allowed_options=resolution_options,
        fallback=defaults.get('sampling_resolution', 'Low'),
    )
    if resolution_var.get() != normalized:
        resolution_var.set(normalized)
    _set_custom_sample_controls_state()
    _apply_resolution_selection(trigger_update=False)

if initial_resolution != CUSTOM_SAMPLING_OPTION:
    initial_custom_samples_text = str(
        int(
            max(
                1,
                resolution_sample_counts.get(
                    initial_resolution, resolution_sample_counts['Low']
                ),
            )
        )
    )
else:
    initial_custom_samples_text = str(int(max(1, simulation_runtime_state.num_samples)))

gui_views.create_sampling_optics_controls(
    parent=mosaic_frame.frame,
    view_state=sampling_optics_controls_view_state,
    resolution_options=resolution_options,
    initial_resolution=initial_resolution,
    custom_samples_text=initial_custom_samples_text,
    resolution_count_text="",
    optics_mode_text=_normalize_optics_mode_label(defaults.get('optics_mode', 'fast')),
    on_apply_custom_samples=_apply_custom_sample_count,
)
resolution_var = sampling_optics_controls_view_state.resolution_var
custom_samples_var = sampling_optics_controls_view_state.custom_samples_var
optics_mode_var = sampling_optics_controls_view_state.optics_mode_var

def on_resolution_option_change(*_):
    _set_custom_sample_controls_state()
    _apply_resolution_selection(trigger_update=True)

_set_custom_sample_controls_state()
_apply_resolution_selection(trigger_update=False)
resolution_var.trace_add('write', on_resolution_option_change)


def on_optics_mode_change(*_):
    simulation_runtime_state.last_simulation_signature = None
    schedule_update()


optics_mode_var.trace_add('write', on_optics_mode_change)


def on_sf_prune_bias_change(*_):

    try:
        raw_value = float(sf_prune_bias_var.get())
    except (tk.TclError, ValueError, TypeError):
        raw_value = float(defaults.get("sf_prune_bias", 0.0))

    clipped_value = _clip_sf_prune_bias(raw_value)
    if not np.isclose(raw_value, clipped_value, rtol=0.0, atol=1e-12):
        sf_prune_bias_var.set(clipped_value)
        return

    _apply_bragg_qr_filters(trigger_update=False)
    simulation_runtime_state.last_simulation_signature = None
    schedule_update()


def on_solve_q_steps_change(*_):

    try:
        raw_value = float(solve_q_steps_var.get())
    except (tk.TclError, ValueError, TypeError):
        raw_value = float(defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS))

    clipped_value = _clip_solve_q_steps(raw_value)
    if not np.isclose(raw_value, float(clipped_value), rtol=0.0, atol=1e-12):
        solve_q_steps_var.set(float(clipped_value))
        return

    simulation_runtime_state.last_simulation_signature = None
    schedule_update()


def on_solve_q_rel_tol_change(*_):

    try:
        raw_value = float(solve_q_rel_tol_var.get())
    except (tk.TclError, ValueError, TypeError):
        raw_value = float(defaults.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL))

    clipped_value = _clip_solve_q_rel_tol(raw_value)
    if not np.isclose(raw_value, float(clipped_value), rtol=0.0, atol=1e-12):
        solve_q_rel_tol_var.set(float(clipped_value))
        return

    simulation_runtime_state.last_simulation_signature = None
    schedule_update()


def _set_solve_q_control_states():
    solve_q_mode_var = structure_factor_pruning_controls_view_state.solve_q_mode_var
    if solve_q_mode_var is None:
        return
    mode = _normalize_solve_q_mode_label(solve_q_mode_var.get())
    gui_views.set_structure_factor_pruning_rel_tol_enabled(
        structure_factor_pruning_controls_view_state,
        enabled=(mode != "uniform"),
    )


def on_solve_q_mode_change(*_):

    normalized = _normalize_solve_q_mode_label(solve_q_mode_var.get())
    if normalized != solve_q_mode_var.get():
        solve_q_mode_var.set(normalized)
        return

    _set_solve_q_control_states()
    simulation_runtime_state.last_simulation_signature = None
    schedule_update()


gui_views.create_structure_factor_pruning_controls(
    parent=mosaic_frame.frame,
    view_state=structure_factor_pruning_controls_view_state,
    sf_prune_bias_range=(SF_PRUNE_BIAS_MIN, SF_PRUNE_BIAS_MAX),
    sf_prune_bias_value=_clip_sf_prune_bias(defaults.get("sf_prune_bias", 0.0)),
    solve_q_mode=_normalize_solve_q_mode_label(
        defaults.get("solve_q_mode", SOLVE_Q_MODE_UNIFORM)
    ),
    solve_q_steps_range=(float(MIN_SOLVE_Q_STEPS), float(MAX_SOLVE_Q_STEPS)),
    solve_q_steps_value=float(
        _clip_solve_q_steps(defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS))
    ),
    solve_q_rel_tol_range=(
        float(MIN_SOLVE_Q_REL_TOL),
        float(MAX_SOLVE_Q_REL_TOL),
    ),
    solve_q_rel_tol_value=float(
        _clip_solve_q_rel_tol(defaults.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL))
    ),
    status_text="",
)
sf_prune_bias_var = structure_factor_pruning_controls_view_state.sf_prune_bias_var
sf_prune_bias_scale = structure_factor_pruning_controls_view_state.sf_prune_bias_scale
sf_prune_status_var = structure_factor_pruning_controls_view_state.sf_prune_status_var
solve_q_mode_var = structure_factor_pruning_controls_view_state.solve_q_mode_var
solve_q_steps_var = structure_factor_pruning_controls_view_state.solve_q_steps_var
solve_q_steps_scale = structure_factor_pruning_controls_view_state.solve_q_steps_scale
solve_q_rel_tol_var = structure_factor_pruning_controls_view_state.solve_q_rel_tol_var
solve_q_rel_tol_scale = (
    structure_factor_pruning_controls_view_state.solve_q_rel_tol_scale
)
sf_prune_bias_var.trace_add('write', on_sf_prune_bias_change)
_update_sf_prune_status_label()
solve_q_steps_var.trace_add('write', on_solve_q_steps_change)
solve_q_rel_tol_var.trace_add('write', on_solve_q_rel_tol_change)
solve_q_mode_var.trace_add('write', on_solve_q_mode_change)
_set_solve_q_control_states()

center_frame = CollapsibleFrame(app_shell_view_state.right_col, text='Beam Controls')
center_frame.pack(fill=tk.X, padx=5, pady=5)

gui_views.create_beam_mosaic_parameter_sliders(
    geometry_parent=geo_frame.frame,
    debye_parent=debye_frame.frame,
    detector_parent=detector_frame.frame,
    lattice_parent=lattice_frame.frame,
    mosaic_parent=mosaic_frame.frame,
    beam_parent=center_frame.frame,
    view_state=beam_mosaic_parameter_sliders_view_state,
    image_size=float(image_size),
    values={
        "theta_initial": defaults["theta_initial"],
        "cor_angle": defaults["cor_angle"],
        "gamma": defaults["gamma"],
        "Gamma": defaults["Gamma"],
        "chi": defaults["chi"],
        "psi_z": defaults["psi_z"],
        "zs": defaults["zs"],
        "zb": defaults["zb"],
        "debye_x": defaults["debye_x"],
        "debye_y": defaults["debye_y"],
        "corto_detector": defaults["corto_detector"],
        "a": defaults["a"],
        "c": defaults["c"],
        "sigma_mosaic_deg": defaults["sigma_mosaic_deg"],
        "gamma_mosaic_deg": defaults["gamma_mosaic_deg"],
        "eta": defaults["eta"],
        "center_x": defaults["center_x"],
        "center_y": defaults["center_y"],
        "bandwidth_percent": _clip_bandwidth_percent(
            defaults.get("bandwidth_percent", bandwidth * 100.0)
        ),
    },
    on_standard_update=schedule_update,
    on_mosaic_update=on_mosaic_slider_change,
)
theta_initial_var = beam_mosaic_parameter_sliders_view_state.theta_initial_var
theta_initial_scale = beam_mosaic_parameter_sliders_view_state.theta_initial_scale
cor_angle_var = beam_mosaic_parameter_sliders_view_state.cor_angle_var
cor_angle_scale = beam_mosaic_parameter_sliders_view_state.cor_angle_scale
gamma_var = beam_mosaic_parameter_sliders_view_state.gamma_var
gamma_scale = beam_mosaic_parameter_sliders_view_state.gamma_scale
Gamma_var = beam_mosaic_parameter_sliders_view_state.Gamma_var
Gamma_scale = beam_mosaic_parameter_sliders_view_state.Gamma_scale
chi_var = beam_mosaic_parameter_sliders_view_state.chi_var
chi_scale = beam_mosaic_parameter_sliders_view_state.chi_scale
psi_z_var = beam_mosaic_parameter_sliders_view_state.psi_z_var
psi_z_scale = beam_mosaic_parameter_sliders_view_state.psi_z_scale
zs_var = beam_mosaic_parameter_sliders_view_state.zs_var
zs_scale = beam_mosaic_parameter_sliders_view_state.zs_scale
zb_var = beam_mosaic_parameter_sliders_view_state.zb_var
zb_scale = beam_mosaic_parameter_sliders_view_state.zb_scale
debye_x_var = beam_mosaic_parameter_sliders_view_state.debye_x_var
debye_x_scale = beam_mosaic_parameter_sliders_view_state.debye_x_scale
debye_y_var = beam_mosaic_parameter_sliders_view_state.debye_y_var
debye_y_scale = beam_mosaic_parameter_sliders_view_state.debye_y_scale
corto_detector_var = beam_mosaic_parameter_sliders_view_state.corto_detector_var
corto_detector_scale = beam_mosaic_parameter_sliders_view_state.corto_detector_scale
a_var = beam_mosaic_parameter_sliders_view_state.a_var
a_scale = beam_mosaic_parameter_sliders_view_state.a_scale
c_var = beam_mosaic_parameter_sliders_view_state.c_var
c_scale = beam_mosaic_parameter_sliders_view_state.c_scale
sigma_mosaic_var = beam_mosaic_parameter_sliders_view_state.sigma_mosaic_var
sigma_mosaic_scale = beam_mosaic_parameter_sliders_view_state.sigma_mosaic_scale
gamma_mosaic_var = beam_mosaic_parameter_sliders_view_state.gamma_mosaic_var
gamma_mosaic_scale = beam_mosaic_parameter_sliders_view_state.gamma_mosaic_scale
eta_var = beam_mosaic_parameter_sliders_view_state.eta_var
eta_scale = beam_mosaic_parameter_sliders_view_state.eta_scale
center_x_var = beam_mosaic_parameter_sliders_view_state.center_x_var
center_x_scale = beam_mosaic_parameter_sliders_view_state.center_x_scale
bandwidth_percent_var = beam_mosaic_parameter_sliders_view_state.bandwidth_percent_var
bandwidth_percent_scale = beam_mosaic_parameter_sliders_view_state.bandwidth_percent_scale
center_y_var = beam_mosaic_parameter_sliders_view_state.center_y_var
center_y_scale = beam_mosaic_parameter_sliders_view_state.center_y_scale


def _clamp_psi_z_var(*_):
    try:
        value = float(psi_z_var.get())
        lo = float(psi_z_scale.cget("from"))
        hi = float(psi_z_scale.cget("to"))
    except Exception:
        return
    clipped = gui_controllers.clamp_slider_value_to_bounds(
        value,
        lower_bound=lo,
        upper_bound=hi,
        fallback=defaults.get("psi_z", 0.0),
    )
    if not math.isclose(value, clipped, rel_tol=0.0, abs_tol=1e-12):
        psi_z_var.set(clipped)


psi_z_var.trace_add("write", _clamp_psi_z_var)
_clamp_psi_z_var()
geometry_fit_parameter_specs = {
    "zb": {
        "label": "Beam Offset",
        "value_var": zb_var,
        "value_slider": zb_scale,
        "step": 0.0001,
    },
    "zs": {
        "label": "Sample Offset",
        "value_var": zs_var,
        "value_slider": zs_scale,
        "step": 0.0001,
    },
    "theta_initial": {
        "label": "Theta Sample Tilt",
        "value_var": theta_initial_var,
        "value_slider": theta_initial_scale,
        "step": 0.01,
    },
    "psi_z": {
        "label": "Goniometer Axis Yaw (about z)",
        "value_var": psi_z_var,
        "value_slider": psi_z_scale,
        "step": 0.01,
    },
    "chi": {
        "label": "Sample Pitch",
        "value_var": chi_var,
        "value_slider": chi_scale,
        "step": 0.001,
    },
    "cor_angle": {
        "label": "Goniometer Axis Pitch (about y)",
        "value_var": cor_angle_var,
        "value_slider": cor_angle_scale,
        "step": 0.01,
    },
    "gamma": {
        "label": "Detector Pitch",
        "value_var": gamma_var,
        "value_slider": gamma_scale,
        "step": 0.001,
    },
    "Gamma": {
        "label": "Detector Yaw",
        "value_var": Gamma_var,
        "value_slider": Gamma_scale,
        "step": 0.001,
    },
    "corto_detector": {
        "label": "Detector Distance",
        "value_var": corto_detector_var,
        "value_slider": corto_detector_scale,
        "step": 0.0001,
    },
    "a": {
        "label": "a Lattice Parameter",
        "value_var": a_var,
        "value_slider": a_scale,
        "step": 0.01,
    },
    "c": {
        "label": "c Lattice Parameter",
        "value_var": c_var,
        "value_slider": c_scale,
        "step": 0.01,
    },
    "center_x": {
        "label": "Beam Center Row",
        "value_var": center_x_var,
        "value_slider": center_x_scale,
        "step": 1.0,
    },
    "center_y": {
        "label": "Beam Center Col",
        "value_var": center_y_var,
        "value_slider": center_y_scale,
        "step": 1.0,
    },
}


def _default_geometry_fit_window(name: str) -> float:
    parameter_name = _geometry_fit_constraint_parameter_name(name)
    control_name = _geometry_fit_constraint_source_name(parameter_name)
    spec = geometry_fit_parameter_specs.get(control_name, {})
    if parameter_name == "theta_offset":
        try:
            current_value = float(_current_geometry_theta_offset(strict=False))
        except Exception:
            current_value = 0.0
    else:
        try:
            current_value = float(spec["value_var"].get())
        except Exception:
            current_value = 0.0
    try:
        step = abs(float(spec.get("step", 0.01)))
    except Exception:
        step = 0.01
    step = max(step, 1.0e-6)
    domain = _current_geometry_fit_parameter_domains([parameter_name]).get(parameter_name)
    domain_span = 0.0
    if isinstance(domain, tuple) and len(domain) >= 2:
        domain_span = max(0.0, float(domain[1]) - float(domain[0]))

    fit_geometry_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(fit_geometry_cfg, dict):
        fit_geometry_cfg = {}
    bounds_cfg = fit_geometry_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}
    entry = bounds_cfg.get(parameter_name)

    default_window = float("nan")
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            lo = float(entry[0])
            hi = float(entry[1])
            default_window = max(abs(current_value - lo), abs(hi - current_value))
        except Exception:
            default_window = float("nan")
    elif isinstance(entry, dict):
        mode = str(entry.get("mode", "absolute")).strip().lower()
        try:
            min_raw = float(entry.get("min")) if entry.get("min") is not None else float("nan")
        except Exception:
            min_raw = float("nan")
        try:
            max_raw = float(entry.get("max")) if entry.get("max") is not None else float("nan")
        except Exception:
            max_raw = float("nan")
        if mode in {"relative", "rel", "relative_min0", "rel_min0"}:
            candidates = [abs(v) for v in (min_raw, max_raw) if np.isfinite(v)]
            if candidates:
                default_window = max(candidates)
        else:
            candidates = [
                abs(current_value - v)
                for v in (min_raw, max_raw)
                if np.isfinite(v)
            ]
            if candidates:
                default_window = max(candidates)

    if not np.isfinite(default_window) or default_window <= 0.0:
        default_window = max(
            step * 10.0,
            0.02 * domain_span,
            0.1 * max(abs(current_value), 1.0),
        )

    if domain_span > 0.0:
        default_window = min(default_window, domain_span)

    return max(float(default_window), step)


def _default_geometry_fit_pull(name: str, window: float) -> float:
    parameter_name = _geometry_fit_constraint_parameter_name(name)
    fit_geometry_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(fit_geometry_cfg, dict):
        fit_geometry_cfg = {}
    priors_cfg = fit_geometry_cfg.get("priors", {}) or {}
    if not isinstance(priors_cfg, dict):
        priors_cfg = {}
    entry = priors_cfg.get(parameter_name)
    if not isinstance(entry, dict):
        return 0.0
    try:
        sigma = float(entry.get("sigma"))
    except Exception:
        sigma = float("nan")
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(window) or window <= 0.0:
        return 0.0
    inferred = (1.0 - min(max(sigma / window, 0.05), 1.0)) / 0.95
    if not np.isfinite(inferred):
        return 0.0
    return min(max(float(inferred), 0.0), 1.0)


gui_views.create_geometry_fit_constraints_panel(
    parent=app_shell_view_state.fit_body,
    root=root,
    view_state=geometry_fit_constraints_view_state,
    after=fit_frame,
    on_mousewheel=lambda event: _scroll_geometry_fit_constraints(event),
)
geometry_fit_constraints_body = geometry_fit_constraints_view_state.body
if geometry_fit_constraints_body is None:
    raise RuntimeError("Geometry-fit constraints body was not created.")


def _scroll_geometry_fit_constraints(event):
    return gui_views.scroll_geometry_fit_constraints_canvas(
        geometry_fit_constraints_view_state,
        pointer_x=root.winfo_pointerx(),
        pointer_y=root.winfo_pointery(),
        event=event,
    )

for name in GEOMETRY_FIT_PARAM_ORDER:
    spec = geometry_fit_parameter_specs.get(name, {})
    try:
        step = abs(float(spec.get("step", 0.01)))
    except Exception:
        step = 0.01
    step = max(step, 1.0e-6)
    default_window = _default_geometry_fit_window(name)
    parameter_name = _geometry_fit_constraint_parameter_name(name)
    domain = _current_geometry_fit_parameter_domains([parameter_name]).get(parameter_name)
    domain_span = 0.0
    if isinstance(domain, tuple) and len(domain) >= 2:
        domain_span = max(0.0, float(domain[1]) - float(domain[0]))
    window_slider_max = max(
        default_window,
        step * 20.0,
        default_window * 4.0,
        0.05 * domain_span,
    )
    if domain_span > 0.0:
        window_slider_max = min(window_slider_max, domain_span)
    window_slider_max = max(window_slider_max, default_window, step)

    row_text = str(spec.get("label", name))
    if name == "theta_initial" and _geometry_fit_uses_shared_theta_offset():
        row_text = "Theta Shared Offset"
    row = ttk.LabelFrame(
        geometry_fit_constraints_body,
        text=row_text,
    )
    window_var, _window_slider = create_slider(
        "Allowed deviation (±)",
        0.0,
        window_slider_max,
        default_window,
        step,
        row,
        allow_range_expand=True,
        range_expand_pad=max(step, default_window * 0.25),
    )
    pull_var, _pull_slider = create_slider(
        "Stay-close pull",
        0.0,
        1.0,
        _default_geometry_fit_pull(name, default_window),
        0.01,
        row,
    )
    ttk.Label(
        row,
        text="0 = free fit, 1 = strongest preference for the starting value.",
        wraplength=360,
        justify="left",
    ).pack(fill=tk.X, padx=6, pady=(0, 4))
    gui_views.set_geometry_fit_constraint_control(
        geometry_fit_constraints_view_state,
        name=name,
        row=row,
        window_var=window_var,
        pull_var=pull_var,
    )

for _toggle_var in geometry_fit_toggle_vars.values():
    _toggle_var.trace_add("write", _sync_geometry_fit_constraint_rows)
_sync_geometry_fit_constraint_rows()
_refresh_geometry_fit_theta_checkbox_label()

# Slider controlling contribution of the first CIF file, only if a second CIF
# was provided.
gui_views.create_cif_weight_controls(
    parent=app_shell_view_state.right_col,
    view_state=cif_weight_controls_view_state,
    has_second_cif=bool(has_second_cif),
    weight1=float(weight1),
    weight2=float(weight2),
)
weight1_var = cif_weight_controls_view_state.weight1_var
weight2_var = cif_weight_controls_view_state.weight2_var
if weight1_var is None or weight2_var is None:
    raise RuntimeError("CIF weight controls did not create slider variables.")


def update_weights(*args):
    """Recompute intensities using the current CIF weights."""
    gui_structure_model.update_weighted_intensities(
        structure_model_state,
        weight1=weight1_var.get(),
        weight2=weight2_var.get(),
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        schedule_update=schedule_update,
    )
    _sync_structure_model_aliases()


if has_second_cif:
    weight1_var.trace_add('write', update_weights)
    weight2_var.trace_add('write', update_weights)
# ---------------------------------------------------------------------------
#  OCCUPANCY CONTROLS: one control per structure site in the loaded CIF.
# ---------------------------------------------------------------------------
occ_vars = [tk.DoubleVar(value=float(val)) for val in occ]
atom_site_fract_vars = [
    {
        "x": tk.DoubleVar(value=float(row["x"])),
        "y": tk.DoubleVar(value=float(row["y"])),
        "z": tk.DoubleVar(value=float(row["z"])),
    }
    for row in atom_site_fractional_metadata
]
structure_model_state.occ_vars = list(occ_vars)
structure_model_state.atom_site_fract_vars = list(atom_site_fract_vars)
_sync_structure_model_aliases()


def _occupancy_label_text(site_idx: int, *, input_label: bool = False) -> str:
    """Build a user-facing occupancy label for a unique atom site."""

    idx = int(site_idx)
    atom_name = (
        occupancy_site_labels[idx]
        if idx < len(occupancy_site_labels)
        else "not in CIF"
    )
    prefix = "Input Occupancy" if input_label else "Occupancy"
    return f"{prefix} Site {idx + 1} ({atom_name})"


def _rebuild_diffraction_inputs(
    new_occ,
    p_vals,
    weights,
    a_axis,
    c_axis,
    *,
    force=False,
    trigger_update=True,
):
    """Refresh cached HT curves and peak lists for the current settings."""
    structure_model_state.two_theta_range = two_theta_range
    gui_structure_model.rebuild_diffraction_inputs(
        structure_model_state,
        new_occ=new_occ,
        p_vals=p_vals,
        weights=weights,
        a_axis=a_axis,
        c_axis=c_axis,
        finite_stack_flag=bool(finite_stack_var.get()),
        layers=int(max(1, stack_layers_var.get())),
        phase_delta_expression_current=_current_phase_delta_expression(),
        phi_l_divisor_current=_current_phi_l_divisor(),
        atom_site_values=_current_atom_site_fractional_values(),
        iodine_z_current=_current_iodine_z(),
        atom_site_override_state=atom_site_override_state,
        simulation_runtime_state=simulation_runtime_state,
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        build_intensity_dataframes=build_intensity_dataframes,
        apply_bragg_qr_filters=_apply_bragg_qr_filters,
        schedule_update=schedule_update,
        weight1=weight1_var.get(),
        weight2=weight2_var.get(),
        tcl_error_types=(tk.TclError,),
        force=force,
        trigger_update=trigger_update,
    )
    _sync_structure_model_aliases()


def update_occupancies(*args):
    """Recompute Hendricks–Teller curves when occupancies or p-values change."""

    try:
        new_occ = [float(var.get()) for var in occ_vars]
    except (tk.TclError, ValueError):
        return

    if not all(np.isfinite(v) for v in new_occ):
        return

    # Keep occupancies physically meaningful and reflect clamped values in the UI.
    clamped_occ = gui_controllers.clamp_site_occupancy_values(
        new_occ,
        fallback_values=occ,
    )
    for var, val in zip(occ_vars, clamped_occ):
        try:
            current = float(var.get())
        except (tk.TclError, ValueError):
            current = val
        if not math.isclose(current, val, rel_tol=1e-12, abs_tol=1e-12):
            var.set(val)
    new_occ = clamped_occ

    try:
        p_vals = [float(p0_var.get()), float(p1_var.get()), float(p2_var.get())]
        w_raw = [float(w0_var.get()), float(w1_var.get()), float(w2_var.get())]
    except (tk.TclError, ValueError):
        return
    weights = gui_controllers.normalize_stacking_weight_values(w_raw)

    _rebuild_diffraction_inputs(new_occ, p_vals, weights, a_var.get(), c_var.get())


def _sync_finite_controls():
    enabled = bool(
        finite_stack_controls_view_state.finite_stack_var is not None
        and finite_stack_controls_view_state.finite_stack_var.get()
    )
    gui_views.set_finite_stack_layer_controls_enabled(
        finite_stack_controls_view_state,
        enabled=enabled,
    )


def _normalize_layer_value(raw_value):
    fallback = (
        stack_layers_var.get()
        if finite_stack_controls_view_state.stack_layers_var is not None
        else 1
    )
    return gui_controllers.normalize_finite_stack_layer_count(raw_value, fallback)


def _normalize_phase_delta_expression_value(raw_value):
    fallback = _current_phase_delta_expression()
    return gui_controllers.normalize_finite_stack_phase_delta_expression(
        raw_value,
        fallback=fallback,
    )


def _normalize_phi_l_divisor_value(raw_value):
    return gui_controllers.normalize_finite_stack_phi_l_divisor(
        raw_value,
        fallback=_current_phi_l_divisor(),
    )


def _sync_layer_entry_from_var(*_):
    if finite_stack_controls_view_state.layers_entry_var is None:
        return
    normalized = gui_controllers.format_finite_stack_layer_count(stack_layers_var.get())
    if finite_stack_controls_view_state.layers_entry_var.get().strip() != normalized:
        gui_views.set_finite_stack_layer_entry_text(
            finite_stack_controls_view_state,
            normalized,
        )


def _commit_layer_entry(_event=None):
    if finite_stack_controls_view_state.layers_entry_var is None:
        return
    value = _normalize_layer_value(finite_stack_controls_view_state.layers_entry_var.get())
    gui_views.ensure_finite_stack_layer_scale_max(
        finite_stack_controls_view_state,
        value,
    )
    changed = stack_layers_var.get() != value
    if changed:
        stack_layers_var.set(value)
    _sync_layer_entry_from_var()
    if changed and finite_stack_var.get():
        update_occupancies()


def _commit_phase_delta_expression_entry(_event=None):
    if finite_stack_controls_view_state.phase_delta_entry_var is None:
        return

    try:
        value = _normalize_phase_delta_expression_value(
            finite_stack_controls_view_state.phase_delta_entry_var.get()
        )
    except ValueError as exc:
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            _current_phase_delta_expression(),
        )
        progress_label.config(text=f"Invalid phase delta equation: {exc}")
        return

    current = _current_phase_delta_expression()
    if current != value:
        phase_delta_expr_var.set(value)
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            value,
        )
        update_occupancies()
        progress_label.config(text="Updated phase delta equation.")
        return

    gui_views.set_finite_stack_phase_delta_entry_text(
        finite_stack_controls_view_state,
        value,
    )


def _sync_phi_l_divisor_entry_from_var(*_):
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is None:
        return
    normalized = gui_controllers.format_finite_stack_phi_l_divisor(
        _current_phi_l_divisor()
    )
    if finite_stack_controls_view_state.phi_l_divisor_entry_var.get().strip() != normalized:
        gui_views.set_finite_stack_phi_l_divisor_entry_text(
            finite_stack_controls_view_state,
            normalized,
        )


def _commit_phi_l_divisor_entry(_event=None):
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is None:
        return

    value = _normalize_phi_l_divisor_value(
        finite_stack_controls_view_state.phi_l_divisor_entry_var.get()
    )
    current = _current_phi_l_divisor()
    if not math.isclose(current, value, rel_tol=1e-12, abs_tol=1e-12):
        phi_l_divisor_var.set(float(value))
        _sync_phi_l_divisor_entry_from_var()
        update_occupancies()
        progress_label.config(text="Updated HT phi L divisor.")
        return

    _sync_phi_l_divisor_entry_from_var()


def _on_finite_toggle():
    _sync_finite_controls()
    update_occupancies()


def _on_layer_slider(val):
    value = _normalize_layer_value(val)
    if stack_layers_var.get() != value:
        stack_layers_var.set(value)
    _sync_layer_entry_from_var()
    if finite_stack_var.get():
        update_occupancies()


# Sliders for three disorder probabilities and weights inside a collapsible frame
gui_views.create_stacking_parameter_panels(
    parent=app_shell_view_state.right_col,
    view_state=stacking_parameter_controls_view_state,
)
gui_views.create_finite_stack_controls(
    parent=stacking_parameter_controls_view_state.stack_frame.frame,
    view_state=finite_stack_controls_view_state,
    finite_stack=bool(defaults['finite_stack']),
    stack_layers=int(defaults['stack_layers']),
    phi_l_divisor=float(defaults['phi_l_divisor']),
    phase_delta_expression=str(defaults['phase_delta_expression']),
    on_toggle_finite_stack=_on_finite_toggle,
    on_layer_slider=_on_layer_slider,
    on_commit_layer_entry=_commit_layer_entry,
    on_commit_phi_l_divisor_entry=_commit_phi_l_divisor_entry,
    on_commit_phase_delta_expression_entry=_commit_phase_delta_expression_entry,
)
finite_stack_var = finite_stack_controls_view_state.finite_stack_var
stack_layers_var = finite_stack_controls_view_state.stack_layers_var
phase_delta_expr_var = finite_stack_controls_view_state.phase_delta_expr_var
phi_l_divisor_var = finite_stack_controls_view_state.phi_l_divisor_var
stack_layers_var.trace_add("write", _sync_layer_entry_from_var)
_sync_layer_entry_from_var()
_sync_finite_controls()
phi_l_divisor_var.trace_add("write", _sync_phi_l_divisor_entry_from_var)
_sync_phi_l_divisor_entry_from_var()
gui_views.create_stacking_probability_sliders(
    parent=stacking_parameter_controls_view_state.stack_frame.frame,
    view_state=stacking_parameter_controls_view_state,
    values={
        "p0": float(defaults["p0"]),
        "w0": float(defaults["w0"]),
        "p1": float(defaults["p1"]),
        "w1": float(defaults["w1"]),
        "p2": float(defaults["p2"]),
        "w2": float(defaults["w2"]),
    },
    on_update=update_occupancies,
)
p0_var = stacking_parameter_controls_view_state.p0_var
w0_var = stacking_parameter_controls_view_state.w0_var
p1_var = stacking_parameter_controls_view_state.p1_var
w1_var = stacking_parameter_controls_view_state.w1_var
p2_var = stacking_parameter_controls_view_state.p2_var
w2_var = stacking_parameter_controls_view_state.w2_var


def _rebuild_occupancy_controls():
    """Recreate occupancy sliders/entries for the current ``occ_vars`` list."""

    gui_views.rebuild_occupancy_controls(
        view_state=stacking_parameter_controls_view_state,
        occ_vars=occ_vars,
        occupancy_label_text=lambda idx: _occupancy_label_text(idx),
        occupancy_input_label_text=lambda idx: _occupancy_label_text(
            idx,
            input_label=True,
        ),
        on_update=update_occupancies,
    )


def _current_occupancy_values():
    """Return the occupancy values currently shown in the controls."""

    return gui_structure_model.current_occupancy_values(
        structure_model_state,
        tcl_error_types=(tk.TclError,),
    )


def _atom_site_fractional_label_text(site_idx: int) -> str:
    idx = int(site_idx)
    if idx < len(atom_site_fractional_metadata):
        return str(atom_site_fractional_metadata[idx].get("label", f"site_{idx + 1}"))
    return f"site_{idx + 1}"


def _rebuild_atom_site_fractional_controls():
    """Recreate x/y/z entry controls for atom-site fractional coordinates."""

    gui_views.rebuild_atom_site_fractional_controls(
        view_state=stacking_parameter_controls_view_state,
        atom_site_fract_vars=atom_site_fract_vars,
        atom_site_label_text=_atom_site_fractional_label_text,
        on_update=update_occupancies,
        empty_text="No _atom_site_fract_x/_y/_z loop found in the active CIF.",
    )


def _reset_structure_model_control_vars(
    occupancy_values,
    atom_site_values,
):
    """Replace occupancy and atom-site Tk variables for the active structure model."""

    global occ_vars, atom_site_fract_vars

    if len(occ_vars) != len(occupancy_values):
        occ_vars = [tk.DoubleVar(value=float(value)) for value in occupancy_values]
    else:
        for occ_var, value in zip(occ_vars, occupancy_values):
            occ_var.set(float(value))

    atom_site_fract_vars = [
        {
            "x": tk.DoubleVar(value=float(x_val)),
            "y": tk.DoubleVar(value=float(y_val)),
            "z": tk.DoubleVar(value=float(z_val)),
        }
        for (x_val, y_val, z_val) in atom_site_values
    ]
    _rebuild_occupancy_controls()
    _rebuild_atom_site_fractional_controls()


def _apply_primary_cif_path(raw_path):
    """Load a new primary CIF file and rebuild diffraction inputs."""

    global occ_vars, atom_site_fract_vars

    text_path = str(raw_path).strip().strip('"').strip("'")
    if not text_path:
        progress_label.config(text="No CIF file path provided.")
        return

    candidate = Path(text_path).expanduser()
    if not candidate.is_file():
        progress_label.config(text=f"CIF file not found: {candidate}")
        return

    snapshot = gui_structure_model.capture_primary_cif_reload_snapshot(
        structure_model_state,
        current_occ_values=_current_occupancy_values(),
        current_atom_site_values=_current_atom_site_fractional_values(),
    )
    try:
        old_slider_a = float(a_var.get())
    except Exception:
        old_slider_a = snapshot.default_a
    try:
        old_slider_c = float(c_var.get())
    except Exception:
        old_slider_c = snapshot.default_c
    try:
        reload_plan = gui_structure_model.prepare_primary_cif_reload_plan(
            structure_model_state,
            str(candidate),
            current_occ_values=snapshot.current_occ_values,
            clamp_site_occupancy_values=gui_controllers.clamp_site_occupancy_values,
        )

        try:
            p_vals = [float(p0_var.get()), float(p1_var.get()), float(p2_var.get())]
        except (tk.TclError, ValueError):
            p_vals = [float(defaults["p0"]), float(defaults["p1"]), float(defaults["p2"])]
        try:
            w_raw = [float(w0_var.get()), float(w1_var.get()), float(w2_var.get())]
        except (tk.TclError, ValueError):
            w_raw = [float(defaults["w0"]), float(defaults["w1"]), float(defaults["w2"])]
        weights = gui_controllers.normalize_stacking_weight_values(w_raw)

        _reset_structure_model_control_vars(
            reload_plan.occ,
            reload_plan.atom_site_values,
        )
        gui_structure_model.apply_primary_cif_reload_plan(
            structure_model_state,
            reload_plan,
            occ_vars=occ_vars,
            atom_site_fract_vars=atom_site_fract_vars,
            has_second_cif=has_second_cif,
        )
        _sync_structure_model_aliases()

        cif_file_var.set(cif_file)
        _reset_atom_site_override_cache()
        _rebuild_diffraction_inputs(
            reload_plan.occ,
            p_vals,
            weights,
            reload_plan.av,
            reload_plan.cv,
            force=True,
            trigger_update=True,
        )
        a_var.set(av)
        c_var.set(cv)
        simulation_runtime_state.last_simulation_signature = None
        progress_label.config(text=f"Loaded CIF: {Path(cif_file).name}")
    except Exception as exc:
        _reset_structure_model_control_vars(
            snapshot.current_occ_values,
            snapshot.current_atom_site_values,
        )
        gui_structure_model.restore_primary_cif_reload_snapshot(
            structure_model_state,
            snapshot,
            occ_vars=occ_vars,
            atom_site_fract_vars=atom_site_fract_vars,
        )
        _reset_atom_site_override_cache()
        _sync_structure_model_aliases()
        a_var.set(float(old_slider_a))
        c_var.set(float(old_slider_c))

        cif_file_var.set(snapshot.cif_file)
        progress_label.config(text=f"Failed to load CIF: {exc}")


def _browse_primary_cif():
    """Open a file picker and apply the selected primary CIF path."""
    gui_structure_model.browse_primary_cif_with_dialog(
        current_cif_path=cif_file,
        file_dialog_dir=get_dir("file_dialog_dir"),
        askopenfilename=filedialog.askopenfilename,
        set_cif_path_text=cif_file_var.set,
        apply_primary_cif_path=_apply_primary_cif_path,
    )


def _apply_primary_cif_from_entry(_event=None):
    """Apply the CIF path currently entered in the CIF control."""

    _apply_primary_cif_path(cif_file_var.get())


def _build_diffuse_ht_request():
    """Collect the active algebraic HT request from GUI state."""

    return gui_structure_model.build_diffuse_ht_request(
        structure_model_state,
        atom_site_override_state,
        p_values=[float(p0_var.get()), float(p1_var.get()), float(p2_var.get())],
        w_values=[float(w0_var.get()), float(w1_var.get()), float(w2_var.get())],
        a_lattice=float(a_var.get()),
        c_lattice=float(c_var.get()),
        lambda_angstrom=float(lambda_),
        mx=int(mx),
        two_theta_range=two_theta_range,
        finite_stack=bool(finite_stack_var.get()),
        stack_layers=int(max(1, stack_layers_var.get())),
        phase_delta_expression=_current_phase_delta_expression(),
        phi_l_divisor=_current_phi_l_divisor(),
        tcl_error_types=(tk.TclError,),
    )


def _open_diffuse_cif_toggle():
    """Open algebraic HT diffuse viewer using the active simulation inputs."""
    gui_structure_model.open_diffuse_ht_view_with_status(
        build_request=_build_diffuse_ht_request,
        open_view=(
            lambda request: open_diffuse_cif_toggle_algebraic(
                cif_path=request.active_cif,
                occ=request.occ,
                p_values=request.p_values,
                w_values=request.w_values,
                a_lattice=request.a_lattice,
                c_lattice=request.c_lattice,
                lambda_angstrom=request.lambda_angstrom,
                mx=request.mx,
                two_theta_max=request.two_theta_max,
                finite_stack=request.finite_stack,
                stack_layers=request.stack_layers,
                iodine_z=request.iodine_z,
                phase_delta_expression=request.phase_delta_expression,
                phi_l_divisor=request.phi_l_divisor,
            )
        ),
        set_status_text=lambda text: progress_label.config(text=text),
        tcl_error_types=(tk.TclError,),
    )


def _export_diffuse_ht_txt():
    """Export algebraic HT values to a fixed-width text table."""
    gui_structure_model.export_diffuse_ht_txt_with_dialog(
        build_request=_build_diffuse_ht_request,
        get_download_dir=lambda: get_dir("downloads"),
        asksaveasfilename=filedialog.asksaveasfilename,
        export_table=(
            lambda save_path, request: export_algebraic_ht_txt(
                output_path=save_path,
                cif_path=request.active_cif,
                occ=request.occ,
                p_values=request.p_values,
                w_values=request.w_values,
                a_lattice=request.a_lattice,
                c_lattice=request.c_lattice,
                lambda_angstrom=request.lambda_angstrom,
                mx=request.mx,
                two_theta_max=request.two_theta_max,
                finite_stack=request.finite_stack,
                stack_layers=request.stack_layers,
                iodine_z=request.iodine_z,
                phase_delta_expression=request.phase_delta_expression,
                phi_l_divisor=request.phi_l_divisor,
            )
        ),
        set_status_text=lambda text: progress_label.config(text=text),
        tcl_error_types=(tk.TclError,),
    )


_rebuild_occupancy_controls()
_rebuild_atom_site_fractional_controls()

gui_views.create_primary_cif_controls(
    parent=app_shell_view_state.right_col,
    view_state=primary_cif_controls_view_state,
    cif_path_text=str(cif_file),
    on_apply_from_entry=_apply_primary_cif_from_entry,
    on_browse_primary_cif=_browse_primary_cif,
    on_open_diffuse_ht=_open_diffuse_cif_toggle,
    on_export_diffuse_ht=_export_diffuse_ht_txt,
)
cif_file_var = primary_cif_controls_view_state.cif_file_var
if cif_file_var is None:
    raise RuntimeError("Primary CIF controls did not create the path variable.")


def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):
    """Entry point for running the GUI application.

    Parameters
    ----------
    write_excel_flag : bool or None, optional
        When ``True`` the initial intensities are written to an Excel
        file in the configured downloads directory.  When ``None`` the
        value from the instrument configuration file is used.
    startup_mode : {"prompt", "simulation", "calibrant"}, optional
        Startup behavior. ``prompt`` shows a launcher GUI asking which mode
        to run, ``simulation`` starts this GUI directly, and ``calibrant``
        launches the hBN calibrant fitter.
    calibrant_bundle : str or None, optional
        Optional NPZ bundle path to preload when launching calibrant mode.
    """

    global write_excel
    if write_excel_flag is not None:
        write_excel = write_excel_flag

    if startup_mode not in {"prompt", "simulation", "calibrant"}:
        raise ValueError(
            "startup_mode must be one of: prompt, simulation, calibrant"
        )

    resolved_mode = startup_mode
    if resolved_mode == "prompt":
        resolved_mode = gui_bootstrap.choose_startup_mode_dialog(root)

    if resolved_mode is None:
        _shutdown_gui()
        return

    if resolved_mode == "calibrant":
        _shutdown_gui()
        gui_bootstrap.launch_calibrant_gui(bundle=calibrant_bundle)
        return

    try:
        root.deiconify()
    except tk.TclError:
        pass

    params_file_path = get_path("parameters_file")
    profile_loaded = False
    if os.path.exists(params_file_path):
        load_parameters(
            params_file_path,
            theta_initial_var,
            cor_angle_var,
            gamma_var,
            Gamma_var,
            chi_var,
            zs_var,
            zb_var,
            debye_x_var,
            debye_y_var,
            corto_detector_var,
            sigma_mosaic_var,
            gamma_mosaic_var,
            eta_var,
            a_var,
            c_var,
            center_x_var,
            center_y_var,
            resolution_var,
            custom_samples_var,
            bandwidth_percent_var=bandwidth_percent_var,
            optics_mode_var=optics_mode_var,
            phase_delta_expr_var=phase_delta_expr_var,
            phi_l_divisor_var=phi_l_divisor_var,
            sf_prune_bias_var=sf_prune_bias_var,
            solve_q_steps_var=solve_q_steps_var,
            solve_q_rel_tol_var=solve_q_rel_tol_var,
            solve_q_mode_var=solve_q_mode_var,
        )
        if finite_stack_controls_view_state.phase_delta_entry_var is not None:
            gui_views.set_finite_stack_phase_delta_entry_text(
                finite_stack_controls_view_state,
                _current_phase_delta_expression(),
            )
        if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
            gui_views.set_finite_stack_phi_l_divisor_entry_text(
                finite_stack_controls_view_state,
                gui_controllers.format_finite_stack_phi_l_divisor(
                    _current_phi_l_divisor()
                ),
            )
        ensure_valid_resolution_choice()
        profile_loaded = True
    else:
        ensure_valid_resolution_choice()

    sample_mode = resolution_var.get()
    sample_count = int(max(1, simulation_runtime_state.num_samples))
    cif_summary = Path(cif_file).name
    if cif_file2:
        cif_summary = f"{cif_summary}, {Path(cif_file2).name}"
    print(
        "Startup ready: "
        f"profile={'loaded' if profile_loaded else 'defaults'}; "
        f"sampling={sample_mode} ({sample_count} samples); "
        f"sf_prune={_current_sf_prune_bias():+.2f}; "
        f"q_mode={_normalize_solve_q_mode_label(solve_q_mode_var.get())}; "
        f"q_steps={_current_solve_q_steps()}; "
        f"q_tol={_current_solve_q_rel_tol():.2e}; "
        f"optics={_normalize_optics_mode_label(optics_mode_var.get())}; "
        f"cif={cif_summary}"
    )

    def _run_initial_startup_work():
        try:
            progress_label.config(text="Initializing simulation...")
            export_initial_excel()
            update_mosaic_cache()
            do_update()
        except Exception as exc:
            progress_label.config(text=f"Startup initialization failed: {exc}")
            try:
                import traceback

                traceback.print_exc()
            except Exception:
                pass
        else:
            progress_label.config(text="Simulation ready.")

    # Let Tk paint the windows first, then run the expensive initial update.
    root.after_idle(_run_initial_startup_work)
    root.mainloop()


if __name__ == "__main__":
    cli_argv = list(sys.argv[1:])
    early_mode = os.environ.get("RA_SIM_EARLY_STARTUP_MODE")
    args = gui_bootstrap.parse_launch_args(cli_argv)

    try:
        override_flag = False if args.no_excel else None
        mode = gui_bootstrap.resolve_startup_mode(
            args.command,
            early_mode=early_mode,
        )
        main(
            write_excel_flag=override_flag,
            startup_mode=mode,
            calibrant_bundle=args.bundle,
        )
    except Exception as exc:
        print("Unhandled exception during startup:", exc)
        import traceback
        traceback.print_exc()




