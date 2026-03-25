#!/usr/bin/env python3

"""Main application entry point for running the Tk based GUI."""

import math
import os
write_excel = False

import re
import argparse
import tempfile
from collections import defaultdict, namedtuple
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Sequence
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import pyFAI
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from scipy.optimize import differential_evolution, least_squares
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from skimage.metrics import mean_squared_error
import spglib
import OSC_Reader
from OSC_Reader import read_osc
import numba
import sys
import pandas as pd
import Dans_Diffraction as dif
import CifFile

from ra_sim.utils.stacking_fault import (
    ht_Iinf_dict,
    ht_dict_to_arrays,
    ht_dict_to_qr_dict,
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
    load_and_format_reference_profiles,
    save_all_parameters,
    load_parameters,
    load_gui_state_file,
    save_gui_state_file,
)
from ra_sim.fitting.optimization import (
    _prepare_reflection_subset,
    build_measured_dict,
    fit_geometry_parameters,
    fit_mosaic_widths_separable,
    simulate_and_compare_hkl,
)
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import (
    hit_tables_to_max_positions,
    OPTICS_MODE_EXACT,
    OPTICS_MODE_FAST,
    process_peaks_parallel_safe as process_peaks_parallel,
    process_qr_rods_parallel_safe as process_qr_rods_parallel,
)
from ra_sim.simulation.intersection_analysis import (
    BeamSamples as IntersectionBeamSamples,
    IntersectionGeometry,
    MosaicParams as IntersectionMosaicParams,
    analyze_reflection_intersection,
    plot_intersection_analysis,
)
from ra_sim.simulation.diffraction_debug import (
    process_peaks_parallel_debug,
    process_qr_rods_parallel_debug,
    dump_debug_log,
)
from ra_sim.simulation.simulation import simulate_diffraction
from ra_sim.gui.sliders import create_slider
from ra_sim.debug_utils import debug_print, is_debug_enabled
from ra_sim.gui.geometry_overlay import (
    build_geometry_fit_overlay_records,
    compute_geometry_overlay_frame_diagnostics,
    normalize_initial_geometry_pairs_display,
)
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
else:
    print("Debug mode off (set RA_SIM_DEBUG=1 for extra output)")

# Keep backend-orientation debug controls disabled and force identity transforms.
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
    get_instrument_config,
)


def _ensure_triplet(values, fallback):
    """Return a 3-element list combining *values* with *fallback*."""

    if not isinstance(values, (list, tuple)):
        return list(fallback)
    merged = list(fallback)
    for idx, val in enumerate(values[:3]):
        merged[idx] = val
    return merged


instrument_config = get_instrument_config().get("instrument", {})
detector_config = instrument_config.get("detector", {})
geometry_config = instrument_config.get("geometry_defaults", {})
beam_config = instrument_config.get("beam", {})
sample_config = instrument_config.get("sample_orientation", {})
debye_config = instrument_config.get("debye_waller", {})
occupancy_config = instrument_config.get("occupancies", {})
hendricks_config = instrument_config.get("hendricks_teller", {})
output_config = instrument_config.get("output", {})
fit_config = instrument_config.get("fit", {})

file_path = get_path_first("simulation_dark_osc_file", "dark_image")
BI = read_osc(file_path)  # Dark (background) image

osc_files = get_path_first("simulation_background_osc_files", "osc_files")
if isinstance(osc_files, str):
    osc_files = [osc_files]
background_images = [read_osc(path) for path in osc_files]
if not background_images:
    raise ValueError(
        "No oscillation images configured in simulation_background_osc_files/osc_files"
    )

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

# Preserve native-orientation copies for fitting/analysis; display variants may
# be rotated for visualization.
background_images_native = [np.array(img) for img in background_images]
background_images_display = [
    np.rot90(img, DISPLAY_ROTATE_K) for img in background_images_native
]

# Parse geometry
poni_file_path = get_path("geometry_poni")
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
    center_text = ""
    if hinted_center_row is not None and hinted_center_col is not None:
        center_text = (
            f", center(row={hinted_center_row:.3f}, col={hinted_center_col:.3f})"
        )
    print(
        "Initialized detector tilt from last hBN fit profile "
        f"(sim γ={Gamma_initial:.4f} deg, sim Γ={gamma_initial:.4f} deg{center_text})."
    )

image_size = detector_config.get("image_size", 3000)
pixel_size_m = float(detector_config.get("pixel_size_m", DEFAULT_PIXEL_SIZE_M))
resolution_sample_counts = {
    "Low": 25,
    "Medium": 250,
    "High": 500,
}
num_samples = resolution_sample_counts["Low"]
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

bandwidth = beam_config.get("bandwidth_percent", 0.7) / 100

# NOTE: We define the default occupancy for each site:
occ = _ensure_triplet(occupancy_config.get("default"), [1.0, 1.0, 1.0])

# When enabled, additional fractional reflections ("rods")
# are injected between integer L values.
include_rods_flag = hendricks_config.get("include_rods", False)

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

# pull the raw text
a_text = blk.get("_cell_length_a")
b_text = blk.get("_cell_length_b")
c_text = blk.get("_cell_length_c")

# strip the '(uncertainty)' and cast
def parse_cif_num(txt):
    # match leading numeric part, e.g. '4.14070' out of '4.14070(3)'
    if txt is None:
        raise ValueError("Missing CIF numeric value")
    if isinstance(txt, (int, float)):
        return float(txt)
    m = re.match(r"[-+0-9\.Ee]+", str(txt).strip())
    if not m:
        raise ValueError(f"Can't parse '{txt}' as a number")
    return float(m.group(0))

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
finite_stack_default = bool(hendricks_config.get("finite_stack", True))
stack_layers_default = int(
    max(1, float(hendricks_config.get("stack_layers", 50)))
)

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
    'center_x': center_default[0],
    'center_y': center_default[1],
    'sampling_resolution': 'Low',
    'finite_stack': finite_stack_default,
    'stack_layers': stack_layers_default,
    'optics_mode': 'fast',
}

# ---------------------------------------------------------------------------
# Replace the old miller_generator call with the new Hendricks–Teller helper.
# ---------------------------------------------------------------------------
def build_ht_cache(p_val, occ_vals, c_axis, finite_stack_flag, stack_layers_count):
    layers = int(max(1, stack_layers_count))
    curves_raw = ht_Iinf_dict(
        cif_path=cif_file,
        mx=mx,
        occ=occ_vals,
        p=p_val,
        L_step=0.01,
        two_theta_max=two_theta_range[1],
        lambda_=lambda_,
        c_lattice=c_axis,
        finite_stack=finite_stack_flag,
        stack_layers=layers,
    )
    curves = {}
    for hk, data in curves_raw.items():
        L_vals = np.asarray(data["L"], dtype=float)
        I_vals = np.asarray(data["I"], dtype=float)
        keep = np.isfinite(L_vals) & np.isfinite(I_vals) & (L_vals > 0.0)
        if not np.any(keep):
            continue
        curves[hk] = {"L": L_vals[keep], "I": I_vals[keep]}
    qr = ht_dict_to_qr_dict(curves)
    arrays = ht_dict_to_arrays(curves)
    return {
        "p": p_val,
        "occ": tuple(occ_vals),
        "ht": curves,
        "qr": qr,
        "arrays": arrays,
        "two_theta_max": two_theta_range[1],
        "c": float(c_axis),
        "finite_stack": bool(finite_stack_flag),
        "stack_layers": layers,
    }

# Precompute curves for the three p values
default_c_axis = float(defaults['c'])
ht_cache_multi = {
    "p0": build_ht_cache(
        defaults['p0'],
        occ,
        default_c_axis,
        defaults['finite_stack'],
        defaults['stack_layers'],
    ),
    "p1": build_ht_cache(
        defaults['p1'],
        occ,
        default_c_axis,
        defaults['finite_stack'],
        defaults['stack_layers'],
    ),
    "p2": build_ht_cache(
        defaults['p2'],
        occ,
        default_c_axis,
        defaults['finite_stack'],
        defaults['stack_layers'],
    ),
}

def combine_ht_dicts(caches, weights):
    import numpy as np
    out = {}
    for cache, w in zip(caches, weights):
        ht_curves = cache["ht"]
        for hk, data in ht_curves.items():
            if hk not in out:
                out[hk] = {
                    "L": data["L"].copy(),
                    "I": w * data["I"].copy(),
                }
            else:
                entry = out[hk]
                if entry["L"].shape != data["L"].shape or not np.allclose(entry["L"], data["L"]):
                    union_L = np.union1d(entry["L"], data["L"])
                    entry_I = np.interp(union_L, entry["L"], entry["I"], left=0.0, right=0.0)
                    add_I = w * np.interp(union_L, data["L"], data["I"], left=0.0, right=0.0)
                    entry["L"] = union_L
                    entry["I"] = entry_I + add_I
                else:
                    entry["I"] += w * data["I"]
    return out

weights_init = np.array([defaults['w0'], defaults['w1'], defaults['w2']], dtype=float)
weights_init /= weights_init.sum() if weights_init.sum() else 1.0
combined_ht = combine_ht_dicts(
    [ht_cache_multi['p0'], ht_cache_multi['p1'], ht_cache_multi['p2']],
    weights_init,
)
combined_qr = ht_dict_to_qr_dict(combined_ht)
miller1, intens1, degeneracy1, details1 = ht_dict_to_arrays(combined_ht)
ht_curves_cache = {
    "curves": combined_ht,
    "qr_curves": combined_qr,
    "arrays": (miller1, intens1, degeneracy1, details1),
    "c": default_c_axis,
    "finite_stack": defaults['finite_stack'],
    "stack_layers": defaults['stack_layers'],
}
_last_occ_for_ht = list(occ)
_last_p_triplet = [defaults['p0'], defaults['p1'], defaults['p2']]
_last_weights = list(weights_init)
_last_c_for_ht = default_c_axis
_last_finite_stack = bool(defaults['finite_stack'])
_last_stack_layers = int(max(1, defaults['stack_layers']))
# ---- convert the dict → arrays compatible with the downstream code ----
debug_print("miller1 shape:", miller1.shape, "intens1 shape:", intens1.shape)
debug_print("miller1 sample:", miller1[:5])

if DEBUG_ENABLED:
    from ra_sim.debug_utils import check_ht_arrays
    check_ht_arrays(miller1, intens1)
    # Manual inspection snippet recommended in the README
    debug_print('miller1 dtype:', miller1.dtype, 'shape:', miller1.shape)
    debug_print('L range:', miller1[:, 2].min(), miller1[:, 2].max())
    debug_print('intens1 dtype:', intens1.dtype, 'min:', intens1.min(), 'max:', intens1.max())
    debug_print('miller1 contiguous:', miller1.flags['C_CONTIGUOUS'])
    debug_print('intens1 contiguous:', intens1.flags['C_CONTIGUOUS'])

has_second_cif = bool(cif_file2)
if has_second_cif:
    miller2, intens2, degeneracy2, details2 = miller_generator(
        mx,
        cif_file2,
        occ,
        lambda_,
        energy,
        intensity_threshold,
        two_theta_range,
    )
    if include_rods_flag:
        miller2, intens2 = inject_fractional_reflections(miller2, intens2, mx)
    union_set = {tuple(hkl) for hkl in miller1} | {tuple(hkl) for hkl in miller2}
    miller = np.array(sorted(union_set), dtype=float)
    debug_print("combined miller count:", miller.shape[0])

    int1_dict = {tuple(h): i for h, i in zip(miller1, intens1)}
    int2_dict = {tuple(h): i for h, i in zip(miller2, intens2)}
    deg_dict1 = {tuple(h): d for h, d in zip(miller1, degeneracy1)}
    deg_dict2 = {tuple(h): d for h, d in zip(miller2, degeneracy2)}
    details_dict1 = {tuple(miller1[i]): details1[i] for i in range(len(miller1))}
    details_dict2 = {tuple(miller2[i]): details2[i] for i in range(len(miller2))}

    intensities_cif1 = np.array([int1_dict.get(tuple(h), 0.0) for h in miller])
    intensities_cif2 = np.array([int2_dict.get(tuple(h), 0.0) for h in miller])
    degeneracy = np.array(
        [deg_dict1.get(tuple(h), 0) + deg_dict2.get(tuple(h), 0) for h in miller],
        dtype=np.int32,
    )
    details = [
        details_dict1.get(tuple(h), []) + details_dict2.get(tuple(h), [])
        for h in miller
    ]

    weight1 = 0.5
    weight2 = 0.5
    intensities = weight1 * intensities_cif1 + weight2 * intensities_cif2
    max_I = intensities.max() if intensities.size else 0.0
    if max_I > 0:
        intensities = intensities * (100.0 / max_I)
    miller1_sim = miller1
    intensities1_sim = intens1
    miller2_sim = miller2
    intensities2_sim = intens2
else:
    miller = miller1
    intensities_cif1 = intens1
    intensities_cif2 = np.zeros_like(intensities_cif1)
    degeneracy = degeneracy1
    details = details1
    weight1 = 1.0
    weight2 = 0.0
    intensities = intensities_cif1.copy()
    miller1_sim = miller1
    intensities1_sim = intens1
    miller2_sim = np.empty((0,3), dtype=np.int32)
    intensities2_sim = np.empty((0,), dtype=np.float64)
    debug_print("single CIF miller count:", miller.shape[0])

# Save simulation data for later use
SIM_MILLER1 = miller1_sim
SIM_INTENS1 = intensities1_sim
SIM_MILLER2 = miller2_sim
SIM_INTENS2 = intensities2_sim

# Build summary and details dataframes using the helper.
df_summary, df_details = build_intensity_dataframes(
    miller, intensities, degeneracy, details
)

def export_initial_excel():
    """Write the initial intensity tables to Excel when enabled."""

    if not write_excel:
        return

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

# Beam center (for plotting and limits)
row_center = int(center_default[0])
col_center = int(center_default[1])

current_background_image = background_images_native[0]
current_background_display = background_images_display[0]
current_background_index = 0
background_visible = True
background_backend_rotation_k = 0
background_backend_flip_x = False
background_backend_flip_y = False


def _get_current_background_native() -> np.ndarray:
    """Return the unrotated background image corresponding to the current index."""

    if 0 <= current_background_index < len(background_images_native):
        return background_images_native[current_background_index]
    return current_background_image


def _get_current_background_display() -> np.ndarray:
    """Return the rotated background image used for GUI display."""

    if 0 <= current_background_index < len(background_images_display):
        return background_images_display[current_background_index]
    return np.rot90(_get_current_background_native(), DISPLAY_ROTATE_K)


def _apply_background_backend_orientation(image: np.ndarray | None) -> np.ndarray | None:
    """Return identity backend orientation for background arrays."""

    if image is None:
        return None
    return np.asarray(image)


def _get_current_background_backend() -> np.ndarray | None:
    """Return the background array used for backend comparisons (debug)."""

    return _apply_background_backend_orientation(_get_current_background_native())


def _rotate_point_for_display(col: float, row: float, shape: tuple[int, ...], k: int):
    """Rotate a single (col, row) pair by ``k`` using the same rule as ``np.rot90``.

    The transformation mirrors what ``np.rot90`` does to the underlying image so
    point overlays stay aligned with whichever orientation we render.
    """

    height, width = shape[:2]
    col_new, row_new = float(col), float(row)

    # Apply the 90° rotation step-by-step to mirror ``np.rot90``'s behavior for
    # any integer ``k`` (positive for CCW, negative for CW).
    for _ in range(k % 4):
        row_new, col_new, height, width = width - 1 - col_new, row_new, width, height

    return col_new, row_new


def _rotate_measured_peaks_for_display(measured, rotated_shape):
    """Rotate measured-peak coordinates to match the displayed background."""

    if measured is None:
        return []

    rotated_entries = []
    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _rotate_point_for_display(
                    updated["x"], updated["y"], rotated_shape, DISPLAY_ROTATE_K
                )
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _rotate_point_for_display(
                    updated["x_pix"], updated["y_pix"], rotated_shape, DISPLAY_ROTATE_K
                )
            rotated_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _rotate_point_for_display(
                seq[3], seq[4], rotated_shape, DISPLAY_ROTATE_K
            )
            rotated_entries.append(type(entry)(seq))
        else:
            rotated_entries.append(entry)

    return rotated_entries


def _unrotate_display_peaks(measured, rotated_shape, *, k=None):
    """Undo a display rotation on peak coordinates.

    Pass ``k`` to match the rotation applied for display (e.g. ``DISPLAY_ROTATE_K``
    or ``SIM_DISPLAY_ROTATE_K``), so the returned points land back in that image's
    native orientation.
    """

    if measured is None:
        return []

    if k is None:
        k = DISPLAY_ROTATE_K
    inv_k = -int(k)

    unrotated = []
    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _rotate_point_for_display(
                    updated["x"], updated["y"], rotated_shape, inv_k
                )
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _rotate_point_for_display(
                    updated["x_pix"], updated["y_pix"], rotated_shape, inv_k
                )
            unrotated.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _rotate_point_for_display(
                seq[3], seq[4], rotated_shape, inv_k
            )
            unrotated.append(type(entry)(seq))
        else:
            unrotated.append(entry)

    return unrotated


def _apply_indexing_mode_to_entries(
    measured,
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
):
    """Swap x/y coordinates when using alternate indexing modes."""

    if measured is None:
        return []

    _ = shape  # retained for signature parity with orientation helpers

    mode = (indexing_mode or "xy").lower()
    if mode == "xy":
        return list(measured)

    swapped_entries = []

    def _swap_pair(col: float, row: float) -> tuple[float, float]:
        return float(row), float(col)

    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _swap_pair(updated["x"], updated["y"])
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _swap_pair(
                    updated["x_pix"], updated["y_pix"]
                )
            swapped_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _swap_pair(seq[3], seq[4])
            swapped_entries.append(type(entry)(seq))
        else:
            swapped_entries.append(entry)

    return swapped_entries


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

    if measured is None:
        return []

    indexed = _apply_indexing_mode_to_entries(
        measured, rotated_shape, indexing_mode=indexing_mode
    )

    k_mod = int(k) % 4
    if k_mod == 0 and not flip_x and not flip_y:
        return list(indexed)

    # Points are already expressed in the requested indexing mode, so avoid
    # double-swapping by keeping further transforms in XY and adjusting the
    # shape to match that frame.
    mode = (indexing_mode or "xy").lower()
    oriented_shape = rotated_shape if mode == "xy" else (rotated_shape[1], rotated_shape[0])

    def _apply_pair(x_val: float, y_val: float) -> tuple[float, float]:
        return _transform_points_orientation(
            [(x_val, y_val)],
            oriented_shape,
            indexing_mode="xy",
            k=k_mod,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_order=flip_order,
        )[0]

    oriented_entries = []
    for entry in indexed:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _apply_pair(updated["x"], updated["y"])
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _apply_pair(
                    updated["x_pix"], updated["y_pix"]
                )
            oriented_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _apply_pair(seq[3], seq[4])
            oriented_entries.append(type(entry)(seq))
        else:
            oriented_entries.append(entry)

    return oriented_entries


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

    if image is None:
        return None

    oriented = np.asarray(image)
    mode = (indexing_mode or "xy").lower()
    if mode == "yx":
        oriented = np.swapaxes(oriented, 0, 1)
    order = (flip_order or "yx").lower()
    if order == "xy":
        if flip_x:
            oriented = np.flip(oriented, axis=1)
        if flip_y:
            oriented = np.flip(oriented, axis=0)
    else:
        if flip_y:
            oriented = np.flip(oriented, axis=0)
        if flip_x:
            oriented = np.flip(oriented, axis=1)
    k_mod = int(k) % 4
    if k_mod:
        oriented = np.rot90(oriented, k_mod)
    return oriented


def _native_sim_to_display_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Rotate native simulation coordinates into the displayed frame."""

    return _rotate_point_for_display(col, row, image_shape, SIM_DISPLAY_ROTATE_K)


def _display_to_native_sim_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Map displayed simulation coordinates back into native simulation frame."""

    return _rotate_point_for_display(col, row, image_shape, -SIM_DISPLAY_ROTATE_K)


def _display_sim_to_native_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Rotate displayed simulation coordinates back to native simulation frame."""

    inv_k = (-SIM_DISPLAY_ROTATE_K) % 4
    return _rotate_point_for_display(col, row, image_shape, inv_k)


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

    base_height, base_width = shape
    mode = (indexing_mode or "xy").lower()
    if mode == "yx":
        height, width = base_width, base_height
    else:
        height, width = base_height, base_width
    transformed: list[tuple[float, float]] = []

    order = (flip_order or "yx").lower()

    def _flip_xy(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_x:
            col_t = width - 1.0 - col_t
        if flip_y:
            row_t = height - 1.0 - row_t
        return col_t, row_t

    def _flip_yx(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_y:
            row_t = height - 1.0 - row_t
        if flip_x:
            col_t = width - 1.0 - col_t
        return col_t, row_t

    flipper = _flip_xy if order == "xy" else _flip_yx

    for col, row in points:
        col_t, row_t = float(col), float(row)
        if mode == "yx":
            col_t, row_t = row_t, col_t
        col_t, row_t = flipper(col_t, row_t)
        col_t, row_t = _rotate_point_for_display(col_t, row_t, (height, width), k)
        transformed.append((col_t, row_t))

    return transformed


def _best_orientation_alignment(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
):
    """Search over 90° rotations and axis flips to minimize RMS distance."""

    if not sim_coords or not meas_coords or len(sim_coords) != len(meas_coords):
        return None

    def _describe(
        k: int, flip_x: bool, flip_y: bool, flip_order: str, indexing_mode: str
    ) -> str:
        parts: list[str] = []
        if k % 4:
            parts.append(f"rot{(k % 4) * 90}° CCW")
        if flip_x:
            parts.append("flip_x")
        if flip_y:
            parts.append("flip_y")
        parts.append(f"order={flip_order}")
        parts.append(f"indexing={indexing_mode}")
        return " + ".join(parts)

    best = None
    for indexing_mode in ("xy", "yx"):
        for flip_order in ("yx", "xy"):
            for k in range(4):
                for flip_x in (False, True):
                    for flip_y in (False, True):
                        transformed = _transform_points_orientation(
                            meas_coords,
                            shape,
                            indexing_mode=indexing_mode,
                            k=k,
                            flip_x=flip_x,
                            flip_y=flip_y,
                            flip_order=flip_order,
                        )
                        deltas = [
                            math.hypot(sx - mx, sy - my)
                            for (sx, sy), (mx, my) in zip(sim_coords, transformed)
                        ]
                        if not deltas:
                            continue
                        rms = math.sqrt(sum(d * d for d in deltas) / len(deltas))
                        mean = sum(deltas) / len(deltas)
                        candidate = {
                            "k": k,
                            "flip_x": flip_x,
                            "flip_y": flip_y,
                            "flip_order": flip_order,
                            "indexing_mode": indexing_mode,
                            "rms": rms,
                            "mean": mean,
                            "label": _describe(
                                k, flip_x, flip_y, flip_order, indexing_mode
                            ),
                        }
                        if best is None or candidate["rms"] < best["rms"]:
                            best = candidate

    return best


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

    transformed = _transform_points_orientation(
        meas_coords,
        shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )
    deltas = [
        math.hypot(sx - mx, sy - my)
        for (sx, sy), (mx, my) in zip(sim_coords, transformed)
    ]
    if not deltas:
        return {
            "rms": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    arr = np.asarray(deltas, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def _select_fit_orientation(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    cfg: dict[str, object] | None = None,
):
    """Choose a measured-peak orientation transform that best aligns to simulation."""

    identity = {
        "k": 0,
        "flip_x": False,
        "flip_y": False,
        "flip_order": "yx",
        "indexing_mode": "xy",
        "label": "identity",
    }
    config = cfg if isinstance(cfg, dict) else {}
    enabled = bool(config.get("enabled", True))
    min_improvement = max(0.0, float(config.get("min_improvement_px", 0.25)))
    max_rms = float(config.get("max_rms_px", np.inf))

    diagnostics = {
        "enabled": bool(enabled),
        "pairs": int(min(len(sim_coords), len(meas_coords))),
        "identity_rms_px": float("nan"),
        "best_rms_px": float("nan"),
        "best_label": "identity",
        "chosen_label": "identity",
        "improvement_px": float("nan"),
        "reason": "identity_fallback",
    }

    if not sim_coords or not meas_coords or len(sim_coords) != len(meas_coords):
        diagnostics["reason"] = "insufficient_pairs"
        return identity, diagnostics

    identity_metrics = _orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode="xy",
        k=0,
        flip_x=False,
        flip_y=False,
        flip_order="yx",
    )
    diagnostics["identity_rms_px"] = float(identity_metrics["rms"])

    best = _best_orientation_alignment(sim_coords, meas_coords, shape)
    if best is None:
        diagnostics["reason"] = "no_candidate"
        return identity, diagnostics

    best_metrics = _orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode=str(best.get("indexing_mode", "xy")),
        k=int(best.get("k", 0)),
        flip_x=bool(best.get("flip_x", False)),
        flip_y=bool(best.get("flip_y", False)),
        flip_order=str(best.get("flip_order", "yx")),
    )
    best_rms = float(best_metrics["rms"])
    identity_rms = float(identity_metrics["rms"])
    improvement = identity_rms - best_rms

    diagnostics.update(
        {
            "best_rms_px": best_rms,
            "best_label": str(best.get("label", "candidate")),
            "improvement_px": float(improvement),
        }
    )

    if not enabled:
        diagnostics["reason"] = "disabled_by_config"
        return identity, diagnostics

    if not np.isfinite(best_rms):
        diagnostics["reason"] = "best_rms_not_finite"
        return identity, diagnostics

    if np.isfinite(max_rms) and best_rms > max_rms:
        diagnostics["reason"] = "best_rms_above_threshold"
        return identity, diagnostics

    if not np.isfinite(improvement) or improvement < min_improvement:
        diagnostics["reason"] = "insufficient_improvement"
        return identity, diagnostics

    chosen = {
        "k": int(best.get("k", 0)),
        "flip_x": bool(best.get("flip_x", False)),
        "flip_y": bool(best.get("flip_y", False)),
        "flip_order": str(best.get("flip_order", "yx")),
        "indexing_mode": str(best.get("indexing_mode", "xy")),
        "label": str(best.get("label", "candidate")),
    }
    diagnostics["chosen_label"] = str(chosen["label"])
    diagnostics["reason"] = "selected_best"
    return chosen, diagnostics


def _aggregate_match_centers(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    sim_millers: list[tuple[int, int, int]],
    meas_millers: list[tuple[int, int, int]],
):
    """Collapse matched peaks by HKL and return centroid pairs."""

    aggregated: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}
    for hkl_sim, hkl_meas, sim_xy, meas_xy in zip(
        sim_millers, meas_millers, sim_coords, meas_coords
    ):
        hkl_key = tuple(int(v) for v in (hkl_sim or hkl_meas))
        entry = aggregated.setdefault(hkl_key, {"sim": [], "meas": []})
        entry["sim"].append(sim_xy)
        entry["meas"].append(meas_xy)

    agg_sim_coords: list[tuple[float, float]] = []
    agg_meas_coords: list[tuple[float, float]] = []
    agg_millers: list[tuple[int, int, int]] = []

    for hkl_key in sorted(aggregated):
        sim_arr = np.array(aggregated[hkl_key]["sim"], dtype=float)
        meas_arr = np.array(aggregated[hkl_key]["meas"], dtype=float)

        sim_center = (float(sim_arr[:, 0].mean()), float(sim_arr[:, 1].mean()))
        meas_center = (float(meas_arr[:, 0].mean()), float(meas_arr[:, 1].mean()))

        agg_sim_coords.append(sim_center)
        agg_meas_coords.append(meas_center)
        agg_millers.append(hkl_key)

    return agg_sim_coords, agg_meas_coords, agg_millers


measured_peaks_raw = np.load(get_path("measured_peaks"), allow_pickle=True)
measured_peaks = _rotate_measured_peaks_for_display(
    measured_peaks_raw,
    current_background_display.shape,
)

###############################################################################
#                                  TK SETUP
###############################################################################
root = tk.Tk()
root.title("Controls and Sliders")

fig_window = tk.Toplevel(root)
fig_window.title("Main Figure")
fig_frame = ttk.Frame(fig_window)
fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


def _shutdown_gui():
    """Close all application windows and end the Tk event loop."""

    for window in (fig_window, root):
        try:
            if window.winfo_exists():
                window.destroy()
        except tk.TclError:
            # Window is already gone or cannot be destroyed cleanly; proceed to
            # shut down the rest of the application.
            pass

    try:
        root.quit()
    except tk.TclError:
        pass


root.protocol("WM_DELETE_WINDOW", _shutdown_gui)
fig_window.protocol("WM_DELETE_WINDOW", _shutdown_gui)

canvas_frame = ttk.Frame(fig_frame)
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

RESPONSIVE_STATUS_BUSY_COLOR = "#c62828"
RESPONSIVE_STATUS_READY_COLOR = "#2e7d32"
RESPONSIVE_STATUS_BLOCK_SIZE = 24
_gui_busy_depth = 0
_gui_ready_after_id = None
responsiveness_indicator_block = None
responsiveness_indicator_container = None


def _raise_responsiveness_indicator_block() -> None:
    if responsiveness_indicator_block is None:
        return
    try:
        if (
            responsiveness_indicator_container is not None
            and responsiveness_indicator_block.winfo_manager() != "pack"
        ):
            responsiveness_indicator_block.pack(
                side=tk.LEFT,
                padx=6,
                pady=4,
            )
        responsiveness_indicator_block.lift()
    except tk.TclError:
        pass


def _set_responsiveness_indicator_color(color: str) -> None:
    if responsiveness_indicator_block is None:
        return
    try:
        responsiveness_indicator_block.configure(bg=str(color))
        _raise_responsiveness_indicator_block()
    except tk.TclError:
        pass


def _apply_responsiveness_indicator_ready() -> None:
    global _gui_ready_after_id

    _gui_ready_after_id = None
    if _gui_busy_depth == 0:
        _set_responsiveness_indicator_color(RESPONSIVE_STATUS_READY_COLOR)


def _mark_gui_busy() -> None:
    global _gui_busy_depth, _gui_ready_after_id

    _gui_busy_depth += 1
    if _gui_ready_after_id is not None:
        try:
            root.after_cancel(_gui_ready_after_id)
        except tk.TclError:
            pass
        _gui_ready_after_id = None
    _set_responsiveness_indicator_color(RESPONSIVE_STATUS_BUSY_COLOR)
    try:
        root.update_idletasks()
    except tk.TclError:
        pass


def _mark_gui_ready() -> None:
    global _gui_busy_depth, _gui_ready_after_id

    _gui_busy_depth = max(0, _gui_busy_depth - 1)
    if _gui_busy_depth != 0:
        return
    try:
        _gui_ready_after_id = root.after_idle(_apply_responsiveness_indicator_ready)
    except tk.TclError:
        _set_responsiveness_indicator_color(RESPONSIVE_STATUS_READY_COLOR)


def _run_with_responsiveness_indicator(callback, *args, **kwargs):
    _mark_gui_busy()
    try:
        return callback(*args, **kwargs)
    finally:
        _mark_gui_ready()


def _wrap_with_responsiveness_indicator(callback):
    @wraps(callback)
    def wrapped(*args, **kwargs):
        return _run_with_responsiveness_indicator(callback, *args, **kwargs)

    return wrapped


try:
    _gui_ready_after_id = root.after_idle(_apply_responsiveness_indicator_ready)
except tk.TclError:
    _gui_ready_after_id = None

display_controls_frame = ttk.Frame(fig_frame)
display_controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

fig, ax = plt.subplots(figsize=(8, 8))
canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

global_image_buffer = np.zeros((image_size, image_size), dtype=np.float64)
unscaled_image_global = None

# ── replace the original imshow call ────────────────────────────
image_display = ax.imshow(
    global_image_buffer,
    cmap=turbo_white0,
    alpha=0.5,
    zorder=1,
    origin='upper'
)


background_display = ax.imshow(
    current_background_display,
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
        "beam_x_array":       profile_cache["beam_x_array"],
        "beam_y_array":       profile_cache["beam_y_array"],
        "theta_array":        profile_cache["theta_array"],
        "phi_array":          profile_cache["phi_array"],
        "wavelength_array":   profile_cache["wavelength_array"],   #  <<< name fixed
        "sigma_mosaic_deg":   sigma_mosaic_var.get(),
        "gamma_mosaic_deg":   gamma_mosaic_var.get(),
        "eta":                eta_var.get(),
    }


def _parse_hkl_from_label(label: object) -> tuple[int, int, int] | None:
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    parts = text.split(",")
    if len(parts) != 3:
        return None
    try:
        return (
            int(round(float(parts[0]))),
            int(round(float(parts[1]))),
            int(round(float(parts[2]))),
        )
    except Exception:
        return None


def _normalize_hkl_key(value: object) -> tuple[int, int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return tuple(int(round(float(part))) for part in value)
        except Exception:
            return None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for ch in "()[]":
        text = text.replace(ch, "")
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        return None
    try:
        return tuple(int(round(float(part))) for part in parts)
    except Exception:
        return None


def _normalize_measured_peaks_for_geometry_fit(
    measured_peaks: Sequence[object],
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for entry in measured_peaks:
        if not isinstance(entry, dict):
            continue
        hkl = _parse_hkl_from_label(entry.get("label"))
        if hkl is None:
            continue
        try:
            x = float(entry.get("x"))
            y = float(entry.get("y"))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        normalized.append(
            {
                "hkl": hkl,
                "label": f"{hkl[0]},{hkl[1]},{hkl[2]}",
                "x": x,
                "y": y,
            }
        )
    return normalized


def _store_geometry_fit_snapshot(
    fitted_params: dict[str, object],
    measured_for_fit: Sequence[object],
) -> None:
    global last_geometry_fit_params, last_geometry_matched_peaks

    snapshot = dict(fitted_params)
    mosaic = dict(snapshot.get("mosaic_params", {}) or {})
    for key in (
        "beam_x_array",
        "beam_y_array",
        "theta_array",
        "phi_array",
        "wavelength_array",
        "wavelength_i_array",
    ):
        if key in mosaic:
            mosaic[key] = np.asarray(mosaic[key], dtype=np.float64).copy()
    snapshot["mosaic_params"] = mosaic

    if "center" in snapshot:
        try:
            ctr = snapshot["center"]
            snapshot["center"] = [float(ctr[0]), float(ctr[1])]
        except Exception:
            pass

    for key in (
        "a",
        "c",
        "lambda",
        "psi",
        "psi_z",
        "zs",
        "zb",
        "chi",
        "debye_x",
        "debye_y",
        "theta_initial",
        "cor_angle",
        "corto_detector",
        "gamma",
        "Gamma",
    ):
        if key in snapshot:
            try:
                snapshot[key] = float(snapshot[key])
            except Exception:
                pass

    last_geometry_fit_params = snapshot
    last_geometry_matched_peaks = _normalize_measured_peaks_for_geometry_fit(
        measured_for_fit
    )


def _copy_geometry_fit_state_value(value):
    """Deep-copy simple GUI fit state without importing ``copy``."""

    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _copy_geometry_fit_state_value(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_copy_geometry_fit_state_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_copy_geometry_fit_state_value(val) for val in value)
    return value


def _current_geometry_fit_ui_params() -> dict[str, object]:
    return {
        "zb": float(zb_var.get()),
        "zs": float(zs_var.get()),
        "theta_initial": float(theta_initial_var.get()),
        "psi_z": float(psi_z_var.get()),
        "chi": float(chi_var.get()),
        "cor_angle": float(cor_angle_var.get()),
        "gamma": float(gamma_var.get()),
        "Gamma": float(Gamma_var.get()),
        "corto_detector": float(corto_detector_var.get()),
        "a": float(a_var.get()),
        "c": float(c_var.get()),
        "center_x": float(center_x_var.get()),
        "center_y": float(center_y_var.get()),
        "center": [float(center_x_var.get()), float(center_y_var.get())],
    }


def _capture_geometry_fit_undo_state() -> dict[str, object]:
    return {
        "ui_params": _copy_geometry_fit_state_value(_current_geometry_fit_ui_params()),
        "profile_cache": _copy_geometry_fit_state_value(profile_cache),
        "last_geometry_fit_params": _copy_geometry_fit_state_value(
            last_geometry_fit_params
        ),
        "last_geometry_matched_peaks": _copy_geometry_fit_state_value(
            last_geometry_matched_peaks
        ),
        "overlay_state": _copy_geometry_fit_state_value(last_geometry_overlay_state),
    }


def _update_geometry_fit_undo_button_state() -> None:
    if undo_geometry_fit_button is None:
        return
    undo_geometry_fit_button.config(
        state=("normal" if geometry_fit_undo_stack else "disabled")
    )


def _push_geometry_fit_undo_state(state: dict[str, object] | None) -> None:
    global geometry_fit_undo_stack

    if not isinstance(state, dict):
        return
    geometry_fit_undo_stack.append(_copy_geometry_fit_state_value(state))
    geometry_fit_undo_stack = geometry_fit_undo_stack[-16:]
    _update_geometry_fit_undo_button_state()


def _restore_geometry_fit_undo_state(state: dict[str, object]) -> None:
    global profile_cache, last_geometry_fit_params, last_geometry_matched_peaks
    global last_geometry_overlay_state, last_simulation_signature

    if not isinstance(state, dict):
        return

    ui_params = state.get("ui_params", {}) or {}
    for name, var in (
        ("zb", zb_var),
        ("zs", zs_var),
        ("theta_initial", theta_initial_var),
        ("psi_z", psi_z_var),
        ("chi", chi_var),
        ("cor_angle", cor_angle_var),
        ("gamma", gamma_var),
        ("Gamma", Gamma_var),
        ("corto_detector", corto_detector_var),
        ("a", a_var),
        ("c", c_var),
        ("center_x", center_x_var),
        ("center_y", center_y_var),
    ):
        try:
            value = float(ui_params.get(name))
        except Exception:
            continue
        if np.isfinite(value):
            var.set(value)

    profile_cache = _copy_geometry_fit_state_value(state.get("profile_cache", {})) or {}
    last_geometry_fit_params = _copy_geometry_fit_state_value(
        state.get("last_geometry_fit_params")
    )
    last_geometry_matched_peaks = _copy_geometry_fit_state_value(
        state.get("last_geometry_matched_peaks", [])
    ) or []
    last_geometry_overlay_state = _copy_geometry_fit_state_value(
        state.get("overlay_state")
    )

    last_simulation_signature = None
    schedule_update()

    _clear_geometry_pick_artists()
    overlay_state = last_geometry_overlay_state or {}
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


def _undo_last_geometry_fit() -> None:
    if not geometry_fit_undo_stack:
        progress_label_geometry.config(text="No geometry fit history available to undo.")
        return

    state = geometry_fit_undo_stack.pop()
    _update_geometry_fit_undo_button_state()
    _restore_geometry_fit_undo_state(state)
    progress_label_geometry.config(text="Restored the previous geometry-fit state.")


def _current_geometry_fit_constraint_state(
    names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    selected_names = list(names) if names is not None else list(geometry_fit_constraint_controls)
    state: dict[str, dict[str, float]] = {}
    for name in selected_names:
        control = geometry_fit_constraint_controls.get(name)
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
        state[name] = {
            "window": float(window),
            "pull": float(pull),
        }
    return state


def _current_geometry_fit_parameter_domains(
    names: Sequence[str] | None = None,
) -> dict[str, tuple[float, float]]:
    selected_names = list(names) if names is not None else list(geometry_fit_parameter_specs)
    domains: dict[str, tuple[float, float]] = {}
    for name in selected_names:
        if name == "center_x" or name == "center_y":
            domains[name] = (0.0, max(float(image_size) - 1.0, 0.0))
            continue
        spec = geometry_fit_parameter_specs.get(name)
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
        if lo > hi:
            lo, hi = hi, lo
        domains[name] = (float(lo), float(hi))
    return domains


def _build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
):
    runtime_cfg = (
        _copy_geometry_fit_state_value(base_config)
        if isinstance(base_config, dict)
        else {}
    )
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    bounds_cfg = runtime_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}
    runtime_cfg["bounds"] = bounds_cfg

    priors_cfg = runtime_cfg.get("priors", {}) or {}
    if not isinstance(priors_cfg, dict):
        priors_cfg = {}
    runtime_cfg["priors"] = priors_cfg

    for name, current in (current_params or {}).items():
        try:
            current_value = float(current)
        except Exception:
            continue
        if not np.isfinite(current_value):
            continue

        control = (control_settings or {}).get(name, {}) or {}
        try:
            window = float(control.get("window", 0.0))
        except Exception:
            window = 0.0
        if not np.isfinite(window):
            window = 0.0
        window = max(0.0, float(window))

        lo = float(current_value - window)
        hi = float(current_value + window)

        domain = (parameter_domains or {}).get(name)
        if isinstance(domain, (list, tuple)) and len(domain) >= 2:
            try:
                domain_lo = float(domain[0])
                domain_hi = float(domain[1])
            except Exception:
                domain_lo = float("nan")
                domain_hi = float("nan")
            if np.isfinite(domain_lo):
                lo = max(lo, float(domain_lo))
            if np.isfinite(domain_hi):
                hi = min(hi, float(domain_hi))

        if hi < lo:
            lo = hi = min(max(current_value, lo), hi)

        bounds_cfg[name] = [float(lo), float(hi)]

        try:
            pull = float(control.get("pull", 0.0))
        except Exception:
            pull = 0.0
        if not np.isfinite(pull):
            pull = 0.0
        pull = min(max(float(pull), 0.0), 1.0)
        if pull > 0.0 and window > 0.0:
            sigma_scale = max(0.05, 1.0 - 0.95 * pull)
            priors_cfg[name] = {
                "center": float(current_value),
                "sigma": float(max(window * sigma_scale, 1.0e-6)),
            }
        else:
            priors_cfg.pop(name, None)

    return runtime_cfg


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
ax.set_ylim(row_center, 0)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.title('Simulated Diffraction Pattern')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
canvas.draw()

# -----------------------------------------------------------
# 1)  Highlight‑marker that we can move each click
# -----------------------------------------------------------
selected_peak_marker, = ax.plot([], [], 'ys',  # yellow square outline
                               markersize=8, markerfacecolor='none',
                               linewidth=1.5, zorder=6)
selected_peak_marker.set_visible(False)

# Geometry click markers (sim vs real)
geometry_pick_artists = []


def _clear_geometry_pick_artists():
    """Remove any geometry fit markers from the plot and reset the cache."""

    global geometry_pick_artists

    for artist in geometry_pick_artists:
        try:
            artist.remove()
        except ValueError:
            pass
    geometry_pick_artists.clear()
    canvas.draw_idle()


def _clear_all_geometry_overlay_artists():
    """Clear all geometry-fit overlays so the image can be inspected cleanly."""

    global last_geometry_overlay_state

    last_geometry_overlay_state = None
    _clear_geometry_pick_artists()
    try:
        progress_label_geometry.config(
            text="Geometry overlays cleared. Run Fit Geometry again to redraw them."
        )
    except Exception:
        pass


def _draw_geometry_fit_overlay(
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw one fixed-background/fitted-simulation overlay record per match."""

    _clear_geometry_pick_artists()

    def _parse_point(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _plot_marker(
        col: float,
        row: float,
        label: str | None,
        color: str,
        marker: str,
        *,
        zorder: int = 7,
    ) -> None:
        point, = ax.plot(
            [float(col)],
            [float(row)],
            marker,
            color=color,
            markersize=8,
            markerfacecolor="none",
            zorder=zorder,
            linestyle="None",
        )
        geometry_pick_artists.append(point)
        if label:
            text = ax.text(
                float(col),
                float(row),
                label,
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
                zorder=zorder + 1,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0),
            )
            geometry_pick_artists.append(text)

    def _plot_arrow(
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        *,
        color: str,
        linestyle: str = "--",
        lw: float = 1.0,
        alpha: float = 0.8,
        annotate: str | None = None,
    ) -> None:
        arrow = ax.annotate(
            annotate or "",
            xy=end_xy,
            xytext=start_xy,
            color=color,
            fontsize=8,
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=lw,
                linestyle=linestyle,
                alpha=alpha,
            ),
            bbox=(
                dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0)
                if annotate
                else None
            ),
            zorder=6,
        )
        geometry_pick_artists.append(arrow)

    limit = max(1, int(max_display_markers))
    for idx, entry in enumerate(overlay_records or []):
        if idx >= limit or not isinstance(entry, dict):
            break

        initial_sim_display = _parse_point(entry.get("initial_sim_display"))
        initial_bg_display = _parse_point(entry.get("initial_bg_display"))
        final_sim_display = _parse_point(entry.get("final_sim_display"))
        final_bg_display = _parse_point(entry.get("final_bg_display"))
        if final_sim_display is None or final_bg_display is None:
            continue

        label = str(entry.get("hkl", entry.get("label", "match")))
        if initial_sim_display is not None:
            _plot_marker(
                initial_sim_display[0],
                initial_sim_display[1],
                None,
                "#0984e3",
                "s",
                zorder=7,
            )
            sim_shift = math.hypot(
                final_sim_display[0] - initial_sim_display[0],
                final_sim_display[1] - initial_sim_display[1],
            )
            if sim_shift > 0.25:
                _plot_arrow(
                    initial_sim_display,
                    final_sim_display,
                    color="#0984e3",
                    linestyle="--",
                    lw=1.0,
                    alpha=0.85,
                )

        if initial_bg_display is not None:
            _plot_marker(
                initial_bg_display[0],
                initial_bg_display[1],
                None,
                "#f39c12",
                "^",
                zorder=7,
            )

        _plot_marker(
            final_sim_display[0],
            final_sim_display[1],
            f"{label} fit sim",
            "#00b894",
            "o",
            zorder=8,
        )

        residual_dist = float(entry.get("overlay_distance_px", np.nan))
        if not np.isfinite(residual_dist):
            residual_dist = math.hypot(
                final_sim_display[0] - final_bg_display[0],
                final_sim_display[1] - final_bg_display[1],
            )
        _plot_arrow(
            final_sim_display,
            final_bg_display,
            color="#2d3436",
            linestyle="-",
            lw=1.1,
            alpha=0.9,
            annotate=f"|Δ|={residual_dist:.1f}px",
        )

    canvas.draw_idle()


def _draw_initial_geometry_pairs_overlay(
    initial_pairs_display: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw only the initially selected simulation/background peak pairs."""

    initial_pairs = normalize_initial_geometry_pairs_display(initial_pairs_display)
    _clear_geometry_pick_artists()

    limit = max(1, int(max_display_markers))
    for idx, entry in enumerate(initial_pairs):
        if idx >= limit:
            break
        sim_display = entry.get("sim_display")
        bg_display = entry.get("bg_display")
        if sim_display is None and bg_display is None:
            continue
        hkl_label = str(entry.get("hkl", entry.get("label", idx)))

        if sim_display is not None:
            sim_pt, = ax.plot(
                [float(sim_display[0])],
                [float(sim_display[1])],
                "s",
                color="#0984e3",
                markersize=8,
                markerfacecolor="none",
                linestyle="None",
                zorder=7,
            )
            geometry_pick_artists.append(sim_pt)
        if bg_display is not None:
            bg_pt, = ax.plot(
                [float(bg_display[0])],
                [float(bg_display[1])],
                "^",
                color="#f39c12",
                markersize=8,
                markerfacecolor="none",
                linestyle="None",
                zorder=7,
            )
            geometry_pick_artists.append(bg_pt)

        if sim_display is not None and bg_display is not None:
            link = ax.annotate(
                hkl_label,
                xy=(float(bg_display[0]), float(bg_display[1])),
                xytext=(float(sim_display[0]), float(sim_display[1])),
                color="#636e72",
                fontsize=8,
                ha="center",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#636e72",
                    lw=1.0,
                    linestyle=":",
                    alpha=0.8,
                ),
                zorder=6,
            )
            geometry_pick_artists.append(link)

    canvas.draw_idle()


def _geometry_overlay_frame_diagnostics(
    overlay_records: Sequence[dict[str, object]] | None,
) -> tuple[dict[str, float], str]:
    """Compare per-match fitted overlay alignment in display space."""

    return compute_geometry_overlay_frame_diagnostics(overlay_records)


def _background_backend_status() -> str:
    return (
        f"k={int(background_backend_rotation_k) % 4} "
        f"flip_x={bool(background_backend_flip_x)} "
        f"flip_y={bool(background_backend_flip_y)}"
    )


def _update_background_backend_status():
    label = globals().get("background_backend_status_label")
    if label is not None:
        label.config(text=_background_backend_status())


def _rotate_background_backend(delta_k: int):
    global background_backend_rotation_k
    background_backend_rotation_k = (int(background_backend_rotation_k) + int(delta_k)) % 4
    _update_background_backend_status()
    _update_chi_square_display()
    schedule_update()


def _toggle_background_backend_flip(axis: str):
    global background_backend_flip_x, background_backend_flip_y
    axis = (axis or "").lower()
    if axis == "x":
        background_backend_flip_x = not background_backend_flip_x
    elif axis == "y":
        background_backend_flip_y = not background_backend_flip_y
    _update_background_backend_status()
    _update_chi_square_display()
    schedule_update()


def _reset_background_backend_orientation():
    global background_backend_rotation_k, background_backend_flip_x, background_backend_flip_y
    background_backend_rotation_k = 0
    background_backend_flip_x = False
    background_backend_flip_y = False
    _update_background_backend_status()
    _update_chi_square_display()
    schedule_update()

# -----------------------------------------------------------
# 2)  Mouse‑click handler
# -----------------------------------------------------------
selected_hkl_target = None


def _nearest_integer_hkl(h: float, k: float, l: float) -> tuple[int, int, int]:
    return (
        int(np.rint(float(h))),
        int(np.rint(float(k))),
        int(np.rint(float(l))),
    )


def _format_hkl_triplet(h: int, k: int, l: int) -> str:
    return f"({int(h)} {int(k)} {int(l)})"


def _source_miller_for_label(source_label: str | None) -> np.ndarray:
    label = str(source_label or "primary").lower()
    if label == "secondary" and isinstance(SIM_MILLER2, np.ndarray) and SIM_MILLER2.size:
        return np.asarray(SIM_MILLER2, dtype=float)
    if isinstance(SIM_MILLER1, np.ndarray) and SIM_MILLER1.size:
        return np.asarray(SIM_MILLER1, dtype=float)
    return np.empty((0, 3), dtype=float)


def _degenerate_hkls_for_qr(
    h: int,
    k: int,
    l: int,
    *,
    source_label: str | None,
) -> list[tuple[int, int, int]]:
    source_miller = _source_miller_for_label(source_label)
    if source_miller.ndim != 2 or source_miller.shape[1] < 3 or source_miller.shape[0] == 0:
        return []

    finite = np.isfinite(source_miller[:, 0]) & np.isfinite(source_miller[:, 1]) & np.isfinite(source_miller[:, 2])
    if not np.any(finite):
        return []
    source_int = np.rint(source_miller[finite, :3]).astype(np.int64, copy=False)

    h_vals = source_int[:, 0]
    k_vals = source_int[:, 1]
    l_vals = source_int[:, 2]
    m_vals = h_vals * h_vals + h_vals * k_vals + k_vals * k_vals
    m_target = int(h * h + h * k + k * k)

    mask = (l_vals == int(l)) & (m_vals == m_target)
    if not np.any(mask):
        rod_mask = m_vals == m_target
        if not np.any(rod_mask):
            return []
        nearest_l = int(l_vals[rod_mask][np.argmin(np.abs(l_vals[rod_mask] - int(l)))])
        mask = rod_mask & (l_vals == nearest_l)

    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for hh, kk, ll in source_int[mask]:
        key = (int(hh), int(kk), int(ll))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(hh), int(kk), int(ll)))

    out.sort(key=lambda vals: (vals[0], vals[1], vals[2]))
    return out


def _selected_peak_qr_and_degenerates(
    H: float,
    K: float,
    L: float,
    selected_peak: dict | None,
) -> tuple[float, list[tuple[int, int, int]]]:
    h_raw, k_raw, l_raw = float(H), float(K), float(L)
    source_label = "primary"
    a_used = float(av)
    if isinstance(selected_peak, dict):
        raw_hkl = selected_peak.get("hkl_raw")
        if isinstance(raw_hkl, (list, tuple, np.ndarray)) and len(raw_hkl) >= 3:
            h_raw = float(raw_hkl[0])
            k_raw = float(raw_hkl[1])
            l_raw = float(raw_hkl[2])
        source_label = str(selected_peak.get("source_label", "primary"))
        try:
            a_used = float(selected_peak.get("av", av))
        except (TypeError, ValueError):
            a_used = float(av)

    h_int, k_int, l_int = _nearest_integer_hkl(h_raw, k_raw, l_raw)
    m_val = float(h_int * h_int + h_int * k_int + k_int * k_int)
    if a_used > 0.0 and np.isfinite(a_used) and m_val >= 0.0:
        qr_val = (2.0 * np.pi / a_used) * np.sqrt((4.0 / 3.0) * m_val)
    else:
        qr_val = float("nan")

    deg_hkls = _degenerate_hkls_for_qr(
        h_int,
        k_int,
        l_int,
        source_label=source_label,
    )
    if not deg_hkls:
        deg_hkls = [(h_int, k_int, l_int)]
    return float(qr_val), deg_hkls


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
    global selected_hkl_target, selected_peak_record
    if idx < 0 or idx >= len(peak_positions):
        return False

    px, py = peak_positions[idx]
    H, K, L = peak_millers[idx]
    I = peak_intensities[idx]
    disp_col, disp_row = (
        (float(selected_display[0]), float(selected_display[1]))
        if selected_display is not None
        else (float(px), float(py))
    )

    selected_peak_marker.set_data([disp_col], [disp_row])
    selected_peak_marker.set_visible(True)

    selected_hkl_target = (int(H), int(K), int(L))
    selected_peak_record = dict(peak_records[idx]) if idx < len(peak_records) else None
    if selected_peak_record is not None:
        if clicked_display is not None:
            selected_peak_record["clicked_display_col"] = float(clicked_display[0])
            selected_peak_record["clicked_display_row"] = float(clicked_display[1])
        if clicked_native is not None:
            selected_peak_record["clicked_native_col"] = float(clicked_native[0])
            selected_peak_record["clicked_native_row"] = float(clicked_native[1])
        if selected_display is not None:
            selected_peak_record["selected_display_col"] = float(selected_display[0])
            selected_peak_record["selected_display_row"] = float(selected_display[1])
        if selected_native is not None:
            selected_peak_record["selected_native_col"] = float(selected_native[0])
            selected_peak_record["selected_native_row"] = float(selected_native[1])
        elif clicked_native is not None:
            selected_peak_record["selected_native_col"] = float(clicked_native[0])
            selected_peak_record["selected_native_row"] = float(clicked_native[1])
        else:
            selected_peak_record["selected_native_col"] = float(selected_peak_record["native_col"])
            selected_peak_record["selected_native_row"] = float(selected_peak_record["native_row"])
    if sync_hkl_vars:
        if "selected_h_var" in globals():
            selected_h_var.set(str(int(H)))
        if "selected_k_var" in globals():
            selected_k_var.set(str(int(K)))
        if "selected_l_var" in globals():
            selected_l_var.set(str(int(L)))

    qr_val, deg_hkls = _selected_peak_qr_and_degenerates(H, K, L, selected_peak_record)
    if selected_peak_record is not None:
        selected_peak_record["qr"] = float(qr_val)
        selected_peak_record["degenerate_hkls"] = [
            (int(hv), int(kv), int(lv)) for hv, kv, lv in deg_hkls
        ]

    shown_deg = deg_hkls[:12]
    deg_text = ", ".join(_format_hkl_triplet(hv, kv, lv) for hv, kv, lv in shown_deg)
    if len(deg_hkls) > len(shown_deg):
        deg_text += f", ... (+{len(deg_hkls) - len(shown_deg)} more)"
    qr_text = f"  Qr={qr_val:.4f} A^-1" if np.isfinite(qr_val) else ""

    progress_label_positions.config(
        text=(
            f"{prefix}: HKL=({int(H)} {int(K)} {int(L)})  pixel=({disp_col:.1f},{disp_row:.1f})  "
            f"I={I:.2g}{qr_text}  HKLs@same Qr,L: {deg_text}"
        )
    )
    canvas.draw_idle()
    return True


def _select_peak_by_hkl(
    h: int,
    k: int,
    l: int,
    *,
    sync_hkl_vars: bool = True,
    silent_if_missing: bool = False,
):
    global selected_hkl_target, selected_peak_record
    target = (int(h), int(k), int(l))

    if not peak_positions:
        if not silent_if_missing:
            progress_label_positions.config(text="Run a simulation first.")
        return False

    matches = [
        idx for idx, hkl in enumerate(peak_millers)
        if tuple(int(np.rint(v)) for v in hkl) == target and peak_positions[idx][0] >= 0
    ]

    if not matches:
        if not silent_if_missing:
            progress_label_positions.config(
                text=f"HKL ({target[0]} {target[1]} {target[2]}) not found in current simulation."
            )
        selected_hkl_target = target
        selected_peak_record = None
        return False

    def _score(i: int) -> float:
        val = peak_intensities[i]
        return float(val) if np.isfinite(val) else float("-inf")

    best_idx = max(matches, key=_score)
    return _select_peak_by_index(
        best_idx,
        prefix="Selected peak",
        sync_hkl_vars=sync_hkl_vars,
    )


def _select_peak_from_hkl_controls():
    global selected_hkl_target
    try:
        h = int(round(float(selected_h_var.get().strip())))
        k = int(round(float(selected_k_var.get().strip())))
        l = int(round(float(selected_l_var.get().strip())))
    except (ValueError, tk.TclError, AttributeError):
        progress_label_positions.config(text="Enter numeric H, K, L values.")
        return

    selected_hkl_target = (h, k, l)
    _select_peak_by_hkl(h, k, l, sync_hkl_vars=True, silent_if_missing=False)


def _clear_selected_peak():
    global selected_hkl_target, selected_peak_record
    selected_hkl_target = None
    selected_peak_record = None
    selected_peak_marker.set_visible(False)
    progress_label_positions.config(text="Peak selection cleared.")
    canvas.draw_idle()


def _open_selected_peak_intersection_figure():
    """Open a Bragg/Ewald intersection analysis plot for the selected peak."""

    if selected_peak_record is None:
        progress_label_positions.config(
            text="Select a Bragg peak first (click a peak or use HKL controls)."
        )
        return

    try:
        h, k, l = tuple(int(v) for v in selected_peak_record["hkl"])
        native_col = float(
            selected_peak_record.get(
                "selected_native_col",
                selected_peak_record.get("native_col"),
            )
        )
        native_row = float(
            selected_peak_record.get(
                "selected_native_row",
                selected_peak_record.get("native_row"),
            )
        )
        lattice_a = float(selected_peak_record["av"])
        lattice_c = float(selected_peak_record["cv"])

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
            beam_x_array=np.asarray(profile_cache["beam_x_array"], dtype=np.float64),
            beam_y_array=np.asarray(profile_cache["beam_y_array"], dtype=np.float64),
            theta_array=np.asarray(profile_cache["theta_array"], dtype=np.float64),
            phi_array=np.asarray(profile_cache["phi_array"], dtype=np.float64),
            wavelength_array=np.asarray(profile_cache["wavelength_array"], dtype=np.float64),
        )
        mosaic = IntersectionMosaicParams(
            sigma_mosaic_deg=float(sigma_mosaic_var.get()),
            gamma_mosaic_deg=float(gamma_mosaic_var.get()),
            eta=float(eta_var.get()),
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
                f"from source={selected_peak_record.get('source_label', 'unknown')}."
            )
        )
    except Exception as exc:
        progress_label_positions.config(
            text=f"Intersection analysis failed for selected peak: {exc}"
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
    h: int,
    k: int,
    l: int,
    av: float,
    cv: float,
) -> tuple[float, float] | None:
    """Simulate a single HKL and return its no-mosaic/no-divergence center."""

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
    beam_x = np.asarray(profile_cache.get("beam_x_array", []), dtype=np.float64).ravel()
    beam_y = np.asarray(profile_cache.get("beam_y_array", []), dtype=np.float64).ravel()
    theta_arr = np.asarray(profile_cache.get("theta_array", []), dtype=np.float64).ravel()
    phi_arr = np.asarray(profile_cache.get("phi_array", []), dtype=np.float64).ravel()
    wavelength_arr = np.asarray(
        profile_cache.get("wavelength_array", []), dtype=np.float64
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
    if event.inaxes is not ax or event.xdata is None:
        return                               # click was outside the image
    if not peak_positions:                   # no simulation yet
        progress_label_positions.config(text="Run a simulation first.")
        return

    click_col = float(event.xdata)
    click_row = float(event.ydata)

    # Find nearest stored peak
    best_i, best_d2 = -1, float("inf")
    best_i_val = float("-inf")
    for i, (px, py) in enumerate(peak_positions):
        if px < 0:              # invalid entries kept as (-1,-1)
            continue
        d2 = (px - click_col) ** 2 + (py - click_row) ** 2
        val = peak_intensities[i]
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

    if best_i < len(peak_records) and isinstance(peak_records[best_i], dict):
        peak_record = peak_records[best_i]
        try:
            rec_h, rec_k, rec_l = tuple(int(np.rint(v)) for v in peak_millers[best_i])
            rec_av = float(peak_record.get("av", float(a_var.get())))
            rec_cv = float(peak_record.get("cv", float(c_var.get())))
            ideal_native = _simulate_ideal_hkl_native_center(
                rec_h, rec_k, rec_l, rec_av, rec_cv
            )
            if ideal_native is not None:
                selected_native = ideal_native
                selected_display = _native_sim_to_display_coords(
                    ideal_native[0],
                    ideal_native[1],
                    sim_shape,
                )
        except Exception:
            selected_display = None
            selected_native = None

    if selected_native is None and best_i < len(peak_records) and isinstance(peak_records[best_i], dict):
        selected_native = (
            float(peak_records[best_i].get("native_col", clicked_native_col)),
            float(peak_records[best_i].get("native_row", clicked_native_row)),
        )

    prefix = f"Nearest peak (Δ={best_d2**0.5:.1f}px)"
    if selected_display is not None:
        snapped_dist = math.hypot(
            float(selected_display[0]) - click_col,
            float(selected_display[1]) - click_row,
        )
        prefix = f"Ideal HKL center (click Δ={snapped_dist:.1f}px)"

    _select_peak_by_index(
        best_i,
        prefix=prefix,
        sync_hkl_vars=True,
        clicked_display=(click_col, click_row),
        clicked_native=(float(clicked_native_col), float(clicked_native_row)),
        selected_display=selected_display,
        selected_native=selected_native,
    )

# -----------------------------------------------------------
# 3)  Bind the handler
# -----------------------------------------------------------
canvas.mpl_connect('button_press_event', on_canvas_click)


# ---------------------------------------------------------------------------
# Display controls for background and simulation intensity scaling
# ---------------------------------------------------------------------------
background_limits_user_override = False
simulation_limits_user_override = False
scale_factor_user_override = False

suppress_background_limit_callback = False
suppress_simulation_limit_callback = False
suppress_scale_factor_callback = False

background_min_var = None
background_max_var = None
background_transparency_var = None
simulation_min_var = None
simulation_max_var = None


def _finite_percentile(array, percentile, fallback):
    if array is None:
        return fallback
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    return float(np.nanpercentile(finite, percentile))


def _ensure_valid_range(min_val, max_val):
    if not np.isfinite(min_val):
        min_val = 0.0
    if not np.isfinite(max_val):
        max_val = max(min_val + 1.0, 1.0)
    if max_val <= min_val:
        max_val = min_val + max(abs(min_val) * 1e-3, 1.0)
    return min_val, max_val


def _apply_background_transparency():
    if background_transparency_var is None:
        return
    transparency = max(0.0, min(1.0, background_transparency_var.get()))
    background_display.set_alpha(1.0 - transparency)


def _apply_background_limits():
    global background_limits_user_override, suppress_background_limit_callback
    if background_min_var is None or background_max_var is None:
        return
    min_val = background_min_var.get()
    max_val = background_max_var.get()
    if min_val >= max_val:
        adjustment = max(abs(max_val) * 1e-6, 1e-6)
        suppress_background_limit_callback = True
        background_min_var.set(max_val - adjustment)
        suppress_background_limit_callback = False
        return
    background_limits_user_override = True
    background_display.set_clim(min_val, max_val)
    _apply_background_transparency()
    canvas.draw_idle()


def _apply_simulation_limits():
    global simulation_limits_user_override, suppress_simulation_limit_callback
    if simulation_min_var is None or simulation_max_var is None:
        return
    min_val = simulation_min_var.get()
    max_val = simulation_max_var.get()
    if min_val >= max_val:
        adjustment = max(abs(max_val) * 1e-6, 1e-6)
        suppress_simulation_limit_callback = True
        simulation_min_var.set(max_val - adjustment)
        suppress_simulation_limit_callback = False
        return
    simulation_limits_user_override = True
    apply_scale_factor_to_existing_results()


background_controls = ttk.LabelFrame(display_controls_frame, text="Background Display")
background_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

simulation_controls = ttk.LabelFrame(display_controls_frame, text="Simulation Display")
simulation_controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)


background_min_candidate = _finite_percentile(current_background_display, 1, 0.0)
background_vmin_default = 0.0
_, background_vmax_default = _ensure_valid_range(
    background_vmin_default,
    _finite_percentile(current_background_display, 99, 1.0),
)

background_slider_min = min(background_min_candidate, 0.0)
background_slider_max = max(background_vmax_default * 5.0, background_slider_min + 1.0)
background_slider_step = max((background_slider_max - background_slider_min) / 500.0, 0.01)

background_min_var, background_min_slider = create_slider(
    "Background Min Intensity",
    background_slider_min,
    background_slider_max,
    background_vmin_default,
    background_slider_step,
    parent=background_controls,
    update_callback=_apply_background_limits,
)

background_max_var, background_max_slider = create_slider(
    "Background Max Intensity",
    background_slider_min,
    background_slider_max,
    background_vmax_default,
    background_slider_step,
    parent=background_controls,
    update_callback=_apply_background_limits,
)

background_transparency_var, _ = create_slider(
    "Background Transparency",
    0.0,
    1.0,
    0.0,
    0.01,
    parent=background_controls,
    update_callback=_apply_background_limits,
)

background_display.set_clim(background_vmin_default, background_vmax_default)
_apply_background_transparency()


simulation_slider_min = 0.0
simulation_slider_max = max(background_slider_max, defaults['vmax'] * 5.0)
# Ensure fine-grained control so the intensity sliders support at least 1e-4 precision.
simulation_slider_step = min(
    max((simulation_slider_max - simulation_slider_min) / 500.0, 1e-6),
    1e-4,
)

simulation_min_var, simulation_min_slider = create_slider(
    "Simulation Min Intensity",
    simulation_slider_min,
    simulation_slider_max,
    0.0,
    simulation_slider_step,
    parent=simulation_controls,
    update_callback=_apply_simulation_limits,
)

simulation_max_var, simulation_max_slider = create_slider(
    "Simulation Max Intensity",
    simulation_slider_min,
    simulation_slider_max,
    background_vmax_default,
    simulation_slider_step,
    parent=simulation_controls,
    update_callback=_apply_simulation_limits,
)

scale_factor_slider_min = 0.0
scale_factor_slider_max = 2.0
scale_factor_step = 0.0001

simulation_scale_factor_var, scale_factor_slider = create_slider(
    "Simulation Scale Factor",
    scale_factor_slider_min,
    scale_factor_slider_max,
    1.0,
    scale_factor_step,
    parent=simulation_controls,
)


def _on_scale_factor_change(*args):
    global scale_factor_user_override
    if suppress_scale_factor_callback:
        return
    if _get_scale_factor_value(default=None) is None:
        return
    scale_factor_user_override = True
    apply_scale_factor_to_existing_results()


simulation_scale_factor_var.trace_add("write", _on_scale_factor_change)


def _update_background_slider_defaults(image, reset_override=False):
    global suppress_background_limit_callback, background_limits_user_override
    if image is None:
        return
    min_candidate = _finite_percentile(image, 1, 0.0)
    max_candidate = _finite_percentile(image, 99, background_max_var.get())
    min_candidate, max_candidate = _ensure_valid_range(min_candidate, max_candidate)
    slider_from = min(float(background_min_slider.cget("from")), min_candidate, 0.0)
    slider_to = max(float(background_min_slider.cget("to")), max_candidate, 1.0)
    background_min_slider.configure(from_=slider_from, to=slider_to)
    background_max_slider.configure(from_=slider_from, to=slider_to)
    suppress_background_limit_callback = True
    if reset_override or not background_limits_user_override:
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
    suppress_background_limit_callback = False
    background_display.set_clim(min_value, max_value)
    if reset_override:
        background_limits_user_override = False


def _update_simulation_sliders_from_image(image, reset_override=False):
    global suppress_simulation_limit_callback, simulation_limits_user_override
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
    suppress_simulation_limit_callback = True
    if reset_override or not simulation_limits_user_override:
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
    suppress_simulation_limit_callback = False
    if reset_override:
        simulation_limits_user_override = False


def _set_scale_factor_value(value, adjust_range=True, reset_override=False):
    global suppress_scale_factor_callback, scale_factor_user_override
    if value is None:
        value = 1.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 1.0
    if not np.isfinite(value):
        value = 1.0
    if value < 0.0:
        value = 0.0

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

    suppress_scale_factor_callback = True
    simulation_scale_factor_var.set(value)
    suppress_scale_factor_callback = False
    if reset_override:
        scale_factor_user_override = False
    else:
        scale_factor_user_override = True


def _get_scale_factor_value(default=1.0):
    try:
        scale = float(simulation_scale_factor_var.get())
    except (tk.TclError, ValueError):
        return default
    if not np.isfinite(scale):
        return default
    return scale


def _install_scale_factor_entry_bindings():
    scale_entry = None
    for child in scale_factor_slider.master.winfo_children():
        if isinstance(child, ttk.Entry):
            scale_entry = child
            break
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
    sim_pixels = sim_pixels[np.isfinite(sim_pixels)]
    bg_pixels = bg_pixels[np.isfinite(bg_pixels)]
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


def _auto_match_scale_factor_to_radial_peak():
    sim_curve = last_1d_integration_data.get("intensities_2theta_sim")
    bg_curve = last_1d_integration_data.get("intensities_2theta_bg")

    if sim_curve is None or bg_curve is None:
        ai = _ai_cache.get("ai")
        sim_img = last_1d_integration_data.get("simulated_2d_image")
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
                last_1d_integration_data["intensities_2theta_sim"] = i2t_sim
                last_1d_integration_data["intensities_2theta_bg"] = i2t_bg
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
    simulation_controls,
    text="Auto-Match Scale (Radial Peak)",
    command=_auto_match_scale_factor_to_radial_peak,
).pack(anchor=tk.W, padx=5, pady=(0, 6))


def _update_chi_square_display():
    try:
        native_background = _get_current_background_backend()
        if (
            background_visible
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
                norm_sim = sim_vals / np.max(sim_vals)
                norm_bg = bg_vals / np.max(bg_vals)
                chi_sq_val = mean_squared_error(norm_bg, norm_sim) * norm_sim.size
                chi_square_label.config(text=f"Chi-Squared: {chi_sq_val:.2e}")
            else:
                chi_square_label.config(text="Chi-Squared: N/A")
        else:
            chi_square_label.config(text="Chi-Squared: N/A")
    except Exception as exc:
        chi_square_label.config(text=f"Chi-Squared Error: {exc}")


def apply_scale_factor_to_existing_results(update_limits=False):
    if unscaled_image_global is None:
        background_display.set_clim(
            background_min_var.get(),
            background_max_var.get(),
        )
        if not show_caked_2d_var.get():
            if background_visible and current_background_display is not None:
                background_display.set_data(current_background_display)
                background_display.set_visible(True)
            else:
                background_display.set_visible(False)
        canvas.draw_idle()
        _update_chi_square_display()
        return

    scale = _get_scale_factor_value(default=1.0)
    scaled_image = unscaled_image_global * scale
    global_image_buffer[:] = scaled_image

    # Keep display limits stable during reruns/parameter changes unless an
    # explicit reset path requests recomputing limits.
    if update_limits:
        _update_simulation_sliders_from_image(
            scaled_image, reset_override=update_limits
        )

    if not show_caked_2d_var.get():
        _set_image_origin(image_display, 'upper')
        image_display.set_extent([0, image_size, image_size, 0])
        image_display.set_data(global_image_buffer)
        image_display.set_clim(
            simulation_min_var.get(),
            simulation_max_var.get(),
        )
        colorbar_main.update_normal(image_display)
    elif last_caked_image_unscaled is not None and last_caked_extent is not None:
        _set_image_origin(image_display, 'lower')
        scaled_caked = last_caked_image_unscaled * scale
        if not simulation_limits_user_override:
            _update_simulation_sliders_from_image(scaled_caked, reset_override=True)
        image_display.set_data(scaled_caked)
        image_display.set_extent(last_caked_extent)

    if show_caked_2d_var.get():
        caked_min = float(simulation_min_var.get())
        caked_max = float(simulation_max_var.get())
        caked_min, caked_max = _ensure_valid_range(caked_min, caked_max)
        image_display.set_clim(caked_min, caked_max)
        if not caked_limits_user_override:
            vmin_caked_var.set(caked_min)
            vmax_caked_var.set(caked_max)

    if (
        show_1d_var.get()
        and "line_1d_rad" in globals()
        and "line_1d_az" in globals()
    ):
        if (
            last_1d_integration_data["radials_sim"] is not None
            and last_1d_integration_data["intensities_2theta_sim"] is not None
        ):
            line_1d_rad.set_data(
                last_1d_integration_data["radials_sim"],
                last_1d_integration_data["intensities_2theta_sim"] * scale,
            )
        if (
            last_1d_integration_data["azimuths_sim"] is not None
            and last_1d_integration_data["intensities_azimuth_sim"] is not None
        ):
            line_1d_az.set_data(
                last_1d_integration_data["azimuths_sim"],
                last_1d_integration_data["intensities_azimuth_sim"] * scale,
            )
        if "canvas_1d" in globals():
            canvas_1d.draw_idle()

    background_display.set_clim(
        background_min_var.get(),
        background_max_var.get(),
    )

    if not show_caked_2d_var.get():
        if background_visible and current_background_display is not None:
            background_display.set_data(current_background_display)
            background_display.set_visible(True)
        else:
            background_display.set_visible(False)

    canvas.draw_idle()
    _update_chi_square_display()
_update_background_slider_defaults(current_background_display, reset_override=True)


# Track caked intensity limits without exposing separate sliders in the UI.
caked_limits_user_override = False

vmin_caked_var = tk.DoubleVar(value=0.0)
vmax_caked_var = tk.DoubleVar(value=2000.0)

slider_frame = ttk.Frame(root, padding=10)
slider_frame.pack(side=tk.LEFT, fill=tk.Y)

left_col = ttk.Frame(slider_frame)
left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

right_col = ttk.Frame(slider_frame)
right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

plot_frame_1d = ttk.Frame(root)
plot_frame_1d.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig_1d, (ax_1d_radial, ax_1d_azim) = plt.subplots(2, 1, figsize=(5, 8))
canvas_1d = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig_1d, master=plot_frame_1d)
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

tth_min_var = tk.DoubleVar(value=0.0)
tth_max_var = tk.DoubleVar(value=60.0)
phi_min_var = tk.DoubleVar(value=-15.0)
phi_max_var = tk.DoubleVar(value=15.0)

tth_min_label_var = tk.StringVar(value=f"{tth_min_var.get():.1f}")
tth_max_label_var = tk.StringVar(value=f"{tth_max_var.get():.1f}")
phi_min_label_var = tk.StringVar(value=f"{phi_min_var.get():.1f}")
phi_max_label_var = tk.StringVar(value=f"{phi_max_var.get():.1f}")
tth_min_entry_var = tk.StringVar(value=f"{tth_min_var.get():.4f}")
tth_max_entry_var = tk.StringVar(value=f"{tth_max_var.get():.4f}")
phi_min_entry_var = tk.StringVar(value=f"{phi_min_var.get():.4f}")
phi_max_entry_var = tk.StringVar(value=f"{phi_max_var.get():.4f}")


def _sync_range_text_vars():
    tth_min_label_var.set(f"{tth_min_var.get():.1f}")
    tth_max_label_var.set(f"{tth_max_var.get():.1f}")
    phi_min_label_var.set(f"{phi_min_var.get():.1f}")
    phi_max_label_var.set(f"{phi_max_var.get():.1f}")
    tth_min_entry_var.set(f"{tth_min_var.get():.4f}")
    tth_max_entry_var.set(f"{tth_max_var.get():.4f}")
    phi_min_entry_var.set(f"{phi_min_var.get():.4f}")
    phi_max_entry_var.set(f"{phi_max_var.get():.4f}")


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
    schedule_update()


def _on_range_var_write(*_args):
    _sync_range_text_vars()


for _var in (tth_min_var, tth_max_var, phi_min_var, phi_max_var):
    _var.trace_add("write", _on_range_var_write)

def tth_min_slider_command(val):
    val_f = float(val)
    tth_min_var.set(val_f)
    _sync_range_text_vars()
    schedule_update()

def tth_max_slider_command(val):
    val_f = float(val)
    tth_max_var.set(val_f)
    _sync_range_text_vars()
    schedule_update()

def phi_min_slider_command(val):
    val_f = float(val)
    phi_min_var.set(val_f)
    _sync_range_text_vars()
    schedule_update()

def phi_max_slider_command(val):
    val_f = float(val)
    phi_max_var.set(val_f)
    _sync_range_text_vars()
    schedule_update()

range_cf = CollapsibleFrame(plot_frame_1d, text='Integration Ranges', expanded=True)
range_cf.pack(side=tk.TOP, fill=tk.X, pady=5)
range_frame = range_cf.frame

tth_min_container = ttk.Frame(range_frame)
tth_min_container.pack(side=tk.TOP, fill=tk.X, pady=2)
ttk.Label(tth_min_container, text="2θ Min (°):").pack(side=tk.LEFT, padx=5)
tth_min_slider = ttk.Scale(
    tth_min_container,
    from_=0.0,
    to=90.0,
    orient=tk.HORIZONTAL,
    variable=tth_min_var,
    command=tth_min_slider_command
)
tth_min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
ttk.Label(tth_min_container, textvariable=tth_min_label_var, width=6).pack(side=tk.LEFT, padx=4)
tth_min_entry = ttk.Entry(tth_min_container, textvariable=tth_min_entry_var, width=8)
tth_min_entry.pack(side=tk.LEFT, padx=(0, 5))
tth_min_entry.bind(
    "<Return>",
    lambda _e: _apply_range_entry(tth_min_entry_var, tth_min_var, tth_min_slider),
)
tth_min_entry.bind(
    "<FocusOut>",
    lambda _e: _apply_range_entry(tth_min_entry_var, tth_min_var, tth_min_slider),
)

tth_max_container = ttk.Frame(range_frame)
tth_max_container.pack(side=tk.TOP, fill=tk.X, pady=2)
ttk.Label(tth_max_container, text="2θ Max (°):").pack(side=tk.LEFT, padx=5)
tth_max_slider = ttk.Scale(
    tth_max_container,
    from_=0.0,
    to=90.0,
    orient=tk.HORIZONTAL,
    variable=tth_max_var,
    command=tth_max_slider_command
)
tth_max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
ttk.Label(tth_max_container, textvariable=tth_max_label_var, width=6).pack(side=tk.LEFT, padx=4)
tth_max_entry = ttk.Entry(tth_max_container, textvariable=tth_max_entry_var, width=8)
tth_max_entry.pack(side=tk.LEFT, padx=(0, 5))
tth_max_entry.bind(
    "<Return>",
    lambda _e: _apply_range_entry(tth_max_entry_var, tth_max_var, tth_max_slider),
)
tth_max_entry.bind(
    "<FocusOut>",
    lambda _e: _apply_range_entry(tth_max_entry_var, tth_max_var, tth_max_slider),
)

phi_min_container = ttk.Frame(range_frame)
phi_min_container.pack(side=tk.TOP, fill=tk.X, pady=2)
ttk.Label(phi_min_container, text="φ Min (°):").pack(side=tk.LEFT, padx=5)
phi_min_slider = ttk.Scale(
    phi_min_container,
    from_=-90.0,
    to=90.0,
    orient=tk.HORIZONTAL,
    variable=phi_min_var,
    command=phi_min_slider_command
)
phi_min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
ttk.Label(phi_min_container, textvariable=phi_min_label_var, width=6).pack(side=tk.LEFT, padx=4)
phi_min_entry = ttk.Entry(phi_min_container, textvariable=phi_min_entry_var, width=8)
phi_min_entry.pack(side=tk.LEFT, padx=(0, 5))
phi_min_entry.bind(
    "<Return>",
    lambda _e: _apply_range_entry(phi_min_entry_var, phi_min_var, phi_min_slider),
)
phi_min_entry.bind(
    "<FocusOut>",
    lambda _e: _apply_range_entry(phi_min_entry_var, phi_min_var, phi_min_slider),
)

phi_max_container = ttk.Frame(range_frame)
phi_max_container.pack(side=tk.TOP, fill=tk.X, pady=2)
ttk.Label(phi_max_container, text="φ Max (°):").pack(side=tk.LEFT, padx=5)
phi_max_slider = ttk.Scale(
    phi_max_container,
    from_=-90.0,
    to=90.0,
    orient=tk.HORIZONTAL,
    variable=phi_max_var,
    command=phi_max_slider_command
)
phi_max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
ttk.Label(phi_max_container, textvariable=phi_max_label_var, width=6).pack(side=tk.LEFT, padx=4)
phi_max_entry = ttk.Entry(phi_max_container, textvariable=phi_max_entry_var, width=8)
phi_max_entry.pack(side=tk.LEFT, padx=(0, 5))
phi_max_entry.bind(
    "<Return>",
    lambda _e: _apply_range_entry(phi_max_entry_var, phi_max_var, phi_max_slider),
)
phi_max_entry.bind(
    "<FocusOut>",
    lambda _e: _apply_range_entry(phi_max_entry_var, phi_max_var, phi_max_slider),
)

_sync_range_text_vars()

PHI_ZERO_OFFSET_DEGREES = 90.0


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

    mask_rad = (radial_2theta >= tth_min) & (radial_2theta <= tth_max)
    radial_filtered = radial_2theta[mask_rad]

    mask_az = (azimuth_vals >= phi_min) & (azimuth_vals <= phi_max)
    azimuth_sub = azimuth_vals[mask_az]

    intensity_sub = intensity[np.ix_(mask_az, mask_rad)]
    intensity_vs_2theta = np.sum(intensity_sub, axis=0)
    intensity_vs_phi = np.sum(intensity_sub, axis=1)

    return intensity_vs_2theta, intensity_vs_phi, azimuth_sub, radial_filtered


def update_integration_region_visuals(ai, sim_res2):
    show_region = show_1d_var.get() and unscaled_image_global is not None

    if not show_region:
        integration_region_overlay.set_visible(False)
        integration_region_rect.set_visible(False)
        return

    tth_values = sorted((tth_min_var.get(), tth_max_var.get()))
    phi_values = sorted((phi_min_var.get(), phi_max_var.get()))
    tth_min, tth_max = tth_values
    phi_min, phi_max = phi_values

    if show_caked_2d_var.get() and sim_res2 is not None:
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

    detector_shape = global_image_buffer.shape
    if _ai_cache.get("detector_shape") != detector_shape:
        try:
            two_theta = ai.twoThetaArray(shape=detector_shape, unit="2th_deg")
        except TypeError:
            two_theta = np.rad2deg(ai.twoThetaArray(shape=detector_shape))

        try:
            phi_vals = ai.chiArray(shape=detector_shape, unit="deg")
        except TypeError:
            phi_vals = np.rad2deg(ai.chiArray(shape=detector_shape))

        _ai_cache["detector_shape"] = detector_shape
        _ai_cache["detector_two_theta"] = two_theta
        _ai_cache["detector_phi"] = phi_vals

    two_theta = _ai_cache.get("detector_two_theta")
    phi_vals = _ai_cache.get("detector_phi")

    if two_theta is None or phi_vals is None:
        integration_region_overlay.set_visible(False)
        return

    phi_vals = _adjust_phi_zero(phi_vals)

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


profile_cache = {}
last_geometry_fit_params = None
last_geometry_matched_peaks = []
last_geometry_overlay_state = None
geometry_fit_undo_stack = []
last_1d_integration_data = {
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

last_caked_image_unscaled = None
last_caked_extent = None

last_res2_background = None
last_res2_sim = None
_ai_cache = {}
undo_geometry_fit_button = None

def update_mosaic_cache():
    """
    Keep the current random beam/mosaic samples unless sampling inputs changed.

    This preserves the same sampled beam positions/divergence across normal
    simulation updates so changing unrelated sliders does not re-randomize the
    detector pattern.
    """
    global profile_cache
    sampling_signature = (
        int(num_samples),
        float(divergence_sigma),
        float(bw_sigma),
        float(lambda_),
        float(bandwidth),
    )

    beam_x_cached = np.asarray(profile_cache.get("beam_x_array", []), dtype=np.float64).ravel()
    beam_y_cached = np.asarray(profile_cache.get("beam_y_array", []), dtype=np.float64).ravel()
    theta_cached = np.asarray(profile_cache.get("theta_array", []), dtype=np.float64).ravel()
    phi_cached = np.asarray(profile_cache.get("phi_array", []), dtype=np.float64).ravel()
    wavelength_cached = np.asarray(profile_cache.get("wavelength_array", []), dtype=np.float64).ravel()

    has_cached_samples = (
        beam_x_cached.size > 0
        and beam_x_cached.size == int(num_samples)
        and beam_y_cached.size == beam_x_cached.size
        and theta_cached.size == beam_x_cached.size
        and phi_cached.size == beam_x_cached.size
        and wavelength_cached.size == beam_x_cached.size
    )

    cached_signature = profile_cache.get("_sampling_signature")
    should_resample = not has_cached_samples
    if has_cached_samples:
        if cached_signature is None:
            # Existing externally-provided samples: keep them and adopt the
            # current signature so future explicit sampling-input changes can
            # trigger a re-sample.
            profile_cache["_sampling_signature"] = sampling_signature
        else:
            should_resample = tuple(cached_signature) != sampling_signature

    if should_resample:
        (beam_x_array,
         beam_y_array,
         theta_array,
         phi_array,
         wavelength_array) = generate_random_profiles(
             num_samples=num_samples,
             divergence_sigma=divergence_sigma,
             bw_sigma=bw_sigma,
             lambda0=lambda_,
             bandwidth=bandwidth
         )
        profile_cache = {
            "beam_x_array": beam_x_array,
            "beam_y_array": beam_y_array,
            "theta_array": theta_array,
            "phi_array": phi_array,
            "wavelength_array": wavelength_array,
            "_sampling_signature": sampling_signature,
        }

    profile_cache.update(
        {
            "sigma_mosaic_deg": sigma_mosaic_var.get(),
            "gamma_mosaic_deg": gamma_mosaic_var.get(),
            "eta": eta_var.get(),
        }
    )

def on_mosaic_slider_change(*args):
    update_mosaic_cache()
    schedule_update()

line_rmin, = ax.plot([], [], color='white', linestyle='-', linewidth=2, zorder=5)
line_rmax, = ax.plot([], [], color='white', linestyle='-', linewidth=2, zorder=5)
line_amin, = ax.plot([], [], color='cyan', linestyle='-', linewidth=2, zorder=5)
line_amax, = ax.plot([], [], color='cyan', linestyle='-', linewidth=2, zorder=5)

update_pending = None
update_running = False

def schedule_update():
    """Queue a throttled simulation/redraw update."""
    global update_pending
    if update_pending is not None:
        root.after_cancel(update_pending)
    update_pending = root.after(1000, do_update)

peak_positions = []
peak_millers = []
peak_intensities = []
peak_records = []
selected_peak_record = None
HKL_PICK_MAX_HITS_PER_REFLECTION = 2
HKL_PICK_MIN_SEPARATION_PX = 2.0
HKL_PICK_MAX_DISTANCE_PX = 12.0

prev_background_visible = True
last_bg_signature = None
last_sim_signature = None
last_simulation_signature = None
stored_max_positions_local = None
stored_sim_image = None
stored_peak_table_lattice = None

###############################################################################
#                              MAIN UPDATE
###############################################################################
def do_update():
    global update_pending, update_running, last_simulation_signature
    global unscaled_image_global, background_visible
    global stored_max_positions_local, stored_sim_image, stored_peak_table_lattice
    global SIM_MILLER1, SIM_INTENS1, SIM_MILLER2, SIM_INTENS2
    global av2, cv2
    global last_caked_image_unscaled, last_caked_extent

    if update_running:
        # another update is in progress; try again shortly
        update_pending = root.after(1000, do_update)
        return

    update_pending = None
    update_running = True
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

    global two_theta_range, _last_c_for_ht
    need_rebuild = False
    if not math.isclose(new_two_theta_max, two_theta_range[1], rel_tol=1e-6, abs_tol=1e-6):
        two_theta_range = (0.0, new_two_theta_max)
        need_rebuild = True
    if not math.isclose(c_updated, _last_c_for_ht, rel_tol=1e-9, abs_tol=1e-9):
        need_rebuild = True

    if need_rebuild:
        current_occ = [occ_var1.get(), occ_var2.get(), occ_var3.get()]
        current_p = [p0_var.get(), p1_var.get(), p2_var.get()]
        weight_values = [w0_var.get(), w1_var.get(), w2_var.get()]
        weight_sum = sum(weight_values) or 1.0
        normalized_weights = [w / weight_sum for w in weight_values]
        _rebuild_diffraction_inputs(
            current_occ,
            current_p,
            normalized_weights,
            c_updated,
            force=True,
            trigger_update=False,
        )

    center_marker.set_xdata([center_y_up])
    center_marker.set_ydata([center_x_up])
    center_marker.set_visible(False)

    mosaic_params = build_mosaic_params()
    optics_mode_flag = _current_optics_mode_flag()


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
            int(optics_mode_flag),
            int(num_samples),
            int(np.size(mosaic_params["beam_x_array"])),
            int(np.size(mosaic_params["theta_array"])),
        )

    # 1 – place near other globals

    # … inside do_update() …
    global stored_max_positions_local        # <- add

    new_sim_sig = get_sim_signature()
    global peak_positions, peak_millers, peak_intensities, peak_records, selected_peak_record
    if new_sim_sig != last_simulation_signature:
        last_simulation_signature = new_sim_sig
        peak_positions.clear()
        peak_millers.clear()
        peak_intensities.clear()
        peak_records.clear()
        selected_peak_record = None
        image_generation_cached = False
        image_generation_start_time = perf_counter()

        def run_one(data, intens_arr, a_val, c_val):
            buf = np.zeros((image_size, image_size), dtype=np.float64)
            if isinstance(data, dict):
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
                )
            else:
                miller_arr = data
                if DEBUG_ENABLED:
                    debug_print("process_peaks_parallel with", miller_arr.shape[0], "reflections")
                    if not np.all(np.isfinite(miller_arr)):
                        debug_print("Non-finite miller indices detected")
                    if not np.all(np.isfinite(intens_arr)):
                        debug_print("Non-finite intensities detected")
                return process_peaks_parallel(
                    miller_arr,
                    intens_arr,
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
                ) + (None,)

        img1, maxpos1, _, _, _, _, _ = run_one(SIM_MILLER1, SIM_INTENS1, a_updated, c_updated)
        if SIM_MILLER2.size > 0:
            img2, maxpos2, _, _, _, _, _ = run_one(SIM_MILLER2, SIM_INTENS2, av2, cv2)
        else:
            img2 = np.zeros_like(img1)
            maxpos2 = []

        w1 = weight1_var.get()
        w2 = weight2_var.get()
        updated_image = w1 * img1 + w2 * img2
        max_positions_local = list(maxpos1) + list(maxpos2)
        peak_table_lattice_local = [
            (float(a_updated), float(c_updated), "primary")
            for _ in maxpos1
        ]
        if maxpos2:
            sec_a = float(av2) if av2 is not None else float(a_updated)
            sec_c = float(cv2) if cv2 is not None else float(c_updated)
            peak_table_lattice_local.extend(
                (sec_a, sec_c, "secondary")
                for _ in maxpos2
            )
        stored_max_positions_local = max_positions_local
        stored_peak_table_lattice = peak_table_lattice_local
        stored_sim_image = updated_image
        image_generation_elapsed_ms = (
            perf_counter() - image_generation_start_time
        ) * 1e3
    else:
        # fall back to the cached arrays
        if stored_max_positions_local is None:
            # first run after programme start – force a simulation
            last_simulation_signature = None
            update_running = False
            return do_update()          # re-enter with computation path
        max_positions_local = stored_max_positions_local
        peak_table_lattice_local = stored_peak_table_lattice
        updated_image       = stored_sim_image

    if not peak_table_lattice_local or len(peak_table_lattice_local) != len(max_positions_local):
        peak_table_lattice_local = [
            (float(a_updated), float(c_updated), "primary")
            for _ in max_positions_local
        ]

    redraw_update_start_time = perf_counter()
    display_image = np.rot90(updated_image, SIM_DISPLAY_ROTATE_K)
    
    # ───── NEW: build peak lists from hit_tables ───────────────────────────
    peak_positions.clear()
    peak_millers.clear()
    peak_intensities.clear()
    peak_records.clear()

    max_hits_per_reflection = max(1, int(HKL_PICK_MAX_HITS_PER_REFLECTION))
    min_sep_sq = float(HKL_PICK_MIN_SEPARATION_PX) ** 2

    # hit_tables is a numba.typed.List of 2-D arrays, one per reflection
    for table_idx, tbl in enumerate(max_positions_local):
        tbl_arr = np.asarray(tbl, dtype=float)
        if tbl_arr.ndim != 2 or tbl_arr.shape[0] == 0:
            continue
        av_used, cv_used, source_label = peak_table_lattice_local[table_idx]
        row_order = np.argsort(tbl_arr[:, 0])[::-1]
        chosen_rows: list[tuple[int, np.ndarray, int, int, float, float]] = []

        # Keep only the strongest distinct hit locations per reflection.
        for row_idx in row_order:
            row = tbl_arr[row_idx]
            I, xpix, ypix, _phi, H, K, L = row
            if not (np.isfinite(I) and np.isfinite(xpix) and np.isfinite(ypix)):
                continue
            cx = int(round(xpix))
            cy = int(round(ypix))
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
            if len(chosen_rows) >= max_hits_per_reflection:
                break

        for row_idx, row, cx, cy, disp_cx, disp_cy in chosen_rows:
            I, _xpix, _ypix, _phi, H, K, L = row
            peak_positions.append((disp_cx, disp_cy))      # display coords
            peak_intensities.append(float(I))
            hkl = tuple(int(np.rint(val)) for val in (H, K, L))
            hkl_raw = (float(H), float(K), float(L))
            m_val = float(H * H + H * K + K * K)
            qr_val = (
                (2.0 * np.pi / float(av_used)) * np.sqrt((4.0 / 3.0) * m_val)
                if float(av_used) > 0.0 and np.isfinite(float(av_used)) and m_val >= 0.0
                else float("nan")
            )
            peak_millers.append(hkl)
            peak_records.append(
                {
                    "display_col": float(disp_cx),
                    "display_row": float(disp_cy),
                    "native_col": float(cx),
                    "native_row": float(cy),
                    "hkl": hkl,
                    "hkl_raw": hkl_raw,
                    "intensity": float(I),
                    "qr": float(qr_val),
                    "phi": float(_phi),
                    "source_table_index": int(table_idx),
                    "source_row_index": int(row_idx),
                    "source_label": str(source_label),
                    "av": float(av_used),
                    "cv": float(cv_used),
                }
            )

    if selected_hkl_target is not None:
        _select_peak_by_hkl(
            selected_hkl_target[0],
            selected_hkl_target[1],
            selected_hkl_target[2],
            sync_hkl_vars=False,
            silent_if_missing=True,
        )

    normalization_scale = 1.0
    native_background = _get_current_background_backend()
    if native_background is not None and display_image is not None:
        normalization_scale = _suggest_scale_factor(
            display_image, native_background
        )
        if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
            normalization_scale = 1.0

    unscaled_image_global = None
    if display_image is not None:
        unscaled_image_global = display_image * normalization_scale
        if peak_intensities and normalization_scale != 1.0:
            peak_intensities[:] = [
                intensity * normalization_scale for intensity in peak_intensities
            ]
            for rec in peak_records:
                rec["intensity"] = float(rec.get("intensity", 0.0)) * normalization_scale

    last_1d_integration_data["simulated_2d_image"] = unscaled_image_global

    if unscaled_image_global is not None:
        if scale_factor_user_override:
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
    global _ai_cache
    sig = (
        corto_det_up,
        center_x_up,
        center_y_up,
        Gamma_updated,
        gamma_updated,
        wave_m,
    )
    if _ai_cache.get("sig") != sig:
        _ai_cache = {
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
    ai = _ai_cache["ai"]

    # Caked 2D or normal 2D?
    sim_res2 = None
    if show_caked_2d_var.get() and unscaled_image_global is not None:
        sim_res2 = caking(unscaled_image_global, ai)
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

        last_caked_image_unscaled = caked_img

        current_scale = _get_scale_factor_value(default=1.0)
        scaled_caked_for_limits = caked_img * current_scale
        auto_vmin, auto_vmax = _auto_caked_limits(scaled_caked_for_limits)

        if not simulation_limits_user_override:
            _update_simulation_sliders_from_image(
                scaled_caked_for_limits, reset_override=True
            )

        if not caked_limits_user_override:
            vmin_caked_var.set(auto_vmin)
            vmax_caked_var.set(auto_vmax)

        vmin_val = float(simulation_min_var.get())
        vmax_val = float(simulation_max_var.get())
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
        native_background = _get_current_background_backend()
        if background_visible and native_background is not None:
            bg_res2 = caking(native_background, ai)
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

        last_caked_extent = [
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
        ax.set_xlim(0.0, 90.0)
        ax.set_ylim(-180.0, 180.0)
        ax.set_xlabel('2θ (degrees)')
        ax.set_ylabel('φ (degrees)')
        ax.set_title('2D Caked Integration')
    else:
        last_caked_image_unscaled = None
        last_caked_extent = None
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Simulated Diffraction Pattern')

        _set_image_origin(background_display, 'upper')
        background_display.set_extent([0, image_size, image_size, 0])
        if background_visible and current_background_display is not None:
            background_display.set_data(current_background_display)
            background_display.set_clim(
                background_min_var.get(),
                background_max_var.get(),
            )
            background_display.set_visible(True)
        else:
            background_display.set_visible(False)
        
    # 1D integration
    if show_1d_var.get() and unscaled_image_global is not None:
        if sim_res2 is None:
            sim_res2 = caking(unscaled_image_global, ai)
        i2t_sim, i_phi_sim, az_sim, rad_sim = caked_up(
            sim_res2,
            tth_min_var.get(),
            tth_max_var.get(),
            phi_min_var.get(),
            phi_max_var.get()
        )
        last_1d_integration_data["radials_sim"] = rad_sim
        last_1d_integration_data["intensities_2theta_sim"] = i2t_sim
        last_1d_integration_data["azimuths_sim"] = az_sim
        last_1d_integration_data["intensities_azimuth_sim"] = i_phi_sim

        scale = _get_scale_factor_value(default=1.0)
        line_1d_rad.set_data(rad_sim, i2t_sim * scale)
        line_1d_az.set_data(az_sim, i_phi_sim * scale)

        native_background = _get_current_background_backend()
        if background_visible and native_background is not None:
            bg_res2 = caking(native_background, ai)
            i2t_bg, i_phi_bg, az_bg, rad_bg = caked_up(
                bg_res2,
                tth_min_var.get(),
                tth_max_var.get(),
                phi_min_var.get(),
                phi_max_var.get()
            )
            last_1d_integration_data["radials_bg"] = rad_bg
            last_1d_integration_data["intensities_2theta_bg"] = i2t_bg
            last_1d_integration_data["azimuths_bg"] = az_bg
            last_1d_integration_data["intensities_azimuth_bg"] = i_phi_bg
            line_1d_rad_bg.set_data(rad_bg, i2t_bg)
            line_1d_az_bg.set_data(az_bg, i_phi_bg)
        else:
            line_1d_rad_bg.set_data([], [])
            line_1d_az_bg.set_data([], [])
            last_1d_integration_data["radials_bg"] = None
            last_1d_integration_data["intensities_2theta_bg"] = None
            last_1d_integration_data["azimuths_bg"] = None
            last_1d_integration_data["intensities_azimuth_bg"] = None

        ax_1d_radial.set_yscale('log' if log_radial_var.get() else 'linear')
        ax_1d_azim.set_yscale('log' if log_azimuth_var.get() else 'linear')

        ax_1d_radial.relim()
        ax_1d_radial.autoscale_view()
        ax_1d_azim.relim()
        ax_1d_azim.autoscale_view()
        canvas_1d.draw_idle()
    else:
        line_1d_rad.set_data([], [])
        line_1d_az.set_data([], [])
        line_1d_rad_bg.set_data([], [])
        line_1d_az_bg.set_data([], [])
        canvas_1d.draw_idle()
        last_1d_integration_data["radials_sim"] = None
        last_1d_integration_data["intensities_2theta_sim"] = None
        last_1d_integration_data["azimuths_sim"] = None
        last_1d_integration_data["intensities_azimuth_sim"] = None
        last_1d_integration_data["radials_bg"] = None
        last_1d_integration_data["intensities_2theta_bg"] = None
        last_1d_integration_data["azimuths_bg"] = None
        last_1d_integration_data["intensities_azimuth_bg"] = None

    # Do not auto-rescale simulation display limits on every update.
    apply_scale_factor_to_existing_results(update_limits=False)

    update_integration_region_visuals(ai, sim_res2)

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
    update_running = False


do_update = _wrap_with_responsiveness_indicator(do_update)

# ── after you’ve updated background_visible in toggle_background() ──
def toggle_background():
    global background_visible
    background_visible = not background_visible
    # ↓ force opaque if the background is hidden, 0.5 otherwise
    image_display.set_alpha(0.5 if background_visible else 1.0)
    schedule_update()


def switch_background():
    global current_background_index, current_background_image, current_background_display
    current_background_index = (current_background_index + 1) % len(background_images_native)
    current_background_image = background_images_native[current_background_index]
    current_background_display = background_images_display[current_background_index]
    background_display.set_data(current_background_display)
    schedule_update()

def reset_to_defaults():
    global caked_limits_user_override
    global simulation_limits_user_override, background_limits_user_override
    global scale_factor_user_override, suppress_simulation_limit_callback
    theta_initial_var.set(defaults['theta_initial'])
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
    a_var.set(defaults['a'])
    c_var.set(defaults['c'])
    resolution_var.set(defaults['sampling_resolution'])
    optics_mode_var.set(_normalize_optics_mode_label(defaults.get('optics_mode', 'fast')))
    center_x_var.set(defaults['center_x'])
    center_y_var.set(defaults['center_y'])
    tth_min_var.set(0.0)
    tth_max_var.set(80.0)
    phi_min_var.set(-15.0)
    phi_max_var.set(15.0)
    show_1d_var.set(False)
    show_caked_2d_var.set(False)
    vmin_caked_var.set(0.0)
    vmax_caked_var.set(2000.0)
    caked_limits_user_override = False

    background_limits_user_override = False
    simulation_limits_user_override = False
    scale_factor_user_override = False

    _update_background_slider_defaults(current_background_display, reset_override=True)

    suppress_simulation_limit_callback = True
    simulation_min_var.set(0.0)
    simulation_max_var.set(background_vmax_default)
    suppress_simulation_limit_callback = False

    _set_scale_factor_value(1.0, adjust_range=False, reset_override=True)

    # ALSO reset occupancies to default
    occ_var1.set(occ[0])
    occ_var2.set(occ[1])
    occ_var3.set(occ[2])
    p0_var.set(defaults['p0'])
    p1_var.set(defaults['p1'])
    p2_var.set(defaults['p2'])
    w0_var.set(defaults['w0'])
    w1_var.set(defaults['w1'])
    w2_var.set(defaults['w2'])
    finite_stack_var.set(defaults['finite_stack'])
    stack_layers_var.set(int(defaults['stack_layers']))
    _sync_finite_controls()

    update_mosaic_cache()
    global last_simulation_signature
    last_simulation_signature = None
    schedule_update()

toggle_button = ttk.Button(
    text="Toggle Background",
    command=toggle_background
)
toggle_button.pack(side=tk.TOP, padx=5, pady=2)

switch_button = ttk.Button(
    text="Switch Background",
    command=switch_background
)
switch_button.pack(side=tk.TOP, padx=5, pady=2)

reset_button_top = ttk.Button(
    text="Reset to Defaults",
    command=reset_to_defaults
)
reset_button_top.pack(side=tk.TOP, padx=5, pady=2)


def open_azimuthal_radial_view():
    view_azimuthal_radial(
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
            num_samples=num_samples,
            divergence_sigma=divergence_sigma,
            bw_sigma=bw_sigma,
            sigma_mosaic_var=sigma_mosaic_var,
            gamma_mosaic_var=gamma_mosaic_var,
            eta_var=eta_var,
            optics_mode=_current_optics_mode_flag(),
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
    )


open_azimuthal_radial_view = _wrap_with_responsiveness_indicator(
    open_azimuthal_radial_view
)

azimuthal_button = ttk.Button(
    text="Azim vs Radial Plot Demo",
    command=open_azimuthal_radial_view
)
azimuthal_button.pack(side=tk.TOP, padx=5, pady=2)

def _compact_status_text(text: object, *, max_chars: int = 120) -> str:
    summary = " ".join(str(text).split())
    if len(summary) > max_chars:
        return summary[: max_chars - 1] + "..."
    return summary


class ConsoleStatusLabel:
    """Mirror verbose GUI status messages to the terminal and keep GUI text compact."""

    def __init__(
        self,
        parent,
        *,
        name: str,
        max_gui_chars: int = 120,
        **label_kwargs,
    ) -> None:
        self._name = str(name)
        self._max_gui_chars = max(16, int(max_gui_chars))
        self._last_full_text = ""
        self._label = ttk.Label(parent, wraplength=0, justify=tk.LEFT, anchor=tk.W, **label_kwargs)

    def config(self, cnf=None, **kwargs):
        options = {}
        if isinstance(cnf, dict):
            options.update(cnf)
        options.update(kwargs)
        if "text" in options:
            raw_text = "" if options["text"] is None else str(options["text"])
            if raw_text != self._last_full_text:
                if raw_text.strip():
                    print(f"[{self._name}] {raw_text}", flush=True)
                self._last_full_text = raw_text
            options["text"] = _compact_status_text(
                raw_text,
                max_chars=self._max_gui_chars,
            )
            options.pop("wraplength", None)
        return self._label.config(**options)

    configure = config

    def cget(self, key):
        return self._label.cget(key)

    def __getattr__(self, name):
        return getattr(self._label, name)


progress_label_positions = ConsoleStatusLabel(
    root,
    name="positions",
    max_gui_chars=110,
)
progress_label_positions.pack(side=tk.BOTTOM, padx=5)

progress_label_geometry = ConsoleStatusLabel(
    root,
    name="geometry",
    max_gui_chars=110,
)
progress_label_geometry.pack(side=tk.BOTTOM, padx=5)

mosaic_progressbar = ttk.Progressbar(root, mode="indeterminate", length=240)
mosaic_progressbar.pack(side=tk.BOTTOM, padx=5, pady=(0, 2))

progress_label_mosaic = ConsoleStatusLabel(
    root,
    name="mosaic",
    max_gui_chars=110,
)
progress_label_mosaic.pack(side=tk.BOTTOM, padx=5)

progress_label = ConsoleStatusLabel(
    root,
    name="gui",
    max_gui_chars=110,
    font=("Helvetica", 8),
)
progress_label.pack(side=tk.BOTTOM, padx=5)

update_timing_label = ttk.Label(
    root,
    text="Timing | image generation: n/a | redraw/update: n/a | total: n/a",
    font=("Helvetica", 8),
)
update_timing_label.pack(side=tk.BOTTOM, padx=5)

chi_square_label = ttk.Label(root, text="Chi-Squared: ", font=("Helvetica", 8))
chi_square_label.pack(side=tk.BOTTOM, padx=5)

responsiveness_indicator_container = ttk.Frame(root)
responsiveness_indicator_container.pack(side=tk.BOTTOM, fill=tk.X)
responsiveness_indicator_block = tk.Frame(
    responsiveness_indicator_container,
    width=RESPONSIVE_STATUS_BLOCK_SIZE,
    height=RESPONSIVE_STATUS_BLOCK_SIZE,
    bg=RESPONSIVE_STATUS_BUSY_COLOR,
    highlightthickness=1,
    highlightbackground="#1f1f1f",
    highlightcolor="#1f1f1f",
)
_raise_responsiveness_indicator_block()
if _gui_busy_depth == 0:
    _set_responsiveness_indicator_color(RESPONSIVE_STATUS_READY_COLOR)

last_hbn_geometry_debug_report = (
    "No hBN geometry debug report yet.\n"
    "Import an hBN bundle to generate one."
)


def show_last_hbn_geometry_debug():
    """Display the most recent hBN->simulation geometry debug report."""

    win = tk.Toplevel(root)
    win.title("hBN Geometry Debug")
    win.geometry("980x560")

    frame = ttk.Frame(win, padding=8)
    frame.pack(fill=tk.BOTH, expand=True)

    text = tk.Text(frame, wrap=tk.NONE)
    text.grid(row=0, column=0, sticky="nsew")
    y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
    x_scroll.grid(row=1, column=0, sticky="ew")
    text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    text.insert("1.0", str(last_hbn_geometry_debug_report))
    text.configure(state=tk.DISABLED)


def import_hbn_tilt_from_bundle():
    """Load an hBN ellipse bundle NPZ and apply its tilt hint to the GUI sliders."""

    bundle_path = filedialog.askopenfilename(
        title="Select hBN bundle (.npz)",
        filetypes=[("hBN bundle", "*.npz"), ("All files", "*.*")],
    )
    if not bundle_path:
        return

    global last_hbn_geometry_debug_report
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
        (
            _,
            _,
            _,
            _,
            distance_info,
            tilt_correction,
            tilt_hint,
            _,
            imported_center,
        ) = _run_with_responsiveness_indicator(load_bundle_npz, bundle_path)
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
        last_hbn_geometry_debug_report = format_hbn_geometry_debug_trace(debug_trace)
        print(last_hbn_geometry_debug_report)
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


_GUI_STATE_EXCLUDED_VAR_NAMES = {
    "geometry_preview_exclude_button_var",
    "hkl_pick_button_var",
    "resolution_count_var",
}


def _is_persistable_gui_var(name: str, value: object) -> bool:
    if not isinstance(value, tk.Variable):
        return False
    if name.startswith("_") or name in _GUI_STATE_EXCLUDED_VAR_NAMES:
        return False
    if name.endswith(("_button_var", "_entry_var", "_label_var", "_status_var")):
        return False
    return True


def _canonicalize_gui_state_background_path(path: object) -> str:
    return os.path.normcase(str(Path(str(path)).expanduser().resolve(strict=False)))


def _background_files_match_loaded_state(file_paths: list[str]) -> bool:
    if not file_paths:
        return False
    if len(file_paths) != len(osc_files):
        return False
    if len(background_images_native) != len(file_paths):
        return False
    if len(background_images_display) != len(file_paths):
        return False
    requested_paths = [
        _canonicalize_gui_state_background_path(path) for path in file_paths
    ]
    current_paths = [
        _canonicalize_gui_state_background_path(path) for path in osc_files
    ]
    return requested_paths == current_paths


def _load_background_files_for_state(
    file_paths: list[str],
    *,
    select_index: int = 0,
) -> None:
    global osc_files, background_images, background_images_native, background_images_display
    global current_background_index, current_background_image, current_background_display

    normalized_paths = [
        str(Path(str(path)).expanduser())
        for path in file_paths
        if path is not None
    ]
    if not normalized_paths:
        return
    if _background_files_match_loaded_state(normalized_paths):
        osc_files = list(normalized_paths)
        index = max(0, min(int(select_index), len(background_images_native) - 1))
        current_background_index = index
        current_background_image = background_images_native[index]
        current_background_display = background_images_display[index]
        background_display.set_data(current_background_display)
        return

    loaded_native = [np.asarray(read_osc(path)) for path in normalized_paths]
    if not loaded_native:
        return
    osc_files = list(normalized_paths)
    background_images = [np.array(img) for img in loaded_native]
    background_images_native = [np.array(img) for img in loaded_native]
    background_images_display = [
        np.rot90(img, DISPLAY_ROTATE_K) for img in background_images_native
    ]
    index = max(0, min(int(select_index), len(background_images_native) - 1))
    current_background_index = index
    current_background_image = background_images_native[index]
    current_background_display = background_images_display[index]
    background_display.set_data(current_background_display)


_load_background_files_for_state = _wrap_with_responsiveness_indicator(
    _load_background_files_for_state
)


def _collect_full_gui_state_snapshot() -> dict[str, object]:
    variables: dict[str, object] = {}
    for name, value in globals().items():
        if not _is_persistable_gui_var(name, value):
            continue
        try:
            variables[name] = value.get()
        except Exception:
            continue

    geometry_state: dict[str, object] = {}
    if isinstance(selected_hkl_target, tuple) and len(selected_hkl_target) == 3:
        geometry_state["selected_hkl_target"] = [
            int(selected_hkl_target[0]),
            int(selected_hkl_target[1]),
            int(selected_hkl_target[2]),
        ]

    return {
        "variables": variables,
        "files": {
            "background_files": [str(Path(str(path)).expanduser()) for path in osc_files],
            "current_background_index": int(current_background_index),
        },
        "flags": {
            "background_visible": bool(background_visible),
            "background_backend_rotation_k": int(background_backend_rotation_k),
            "background_backend_flip_x": bool(background_backend_flip_x),
            "background_backend_flip_y": bool(background_backend_flip_y),
            "background_limits_user_override": bool(background_limits_user_override),
            "simulation_limits_user_override": bool(simulation_limits_user_override),
            "scale_factor_user_override": bool(scale_factor_user_override),
        },
        "geometry": geometry_state,
    }


def _apply_full_gui_state_snapshot(snapshot: dict[str, object]) -> str:
    global background_visible
    global background_backend_rotation_k, background_backend_flip_x, background_backend_flip_y
    global background_limits_user_override, simulation_limits_user_override
    global caked_limits_user_override
    global scale_factor_user_override, selected_hkl_target

    warnings: list[str] = []

    files = snapshot.get("files", {})
    if isinstance(files, dict):
        raw_background_paths = files.get("background_files", [])
        background_paths = []
        if isinstance(raw_background_paths, list):
            for raw_path in raw_background_paths:
                if raw_path is None:
                    continue
                candidate = Path(str(raw_path)).expanduser()
                if candidate.is_file():
                    background_paths.append(str(candidate))
                else:
                    warnings.append(f"background missing: {candidate}")
        if background_paths:
            try:
                _load_background_files_for_state(
                    background_paths,
                    select_index=int(files.get("current_background_index", 0)),
                )
            except Exception as exc:
                warnings.append(f"backgrounds: {exc}")

    variables = snapshot.get("variables", {})
    if isinstance(variables, dict):
        for name, stored_value in variables.items():
            target_var = globals().get(str(name))
            if not isinstance(target_var, tk.Variable):
                continue
            try:
                if target_var.get() == stored_value:
                    continue
            except Exception:
                pass
            try:
                target_var.set(stored_value)
            except Exception as exc:
                warnings.append(f"{name}: {exc}")

    flags = snapshot.get("flags", {})
    if isinstance(flags, dict):
        desired_background_visible = bool(flags.get("background_visible", background_visible))
        if desired_background_visible != bool(background_visible):
            background_visible = desired_background_visible
            image_display.set_alpha(0.5 if background_visible else 1.0)
        background_backend_rotation_k = int(flags.get("background_backend_rotation_k", background_backend_rotation_k)) % 4
        background_backend_flip_x = bool(flags.get("background_backend_flip_x", background_backend_flip_x))
        background_backend_flip_y = bool(flags.get("background_backend_flip_y", background_backend_flip_y))
        background_limits_user_override = bool(
            flags.get("background_limits_user_override", background_limits_user_override)
        )
        simulation_limits_user_override = bool(
            flags.get("simulation_limits_user_override", simulation_limits_user_override)
        )
        scale_factor_user_override = bool(
            flags.get("scale_factor_user_override", scale_factor_user_override)
        )

    geometry_state = snapshot.get("geometry", {})
    if isinstance(geometry_state, dict):
        target_hkl = geometry_state.get("selected_hkl_target")
        if isinstance(target_hkl, (list, tuple)) and len(target_hkl) >= 3:
            try:
                selected_hkl_target = (
                    int(target_hkl[0]),
                    int(target_hkl[1]),
                    int(target_hkl[2]),
                )
            except Exception:
                selected_hkl_target = None

    _sync_finite_controls()
    ensure_valid_resolution_choice()
    if not show_caked_2d_var.get():
        caked_limits_user_override = False
    else:
        simulation_limits_user_override = False
    _update_background_backend_status()
    _update_chi_square_display()
    schedule_update()

    summary = "Imported GUI state snapshot."
    if warnings:
        summary += " Warnings: " + "; ".join(warnings[:4])
        if len(warnings) > 4:
            summary += f"; +{len(warnings) - 4} more"
    return summary


_apply_full_gui_state_snapshot = _wrap_with_responsiveness_indicator(
    _apply_full_gui_state_snapshot
)


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
            metadata={"entrypoint": "ra_sim.gui.app"},
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

    if Path(file_path).suffix.lower() == ".npy":
        _mark_gui_busy()
        try:
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
                None,
                optics_mode_var=optics_mode_var,
            )
            ensure_valid_resolution_choice()
            schedule_update()
        finally:
            _mark_gui_ready()
        progress_label.config(text=message)
        return

    _mark_gui_busy()
    try:
        payload = load_gui_state_file(file_path)
        message = _apply_full_gui_state_snapshot(payload.get("state", {}))
    except Exception as exc:
        progress_label.config(text=f"Failed to import GUI state: {exc}")
        return
    finally:
        _mark_gui_ready()
    progress_label.config(text=message)


save_button = ttk.Button(
    text="Export GUI State...",
    command=_export_full_gui_state,
)
save_button.pack(side=tk.TOP, padx=5, pady=2)

load_button = ttk.Button(
    text="Import GUI State...",
    command=_import_full_gui_state,
)
load_button.pack(side=tk.TOP, padx=5, pady=2)

import_hbn_button = ttk.Button(
    text="Import hBN Bundle Tilt",
    command=import_hbn_tilt_from_bundle,
)
import_hbn_button.pack(side=tk.TOP, padx=5, pady=2)

if HBN_GEOMETRY_DEBUG_ENABLED:
    show_hbn_debug_button = ttk.Button(
        text="Show hBN Geometry Debug",
        command=show_last_hbn_geometry_debug,
    )
    show_hbn_debug_button.pack(side=tk.TOP, padx=5, pady=2)

# Frame for selecting which geometry params to fit
fit_frame = ttk.LabelFrame(root, text="Fit geometry parameters")
fit_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

fit_zb_var    = tk.BooleanVar(value=True)
fit_zs_var    = tk.BooleanVar(value=True)
fit_theta_var = tk.BooleanVar(value=True)  # theta_initial
fit_psi_z_var = tk.BooleanVar(value=True)
fit_chi_var   = tk.BooleanVar(value=True)
fit_cor_var   = tk.BooleanVar(value=True)
fit_gamma_var = tk.BooleanVar(value=True)
fit_Gamma_var = tk.BooleanVar(value=True)
fit_corto_var = tk.BooleanVar(value=True)
fit_a_var = tk.BooleanVar(value=False)
fit_c_var = tk.BooleanVar(value=False)
fit_center_x_var = tk.BooleanVar(value=False)
fit_center_y_var = tk.BooleanVar(value=False)

ttk.Checkbutton(fit_frame, text="zb",    variable=fit_zb_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="zs",    variable=fit_zs_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="theta", variable=fit_theta_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(
    fit_frame,
    text="Goniometer Axis Yaw (about z)",
    variable=fit_psi_z_var,
).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="chi",   variable=fit_chi_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(
    fit_frame,
    text="Goniometer Axis Pitch (about y)",
    variable=fit_cor_var,
).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="gamma", variable=fit_gamma_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="Gamma", variable=fit_Gamma_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="Corto", variable=fit_corto_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="a lattice", variable=fit_a_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="c lattice", variable=fit_c_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="center row", variable=fit_center_x_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="center col", variable=fit_center_y_var).pack(side=tk.LEFT, padx=2)

GEOMETRY_FIT_PARAM_ORDER = [
    "zb",
    "zs",
    "theta_initial",
    "psi_z",
    "chi",
    "cor_angle",
    "gamma",
    "Gamma",
    "corto_detector",
    "a",
    "c",
    "center_x",
    "center_y",
]
geometry_fit_toggle_vars = {
    "zb": fit_zb_var,
    "zs": fit_zs_var,
    "theta_initial": fit_theta_var,
    "psi_z": fit_psi_z_var,
    "chi": fit_chi_var,
    "cor_angle": fit_cor_var,
    "gamma": fit_gamma_var,
    "Gamma": fit_Gamma_var,
    "corto_detector": fit_corto_var,
    "a": fit_a_var,
    "c": fit_c_var,
    "center_x": fit_center_x_var,
    "center_y": fit_center_y_var,
}
geometry_fit_parameter_specs = {}
geometry_fit_constraint_controls = {}


def _sync_geometry_fit_constraint_rows(*_args) -> None:
    for name in GEOMETRY_FIT_PARAM_ORDER:
        control = geometry_fit_constraint_controls.get(name)
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
    background_backend_frame = ttk.LabelFrame(root, text="Background Backend (debug)")
    background_backend_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    background_backend_status_label = ttk.Label(
        background_backend_frame,
        text=_background_backend_status(),
    )
    background_backend_status_label.pack(side=tk.LEFT, padx=4)

    ttk.Button(
        background_backend_frame,
        text="Rot -90",
        command=lambda: _rotate_background_backend(-1),
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        background_backend_frame,
        text="Rot +90",
        command=lambda: _rotate_background_backend(1),
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        background_backend_frame,
        text="Flip X",
        command=lambda: _toggle_background_backend_flip("x"),
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        background_backend_frame,
        text="Flip Y",
        command=lambda: _toggle_background_backend_flip("y"),
    ).pack(side=tk.LEFT, padx=2)
    ttk.Button(
        background_backend_frame,
        text="Reset",
        command=_reset_background_backend_orientation,
    ).pack(side=tk.LEFT, padx=2)

backend_rotation_var = tk.IntVar(value=0)
backend_flip_y_axis_var = tk.BooleanVar(value=False)
backend_flip_x_axis_var = tk.BooleanVar(value=False)
backend_flip_order_var = tk.StringVar(value="yx")

if BACKEND_ORIENTATION_UI_ENABLED:
    backend_orient_frame = ttk.LabelFrame(root, text="Backend orientation (debug)")
    backend_orient_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    ttk.Label(backend_orient_frame, text="Rotate ×90° (k):").pack(side=tk.LEFT, padx=2)
    tk.Spinbox(
        backend_orient_frame,
        from_=-3,
        to=3,
        width=4,
        textvariable=backend_rotation_var,
    ).pack(side=tk.LEFT, padx=2)

    ttk.Checkbutton(
        backend_orient_frame,
        text="Flip about y-axis",
        variable=backend_flip_y_axis_var,
    ).pack(side=tk.LEFT, padx=2)

    ttk.Checkbutton(
        backend_orient_frame,
        text="Flip about x-axis",
        variable=backend_flip_x_axis_var,
    ).pack(side=tk.LEFT, padx=2)

    ttk.Label(backend_orient_frame, text="Flip order:").pack(side=tk.LEFT, padx=2)
    tk.OptionMenu(
        backend_orient_frame,
        backend_flip_order_var,
        "yx",
        "yx",
        "xy",
    ).pack(side=tk.LEFT, padx=2)

    ttk.Button(
        backend_orient_frame,
        text="Reset",  # return to no rotation/flip
        command=lambda: (
            backend_rotation_var.set(0),
            backend_flip_y_axis_var.set(False),
            backend_flip_x_axis_var.set(False),
            backend_flip_order_var.set("yx"),
        ),
    ).pack(side=tk.LEFT, padx=4)


def _manual_orientation_choice() -> dict:
    """Return the orientation selected in the debug orientation controls."""

    try:
        k = int(backend_rotation_var.get())
    except (TypeError, ValueError, tk.TclError):
        k = 0

    flip_x = bool(backend_flip_x_axis_var.get())
    flip_y = bool(backend_flip_y_axis_var.get())
    flip_order = str(backend_flip_order_var.get()).lower()
    if flip_order not in {"xy", "yx"}:
        flip_order = "yx"

    return {
        "k": k,
        "flip_x": flip_x,
        "flip_y": flip_y,
        "flip_order": flip_order,
        "indexing_mode": "xy",
        "label": f"manual(k={k}, flip_x={flip_x}, flip_y={flip_y}, order={flip_order})",
    }


def _extract_simulated_fit_peaks(
    hit_tables: Sequence[object],
    image_shape: tuple[int, int],
    original_table_indices: np.ndarray,
) -> list[dict[str, object]]:
    """Extract distinct simulated detector spots while preserving source rows."""

    max_hits_per_reflection = max(1, int(HKL_PICK_MAX_HITS_PER_REFLECTION))
    min_sep_sq = float(HKL_PICK_MIN_SEPARATION_PX) ** 2
    simulated_peaks: list[dict[str, object]] = []

    for local_table_idx, tbl in enumerate(hit_tables):
        tbl_arr = np.asarray(tbl, dtype=float)
        if tbl_arr.ndim != 2 or tbl_arr.shape[0] == 0:
            continue

        row_order = np.argsort(tbl_arr[:, 0])[::-1]
        chosen_rows: list[tuple[int, np.ndarray, float, float]] = []
        for row_idx in row_order:
            row = tbl_arr[row_idx]
            intensity, sim_col, sim_row, _phi, *_ = row
            if not (
                np.isfinite(intensity)
                and np.isfinite(sim_col)
                and np.isfinite(sim_row)
            ):
                continue
            disp_col, disp_row = _native_sim_to_display_coords(
                float(sim_col),
                float(sim_row),
                image_shape,
            )
            too_close = False
            for _, _, prev_col, prev_row in chosen_rows:
                d2 = (disp_col - prev_col) ** 2 + (disp_row - prev_row) ** 2
                if d2 < min_sep_sq:
                    too_close = True
                    break
            if too_close:
                continue
            chosen_rows.append((int(row_idx), row, float(disp_col), float(disp_row)))
            if len(chosen_rows) >= max_hits_per_reflection:
                break

        original_table_idx = (
            int(original_table_indices[local_table_idx])
            if local_table_idx < int(original_table_indices.shape[0])
            else int(local_table_idx)
        )
        for row_idx, row, _disp_col, _disp_row in chosen_rows:
            intensity, sim_col, sim_row, phi_val, H, K, L = row
            hkl = tuple(int(np.rint(val)) for val in (H, K, L))
            simulated_peaks.append(
                {
                    "hkl": hkl,
                    "label": f"{hkl[0]},{hkl[1]},{hkl[2]}",
                    "sim_col": float(sim_col),
                    "sim_row": float(sim_row),
                    "weight": float(abs(intensity)),
                    "source_table_index": int(original_table_idx),
                    "source_row_index": int(row_idx),
                    "phi": float(phi_val),
                }
            )
    return simulated_peaks


def _simulate_hkl_peak_centers_for_fit(
    miller_array: np.ndarray,
    intensity_array: np.ndarray,
    image_size: int,
    param_set: dict[str, object],
    *,
    measured_peaks: Sequence[object] | None = None,
    force_single_ray: bool = True,
) -> list[dict[str, object]]:
    """Simulate detector spots for geometry auto-match/rematch."""

    sim_subset = _prepare_reflection_subset(
        np.asarray(miller_array, dtype=np.float64),
        np.asarray(intensity_array, dtype=np.float64),
        measured_peaks,
    )
    sim_miller = sim_subset.miller
    sim_intensity = sim_subset.intensities
    original_table_indices = sim_subset.original_indices

    mosaic = dict(param_set.get("mosaic_params", {}))
    wavelength_array = mosaic.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic.get("wavelength_i_array")
    if wavelength_array is None:
        wavelength_array = np.full(
            int(np.size(mosaic.get("beam_x_array", []))),
            float(param_set.get("lambda", 1.0)),
            dtype=np.float64,
        )

    if sim_miller.ndim != 2 or sim_miller.shape[1] != 3 or sim_miller.shape[0] == 0:
        return []

    beam_x = np.asarray(mosaic["beam_x_array"], dtype=np.float64)
    beam_y = np.asarray(mosaic["beam_y_array"], dtype=np.float64)
    theta_arr = np.asarray(mosaic["theta_array"], dtype=np.float64)
    phi_arr = np.asarray(mosaic["phi_array"], dtype=np.float64)
    wavelength_arr = np.asarray(wavelength_array, dtype=np.float64)
    center = [float(param_set["center"][0]), float(param_set["center"][1])]

    def _run_preview_sim(
        *,
        single_sample_indices: np.ndarray | None = None,
        best_sample_indices_out: np.ndarray | None = None,
        collect_hit_tables: bool = True,
    ):
        sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
        return process_peaks_parallel(
            sim_miller,
            sim_intensity,
            image_size,
            float(param_set["a"]),
            float(param_set["c"]),
            wavelength_arr,
            sim_buffer,
            float(param_set["corto_detector"]),
            float(param_set["gamma"]),
            float(param_set["Gamma"]),
            float(param_set["chi"]),
            float(param_set.get("psi", 0.0)),
            float(param_set.get("psi_z", 0.0)),
            float(param_set["zs"]),
            float(param_set["zb"]),
            param_set["n2"],
            beam_x,
            beam_y,
            theta_arr,
            phi_arr,
            float(mosaic["sigma_mosaic_deg"]),
            float(mosaic["gamma_mosaic_deg"]),
            float(mosaic["eta"]),
            wavelength_arr,
            float(param_set["debye_x"]),
            float(param_set["debye_y"]),
            center,
            float(param_set["theta_initial"]),
            float(param_set.get("cor_angle", 0.0)),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            save_flag=0,
            optics_mode=int(param_set.get("optics_mode", 0)),
            single_sample_indices=single_sample_indices,
            best_sample_indices_out=best_sample_indices_out,
            collect_hit_tables=collect_hit_tables,
        )

    forced_sample_indices: np.ndarray | None = None
    if force_single_ray and measured_peaks and sim_miller.shape[0] > 0:
        try:
            best_indices = np.full(sim_miller.shape[0], -1, dtype=np.int64)
            _run_preview_sim(
                best_sample_indices_out=best_indices,
                collect_hit_tables=False,
            )
            forced_sample_indices = best_indices
        except Exception:
            forced_sample_indices = None

    _, hit_tables, *_ = _run_preview_sim(
        single_sample_indices=forced_sample_indices,
        collect_hit_tables=True,
    )
    return _extract_simulated_fit_peaks(
        hit_tables,
        (int(image_size), int(image_size)),
        original_table_indices,
    )


def _auto_match_background_peaks(
    simulated_peaks: list[dict[str, object]],
    background_image: np.ndarray,
    cfg: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Match by nearest background peak with Voronoi ownership constraint.

    Each detected background peak is owned by its nearest simulated peak. For
    each simulated peak, we then pick the nearest owned background peak.
    """

    config = cfg if isinstance(cfg, dict) else {}
    search_radius = max(1.0, float(config.get("search_radius_px", 24.0)))
    local_max_size = int(config.get("local_max_size_px", 5))
    local_max_size = max(3, local_max_size)
    if local_max_size % 2 == 0:
        local_max_size += 1
    smooth_sigma = max(0.0, float(config.get("smooth_sigma_px", 3.0)))
    min_prominence_sigma = float(config.get("min_prominence_sigma", 2.0))
    min_match_prominence_sigma = float(
        config.get("min_match_prominence_sigma", min_prominence_sigma)
    )
    k_neighbors = max(1, int(config.get("k_neighbors", 12)))
    fallback_percentile = float(config.get("fallback_percentile", 99.5))
    fallback_percentile = min(100.0, max(50.0, fallback_percentile))
    min_confidence = float(config.get("min_confidence", 0.0))
    max_candidate_peaks = max(50, int(config.get("max_candidate_peaks", 1200)))
    ambiguity_ratio_min = max(1.0, float(config.get("ambiguity_ratio_min", 1.15)))
    ambiguity_margin_px = max(0.0, float(config.get("ambiguity_margin_px", 2.0)))
    distance_sigma_clip = max(0.0, float(config.get("distance_sigma_clip", 3.5)))
    max_match_distance_px = float(config.get("max_match_distance_px", np.inf))
    if not np.isfinite(max_match_distance_px):
        max_match_distance_px = np.inf

    img = np.asarray(background_image, dtype=float)
    valid_mask = np.isfinite(img)
    if img.ndim != 2 or not np.any(valid_mask):
        return [], {"simulated_count": float(len(simulated_peaks))}

    baseline = float(np.median(img[valid_mask]))
    work = np.where(valid_mask, img, baseline)
    smooth = gaussian_filter(work, sigma=smooth_sigma, mode="nearest")
    prominence = work - smooth
    local_max = work == maximum_filter(work, size=local_max_size, mode="nearest")

    prom_vals = prominence[valid_mask]
    prom_center = float(np.median(prom_vals))
    mad = float(np.median(np.abs(prom_vals - prom_center)))
    sigma_est = 1.4826 * mad
    if not np.isfinite(sigma_est) or sigma_est <= 1e-12:
        sigma_est = float(np.std(prom_vals))
    if not np.isfinite(sigma_est) or sigma_est <= 1e-12:
        sigma_est = 1.0

    candidate_floor = prom_center + min_prominence_sigma * sigma_est
    candidate_mask = local_max & valid_mask & (prominence >= candidate_floor)
    if not np.any(candidate_mask):
        fallback_floor = float(np.percentile(prom_vals, fallback_percentile))
        candidate_mask = local_max & valid_mask & (prominence >= fallback_floor)

    rows, cols = np.nonzero(candidate_mask)
    if rows.size == 0:
        return [], {
            "simulated_count": float(len(simulated_peaks)),
            "sigma_est": float(sigma_est),
            "candidate_count": 0.0,
            "within_radius_count": 0.0,
            "unambiguous_count": 0.0,
            "matched_pre_clip_count": 0.0,
            "clipped_count": 0.0,
        }

    candidate_coords = np.column_stack((cols.astype(float), rows.astype(float)))
    candidate_prom_sigma = (prominence[rows, cols] - prom_center) / (sigma_est + 1e-12)
    if candidate_coords.shape[0] > max_candidate_peaks:
        keep_idx = np.argsort(candidate_prom_sigma)[-max_candidate_peaks:]
        candidate_coords = candidate_coords[keep_idx]
        candidate_prom_sigma = candidate_prom_sigma[keep_idx]

    ordered_simulated = sorted(
        simulated_peaks,
        key=lambda entry: float(entry.get("weight", 0.0)),
        reverse=True,
    )
    n_sim = len(ordered_simulated)
    n_cand = int(candidate_coords.shape[0])
    if n_sim == 0 or n_cand == 0:
        return [], {
            "simulated_count": float(len(simulated_peaks)),
            "candidate_count": float(n_cand),
            "sigma_est": float(sigma_est),
            "within_radius_count": 0.0,
            "unambiguous_count": 0.0,
            "matched_pre_clip_count": 0.0,
            "clipped_count": 0.0,
        }

    sim_coords = np.zeros((n_sim, 2), dtype=float)
    for i, entry in enumerate(ordered_simulated):
        sim_coords[i, 0] = float(entry["sim_col"])
        sim_coords[i, 1] = float(entry["sim_row"])

    sim_tree = cKDTree(sim_coords)
    k_query = min(max(2, k_neighbors), n_sim) if n_sim > 1 else 1
    cand_dists, cand_idx = sim_tree.query(candidate_coords, k=k_query)
    cand_dists = np.asarray(cand_dists, dtype=float)
    cand_idx = np.asarray(cand_idx, dtype=int)
    if cand_dists.ndim == 1:
        cand_dists = cand_dists[:, np.newaxis]
        cand_idx = cand_idx[:, np.newaxis]

    owner_dist = cand_dists[:, 0]
    owner_idx = cand_idx[:, 0]
    within_radius = np.isfinite(owner_dist) & (owner_dist <= search_radius)

    if cand_dists.shape[1] >= 2:
        second_dist = cand_dists[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = second_dist / np.maximum(owner_dist, 1e-9)
        unambiguous = (
            ~np.isfinite(second_dist)
            | ((second_dist - owner_dist) >= ambiguity_margin_px)
            | (ratio >= ambiguity_ratio_min)
        )
    else:
        unambiguous = np.ones_like(within_radius, dtype=bool)

    candidate_allowed = within_radius & unambiguous

    best_cand_for_sim = np.full(n_sim, -1, dtype=int)
    best_dist_for_sim = np.full(n_sim, np.inf, dtype=float)
    best_prom_for_sim = np.full(n_sim, -np.inf, dtype=float)

    for cand_i in range(n_cand):
        if not candidate_allowed[cand_i]:
            continue
        owner = int(owner_idx[cand_i])
        if owner < 0 or owner >= n_sim:
            continue

        prom_sigma = float(candidate_prom_sigma[cand_i])
        if prom_sigma < min_match_prominence_sigma:
            continue

        dist = float(owner_dist[cand_i])
        current_best = best_dist_for_sim[owner]
        if (dist + 1e-12) < current_best or (
            abs(dist - current_best) <= 1e-12 and prom_sigma > best_prom_for_sim[owner]
        ):
            best_dist_for_sim[owner] = dist
            best_prom_for_sim[owner] = prom_sigma
            best_cand_for_sim[owner] = cand_i

    pre_clip_matches: list[dict[str, object]] = []
    for sim_idx in range(n_sim):
        cand_i = int(best_cand_for_sim[sim_idx])
        if cand_i < 0:
            continue
        entry = ordered_simulated[sim_idx]
        sim_col = float(entry["sim_col"])
        sim_row = float(entry["sim_row"])
        col = float(candidate_coords[cand_i, 0])
        row = float(candidate_coords[cand_i, 1])
        dist_px = float(best_dist_for_sim[sim_idx])
        prom_sigma = float(best_prom_for_sim[sim_idx])
        confidence = max(0.0, prom_sigma) / (1.0 + max(0.0, dist_px))
        if confidence < min_confidence:
            continue
        if dist_px > max_match_distance_px:
            continue
        pre_clip_matches.append(
            {
                "hkl": tuple(int(v) for v in entry["hkl"]),
                "label": str(entry["label"]),
                "x": col,
                "y": row,
                "sim_x": sim_col,
                "sim_y": sim_row,
                "distance_px": dist_px,
                "prominence_sigma": prom_sigma,
                "confidence": float(confidence),
                "weight": float(entry.get("weight", 0.0)),
                "source_table_index": entry.get("source_table_index"),
                "source_row_index": entry.get("source_row_index"),
                "phi": entry.get("phi"),
            }
        )

    matches = list(pre_clip_matches)
    clipped_count = 0
    clip_limit = np.inf
    if matches:
        dists = np.asarray([float(m["distance_px"]) for m in matches], dtype=float)
        dist_med = float(np.median(dists))
        dist_mad = float(np.median(np.abs(dists - dist_med)))
        dist_sigma = 1.4826 * dist_mad
        clip_limit = search_radius
        if np.isfinite(max_match_distance_px):
            clip_limit = min(clip_limit, float(max_match_distance_px))
        if np.isfinite(distance_sigma_clip):
            if np.isfinite(dist_sigma) and dist_sigma > 1e-9:
                clip_limit = min(clip_limit, dist_med + distance_sigma_clip * dist_sigma)
            else:
                clip_limit = min(clip_limit, dist_med + max(1.0, distance_sigma_clip))
        matches = [
            m for m in matches if float(m["distance_px"]) <= float(clip_limit) + 1e-9
        ]
        clipped_count = max(0, len(pre_clip_matches) - len(matches))
    else:
        dist_med = float("nan")
        dist_sigma = float("nan")

    match_dists = np.asarray([float(m["distance_px"]) for m in matches], dtype=float)
    match_conf = np.asarray([float(m["confidence"]) for m in matches], dtype=float)

    stats = {
        "simulated_count": float(len(simulated_peaks)),
        "candidate_count": float(candidate_coords.shape[0]),
        "within_radius_count": float(np.count_nonzero(within_radius)),
        "unambiguous_count": float(np.count_nonzero(candidate_allowed)),
        "matched_pre_clip_count": float(len(pre_clip_matches)),
        "matched_count": float(len(matches)),
        "clipped_count": float(clipped_count),
        "sigma_est": float(sigma_est),
        "prominence_center": float(prom_center),
        "search_radius_px": float(search_radius),
        "k_query_used": float(k_query),
        "ambiguity_ratio_min": float(ambiguity_ratio_min),
        "ambiguity_margin_px": float(ambiguity_margin_px),
        "distance_sigma_clip": float(distance_sigma_clip),
        "distance_clip_limit_px": float(clip_limit) if np.isfinite(clip_limit) else float("nan"),
        "distance_median_pre_clip_px": float(dist_med) if np.isfinite(dist_med) else float("nan"),
        "distance_sigma_pre_clip_px": float(dist_sigma) if np.isfinite(dist_sigma) else float("nan"),
        "mean_match_distance_px": float(np.mean(match_dists)) if match_dists.size else float("nan"),
        "p90_match_distance_px": float(np.percentile(match_dists, 90.0)) if match_dists.size else float("nan"),
        "median_match_confidence": float(np.median(match_conf)) if match_conf.size else float("nan"),
    }
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
    target_min_matches = max(1, int(min_matches))
    attempts: list[dict[str, float | str]] = []

    def _run_attempt(
        attempt_cfg: dict[str, object], label: str
    ) -> tuple[list[dict[str, object]], dict[str, float]]:
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
                "within_radius_count": float(stats_i.get("within_radius_count", np.nan)),
                "matched_pre_clip_count": float(
                    stats_i.get("matched_pre_clip_count", np.nan)
                ),
                "clipped_count": float(stats_i.get("clipped_count", np.nan)),
            }
        )
        return matches_i, stats_i

    best_matches, best_stats = _run_attempt(base_cfg, "base")
    best_cfg = dict(base_cfg)

    if len(best_matches) >= target_min_matches:
        return best_matches, best_stats, best_cfg, attempts

    if not bool(base_cfg.get("relax_on_low_matches", True)):
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
            break

    return best_matches, best_stats, best_cfg, attempts


def on_fit_geometry_click():
    global profile_cache, last_simulation_signature, last_geometry_overlay_state
    _clear_geometry_pick_artists()

    # first, reconstruct the same mosaic_params dict you use in do_update()
    mosaic_params = build_mosaic_params()


    # assemble the params dict with exactly the keys the optimizer expects
    params = {
        'a':                  a_var.get(),
        'c':                  c_var.get(),
        'lambda':             lambda_,          # not 'lambda_'
        'psi':                psi,
        'psi_z':              psi_z_var.get(),
        'zs':                 zs_var.get(),
        'zb':                 zb_var.get(),
        'chi':                chi_var.get(),
        'n2':                 n2,
        'mosaic_params':      mosaic_params,
        'debye_x':            debye_x_var.get(),
        'debye_y':            debye_y_var.get(),
        'center':             [center_x_var.get(), center_y_var.get()],
        'center_x':           center_x_var.get(),
        'center_y':           center_y_var.get(),
        'theta_initial':      theta_initial_var.get(),
        'uv1':                np.array([1.0,0.0,0.0]),
        'uv2':                np.array([0.0,1.0,0.0]),
        'corto_detector':     corto_detector_var.get(),
        'gamma':              gamma_var.get(),
        'Gamma':              Gamma_var.get(),
        'cor_angle':          cor_angle_var.get(),
        'optics_mode':        _current_optics_mode_flag(),
    }

    # build the list of which of those to vary
    var_names = []
    if fit_zb_var.get():    var_names.append('zb')
    if fit_zs_var.get():    var_names.append('zs')
    if fit_theta_var.get(): var_names.append('theta_initial')
    if fit_psi_z_var.get(): var_names.append('psi_z')
    if fit_chi_var.get():   var_names.append('chi')
    if fit_cor_var.get():   var_names.append('cor_angle')
    if fit_gamma_var.get(): var_names.append('gamma')
    if fit_Gamma_var.get(): var_names.append('Gamma')
    if fit_corto_var.get(): var_names.append('corto_detector')
    if fit_a_var.get(): var_names.append('a')
    if fit_c_var.get(): var_names.append('c')
    if fit_center_x_var.get(): var_names.append('center_x')
    if fit_center_y_var.get(): var_names.append('center_y')
    if not var_names:
        progress_label_geometry.config(text="No parameters selected!")
        return

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(geometry_refine_cfg, dict):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, dict):
        auto_match_cfg = {}
    orientation_cfg = geometry_refine_cfg.get("orientation", {}) or {}
    if not isinstance(orientation_cfg, dict):
        orientation_cfg = {}
    geometry_runtime_cfg = _build_geometry_fit_runtime_config(
        geometry_refine_cfg,
        {
            name: params.get(name)
            for name in var_names
        },
        _current_geometry_fit_constraint_state(var_names),
        _current_geometry_fit_parameter_domains(var_names),
    )

    native_background = _get_current_background_native()
    backend_background = _get_current_background_backend()
    display_background = current_background_display
    if display_background is None and native_background is not None:
        display_background = np.rot90(native_background, DISPLAY_ROTATE_K)
    if native_background is None or display_background is None:
        progress_label_geometry.config(
            text="Geometry fit unavailable: no background image is loaded."
        )
        return
    if backend_background is None:
        backend_background = native_background

    miller_array = np.asarray(miller, dtype=np.float64)
    intensity_array = np.asarray(intensities, dtype=np.float64)
    if miller_array.ndim != 2 or miller_array.shape[1] != 3 or miller_array.size == 0:
        progress_label_geometry.config(
            text="Geometry fit unavailable: no simulated reflections are available."
        )
        return
    if intensity_array.shape[0] != miller_array.shape[0]:
        progress_label_geometry.config(
            text="Geometry fit unavailable: intensity array does not match HKLs."
        )
        return

    try:
        simulated_peaks = _simulate_hkl_peak_centers_for_fit(
            miller_array,
            intensity_array,
            image_size,
            params,
        )
    except Exception as exc:
        progress_label_geometry.config(
            text=f"Geometry fit unavailable: failed to simulate peak centers ({exc})."
        )
        return

    if not simulated_peaks:
        progress_label_geometry.config(
            text="Geometry fit unavailable: no simulated Bragg peak centers were found."
        )
        return

    default_min_matches = max(6, len(var_names) + 2)
    min_matches = int(auto_match_cfg.get("min_matches", default_min_matches))
    min_matches = max(1, min_matches)

    matched_pairs, match_stats, effective_auto_match_cfg, auto_match_attempts = (
        _auto_match_background_peaks_with_relaxation(
            simulated_peaks,
            np.asarray(display_background, dtype=float),
            auto_match_cfg,
            min_matches=min_matches,
        )
    )

    if len(matched_pairs) < min_matches:
        simulated_count = int(match_stats.get("simulated_count", len(simulated_peaks)))
        best_radius = float(
            effective_auto_match_cfg.get(
                "search_radius_px", auto_match_cfg.get("search_radius_px", 24.0)
            )
        )
        progress_label_geometry.config(
            text=(
                "Geometry fit cancelled: auto-match found "
                f"{len(matched_pairs)}/{simulated_count} confident peaks "
                f"(need at least {min_matches}); "
                f"best radius={best_radius:.1f}px. Try Auto-Match Scale, then retry."
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
        normalized_hkl = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if normalized_hkl is None:
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
                "overlay_match_index": int(
                    entry.get("overlay_match_index", len(initial_auto_pairs_display))
                ),
                "hkl": normalized_hkl,
                "sim_display": (float(sim_display[0]), float(sim_display[1])),
                "bg_display": (float(bg_col), float(bg_row)),
            }
        )
    preview_marker_limit = int(auto_match_cfg.get("max_display_markers", 120))
    preview_marker_limit = max(1, preview_marker_limit)
    _draw_initial_geometry_pairs_overlay(
        initial_auto_pairs_display,
        max_display_markers=preview_marker_limit,
    )

    def _mark_auto_pick(col: float, row: float, label: str, color: str, marker: str):
        point, = ax.plot(
            [col],
            [row],
            marker,
            color=color,
            markersize=8,
            markerfacecolor='none',
            zorder=7,
            linestyle='None',
        )
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
        geometry_pick_artists.extend([point, text])

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

    _last_fit_status = {"text": ""}

    def _fit_status_update(text: str) -> None:
        message = "" if text is None else str(text).strip()
        if not message:
            return
        progress_label_geometry.config(text=message)
        root.update_idletasks()
        if message != _last_fit_status["text"]:
            _log_line(f"[status] {message}")
            _last_fit_status["text"] = message

    progress_label_geometry.config(text="Running geometry fit (auto-matched peaks)…")
    root.update_idletasks()

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
            hkl_value = entry.get("hkl")
            if not isinstance(hkl_value, tuple) or len(hkl_value) != 3:
                try:
                    hkl_value = tuple(
                        int(round(float(part)))
                        for part in str(entry.get("label", "")).split(",")[:3]
                    )
                except Exception:
                    hkl_value = None
            measured_entry = dict(entry)
            measured_entry["label"] = str(entry["label"])
            measured_entry["x"] = x_val
            measured_entry["y"] = y_val
            measured_entry["overlay_match_index"] = int(
                entry.get("overlay_match_index", len(measured_from_display))
            )
            if isinstance(hkl_value, tuple) and len(hkl_value) == 3:
                measured_entry["hkl"] = hkl_value
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
                    f"within={int(float(entry.get('within_radius_count', 0.0)))} "
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
                f"candidate_peaks={int(match_stats.get('candidate_count', 0))}",
                f"within_radius_candidates={int(match_stats.get('within_radius_count', 0))}",
                f"unambiguous_candidates={int(match_stats.get('unambiguous_count', 0))}",
                f"matched_pre_clip={int(match_stats.get('matched_pre_clip_count', 0))}",
                f"clipped_matches={int(match_stats.get('clipped_count', 0))}",
                f"matched_peaks={len(matched_pairs)}",
                f"prominence_sigma_est={float(match_stats.get('sigma_est', np.nan)):.6f}",
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
                    f"src=({int(entry.get('source_table_index', -1))},"
                    f"{int(entry.get('source_row_index', -1))}) "
                    f"sim=({entry['sim_x']:.3f},{entry['sim_y']:.3f}) "
                    f"meas=({entry['x']:.3f},{entry['y']:.3f}) "
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
        _log_section(
            "Orientation diagnostics:",
            [
                f"pairs={orientation_diag.get('pairs', 0)}",
                f"enabled={orientation_diag.get('enabled', True)}",
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
            progress_label_geometry.config(
                text=(
                    f"Geometry fit iteration {iter_idx + 1}/{fit_iterations} "
                    f"(matches={len(current_matched_pairs)})…"
                )
            )
            root.update_idletasks()

            result = fit_geometry_parameters(
                miller,
                intensities,
                image_size,
                current_fit_params,
                current_measured_for_fit,
                var_names,
                pixel_tol=float('inf'),
                experimental_image=experimental_image_for_fit,
                refinement_config=geometry_runtime_cfg,
                status_callback=_fit_status_update,
            )

            if getattr(result, "x", None) is None or len(result.x) != len(var_names):
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

            if iter_idx + 1 >= fit_iterations:
                break

            try:
                sim_iter = _simulate_hkl_peak_centers_for_fit(
                    miller_array,
                    intensity_array,
                    image_size,
                    current_fit_params,
                    measured_peaks=current_measured_for_fit,
                )
                if not sim_iter:
                    iteration_logs.append(
                        f"iter={iter_idx + 1}: rematch skipped (no simulated peaks)"
                    )
                    break

                matched_iter, stats_iter, _, rematch_attempts = (
                    _auto_match_background_peaks_with_relaxation(
                        sim_iter,
                        np.asarray(display_background, dtype=float),
                        auto_match_cfg,
                        min_matches=min_matches,
                    )
                )
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
                            f"radius={float(best_attempt.get('search_radius_px', np.nan)):.1f}px)"
                        )
                    iteration_logs.append(
                        (
                            f"iter={iter_idx + 1}: rematch skipped "
                            f"({len(matched_iter)} < min_matches={min_matches}"
                            f"{best_attempt_note})"
                        )
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
                    hkl_value = entry.get("hkl")
                    if not isinstance(hkl_value, tuple) or len(hkl_value) != 3:
                        try:
                            hkl_value = tuple(
                                int(round(float(part)))
                                for part in str(entry.get("label", "")).split(",")[:3]
                            )
                        except Exception:
                            hkl_value = None
                    measured_entry = dict(entry)
                    measured_entry["label"] = str(entry["label"])
                    measured_entry["x"] = x_val
                    measured_entry["y"] = y_val
                    measured_entry["overlay_match_index"] = int(
                        entry.get("overlay_match_index", len(measured_iter_display))
                    )
                    if isinstance(hkl_value, tuple) and len(hkl_value) == 3:
                        measured_entry["hkl"] = hkl_value
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

                matched_iter = _attach_overlay_match_indices(matched_iter)
                current_matched_pairs = matched_iter
                current_measured_for_fit = measured_iter_for_fit
                match_stats = stats_iter
                progress_label_geometry.config(
                    text=(
                        f"Geometry fit iteration {iter_idx + 1}/{fit_iterations} "
                        f"rematch complete (matches={len(current_matched_pairs)})."
                    )
                )
                root.update_idletasks()
            except Exception as rematch_exc:
                iteration_logs.append(
                    f"iter={iter_idx + 1}: rematch failed ({rematch_exc})"
                )
                break

        if result is None:
            raise RuntimeError("Geometry optimizer did not run.")

        matched_pairs = current_matched_pairs
        measured_for_fit = current_measured_for_fit

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
        point_match_summary = getattr(result, "point_match_summary", None)
        if isinstance(point_match_summary, dict):
            _log_section(
                "Point-match execution:",
                [
                    f"matched_pairs={int(point_match_summary.get('matched_pair_count', 0))}",
                    f"peak_weighting_mode={point_match_summary.get('peak_weighting_mode', 'uniform')}",
                    f"custom_sigma_count={int(point_match_summary.get('custom_sigma_count', 0))}",
                    f"anisotropic_sigma_count={int(point_match_summary.get('anisotropic_sigma_count', 0))}",
                    f"unweighted_peak_rms_px={float(point_match_summary.get('unweighted_peak_rms_px', np.nan)):.6f}",
                    f"simulated_reflections={int(point_match_summary.get('simulated_reflection_count', 0))}",
                    f"total_reflections={int(point_match_summary.get('total_reflection_count', 0))}",
                    f"subset_reduced={bool(point_match_summary.get('subset_reduced', False))}",
                    f"fixed_source_reflections={int(point_match_summary.get('fixed_source_reflection_count', 0))}",
                    f"fallback_hkls={int(point_match_summary.get('subset_fallback_hkl_count', 0))}",
                    f"single_ray_enabled={bool(point_match_summary.get('single_ray_enabled', False))}",
                    f"single_ray_forced_count={int(point_match_summary.get('single_ray_forced_count', 0))}",
                ],
            )
        ridge_refinement_summary = getattr(result, "ridge_refinement_summary", None)
        if isinstance(ridge_refinement_summary, dict):
            _log_section(
                "Ridge refinement:",
                [
                    f"enabled={bool(ridge_refinement_summary.get('enabled', False))}",
                    f"status={ridge_refinement_summary.get('status', 'unknown')}",
                    f"reason={ridge_refinement_summary.get('reason', 'n/a')}",
                    f"accepted={bool(ridge_refinement_summary.get('accepted', False))}",
                    f"ridge_cost_before={float(ridge_refinement_summary.get('ridge_cost_before', np.nan)):.6f}",
                    f"ridge_cost_after={float(ridge_refinement_summary.get('ridge_cost_after', np.nan)):.6f}",
                    f"point_rms_before_px={float(ridge_refinement_summary.get('point_rms_before_px', np.nan)):.6f}",
                    f"point_rms_after_px={float(ridge_refinement_summary.get('point_rms_after_px', np.nan)):.6f}",
                ],
            )
        image_refinement_summary = getattr(result, "image_refinement_summary", None)
        if isinstance(image_refinement_summary, dict):
            _log_section(
                "ROI/image refinement:",
                [
                    f"enabled={bool(image_refinement_summary.get('enabled', False))}",
                    f"status={image_refinement_summary.get('status', 'unknown')}",
                    f"reason={image_refinement_summary.get('reason', 'n/a')}",
                    f"accepted={bool(image_refinement_summary.get('accepted', False))}",
                    f"preview_roi_count={int(image_refinement_summary.get('preview_roi_count', 0))}",
                    f"selected_roi_count={int(image_refinement_summary.get('selected_roi_count', 0))}",
                    f"image_cost_before={float(image_refinement_summary.get('image_cost_before', np.nan)):.6f}",
                    f"image_cost_after={float(image_refinement_summary.get('image_cost_after', np.nan)):.6f}",
                    f"point_rms_before_px={float(image_refinement_summary.get('point_rms_before_px', np.nan)):.6f}",
                    f"point_rms_after_px={float(image_refinement_summary.get('point_rms_after_px', np.nan)):.6f}",
                ],
            )
        reparameterization_summary = getattr(result, "reparameterization_summary", None)
        if isinstance(reparameterization_summary, dict):
            _log_section(
                "Pair reparameterization:",
                [
                    f"enabled={bool(reparameterization_summary.get('enabled', False))}",
                    f"status={reparameterization_summary.get('status', 'unknown')}",
                    f"reason={reparameterization_summary.get('reason', 'n/a')}",
                    f"accepted={bool(reparameterization_summary.get('accepted', False))}",
                    f"pairs={list(reparameterization_summary.get('pairs', []))}",
                    f"start_cost={float(reparameterization_summary.get('start_cost', np.nan)):.6f}",
                    f"final_cost={float(reparameterization_summary.get('final_cost', np.nan)):.6f}",
                ],
            )
        staged_release_summary = getattr(result, "staged_release_summary", None)
        if isinstance(staged_release_summary, dict):
            stage_lines = [
                f"enabled={bool(staged_release_summary.get('enabled', False))}",
                f"status={staged_release_summary.get('status', 'unknown')}",
                f"reason={staged_release_summary.get('reason', 'n/a')}",
                f"accepted={bool(staged_release_summary.get('accepted', False))}",
                f"planned_stage_count={int(staged_release_summary.get('planned_stage_count', 0))}",
                f"accepted_stage_count={int(staged_release_summary.get('accepted_stage_count', 0))}",
            ]
            for stage_entry in staged_release_summary.get("stages", []) or []:
                if not isinstance(stage_entry, dict):
                    continue
                stage_lines.append(
                    "stage[{stage}] status={status} reason={reason} active={active} fixed={fixed} "
                    "start_cost={start_cost:.6f} final_cost={final_cost:.6f}".format(
                        stage=int(stage_entry.get("stage", -1)),
                        status=str(stage_entry.get("status", "unknown")),
                        reason=str(stage_entry.get("reason", "n/a")),
                        active=list(stage_entry.get("active_parameters", [])),
                        fixed=list(stage_entry.get("fixed_parameters", [])),
                        start_cost=float(stage_entry.get("start_cost", np.nan)),
                        final_cost=float(stage_entry.get("final_cost", np.nan)),
                    )
                )
            _log_section("Staged release:", stage_lines)
        auto_freeze_summary = getattr(result, "auto_freeze_summary", None)
        if isinstance(auto_freeze_summary, dict):
            _log_section(
                "Auto-freeze:",
                [
                    f"enabled={bool(auto_freeze_summary.get('enabled', False))}",
                    f"status={auto_freeze_summary.get('status', 'unknown')}",
                    f"reason={auto_freeze_summary.get('reason', 'n/a')}",
                    f"accepted={bool(auto_freeze_summary.get('accepted', False))}",
                    f"fixed_parameters={list(auto_freeze_summary.get('fixed_parameters', []))}",
                    f"active_parameters={list(auto_freeze_summary.get('active_parameters', []))}",
                    f"start_cost={float(auto_freeze_summary.get('start_cost', np.nan)):.6f}",
                    f"final_cost={float(auto_freeze_summary.get('final_cost', np.nan)):.6f}",
                ],
            )
        selective_thaw_summary = getattr(result, "selective_thaw_summary", None)
        if isinstance(selective_thaw_summary, dict):
            thaw_lines = [
                f"enabled={bool(selective_thaw_summary.get('enabled', False))}",
                f"status={selective_thaw_summary.get('status', 'unknown')}",
                f"reason={selective_thaw_summary.get('reason', 'n/a')}",
                f"accepted={bool(selective_thaw_summary.get('accepted', False))}",
                f"thawed_parameters={list(selective_thaw_summary.get('thawed_parameters', []))}",
                f"remaining_fixed_parameters={list(selective_thaw_summary.get('remaining_fixed_parameters', []))}",
                f"start_cost={float(selective_thaw_summary.get('start_cost', np.nan)):.6f}",
                f"final_cost={float(selective_thaw_summary.get('final_cost', np.nan)):.6f}",
            ]
            for step_entry in selective_thaw_summary.get("steps", []) or []:
                if not isinstance(step_entry, dict):
                    continue
                thaw_lines.append(
                    "step[{step}] accepted={accepted} reason={reason} thawed={thawed} fixed={fixed} "
                    "start_cost={start_cost:.6f} final_cost={final_cost:.6f}".format(
                        step=int(step_entry.get("step", -1)),
                        accepted=bool(step_entry.get("accepted_parameters", [])),
                        reason=str(
                            step_entry.get(
                                "reason",
                                selective_thaw_summary.get("reason", "n/a"),
                            )
                        ),
                        thawed=list(step_entry.get("accepted_parameters", [])),
                        fixed=list(step_entry.get("remaining_fixed_parameters", [])),
                        start_cost=float(step_entry.get("start_cost", np.nan)),
                        final_cost=float(step_entry.get("final_cost", np.nan)),
                    )
                )
            _log_section("Selective thaw:", thaw_lines)
        identifiability_summary = getattr(result, "identifiability_summary", None)
        if isinstance(identifiability_summary, dict):
            _log_section(
                "Identifiability diagnostics:",
                [
                    f"enabled={bool(identifiability_summary.get('enabled', False))}",
                    f"status={identifiability_summary.get('status', 'unknown')}",
                    f"reason={identifiability_summary.get('reason', 'n/a')}",
                    f"rank={int(identifiability_summary.get('rank', 0))}",
                    f"num_valid_parameters={int(identifiability_summary.get('num_valid_parameters', 0))}",
                    f"num_residuals={int(identifiability_summary.get('num_residuals', 0))}",
                    f"condition_number={float(identifiability_summary.get('condition_number', np.nan)):.6g}",
                    f"dominant_group={identifiability_summary.get('dominant_group', 'unknown')}",
                    f"underconstrained={bool(identifiability_summary.get('underconstrained', False))}",
                ],
            )

        undo_state = _capture_geometry_fit_undo_state()
        for name in var_names:
            val = float(current_fit_params.get(name, params.get(name, 0.0)))
            if name == 'zb':
                zb_var.set(val)
            elif name == 'zs':
                zs_var.set(val)
            elif name == 'theta_initial':
                theta_initial_var.set(val)
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

        profile_cache = dict(profile_cache)
        profile_cache.update(mosaic_params)
        profile_cache.update(
            {
                "theta_initial": theta_initial_var.get(),
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

        last_simulation_signature = None
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
        _store_geometry_fit_snapshot(fitted_params, measured_for_fit)

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
        overlay_records = build_geometry_fit_overlay_records(
            initial_auto_pairs_display,
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
                initial_auto_pairs_display,
                max_display_markers=max_display_markers,
            )
        last_geometry_overlay_state = {
            "overlay_records": _copy_geometry_fit_state_value(overlay_records),
            "initial_pairs_display": _copy_geometry_fit_state_value(
                initial_auto_pairs_display
            ),
            "max_display_markers": int(max_display_markers),
        }

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
        progress_label_geometry.config(
            text=(
                f"{base_summary}\n"
                f"Auto-matched peaks: {len(matched_pairs)}/"
                f"{int(match_stats.get('simulated_count', len(simulated_peaks)))}\n"
                f"Saved {len(export_recs)} peak records → {save_path}\n"
                f"Fit log → {log_path}"
            )
        )
        _push_geometry_fit_undo_state(undo_state)
        return
    except Exception as exc:
        _log_line(f"Geometry fit failed: {exc}")
        progress_label_geometry.config(text=f"Geometry fit failed: {exc}")
        return
    finally:
        try:
            log_file.close()
        except Exception:
            pass

    def _nearest_simulated_peak(col: float, row: float):
        """Return (index, distance^2) of the nearest simulated peak, or (None, inf)."""

        best_idx, best_d2 = None, float("inf")
        for idx, (px, py) in enumerate(peak_positions):
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
        r1 = min(current_background_display.shape[0], r + search_radius + 1)
        c0 = max(0, c - search_radius)
        c1 = min(current_background_display.shape[1], c + search_radius + 1)

        window = np.asarray(current_background_display[r0:r1, c0:c1], dtype=float)
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
        geometry_pick_artists.extend([point, text])
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
        global profile_cache, last_simulation_signature, last_geometry_overlay_state
        nonlocal click_cid
        if click_cid is not None:
            canvas.mpl_disconnect(click_cid)
            click_cid = None
        canvas_widget.configure(cursor="")

    def _on_geometry_pick(event):
        global profile_cache, last_simulation_signature, last_geometry_overlay_state
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

            _last_fit_status = {"text": ""}

            def _fit_status_update(text: str) -> None:
                message = "" if text is None else str(text).strip()
                if not message:
                    return
                progress_label_geometry.config(text=message)
                root.update_idletasks()
                if message != _last_fit_status["text"]:
                    _log_line(f"[status] {message}")
                    _last_fit_status["text"] = message

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
                current_background_display.shape,
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
                    measured_peaks=measured_native,
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

                result = _run_with_responsiveness_indicator(
                    fit_geometry_parameters,
                    miller,
                    intensities,
                    image_size,
                    params,
                    measured_for_fit,
                    var_names,
                    pixel_tol=float('inf'),
                    experimental_image=experimental_image_for_fit,
                    refinement_config=geometry_runtime_cfg,
                    status_callback=_fit_status_update,
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
                point_match_summary = getattr(result, "point_match_summary", None)
                if isinstance(point_match_summary, dict):
                    _log_section(
                        "Point-match execution:",
                        [
                            f"matched_pairs={int(point_match_summary.get('matched_pair_count', 0))}",
                            f"peak_weighting_mode={point_match_summary.get('peak_weighting_mode', 'uniform')}",
                            f"custom_sigma_count={int(point_match_summary.get('custom_sigma_count', 0))}",
                            f"anisotropic_sigma_count={int(point_match_summary.get('anisotropic_sigma_count', 0))}",
                            f"unweighted_peak_rms_px={float(point_match_summary.get('unweighted_peak_rms_px', np.nan)):.6f}",
                            f"simulated_reflections={int(point_match_summary.get('simulated_reflection_count', 0))}",
                            f"total_reflections={int(point_match_summary.get('total_reflection_count', 0))}",
                            f"subset_reduced={bool(point_match_summary.get('subset_reduced', False))}",
                            f"fixed_source_reflections={int(point_match_summary.get('fixed_source_reflection_count', 0))}",
                            f"fallback_hkls={int(point_match_summary.get('subset_fallback_hkl_count', 0))}",
                            f"single_ray_enabled={bool(point_match_summary.get('single_ray_enabled', False))}",
                            f"single_ray_forced_count={int(point_match_summary.get('single_ray_forced_count', 0))}",
                        ],
                    )
                ridge_refinement_summary = getattr(result, "ridge_refinement_summary", None)
                if isinstance(ridge_refinement_summary, dict):
                    _log_section(
                        "Ridge refinement:",
                        [
                            f"enabled={bool(ridge_refinement_summary.get('enabled', False))}",
                            f"status={ridge_refinement_summary.get('status', 'unknown')}",
                            f"reason={ridge_refinement_summary.get('reason', 'n/a')}",
                            f"accepted={bool(ridge_refinement_summary.get('accepted', False))}",
                            f"ridge_cost_before={float(ridge_refinement_summary.get('ridge_cost_before', np.nan)):.6f}",
                            f"ridge_cost_after={float(ridge_refinement_summary.get('ridge_cost_after', np.nan)):.6f}",
                            f"point_rms_before_px={float(ridge_refinement_summary.get('point_rms_before_px', np.nan)):.6f}",
                            f"point_rms_after_px={float(ridge_refinement_summary.get('point_rms_after_px', np.nan)):.6f}",
                        ],
                    )
                image_refinement_summary = getattr(result, "image_refinement_summary", None)
                if isinstance(image_refinement_summary, dict):
                    _log_section(
                        "ROI/image refinement:",
                        [
                            f"enabled={bool(image_refinement_summary.get('enabled', False))}",
                            f"status={image_refinement_summary.get('status', 'unknown')}",
                            f"reason={image_refinement_summary.get('reason', 'n/a')}",
                            f"accepted={bool(image_refinement_summary.get('accepted', False))}",
                            f"preview_roi_count={int(image_refinement_summary.get('preview_roi_count', 0))}",
                            f"selected_roi_count={int(image_refinement_summary.get('selected_roi_count', 0))}",
                            f"image_cost_before={float(image_refinement_summary.get('image_cost_before', np.nan)):.6f}",
                            f"image_cost_after={float(image_refinement_summary.get('image_cost_after', np.nan)):.6f}",
                            f"point_rms_before_px={float(image_refinement_summary.get('point_rms_before_px', np.nan)):.6f}",
                            f"point_rms_after_px={float(image_refinement_summary.get('point_rms_after_px', np.nan)):.6f}",
                        ],
                    )
                reparameterization_summary = getattr(result, "reparameterization_summary", None)
                if isinstance(reparameterization_summary, dict):
                    _log_section(
                        "Pair reparameterization:",
                        [
                            f"enabled={bool(reparameterization_summary.get('enabled', False))}",
                            f"status={reparameterization_summary.get('status', 'unknown')}",
                            f"reason={reparameterization_summary.get('reason', 'n/a')}",
                            f"accepted={bool(reparameterization_summary.get('accepted', False))}",
                            f"pairs={list(reparameterization_summary.get('pairs', []))}",
                            f"start_cost={float(reparameterization_summary.get('start_cost', np.nan)):.6f}",
                            f"final_cost={float(reparameterization_summary.get('final_cost', np.nan)):.6f}",
                        ],
                    )
                staged_release_summary = getattr(result, "staged_release_summary", None)
                if isinstance(staged_release_summary, dict):
                    stage_lines = [
                        f"enabled={bool(staged_release_summary.get('enabled', False))}",
                        f"status={staged_release_summary.get('status', 'unknown')}",
                        f"reason={staged_release_summary.get('reason', 'n/a')}",
                        f"accepted={bool(staged_release_summary.get('accepted', False))}",
                        f"planned_stage_count={int(staged_release_summary.get('planned_stage_count', 0))}",
                        f"accepted_stage_count={int(staged_release_summary.get('accepted_stage_count', 0))}",
                    ]
                    for stage_entry in staged_release_summary.get("stages", []) or []:
                        if not isinstance(stage_entry, dict):
                            continue
                        stage_lines.append(
                            "stage[{stage}] status={status} reason={reason} active={active} fixed={fixed} "
                            "start_cost={start_cost:.6f} final_cost={final_cost:.6f}".format(
                                stage=int(stage_entry.get("stage", -1)),
                                status=str(stage_entry.get("status", "unknown")),
                                reason=str(stage_entry.get("reason", "n/a")),
                                active=list(stage_entry.get("active_parameters", [])),
                                fixed=list(stage_entry.get("fixed_parameters", [])),
                                start_cost=float(stage_entry.get("start_cost", np.nan)),
                                final_cost=float(stage_entry.get("final_cost", np.nan)),
                            )
                        )
                    _log_section("Staged release:", stage_lines)
                auto_freeze_summary = getattr(result, "auto_freeze_summary", None)
                if isinstance(auto_freeze_summary, dict):
                    _log_section(
                        "Auto-freeze:",
                        [
                            f"enabled={bool(auto_freeze_summary.get('enabled', False))}",
                            f"status={auto_freeze_summary.get('status', 'unknown')}",
                            f"reason={auto_freeze_summary.get('reason', 'n/a')}",
                            f"accepted={bool(auto_freeze_summary.get('accepted', False))}",
                            f"fixed_parameters={list(auto_freeze_summary.get('fixed_parameters', []))}",
                            f"active_parameters={list(auto_freeze_summary.get('active_parameters', []))}",
                            f"start_cost={float(auto_freeze_summary.get('start_cost', np.nan)):.6f}",
                            f"final_cost={float(auto_freeze_summary.get('final_cost', np.nan)):.6f}",
                        ],
                    )
                selective_thaw_summary = getattr(result, "selective_thaw_summary", None)
                if isinstance(selective_thaw_summary, dict):
                    thaw_lines = [
                        f"enabled={bool(selective_thaw_summary.get('enabled', False))}",
                        f"status={selective_thaw_summary.get('status', 'unknown')}",
                        f"reason={selective_thaw_summary.get('reason', 'n/a')}",
                        f"accepted={bool(selective_thaw_summary.get('accepted', False))}",
                        f"thawed_parameters={list(selective_thaw_summary.get('thawed_parameters', []))}",
                        f"remaining_fixed_parameters={list(selective_thaw_summary.get('remaining_fixed_parameters', []))}",
                        f"start_cost={float(selective_thaw_summary.get('start_cost', np.nan)):.6f}",
                        f"final_cost={float(selective_thaw_summary.get('final_cost', np.nan)):.6f}",
                    ]
                    for step_entry in selective_thaw_summary.get("steps", []) or []:
                        if not isinstance(step_entry, dict):
                            continue
                        thaw_lines.append(
                            "step[{step}] accepted={accepted} reason={reason} thawed={thawed} fixed={fixed} "
                            "start_cost={start_cost:.6f} final_cost={final_cost:.6f}".format(
                                step=int(step_entry.get("step", -1)),
                                accepted=bool(step_entry.get("accepted_parameters", [])),
                                reason=str(
                                    step_entry.get(
                                        "reason",
                                        selective_thaw_summary.get("reason", "n/a"),
                                    )
                                ),
                                thawed=list(step_entry.get("accepted_parameters", [])),
                                fixed=list(step_entry.get("remaining_fixed_parameters", [])),
                                start_cost=float(step_entry.get("start_cost", np.nan)),
                                final_cost=float(step_entry.get("final_cost", np.nan)),
                            )
                        )
                    _log_section("Selective thaw:", thaw_lines)
                identifiability_summary = getattr(result, "identifiability_summary", None)
                if isinstance(identifiability_summary, dict):
                    _log_section(
                        "Identifiability diagnostics:",
                        [
                            f"enabled={bool(identifiability_summary.get('enabled', False))}",
                            f"status={identifiability_summary.get('status', 'unknown')}",
                            f"reason={identifiability_summary.get('reason', 'n/a')}",
                            f"rank={int(identifiability_summary.get('rank', 0))}",
                            f"num_valid_parameters={int(identifiability_summary.get('num_valid_parameters', 0))}",
                            f"num_residuals={int(identifiability_summary.get('num_residuals', 0))}",
                            f"condition_number={float(identifiability_summary.get('condition_number', np.nan)):.6g}",
                            f"dominant_group={identifiability_summary.get('dominant_group', 'unknown')}",
                            f"underconstrained={bool(identifiability_summary.get('underconstrained', False))}",
                        ],
                    )

                undo_state = _capture_geometry_fit_undo_state()
                for name, val in zip(var_names, result.x):
                    if name == 'zb':               zb_var.set(val)
                    elif name == 'zs':             zs_var.set(val)
                    elif name == 'theta_initial':  theta_initial_var.set(val)
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

                # Keep the cached profile in sync with the fitted geometry so the
                # next simulation uses the updated parameters even when diagnostics
                # are disabled.
                profile_cache = dict(profile_cache)
                profile_cache.update(mosaic_params)
                profile_cache.update(
                    {
                        "theta_initial": theta_initial_var.get(),
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
                last_simulation_signature = None
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
                _store_geometry_fit_snapshot(fitted_params, measured_for_fit)

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

            pixel_offsets = []

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

            max_display_markers = int(auto_match_cfg.get("max_display_markers", 120))
            max_display_markers = max(1, max_display_markers)
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
            last_geometry_overlay_state = {
                "overlay_records": _copy_geometry_fit_state_value(overlay_records),
                "initial_pairs_display": _copy_geometry_fit_state_value(
                    initial_manual_pairs_display
                ),
                "max_display_markers": int(max_display_markers),
            }

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

            if DEBUG_ENABLED:
                final_text = f"{base_summary}\n{orientation_report}\nFit log → {log_path}"
            else:
                final_text = (
                    f"{base_summary}\n\nSaved {len(export_recs)} peak records →\n{save_path}"
                    + f"\n\nPixel offsets:\n{dist_report}\nFit log → {log_path}"
                )

            progress_label_geometry.config(text=final_text)
            _push_geometry_fit_undo_state(undo_state)
            return

        col, row = float(event.xdata), float(event.ydata)

        if selection_state["expecting"] == "sim":
            idx, _ = _nearest_simulated_peak(col, row)
            if idx is None:
                progress_label_geometry.config(text="No simulated peaks available to pick.")
                return
            selection_state["pending_hkl"] = peak_millers[idx]
            sim_col, sim_row = peak_positions[idx]
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

    global profile_cache, last_simulation_signature
    global last_geometry_fit_params, last_geometry_matched_peaks

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

    if not isinstance(last_geometry_fit_params, dict):
        progress_label_mosaic.config(
            text=(
                "Mosaic fit unavailable: geometry lock is missing. "
                "Run geometry fitting first."
            )
        )
        return

    if not last_geometry_matched_peaks:
        progress_label_mosaic.config(
            text=(
                "Mosaic fit unavailable: no geometry-matched peaks were saved. "
                "Run geometry fitting first."
            )
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

    current_hkls = {
        tuple(int(round(val)) for val in row)
        for row in np.asarray(miller_array, dtype=np.float64)
    }
    locked_hkls = {
        tuple(int(v) for v in entry.get("hkl", ()))
        for entry in last_geometry_matched_peaks
        if isinstance(entry, dict) and entry.get("hkl") is not None
    }
    if not locked_hkls:
        progress_label_mosaic.config(
            text=(
                "Mosaic fit unavailable: geometry-matched HKLs are missing from the lock state. "
                "Run geometry fitting first."
            )
        )
        return
    if not locked_hkls.issubset(current_hkls):
        progress_label_mosaic.config(
            text=(
                "Mosaic fit unavailable: current reflection list does not match geometry lock. "
                "Re-run geometry fitting."
            )
        )
        return

    params = dict(last_geometry_fit_params)
    frozen_mosaic = dict(params.get("mosaic_params", {}) or {})
    required_keys = (
        "beam_x_array",
        "beam_y_array",
        "theta_array",
        "phi_array",
        "wavelength_array",
    )
    missing = [key for key in required_keys if not np.asarray(frozen_mosaic.get(key)).size]
    if missing:
        progress_label_mosaic.config(
            text=(
                "Mosaic fit unavailable: geometry lock does not include full beam samples. "
                "Re-run geometry fitting."
            )
        )
        return

    frozen_mosaic["sigma_mosaic_deg"] = float(sigma_mosaic_var.get())
    frozen_mosaic["gamma_mosaic_deg"] = float(gamma_mosaic_var.get())
    frozen_mosaic["eta"] = float(eta_var.get())
    params["mosaic_params"] = frozen_mosaic

    fallback_params = {
        "a": a_var.get(),
        "c": c_var.get(),
        "lambda": lambda_,
        "psi": psi,
        "psi_z": psi_z_var.get(),
        "zs": zs_var.get(),
        "zb": zb_var.get(),
        "chi": chi_var.get(),
        "n2": n2,
        "debye_x": debye_x_var.get(),
        "debye_y": debye_y_var.get(),
        "center": [center_x_var.get(), center_y_var.get()],
        "theta_initial": theta_initial_var.get(),
        "uv1": np.array([1.0, 0.0, 0.0]),
        "uv2": np.array([0.0, 1.0, 0.0]),
        "corto_detector": corto_detector_var.get(),
        "gamma": gamma_var.get(),
        "Gamma": Gamma_var.get(),
        "cor_angle": cor_angle_var.get(),
        "optics_mode": _current_optics_mode_flag(),
    }
    for key, value in fallback_params.items():
        params.setdefault(key, value)

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
            num_peaks=36,
            roi_half_width=8,
            stratify="twotheta",
            stratify_bins=6,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=80,
            bounds=(
                np.array([0.03, 0.03, 0.0], dtype=np.float64),
                np.array([3.0, 3.0, 1.0], dtype=np.float64),
            ),
            measured_peaks=last_geometry_matched_peaks,
            peak_source="geometry",
            max_restarts=2,
            roi_normalization="sqrt_npix",
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

    # Keep the live UI state aligned with the geometry-locked snapshot used
    # during optimization so the rendered post-fit pattern matches the fit.
    def _set_var_from_locked(var, key: str):
        if key not in params:
            return
        try:
            value = float(params[key])
        except Exception:
            return
        if np.isfinite(value):
            var.set(value)

    _set_var_from_locked(a_var, "a")
    _set_var_from_locked(c_var, "c")
    _set_var_from_locked(zs_var, "zs")
    _set_var_from_locked(zb_var, "zb")
    _set_var_from_locked(chi_var, "chi")
    _set_var_from_locked(psi_z_var, "psi_z")
    _set_var_from_locked(theta_initial_var, "theta_initial")
    _set_var_from_locked(cor_angle_var, "cor_angle")
    _set_var_from_locked(gamma_var, "gamma")
    _set_var_from_locked(Gamma_var, "Gamma")
    _set_var_from_locked(corto_detector_var, "corto_detector")
    locked_center = params.get("center")
    if isinstance(locked_center, (list, tuple, np.ndarray)) and len(locked_center) >= 2:
        try:
            cx = float(locked_center[0])
            cy = float(locked_center[1])
            if np.isfinite(cx) and np.isfinite(cy):
                center_x_var.set(cx)
                center_y_var.set(cy)
        except Exception:
            pass

    best_params = getattr(result, "best_params", None)
    if best_params and "mosaic_params" in best_params:
        profile_cache = dict(best_params["mosaic_params"])
    else:
        profile_cache = dict(mosaic_params)
        profile_cache.update(
            {
                "sigma_mosaic_deg": sigma_deg,
                "gamma_mosaic_deg": gamma_deg,
                "eta": eta_val,
            }
        )

    last_simulation_signature = None
    schedule_update()

    residual_norm = 0.0
    if getattr(result, "fun", None) is not None and result.fun.size:
        residual_norm = float(np.linalg.norm(result.fun))
    selected_rois = list(getattr(result, "selected_rois", []) or [])
    rejected_rois = list(getattr(result, "rejected_rois", []) or [])
    roi_diagnostics = list(getattr(result, "roi_diagnostics", []) or [])
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

    initial_cost = float(getattr(result, "initial_cost", np.nan))
    final_cost = float(getattr(result, "final_cost", np.nan))
    cost_reduction = float(getattr(result, "cost_reduction", np.nan))
    outlier_fraction = float(getattr(result, "outlier_fraction", np.nan))
    acceptance_passed = bool(getattr(result, "acceptance_passed", False))
    boundary_warning = str(getattr(result, "boundary_warning", "") or "").strip()
    worst_rois = list(getattr(result, "top_worst_rois", []) or [])

    if np.isfinite(cost_reduction):
        reduction_pct = 100.0 * cost_reduction
    else:
        reduction_pct = float("nan")

    quality_line = (
        "Quality gates: passed"
        if acceptance_passed
        else "Quality gates: needs review"
    )
    if np.isfinite(outlier_fraction):
        quality_line += f" (outliers={100.0 * outlier_fraction:.1f}%)"

    worst_summary = ""
    if worst_rois:
        rows: list[str] = []
        for diag in worst_rois[:5]:
            hkl = tuple(diag.get("hkl", ()))
            rms = float(diag.get("rms", np.nan))
            if len(hkl) == 3:
                rows.append(f"{hkl}: RMS={rms:.3f}")
        if rows:
            worst_summary = "\nWorst ROIs: " + ", ".join(rows)

    warning_summary = f"\n{boundary_warning}" if boundary_warning else ""

    progress_label_mosaic.config(
        text=(
            f"Mosaic fit {status_text}\n"
            f"σ={sigma_deg:.3f}°, γ={gamma_deg:.3f}°, η={eta_val:.3f}\n"
            f"Robust cost={final_cost:.4g} (init={initial_cost:.4g}, Δ={reduction_pct:.1f}%)\n"
            f"Residual norm={residual_norm:.2f} using {roi_count} ROIs"
            f" ({len(rejected_rois)} rejected, {len(roi_diagnostics)} diagnosed)\n"
            f"{quality_line}"
            f"{warning_summary}"
            f"{peaks_summary}"
            f"{worst_summary}"
        )
    )


on_fit_geometry_click = _wrap_with_responsiveness_indicator(on_fit_geometry_click)
on_fit_mosaic_click = _wrap_with_responsiveness_indicator(on_fit_mosaic_click)


fit_button_geometry = ttk.Button(
    root,
    text="Fit Positions & Geometry",
    command=on_fit_geometry_click
)
fit_button_geometry.pack(side=tk.TOP, padx=5, pady=2)
fit_button_geometry.config(text="Fit Geometry (LSQ)", command=on_fit_geometry_click)

undo_geometry_fit_button = ttk.Button(
    root,
    text="Undo Last Geometry Fit",
    command=_undo_last_geometry_fit,
    state="disabled",
)
undo_geometry_fit_button.pack(side=tk.TOP, padx=5, pady=2)
_update_geometry_fit_undo_button_state()

clear_geometry_markers_button = ttk.Button(
    root,
    text="Clear Geometry Overlays",
    command=_clear_all_geometry_overlay_artists,
)
clear_geometry_markers_button.pack(side=tk.TOP, padx=5, pady=2)

fit_button_mosaic = ttk.Button(
    root,
    text="Fit Mosaic Widths",
    command=on_fit_mosaic_click,
)
fit_button_mosaic.pack(side=tk.TOP, padx=5, pady=2)

hkl_lookup_frame = ttk.LabelFrame(root, text="Peak Lookup (HKL)")
# Make this control easy to find: pack it above the geometry/mosaic fit buttons
# even though it is defined later in the file.
hkl_lookup_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=4, before=fit_button_geometry)

selected_h_var = tk.StringVar(value="0")
selected_k_var = tk.StringVar(value="0")
selected_l_var = tk.StringVar(value="0")

ttk.Label(hkl_lookup_frame, text="H").pack(side=tk.LEFT, padx=(4, 2))
h_entry = ttk.Entry(hkl_lookup_frame, width=5, textvariable=selected_h_var)
h_entry.pack(side=tk.LEFT, padx=(0, 6))

ttk.Label(hkl_lookup_frame, text="K").pack(side=tk.LEFT, padx=(0, 2))
k_entry = ttk.Entry(hkl_lookup_frame, width=5, textvariable=selected_k_var)
k_entry.pack(side=tk.LEFT, padx=(0, 6))

ttk.Label(hkl_lookup_frame, text="L").pack(side=tk.LEFT, padx=(0, 2))
l_entry = ttk.Entry(hkl_lookup_frame, width=5, textvariable=selected_l_var)
l_entry.pack(side=tk.LEFT, padx=(0, 8))

ttk.Button(
    hkl_lookup_frame,
    text="Select HKL",
    command=_select_peak_from_hkl_controls,
).pack(side=tk.LEFT, padx=(0, 4))

ttk.Button(
    hkl_lookup_frame,
    text="Clear",
    command=_clear_selected_peak,
).pack(side=tk.LEFT, padx=(0, 4))

ttk.Button(
    hkl_lookup_frame,
    text="Show Bragg/Ewald",
    command=_open_selected_peak_intersection_figure,
).pack(side=tk.LEFT, padx=(0, 4))

for _entry in (h_entry, k_entry, l_entry):
    _entry.bind("<Return>", lambda _event: _select_peak_from_hkl_controls())

show_1d_var = tk.BooleanVar(value=False)
def toggle_1d_plots():
    schedule_update()

check_1d = ttk.Checkbutton(
    text="Show 1D Integration",
    variable=show_1d_var,
    command=toggle_1d_plots
)
check_1d.pack(side=tk.TOP, padx=5, pady=2)

show_caked_2d_var = tk.BooleanVar(value=False)
def toggle_caked_2d():
    global caked_limits_user_override, simulation_limits_user_override
    if not show_caked_2d_var.get():
        caked_limits_user_override = False
    else:
        simulation_limits_user_override = False
    schedule_update()

check_2d = ttk.Checkbutton(
    text="Show 2D Caking",
    variable=show_caked_2d_var,
    command=toggle_caked_2d
)
check_2d.pack(side=tk.TOP, padx=5, pady=2)

log_radial_var = tk.BooleanVar(value=False)
log_azimuth_var = tk.BooleanVar(value=False)

def toggle_log_radial():
    schedule_update()

def toggle_log_azimuth():
    schedule_update()

check_log_radial = ttk.Checkbutton(
    text="Log Radial",
    variable=log_radial_var,
    command=toggle_log_radial
)
check_log_radial.pack(side=tk.TOP, padx=5, pady=2)

check_log_azimuth = ttk.Checkbutton(
    text="Log Azimuth",
    variable=log_azimuth_var,
    command=toggle_log_azimuth
)
check_log_azimuth.pack(side=tk.TOP, padx=5, pady=2)

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
    # exactly what the user sees.  ``last_1d_integration_data`` may be empty if
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

snapshot_button = ttk.Button(
    text="Save 1D Snapshot",
    command=save_1d_snapshot
)
snapshot_button.pack(side=tk.TOP, padx=5, pady=2)

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
        "a": a_var.get(),
        "c": c_var.get(),
        "center_x": center_x_var.get(),
        "center_y": center_y_var.get(),
    }

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    
    mosaic_params = {
        "beam_x_array": profile_cache.get("beam_x_array", []),
        "beam_y_array": profile_cache.get("beam_y_array", []),
        "theta_array":  profile_cache.get("theta_array", []),
        "phi_array":    profile_cache.get("phi_array", []),
        "wavelength_array": profile_cache.get("wavelength_array", []),
        "sigma_mosaic_deg": profile_cache.get("sigma_mosaic_deg", 0.0),
        "gamma_mosaic_deg": profile_cache.get("gamma_mosaic_deg", 0.0),
        "eta": profile_cache.get("eta", 0.0)
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

save_q_button = ttk.Button(
    text="Save Q-Space Snapshot",
    command=save_q_space_representation
)
save_q_button.pack(side=tk.TOP, padx=5, pady=2)

def save_1d_permutations():
    pass

save_1d_grid_button = ttk.Button(
    text="Save 1D Grid",
    command=save_1d_permutations
)
save_1d_grid_button.pack(side=tk.TOP, padx=5, pady=2)

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
        "beam_x_array": profile_cache.get("beam_x_array", []),
        "beam_y_array": profile_cache.get("beam_y_array", []),
        "theta_array":  profile_cache.get("theta_array", []),
        "phi_array":    profile_cache.get("phi_array", []),
        "wavelength_array": profile_cache.get("wavelength_array", []),
        "sigma_mosaic_deg": profile_cache.get("sigma_mosaic_deg", 0.0),
        "gamma_mosaic_deg": profile_cache.get("gamma_mosaic_deg", 0.0),
        "eta": profile_cache.get("eta", 0.0)
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


run_debug_simulation = _wrap_with_responsiveness_indicator(run_debug_simulation)

debug_button = ttk.Button(
    text="Run Debug Simulation",
    command=run_debug_simulation
)
debug_button.pack(side=tk.TOP, padx=5, pady=2)

# Button to force a full update (re-read occupancies and recalc everything).
force_update_button = ttk.Button(
    text="Force Update",
    command=lambda: update_occupancies()
)
force_update_button.pack(side=tk.TOP, padx=5, pady=2)

# Group related sliders in collapsible sections so the interface remains
# manageable as more controls are added.
geo_frame = CollapsibleFrame(left_col, text='Geometry', expanded=True)
geo_frame.pack(fill=tk.X, padx=5, pady=5)

debye_frame = CollapsibleFrame(left_col, text='Debye Parameters')
debye_frame.pack(fill=tk.X, padx=5, pady=5)

detector_frame = CollapsibleFrame(right_col, text='Detector')
detector_frame.pack(fill=tk.X, padx=5, pady=5)

lattice_frame = CollapsibleFrame(right_col, text='Lattice Parameters', expanded=True)
lattice_frame.pack(fill=tk.X, padx=5, pady=5)

mosaic_frame = CollapsibleFrame(right_col, text='Mosaic Broadening')
mosaic_frame.pack(fill=tk.X, padx=5, pady=5)

initial_resolution = defaults.get('sampling_resolution', 'Low')
if initial_resolution not in resolution_sample_counts:
    initial_resolution = 'Low'

resolution_var = tk.StringVar(value=initial_resolution)
resolution_count_var = tk.StringVar()

def _refresh_resolution_display():
    count = resolution_sample_counts.get(
        resolution_var.get(), resolution_sample_counts['Low']
    )
    resolution_count_var.set(f"{count:,} samples" if count >= 1000 else f"{count} samples")

def ensure_valid_resolution_choice():
    if resolution_var.get() not in resolution_sample_counts:
        resolution_var.set(defaults['sampling_resolution'])

num_samples = resolution_sample_counts.get(initial_resolution, num_samples)
_refresh_resolution_display()

resolution_selector_frame = ttk.Frame(mosaic_frame.frame)
resolution_selector_frame.pack(fill=tk.X, pady=5)
ttk.Label(resolution_selector_frame, text='Sampling Resolution').pack(anchor=tk.W, padx=5)
resolution_menu = ttk.OptionMenu(
    resolution_selector_frame,
    resolution_var,
    resolution_var.get(),
    *resolution_sample_counts.keys(),
)
resolution_menu.pack(fill=tk.X, padx=5, pady=(2, 0))
ttk.Label(
    resolution_selector_frame,
    textvariable=resolution_count_var,
).pack(anchor=tk.W, padx=5, pady=(2, 0))

def on_resolution_option_change(*_):
    global num_samples
    num_samples = resolution_sample_counts.get(
        resolution_var.get(), resolution_sample_counts['Low']
    )
    _refresh_resolution_display()
    update_mosaic_cache()
    schedule_update()

resolution_var.trace_add('write', on_resolution_option_change)

optics_mode_var = tk.StringVar(value=_normalize_optics_mode_label(defaults.get('optics_mode', 'fast')))
optics_mode_frame = ttk.Frame(mosaic_frame.frame)
optics_mode_frame.pack(fill=tk.X, pady=(6, 2))
ttk.Label(optics_mode_frame, text='Optics Transport').pack(anchor=tk.W, padx=5)
ttk.Radiobutton(
    optics_mode_frame,
    text='Original Fast Approx (Fresnel + Beer-Lambert)',
    variable=optics_mode_var,
    value='fast',
).pack(anchor=tk.W, padx=12)
ttk.Radiobutton(
    optics_mode_frame,
    text='Complex-k DWBA slab optics (Precise)',
    variable=optics_mode_var,
    value='exact',
).pack(anchor=tk.W, padx=12)


def on_optics_mode_change(*_):
    global last_simulation_signature
    last_simulation_signature = None
    schedule_update()


optics_mode_var.trace_add('write', on_optics_mode_change)

center_frame = CollapsibleFrame(right_col, text='Beam Center')
center_frame.pack(fill=tk.X, padx=5, pady=5)

def make_slider(
    label_str,
    min_val,
    max_val,
    init_val,
    step,
    parent,
    mosaic=False,
    **slider_kwargs,
):
    var, scale = create_slider(
        label_str,
        min_val,
        max_val,
        init_val,
        step,
        parent,
        on_mosaic_slider_change if mosaic else schedule_update,
        **slider_kwargs,
    )
    return var, scale

theta_initial_var, theta_initial_scale = make_slider(
    'Theta Initial', 0.5, 30.0, defaults['theta_initial'], 0.01, geo_frame.frame
)
cor_angle_var, cor_angle_scale = make_slider(
    'Goniometer Axis Pitch (about y)',
    -5.0,
    5.0,
    defaults['cor_angle'],
    0.01,
    geo_frame.frame,
)
gamma_var, gamma_scale = make_slider(
    'Gamma',
    -4,
    4,
    defaults['gamma'],
    0.001,
    geo_frame.frame,
    allow_range_expand=True,
)
Gamma_var, Gamma_scale = make_slider(
    'Detector Rotation Γ',
    -4,
    4,
    defaults['Gamma'],
    0.001,
    geo_frame.frame,
    allow_range_expand=True,
)
chi_var, chi_scale = make_slider(
    'Chi', -1, 1, defaults['chi'], 0.001, geo_frame.frame
)
psi_z_var, psi_z_scale = make_slider(
    'Goniometer Axis Yaw (about z)',
    -5.0,
    5.0,
    defaults['psi_z'],
    0.01,
    geo_frame.frame,
)


def _clamp_psi_z_var(*_):
    try:
        value = float(psi_z_var.get())
        lo = float(psi_z_scale.cget("from"))
        hi = float(psi_z_scale.cget("to"))
    except Exception:
        return
    if hi < lo:
        lo, hi = hi, lo
    clipped = min(max(value, lo), hi)
    if not math.isclose(value, clipped, rel_tol=0.0, abs_tol=1e-12):
        psi_z_var.set(clipped)


psi_z_var.trace_add("write", _clamp_psi_z_var)
_clamp_psi_z_var()

zs_var, zs_scale = make_slider(
    'Zs', -2.0e-3, 2e-3, defaults['zs'], 0.0001, geo_frame.frame
)
zb_var, zb_scale = make_slider(
    'Zb', -2.0e-3, 2e-3, defaults['zb'], 0.0001, geo_frame.frame
)
debye_x_var, debye_x_scale = make_slider(
    'Debye Qz', 0.0, 1.0, defaults['debye_x'], 0.001, debye_frame.frame
)
debye_y_var, debye_y_scale = make_slider(
    'Debye Qr', 0.0, 1.0, defaults['debye_y'], 0.001, debye_frame.frame
)
corto_detector_var, corto_detector_scale = make_slider(
    'CortoDetector', 0.0, 100e-3, defaults['corto_detector'], 0.1e-3, detector_frame.frame
)
a_var, a_scale = make_slider(
    'a (Å)',
    3.5,
    8.0,
    defaults['a'],
    0.01,
    lattice_frame.frame,
    allow_range_expand=True,
    range_expand_pad=0.1,
)
c_var, c_scale = make_slider(
    'c (Å)',
    20.0,
    40.0,
    defaults['c'],
    0.01,
    lattice_frame.frame,
    allow_range_expand=True,
    range_expand_pad=0.5,
)
sigma_mosaic_var, sigma_mosaic_scale = make_slider(
    'σ Mosaic (deg)', 0.0, 5.0, defaults['sigma_mosaic_deg'], 0.01, mosaic_frame.frame, mosaic=True
)
gamma_mosaic_var, gamma_mosaic_scale = make_slider(
    'γ Mosaic (deg)', 0.0, 5.0, defaults['gamma_mosaic_deg'], 0.01, mosaic_frame.frame, mosaic=True
)
eta_var, eta_scale = make_slider(
    'η (fraction)', 0.0, 1.0, defaults['eta'], 0.001, mosaic_frame.frame, mosaic=True
)
beam_center_slider_max = max(3000.0, float(image_size))
center_x_var, center_x_scale = make_slider(
    'Beam Center Row',
    0.0,
    beam_center_slider_max,
    defaults['center_x'],
    1.0,
    center_frame.frame
)
center_y_var, center_y_scale = make_slider(
    'Beam Center Col',
    0.0,
    beam_center_slider_max,
    defaults['center_y'],
    1.0,
    center_frame.frame
)

geometry_fit_parameter_specs = {
    "zb": {
        "label": "Zb",
        "value_var": zb_var,
        "value_slider": zb_scale,
        "step": 0.0001,
    },
    "zs": {
        "label": "Zs",
        "value_var": zs_var,
        "value_slider": zs_scale,
        "step": 0.0001,
    },
    "theta_initial": {
        "label": "Theta Initial",
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
        "label": "Chi",
        "value_var": chi_var,
        "value_slider": chi_scale,
        "step": 0.001,
    },
    "cor_angle": {
        "label": "Goniometer Axis Pitch (about y)",
        "value_var": cor_angle_var,
        "value_slider": cor_angle_scale,
        "step": 0.001,
    },
    "gamma": {
        "label": "gamma",
        "value_var": gamma_var,
        "value_slider": gamma_scale,
        "step": 0.001,
    },
    "Gamma": {
        "label": "Gamma",
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
    spec = geometry_fit_parameter_specs.get(name, {})
    try:
        current_value = float(spec["value_var"].get())
    except Exception:
        current_value = 0.0
    try:
        step = abs(float(spec.get("step", 0.01)))
    except Exception:
        step = 0.01
    step = max(step, 1.0e-6)
    domain = _current_geometry_fit_parameter_domains([name]).get(name)
    domain_span = 0.0
    if isinstance(domain, tuple) and len(domain) >= 2:
        domain_span = max(0.0, float(domain[1]) - float(domain[0]))

    fit_geometry_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(fit_geometry_cfg, dict):
        fit_geometry_cfg = {}
    bounds_cfg = fit_geometry_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}
    entry = bounds_cfg.get(name)

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
        fallback_window = max(
            step * 10.0,
            0.02 * domain_span,
            0.1 * max(abs(current_value), 1.0),
        )
        default_window = fallback_window

    if domain_span > 0.0:
        default_window = min(default_window, domain_span)

    return max(float(default_window), step)


def _default_geometry_fit_pull(name: str, window: float) -> float:
    fit_geometry_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(fit_geometry_cfg, dict):
        fit_geometry_cfg = {}
    priors_cfg = fit_geometry_cfg.get("priors", {}) or {}
    if not isinstance(priors_cfg, dict):
        priors_cfg = {}
    entry = priors_cfg.get(name)
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


geometry_fit_constraints_frame = CollapsibleFrame(
    root,
    text="Geometry Fit Constraints",
    expanded=True,
)
geometry_fit_constraints_frame.pack(
    side=tk.TOP,
    fill=tk.X,
    padx=5,
    pady=5,
    after=fit_frame,
)
ttk.Label(
    geometry_fit_constraints_frame.frame,
    text=(
        "Each window is applied as current value \u00b1 deviation during geometry fitting. "
        "Stay-close adds a soft pull back to the starting guess."
    ),
    wraplength=420,
    justify="left",
).pack(fill=tk.X, padx=6, pady=(2, 6))

geometry_fit_constraints_list_frame = ttk.Frame(geometry_fit_constraints_frame.frame)
geometry_fit_constraints_list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

geometry_fit_constraints_canvas_height = int(
    min(max(root.winfo_screenheight() * 0.35, 220), 420)
)
geometry_fit_constraints_canvas = tk.Canvas(
    geometry_fit_constraints_list_frame,
    highlightthickness=0,
    borderwidth=0,
    height=geometry_fit_constraints_canvas_height,
)
geometry_fit_constraints_scrollbar = ttk.Scrollbar(
    geometry_fit_constraints_list_frame,
    orient=tk.VERTICAL,
    command=geometry_fit_constraints_canvas.yview,
)
geometry_fit_constraints_canvas.configure(
    yscrollcommand=geometry_fit_constraints_scrollbar.set
)
geometry_fit_constraints_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
geometry_fit_constraints_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

geometry_fit_constraints_body = ttk.Frame(geometry_fit_constraints_canvas)
geometry_fit_constraints_body_window = geometry_fit_constraints_canvas.create_window(
    (0, 0),
    window=geometry_fit_constraints_body,
    anchor="nw",
)


def _refresh_geometry_fit_constraints_scrollregion(_event=None) -> None:
    geometry_fit_constraints_canvas.configure(
        scrollregion=geometry_fit_constraints_canvas.bbox("all")
    )


def _resize_geometry_fit_constraints_body(event) -> None:
    geometry_fit_constraints_canvas.itemconfigure(
        geometry_fit_constraints_body_window,
        width=event.width,
    )


def _scroll_geometry_fit_constraints(event):
    pointer_x = root.winfo_pointerx()
    pointer_y = root.winfo_pointery()
    canvas_x = geometry_fit_constraints_canvas.winfo_rootx()
    canvas_y = geometry_fit_constraints_canvas.winfo_rooty()
    if not (
        canvas_x <= pointer_x <= canvas_x + geometry_fit_constraints_canvas.winfo_width()
        and canvas_y <= pointer_y <= canvas_y + geometry_fit_constraints_canvas.winfo_height()
    ):
        return None

    delta = 0
    if getattr(event, "delta", 0):
        delta = int(-event.delta / 120)
    elif getattr(event, "num", None) == 4:
        delta = -1
    elif getattr(event, "num", None) == 5:
        delta = 1
    if delta:
        geometry_fit_constraints_canvas.yview_scroll(delta, "units")
        return "break"
    return None


geometry_fit_constraints_body.bind(
    "<Configure>",
    _refresh_geometry_fit_constraints_scrollregion,
)
geometry_fit_constraints_canvas.bind(
    "<Configure>",
    _resize_geometry_fit_constraints_body,
)
root.bind_all("<MouseWheel>", _scroll_geometry_fit_constraints, add="+")
root.bind_all("<Button-4>", _scroll_geometry_fit_constraints, add="+")
root.bind_all("<Button-5>", _scroll_geometry_fit_constraints, add="+")

for name in GEOMETRY_FIT_PARAM_ORDER:
    spec = geometry_fit_parameter_specs.get(name, {})
    try:
        step = abs(float(spec.get("step", 0.01)))
    except Exception:
        step = 0.01
    step = max(step, 1.0e-6)
    default_window = _default_geometry_fit_window(name)
    domain = _current_geometry_fit_parameter_domains([name]).get(name)
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

    row = ttk.LabelFrame(
        geometry_fit_constraints_body,
        text=str(spec.get("label", name)),
    )
    window_var, _window_slider = create_slider(
        "Allowed deviation (\u00b1)",
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
    geometry_fit_constraint_controls[name] = {
        "row": row,
        "window_var": window_var,
        "pull_var": pull_var,
        "_mapped": False,
    }

for _toggle_var in geometry_fit_toggle_vars.values():
    _toggle_var.trace_add("write", _sync_geometry_fit_constraint_rows)
_sync_geometry_fit_constraint_rows()

# Slider controlling contribution of the first CIF file, only if a second CIF
# was provided.
if has_second_cif:
    weights_frame = CollapsibleFrame(right_col, text='CIF Weights')
    weights_frame.pack(fill=tk.X, padx=5, pady=5)
    weight1_var, _ = make_slider(
        'CIF1 Weight', 0.0, 1.0, weight1, 0.01, weights_frame.frame
    )
    weight2_var, _ = make_slider(
        'CIF2 Weight', 0.0, 1.0, weight2, 0.01, weights_frame.frame
    )

    def update_weights(*args):
        """Recompute intensities using the current CIF weights."""
        global intensities, df_summary, last_simulation_signature
        w1 = weight1_var.get()
        w2 = weight2_var.get()
        intensities = w1 * intensities_cif1 + w2 * intensities_cif2
        max_I = intensities.max() if intensities.size else 0.0
        if max_I > 0:
            intensities = intensities * (100.0 / max_I)

        df_summary['Intensity'] = intensities
        last_simulation_signature = None
        schedule_update()

    weight1_var.trace_add('write', update_weights)
    weight2_var.trace_add('write', update_weights)
else:
    weight1_var = tk.DoubleVar(value=1.0)
    weight2_var = tk.DoubleVar(value=0.0)
# ---------------------------------------------------------------------------
#  OCCUPANCY SLIDERS: Sliders for occ[0], occ[1], occ[2]
# ---------------------------------------------------------------------------
occ_var1 = tk.DoubleVar(value=occ[0])
occ_var2 = tk.DoubleVar(value=occ[1])
occ_var3 = tk.DoubleVar(value=occ[2])
finite_stack_var = tk.BooleanVar(value=defaults['finite_stack'])
stack_layers_var = tk.IntVar(value=int(defaults['stack_layers']))
_layers_scale_widget = None
_layers_entry_widget = None
_layers_entry_var = None


def _rebuild_diffraction_inputs(
    new_occ,
    p_vals,
    weights,
    c_axis,
    *,
    force=False,
    trigger_update=True,
):
    """Refresh cached HT curves and peak lists for the current settings."""

    global miller, intensities, degeneracy, details
    global df_summary, df_details
    global last_simulation_signature
    global SIM_MILLER1, SIM_INTENS1, SIM_MILLER2, SIM_INTENS2
    global ht_curves_cache, ht_cache_multi, _last_occ_for_ht, _last_p_triplet, _last_weights
    global _last_c_for_ht, _last_finite_stack, _last_stack_layers
    global intensities_cif1, intensities_cif2

    finite_flag = bool(finite_stack_var.get())
    layers = int(max(1, stack_layers_var.get()))

    if (
        not force
        and list(new_occ) == _last_occ_for_ht
        and list(p_vals) == _last_p_triplet
        and list(weights) == _last_weights
        and math.isclose(float(c_axis), _last_c_for_ht, rel_tol=1e-9, abs_tol=1e-9)
        and _last_finite_stack == finite_flag
        and (not finite_flag or _last_stack_layers == layers)
    ):
        last_simulation_signature = None
        if trigger_update:
            schedule_update()
        return

    def get_cache(label, p_val):
        cache = ht_cache_multi.get(label)
        if (
            cache is None
            or cache["p"] != p_val
            or list(cache["occ"]) != list(new_occ)
            or cache.get("two_theta_max") != two_theta_range[1]
            or not math.isclose(cache.get("c", float("nan")), float(c_axis), rel_tol=1e-9, abs_tol=1e-9)
            or bool(cache.get("finite_stack")) != finite_flag
            or (finite_flag and cache.get("stack_layers") != layers)
        ):
            cache = build_ht_cache(p_val, new_occ, c_axis, finite_flag, layers)
            ht_cache_multi[label] = cache
        return cache

    caches = [
        get_cache("p0", p_vals[0]),
        get_cache("p1", p_vals[1]),
        get_cache("p2", p_vals[2]),
    ]

    combined_ht_local = combine_ht_dicts(caches, weights)
    combined_qr_local = ht_dict_to_qr_dict(combined_ht_local)
    arrays_local = ht_dict_to_arrays(combined_ht_local)
    ht_curves_cache = {
        "curves": combined_ht_local,
        "qr_curves": combined_qr_local,
        "arrays": arrays_local,
        "c": float(c_axis),
        "finite_stack": finite_flag,
        "stack_layers": layers,
    }
    _last_occ_for_ht = list(new_occ)
    _last_p_triplet = list(p_vals)
    _last_weights = list(weights)
    _last_c_for_ht = float(c_axis)
    _last_finite_stack = finite_flag
    _last_stack_layers = layers

    m1, i1, d1, det1 = arrays_local

    deg_dict1 = {tuple(m1[i]): int(d1[i]) for i in range(len(m1))}
    det_dict1 = {tuple(m1[i]): det1[i] for i in range(len(m1))}

    if has_second_cif:
        m2, i2, d2, det2 = miller_generator(
            mx,
            cif_file2,
            new_occ,
            lambda_,
            energy,
            intensity_threshold,
            two_theta_range,
        )
        if include_rods_flag:
            m2, i2 = inject_fractional_reflections(m2, i2, mx)

        deg_dict2 = {tuple(m2[i]): int(d2[i]) for i in range(len(m2))}
        det_dict2 = {tuple(m2[i]): det2[i] for i in range(len(m2))}

        union = {tuple(h) for h in m1} | {tuple(h) for h in m2}
        miller = np.array(sorted(union), dtype=float)
        int1 = {tuple(h): v for h, v in zip(m1, i1)}
        int2 = {tuple(h): v for h, v in zip(m2, i2)}
        intensities_cif1 = np.array([int1.get(tuple(h), 0.0) for h in miller])
        intensities_cif2 = np.array([int2.get(tuple(h), 0.0) for h in miller])
        w1 = weight1_var.get()
        w2 = weight2_var.get()
        intensities = w1 * intensities_cif1 + w2 * intensities_cif2
        max_I = intensities.max() if intensities.size else 0.0
        if max_I > 0:
            intensities = intensities * (100.0 / max_I)

        SIM_MILLER1 = m1
        SIM_INTENS1 = i1
        SIM_MILLER2 = m2
        SIM_INTENS2 = i2

        degeneracy = np.array(
            [deg_dict1.get(tuple(h), 0) + deg_dict2.get(tuple(h), 0) for h in miller],
            dtype=np.int32,
        )
        details = [
            det_dict1.get(tuple(h), []) + det_dict2.get(tuple(h), [])
            for h in miller
        ]
    else:
        miller = m1
        intensities_cif1 = i1
        intensities_cif2 = np.zeros_like(intensities_cif1)
        intensities = intensities_cif1
        degeneracy = d1
        details = det1

        SIM_MILLER1 = m1
        SIM_INTENS1 = i1
        SIM_MILLER2 = np.empty((0, 3), dtype=np.int32)
        SIM_INTENS2 = np.empty((0,), dtype=np.float64)

    df_summary, df_details = build_intensity_dataframes(
        miller, intensities, degeneracy, details
    )

    last_simulation_signature = None
    if trigger_update:
        schedule_update()


_rebuild_diffraction_inputs = _wrap_with_responsiveness_indicator(
    _rebuild_diffraction_inputs
)


def update_occupancies(*args):
    """Recompute Hendricks–Teller curves when occupancies or p-values change."""

    new_occ = [occ_var1.get(), occ_var2.get(), occ_var3.get()]
    p_vals = [p0_var.get(), p1_var.get(), p2_var.get()]
    w_raw = [w0_var.get(), w1_var.get(), w2_var.get()]
    w_sum = sum(w_raw) or 1.0
    weights = [w / w_sum for w in w_raw]

    _rebuild_diffraction_inputs(new_occ, p_vals, weights, c_var.get())


def _sync_finite_controls():
    global _layers_scale_widget, _layers_entry_widget
    state = tk.NORMAL if finite_stack_var.get() else tk.DISABLED
    if _layers_scale_widget is not None:
        _layers_scale_widget.configure(state=state)
    if _layers_entry_widget is not None:
        _layers_entry_widget.configure(state=state)


def _normalize_layer_value(raw_value):
    try:
        value = int(round(float(raw_value)))
    except (TypeError, ValueError):
        value = stack_layers_var.get()
    if value < 1:
        value = 1
    return value


def _sync_layer_entry_from_var(*_):
    global _layers_entry_var
    if _layers_entry_var is None:
        return
    normalized = str(int(max(1, stack_layers_var.get())))
    if _layers_entry_var.get().strip() != normalized:
        _layers_entry_var.set(normalized)


def _commit_layer_entry(_event=None):
    global _layers_scale_widget, _layers_entry_var
    if _layers_entry_var is None:
        return
    value = _normalize_layer_value(_layers_entry_var.get())
    if _layers_scale_widget is not None:
        current_to = int(round(float(_layers_scale_widget.cget("to"))))
        if value > current_to:
            _layers_scale_widget.configure(to=value)
    changed = stack_layers_var.get() != value
    if changed:
        stack_layers_var.set(value)
    _sync_layer_entry_from_var()
    if changed and finite_stack_var.get():
        update_occupancies()


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
stack_frame = CollapsibleFrame(right_col, text='Stacking Probabilities')
stack_frame.pack(fill=tk.X, padx=5, pady=5)
finite_frame = ttk.Frame(stack_frame.frame)
finite_frame.pack(fill=tk.X, padx=5, pady=5)
ttk.Checkbutton(
    finite_frame,
    text="Finite Stack",
    variable=finite_stack_var,
    command=_on_finite_toggle,
).pack(anchor=tk.W, padx=5, pady=2)

layers_row = ttk.Frame(finite_frame)
layers_row.pack(fill=tk.X, padx=5, pady=2)
ttk.Label(layers_row, text="Layers:").grid(row=0, column=0, sticky="w")
_layers_entry_var = tk.StringVar(value=str(int(max(1, stack_layers_var.get()))))
layers_entry = ttk.Entry(
    layers_row,
    textvariable=_layers_entry_var,
    width=8,
    justify="right",
)
layers_entry.grid(row=0, column=2, sticky="e", padx=(5, 0))
layers_entry.bind("<Return>", _commit_layer_entry)
layers_entry.bind("<FocusOut>", _commit_layer_entry)

layers_scale = tk.Scale(
    layers_row,
    from_=1,
    to=1000,
    orient=tk.HORIZONTAL,
    resolution=1,
    showvalue=False,
    variable=stack_layers_var,
    command=_on_layer_slider,
)
layers_scale.grid(row=0, column=1, sticky="ew", padx=(5, 5))
layers_row.columnconfigure(1, weight=1)

_layers_scale_widget = layers_scale
_layers_entry_widget = layers_entry
stack_layers_var.trace_add("write", _sync_layer_entry_from_var)
_sync_layer_entry_from_var()
_sync_finite_controls()
p0_var, _ = create_slider('p≈0', 0.0, 0.2, defaults['p0'], 0.001,
                          stack_frame.frame, update_occupancies)
w0_var, _ = create_slider('w(p≈0)%', 0.0, 100.0, defaults['w0'], 0.1,
                          stack_frame.frame, update_occupancies)
p1_var, _ = create_slider('p≈1', 0.8, 1.0, defaults['p1'], 0.001,
                          stack_frame.frame, update_occupancies)
w1_var, _ = create_slider('w(p≈1)%', 0.0, 100.0, defaults['w1'], 0.1,
                          stack_frame.frame, update_occupancies)
p2_var, _ = create_slider('p', 0.0, 1.0, defaults['p2'], 0.001,
                          stack_frame.frame, update_occupancies)
w2_var, _ = create_slider('w(p)%', 0.0, 100.0, defaults['w2'], 0.1,
                          stack_frame.frame, update_occupancies)

# Occupancy sliders grouped in a collapsible frame
occ_frame = CollapsibleFrame(right_col, text='Site Occupancies')
occ_frame.pack(fill=tk.X, padx=5, pady=5)

ttk.Label(occ_frame.frame, text="Occupancy Site 1").pack(padx=5, pady=2)
occ_scale1 = ttk.Scale(
    occ_frame.frame,
    from_=0.0,
    to=1.0,
    orient=tk.HORIZONTAL,
    variable=occ_var1,
    command=update_occupancies
)
occ_scale1.pack(fill=tk.X, padx=5, pady=2)

ttk.Label(occ_frame.frame, text="Occupancy Site 2").pack(padx=5, pady=2)
occ_scale2 = ttk.Scale(
    occ_frame.frame,
    from_=0.0,
    to=1.0,
    orient=tk.HORIZONTAL,
    variable=occ_var2,
    command=update_occupancies
)
occ_scale2.pack(fill=tk.X, padx=5, pady=2)

ttk.Label(occ_frame.frame, text="Occupancy Site 3").pack(padx=5, pady=2)
occ_scale3 = ttk.Scale(
    occ_frame.frame,
    from_=0.0,
    to=1.0,
    orient=tk.HORIZONTAL,
    variable=occ_var3,
    command=update_occupancies
)
occ_scale3.pack(fill=tk.X, padx=5, pady=2)

# --- Add numeric input fields and a Force Update button ---
occ_entry_frame = ttk.Frame(occ_frame.frame)
occ_entry_frame.pack(fill=tk.X, padx=5, pady=5)

# Occupancy input for Site 1.
ttk.Label(occ_entry_frame, text="Input Occupancy Site 1:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
occ_entry1 = ttk.Entry(occ_entry_frame, textvariable=occ_var1, width=5)
occ_entry1.grid(row=0, column=1, padx=5, pady=2)

# Occupancy input for Site 2.
ttk.Label(occ_entry_frame, text="Input Occupancy Site 2:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
occ_entry2 = ttk.Entry(occ_entry_frame, textvariable=occ_var2, width=5)
occ_entry2.grid(row=1, column=1, padx=5, pady=2)

# Occupancy input for Site 3.
ttk.Label(occ_entry_frame, text="Input Occupancy Site 3:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
occ_entry3 = ttk.Entry(occ_entry_frame, textvariable=occ_var3, width=5)
occ_entry3.grid(row=2, column=1, padx=5, pady=2)

def main(write_excel_flag=None):
    """Entry point for running the GUI application.

    Parameters
    ----------
    write_excel_flag : bool or None, optional
        When ``True`` the initial intensities are written to an Excel
        file in the configured downloads directory.  When ``None`` the
        value from the instrument configuration file is used.
    """

    global write_excel
    if write_excel_flag is not None:
        write_excel = write_excel_flag

    params_file_path = get_path("parameters_file")
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
            None,
            optics_mode_var=optics_mode_var,
        )
        ensure_valid_resolution_choice()
        print("Loaded saved profile from", params_file_path)
    else:
        print("No saved profile found; using default parameters.")

    export_initial_excel()
    update_mosaic_cache()
    do_update()
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RA Simulation GUI")
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Do not write the initial intensity Excel file",
    )
    args = parser.parse_args()

    try:
        override_flag = False if args.no_excel else None
        main(write_excel_flag=override_flag)
    except Exception as exc:
        print("Unhandled exception during startup:", exc)
        import traceback
        traceback.print_exc()
