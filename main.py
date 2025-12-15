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
from pathlib import Path
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
    qr_dict_to_arrays,
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
)
from ra_sim.fitting.optimization import (
    simulate_and_compare_hkl,
    fit_geometry_parameters,
    fit_mosaic_widths_separable,
)
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import (
    hit_tables_to_max_positions,
    process_peaks_parallel,
    process_qr_rods_parallel,
)
from ra_sim.simulation.diffraction_debug import (
    process_peaks_parallel_debug,
    process_qr_rods_parallel_debug,
    dump_debug_log,
)
from ra_sim.simulation.simulation import simulate_diffraction
from ra_sim.gui.sliders import create_slider
from ra_sim.debug_utils import debug_print, is_debug_enabled
from ra_sim.hbn import estimate_detector_tilt, load_bundle_npz, load_tilt_hint
from ra_sim.gui.collapsible import CollapsibleFrame


turbo = matplotlib.colormaps.get_cmap('turbo').resampled(256)
turbo_rgba = turbo(np.linspace(0, 1, 256))
turbo_rgba[0] = [1.0, 1.0, 1.0, 1.0]       # make the 0-bin white
turbo_white0 = ListedColormap(turbo_rgba, name='turbo_white0')
turbo_white0.set_bad('white')              # NaNs will also show white


# Force TkAgg backend to ensure GUI usage
matplotlib.use('TkAgg')
# Enable extra diagnostics when the RA_SIM_DEBUG environment variable is set.
DEBUG_ENABLED = is_debug_enabled()
if DEBUG_ENABLED:
    print("Debug mode active (RA_SIM_DEBUG=1)")
    from ra_sim.debug_utils import enable_numba_logging
    enable_numba_logging()
else:
    print("Debug mode off (set RA_SIM_DEBUG=1 for extra output)")


###############################################################################
#                          DATA & PARAMETER SETUP
###############################################################################
from ra_sim.path_config import get_path, get_dir, get_instrument_config


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

file_path = get_path("dark_image")
BI = read_osc(file_path)  # Dark (background) image

osc_files = get_path("osc_files")
if isinstance(osc_files, str):
    osc_files = [osc_files]
background_images = [read_osc(path) for path in osc_files]
if not background_images:
    raise ValueError("No oscillation images configured in osc_files")

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
if tilt_hint:
    Gamma_initial = float(tilt_hint.get("rot1_rad", Gamma_initial))
    gamma_initial = float(tilt_hint.get("rot2_rad", gamma_initial))
    print(
        "Initialized detector tilt from last hBN fit profile "
        f"(Rot1={Gamma_initial:.4f} rad, Rot2={gamma_initial:.4f} rad)."
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
zb = sample_config.get("zb", 0.0)
bw_sigma = beam_config.get("bandwidth_sigma_fraction", 0.05e-3) * fwhm2sigma
zs = sample_config.get("zs", 0.0)
debye_x = debye_config.get("x", 0.0)
debye_y = debye_config.get("y", 0.0)
n2 = IndexofRefraction()

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
a_text = blk["_cell_length_a"]
b_text = blk["_cell_length_b"]
c_text = blk["_cell_length_c"]

# strip the '(uncertainty)' and cast
def parse_cif_num(txt):
    # match leading numeric part, e.g. '4.14070' out of '4.14070(3)'
    m = re.match(r"[-+0-9\.Ee]+", txt)
    if not m:
        raise ValueError(f"Can't parse '{txt}' as a number")
    return float(m.group(0))

av = parse_cif_num(a_text)
bv = parse_cif_num(b_text)
cv = parse_cif_num(c_text) * 3

if cif_file2:
    cf2  = CifFile.ReadCif(cif_file2)
    blk2 = cf2[list(cf2.keys())[0]]
    a2_text = blk2.get("_cell_length_a")
    c2_text = blk2.get("_cell_length_c")
    av2 = parse_cif_num(a2_text) if a2_text else av
    cv2 = parse_cif_num(c2_text) if c2_text else cv
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
}

# ---------------------------------------------------------------------------
# Replace the old miller_generator call with the new Hendricks–Teller helper.
# ---------------------------------------------------------------------------
def build_ht_cache(p_val, occ_vals, c_axis, finite_stack_flag, stack_layers_count):
    layers = int(max(1, stack_layers_count))
    curves = ht_Iinf_dict(
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
    qr = ht_dict_to_qr_dict(curves)
    arrays = qr_dict_to_arrays(qr)
    return {
        "p": p_val,
        "occ": tuple(occ_vals),
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

def combine_qr_dicts(caches, weights):
    import numpy as np
    out = {}
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
    return out

weights_init = np.array([defaults['w0'], defaults['w1'], defaults['w2']], dtype=float)
weights_init /= weights_init.sum() if weights_init.sum() else 1.0
combined_qr = combine_qr_dicts(
    [ht_cache_multi['p0'], ht_cache_multi['p1'], ht_cache_multi['p2']],
    weights_init,
)
miller1, intens1, degeneracy1, details1 = qr_dict_to_arrays(combined_qr)
ht_curves_cache = {
    "curves": combined_qr,
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

# Background and simulated overlays are both rotated for display. Negative ``k``
# rotates clockwise, positive rotates counter-clockwise. The simulation uses a
# fixed offset relative to the background so they stay aligned without
# additional user interaction.
DISPLAY_ROTATE_K = -1
SIM_DISPLAY_ROTATE_K = DISPLAY_ROTATE_K - 3

# Keep the real images but rotate them for display so they start aligned with
# the simulated overlay orientation.
background_images = [np.rot90(img, DISPLAY_ROTATE_K) for img in background_images]

current_background_image = background_images[0]
current_background_index = 0
background_visible = True


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


def _unrotate_display_peaks(measured, rotated_shape):
    """Map displayed peak coordinates into the simulation's native orientation."""

    if measured is None:
        return []

    unrotated = []
    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _rotate_point_for_display(
                    updated["x"], updated["y"], rotated_shape, -SIM_DISPLAY_ROTATE_K
                )
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _rotate_point_for_display(
                    updated["x_pix"], updated["y_pix"], rotated_shape, -SIM_DISPLAY_ROTATE_K
                )
            unrotated.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _rotate_point_for_display(
                seq[3], seq[4], rotated_shape, -SIM_DISPLAY_ROTATE_K
            )
            unrotated.append(type(entry)(seq))
        else:
            unrotated.append(entry)

    return unrotated


def _native_sim_to_display_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Rotate native simulation coordinates into the displayed frame."""

    return _rotate_point_for_display(col, row, image_shape, SIM_DISPLAY_ROTATE_K)


def _transform_points_orientation(
    points: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
) -> list[tuple[float, float]]:
    """Apply flips/rotations to a list of (col, row) points for diagnostics."""

    height, width = shape
    transformed: list[tuple[float, float]] = []

    for col, row in points:
        col_t, row_t = float(col), float(row)
        if flip_x:
            col_t = width - 1.0 - col_t
        if flip_y:
            row_t = height - 1.0 - row_t
        col_t, row_t = _rotate_point_for_display(col_t, row_t, shape, k)
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

    def _describe(k: int, flip_x: bool, flip_y: bool) -> str:
        parts: list[str] = []
        if k % 4:
            parts.append(f"rot{(k % 4) * 90}° CCW")
        if flip_x:
            parts.append("flip_x")
        if flip_y:
            parts.append("flip_y")
        return " + ".join(parts) if parts else "identity"

    best = None
    for k in range(4):
        for flip_x in (False, True):
            for flip_y in (False, True):
                transformed = _transform_points_orientation(
                    meas_coords, shape, k=k, flip_x=flip_x, flip_y=flip_y
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
                    "rms": rms,
                    "mean": mean,
                    "label": _describe(k, flip_x, flip_y),
                }
                if best is None or candidate["rms"] < best["rms"]:
                    best = candidate

    return best


measured_peaks_raw = np.load(get_path("measured_peaks"), allow_pickle=True)
measured_peaks = _rotate_measured_peaks_for_display(
    measured_peaks_raw,
    current_background_image.shape,
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
    current_background_image,
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

# -----------------------------------------------------------
# 2)  Mouse‑click handler
# -----------------------------------------------------------
def on_canvas_click(event):
    if event.inaxes is not ax or event.xdata is None:
        return                               # click was outside the image
    if not peak_positions:                   # no simulation yet
        progress_label_positions.config(text="Run a simulation first.")
        return

    # (x,y) from Matplotlib → integer detector pixels
    cx, cy = int(round(event.xdata)), int(round(event.ydata))

    # Find nearest stored peak
    best_i, best_d2 = -1, float("inf")
    for i, (px, py) in enumerate(peak_positions):
        if px < 0:              # invalid entries kept as (-1,-1)
            continue
        d2 = (px - cx)**2 + (py - cy)**2
        if d2 < best_d2:
            best_i, best_d2 = i, d2

    if best_i == -1:
        progress_label_positions.config(text="No peaks on screen.")
        return

    # Update GUI
    px, py      = peak_positions[best_i]
    H,K,L       = peak_millers[best_i]
    I           = peak_intensities[best_i]
    selected_peak_marker.set_data([px], [py])
    selected_peak_marker.set_visible(True)

    progress_label_positions.config(
        text=f"Nearest peak: HKL=({H} {K} {L})  "
             f"pixel=({px},{py})  Δ={best_d2**0.5:.1f}px  I={I:.2g}"
    )
    canvas.draw_idle()

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


background_min_candidate = _finite_percentile(current_background_image, 1, 0.0)
background_vmin_default = 0.0
_, background_vmax_default = _ensure_valid_range(
    background_vmin_default,
    _finite_percentile(current_background_image, 99, 1.0),
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
    slider_min = float(scale_factor_slider.cget("from"))
    slider_max = float(scale_factor_slider.cget("to"))
    if not np.isfinite(value):
        value = 1.0
    value = float(value)
    value = min(max(value, slider_min), slider_max)
    suppress_scale_factor_callback = True
    simulation_scale_factor_var.set(value)
    suppress_scale_factor_callback = False
    if reset_override:
        scale_factor_user_override = False


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


def _update_chi_square_display():
    try:
        if (
            background_visible
            and current_background_image is not None
            and global_image_buffer.size
        ):
            sim_vals = np.asarray(global_image_buffer, dtype=float)
            bg_vals = np.asarray(current_background_image, dtype=float)
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
            if background_visible and current_background_image is not None:
                background_display.set_data(current_background_image)
                background_display.set_visible(True)
            else:
                background_display.set_visible(False)
        canvas.draw_idle()
        _update_chi_square_display()
        return

    scale = simulation_scale_factor_var.get()
    scaled_image = unscaled_image_global * scale
    global_image_buffer[:] = scaled_image

    if update_limits or not simulation_limits_user_override:
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
        image_display.set_data(scaled_caked)
        image_display.set_extent(last_caked_extent)

    if show_caked_2d_var.get():
        image_display.set_clim(
            float(vmin_caked_var.get()),
            float(vmax_caked_var.get()),
        )

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
        if background_visible and current_background_image is not None:
            background_display.set_data(current_background_image)
            background_display.set_visible(True)
        else:
            background_display.set_visible(False)

    canvas.draw_idle()
    _update_chi_square_display()
_update_background_slider_defaults(current_background_image, reset_override=True)


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

def tth_min_slider_command(val):
    val_f = float(val)
    tth_min_var.set(val_f)
    tth_min_label_var.set(f"{val_f:.1f}")
    schedule_update()

def tth_max_slider_command(val):
    val_f = float(val)
    tth_max_var.set(val_f)
    tth_max_label_var.set(f"{val_f:.1f}")
    schedule_update()

def phi_min_slider_command(val):
    val_f = float(val)
    phi_min_var.set(val_f)
    phi_min_label_var.set(f"{val_f:.1f}")
    schedule_update()

def phi_max_slider_command(val):
    val_f = float(val)
    phi_max_var.set(val_f)
    phi_max_label_var.set(f"{val_f:.1f}")
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
ttk.Label(tth_min_container, textvariable=tth_min_label_var, width=5).pack(side=tk.LEFT, padx=5)

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
ttk.Label(tth_max_container, textvariable=tth_max_label_var, width=5).pack(side=tk.LEFT, padx=5)

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
ttk.Label(phi_min_container, textvariable=phi_min_label_var, width=5).pack(side=tk.LEFT, padx=5)

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
ttk.Label(phi_max_container, textvariable=phi_max_label_var, width=5).pack(side=tk.LEFT, padx=5)

PHI_ZERO_OFFSET_DEGREES = -90.0


def _adjust_phi_zero(phi_values):
    """Shift azimuthal values so φ=0 is rotated clockwise by ``PHI_ZERO_OFFSET_DEGREES``."""

    return np.asarray(phi_values) - PHI_ZERO_OFFSET_DEGREES


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

def update_mosaic_cache():
    """
    Regenerate random mosaic profiles if mosaic sliders changed.
    """
    global profile_cache
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
        "sigma_mosaic_deg": sigma_mosaic_var.get(),
        "gamma_mosaic_deg": gamma_mosaic_var.get(),
        "eta": eta_var.get()
    }

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
    """Throttle updates so heavy simulations don't overlap."""
    global update_pending
    if update_pending is not None:
        root.after_cancel(update_pending)
    update_pending = root.after(100, do_update)

peak_positions = []
peak_millers = []
peak_intensities = []

prev_background_visible = True
last_bg_signature = None
last_sim_signature = None
last_simulation_signature = None
stored_max_positions_local = None
stored_sim_image = None

###############################################################################
#                              MAIN UPDATE
###############################################################################
def do_update():
    global update_pending, update_running, last_simulation_signature
    global unscaled_image_global, background_visible
    global stored_max_positions_local, stored_sim_image
    global SIM_MILLER1, SIM_INTENS1, SIM_MILLER2, SIM_INTENS2
    global av2, cv2
    global last_caked_image_unscaled, last_caked_extent

    if update_running:
        # another update is in progress; try again shortly
        update_pending = root.after(100, do_update)
        return

    update_pending = None
    update_running = True

    gamma_updated      = float(gamma_var.get())
    Gamma_updated      = float(Gamma_var.get())
    chi_updated        = float(chi_var.get())
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


    def get_sim_signature():
        return (
            round(gamma_updated, 6),
            round(Gamma_updated, 6),
            round(chi_updated, 6),
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
            round(mosaic_params["eta"], 6)
        )

    # 1 – place near other globals

    # … inside do_update() …
    global stored_max_positions_local        # <- add

    new_sim_sig = get_sim_signature()
    global peak_positions, peak_millers, peak_intensities
    if new_sim_sig != last_simulation_signature:
        last_simulation_signature = new_sim_sig
        peak_positions.clear()
        peak_millers.clear()
        peak_intensities.clear()

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
                ) + (None,)

        img1, maxpos1, _, _, _, _, _ = run_one(ht_curves_cache["curves"], None, a_updated, c_updated)
        if SIM_MILLER2.size > 0:
            img2, maxpos2, _, _, _, _, _ = run_one(SIM_MILLER2, SIM_INTENS2, av2, cv2)
        else:
            img2 = np.zeros_like(img1)
            maxpos2 = []

        w1 = weight1_var.get()
        w2 = weight2_var.get()
        updated_image = w1 * img1 + w2 * img2
        max_positions_local = list(maxpos1) + list(maxpos2)
        stored_max_positions_local = max_positions_local
        stored_sim_image = updated_image
    else:
        # fall back to the cached arrays
        if stored_max_positions_local is None:
            # first run after programme start – force a simulation
            last_simulation_signature = None
            update_running = False
            return do_update()          # re-enter with computation path
        max_positions_local = stored_max_positions_local
        updated_image       = stored_sim_image

    display_image = np.rot90(updated_image, SIM_DISPLAY_ROTATE_K)
    
    # ───── NEW: build peak lists from hit_tables ───────────────────────────
    peak_positions.clear()
    peak_millers.clear()
    peak_intensities.clear()

    # hit_tables is a numba.typed.List of 2-D arrays, one per reflection
    for tbl in max_positions_local:
        if tbl.shape[0] == 0:          # nothing recorded for this HKL
            continue
        # each row → (I  xpix  ypix  ϕ  H  K  L)
        for row in tbl:
            I, xpix, ypix, _phi, H, K, L = row
            if not (np.isfinite(xpix) and np.isfinite(ypix)):
                continue
            cx = int(round(xpix))
            cy = int(round(ypix))
            disp_cx, disp_cy = _native_sim_to_display_coords(cx, cy, updated_image.shape)
            peak_positions.append((disp_cx, disp_cy))      # display coords
            peak_intensities.append(I)
            hkl = tuple(int(np.rint(val)) for val in (H, K, L))
            peak_millers.append(hkl)

    normalization_scale = 1.0
    if current_background_image is not None and display_image is not None:
        normalization_scale = _suggest_scale_factor(
            display_image, current_background_image
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

    last_1d_integration_data["simulated_2d_image"] = unscaled_image_global

    if unscaled_image_global is not None:
        if scale_factor_user_override:
            _set_scale_factor_value(
                simulation_scale_factor_var.get(),
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
            "ai": pyFAI.AzimuthalIntegrator(
                dist=corto_det_up,
                poni1=center_x_up * pixel_size_m,
                poni2=center_y_up * pixel_size_m,
                rot1=np.deg2rad(Gamma_updated),
                rot2=np.deg2rad(gamma_updated),
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

        current_scale = simulation_scale_factor_var.get()
        scaled_caked_for_limits = caked_img * current_scale
        auto_vmin, auto_vmax = _auto_caked_limits(scaled_caked_for_limits)

        if not caked_limits_user_override:
            vmin_caked_var.set(auto_vmin)
            vmax_caked_var.set(auto_vmax)

        vmin_val = float(vmin_caked_var.get())
        vmax_val = float(vmax_caked_var.get())
        global_sim_max = float(simulation_max_var.get())

        if not math.isfinite(vmin_val):
            vmin_val = auto_vmin
        if not math.isfinite(vmax_val):
            vmax_val = auto_vmax
        if not math.isfinite(global_sim_max):
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
        if background_visible and current_background_image is not None:
            bg_res2 = caking(current_background_image, ai)
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
        if background_visible and current_background_image is not None:
            background_display.set_data(current_background_image)
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

        scale = simulation_scale_factor_var.get()
        line_1d_rad.set_data(rad_sim, i2t_sim * scale)
        line_1d_az.set_data(az_sim, i_phi_sim * scale)

        if background_visible and current_background_image is not None:
            bg_res2 = caking(current_background_image, ai)
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

    apply_scale_factor_to_existing_results(update_limits=True)

    update_integration_region_visuals(ai, sim_res2)

    # mark update completion so future updates can run
    update_running = False

# ── after you’ve updated background_visible in toggle_background() ──
def toggle_background():
    global background_visible
    background_visible = not background_visible
    # ↓ force opaque if the background is hidden, 0.5 otherwise
    image_display.set_alpha(0.5 if background_visible else 1.0)
    schedule_update()


def switch_background():
    global current_background_index, current_background_image
    current_background_index = (current_background_index + 1) % len(background_images)
    current_background_image = background_images[current_background_index]
    background_display.set_data(current_background_image)
    _update_background_slider_defaults(current_background_image, reset_override=True)
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
    center_x_var.set(defaults['center_x'])
    center_y_var.set(defaults['center_y'])
    tth_min_var.set(0.0)
    tth_max_var.set(80.0)
    phi_min_var.set(75.0)
    phi_max_var.set(105.0)
    show_1d_var.set(False)
    show_caked_2d_var.set(False)
    vmin_caked_var.set(0.0)
    vmax_caked_var.set(2000.0)
    caked_limits_user_override = False

    background_limits_user_override = False
    simulation_limits_user_override = False
    scale_factor_user_override = False

    _update_background_slider_defaults(current_background_image, reset_override=True)

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

azimuthal_button = ttk.Button(
    text="Azim vs Radial Plot Demo",
    command=lambda: view_azimuthal_radial(
        simulate_diffraction(
            theta_initial=theta_initial_var.get(),
            cor_angle=cor_angle_var.get(),
            gamma=gamma_var.get(),
            Gamma=Gamma_var.get(),
            chi=chi_var.get(),
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
            eta_var=eta_var
        ),
        [center_x_var.get(), center_y_var.get()],
        {
            'pixel_size': pixel_size_m,
            'poni1': (center_x_var.get()) * pixel_size_m,
            'poni2': (center_y_var.get()) * pixel_size_m,
            'dist': corto_detector_var.get(),
            'rot1': np.deg2rad(Gamma_var.get()),
            'rot2': np.deg2rad(gamma_var.get()),
            'rot3': 0.0,
            'wavelength': wave_m
        }
    )
)
azimuthal_button.pack(side=tk.TOP, padx=5, pady=2)

progress_label_positions = ttk.Label(root, text="", wraplength=300, justify=tk.LEFT)
progress_label_positions.pack(side=tk.BOTTOM, padx=5)

progress_label_geometry = ttk.Label(root, text="")
progress_label_geometry.pack(side=tk.BOTTOM, padx=5)

mosaic_progressbar = ttk.Progressbar(root, mode="indeterminate", length=240)
mosaic_progressbar.pack(side=tk.BOTTOM, padx=5, pady=(0, 2))

progress_label_mosaic = ttk.Label(root, text="", wraplength=300, justify=tk.LEFT)
progress_label_mosaic.pack(side=tk.BOTTOM, padx=5)

progress_label = ttk.Label(root, text="", font=("Helvetica", 8))
progress_label.pack(side=tk.BOTTOM, padx=5)

chi_square_label = ttk.Label(root, text="Chi-Squared: ", font=("Helvetica", 8))
chi_square_label.pack(side=tk.BOTTOM, padx=5)


def import_hbn_tilt_from_bundle():
    """Load an hBN ellipse bundle NPZ and apply its tilt hint to the GUI sliders."""

    bundle_path = filedialog.askopenfilename(
        title="Select hBN bundle (.npz)",
        filetypes=[("hBN bundle", "*.npz"), ("All files", "*.*")],
    )
    if not bundle_path:
        return

    mean_dist = None
    rot1_deg = None
    rot2_deg = None
    try:
        _, _, _, ellipses, distance_info, tilt_correction, tilt_hint, _, _ = load_bundle_npz(
            bundle_path
        )
    except Exception as exc:  # pragma: no cover - GUI interaction
        progress_label.config(text=f"Failed to load bundle: {exc}")
        return

    if tilt_correction:
        rot1_deg = float(tilt_correction.get("tilt_x_deg", 0.0))
        rot2_deg = float(tilt_correction.get("tilt_y_deg", 0.0))
    elif tilt_hint:
        rot1_deg = float(np.degrees(tilt_hint.get("rot1_rad", 0.0)))
        rot2_deg = float(np.degrees(tilt_hint.get("rot2_rad", 0.0)))
    else:
        tilt_hint_local = estimate_detector_tilt(ellipses)
        if tilt_hint_local:
            rot1_deg = float(np.degrees(tilt_hint_local["rot1_rad"]))
            rot2_deg = float(np.degrees(tilt_hint_local["rot2_rad"]))

    if rot1_deg is None or rot2_deg is None:
        progress_label.config(text="Bundle loaded, but no tilt information was found.")
        return
    Gamma_var.set(rot1_deg)
    gamma_var.set(rot2_deg)
    if distance_info and isinstance(distance_info, dict):
        mean_dist = distance_info.get("mean_m")
        if mean_dist is not None:
            try:
                corto_detector_var.set(float(mean_dist))
            except Exception:
                pass
    schedule_update()
    progress_label.config(
        text=(
            "Applied hBN bundle tilt hint "
            f"(Γ={rot1_deg:.3f}°, γ={rot2_deg:.3f}°)"
            + (
                f" and distance {mean_dist:.4f} m"  # type: ignore[arg-type]
                if distance_info and mean_dist is not None
                else ""
            )
            + f" from {bundle_path}"
        )
    )


save_button = ttk.Button(
    text="Save Params",
    command=lambda: save_all_parameters(
        get_path("parameters_file"),
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
    )
)
save_button.pack(side=tk.TOP, padx=5, pady=2)

load_button = ttk.Button(
    text="Load Params",
    command=lambda: (
        progress_label.config(
            text=load_parameters(
                get_path("parameters_file"),
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
            )
        ),
        ensure_valid_resolution_choice(),
        schedule_update()
    )
)
load_button.pack(side=tk.TOP, padx=5, pady=2)

import_hbn_button = ttk.Button(
    text="Import hBN Bundle Tilt",
    command=import_hbn_tilt_from_bundle,
)
import_hbn_button.pack(side=tk.TOP, padx=5, pady=2)

# Frame for selecting which geometry params to fit
fit_frame = ttk.LabelFrame(root, text="Fit geometry parameters")
fit_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

fit_zb_var    = tk.BooleanVar(value=True)
fit_zs_var    = tk.BooleanVar(value=True)
fit_theta_var = tk.BooleanVar(value=True)  # theta_initial
fit_chi_var   = tk.BooleanVar(value=True)
fit_cor_var   = tk.BooleanVar(value=True)

ttk.Checkbutton(fit_frame, text="zb",    variable=fit_zb_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="zs",    variable=fit_zs_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="theta", variable=fit_theta_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="chi",   variable=fit_chi_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="CoR",   variable=fit_cor_var).pack(side=tk.LEFT, padx=2)

def on_fit_geometry_click():
    _clear_geometry_pick_artists()

    # first, reconstruct the same mosaic_params dict you use in do_update()
    mosaic_params = build_mosaic_params()


    # assemble the params dict with exactly the keys the optimizer expects
    params = {
        'a':                  a_var.get(),
        'c':                  c_var.get(),
        'lambda':             lambda_,          # not 'lambda_'
        'psi':                psi,
        'zs':                 zs_var.get(),
        'zb':                 zb_var.get(),
        'chi':                chi_var.get(),
        'n2':                 n2,
        'mosaic_params':      mosaic_params,
        'debye_x':            debye_x_var.get(),
        'debye_y':            debye_y_var.get(),
        'center':             [center_x_var.get(), center_y_var.get()],
        'theta_initial':      theta_initial_var.get(),
        'uv1':                np.array([1.0,0.0,0.0]),
        'uv2':                np.array([0.0,1.0,0.0]),
        'corto_detector':     corto_detector_var.get(),
        'gamma':              gamma_var.get(),
        'Gamma':              Gamma_var.get(),
        'cor_angle':          cor_angle_var.get(),
    }

    # build the list of which of those to vary
    var_names = []
    if fit_zb_var.get():    var_names.append('zb')
    if fit_zs_var.get():    var_names.append('zs')
    if fit_theta_var.get(): var_names.append('theta_initial')
    if fit_chi_var.get():   var_names.append('chi')
    if fit_cor_var.get():   var_names.append('cor_angle')
    if not var_names:
        progress_label_geometry.config(text="No parameters selected!")
        return

    if not peak_positions:
        progress_label_geometry.config(text="Run a simulation first to pick peaks.")
        return

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
        r1 = min(current_background_image.shape[0], r + search_radius + 1)
        c0 = max(0, c - search_radius)
        c1 = min(current_background_image.shape[1], c + search_radius + 1)

        window = np.asarray(current_background_image[r0:r1, c0:c1], dtype=float)
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
    selection_state = {"expecting": "sim", "pending_hkl": None}
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

        if event.button == 3:
            _finish_pair_collection()
            if not picked_pairs:
                progress_label_geometry.config(text="No peak pairs selected; fit cancelled.")
                return
            measured_from_clicks = [
                {"label": f"{h},{k},{l}", "x": float(x), "y": float(y)}
                for (h, k, l), (x, y) in picked_pairs
            ]
            measured_for_fit = _unrotate_display_peaks(
                measured_from_clicks, current_background_image.shape
            )

            result = fit_geometry_parameters(
                miller,
                intensities,
                image_size,
                params,
                measured_for_fit,
                var_names,
                pixel_tol=float('inf'),
                experimental_image=current_background_image,
            )

            for name, val in zip(var_names, result.x):
                if name == 'zb':               zb_var.set(val)
                elif name == 'zs':             zs_var.set(val)
                elif name == 'theta_initial':  theta_initial_var.set(val)
                elif name == 'chi':            chi_var.set(val)
                elif name == 'cor_angle':      cor_angle_var.set(val)

            schedule_update()

            rms = np.sqrt(np.mean(result.fun**2)) if getattr(result, 'fun', None) is not None and result.fun.size else 0.0
            txt = "Fit complete:\n" + \
                  "\n".join(f"{n} = {v:.4f}" for n,v in zip(var_names, result.x)) + \
                  f"\nRMS residual = {rms:.2f} px"
            progress_label_geometry.config(text=txt)

            fitted_params = dict(params)
            fitted_params.update({
                'zb': zb_var.get(),
                'zs': zs_var.get(),
                'theta_initial': theta_initial_var.get(),
                'chi': chi_var.get(),
                'cor_angle': cor_angle_var.get(),
            })

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

            def _to_display_frame(col: float, row: float, *, k: int) -> tuple[float, float]:
                """Rotate native coordinates into the currently displayed frame."""

                return _rotate_point_for_display(
                    float(col), float(row), (image_size, image_size), k
                )

            pixel_offsets = []
            for hkl, (s_xy, m_xy) in zip(sim_millers, zip(sim_coords, meas_coords)):
                sx, sy = s_xy
                mx, my = m_xy
                dx = sx - mx
                dy = sy - my
                dist = math.hypot(dx, dy)
                pixel_offsets.append((hkl, dx, dy, dist))

                disp_sx, disp_sy = _to_display_frame(
                    sx, sy, k=SIM_DISPLAY_ROTATE_K
                )
                disp_mx, disp_my = _to_display_frame(
                    mx, my, k=DISPLAY_ROTATE_K
                )

                line, = ax.plot(
                    [disp_sx, disp_mx],
                    [disp_sy, disp_my],
                    color='#0984e3',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=5,
                )
                mid_x = 0.5 * (disp_sx + disp_mx)
                mid_y = 0.5 * (disp_sy + disp_my)
                label = ax.text(
                    mid_x,
                    mid_y,
                    f"|Δ|={dist:.1f}px",
                    color='#2d3436',
                    fontsize=8,
                    ha='center',
                    va='center',
                    zorder=6,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.0),
                )
                geometry_pick_artists.extend([line, label])

            canvas.draw_idle()

            export_recs = []
            for hkl, (x, y), (_, _, _, dist) in zip(sim_millers, sim_coords, pixel_offsets):
                export_recs.append({
                    'source': 'sim',
                    'hkl': tuple(int(v) for v in hkl),
                    'x': int(x),
                    'y': int(y),
                    'dist_px': float(dist),
                })
            for hkl, (x, y), (_, _, _, dist) in zip(meas_millers, meas_coords, pixel_offsets):
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

            if pixel_offsets:
                dist_lines = [
                    f"HKL={hkl}: |Δ|={dist:.2f}px (dx={dx:.2f}, dy={dy:.2f})"
                    for hkl, dx, dy, dist in pixel_offsets
                ]
                dist_report = "\n".join(dist_lines)
            else:
                dist_report = "No matched peaks to report distances."

            if DEBUG_ENABLED:
                best_orientation = _best_orientation_alignment(
                    sim_coords, meas_coords, (image_size, image_size)
                )
                if best_orientation is not None:
                    orientation_report = f"Best flip/rotation match: {best_orientation['label']}"
                else:
                    orientation_report = "Best flip/rotation match: unavailable"
                progress_label_geometry.config(text=orientation_report)
            else:
                progress_label_geometry.config(
                    text=(
                        progress_label_geometry.cget('text')
                        + f'\n\nSaved {len(export_recs)} peak records →\n{save_path}'
                        + f"\n\nPixel offsets:\n{dist_report}"
                    )
                )
            return

        col, row = float(event.xdata), float(event.ydata)

        if selection_state["expecting"] == "sim":
            idx, _ = _nearest_simulated_peak(col, row)
            if idx is None:
                progress_label_geometry.config(text="No simulated peaks available to pick.")
                return
            selection_state["pending_hkl"] = peak_millers[idx]
            sim_col, sim_row = peak_positions[idx]
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
            progress_label_geometry.config(text="Pick a simulated peak first.")
            return

        peak_col, peak_row = _peak_maximum_near(col, row, search_radius=6)
        _mark_pick(peak_col, peak_row, f"{pending} real", '#e17055', 'x')

        picked_pairs.append((pending, (peak_col, peak_row)))
        progress_label_geometry.config(
            text=(
                f"Recorded pair for HKL={pending} at real px=({peak_col:.1f},{peak_row:.1f}). "
                "Select another simulated peak or right click to fit."
            )
        )
        selection_state["expecting"] = "sim"

    click_cid = canvas.mpl_connect('button_press_event', _on_geometry_pick)
    return


def on_fit_mosaic_click():
    """Run the separable mosaic-width optimizer and apply the results."""

    global profile_cache, last_simulation_signature

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

    experimental_image = np.asarray(current_background_image, dtype=np.float64)
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


fit_button_geometry = ttk.Button(
    root,
    text="Fit Positions & Geometry",
    command=on_fit_geometry_click
)
fit_button_geometry.pack(side=tk.TOP, padx=5, pady=2)
fit_button_geometry.config(text="Fit Geometry (LSQ)", command=on_fit_geometry_click)

fit_button_mosaic = ttk.Button(
    root,
    text="Fit Mosaic Widths",
    command=on_fit_mosaic_click,
)
fit_button_mosaic.pack(side=tk.TOP, padx=5, pady=2)

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
    global caked_limits_user_override
    if not show_caked_2d_var.get():
        caked_limits_user_override = False
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
        save_flag=1
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

center_frame = CollapsibleFrame(right_col, text='Beam Center')
center_frame.pack(fill=tk.X, padx=5, pady=5)

def make_slider(label_str, min_val, max_val, init_val, step, parent, mosaic=False):
    var, scale = create_slider(
        label_str,
        min_val,
        max_val,
        init_val,
        step,
        parent,
        on_mosaic_slider_change if mosaic else schedule_update
    )
    return var, scale

theta_initial_var, theta_initial_scale = make_slider(
    'Theta Initial', 0.5, 30.0, defaults['theta_initial'], 0.01, geo_frame.frame
)
cor_angle_var, cor_angle_scale = make_slider(
    'CoR Axis Angle', -5.0, 5.0, defaults['cor_angle'], 0.01, geo_frame.frame
)
gamma_var, gamma_scale = make_slider(
    'Gamma', -4, 4, defaults['gamma'], 0.001, geo_frame.frame
)
Gamma_var, Gamma_scale = make_slider(
    'Detector Rotation Γ', -4, 4, defaults['Gamma'], 0.001, geo_frame.frame
)
chi_var, chi_scale = make_slider(
    'Chi', -1, 1, defaults['chi'], 0.001, geo_frame.frame
)
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
    'a (Å)', 3.5, 8.0, defaults['a'], 0.01, lattice_frame.frame
)
c_var, c_scale = make_slider(
    'c (Å)', 20.0, 40.0, defaults['c'], 0.01, lattice_frame.frame
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
center_x_var, center_x_scale = make_slider(
    'Beam Center Row',
    center_default[0]-100.0,
    center_default[0]+100.0,
    defaults['center_x'],
    1.0,
    center_frame.frame
)
center_y_var, center_y_scale = make_slider(
    'Beam Center Col',
    center_default[1]-100.0,
    center_default[1]+100.0,
    defaults['center_y'],
    1.0,
    center_frame.frame
)

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

    combined_qr_local = combine_qr_dicts(caches, weights)
    arrays_local = qr_dict_to_arrays(combined_qr_local)
    ht_curves_cache = {
        "curves": combined_qr_local,
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


def update_occupancies(*args):
    """Recompute Hendricks–Teller curves when occupancies or p-values change."""

    new_occ = [occ_var1.get(), occ_var2.get(), occ_var3.get()]
    p_vals = [p0_var.get(), p1_var.get(), p2_var.get()]
    w_raw = [w0_var.get(), w1_var.get(), w2_var.get()]
    w_sum = sum(w_raw) or 1.0
    weights = [w / w_sum for w in w_raw]

    _rebuild_diffraction_inputs(new_occ, p_vals, weights, c_var.get())


def _sync_finite_controls():
    global _layers_scale_widget
    if _layers_scale_widget is None:
        return
    state = tk.NORMAL if finite_stack_var.get() else tk.DISABLED
    _layers_scale_widget.configure(state=state)


def _on_finite_toggle():
    _sync_finite_controls()
    update_occupancies()


def _on_layer_slider(val):
    try:
        value = int(round(float(val)))
    except (TypeError, ValueError):
        value = stack_layers_var.get()
    if stack_layers_var.get() != value:
        stack_layers_var.set(value)
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
ttk.Label(layers_row, textvariable=stack_layers_var, width=6).grid(
    row=0,
    column=2,
    sticky="e",
    padx=(5, 0),
)

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
