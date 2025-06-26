#!/usr/bin/env python3

"""Main application entry point for running the Tk based GUI."""

import math
import os

# Enable debug messages automatically. Set RA_SIM_DEBUG=0 to disable.
os.environ.setdefault("RA_SIM_DEBUG", "0")
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
from matplotlib import cm
from matplotlib.colors import ListedColormap
import pyFAI
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from scipy.optimize import differential_evolution, least_squares
from skimage.metrics import mean_squared_error
import spglib
import OSC_Reader
from OSC_Reader import read_osc
import numba
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
)
from ra_sim.io.data_loading import (
    load_and_format_reference_profiles,
    save_all_parameters,
    load_parameters,
)
from ra_sim.fitting.optimization import (
    simulate_and_compare_hkl,
    fit_geometry_parameters,
)
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import (
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


turbo = cm.get_cmap('turbo', 256)          # 256-step version of ‘turbo’
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
from ra_sim.path_config import get_path, get_dir

file_path = get_path("dark_image")
BI = read_osc(file_path)  # Dark (background) image

osc_files = get_path("osc_files")
file1 = read_osc(osc_files[0])
#file1 = read_osc(r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Varying\Images\Bi2Se3_5m_5d.osc")
file2 = read_osc(osc_files[1])
file3 = read_osc(osc_files[2])
file4 = read_osc(osc_files[3])

#bg1 = np.load(r"C:\Users\Kenpo\Downloads\background_6d.npy")
#bg2 = np.load(r"C:\Users\Kenpo\Downloads\background_10d.npy")
#bg3 = np.load(r"C:\Users\Kenpo\Downloads\background_15d.npy")
#bg4 = np.load(r"C:\Users\Kenpo\Downloads\background_30d.npy")

# Rotate and subtract dark image
files = [file1, file2, file3, file4]
#bg_data_list = [bg1, bg2, bg3, bg4]
#for i in range(len(files)):
#    files[i] = np.rot90(files[i], k=3) - BI

background_images = files

# Parse geometry
poni_file_path = get_path("geometry_poni")
parameters = parse_poni_file(poni_file_path)

Distance_CoR_to_Detector = parameters.get("Dist", 0.075)
Gamma_initial = parameters.get("Rot1", 0.0)
gamma_initial = parameters.get("Rot2", 0.0)
poni1 = parameters.get("Poni1", 0.0)
poni2 = parameters.get("Poni2", 0.0)
wave_m = parameters.get("Wavelength", 1e-10)
lambda_ = wave_m * 1e10  # Convert m -> Å

image_size = 3000
num_samples = 1000

# Approximate beam center
center_default = [
    (poni2 / (100e-6)),
    3000 - (poni1 / (100e-6))
]

mx = 19

fwhm2sigma = 1 / (2 * math.sqrt(2 * math.log(2)))
divergence_sigma = math.radians(0.05 * fwhm2sigma)

sigma_mosaic = math.radians(0.8 * fwhm2sigma)
gamma_mosaic = math.radians(0.7 * fwhm2sigma)
eta = 0.0

theta_initial = 6.0
chi = 0.0
psi = 0.0
zb = 0.0
bw_sigma = 0.05e-3 * fwhm2sigma
zs = 0.0
debye_x = 0.0
debye_y = 0.0
n2 = IndexofRefraction()

bandwidth = 0.7 / 100  # 0.7%

# NOTE: We define the default occupancy for each site:
occ = [1.0, 1.0, 1.0]

# When enabled, additional fractional reflections ("rods")
# are injected between integer L values.
include_rods_flag = False

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
cv = parse_cif_num(c_text)

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
    
lambda_ = 1.54   # X-ray wavelength in Å (e.g., Cu Kα)
energy = 6.62607e-34 * 2.99792458e8 / (lambda_*1e-10) / (1.602176634e-19)    # keV
intensity_threshold = 1.0
two_theta_range = (0, 70)

# ---------------------------------------------------------------------------
# Default GUI/fit parameter values. These must be defined before any calls
# that reference them (e.g. ``ht_Iinf_dict`` below).
# ---------------------------------------------------------------------------
defaults = {
    'theta_initial': 5.0,
    'gamma': Gamma_initial,
    'Gamma': gamma_initial,
    'chi': 0.0,
    'zs': 0.0,
    'zb': 0.0,
    'debye_x': 0.0,
    'debye_y': 0.0,
    'corto_detector': Distance_CoR_to_Detector,
    'sigma_mosaic_deg': np.degrees(sigma_mosaic),
    'gamma_mosaic_deg': np.degrees(gamma_mosaic),
    'eta': 0.0,
    'a': av,
    'c': cv,
    'vmax': 1000,
    'p': 0.5,
    'center_x': center_default[0],
    'center_y': center_default[1],
}

# ---------------------------------------------------------------------------
# Replace the old miller_generator call with the new Hendricks–Teller helper.
# ---------------------------------------------------------------------------
ht_curves = ht_Iinf_dict(                 # ← new core
    cif_path=cif_file,
    mx=mx,                                # generates all (h,k) for |h|,|k|<mx
    occ=occ,                              # same occupancy-scaling rules
    p=defaults['p'],                      # disorder probability
    L_step= 0.01,
    two_theta_max=two_theta_range[1],
    lambda_=lambda_,
)

# Convert to Qr rods by summing curves with identical radial index.
qr_curves = ht_dict_to_qr_dict(ht_curves)

# Cache the initial Qr curves along with the occupancy and p values so that
# subsequent updates can reuse them unless these parameters change.
miller1, intens1, degeneracy1, details1 = qr_dict_to_arrays(qr_curves)
ht_curves_cache = {
    "curves": qr_curves,
    "arrays": (miller1, intens1, degeneracy1, details1),
}
_last_occ_for_ht = list(occ)
_last_p_for_ht = float(defaults['p'])
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

if write_excel:
    # Save the initial intensities to Excel in the configured downloads
    # directory.
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

# Zero out beamstop region near center
row_center = int(center_default[0])
col_center = int(center_default[1])
half_size = 40

for bg in background_images:
    rmin = max(0, row_center - half_size)
    rmax = min(bg.shape[0], row_center + half_size)
    cmin = max(0, col_center - half_size)
    cmax = min(bg.shape[1], col_center + half_size)
    bg[rmin:rmax, cmin:cmax] = 0.0

current_background_image = background_images[0]
current_background_index = 0
background_visible = True

measured_peaks = np.load(get_path("measured_peaks"), allow_pickle=True)

###############################################################################
#                                  TK SETUP
###############################################################################
root = tk.Tk()
root.title("Controls and Sliders")

fig_window = tk.Toplevel(root)
fig_window.title("Main Figure")
fig_frame = ttk.Frame(fig_window)
fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas_frame = ttk.Frame(fig_frame)
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

vmax_frame = ttk.Frame(fig_frame)
vmax_frame.pack(side=tk.BOTTOM, fill=tk.X)

fig, ax = plt.subplots(figsize=(8, 8))
canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

global_image_buffer = np.zeros((image_size, image_size), dtype=np.float64)
unscaled_image_global = None

# ── replace the original imshow call ────────────────────────────
image_display = ax.imshow(
    global_image_buffer,
    cmap=turbo_white0,     # ← use the custom map
    vmin=0,
    vmax=1000,
    alpha=0.5,
    zorder=1,
    origin='upper'
)


background_display = ax.imshow(
    current_background_image,
    cmap='turbo',
    vmin=0,
    vmax=1e3,
    zorder=0,
    origin='upper'
)
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
# Separate "vmax" for normal 2D vs. caked 2D images
# ---------------------------------------------------------------------------
vmax_label = ttk.Label(vmax_frame, text="Max Value (Normal)")
vmax_label.pack(side=tk.LEFT, padx=5)

vmax_var = tk.DoubleVar(value=defaults['vmax'])

def vmax_slider_command(val):
    v = float(val)
    vmax_var.set(v)
    schedule_update()

vmax_slider = ttk.Scale(
    vmax_frame,
    from_=0,
    to=3000,
    orient=tk.HORIZONTAL,
    variable=vmax_var,
    command=vmax_slider_command
)
vmax_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

# Frame for caked vrange
caked_vrange_frame = ttk.Frame(fig_frame)
caked_vrange_frame.pack(side=tk.BOTTOM, fill=tk.X)

vmin_caked_label = ttk.Label(caked_vrange_frame, text="vmin (Caked)")
vmin_caked_label.pack(side=tk.LEFT, padx=5)

vmin_caked_var = tk.DoubleVar(value=0.0)
def vmin_caked_slider_command(val):
    v = float(val)
    vmin_caked_var.set(v)
    schedule_update()

vmin_caked_slider = ttk.Scale(
    caked_vrange_frame,
    from_=0,
    to=1000,
    orient=tk.HORIZONTAL,
    variable=vmin_caked_var,
    command=vmin_caked_slider_command
)
vmin_caked_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

vmax_caked_label = ttk.Label(caked_vrange_frame, text="vmax (Caked)")
vmax_caked_label.pack(side=tk.LEFT, padx=5)

vmax_caked_var = tk.DoubleVar(value=2000.0)
def vmax_caked_slider_command(val):
    v = float(val)
    vmax_caked_var.set(v)
    schedule_update()

vmax_caked_slider = ttk.Scale(
    caked_vrange_frame,
    from_=0,
    to=5000,
    orient=tk.HORIZONTAL,
    variable=vmax_caked_var,
    command=vmax_caked_slider_command
)
vmax_caked_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

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

range_frame = ttk.LabelFrame(plot_frame_1d, text="Integration Ranges")
range_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

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

def caking(data, ai):
    return ai.integrate2d(
        data,
        npt_rad=1000,
        npt_azim=720,
        correctSolidAngle=True,
        method="lut",
        unit="2th_deg"
    )

def caked_up(res2, tth_min, tth_max, phi_min, phi_max):
    intensity = res2.intensity
    radial_2theta = res2.radial
    azimuth_vals = res2.azimuthal

    mask_rad = (radial_2theta >= tth_min) & (radial_2theta <= tth_max)
    radial_filtered = radial_2theta[mask_rad]

    mask_az = (azimuth_vals >= phi_min) & (azimuth_vals <= phi_max)
    azimuth_sub = azimuth_vals[mask_az]

    intensity_sub = intensity[np.ix_(mask_az, mask_rad)]
    intensity_vs_2theta = np.sum(intensity_sub, axis=0)
    intensity_vs_phi = np.sum(intensity_sub, axis=1)

    return intensity_vs_2theta, intensity_vs_phi, azimuth_sub, radial_filtered

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

last_res2_background = None
last_res2_sim = None

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
def schedule_update():
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
    global update_pending, last_simulation_signature
    global unscaled_image_global, background_visible
    global stored_max_positions_local, stored_sim_image
    global SIM_MILLER1, SIM_INTENS1, SIM_MILLER2, SIM_INTENS2
    global av2, cv2

    update_pending = None

    gamma_updated      = float(gamma_var.get())
    Gamma_updated      = float(Gamma_var.get())
    chi_updated        = float(chi_var.get())
    zs_updated         = float(zs_var.get())
    zb_updated         = float(zb_var.get())
    a_updated          = float(a_var.get())
    c_updated          = float(c_var.get())
    theta_init_up      = float(theta_initial_var.get())
    debye_x_updated    = float(debye_x_var.get())
    debye_y_updated    = float(debye_y_var.get())
    corto_det_up       = float(corto_detector_var.get())
    center_x_up        = float(center_x_var.get())
    center_y_up        = float(center_y_var.get())

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
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    save_flag=0,
                )

        img1, maxpos1, _, _, _, _ = run_one(ht_curves_cache["curves"], None, a_updated, c_updated)
        if SIM_MILLER2.size > 0:
            img2, maxpos2, _, _, _, _ = run_one(SIM_MILLER2, SIM_INTENS2, av2, cv2)
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
            return do_update()          # re-enter with computation path
        max_positions_local = stored_max_positions_local
        updated_image       = stored_sim_image
    
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
            peak_positions.append((cx, cy))      # (column,x) == (MPL x)
            peak_intensities.append(I)
            peak_millers.append((int(H), int(K), int(L)))

        # Store the unscaled image globally
        unscaled_image_global = updated_image

    def scale_image_for_display(unscaled_img):
        if unscaled_img is None:
            return np.zeros((image_size, image_size), dtype=np.float64)
        disp_img = unscaled_img.copy()
        if current_background_image is not None:
            max_bg = np.max(current_background_image)
            max_sim = np.max(disp_img)
            if (max_bg > 0) and (max_sim > 0):
                disp_img *= (max_bg / max_sim)
        return disp_img

    disp_image = scale_image_for_display(unscaled_image_global)
    global_image_buffer[:] = disp_image

    # Check if we show caked 2D
    if show_caked_2d_var.get() and unscaled_image_global is not None:
        image_display.set_clim(vmin_caked_var.get(), vmax_caked_var.get())
    else:
        image_display.set_clim(0, vmax_var.get())

    image_display.set_data(global_image_buffer)

    background_display.set_visible(background_visible)
    if background_visible:
        background_display.set_data(current_background_image)
    else:
        background_display.set_data(np.zeros_like(current_background_image))

    try:
        norm_sim = (global_image_buffer / np.max(global_image_buffer)
                    if np.max(global_image_buffer) > 0 else global_image_buffer)
        norm_bg = (current_background_image / np.max(current_background_image)
                   if (current_background_image is not None and np.max(current_background_image) > 0)
                   else current_background_image)
        if norm_bg is not None and norm_bg.shape == norm_sim.shape:
            chi_sq_val = mean_squared_error(norm_bg, norm_sim) * norm_sim.size
            chi_square_label.config(text=f"Chi-Squared: {chi_sq_val:.2e}")
        else:
            chi_square_label.config(text="Chi-Squared: N/A")
    except Exception as e:
        chi_square_label.config(text=f"Chi-Squared: Error - {e}")

    last_1d_integration_data["simulated_2d_image"] = unscaled_image_global

    ai = pyFAI.AzimuthalIntegrator(
        dist=corto_det_up,
        poni1=center_x_up * 100e-6,
        poni2=center_y_up * 100e-6,
        rot1=np.deg2rad(Gamma_updated),
        rot2=np.deg2rad(gamma_updated),
        rot3=0.0,
        wavelength=wave_m,
        pixel1=100e-6,
        pixel2=100e-6
    )

    # Caked 2D or normal 2D?
    if show_caked_2d_var.get() and unscaled_image_global is not None:
        sim_res2 = caking(unscaled_image_global, ai)
        caked_img = sim_res2.intensity
        image_display.set_data(caked_img)
        image_display.set_clim(vmin_caked_var.get(), vmax_caked_var.get())
        ax.set_title('2D Caked Integration')
        background_display.set_visible(False)
    else:
        if unscaled_image_global is not None:
            disp_image = scale_image_for_display(unscaled_image_global)
            image_display.set_data(disp_image)
        else:
            image_display.set_data(np.zeros((image_size, image_size)))
        image_display.set_clim(0, vmax_var.get())
        ax.set_title('Simulated Diffraction Pattern')
        background_display.set_visible(background_visible)
        
    # 1D integration
    if show_1d_var.get() and unscaled_image_global is not None:
        sim_res2 = caking(unscaled_image_global, ai)
        i2t_sim, i_phi_sim, az_sim, rad_sim = caked_up(
            sim_res2,
            tth_min_var.get(),
            tth_max_var.get(),
            phi_min_var.get(),
            phi_max_var.get()
        )
        line_1d_rad.set_data(rad_sim, i2t_sim)
        line_1d_az.set_data(az_sim, i_phi_sim)

        if background_visible and current_background_image is not None:
            bg_res2 = caking(current_background_image, ai)
            i2t_bg, i_phi_bg, az_bg, rad_bg = caked_up(
                bg_res2,
                tth_min_var.get(),
                tth_max_var.get(),
                phi_min_var.get(),
                phi_max_var.get()
            )
            line_1d_rad_bg.set_data(rad_bg, i2t_bg)
            line_1d_az_bg.set_data(az_bg, i_phi_bg)
        else:
            line_1d_rad_bg.set_data([], [])
            line_1d_az_bg.set_data([], [])

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

    canvas.draw_idle()

    try:
        if background_visible and current_background_image is not None:
            norm_sim = (unscaled_image_global / np.max(unscaled_image_global)
                        if np.max(unscaled_image_global) > 0 else unscaled_image_global)
            norm_bg = (current_background_image / np.max(current_background_image)
                       if np.max(current_background_image) > 0 else current_background_image)
            if norm_bg is not None and norm_bg.shape == norm_sim.shape:
                chi_sq_val = mean_squared_error(norm_bg, norm_sim) * norm_sim.size
                chi_square_label.config(text=f"Chi-Squared: {chi_sq_val:.2e}")
            else:
                chi_square_label.config(text="Chi-Squared: N/A")
        else:
            chi_square_label.config(text="Chi-Squared: N/A")
    except Exception as e:
        chi_square_label.config(text=f"Chi-Squared Error: {e}")

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
    schedule_update()

def reset_to_defaults():
    theta_initial_var.set(defaults['theta_initial'])
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
    vmax_var.set(defaults['vmax'])
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

    # ALSO reset occupancies to default
    occ_var1.set(1.0)
    occ_var2.set(0.5)
    occ_var3.set(0.5)

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
            'pixel_size': 100e-6,
            'poni1': (center_x_var.get()) * 100e-6,
            'poni2': (center_y_var.get()) * 100e-6,
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

progress_label = ttk.Label(root, text="", font=("Helvetica", 8))
progress_label.pack(side=tk.BOTTOM, padx=5)

chi_square_label = ttk.Label(root, text="Chi-Squared: ", font=("Helvetica", 8))
chi_square_label.pack(side=tk.BOTTOM, padx=5)

save_button = ttk.Button(
    text="Save Params",
    command=lambda: save_all_parameters(
        get_path("parameters_file"),
        theta_initial_var,
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
        center_y_var
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
                center_y_var
            )
        ),
        schedule_update()
    )
)
load_button.pack(side=tk.TOP, padx=5, pady=2)

# Frame for selecting which geometry params to fit
fit_frame = ttk.LabelFrame(root, text="Fit geometry parameters")
fit_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

fit_zb_var    = tk.BooleanVar(value=True)
fit_zs_var    = tk.BooleanVar(value=True)
fit_theta_var = tk.BooleanVar(value=True)  # theta_initial
fit_chi_var   = tk.BooleanVar(value=True)

ttk.Checkbutton(fit_frame, text="zb",    variable=fit_zb_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="zs",    variable=fit_zs_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="theta", variable=fit_theta_var).pack(side=tk.LEFT, padx=2)
ttk.Checkbutton(fit_frame, text="chi",   variable=fit_chi_var).pack(side=tk.LEFT, padx=2)


def on_fit_geometry_click():
    # Assemble params dict (must include all keys used by simulate_and_compare_hkl)
    params = {
        'av': a_var.get(),
        'cv': c_var.get(),
        'lambda_': lambda_,
        'psi': psi,
        'zs': zs_var.get(),
        'zb': zb_var.get(),
        'chi': chi_var.get(),
        'n2': n2,
        'beam_x_array': profile_cache['beam_x_array'],
        'beam_y_array': profile_cache['beam_y_array'],
        'theta_array': profile_cache['theta_array'],
        'phi_array': profile_cache['phi_array'],
        'sigma_mosaic_deg': sigma_mosaic_var.get(),
        'gamma_mosaic_deg': gamma_mosaic_var.get(),
        'eta': eta_var.get(),
        'wavelength_array': profile_cache['wavelength_array'],
        'debye_x': debye_x_var.get(),
        'debye_y': debye_y_var.get(),
        'center': [center_x_var.get(), center_y_var.get()],
        'theta_initial': theta_initial_var.get(),
        'uv1': np.array([1.0,0.0,0.0]),
        'uv2': np.array([0.0,1.0,0.0]),
        'corto_detector': corto_detector_var.get(),
        'gamma': gamma_var.get(),
        'Gamma': Gamma_var.get(),
    }

    # Build list of parameters to fit
    var_names = []
    if fit_zb_var.get():    var_names.append('zb')
    if fit_zs_var.get():    var_names.append('zs')
    if fit_theta_var.get(): var_names.append('theta_initial')
    if fit_chi_var.get():   var_names.append('chi')

    if not var_names:
        progress_label_geometry.config(text="No parameters selected!")
        return

    # Run least-squares fit (infinite tol → pure HKL matching)
    result = fit_geometry_parameters(
        miller, intensities, image_size,
        params, measured_peaks,
        var_names,
        pixel_tol=float('inf')
    )

    # Update sliders with fitted values
    for name, val in zip(var_names, result.x):
        if name == 'zb':            zb_var.set(val)
        elif name == 'zs':          zs_var.set(val)
        elif name == 'theta_initial': theta_initial_var.set(val)
        elif name == 'chi':         chi_var.set(val)

    # Redraw the figure with new geometry
    schedule_update()

    # Show summary
    rms = np.sqrt(np.mean(result.fun**2)) if result.fun.size else 0.0
    txt = "Fit complete:\n"
    txt += "\n".join(f"{n} = {v:.4f}" for n, v in zip(var_names, result.x))
    txt += f"\nRMS residual = {rms:.2f} px"
    progress_label_geometry.config(text=txt)
    
    # ─────────────────────────────────────────────────────────────────────
    # χ² minimisation (unchanged)
    result = fit_geometry_parameters(
        miller, intensities, image_size,
        params, measured_peaks,
        var_names,
        pixel_tol=float('inf')
    )
    # ─────────────────────────────────────────────────────────────────────
    # write the fitted values back into the sliders (unchanged)
    for name, val in zip(var_names, result.x):
        if   name == 'zb':            zb_var.set(val)
        elif name == 'zs':            zs_var.set(val)
        elif name == 'theta_initial': theta_initial_var.set(val)
        elif name == 'chi':           chi_var.set(val)

    schedule_update()         # causes a new simulation & redraw
    # ─────────────────────────────────────────────────────────────────────
    # ❶  RE-RUN THE COMPARISON WITH THE *FITTED* PARAMETERS
    #     (this picks up the brand-new slider values)
    fitted_params = dict(params)       # shallow copy
    fitted_params.update({
        'zb'            : zb_var.get(),
        'zs'            : zs_var.get(),
        'theta_initial' : theta_initial_var.get(),
        'chi'           : chi_var.get(),
    })

    (D, label_match, pixel_match, match_matrix,
     sim_coords, sim_millers,
     meas_coords, meas_millers) = simulate_and_compare_hkl(
        miller, intensities,            # same reflections
        image_size,
        fitted_params,                  #   ↖ fitted geometry!
        measured_peaks,
        pixel_tol=float('inf')          # keep *all* simulated peaks
    )

    # ─────────────────────────────────────────────────────────────────────
    # ❷  BUILD A UNIFIED LIST OF RECORDS
    export_recs = []

    #   ▸ simulated peaks
    for hkl, (x, y) in zip(sim_millers, sim_coords):
        export_recs.append({
            'source' : 'sim',
            'hkl'    : tuple(int(v) for v in hkl),
            'x'      : int(x),
            'y'      : int(y),
        })

    #   ▸ measured peaks  (stored exactly as in blobs.npy)
    for hkl, (x, y) in zip(meas_millers, meas_coords):
        export_recs.append({
            'source' : 'meas',
            'hkl'    : tuple(int(v) for v in hkl),
            'x'      : int(x),
            'y'      : int(y),
        })

    # ─────────────────────────────────────────────────────────────────────
    # ❸  SAVE AUTOMATICALLY INTO configured downloads directory

    download_dir = get_dir("downloads")

    stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path  = download_dir / f"matched_peaks_{stamp}.npy"

    np.save(save_path, np.array(export_recs, dtype=object), allow_pickle=True)

    progress_label_geometry.config(
        text=(progress_label_geometry.cget('text')
            + f'\n\nSaved {len(export_recs)} peak records →\n{save_path}')
    )
    # ─────────────────────────────────────────────────────────────────────

def on_fit_geometry_click():
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
    }

    # build the list of which of those to vary
    var_names = []
    if fit_zb_var.get():    var_names.append('zb')
    if fit_zs_var.get():    var_names.append('zs')
    if fit_theta_var.get(): var_names.append('theta_initial')
    if fit_chi_var.get():   var_names.append('chi')
    if not var_names:
        progress_label_geometry.config(text="No parameters selected!")
        return

    # now call the fitter
    result = fit_geometry_parameters(
        miller, intensities, image_size,
        params,
        measured_peaks,
        var_names,
        pixel_tol=float('inf')
    )

    # unpack fitted values back onto the sliders
    for name, val in zip(var_names, result.x):
        if name == 'zb':            zb_var.set(val)
        elif name == 'zs':          zs_var.set(val)
        elif name == 'theta_initial': theta_initial_var.set(val)
        elif name == 'chi':         chi_var.set(val)

    # redraw with the new geometry
    schedule_update()

    rms = np.sqrt(np.mean(result.fun**2)) if result.fun.size else 0.0
    txt = "Fit complete:\n" + \
          "\n".join(f"{n} = {v:.4f}" for n,v in zip(var_names, result.x)) + \
          f"\nRMS residual = {rms:.2f} px"
    progress_label_geometry.config(text=txt)

fit_button_geometry = ttk.Button(
    root,
    text="Fit Positions & Geometry",
    command=on_fit_geometry_click
)
fit_button_geometry.pack(side=tk.TOP, padx=5, pady=2)
fit_button_geometry.config(text="Fit Geometry (LSQ)", command=on_fit_geometry_click)

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

# Option to add fractional rods between integer L values
include_rods_var = tk.BooleanVar(value=include_rods_flag)
def toggle_rods():
    global include_rods_flag
    include_rods_flag = include_rods_var.get()
    update_occupancies()

check_rods = ttk.Checkbutton(
    text="Include Rods",
    variable=include_rods_var,
    command=toggle_rods
)
check_rods.pack(side=tk.TOP, padx=5, pady=2)

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
    
    sim_img = last_1d_integration_data.get("simulated_2d_image")
    if sim_img is None:
        progress_label.config(text="No simulated image available to save!")
        return

    sim_img = np.asarray(sim_img, dtype=np.float64)
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

    image_result, max_positions_local, q_data, q_count, _, _ = process_peaks_parallel(
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
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=1
    )

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
    'Theta Initial', 0.5, 30.0, defaults['theta_initial'], 0.01, left_col
)
gamma_var, gamma_scale = make_slider(
    'Gamma', -4, 4, defaults['gamma'], 0.001, left_col
)
Gamma_var, Gamma_scale = make_slider(
    'Detector Rotation Γ', -4, 4, defaults['Gamma'], 0.001, left_col
)
chi_var, chi_scale = make_slider(
    'Chi', -1, 1, defaults['chi'], 0.001, left_col
)
zs_var, zs_scale = make_slider(
    'Zs', -2.0e-3, 2e-3, defaults['zs'], 0.0001, left_col
)
zb_var, zb_scale = make_slider(
    'Zb', -2.0e-3, 2e-3, defaults['zb'], 0.0001, left_col
)
debye_x_var, debye_x_scale = make_slider(
    'Debye Qz', 0.0, 1.0, defaults['debye_x'], 0.001, left_col
)
debye_y_var, debye_y_scale = make_slider(
    'Debye Qr', 0.0, 1.0, defaults['debye_y'], 0.001, left_col
)
corto_detector_var, corto_detector_scale = make_slider(
    'CortoDetector', 0.0, 100e-3, defaults['corto_detector'], 0.1e-3, right_col
)
a_var, a_scale = make_slider(
    'a (Å)', 3.5, 8.0, defaults['a'], 0.01, right_col
)
c_var, c_scale = make_slider(
    'c (Å)', 20.0, 40.0, defaults['c'], 0.01, right_col
)
sigma_mosaic_var, sigma_mosaic_scale = make_slider(
    'σ Mosaic (deg)', 0.0, 5.0, defaults['sigma_mosaic_deg'], 0.01, right_col, mosaic=True
)
gamma_mosaic_var, gamma_mosaic_scale = make_slider(
    'γ Mosaic (deg)', 0.0, 5.0, defaults['gamma_mosaic_deg'], 0.01, right_col, mosaic=True
)
eta_var, eta_scale = make_slider(
    'η (fraction)', 0.0, 1.0, defaults['eta'], 0.001, right_col, mosaic=True
)
center_x_var, center_x_scale = make_slider(
    'Beam Center Row',
    center_default[0]-100.0,
    center_default[0]+100.0,
    defaults['center_x'],
    1.0,
    right_col
)
center_y_var, center_y_scale = make_slider(
    'Beam Center Col',
    center_default[1]-100.0,
    center_default[1]+100.0,
    defaults['center_y'],
    1.0,
    right_col
)

# Slider controlling contribution of the first CIF file, only if a second CIF
# was provided.
if has_second_cif:
    weight1_var, _ = make_slider(
        'CIF1 Weight', 0.0, 1.0, weight1, 0.01, right_col
    )
    weight2_var, _ = make_slider(
        'CIF2 Weight', 0.0, 1.0, weight2, 0.01, right_col
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

vmax_slider.config(command=vmax_slider_command)
# ---------------------------------------------------------------------------
#  OCCUPANCY SLIDERS: Sliders for occ[0], occ[1], occ[2]
# ---------------------------------------------------------------------------
occ_var1 = tk.DoubleVar(value=1.0)
occ_var2 = tk.DoubleVar(value=1.0)
occ_var3 = tk.DoubleVar(value=1.0)

def update_occupancies(*args):
    """
    Re-run miller_generator with updated occupancies,
    then force the simulation to recalc.
    """
    global miller, intensities, degeneracy, details
    global df_summary, df_details
    global last_simulation_signature
    global SIM_MILLER1, SIM_INTENS1, SIM_MILLER2, SIM_INTENS2
    global ht_curves_cache, _last_occ_for_ht, _last_p_for_ht

    # Grab new occupancy values from the variables (they may have been updated via the slider or Entry)
    new_occ = [occ_var1.get(), occ_var2.get(), occ_var3.get()]

    global intensities_cif1, intensities_cif2
    new_p = p_var.get()

    # Reuse the cached ht_curves unless occupancy or p has changed
    if (
        ht_curves_cache is None
        or list(new_occ) != list(_last_occ_for_ht)
        or not math.isclose(new_p, _last_p_for_ht, rel_tol=0.0, abs_tol=1e-12)
    ):
        ht_curves_local = ht_Iinf_dict(
            cif_path=cif_file,
            mx=mx,
            occ=new_occ,
            p=new_p,
            L_step=0.02,
            two_theta_max=two_theta_range[1],
            lambda_=lambda_,
        )
        qr_local = ht_dict_to_qr_dict(ht_curves_local)
        arrays_local = qr_dict_to_arrays(qr_local)
        ht_curves_cache = {"curves": qr_local, "arrays": arrays_local}
        _last_occ_for_ht = list(new_occ)
        _last_p_for_ht = float(new_p)
    else:
        ht_curves_local = ht_curves_cache["curves"]
        arrays_local = ht_curves_cache["arrays"]

    m1, i1, d1, det1 = arrays_local

    # Convert arrays → dictionaries for quick lookup
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
        if include_rods_var.get():
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
        SIM_MILLER2 = np.empty((0,3), dtype=np.int32)
        SIM_INTENS2 = np.empty((0,), dtype=np.float64)

    # (Re-)build the summary and details DataFrames.
    df_summary, df_details = build_intensity_dataframes(
        miller, intensities, degeneracy, details
    )

    # Reset the simulation signature so the next update is forced.
    last_simulation_signature = None
    schedule_update()

# Slider for stacking disorder probability p
p_var, _ = create_slider(
    'Disorder p', 0.0, 1.0, defaults['p'], 0.01, right_col, update_occupancies
)

# Existing occupancy slider for site 1.
ttk.Label(right_col, text="Occupancy Site 1").pack(padx=5, pady=2)
occ_scale1 = ttk.Scale(
    right_col,
    from_=0.0,
    to=1.0,
    orient=tk.HORIZONTAL,
    variable=occ_var1,
    command=update_occupancies
)
occ_scale1.pack(fill=tk.X, padx=5, pady=2)

# Existing occupancy slider for site 2.
ttk.Label(right_col, text="Occupancy Site 2").pack(padx=5, pady=2)
occ_scale2 = ttk.Scale(
    right_col,
    from_=0.0,
    to=1.0,
    orient=tk.HORIZONTAL,
    variable=occ_var2,
    command=update_occupancies
)
occ_scale2.pack(fill=tk.X, padx=5, pady=2)

# Existing occupancy slider for site 3.
ttk.Label(right_col, text="Occupancy Site 3").pack(padx=5, pady=2)
occ_scale3 = ttk.Scale(
    right_col,
    from_=0.0,
    to=1.0,
    orient=tk.HORIZONTAL,
    variable=occ_var3,
    command=update_occupancies
)
occ_scale3.pack(fill=tk.X, padx=5, pady=2)

# --- Add numeric input fields and a Force Update button ---
occ_entry_frame = ttk.Frame(right_col)
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

# Button to force a full update (re-read occupancies and recalc everything).
force_update_button = ttk.Button(occ_entry_frame, text="Force Update", command=update_occupancies)
force_update_button.grid(row=3, column=0, columnspan=2, pady=5)

def main(write_excel: bool = True):
    """Entry point for running the GUI application.

    Parameters
    ----------
    write_excel : bool, optional
        When ``True`` the initial intensities are written to an Excel
        file in the configured downloads directory.
    """

    params_file_path = get_path("parameters_file")
    if os.path.exists(params_file_path):
        load_parameters(
            params_file_path,
            theta_initial_var,
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
            center_y_var
        )
        print("Loaded saved profile from", params_file_path)
    else:
        print("No saved profile found; using default parameters.")

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
        main(write_excel=not args.no_excel)
    except Exception as exc:
        print("Unhandled exception during startup:", exc)
        import traceback
        traceback.print_exc()
