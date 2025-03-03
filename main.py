# ==================== main.py ====================
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import pyFAI
import scipy.optimize
from scipy.optimize import differential_evolution, least_squares
from skimage.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog  # for snapshot-saving dialogs
from ra_sim.utils.calculations import IndexofRefraction, fresnel_transmission

# Local modules
from ra_sim.io.file_parsing import parse_poni_file, Open_ASC
from ra_sim.utils.tools import miller_generator, view_azimuthal_radial, detect_blobs
from ra_sim.io.data_loading import (
    load_and_format_reference_profiles,
    save_all_parameters,
    load_parameters
)
from ra_sim.StructureFactor.AtomicCoordinates import get_Atomic_Coordinates
from ra_sim.StructureFactor.StructureFactor import calculate_structure_factor
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import process_peaks_parallel

from ra_sim.simulation.simulation import simulate_diffraction
from ra_sim.gui.sliders import create_slider
from ra_sim.fitting.optimization import (
    run_optimization_positions_geometry_local,
    run_optimization_mosaic_1d_local
)
import numba
import os
import spglib
import OSC_Reader
from OSC_Reader import read_osc

# Select the TkAgg backend for matplotlib, ensuring GUI compatibility
matplotlib.use('TkAgg')
DEBUG_ENABLED = False

###############################################################################
#                          DATA & PARAMETER SETUP
###############################################################################
file_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Varying\Images\darkImg.osc"
BI = OSC_Reader.read_osc(file_path)  # Dark (background) image

file1 = OSC_Reader.read_osc(r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Varying\Images\Bi2Se3_6d_5m.osc")
file2 = OSC_Reader.read_osc(r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Varying\Images\Bi2Se3_10d_5m.osc")
file3 = OSC_Reader.read_osc(r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Varying\Images\Bi2Se3_15d_5m.osc")
file4 = OSC_Reader.read_osc(r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Varying\Images\Bi2Se3_30d_2m.osc")

bg1 = np.load(r"C:\Users\Kenpo\Downloads\background_6d.npy")
bg2 = np.load(r"C:\Users\Kenpo\Downloads\background_10d.npy")
bg3 = np.load(r"C:\Users\Kenpo\Downloads\background_15d.npy")
bg4 = np.load(r"C:\Users\Kenpo\Downloads\background_30d.npy")

# make a list of all files
files = [file1, file2, file3, file4]
bg_data_list = [bg1, bg2, bg3, bg4]

# rotate the images of all files
for i in range(len(files)):
    files[i] = np.rot90(files[i], k=3) - BI

background_images = files

# Parse geometry from a .poni file (pyFAI format)
poni_file_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\geometry.poni"
parameters = parse_poni_file(poni_file_path)

Distance_CoR_to_Detector = parameters.get("Dist", 0.075)   # [m]
Gamma_initial = parameters.get("Rot1", 0.0)                # [rad]
gamma_initial = parameters.get("Rot2", 0.0)                # [rad]
poni1 = parameters.get("Poni1", 0.0)                       # [m]
poni2 = parameters.get("Poni2", 0.0)                       # [m]
wave_m = parameters.get("Wavelength", 1e-10)               # [m]
lambda_ = wave_m * 1e10                                    # [Å]

# Detector image size (square array of 3000×3000)
image_size = 3000

# Approximate beam center, in pixel coordinates (row, column)
center_default = [
    3000 - (poni2 / (100e-6)),  # row index
    3000 - (poni1 / (100e-6))   # column index
]

mx = 19
num_samples = 1000

av = 4.14
bv = av
cv = 28.63600

import math
fwhm2sigma = 1 / (2 * math.sqrt(2 * math.log(2)))  # factor to convert FWHM to σ
divergence_sigma = math.radians(0.05 * fwhm2sigma)
sigma_mosaic = math.radians(0.8 * fwhm2sigma)
gamma_mosaic = math.radians(0.7 * fwhm2sigma)
eta = 0.0

theta_initial = 6.0
chi = 0.0
psi = 0.0
zb = 0.0
bw_sigma = 0.25e-3 * fwhm2sigma
zs = 0.0
debye_x = 0.0
debye_y = 0.0
n2 = IndexofRefraction()

# Bandwidth = 0.7% => 0.7/100
bandwidth = 0.7 / 100  # fractional standard deviation for the beam's bandpass

# Lattice & atomic positions
lattice = [[av, 0, 0], [0, bv, 0], [0, 0, cv]]
positions = np.array([
    [0.00000, 0.00000, 0.40080],  # Bi
    [0.00000, 0.00000, 0.00000],  # Se1
    [0.00000, 0.00000, 0.21170]   # Se2
])
numbers = [83, 34, 34]
atomic_labels = ['Bi', 'Se1', 'Se2']
import spglib
space_group_operations = spglib.get_symmetry_from_database(458)

# Example symmetrical polynomial data from AFF.npy
data = np.load(
    r"C:\Users\Kenpo\OneDrive\Research\Uniaxially Twisted Project\UniTwist2D\repo\AFF.npy",
    allow_pickle=True
)
import sympy as sp
q = sp.Symbol('q')
Pbsol = sp.lambdify(q, data[0], modules='numpy')
Isol = sp.lambdify(q, data[1], modules='numpy')
data = [Pbsol, Isol]

occ = np.array([1, 1, 1])
cell_params, atoms = get_Atomic_Coordinates(positions, space_group_operations, atomic_labels)
miller, intensities = miller_generator(mx, av, cv, lambda_, atoms, data, occ)

# Zero out a beamstop region near the center
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

measured_peaks = np.load(r"C:\Users\Kenpo\OneDrive\Documents\GitHub\blobs.npy", allow_pickle=True)

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
    'center_x': center_default[0],
    'center_y': center_default[1]
}

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

image_display = ax.imshow(
    global_image_buffer,
    cmap='turbo',
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

# --- Main colorbar for normal image ---
colorbar_main = fig.colorbar(image_display, ax=ax, label='Intensity', shrink=0.6, pad=0.02)

# ------------------------------------------------------------------
# Create a dedicated colorbar axis for the caked 2D plot:
# We'll link it to any caked-imshow in code, but keep it hidden
# if we are not in "caking mode."
caked_cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
caked_cbar_ax.set_visible(False)
caked_colorbar = fig.colorbar(image_display, cax=caked_cbar_ax)
caked_colorbar.set_label('Intensity (binned)')
caked_colorbar.ax.set_visible(False)
# ------------------------------------------------------------------

center_marker, = ax.plot(
    center_default[1],
    3000 - center_default[0],
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

vmax_label = ttk.Label(vmax_frame, text="Max Value")
vmax_label.pack(side=tk.LEFT, padx=5)

vmax_var = tk.DoubleVar(value=defaults['vmax'])
def vmax_slider_command(val):
    v = float(val)
    vmax_var.set(v)
    schedule_update()

vmax_slider = ttk.Scale(
    vmax_frame,
    from_=0,
    to=2000,
    orient=tk.HORIZONTAL,
    variable=vmax_var,
    command=vmax_slider_command
)
vmax_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

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

# 2θ (tth) Min
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

# 2θ (tth) Max
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

# φ (phi) Min
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

# φ (phi) Max
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

from collections import namedtuple

def caking(data, ai):
    """
    Perform 2D integration (caking) using pyFAI.
    Returns the pyFAI integrate2d result.
    """
    return ai.integrate2d(
        data,
        npt_rad=1000,
        npt_azim=720,
        correctSolidAngle=True,
        method="lut",
        unit="2th_deg"
    )

def caked_up(res2, tth_min, tth_max, phi_min, phi_max):
    """
    Extract 1D lineouts from a 2D integrated result (res2).
    """
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

def tth_phi_to_pixel(tth_deg, phi_deg, dist_m, px_size_m, center_row, center_col):
    tth_rad = np.deg2rad(tth_deg)
    phi_rad = np.deg2rad(phi_deg)
    r_pixels = (dist_m / px_size_m) * np.tan(tth_rad)
    x = center_col + r_pixels * np.cos(phi_rad)
    y = center_row + r_pixels * np.sin(phi_rad)
    return x, y

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
}

last_res2_background = None
last_res2_sim = None

def update_mosaic_cache():
    """
    Rebuild random mosaic profiles if mosaic sliders changed.
    """
    global profile_cache
    sigma_mosaic_deg = sigma_mosaic_var.get()
    gamma_mosaic_deg = gamma_mosaic_var.get()
    eta_val = eta_var.get()

    (beam_x_array,
     beam_y_array,
     theta_array,
     phi_array,
     wavelength_array) = generate_random_profiles(
         num_samples=num_samples,
         divergence_sigma=divergence_sigma,
         bw_sigma=bw_sigma,
         lambda0=lambda_,     # nominal wavelength (Å)
         bandwidth=bandwidth  # e.g. 0.007 for ±0.7%
     )

    profile_cache = {
        "beam_x_array": beam_x_array,
        "beam_y_array": beam_y_array,
        "theta_array": theta_array,
        "phi_array": phi_array,
        "wavelength_array": wavelength_array,  # store new bandpass array
        "sigma_mosaic_deg": sigma_mosaic_deg,
        "gamma_mosaic_deg": gamma_mosaic_deg,
        "eta": eta_val
    }

def on_mosaic_slider_change(*args):
    update_mosaic_cache()
    schedule_update()

line_rmin, = ax.plot([], [], color='white', linestyle='-', linewidth=2, zorder=5)
line_rmax, = ax.plot([], [], color='white', linestyle='-', linewidth=2, zorder=5)
line_amin, = ax.plot([], [], color='cyan',  linestyle='-', linewidth=2, zorder=5)
line_amax, = ax.plot([], [], color='cyan',  linestyle='-', linewidth=2, zorder=5)

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

def get_bg_signature(ai, bg_image):
    return (
        round(ai.dist, 6),
        round(ai.poni1, 6),
        round(ai.poni2, 6),
        round(ai.rot1, 6),
        round(ai.rot2, 6),
        round(ai.rot3, 6),
        round(ai.wavelength, 9),
        id(bg_image)
    )

def get_sim_signature(
    gamma_val,
    Gamma_val,
    chi_val,
    zs_val,
    zb_val,
    debye_x_val,
    debye_y_val,
    a_val,
    c_val,
    theta_init_val,
    center_x_val,
    center_y_val,
    mosaic_dict
):
    return (
        round(gamma_val, 6),
        round(Gamma_val, 6),
        round(chi_val, 6),
        round(zs_val, 9),
        round(zb_val, 9),
        round(debye_x_val, 6),
        round(debye_y_val, 6),
        round(a_val, 6),
        round(c_val, 6),
        round(theta_init_val, 6),
        round(center_x_val, 3),
        round(center_y_val, 3),
        round(mosaic_dict["sigma_mosaic_deg"], 6),
        round(mosaic_dict["gamma_mosaic_deg"], 6),
        round(mosaic_dict["eta"], 6)
    )

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

_transform_cache = {}

@numba.njit(parallel=True)
def compute_transform_grid(phi_vals, th_vals, gamma, Gamma, distance_m):

    n_phi = phi_vals.size
    n_th  = th_vals.size

    cos_g = np.cos(gamma)
    sin_g = np.sin(gamma)
    cos_G = np.cos(Gamma)
    sin_G = np.sin(Gamma)

    T_phi = np.empty((n_phi, n_th), dtype=np.float64)
    T_r   = np.empty((n_phi, n_th), dtype=np.float64)

    for i in numba.prange(n_phi):
        phi = phi_vals[i]
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for j in range(n_th):
            th = th_vals[j]
            sin_th = np.sin(th)
            cos_th = np.cos(th)

            # Denominator
            denom = ((cos_g * sin_G * sin_phi - sin_g * cos_phi) * sin_th
                     + cos_G * cos_g * cos_th)
            lead = distance_m * sin_th / denom

            # Final x,z
            T_x = lead * (sin_phi * cos_g - cos_phi * sin_g * sin_G)
            T_z = lead * (cos_G * cos_phi)

            # T_y = 0 => angle = arctan2(T_z, T_x)
            T_phi[i, j] = np.arctan2(T_z, T_x)
            T_r[i, j]   = np.sqrt(T_x*T_x + T_z*T_z)

    return T_phi, T_r

def get_transformation_matrices(gamma, Gamma, distance_m, radial, azimuthal):
    key = (
        float(gamma),
        float(Gamma),
        float(distance_m),
        radial.shape,
        azimuthal.shape,
        hash(radial.tobytes()),
        hash(azimuthal.tobytes())
    )
    if key in _transform_cache:
        return _transform_cache[key]

    T_phi, T_r = compute_transform_grid(azimuthal, radial, gamma, Gamma, distance_m)
    _transform_cache[key] = (T_phi, T_r)
    return T_phi, T_r

legend_handle = None  # global reference to store the legend

def do_update():
    global update_pending, last_res2_sim, last_res2_background
    global unscaled_image_global, last_simulation_signature

    # Cancel any pending update
    update_pending = None

    # ---------------------------------------------------------------------
    # Retrieve current slider/parameter values and cast to float
    # so that Numba's np.radians(...) doesn't see a Python list
    # ---------------------------------------------------------------------
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

    # Update beam center marker on the main image display
    center_marker.set_xdata([center_y_up])
    center_marker.set_ydata([center_x_up])
    center_marker.set_visible(True)

    # Build mosaic parameters dictionary (used later in simulation)
    mosaic_params = {
        "beam_x_array":     profile_cache.get("beam_x_array", []),
        "beam_y_array":     profile_cache.get("beam_y_array", []),
        "beam_i_array":     profile_cache.get("beam_intensity_array", []),
        "beta_array":       profile_cache.get("beta_array", []),
        "kappa_array":      profile_cache.get("kappa_array", []),
        "mosaic_i_array":   profile_cache.get("mosaic_intensity_array", []),
        "theta_array":      profile_cache.get("theta_array", []),
        "phi_array":        profile_cache.get("phi_array", []),
        "div_i_array":      profile_cache.get("divergence_intensity_array", []),
        # NEW: The per-sample wavelength array
        "wavelength_i_array": profile_cache.get("wavelength_array", []),
        "sigma_mosaic_deg": sigma_mosaic_var.get(),
        "gamma_mosaic_deg": gamma_mosaic_var.get(),
        "eta":              eta_var.get()
    }

    # ---------------------------------------------------------------------
    # Check if simulation parameters have changed; if so, re-run simulation
    # ---------------------------------------------------------------------
    new_sim_sig = get_sim_signature(
        gamma_updated, Gamma_updated, chi_updated,
        zs_updated, zb_updated, debye_x_updated, debye_y_updated,
        a_updated, c_updated, theta_init_up, center_x_up, center_y_up,
        mosaic_params
    )
    if new_sim_sig != last_simulation_signature:
        last_simulation_signature = new_sim_sig

        # Clear old peak lists
        peak_positions.clear()
        peak_millers.clear()
        peak_intensities.clear()

        # Create a new simulation buffer image and update using the peak processing routine
        sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)

        # Pass mosaic_params["wavelength_i_array"] into process_peaks_parallel
        updated_image, max_positions_local, _, _ = process_peaks_parallel(
            miller, intensities, image_size,
            a_updated, c_updated, lambda_, sim_buffer, corto_det_up,
            gamma_updated, Gamma_updated, chi_updated, psi,
            zs_updated, zb_updated, n2,
            mosaic_params["beam_x_array"],
            mosaic_params["beam_y_array"],
            mosaic_params["theta_array"],
            mosaic_params["phi_array"],
            mosaic_params["sigma_mosaic_deg"],
            mosaic_params["gamma_mosaic_deg"],
            mosaic_params["eta"],
            mosaic_params["wavelength_i_array"],  # <--- pass the array
            debye_x_updated, debye_y_updated,
            [center_x_up, center_y_up],
            theta_init_up,
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            save_flag=0
        )
        # For each Miller index, record peak positions (if detected)
        for i, (H, K, L) in enumerate(miller):
            mx0, my0, mv0, mx1, my1, mv1 = max_positions_local[i, :]
            if np.isfinite(mx0) and np.isfinite(my0):
                peak_positions.append((int(round(mx0)), int(round(my0))))
                peak_intensities.append(mv0)
                peak_millers.append((H, K, L))
            else:
                # Use a sentinel (e.g., (-1,-1)) if invalid
                peak_positions.append((-1, -1))
                peak_intensities.append(mv0)
                peak_millers.append((H, K, L))
                
            if np.isfinite(mx1) and np.isfinite(my1):
                peak_positions.append((int(round(mx1)), int(round(my1))))
                peak_intensities.append(mv1)
                peak_millers.append((H, K, L))
            else:
                peak_positions.append((-1, -1))
                peak_intensities.append(mv1)
                peak_millers.append((H, K, L))


        unscaled_image_global = updated_image

    # ---------------------------------------------------------------------
    # Update main simulation image display using normalization relative to background
    # ---------------------------------------------------------------------
    disp_image = scale_image_for_display(unscaled_image_global)
    global_image_buffer[:] = disp_image
    image_display.set_clim(vmin=0, vmax=vmax_var.get())
    image_display.set_data(global_image_buffer)

    # Compute chi-squared between normalized simulation and background (if possible)
    try:
        norm_sim = (global_image_buffer / np.max(global_image_buffer)
                    if np.max(global_image_buffer) > 0 else global_image_buffer)
        norm_bg  = (current_background_image / np.max(current_background_image)
                    if (current_background_image is not None and np.max(current_background_image) > 0)
                    else current_background_image)
        if norm_bg is not None and norm_bg.shape == norm_sim.shape:
            chi_sq_val = mean_squared_error(norm_bg, norm_sim) * norm_sim.size
            chi_square_label.config(text=f"Chi-Squared: {chi_sq_val:.2e}")
        else:
            chi_square_label.config(text="Chi-Squared: N/A")
    except Exception as e:
        chi_square_label.config(text=f"Chi-Squared: Error - {e}")

    # ---------------------------------------------------------------------
    # Set up pyFAI integrator for caking (2D integration)
    # ---------------------------------------------------------------------
    ai = pyFAI.integrator.azimuthal.AzimuthalIntegrator(
        dist=corto_det_up,
        poni1=center_x_up * 100e-6,
        poni2=center_y_up * 100e-6,
        pixel1=100e-6,
        pixel2=100e-6,
        rot1=0.0,
        rot2=0.0,
        rot3=0.0,
        wavelength=1.5406e-10
    )

    compute_caking = show_caked_2d_var.get()

    if compute_caking:
        # ----- Background caking (if background is visible) -----
        if background_visible and (current_background_image is not None):
            res_bg = caking(current_background_image, ai)
            radial_bg = np.deg2rad(res_bg.radial)
            phi_bg = np.deg2rad(((res_bg.azimuthal + 180) % 360) - 180)

            T_phi_bg, T_r_bg = get_transformation_matrices(
                np.deg2rad(gamma_updated), np.deg2rad(Gamma_updated),
                corto_det_up, radial_bg, phi_bg
            )

            from collections import namedtuple
            CakedResult = namedtuple("CakedResult", ["intensity", "radial", "azimuthal"])
            last_res2_background = CakedResult(
                intensity=res_bg.intensity.copy() - bg_data_list[current_background_index],
                radial=T_r_bg,
                azimuthal=T_phi_bg
            )
        else:
            last_res2_background = None

        # ----- Simulation caking -----
        res_sim = caking(global_image_buffer, ai)
        radial_sim = np.deg2rad(res_sim.radial)
        phi_sim = np.deg2rad(((res_sim.azimuthal + 180) % 360) - 180)

        T_phi_sim, T_r_sim = get_transformation_matrices(
            np.deg2rad(gamma_updated), np.deg2rad(Gamma_updated),
            corto_det_up, radial_sim, phi_sim
        )
        sim_intensity = res_sim.intensity.copy()

        from collections import namedtuple
        CakedResult = namedtuple("CakedResult", ["intensity", "radial", "azimuthal"])
        last_res2_sim = CakedResult(
            intensity=sim_intensity,
            radial=T_r_sim,
            azimuthal=T_phi_sim
        )

        # ----- 2D Binning for 1D Integration (both simulation & background) -----
        num_bins_2theta = 1000
        num_bins_phi    = 360

        # --- Simulation 2D binning ---
        intens_flat_sim = last_res2_sim.intensity.ravel()
        T_r_flat_sim    = last_res2_sim.radial.ravel()
        T_phi_flat_sim  = last_res2_sim.azimuthal.ravel()

        two_theta_rad_sim = np.arctan(T_r_flat_sim / corto_det_up)
        two_theta_deg_sim = np.degrees(two_theta_rad_sim)
        phi_deg_sim = np.degrees(T_phi_flat_sim)
        # Shift azimuth values into [-180,180] range
        phi_shifted_sim = np.where(phi_deg_sim >= 0, phi_deg_sim - 180, phi_deg_sim + 180)

        mask_sim = ((intens_flat_sim > 0) &
                    (phi_shifted_sim >= -90) & (phi_shifted_sim <= 90) &
                    (two_theta_deg_sim >= 0) & (two_theta_deg_sim <= 90))
        masked_intens_sim    = intens_flat_sim[mask_sim]
        masked_two_theta_sim = two_theta_deg_sim[mask_sim]
        masked_phi_sim       = phi_shifted_sim[mask_sim]

        bin_sums_sim, tth_edges, phi_edges = np.histogram2d(
            masked_two_theta_sim, masked_phi_sim,
            bins=(num_bins_2theta, num_bins_phi),
            range=[[0, 90], [-90, 90]],
            weights=masked_intens_sim
        )
        bin_counts_sim, _, _ = np.histogram2d(
            masked_two_theta_sim, masked_phi_sim,
            bins=(num_bins_2theta, num_bins_phi),
            range=[[0, 90], [-90, 90]]
        )
        bin_avg_sim = bin_sums_sim / bin_counts_sim
        bin_avg_sim[bin_counts_sim == 0] = np.nan

        integrated_2theta_sim = np.nansum(bin_avg_sim, axis=1)
        integrated_phi_sim    = np.nansum(bin_avg_sim, axis=0)

        # --- Background 2D binning ---
        if background_visible and (last_res2_background is not None):
            intens_flat_bg = last_res2_background.intensity.ravel()
            T_r_flat_bg    = last_res2_background.radial.ravel()
            T_phi_flat_bg  = last_res2_background.azimuthal.ravel()

            two_theta_rad_bg = np.arctan(T_r_flat_bg / corto_det_up)
            two_theta_deg_bg = np.degrees(two_theta_rad_bg)
            phi_deg_bg = np.degrees(T_phi_flat_bg)
            phi_shifted_bg = np.where(phi_deg_bg >= 0, phi_deg_bg - 180, phi_deg_bg + 180)

            mask_bg = ((intens_flat_bg > 0) &
                       (phi_shifted_bg >= -90) & (phi_shifted_bg <= 90) &
                       (two_theta_deg_bg >= 0) & (two_theta_deg_bg <= 90))
            masked_intens_bg    = intens_flat_bg[mask_bg]
            masked_two_theta_bg = two_theta_deg_bg[mask_bg]
            masked_phi_bg       = phi_shifted_bg[mask_bg]

            bin_sums_bg, _, _ = np.histogram2d(
                masked_two_theta_bg, masked_phi_bg,
                bins=(num_bins_2theta, num_bins_phi),
                range=[[0, 90], [-90, 90]],
                weights=masked_intens_bg
            )
            bin_counts_bg, _, _ = np.histogram2d(
                masked_two_theta_bg, masked_phi_bg,
                bins=(num_bins_2theta, num_bins_phi),
                range=[[0, 90], [-90, 90]]
            )
            bin_avg_bg = bin_sums_bg / bin_counts_bg
            bin_avg_bg[bin_counts_bg == 0] = np.nan

            integrated_2theta_bg = np.nansum(bin_avg_bg, axis=1)
            integrated_phi_bg    = np.nansum(bin_avg_bg, axis=0)
        else:
            integrated_2theta_bg = np.zeros(num_bins_2theta)
            integrated_phi_bg    = np.zeros(num_bins_phi)

        # ----- Normalize simulation 1D integrations to the background's max -----
        max_bg_radial  = np.nanmax(integrated_2theta_bg)
        max_bg_azimuth = np.nanmax(integrated_phi_bg)

        if np.nanmax(integrated_2theta_sim) > 0:
            integrated_2theta_sim = integrated_2theta_sim * (max_bg_radial / np.nanmax(integrated_2theta_sim))
        if np.nanmax(integrated_phi_sim) > 0:
            integrated_phi_sim = integrated_phi_sim * (max_bg_azimuth / np.nanmax(integrated_phi_sim))

        # Update 1D integration plots if enabled
        if show_1d_var.get():
            tth_centers = (tth_edges[:-1] + tth_edges[1:]) / 2
            phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

            line_1d_rad.set_xdata(tth_centers)
            line_1d_rad.set_ydata(integrated_2theta_sim)
            line_1d_rad_bg.set_xdata(tth_centers)
            line_1d_rad_bg.set_ydata(integrated_2theta_bg)

            line_1d_az.set_xdata(phi_centers)
            line_1d_az.set_ydata(integrated_phi_sim)
            line_1d_az_bg.set_xdata(phi_centers)
            line_1d_az_bg.set_ydata(integrated_phi_bg)

            ax_1d_radial.relim()
            ax_1d_radial.autoscale_view()
            ax_1d_radial.set_xlim(tth_min_var.get(), tth_max_var.get())

            phi_min_eff = -phi_max_var.get()
            phi_max_eff = -phi_min_var.get()
            ax_1d_azim.relim()
            ax_1d_azim.autoscale_view()
            ax_1d_azim.set_xlim(phi_min_eff, phi_max_eff)

            canvas_1d.draw_idle()
        else:
            line_1d_rad.set_xdata([])
            line_1d_rad.set_ydata([])
            line_1d_rad_bg.set_xdata([])
            line_1d_rad_bg.set_ydata([])
            line_1d_az.set_xdata([])
            line_1d_az.set_ydata([])
            line_1d_az_bg.set_xdata([])
            line_1d_az_bg.set_ydata([])
            canvas_1d.draw_idle()

        # -------------------------------
        # 2D Caked Display: Overlay simulation and background images
        # -------------------------------
        if show_caked_2d_var.get():
            sim_alpha = 0.5 if background_visible else 1.0
            ax.clear()
            # Hide the main colorbar and show the dedicated caked colorbar
            colorbar_main.ax.set_visible(False)
            caked_colorbar.ax.set_visible(True)
            caked_cbar_ax.set_visible(True)

            # First plot the background caked image (if available) with transparency
            if background_visible and (last_res2_background is not None):
                im_bg = ax.imshow(
                    bin_avg_bg.T,
                    origin='lower',
                    extent=[tth_edges[0], tth_edges[-1], phi_edges[0], phi_edges[-1]],
                    aspect='auto',
                    cmap='grey',
                    interpolation='bilinear',
                    alpha=0.5,
                    vmin=0,
                    vmax=500
                )
            # Then overlay the simulation caked image
            im_sim = ax.imshow(
                bin_avg_sim.T,
                origin='lower',
                extent=[tth_edges[0], tth_edges[-1], phi_edges[0], phi_edges[-1]],
                aspect='auto',
                cmap='turbo',
                interpolation='bilinear',
                alpha=sim_alpha,
                vmin=0,
                vmax=500
            )
            im_sim.set_clim(0, 500)
            caked_colorbar.mappable = im_sim
            caked_colorbar.update_normal(im_sim)

            ax.set_title("2θ vs φ Binned Heatmap (Overlay)")
            ax.set_xlabel("2θ (degrees)")
            ax.set_ylabel("φ (degrees)")
            ax.set_xlim(0, 70)
            ax.set_ylim(-90, 90)
            canvas.draw_idle()
        else:
            # If 2D caking is not selected, revert to the normal image display
            ax.clear()
            caked_colorbar.ax.set_visible(False)
            caked_cbar_ax.set_visible(False)
            colorbar_main.ax.set_visible(True)

            sim_alpha = 0.5 if background_visible else 1.0

            if background_visible and current_background_image is not None:
                ax.imshow(current_background_image, cmap='turbo', vmin=0, vmax=1e3, zorder=0, origin='upper')

            ax.imshow(global_image_buffer, cmap='turbo', vmin=0, vmax=vmax_var.get(), alpha=sim_alpha, zorder=1, origin='upper')
            center_marker.set_visible(True)
            ax.set_xlim(0, image_size)
            ax.set_ylim(center_default[0], 0)
            ax.set_title("Original Image")
            canvas.draw_idle()
    else:
        # If caking is not computed, show the normal image view.
        ax.clear()
        caked_colorbar.ax.set_visible(False)
        caked_cbar_ax.set_visible(False)
        colorbar_main.ax.set_visible(True)
        if background_visible and current_background_image is not None:
            ax.imshow(current_background_image, cmap='turbo', vmin=0, vmax=1e3, zorder=0, origin='upper')

        sim_alpha = 0.5 if background_visible else 1.0
        if background_visible and current_background_image is not None:
            ax.imshow(current_background_image, cmap='turbo', vmin=0, vmax=1e3, zorder=0, origin='upper')

        ax.imshow(global_image_buffer, cmap='turbo', vmin=0, vmax=vmax_var.get(), alpha=sim_alpha, zorder=1, origin='upper')
        center_marker.set_visible(True)
        ax.set_xlim(0, image_size)
        ax.set_ylim(center_default[0], 0)
        ax.set_title("Original Image")
        canvas.draw_idle()

# --------------------------
# Create mosaic-based sliders
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
    'Theta Initial', 5.0, 20.0, defaults['theta_initial'], 0.0001, left_col
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

vmax_slider.config(command=vmax_slider_command)

top_frame = ttk.Frame(root, padding=5)
top_frame.pack(side=tk.TOP, fill=tk.X)

left_top_frame = ttk.Frame(top_frame)
left_top_frame.pack(side=tk.LEFT, fill=tk.X)

button_frame = ttk.Frame(top_frame)
button_frame.pack(side=tk.RIGHT, anchor='ne', padx=5)

progress_label_positions = ttk.Label(left_top_frame, text="", wraplength=300, justify=tk.LEFT)
progress_label_positions.pack(side=tk.TOP, padx=5)

progress_label_geometry = ttk.Label(left_top_frame, text="")
progress_label_geometry.pack(side=tk.TOP, padx=5)

progress_label = ttk.Label(left_top_frame, text="", font=("Helvetica", 8))
progress_label.pack(side=tk.TOP, padx=5)

chi_square_label = ttk.Label(left_top_frame, text="Chi-Squared: ", font=("Helvetica", 8))
chi_square_label.pack(side=tk.TOP, padx=5)

info_label = ttk.Label(left_top_frame, text="", wraplength=200, font=("Helvetica", 8))
info_label.pack(side=tk.TOP, padx=5)

def toggle_background():
    global background_visible
    background_visible = not background_visible
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
    update_mosaic_cache()
    global last_simulation_signature
    last_simulation_signature = None
    schedule_update()

toggle_button = ttk.Button(
    button_frame,
    text="Toggle Background",
    command=toggle_background
)
toggle_button.pack(side=tk.TOP, padx=5, pady=2)

switch_button = ttk.Button(
    button_frame,
    text="Switch Background",
    command=switch_background
)
switch_button.pack(side=tk.TOP, padx=5, pady=2)

reset_button_top = ttk.Button(
    button_frame,
    text="Reset to Defaults",
    command=reset_to_defaults
)
reset_button_top.pack(side=tk.TOP, padx=5, pady=2)

from ra_sim.simulation.simulation import simulate_diffraction

azimuthal_button = ttk.Button(
    button_frame,
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

from ra_sim.io.data_loading import save_all_parameters, load_parameters

save_button = ttk.Button(
    button_frame,
    text="Save Params",
    command=lambda: save_all_parameters(
        r"C:\Users\Kenpo\parameters.npy",
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
    button_frame,
    text="Load Params",
    command=lambda: (
        progress_label.config(
            text=load_parameters(
                r"C:\Users\Kenpo\parameters.npy",
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

from ra_sim.fitting.optimization import (
    run_optimization_positions_geometry_local,
    run_optimization_mosaic_1d_local
)

fit_button_mosaic_1d = ttk.Button(
    button_frame,
    text="Optimize Mosaic (1D)",
    command=lambda: run_optimization_mosaic_1d_local(
        fit_button=fit_button_mosaic_1d,
        progress_label=progress_label_positions,
        gamma_const=gamma_var.get(),
        Gamma_const=Gamma_var.get(),
        dist_const=corto_detector_var.get(),
        theta_i_const=theta_initial_var.get(),
        zs_const=zs_var.get(),
        zb_const=zb_var.get(),
        chi_const=chi_var.get(),
        a_const=a_var.get(),
        c_const=c_var.get(),
        center_xy=[center_x_var.get(), center_y_var.get()],
        miller=miller,
        intensities=intensities,
        image_size=image_size,
        lambda_=lambda_,
        psi=psi,
        n2=n2,
        rmin=tth_min_var.get(),
        rmax=tth_max_var.get(),
        amin_deg=phi_min_var.get(),
        amax_deg=phi_max_var.get(),
        background_image=current_background_image,
        mosaic_sigma_var=sigma_mosaic_var,
        mosaic_gamma_var=gamma_mosaic_var,
        mosaic_eta_var=eta_var,
        num_samples=num_samples,
        divergence_sigma=divergence_sigma,
        bw_sigma=bw_sigma,
        update_gui=do_update
    )
)
fit_button_mosaic_1d.pack(side=tk.TOP, padx=5, pady=2)

fit_button_geometry = ttk.Button(
    button_frame,
    text="Optimize Geometry Only",
    command=lambda: run_optimization_positions_geometry_local(
        fit_button=fit_button_geometry,
        progress_label=progress_label_geometry,
        miller=miller,
        intensities=intensities,
        image_size=image_size,
        av=a_var.get(),
        cv=c_var.get(),
        lambda_=lambda_,
        psi=psi,
        n2=n2,
        center=[center_x_var.get(), center_y_var.get()],
        measured_peaks=measured_peaks,
        mosaic_sigma_const=sigma_mosaic_var.get(),
        mosaic_gamma_const=gamma_mosaic_var.get(),
        mosaic_eta_const=eta_var.get(),
        gamma_var=gamma_var,
        Gamma_var=Gamma_var,
        dist_var=corto_detector_var,
        theta_i_var=theta_initial_var,
        zs_var=zs_var,
        zb_var=zb_var,
        chi_var=chi_var,
        a_var=a_var,
        c_var=c_var,
        center_x_var=center_x_var,
        center_y_var=center_y_var,
        update_gui=do_update
    )
)
fit_button_geometry.pack(side=tk.TOP, padx=5, pady=2)

show_1d_var = tk.BooleanVar(value=False)
def toggle_1d_plots():
    schedule_update()

check_1d = ttk.Checkbutton(
    button_frame,
    text="Show 1D Integration",
    variable=show_1d_var,
    command=toggle_1d_plots
)
check_1d.pack(side=tk.TOP, padx=5, pady=2)

show_caked_2d_var = tk.BooleanVar(value=False)
def toggle_caked_2d():
    schedule_update()

check_2d = ttk.Checkbutton(
    button_frame,
    text="Show 2D Caking",
    variable=show_caked_2d_var,
    command=toggle_caked_2d
)
check_2d.pack(side=tk.TOP, padx=5, pady=2)

original_rad_sim = []
original_rad_bg  = []
original_az_sim  = []
original_az_bg   = []

log_radial_var = tk.BooleanVar(value=False)
log_azimuth_var = tk.BooleanVar(value=False)

def toggle_log_radial():
    global original_rad_sim, original_rad_bg
    if log_radial_var.get():
        original_rad_sim = np.asarray(line_1d_rad.get_ydata()).copy()
        original_rad_bg  = np.asarray(line_1d_rad_bg.get_ydata()).copy()
        clipped_sim = np.where(original_rad_sim < 1, 1, original_rad_sim)
        clipped_bg  = np.where(original_rad_bg < 1, 1, original_rad_bg)
        line_1d_rad.set_ydata(clipped_sim)
        line_1d_rad_bg.set_ydata(clipped_bg)
        ax_1d_radial.set_yscale('log')
        ax_1d_radial.set_ylabel('Intensity (log)')
    else:
        if len(original_rad_sim) > 0:
            line_1d_rad.set_ydata(original_rad_sim)
        if len(original_rad_bg) > 0:
            line_1d_rad_bg.set_ydata(original_rad_bg)
        ax_1d_radial.set_yscale('linear')
        ax_1d_radial.set_ylabel('Intensity')
    canvas_1d.draw_idle()

def toggle_log_azimuth():
    global original_az_sim, original_az_bg
    if log_azimuth_var.get():
        original_az_sim = line_1d_az.get_ydata().copy()
        original_az_bg  = line_1d_az_bg.get_ydata().copy()
        clipped_sim = np.where(original_az_sim <= 0, 1, original_az_sim)
        clipped_bg  = np.where(original_az_bg <= 0, 1, original_az_bg)
        line_1d_az.set_ydata(clipped_sim)
        line_1d_az_bg.set_ydata(clipped_bg)
        ax_1d_azim.set_yscale('log')
        ax_1d_azim.set_ylabel('Intensity (log)')
    else:
        if len(original_az_sim) > 0:
            line_1d_az.set_ydata(original_az_sim)
        if len(original_az_bg) > 0:
            line_1d_az_bg.set_ydata(original_az_bg)
        ax_1d_azim.set_yscale('linear')
        ax_1d_azim.set_ylabel('Intensity')
    canvas_1d.draw_idle()

check_log_radial = ttk.Checkbutton(
    button_frame,
    text="Log Radial",
    variable=log_radial_var,
    command=toggle_log_radial
)
check_log_radial.pack(side=tk.TOP, padx=5, pady=2)

check_log_azimuth = ttk.Checkbutton(
    button_frame,
    text="Log Azimuth",
    variable=log_azimuth_var,
    command=toggle_log_azimuth
)
check_log_azimuth.pack(side=tk.TOP, padx=5, pady=2)

def save_1d_snapshot():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
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
        "tth_min": tth_min_var.get(),
        "tth_max": tth_max_var.get(),
        "phi_min": phi_min_var.get(),
        "phi_max": phi_max_var.get(),
    }

    data_for_saving = {
        "radials_sim": last_1d_integration_data["radials_sim"],
        "intensities_2theta_sim": last_1d_integration_data["intensities_2theta_sim"],
        "azimuths_sim": last_1d_integration_data["azimuths_sim"],
        "intensities_azimuth_sim": last_1d_integration_data["intensities_azimuth_sim"],
        "radials_bg": last_1d_integration_data["radials_bg"],
        "intensities_2theta_bg": last_1d_integration_data["intensities_2theta_bg"],
        "azimuths_bg": last_1d_integration_data["azimuths_bg"],
        "intensities_azimuth_bg": last_1d_integration_data["intensities_azimuth_bg"],
        "parameters": param_dict,
    }

    np.save(file_path, data_for_saving, allow_pickle=True)
    progress_label.config(text=f"Saved 1D snapshot to {file_path}")

snapshot_button = ttk.Button(
    button_frame,
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
        "beam_x_array":  profile_cache.get("beam_x_array", []),
        "beam_y_array":  profile_cache.get("beam_y_array", []),
        "beam_i_array":  profile_cache.get("beam_intensity_array", []),
        "beta_array":    profile_cache.get("beta_array", []),
        "kappa_array":   profile_cache.get("kappa_array", []),
        "mosaic_i_array":profile_cache.get("mosaic_intensity_array", []),
        "theta_array":   profile_cache.get("theta_array", []),
        "phi_array":     profile_cache.get("phi_array", []),
        "div_i_array":   profile_cache.get("divergence_intensity_array", []),
        # you’d also want the wavelength array if saving Q-space
        "wavelength_i_array": profile_cache.get("wavelength_array", []),
        "sigma_mosaic_deg": sigma_mosaic_var.get(),
        "gamma_mosaic_deg": gamma_mosaic_var.get(),
        "eta": eta_var.get()
    }

    image_result, max_positions_local, q_data, q_count = process_peaks_parallel(
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
        mosaic_params["wavelength_i_array"],  # <--- pass the array
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
    button_frame,
    text="Save Q-Space Snapshot",
    command=save_q_space_representation
)
save_q_button.pack(side=tk.TOP, padx=5, pady=2)

def save_1d_permutations():
    # etc. ...
    # [Left as-is in your code—simply remember to incorporate wavelength_i_array if needed in a loop]
    pass

save_1d_grid_button = ttk.Button(
    button_frame,
    text="Save 1D Grid",
    command=save_1d_permutations
)
save_1d_grid_button.pack(side=tk.TOP, padx=5, pady=2)

def run_debug_simulation():
    """
    Run the debug version of the simulation with current slider/parameter values,
    then dump the debug log.
    """
    from ra_sim.simulation.diffraction_debug import process_peaks_parallel_debug, dump_debug_log

    # 1) Gather current GUI slider values, cast to float
    gamma_val   = float(gamma_var.get())
    Gamma_val   = float(Gamma_var.get())
    chi_val     = float(chi_var.get())
    zs_val      = float(zs_var.get())
    zb_val      = float(zb_var.get())
    a_val       = float(a_var.get())
    c_val       = float(c_var.get())
    theta_val   = float(theta_initial_var.get())
    debye_x_val = float(debye_x_var.get())
    debye_y_val = float(debye_y_var.get())
    corto_val   = float(corto_detector_var.get())
    center_x_val= float(center_x_var.get())
    center_y_val= float(center_y_var.get())

    mosaic_params = {
        "beam_x_array":     profile_cache.get("beam_x_array", []),
        "beam_y_array":     profile_cache.get("beam_y_array", []),
        "theta_array":      profile_cache.get("theta_array", []),
        "phi_array":        profile_cache.get("phi_array", []),
        "sigma_mosaic_deg": sigma_mosaic_var.get(),
        "gamma_mosaic_deg": gamma_mosaic_var.get(),
        "eta":              eta_var.get(),
        "wave_i_array":     profile_cache.get("wavelength_array", [])
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
        mosaic_params["beam_i_array"],
        mosaic_params["beta_array"],
        mosaic_params["kappa_array"],
        mosaic_params["mosaic_i_array"],
        mosaic_params["theta_array"],
        mosaic_params["phi_array"],
        mosaic_params["div_i_array"],
        mosaic_params["wavelength_i_array"],  # debug version also needs bandpass
        debye_x_val,
        debye_y_val,
        [center_x_val, center_y_val],
        theta_val,
        theta_val + 0.1,  # small range for demonstration
        0.1,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0
    )

    progress_label.config(text="Debug simulation complete. Log saved.")

debug_button = ttk.Button(
    button_frame,
    text="Run Debug Simulation",
    command=run_debug_simulation
)
debug_button.pack(side=tk.TOP, padx=5, pady=2)

###############################################################################
# Finally, initialize everything:
###############################################################################
update_mosaic_cache()
do_update()
root.mainloop()
