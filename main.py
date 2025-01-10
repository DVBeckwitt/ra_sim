import os
import time
import json
import pickle
import threading
import traceback
import itertools

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import fabio
import pyFAI
from pyFAI.gui import jupyter
from pyFAI.gui.jupyter.calib import Calibration
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

from IPython.display import display

import scipy.optimize
from scipy.optimize import differential_evolution, least_squares

from numba import njit, int64, float64, prange

import spglib
from bayes_opt import BayesianOptimization
from skimage.metrics import mean_squared_error

matplotlib.use('TkAgg')

# --- Import from local modules ---
from ra_sim.io.file_parsing import parse_poni_file, Open_ASC
from ra_sim.utils.tools import (
    setup_azimuthal_integrator,
    miller_generator,
    view_azimuthal_radial,
    detect_blobs
)
from ra_sim.io.data_loading import load_and_format_reference_profiles, save_all_parameters, load_parameters
from ra_sim.StructureFactor.AtomicCoordinates import get_Atomic_Coordinates
from ra_sim.StructureFactor.StructureFactor import calculate_structure_factor
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import process_peaks_parallel
from ra_sim.simulation.simulation import simulate_diffraction
from ra_sim.gui.sliders import create_slider
from ra_sim.fitting.optimization import (
    run_optimization_positions_geometry,  # geometry-only
    run_optimization_mosaic              # mosaic-only
)

# -------------------------------------------------
#               DEFINE ALL DEFAULTS
# -------------------------------------------------

file_path_1 = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\In-Plane\3\Bi2Se3_6d_5m.asc"
file_path_2 = r"C:\Users\Kenpo\OneDrive\Research\Uniaxially Twisted Project\UniTwist2D\subroutines\Bi2Se3_10d_5m.asc"

poni_file_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\geometry.poni"
parameters = parse_poni_file(poni_file_path)
ai_initial = setup_azimuthal_integrator(parameters)

mx = 19
num_samples = 1000

q_c = 3.286
av = 4.14
bv = av
cv = 28.63600

lambda_ = ai_initial._wavelength * 1e10
center_default = [3000 - ai_initial.poni2 * 1e4, 3000 - ai_initial.poni1 * 1e4]

fwhm2sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
divergence_sigma = np.deg2rad(0.1 * fwhm2sigma)

sigma_mosaic = np.deg2rad(0.8 * fwhm2sigma)
gamma_mosaic = np.deg2rad(0.7 * fwhm2sigma)
eta = 0.0

theta_initial = 6
gamma_initial = ai_initial.rot2
Gamma_initial = ai_initial.rot1

chi = 0.0
psi = 0.0

zb = 0.0
bw_sigma = 0.25e-3 * fwhm2sigma
Distance_CoR_to_Detector = ai_initial.dist

zs = 0.0
debye_x = 0.0
debye_y = 0.0
n2 = 1

lattice = [[av, 0, 0], [0, bv, 0], [0, 0, cv]]
positions = np.array([
    [0.00000, 0.00000, 0.40080],  # Bi
    [0.00000, 0.00000, 0.00000],  # Se1
    [0.00000, 0.00000, 0.21170]   # Se2
])
numbers = [83, 34, 34]
atomic_labels = ['Bi', 'Se1', 'Se2']
space_group_operations = spglib.get_symmetry_from_database(458)

input_file = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\In-Plane\3\integrated_peak_intensities_real.npy"
reference_profiles = load_and_format_reference_profiles(input_file)

data = np.load(r"C:\Users\Kenpo\OneDrive\Research\Uniaxially Twisted Project\UniTwist2D\repo\AFF.npy", allow_pickle=True)

q = sp.Symbol('q')
Pbsol = sp.lambdify(q, data[0], modules='numpy')
Isol = sp.lambdify(q, data[1], modules='numpy')
data = [Pbsol, Isol]

occ = np.array([1, 1, 1])
cell_params, atoms = get_Atomic_Coordinates(positions, space_group_operations, atomic_labels)
miller, intensities = miller_generator(mx, av, cv, lambda_, atoms, data, occ)

image_size = 3000

background_image_1 = Open_ASC(file_path_1)
background_image_2 = Open_ASC(file_path_2)
background_images = [background_image_1, background_image_2]
current_background_image = background_image_1
current_background_index = 0
background_visible = True

measured_peaks = np.load(r"C:\Users\Kenpo\Downloads\blobs.npy", allow_pickle=True)

defaults = {
    'theta_initial': 6.0,
    'gamma': gamma_initial,
    'Gamma': Gamma_initial,
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
    'vmax': 1e6,
    'center_x': center_default[0],
    'center_y': center_default[1]
}

# -------------------------------------------------
# Caching random profiles at startup
# -------------------------------------------------
profile_cache = None
def generate_profiles_once(
    num_samples=1000,
    divergence_sigma=divergence_sigma,
    bw_sigma=bw_sigma,
    sigma_mosaic_deg=np.degrees(sigma_mosaic),
    gamma_mosaic_deg=np.degrees(gamma_mosaic),
    eta=eta
):
    global profile_cache
    (beam_x_array, beam_y_array, beam_intensity_array,
     beta_array, kappa_array, mosaic_intensity_array,
     theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
         num_samples,
         divergence_sigma,
         bw_sigma,
         sigma_mosaic_deg,
         gamma_mosaic_deg,
         eta
    )
    profile_cache = {
        "beam_x_array": beam_x_array,
        "beam_y_array": beam_y_array,
        "beam_intensity_array": beam_intensity_array,
        "beta_array": beta_array,
        "kappa_array": kappa_array,
        "mosaic_intensity_array": mosaic_intensity_array,
        "theta_array": theta_array,
        "phi_array": phi_array,
        "divergence_intensity_array": divergence_intensity_array
    }

# Call once to generate random distributions
generate_profiles_once()

# -------------------------------------------------
# Reusable image buffer to avoid repeated np.zeros
# -------------------------------------------------
global_image_buffer = np.zeros((image_size, image_size), dtype=np.float64)

# --- Tkinter GUI ---
root = tk.Tk()
root.title("Controls and Sliders")

# Create top-level window for the figure
fig_window = tk.Toplevel(root)
fig_window.title("Main Figure")
fig_frame = ttk.Frame(fig_window)
fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas_frame = ttk.Frame(fig_frame)
canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

vmax_frame = ttk.Frame(fig_frame)
vmax_frame.pack(side=tk.BOTTOM, fill=tk.X)

fig, ax = plt.subplots(figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas.draw()

vmax_label = ttk.Label(vmax_frame, text="Max Value")
vmax_label.pack(side=tk.LEFT, padx=5)

vmax_var = tk.DoubleVar(value=defaults['vmax'])
def vmax_slider_command(val):
    v = float(val)
    vmax_var.set(v)
    update()
vmax_slider = ttk.Scale(
    vmax_frame,
    from_=1e4,
    to=1e8,
    orient=tk.HORIZONTAL,
    variable=vmax_var,
    command=vmax_slider_command
)
vmax_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

slider_frame = ttk.Frame(root, padding=10)
slider_frame.pack(side=tk.LEFT, fill=tk.Y)

background_display = ax.imshow(
    current_background_image,
    cmap='turbo',
    vmin=0,
    vmax=1e3,
    zorder=0,
    origin='upper'
)

# Red center marker
center_marker, = ax.plot(center_default[1], 3000 - center_default[0],
                         'ro', markersize=5, zorder=2)

# Show one initial (dummy) diffraction image
image_display = ax.imshow(
    global_image_buffer,
    cmap='turbo',
    vmin=0,
    vmax=1e5,
    alpha=0.5,
    zorder=1,
    origin='upper'
)
colorbar = plt.colorbar(image_display, ax=ax, label='Intensity', shrink=0.6, pad=0.02)
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.title('Simulated Diffraction Pattern')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
ax.set_ylim(3000 - center_default[0], 0)

legend_handle = None
colorbar_handle = None

peak_positions = []
peak_millers = []
peak_intensities = []
updated_image_global = None

def update():
    global legend_handle, colorbar_handle
    global updated_image_global, peak_positions, peak_millers, peak_intensities
    global center_marker, global_image_buffer

    peak_positions = []
    peak_millers = []
    peak_intensities = []

    gamma_updated = round(gamma_var.get(), 3)
    Gamma_updated = round(Gamma_var.get(), 3)
    chi_updated   = round(chi_var.get(), 3)
    zs_updated    = round(zs_var.get(), 6)
    zb_updated    = round(zb_var.get(), 6)
    a_updated     = a_var.get()
    c_updated     = c_var.get()
    theta_initial_updated = round(theta_initial_var.get(), 2)
    debye_x_updated       = round(debye_x_var.get(), 3)
    debye_y_updated       = round(debye_y_var.get(), 3)
    corto_detector_updated= round(corto_detector_var.get(), 4)
    current_vmax         = vmax_var.get()

    center_x_updated     = round(center_x_var.get())
    center_y_updated     = round(center_y_var.get())
    current_center       = [center_x_updated, center_y_updated]

    # Move red dot
    center_marker.set_xdata([center_y_updated])
    center_marker.set_ydata([3000 - center_x_updated])

    image_display.set_clim(vmax=current_vmax)

    # 1) Fill image buffer with 0 instead of making a new array
    global_image_buffer.fill(0.0)

    # 2) Grab cached profiles
    global profile_cache
    beam_x_arr = profile_cache["beam_x_array"]
    beam_y_arr = profile_cache["beam_y_array"]
    beam_intensity_arr = profile_cache["beam_intensity_array"]
    beta_arr = profile_cache["beta_array"]
    kappa_arr = profile_cache["kappa_array"]
    mosaic_intensity_arr = profile_cache["mosaic_intensity_array"]
    theta_arr = profile_cache["theta_array"]
    phi_arr = profile_cache["phi_array"]
    divergence_intensity_arr = profile_cache["divergence_intensity_array"]

    # 3) Actually fill the buffer by calling process_peaks_parallel
    updated_image, max_positions_local = process_peaks_parallel(
        miller,
        intensities,
        image_size,
        a_updated,
        c_updated,
        lambda_,
        global_image_buffer,  # pass the same buffer
        corto_detector_updated,
        gamma_updated,
        Gamma_updated,
        chi_updated,
        psi,
        zs_updated,
        zb_updated,
        n2,
        beam_x_arr,
        beam_y_arr,
        beam_intensity_arr,
        beta_arr,
        kappa_arr,
        mosaic_intensity_arr,
        theta_arr,
        phi_arr,
        divergence_intensity_arr,
        debye_x_updated,
        debye_y_updated,
        current_center,
        theta_initial_updated,
        theta_initial_updated + 0.1,
        0.1,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0])
    )

    # 4) Build single-peak arrays
    import math
    for i in range(len(miller)):
        H, K, L = miller[i]
        mx0, my0, mv0, mx1, my1, mv1 = max_positions_local[i, :]

        if not (math.isnan(mx0) or math.isnan(my0)):
            peak_positions.append((int(round(mx0)), int(round(my0))))
            peak_intensities.append(mv0)
            peak_millers.append((H, K, L))

        if not (math.isnan(mx1) or math.isnan(my1)):
            peak_positions.append((int(round(mx1)), int(round(my1))))
            peak_intensities.append(mv1)
            peak_millers.append((H, K, L))

    # 5) Update displayed image
    image_display.set_data(global_image_buffer)  # same reference
    image_display.set_clim(vmax=current_vmax)

    if colorbar_handle:
        colorbar_handle.remove()
    if legend_handle:
        legend_handle.remove()

    legend_handle = ax.legend(['Simulated Diffraction Pattern'], loc='upper left', fontsize='small')

    # 6) Compute chi²
    try:
        if np.max(global_image_buffer) > 0:
            updated_image_norm = global_image_buffer / np.max(global_image_buffer)
        else:
            updated_image_norm = global_image_buffer

        if np.max(current_background_image) > 0:
            reference_norm = current_background_image / np.max(current_background_image)
        else:
            reference_norm = current_background_image

        chi_squared_value = mean_squared_error(reference_norm, updated_image_norm) * updated_image.size
        chi_square_label.config(text=f"Chi-Squared: {chi_squared_value:.2e}")
    except Exception as e:
        chi_square_label.config(text=f"Chi-Squared: Error - {e}")

    updated_image_global = updated_image
    canvas.draw_idle()

# -------------------------------------------------
# CREATE SLIDERS
# -------------------------------------------------
def make_slider(label_str, min_val, max_val, init_val, step):
    return create_slider(
        label_str, min_val, max_val, init_val, step,
        parent=slider_frame, update_callback=update
    )

theta_initial_var, theta_initial_scale = make_slider('Theta Initial', 5.0, 20.0, defaults['theta_initial'], 0.01)
gamma_var, gamma_scale = make_slider('Gamma', -2, 2, defaults['gamma'], 0.001)
Gamma_var, Gamma_scale = make_slider('Detector Rotation Gamma', -2, 2, defaults['Gamma'], 0.001)
chi_var, chi_scale = make_slider('Chi', -1, 1, defaults['chi'], 0.001)
zs_var, zs_scale = make_slider('Zs', 0.0, 2e-3, defaults['zs'], 0.0001)
zb_var, zb_scale = make_slider('Zb', 0.0, 2e-3, defaults['zb'], 0.0001)
debye_x_var, debye_x_scale = make_slider('Debye Qz', 0.0, 1.0, defaults['debye_x'], 0.001)
debye_y_var, debye_y_scale = make_slider('Debye Qr', 0.0, 1.0, defaults['debye_y'], 0.001)
corto_detector_var, corto_detector_scale = make_slider('CortoDetector', 0.0, 100e-3, defaults['corto_detector'], 0.1e-3)
sigma_mosaic_var, sigma_mosaic_scale = make_slider('Sigma Mosaic (deg)', 0.0, 5.0, defaults['sigma_mosaic_deg'], 0.01)
gamma_mosaic_var, gamma_mosaic_scale = make_slider('Gamma Mosaic (deg)', 0.0, 5.0, defaults['gamma_mosaic_deg'], 0.01)
eta_var, eta_scale = make_slider('Eta (fraction)', 0.0, 1.0, defaults['eta'], 0.001)
a_var, a_scale = make_slider('a (Å)', 3.5, 8.0, defaults['a'], 0.01)
c_var, c_scale = make_slider('c (Å)', 20.0, 40.0, defaults['c'], 0.01)
vmax_var, vmax_scale = make_slider('Max Value', 1e4, 1e8, defaults['vmax'], 1e4)

# Beam center sliders
center_x_var, center_x_scale = make_slider('Beam Center Y', center_default[0]-100.0, center_default[0]+100.0, defaults['center_x'], 1.0)
center_y_var, center_y_scale = make_slider('Beam Center X', center_default[1]-100.0, center_default[1]+100.0, defaults['center_y'], 1.0)

iteration_counter = {'count': 0}
best_chi_square = [np.inf]

def update_progress(result):
    progress_label.config(text=f"Current Best Loss: {result.cost:.2e}\nParameters: {result.x}")

pbounds = {
    'chi': (chi_scale.cget('from'), chi_scale.cget('to')),
    'zs': (zs_scale.cget('from'), zs_scale.cget('to')),
    'zb': (zb_scale.cget('from'), zb_scale.cget('to')),
    'eta': (0.0, 1.0),
    'sigma_mosaic_deg': (0.0, 5.0),
    'gamma_mosaic_deg': (0.0, 5.0)
}

initial_params = {
    'chi': chi_var.get(),
    'zs': zs_var.get(),
    'zb': zb_var.get(),
    'eta': eta_var.get(),
    'sigma_mosaic_deg': sigma_mosaic_var.get(),
    'gamma_mosaic_deg': gamma_mosaic_var.get()
}

# -------------------------------------------------
#  OPTIMIZATION BUTTONS
# -------------------------------------------------
fit_button_geometry = ttk.Button(
    root,
    text="Optimize Geometry Only",
    command=lambda: run_optimization_positions_geometry(
        fit_button=fit_button_geometry,
        progress_label=progress_label_geometry,
        miller=miller,
        intensities=intensities,
        image_size=image_size,
        av=av,
        cv=cv,
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
        update_gui=update
    )
)
fit_button_geometry.pack(side=tk.LEFT, padx=5)

progress_label_geometry = ttk.Label(root, text="")
progress_label_geometry.pack(side=tk.LEFT, padx=5)

fit_button_mosaic = ttk.Button(
    root,
    text="Optimize Mosaic Only",
    command=lambda: run_optimization_mosaic(
        fit_button=fit_button_mosaic,
        progress_label=progress_label_mosaic,
        gamma_const=gamma_var.get(),
        Gamma_const=Gamma_var.get(),
        dist_const=corto_detector_var.get(),
        theta_i_const=theta_initial_var.get(),
        zs_const=zs_var.get(),
        zb_const=zb_var.get(),
        miller=miller,
        intensities=intensities,
        image_size=image_size,
        av=av,
        cv=cv,
        lambda_=lambda_,
        psi=psi,
        n2=n2,
        center=[center_x_var.get(), center_y_var.get()],
        measured_peaks=measured_peaks
    )
)
fit_button_mosaic.pack(side=tk.LEFT, padx=5)

progress_label_mosaic = ttk.Label(root, text="")
progress_label_mosaic.pack(side=tk.LEFT, padx=5)

def toggle_background():
    global background_visible, background_display, image_display
    if background_visible:
        background_display.set_visible(False)
        image_display.set_alpha(1.0)
    else:
        background_display.set_visible(True)
        image_display.set_alpha(0.5)
    background_visible = not background_visible
    canvas.draw_idle()

def switch_background():
    global current_background_index, background_display
    current_background_index = (current_background_index + 1) % len(background_images)
    background_display.set_data(background_images[current_background_index])
    canvas.draw_idle()

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
    update()

param_button_frame = ttk.LabelFrame(slider_frame, text="Parameters & Actions", padding=5)
param_button_frame.pack(pady=5, fill=tk.X)

bottom_frame = ttk.Frame(slider_frame, padding=5)
bottom_frame.pack(pady=5, fill=tk.X)

reset_button = ttk.Button(bottom_frame, text="Reset to Defaults", command=reset_to_defaults)
reset_button.pack(pady=5, fill=tk.X)

progress_label = ttk.Label(bottom_frame, text="", wraplength=300, justify=tk.LEFT)
progress_label.pack(pady=5, fill=tk.BOTH, expand=True)

top_frame = ttk.Frame(root, padding=5)
top_frame.pack(side=tk.TOP, fill=tk.X)

toggle_button = ttk.Button(top_frame, text="Toggle Background", command=toggle_background)
toggle_button.pack(side=tk.LEFT, padx=5)

switch_button = ttk.Button(top_frame, text="Switch Background", command=switch_background)
switch_button.pack(side=tk.LEFT, padx=5)

reset_button = ttk.Button(top_frame, text="Reset to Defaults", command=reset_to_defaults)
reset_button.pack(side=tk.LEFT, padx=5)

azimuthal_button = ttk.Button(
    top_frame,
    text="Azimuthal vs Radial",
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
            'poni1': (3000 - center_x_var.get()) * 100e-6,
            'poni2': (3000 - center_y_var.get()) * 100e-6,
            'dist': corto_detector_var.get(),
            'rot1': np.deg2rad(Gamma_var.get()),
            'rot2': np.deg2rad(gamma_var.get()),
            'rot3': 0.0,
            'wavelength': ai_initial._wavelength
        }
    )
)
azimuthal_button.pack(side=tk.LEFT, padx=5)

progress_label_positions = ttk.Label(root, text="", wraplength=300, justify=tk.LEFT)
progress_label_positions.pack(side=tk.LEFT, padx=5)

save_button = ttk.Button(
    top_frame,
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
save_button.pack(side=tk.LEFT, padx=5)

load_button = ttk.Button(
    top_frame,
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
        update()
    )
)
load_button.pack(side=tk.LEFT, padx=5)

progress_label = ttk.Label(top_frame, text="", font=("Helvetica", 8))
progress_label.pack(side=tk.LEFT, padx=5)

chi_square_label = ttk.Label(top_frame, text="Chi-Squared: ", font=("Helvetica", 8))
chi_square_label.pack(side=tk.LEFT, padx=5)

info_label = ttk.Label(top_frame, text="", wraplength=200, font=("Helvetica", 8))
info_label.pack(side=tk.LEFT, padx=5)

control_frame = ttk.Frame(root, padding=10, width=200)
control_frame.pack(side=tk.RIGHT, fill=tk.Y)
control_frame.pack_propagate(False)

def on_click(event):
    global peak_positions, peak_millers, peak_intensities
    x_click = event.xdata
    y_click = event.ydata
    if x_click is None or y_click is None:
        return
    x_click = int(round(x_click))
    y_click = int(round(y_click))

    distances = []
    for (px, py) in peak_positions:
        dx = x_click - px
        dy = y_click - py
        distances.append(np.sqrt(dx*dx + dy*dy))
    distances = np.array(distances)
    min_index = int(np.argmin(distances))

    (H, K, L) = peak_millers[min_index]
    best_intensity = peak_intensities[min_index]
    best_px, best_py = peak_positions[min_index]

    info_text = (
        f"Nearest reflection:\n"
        f"H={H}, K={K}, L={L}\n"
        f"Pixel coords: ({best_px}, {best_py})\n"
        f"Sim. intensity: {best_intensity:.1f}"
    )
    print(info_text)
    info_label.config(text=info_text)

fig.canvas.mpl_connect('button_press_event', on_click)

# Trigger initial update
update()

# Start the GUI
root.mainloop()
