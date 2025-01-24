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
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

from IPython.display import display

import scipy.optimize
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt
from numba import njit, int64, float64, prange

import spglib
from skimage.metrics import mean_squared_error

# For the updated 1D plotting:
import plotly.graph_objects as go

matplotlib.use('TkAgg')

# --- Import from local modules ---
from ra_sim.io.file_parsing import parse_poni_file, Open_ASC
from ra_sim.utils.tools import (
    setup_azimuthal_integrator,
    miller_generator,
    view_azimuthal_radial,
    detect_blobs
)
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
# -- The NEW local solver calls (updated optimization.py) --
from ra_sim.fitting.optimization import (
    run_optimization_positions_geometry_local,
)
import OSC_Reader
from OSC_Reader import read_osc


###############################################################################
#               DEFINE ALL DEFAULTS, LOAD FILES, ETC.
###############################################################################

# Load the data
file_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Varying\Images\darkImg.osc"
BI = OSC_Reader.read_osc(file_path)

file_path_1 = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Analysis\Bi2Se3\In-Plane\5\Bi2Se3_5d_5m.asc"
file_path_2 = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Analysis\Bi2Se3\In-Plane\10\Bi2Se3_10d_5m.asc"

poni_file_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL.07.25.2024\Analysis\geometry.poni"
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

import math

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

background_image_1 = Open_ASC(file_path_1) - BI
background_image_2 = Open_ASC(file_path_2) - BI
background_images = [background_image_1, background_image_2]
current_background_image = background_image_1
current_background_index = 0
background_visible = True

measured_peaks = np.load(r"C:\Users\Kenpo\OneDrive\Documents\GitHub\blobs.npy", allow_pickle=True)

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

###############################################################################
#   GUI Setup
###############################################################################

global_image_buffer = np.zeros((image_size, image_size), dtype=np.float64)

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

center_marker, = ax.plot(center_default[1], 3000 - center_default[0],
                         'ro', markersize=5, zorder=2)

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

profile_cache = {
    "beam_x_array": None,
    "beam_y_array": None,
    "beam_intensity_array": None,
    "beta_array": None,
    "kappa_array": None,
    "mosaic_intensity_array": None,
    "theta_array": None,
    "phi_array": None,
    "divergence_intensity_array": None
}

def update_mosaic_cache():
    global profile_cache
    sigma_mosaic_deg = sigma_mosaic_var.get()
    gamma_mosaic_deg = gamma_mosaic_var.get()
    eta_val = eta_var.get()
    (beam_x_array, beam_y_array, beam_intensity_array,
     beta_array, kappa_array, mosaic_intensity_array,
     theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
         num_samples=num_samples,
         divergence_sigma=divergence_sigma,
         bw_sigma=bw_sigma,
         sigma_mosaic_deg=sigma_mosaic_deg,
         gamma_mosaic_deg=gamma_mosaic_deg,
         eta=eta_val
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

def on_mosaic_slider_change(*args):
    update_mosaic_cache()
    update()

def update():
    global profile_cache
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

    center_marker.set_xdata([center_y_updated])
    center_marker.set_ydata([3000 - center_x_updated])

    image_display.set_clim(vmax=current_vmax)
    global_image_buffer.fill(0.0)

    beam_x_arr = profile_cache["beam_x_array"]
    beam_y_arr = profile_cache["beam_y_array"]
    beam_intensity_arr = profile_cache["beam_intensity_array"]
    beta_arr = profile_cache["beta_array"]
    kappa_arr = profile_cache["kappa_array"]
    mosaic_intensity_arr = profile_cache["mosaic_intensity_array"]
    theta_arr = profile_cache["theta_array"]
    phi_arr = profile_cache["phi_array"]
    divergence_intensity_arr = profile_cache["divergence_intensity_array"]

    updated_image, max_positions_local = process_peaks_parallel(
        miller,
        intensities,
        image_size,
        a_updated,
        c_updated,
        lambda_,
        global_image_buffer,  # pass reference
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

    image_display.set_data(global_image_buffer)
    image_display.set_clim(vmax=current_vmax)

    if colorbar_handle:
        colorbar_handle.remove()
    if legend_handle:
        legend_handle.remove()
    legend_handle = ax.legend(['Simulated Diffraction Pattern'], loc='upper left', fontsize='small')

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

def make_slider(label_str, min_val, max_val, init_val, step):
    return create_slider(
        label_str, min_val, max_val, init_val, step,
        parent=slider_frame, update_callback=update
    )

theta_initial_var, theta_initial_scale = make_slider('Theta Initial', 5.0, 20.0, defaults['theta_initial'], 0.01)
gamma_var, gamma_scale = make_slider('Gamma', -4, 4, defaults['gamma'], 0.001)
Gamma_var, Gamma_scale = make_slider('Detector Rotation Gamma', -4, 4, defaults['Gamma'], 0.001)
chi_var, chi_scale = make_slider('Chi', -1, 1, defaults['chi'], 0.001)
zs_var, zs_scale = make_slider('Zs', -2.0e-3, 2e-3, defaults['zs'], 0.0001)
zb_var, zb_scale = make_slider('Zb', -2.0e-3, 2e-3, defaults['zb'], 0.0001)
debye_x_var, debye_x_scale = make_slider('Debye Qz', 0.0, 1.0, defaults['debye_x'], 0.001)
debye_y_var, debye_y_scale = make_slider('Debye Qr', 0.0, 1.0, defaults['debye_y'], 0.001)
corto_detector_var, corto_detector_scale = make_slider('CortoDetector', 0.0, 100e-3, defaults['corto_detector'], 0.1e-3)
a_var, a_scale = make_slider('a (Å)', 3.5, 8.0, defaults['a'], 0.01)
c_var, c_scale = make_slider('c (Å)', 20.0, 40.0, defaults['c'], 0.01)
vmax_var, vmax_scale = make_slider('Max Value', 1e4, 1e8, defaults['vmax'], 1e4)

def make_mosaic_slider(label_str, min_val, max_val, init_val, step):
    return create_slider(
        label_str, min_val, max_val, init_val, step,
        parent=slider_frame,
        update_callback=on_mosaic_slider_change
    )

sigma_mosaic_var, sigma_mosaic_scale = make_mosaic_slider(
    'Sigma Mosaic (deg)', 0.0, 5.0, defaults['sigma_mosaic_deg'], 0.01
)
gamma_mosaic_var, gamma_mosaic_scale = make_mosaic_slider(
    'Gamma Mosaic (deg)', 0.0, 5.0, defaults['gamma_mosaic_deg'], 0.01
)
eta_var, eta_scale = make_mosaic_slider(
    'Eta (fraction)', 0.0, 1.0, defaults['eta'], 0.001
)

center_x_var, center_x_scale = make_slider('Beam Center Y', center_default[0]-100.0, center_default[0]+100.0, defaults['center_x'], 1.0)
center_y_var, center_y_scale = make_slider('Beam Center X', center_default[1]-100.0, center_default[1]+100.0, defaults['center_y'], 1.0)

iteration_counter = {'count': 0}
best_chi_square = [np.inf]

progress_label_geometry = ttk.Label(root, text="")
progress_label_geometry.pack(side=tk.LEFT, padx=5)

progress_label_mosaic = ttk.Label(root, text="")
progress_label_mosaic.pack(side=tk.LEFT, padx=5)

fit_button_geometry = ttk.Button(
    root,
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
        update_gui=update
    )
)
fit_button_geometry.pack(side=tk.LEFT, padx=5)


###############################################################################
# ADDED/CHANGED CODE FOR CONTINUOUSLY UPDATED PLOT
###############################################################################

# A global flag to indicate whether the continuous 1D-plot is running
mosaic_plot_running = False
# A global reference to the plotly figure (so we can reuse it)
mosaic_fig = None

def int_vs_tth(gamma_const, Gamma_const, dist_const, theta_i_const,
               image_size, av, cv, lambda_, psi, center, image):
    """
    Perform integration over 2theta and update or create the Plotly figure.
    """
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    # Build an AzimuthalIntegrator using updated geometry
    ai = AzimuthalIntegrator(
        dist=dist_const,
        poni1=center[0],     # caution: these might need re-checking if your geometry differs
        poni2=center[1],
        pixel1=100e-6,
        pixel2=100e-6,
        rot1=Gamma_const,
        rot2=gamma_const,
        rot3=0.0,
        wavelength=lambda_ * 1e-10  # if your lambda_ is in Å
    )

    main_1d = ai.integrate1d(
        image,
        npt=3000,
        radial_range=(0.1, 50),      # in degrees if unit="2th_deg"
        azimuth_range=(0, 360),
        unit="2th_deg",
        error_model="poisson"
    )

    x_values = main_1d.radial      # 2theta
    y_values = main_1d.intensity
    y_errors = main_1d.sigma if main_1d.sigma is not None else np.sqrt(np.abs(y_values))

    # If the global figure doesn't exist, create it once. Otherwise, just update it.
    global mosaic_fig
    if mosaic_fig is None:
        # Create a new figure the first time
        mosaic_fig = go.Figure()
        mosaic_fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            error_y=dict(type='data', array=y_errors, visible=True),
            name="Simulated"
        ))
        mosaic_fig.update_layout(
            title='Simulated Mosaic Diffraction Pattern',
            xaxis_title='2θ (degrees)',
            yaxis_title='Intensity'
        )
        mosaic_fig.show()
    else:
        # Just update the data in the existing figure
        mosaic_fig.data[0].x = x_values
        mosaic_fig.data[0].y = y_values
        mosaic_fig.data[0].error_y.array = y_errors
        mosaic_fig.update_layout(title='Simulated Mosaic Diffraction Pattern (Updated)')

def update_1d_plot():
    """
    Repeatedly called if mosaic_plot_running is True.
    Calls int_vs_tth(...) to integrate and update the figure.
    """
    global mosaic_plot_running, updated_image_global

    if not mosaic_plot_running:
        return  # If toggled off, do nothing.

    if updated_image_global is None:
        # We have no updated_image yet, so just schedule next run
        root.after(2000, update_1d_plot)
        return

    # Call your integration/plot function
    int_vs_tth(
        gamma_const=gamma_var.get(),
        Gamma_const=Gamma_var.get(),
        dist_const=corto_detector_var.get(),
        theta_i_const=theta_initial_var.get(),
        image_size=image_size,
        av=a_var.get(),
        cv=c_var.get(),
        lambda_=lambda_,
        psi=psi,
        center=[center_x_var.get(), center_y_var.get()],
        image=updated_image_global
    )

    # Schedule the next update after 2 seconds (modify as desired)
    root.after(2000, update_1d_plot)

def toggle_mosaic_plot():
    """
    Toggles the global mosaic_plot_running flag.
    If turning on, calls update_1d_plot() once, which keeps re-calling itself.
    """
    global mosaic_plot_running, mosaic_fig
    mosaic_plot_running = not mosaic_plot_running
    if mosaic_plot_running:
        progress_label_mosaic.config(text="1D Slice: ON")
        # Start or re-start the updating
        update_1d_plot()
    else:
        progress_label_mosaic.config(text="1D Slice: OFF")
        # Optionally clear out the mosaic_fig or let it remain
        # mosaic_fig = None

###############################################################################


fit_button_mosaic = ttk.Button(
    root,
    text="Show 1D Slice",
    command=toggle_mosaic_plot
)
fit_button_mosaic.pack(side=tk.LEFT, padx=5)


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

# Initialize mosaic cache, then an update
update_mosaic_cache()
update()

root.mainloop()
