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

from ra_sim.io.file_parsing import parse_poni_file, Open_ASC
from ra_sim.utils.tools import setup_azimuthal_integrator, miller_generator, view_azimuthal_radial, detect_blobs
from ra_sim.io.data_loading import load_and_format_reference_profiles, save_all_parameters, load_parameters
from ra_sim.StructureFactor.AtomicCoordinates import get_Atomic_Coordinates
from ra_sim.StructureFactor.StructureFactor import calculate_structure_factor
from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import process_peaks_parallel
from ra_sim.simulation.simulation import simulate_diffraction
from ra_sim.gui.sliders import create_slider
from ra_sim.fitting.optimization import (
    run_optimization_positions_geometry,  # geometry-only
    run_optimization_mosaic             # mosaic-only
)

# Load background images
file_path_1 = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\In-Plane\3\Bi2Se3_6d_5m.asc"
file_path_2 = 'C:/Users/Kenpo/OneDrive/Research/Uniaxially Twisted Project/UniTwist2D/subroutines/Bi2Se3_10d_5m.asc'  # Assuming a second background file

# Path to your .poni file
file_path = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\geometry.poni"
parameters = parse_poni_file(file_path)
ai_initial = setup_azimuthal_integrator(parameters)


mx = 18+1  # max number of miller indices to permute
previous_params = None


# Example parameters (replace these with your desired values)
num_samples = 1000

# Constants
q_c = 3.286
av = 4.14
bv = av
cv = 28.63600

lambda_ = ai_initial._wavelength * 1e10 # Given
center = [3000-ai_initial.poni2*1e4, 3000-ai_initial.poni1*1e4]
Diffuse_check = False
diffuse_theta = 5

fwhm2sigma = 1 / (2 * np.sqrt(2 * np.log(2)))

divergence_sigma = np.deg2rad(0.1 * fwhm2sigma)


sigma_mosaic = np.deg2rad(3 * fwhm2sigma)
gamma_mosaic = np.deg2rad(0.7 * fwhm2sigma)
eta = 0.08          # fraction of Lorentzian in pseudo-Voigt

theta_initial = 6

gamma = ai_initial.rot2 # rotation along z of detector
Gamma = ai_initial.rot1  # rotation along x of detector


chi = np.deg2rad(0) # rotation along y of sample
psi = 0  # rotation along z of sample


zb = -0.1 * 1e-3
bw_sigma = 0.25e-3 * fwhm2sigma

Distance_CoR_to_Detector = ai_initial.dist 

zs = 0.0 * 1e-3

debye_x = 0.4
debye_y = 0.5

n2 = 1

# Declare global arrays for beam profile, mosaic profile, and divergence profile
beam_x_array = np.zeros(num_samples)
beam_y_array = np.zeros(num_samples)
beam_intensity_array = np.zeros(num_samples)
beta_array = np.zeros(num_samples)
kappa_array = np.zeros(num_samples)
mosaic_intensity_array = np.zeros(num_samples)
theta_array = np.zeros(num_samples)
phi_array = np.zeros(num_samples)
divergence_intensity_array = np.zeros(num_samples)


# Define lattice parameters and atomic positions
#av, bv, cv = 4.14300, 4.14300, 28.63600
lattice = [[av, 0, 0], [0, bv, 0], [0, 0, cv]]  # Define lattice as a list of lists

# Define atomic positions
positions = np.array([
    [0.00000, 0.00000, 0.40080],  # Bi
    [0.00000, 0.00000, 0.00000],  # Se1
    [0.00000, 0.00000, 0.21170]   # Se2
])

# Define atomic numbers
numbers = [83, 34, 34]  # Bi=83, Se=34
atomic_labels = ['Bi', 'Se1', 'Se2']

# Get symmetry information for the specific space group (R -3 m)
space_group_operations = spglib.get_symmetry_from_database(458)



input_file = r"C:\Users\Kenpo\OneDrive\Research\Rigaku XRD\ORNL_4_12_24\Analysis\Bi2Se3\In-Plane\3\integrated_peak_intensities_real.npy"
reference_profiles = load_and_format_reference_profiles(input_file)

# Import the data pickleable
data = np.load('C:/Users/Kenpo/OneDrive/Research/Uniaxially Twisted Project/UniTwist2D/repo/AFF.npy', allow_pickle=True)


# Define q as a symbolic variable
q = sp.Symbol('q')
# Read the structure from the file
cell_params, atoms = get_Atomic_Coordinates(positions, space_group_operations, atomic_labels)

# lamdify data[0] and data[1]
Pbsol = sp.lambdify(q, data[0], modules='numpy')
Isol = sp.lambdify(q, data[1], modules='numpy')
data = [Pbsol, Isol]    

occ = np.array([1, 1, 1])  # Occupancies as a numpy array for potential vectorization

miller, intensities = miller_generator(mx, av, cv, lambda_, atoms, data, occ)


image_size = 3000

# Declare global variables for the Bragg peak positions and indices
bragg_positions = []
bragg_miller_indices = []


background_image_1 = Open_ASC(file_path_1)
background_image_2 = Open_ASC(file_path_2)


# load blobs temporarily for testing from npy (later will add feature)
measured_peaks = np.load(r"C:\Users\Kenpo\Downloads\blobs.npy", allow_pickle=True)

current_background_image = background_image_1
background_images = [background_image_1, background_image_2]
current_background_index = 0
background_visible = True


# --- Tkinter GUI Setup ---# Create the main root window
root = tk.Tk()
root.title("Controls and Sliders")

# Create top-level window for the figure
fig_window = tk.Toplevel(root)
fig_window.title("Main Figure")

# Define fig_frame before using it
fig_frame = ttk.Frame(fig_window)
fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Now create the figure and canvas after fig_frame is defined
fig, ax = plt.subplots(figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=fig_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas.draw()



# Add sliders and controls in root
slider_frame = ttk.Frame(root, padding=10)
slider_frame.pack(side=tk.LEFT, fill=tk.Y)


# Display the background image first
background_display = ax.imshow(current_background_image, cmap='turbo', vmin=0, vmax=1e3, zorder=0, origin='upper')  # Set zorder=0 to keep it at the bottom

# Placeholder for process_peaks_parallel function
# (Assuming this function is properly implemented above with @njit)
# If process_peaks_parallel requires additional setup, ensure it's defined before usage.
image = np.zeros((image_size, image_size))


pixel_size = 0.1  # Define this if not already defined

# Compute coordinates and angles using the high-precision float64 data type
y_coords, x_coords = np.indices(image.shape, dtype=np.int32)
x_coords2 = np.round(x_coords / 100e-6 - center[1]).astype(np.int32)
y_coords2 = np.round(y_coords / 100e-6 + center[0]).astype(np.int32)

r = np.sqrt(x_coords2**2 + y_coords2**2)
theta = np.arctan(r / 75e-3)
phi = np.arctan2(x_coords2, y_coords2)

# Calculate qr and qz for each pixel
k = 2 * np.pi / 1.5406  # Using float64 precision by default

qx = np.sin(theta) * np.sin(phi) * k
cos_th, sin_th = np.cos(theta), np.sin(theta)
cos_phi, sin_phi = np.cos(phi), np.sin(phi)


unit_x = np.array([1.0, 0.0, 0.0])
n_detector = np.array([0.0, 1.0, 0.0])

image,max_positions = process_peaks_parallel(
    miller, intensities, image_size, av, cv, lambda_, np.zeros((image_size, image_size)),
    Distance_CoR_to_Detector, gamma, Gamma, chi, psi, zs, zb, n2,
    beam_x_array, beam_y_array, beam_intensity_array,
    beta_array, kappa_array, mosaic_intensity_array,
    theta_array, phi_array, divergence_intensity_array,
    debye_x, debye_y, center, 6.6, 6.7, 0.1,
    unit_x, n_detector
)


image_display = ax.imshow(image, cmap='turbo', vmin=0, vmax=1e5, alpha=0.5, zorder=1, origin='upper')  # Set alpha to make it semi-transparent
ax.plot(center[1],3000- center[0], 'ro', markersize=5)

colorbar = plt.colorbar(image_display, ax=ax, label='Intensity', shrink=0.6, pad=0.02)  # Use shrink to make it smaller, pad to reduce space

# Adjust the plot layout to make the image larger
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Reduce margins to make the image fill more of the space

plt.title('Simulated Diffraction Pattern')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')


def update():
    global inset_ax, legend_handle, colorbar_handle
    global bragg_positions, bragg_miller_indices, bragg_pixel_positions
    global updated_image_global
    
    
    # We'll store just one "best sign" peak per reflection in these:
    global peak_positions, peak_millers
    peak_positions = []
    peak_millers   = []
    global peak_intensities
    peak_intensities = []
    # 1) Read current slider values
    gamma_updated = round(gamma_var.get(), 3)
    Gamma_updated = round(Gamma_var.get(), 3)
    chi_updated   = round(chi_var.get(), 3)
    zs_updated    = round(zs_var.get(), 6)
    zb_updated    = round(zb_var.get(), 6)
    a_updated = a_var.get()   # lattice constant a
    c_updated = c_var.get()   # lattice constant c
    # Some additional parameters:
    theta_initial_updated = round(theta_initial_var.get(), 2)
    debye_x_updated       = round(debye_x_var.get(), 3)
    debye_y_updated       = round(debye_y_var.get(), 3)
    corto_detector_updated= round(corto_detector_var.get(), 2)

    # Mosaic parameters from mosaic sliders
    current_sigma_mosaic = sigma_mosaic_var.get()
    current_gamma_mosaic = gamma_mosaic_var.get()
    current_eta          = eta_var.get()

    # 2) Regenerate random profiles
    (beam_x_array, beam_y_array, beam_intensity_array,
     beta_array, kappa_array, mosaic_intensity_array,
     theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
         num_samples,
         divergence_sigma,
         bw_sigma,
         current_sigma_mosaic,
         current_gamma_mosaic,
         current_eta
    )

    # 3) Call process_peaks_parallel for the new diffraction image
    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])

    updated_image, max_positions = process_peaks_parallel(
        miller, intensities, image_size, a_updated, c_updated, lambda_, np.zeros((image_size, image_size)),
        corto_detector_updated, gamma_updated, Gamma_updated, chi_updated, psi,
        zs_updated, zb_updated, n2,
        beam_x_array, beam_y_array, beam_intensity_array,
        beta_array, kappa_array, mosaic_intensity_array,
        theta_array, phi_array, divergence_intensity_array,
        debye_x_updated, debye_y_updated, center,
        theta_initial_updated, theta_initial_updated + 0.1, 0.1,
        unit_x, n_detector
    )

    # 4) Build single-peak arrays "peak_positions" and "peak_millers"
    import math
    for i in range(len(miller)):
        H, K, L = miller[i]               # reflection
        mx0, my0, mv0, mx1, my1, mv1 = max_positions[i, :]

        import math

        mx0, my0, mv0, mx1, my1, mv1 = max_positions[i]  # from process_peaks_parallel

        # Always store sign=0
        if not (math.isnan(mx0) or math.isnan(my0)):
            peak_positions.append((int(round(mx0)), int(round(my0))))
            peak_intensities.append(mv0)
            peak_millers.append((H, K, L))  # or label the sign if you like

        # Always store sign=1
        if not (math.isnan(mx1) or math.isnan(my1)):
            peak_positions.append((int(round(mx1)), int(round(my1))))
            peak_intensities.append(mv1)
            peak_millers.append((H, K, L))

    # 5) Update the displayed diffraction image in the GUI
    image_display.set_data(updated_image)

    # Update color scale
    current_vmax = vmax_var.get()
    image_display.set_clim(vmax=current_vmax)

    # Remove old colorbar if it exists
    if colorbar_handle:
        colorbar_handle.remove()

    # Remove old legend if it exists
    if legend_handle:
        legend_handle.remove()

    # Add a new legend
    legend_handle = ax.legend(['Simulated Diffraction Pattern'], loc='upper left', fontsize='small')

    # 6) Compute χ² between the simulated image and the reference background
    try:
        if np.max(updated_image) > 0:
            updated_image_norm = updated_image / np.max(updated_image)
        else:
            updated_image_norm = updated_image

        if np.max(current_background_image) > 0:
            reference_norm = current_background_image / np.max(current_background_image)
        else:
            reference_norm = current_background_image

        chi_squared_value = mean_squared_error(reference_norm, updated_image_norm) * updated_image.size
        chi_square_label.config(text=f"Chi-Squared: {chi_squared_value:.2e}")

    except Exception as e:
        chi_square_label.config(text=f"Chi-Squared: Error - {e}")

    # 7) Keep a copy of the updated image if needed
    updated_image_global = updated_image

    # Redraw the canvas
    canvas.draw_idle()


# --- Slider Creation ---

theta_initial_var, theta_initial_scale = create_slider('Theta Initial', 5.0, 20.0, 6.0, 0.01, parent=slider_frame, update_callback=update)
gamma_var, gamma_scale = create_slider('Gamma', -5, 5, gamma, 0.001, parent=slider_frame, update_callback=update)
Gamma_var, Gamma_scale = create_slider('Detector Rotation Gamma', -5, 5, Gamma, 0.001, parent=slider_frame, update_callback=update)
chi_var, chi_scale = create_slider('Chi', -1, 1, 0.0, 0.001, parent=slider_frame, update_callback=update)
zs_var, zs_scale = create_slider('Zs', 0.0, 5e-3, 0.0, 0.0001, parent=slider_frame, update_callback=update)
zb_var, zb_scale = create_slider('Zb', 0.0, 5e-3, 0.0, 0.0001, parent=slider_frame, update_callback=update)
debye_x_var, debye_x_scale = create_slider('Debye Qz', 0.0, 1, 0.0, 0.001, parent=slider_frame, update_callback=update)
debye_y_var, debye_y_scale = create_slider('Debye Qr', 0.0, 1, 0.0, 0.001, parent=slider_frame, update_callback=update)
corto_detector_var, corto_detector_scale = create_slider('CortoDetector', 0.0, 100e-3, Distance_CoR_to_Detector, 0.1e-3, parent=slider_frame, update_callback=update)
sigma_mosaic_var, sigma_mosaic_scale = create_slider('Sigma Mosaic (deg)', 0.0, 5.0, np.degrees(sigma_mosaic), 0.01, parent=slider_frame, update_callback=update)
gamma_mosaic_var, gamma_mosaic_scale = create_slider('Gamma Mosaic (deg)', 0.0, 5.0, np.degrees(gamma_mosaic), 0.01, parent=slider_frame, update_callback=update)
eta_var, eta_scale = create_slider('Eta (fraction)', 0.0, 1.0, eta, 0.001, parent=slider_frame, update_callback=update)
a_var, a_scale = create_slider('a (Å)', 3.5, 8, av, 0.01, parent=slider_frame, update_callback=update)
c_var, c_scale = create_slider('c (Å)', 20.0, 40.0, cv, 0.01, parent=slider_frame, update_callback=update)


# Call the new function with all required parameters
(beam_x_array, beam_y_array, beam_intensity_array,
 beta_array, kappa_array, mosaic_intensity_array,
 theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
    num_samples,
    divergence_sigma,
    bw_sigma,
    np.degrees(sigma_mosaic),
    np.degrees(gamma_mosaic),
    eta
)

# Define bounds based on the Scale widgets
bounds = [
    (theta_initial_scale.cget('from'), theta_initial_scale.cget('to')),    # Theta Initial
    (gamma_scale.cget('from'), gamma_scale.cget('to')),                    # Gamma
    (Gamma_scale.cget('from'), Gamma_scale.cget('to')),                    # Detector Rotation Gamma
    (chi_scale.cget('from'), chi_scale.cget('to')),                        # Chi
    (zs_scale.cget('from'), zs_scale.cget('to')),                          # Zs
    (zb_scale.cget('from'), zb_scale.cget('to'))                           # Zb
]

# --- GUI Update Functions ---

# Global variables for optimization tracking
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
        center=center,
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
        a_var=a_var,     # pass the actual slider DoubleVar
        c_var=c_var,     # pass the actual slider DoubleVar
        update_gui=update
    )
)


fit_button_geometry.pack(side=tk.LEFT, padx=5)



fit_button_geometry.pack(side=tk.LEFT, padx=5)

fit_button_mosaic = ttk.Button(
    root,
    text="Optimize Mosaic Only",
    command=lambda: run_optimization_mosaic(
        fit_button=fit_button_mosaic,
        progress_label=progress_label_mosaic,
        # fix geometry from slider values (not static):
        gamma_const=gamma_var.get(),
        Gamma_const=Gamma_var.get(),
        dist_const=corto_detector_var.get(),
        theta_i_const=theta_initial_var.get(),
        zs_const=zs_var.get(),
        zb_const=zb_var.get(),

        # data
        miller=miller,
        intensities=intensities,
        image_size=image_size,
        av=av,
        cv=cv,
        lambda_=lambda_,
        psi=psi,
        n2=n2,
        center=center,
        measured_peaks=measured_peaks
    )
)


fit_button_mosaic.pack(side=tk.LEFT, padx=5)

progress_label_geometry = ttk.Label(root, text="")
progress_label_geometry.pack(side=tk.LEFT, padx=5)

progress_label_mosaic = ttk.Label(root, text="")
progress_label_mosaic.pack(side=tk.LEFT, padx=5)


# --- Background Management Functions ---

# Function to toggle background visibility and adjust opacity of dynamic image
def toggle_background():
    global background_visible, background_display, image_display

    if background_visible:
        # Hide the background and set the dynamic image opacity to fully opaque
        background_display.set_visible(False)
        image_display.set_alpha(1.0)  # Set the opacity of the data to fully opaque
    else:
        # Show the current background and set the dynamic image opacity to semi-transparent
        background_display.set_visible(True)
        image_display.set_alpha(0.5)  # Set the opacity to allow the background to be visible

    # Toggle the state of background visibility
    background_visible = not background_visible

    # Refresh the canvas to apply the changes
    canvas.draw_idle()

# Function to switch between background images
def switch_background():
    global current_background_index, background_display

    # Update the current background index (toggle between 0 and 1)
    current_background_index = (current_background_index + 1) % len(background_images)
    
    # Update the background image
    background_display.set_data(background_images[current_background_index])

    # Refresh the canvas to show the new background
    canvas.draw_idle()
    
# Function to reset all slider values to their initial conditions
def reset_to_defaults():
    """
    Reset all sliders to their default values.
    """
    gamma_var.set(0.0)
    Gamma_var.set(0.0)
    chi_var.set(0.0)
    zs_var.set(0.0)
    zb_var.set(0.0)
    theta_initial_var.set(5.0)
    debye_x_var.set(0.0)
    debye_y_var.set(0.0)
    corto_detector_var.set(20e-3)

    # Update the figure with default values
    update()


# Create a frame below the figure for the vmax slider
vmax_frame = ttk.Frame(fig_frame)
vmax_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Create a label for the vmax slider
vmax_label = ttk.Label(vmax_frame, text="Max Value")
vmax_label.pack(side=tk.LEFT, padx=5)

# Variable for the vmax slider
vmax_var = tk.DoubleVar(value=1e7)  # Initial max value

def vmax_slider_command(val):
    # Round or just float the val
    v = float(val)
    vmax_var.set(v)
    update()  # Recalculate the plot with new vmax
# Create a frame for parameter buttons (Fit, Save, Load)
param_button_frame = ttk.LabelFrame(slider_frame, text="Parameters & Actions", padding=5)
param_button_frame.pack(pady=5, fill=tk.X)


# Create a Scale widget for vmax
vmax_slider = ttk.Scale(vmax_frame, from_=1e4, to=1e8, orient=tk.HORIZONTAL, variable=vmax_var, command=vmax_slider_command)
vmax_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)


# Frame for reset & error display
bottom_frame = ttk.Frame(slider_frame, padding=5)
bottom_frame.pack(pady=5, fill=tk.X)

reset_button = ttk.Button(bottom_frame, text="Reset to Defaults", command=reset_to_defaults)
reset_button.pack(pady=5, fill=tk.X)

progress_label = ttk.Label(bottom_frame, text="", wraplength=300, justify=tk.LEFT)
progress_label.pack(pady=5, fill=tk.BOTH, expand=True)

from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Declare global variables for the legend and colorbar handles
legend_handle = None
colorbar_handle = None

    
    

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
            av=av,
            cv=cv,
            lambda_=lambda_,
            psi=psi,
            n2=n2,
            center=center,
            num_samples=num_samples,
            divergence_sigma=divergence_sigma,
            bw_sigma=bw_sigma,
            sigma_mosaic_var=sigma_mosaic_var,
            gamma_mosaic_var=gamma_mosaic_var,
            eta_var=eta_var
        ),
        center,
        {
            'pixel_size': 100e-6,
            'poni1': (3000 - center[0]) * 100e-6,
            'poni2': (3000 - center[1]) * 100e-6,
            'dist': corto_detector_var.get(),
            'rot1': np.deg2rad(Gamma_var.get()),
            'rot2': np.deg2rad(gamma_var.get()),
            'rot3': 0.0,
            'wavelength': ai_initial._wavelength
        }
    )
)

azimuthal_button.pack(side=tk.LEFT, padx=5)

# Create a label to show the progress or final message
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
        a_var,  # new
        c_var   # new
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
                a_var,  # <-- new
                c_var   # <-- new
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

# y limit for the plot up to 3000-c[0]
ax.set_ylim(3000- center[0], 0)

# Now the sliders and figure frames:
fig_frame = ttk.Frame(root)
fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control_frame = ttk.Frame(root, padding=10, width=200)
control_frame.pack(side=tk.RIGHT, fill=tk.Y)
control_frame.pack_propagate(False)

def on_click(event):
    global peak_positions, peak_millers, peak_intensities
    
    x_click = event.xdata
    y_click = event.ydata
    if x_click is None or y_click is None:
        return  # clicked outside the axes

    x_click = int(round(x_click))
    y_click = int(round(y_click))

    # 1) Find nearest single-peak pixel among "peak_positions"
    distances = []
    for (px, py) in peak_positions:
        dx = x_click - px
        dy = y_click - py
        distances.append(np.sqrt(dx*dx + dy*dy))
    distances = np.array(distances)
    min_index = int(np.argmin(distances))

    # 2) Retrieve data for that reflection
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




# Connect the click event to the figure
fig.canvas.mpl_connect('button_press_event', on_click)

# --- Start the Tkinter Main Loop ---
# Call update function when sliders are adjusted
update()
root.mainloop()