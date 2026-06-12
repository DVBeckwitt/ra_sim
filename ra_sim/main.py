import json
import pyFAI
import fabio
from matplotlib.pyplot import subplots
from pyFAI.gui import jupyter
from pyFAI.gui.jupyter.calib import Calibration
from numpy import sin, cos, sqrt, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import time
import scipy.optimize  # Import optimization module
import threading
import traceback
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import itertools
import matplotlib.image
import pandas as pd
import matplotlib.image as mpimg
import IPython.display as display
from IPython.display import display
import pickle
from numba import njit, int64, float64, prange
from scipy.optimize import differential_evolution  # Import here
from scipy.optimize import least_squares
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator   
import os

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import matplotlib.image
import pandas as pd
import matplotlib.image as mpimg
import IPython.display as display
from IPython.display import display
import pickle 

import numpy as np
import spglib

import pickle 
import pandas as pd
import sympy as sp
import numpy as np
import itertools

from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.utils.tools import setup_azimuthal_integrator
from ra_sim.io.data_loading  import load_and_format_reference_profiles
from ra_sim.StructureFactor.AtomicCoordinates  import get_Atomic_Coordinates
from ra_sim.StructureFactor.StructureFactor  import calculate_structure_factor
from ra_sim.utils.tools import miller_generator

from ra_sim.simulation.mosaic_profiles import generate_random_profiles

from ra_sim.simulation.diffraction import process_peaks_parallel, compute_bragg_peak_position

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

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

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


with open(file_path_1, 'r') as file:
    lines = file.readlines()
    pixel_lines = lines[6:]
    pixels = [list(map(int, line.split())) for line in pixel_lines]
    flattened_pixels = np.array(pixels).flatten()
    background_image_1 = flattened_pixels.reshape((3000, 3000))
    background_image_1 = np.rot90(background_image_1, k=3)  # Rotate the first background image by -90 deg

with open(file_path_2, 'r') as file:
    lines = file.readlines()
    pixel_lines = lines[6:]
    pixels = [list(map(int, line.split())) for line in pixel_lines]
    flattened_pixels = np.array(pixels).flatten()
    background_image_2 = flattened_pixels.reshape((3000, 3000))
    background_image_2 = np.rot90(background_image_2, k=3)  # Rotate the second background image by -90 deg

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

# Now proceed with other code that relies on fig_frame being available.


# Add sliders and controls in root
slider_frame = ttk.Frame(root, padding=10)
slider_frame.pack(side=tk.LEFT, fill=tk.Y)

theta_label = ttk.Label(slider_frame, text="Theta Initial")
theta_label.pack(pady=5)
theta_var = tk.DoubleVar(value=5.0)
theta_scale = ttk.Scale(slider_frame, from_=0, to=20, variable=theta_var, orient=tk.HORIZONTAL)
theta_scale.pack(fill=tk.X, padx=5)


# Display the background image first
background_display = ax.imshow(current_background_image, cmap='turbo', vmin=0, vmax=1e3, zorder=0)  # Set zorder=0 to keep it at the bottom

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

image = process_peaks_parallel(
    miller, intensities, image_size, av, cv, lambda_, np.zeros((image_size, image_size)),
    Distance_CoR_to_Detector, gamma, Gamma, chi, psi, zs, zb, n2,
    beam_x_array, beam_y_array, beam_intensity_array,
    beta_array, kappa_array, mosaic_intensity_array,
    theta_array, phi_array, divergence_intensity_array,
    debye_x, debye_y, center, 6.6, 6.7, 0.1,
    unit_x, n_detector
)


image_display = ax.imshow(image, cmap='turbo', vmin=0, vmax=1e5, alpha=0.5, zorder=1)  # Set alpha to make it semi-transparent

# Set the y-axis limits to only display pixels above y = 1600
#ax.set_ylim(1600, 0)  # Assuming y increases downward
#ax.set_xlim(0, 3000)  # x-limits remain the same
# Adjust colorbar to be smaller
colorbar = plt.colorbar(image_display, ax=ax, label='Intensity', shrink=0.6, pad=0.02)  # Use shrink to make it smaller, pad to reduce space

# Adjust the plot layout to make the image larger
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Reduce margins to make the image fill more of the space

plt.title('Simulated Diffraction Pattern')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')

# --- Slider Creation Function ---
def create_slider(label, min_val, max_val, initial_val, step_size, parent=None):
    if parent is None:
        parent = slider_frame  # default parent if none provided
    
    frame = ttk.Frame(parent)
    frame.pack(pady=5, fill=tk.X)
    label_widget = ttk.Label(frame, text=label, font=("Helvetica", 10))
    label_widget.pack(anchor=tk.W)
    slider_var = tk.DoubleVar(value=initial_val)

    def slider_command(val):
        precise_value = round(float(val) / step_size) * step_size
        slider_var.set(precise_value)
        update()

    slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=slider_var, command=slider_command)
    slider.pack(fill=tk.X, padx=5)

    entry = ttk.Entry(frame, textvariable=slider_var, width=10)
    entry.pack(side=tk.RIGHT, padx=5)

    def on_key(event):
        if event.keysym == 'Left':
            new_val = slider_var.get() - step_size + 1.0
            new_val = max(new_val, min_val)
            slider_var.set(round(new_val / step_size) * step_size)
            update()
        elif event.keysym == 'Right':
            new_val = slider_var.get() + step_size - 1.0
            new_val = min(new_val, max_val)
            slider_var.set(round(new_val / step_size) * step_size)
            update()

    def on_click(event):
        slider.focus_set()
        slider.bind('<KeyPress>', on_key)

    slider.bind('<Button-1>', on_click)

    return slider_var, slider

# --- Slider Creation ---

# Recreate all sliders with the parent parameter:
theta_initial_var, theta_initial_scale = create_slider('Theta Initial', 5.0, 20.0, 6.0, 0.01, parent=slider_frame)
gamma_var, gamma_scale = create_slider('Gamma', -5, 5, gamma, 0.001, parent=slider_frame)
Gamma_var, Gamma_scale = create_slider('Detector Rotation Gamma', -5, 5, Gamma, 0.001, parent=slider_frame)
chi_var, chi_scale = create_slider('Chi', -1, 1, 0.0, 0.001, parent=slider_frame)
zs_var, zs_scale = create_slider('Zs', 0.0, 5e-3, 0.0, 0.0001, parent=slider_frame)
zb_var, zb_scale = create_slider('Zb', 0.0, 5e-3, 0.0, 0.0001, parent=slider_frame)
debye_x_var, debye_x_scale = create_slider('Debye Qz', 0.0, 1, 0.0, 0.001, parent=slider_frame)
debye_y_var, debye_y_scale = create_slider('Debye Qr', 0.0, 1, 0.0, 0.001, parent=slider_frame)
corto_detector_var, corto_detector_scale = create_slider('CortoDetector', 0.0, 100e-3, Distance_CoR_to_Detector, 0.001, parent=slider_frame)
sigma_mosaic_var, sigma_mosaic_scale = create_slider('Sigma Mosaic (deg)', 0.0, 5.0, np.degrees(sigma_mosaic), 0.01, parent=slider_frame)
gamma_mosaic_var, gamma_mosaic_scale = create_slider('Gamma Mosaic (deg)', 0.0, 5.0, np.degrees(gamma_mosaic), 0.01, parent=slider_frame)
eta_var, eta_scale = create_slider('Eta (fraction)', 0.0, 1.0, eta, 0.001, parent=slider_frame)

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

# image size

# --- Define Optimization Bounds ---

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
def optimization_complete(result):
    """
    Handles the completion of the optimization process.
    Updates the GUI sliders with the optimized parameters.
    """
    fit_button.config(state=tk.NORMAL)
    if result.success:
        optimal_params = result.x  # [gamma, Gamma, chi, zs, zb]
        gamma_var.set(optimal_params[0])
        Gamma_var.set(optimal_params[1])
        chi_var.set(optimal_params[2])
        zs_var.set(optimal_params[3])
        zb_var.set(optimal_params[4])
        # Do NOT set the fixed parameters
        update()

        # Calculate the cost from the residuals
        cost = np.sum(result.fun ** 2)

        progress_label.config(text=f"Optimization complete.\n"
                                   f"Best Loss: {cost:.2e}\n"
                                   f"Optimal Parameters:\n"
                                   f"Gamma: {optimal_params[0]:.4f}\n"
                                   f"Detector Rotation Gamma: {optimal_params[1]:.4f}\n"
                                   f"Chi: {optimal_params[2]:.4f}\n"
                                   f"Zs: {optimal_params[3]:.6f}\n"
                                   f"Zb: {optimal_params[4]:.6f}")
    else:
        progress_label.config(text="Optimization failed.")


def process_data(ai, data, regions_of_interest):
    # Perform 2D integration
    res2 = ai.integrate2d(
        data,
        npt_rad=3000,
        npt_azim=1000,
        unit="2th_deg",
    )

    # Extract arrays
    intensity = res2.intensity  # shape: (npt_azim, npt_rad)
    radial = res2.radial
    azimuthal = res2.azimuthal

    # Adjust azimuthal values
    azimuthal_adjusted = np.where(azimuthal < 0, azimuthal + 180, azimuthal - 180)

    # Sort by adjusted azimuthal angle
    sort_indices = np.argsort(azimuthal_adjusted)
    azimuthal_adjusted_sorted = azimuthal_adjusted[sort_indices]
    intensity_sorted = intensity[sort_indices, :]

    # Restrict azimuthal range to -90 to 90
    mask = (azimuthal_adjusted_sorted > -90) & (azimuthal_adjusted_sorted < 90)
    azimuthal_adjusted_sorted = azimuthal_adjusted_sorted[mask]
    intensity_sorted = intensity_sorted[mask, :]

    peak_data = []

    # Iterate over each region of interest
    for region in regions_of_interest:
        theta_min, theta_max = region['theta_min'], region['theta_max']
        phi_min, phi_max = region['phi_min'], region['phi_max']

        # Filter azimuthal
        mask_az = (azimuthal_adjusted_sorted >= phi_min) & (azimuthal_adjusted_sorted <= phi_max)
        intensity_filtered_azimuth = intensity_sorted[mask_az, :]
        azimuthal_filtered = azimuthal_adjusted_sorted[mask_az]

        # Filter radial
        mask_rad = (radial >= theta_min) & (radial <= theta_max)
        intensity_filtered = intensity_filtered_azimuth[:, mask_rad]
        radial_filtered = radial[mask_rad]

        if intensity_filtered.size == 0:
            continue

        # Sum along azimuthal to get I(2θ)
        intensity_1d = np.sum(intensity_filtered, axis=0)

        # Sum along radial to get I(φ)
        intensity_1d_phi = np.sum(intensity_filtered, axis=1)

        peak_data.append({
            'Region': region['name'],
            'Radial (2θ)': radial_filtered,
            'Intensity': intensity_1d,
            'Azimuthal Angle (φ)': azimuthal_filtered,
            'Azimuthal Intensity': intensity_1d_phi
        })

    return peak_data

def simulate_diffraction(theta_initial, gamma, Gamma, chi, zs, zb, debye_x_value, debye_y_value, corto_detector_value):
    # Get updated parameters for mosaic and eta
    current_sigma_mosaic = np.radians(sigma_mosaic_var.get())
    current_gamma_mosaic = np.radians(gamma_mosaic_var.get())
    current_eta = eta_var.get()

    # Regenerate the random profiles with updated parameters
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

    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])

    simulated_image = process_peaks_parallel(
        miller, intensities, image_size, av, cv, lambda_, np.zeros((image_size, image_size)),
        corto_detector_value, gamma, Gamma, chi, psi, zs, zb, n2,
        beam_x_array, beam_y_array, beam_intensity_array,
        beta_array, kappa_array, mosaic_intensity_array,
        theta_array, phi_array, divergence_intensity_array,
        debye_x_value, debye_y_value, center,
        theta_initial, theta_initial + 0.1, 0.1,
        unit_x, n_detector
    )
    # Rotate if needed
    #simulated_image = np.rot90(simulated_image, k=3)
    return simulated_image

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
from bayes_opt import BayesianOptimization


def compute_cost(integrated_data, reference_profiles):
    # We'll accumulate squared differences for all regions and then take the mean
    squared_errors = []
    for region_data in integrated_data:
        region_name = region_data['Region']
        
        # Extract simulated profiles
        sim_theta = region_data['Radial (2θ)']
        sim_intensity = region_data['Intensity']
        
        # Extract reference profiles
        ref_theta = reference_profiles[region_name]['Radial (2θ)']
        ref_intensity = reference_profiles[region_name]['Intensity']
        
        # Ensure both arrays are aligned. If they are not the same length or grid, 
        # you may need interpolation. For simplicity, let's assume they match in length and order.
        if len(sim_theta) == len(ref_theta):
            # Compute squared error for this region
            diff = sim_intensity - ref_intensity
            squared_errors.extend(diff**2)
        else:
            # If lengths differ, consider interpolation or a different approach
            # For now, just skip or handle gracefully:
            continue
    
    if len(squared_errors) == 0:
        # No data to compare, return a large cost
        return 1e9
    
    # Mean Squared Error
    mse = np.mean(squared_errors)
    return mse


def objective_function_bayesian(_, __, chi_val, zs_val, zb_val, eta_val, sigma_mosaic_deg_val, gamma_mosaic_deg_val):
    # Set the sliders to these trial values
    eta_var.set(eta_val)
    sigma_mosaic_var.set(sigma_mosaic_deg_val)
    gamma_mosaic_var.set(gamma_mosaic_deg_val)
    chi_var.set(chi_val)
    zs_var.set(zs_val)
    zb_var.set(zb_val)

    # Now run the simulation with these values
    gamma_fixed = gamma_var.get()
    Gamma_fixed = Gamma_var.get()
    theta_initial_val = theta_initial_var.get()
    debye_x_val = debye_x_var.get()
    debye_y_val = debye_y_var.get()
    corto_detector_val = corto_detector_var.get()

    try:
        simulated_image = simulate_diffraction(
            theta_initial_val, gamma_fixed, Gamma_fixed, chi_val, zs_val, zb_val,
            debye_x_val, debye_y_val, corto_detector_val
        )

        # rotate the image ccw by 90 degrees
        simulated_image = np.rot90(simulated_image, k=1)
    
        # Create a new AzimuthalIntegrator (ai) with the current parameters
        pixel_size = 100e-6  # 100 micrometers per pixel
        center_x = center[0]
        center_y = center[1]
        poni1 = (3000 - center_x) * pixel_size
        poni2 = (3000 - center_y) * pixel_size
        
        # Convert rotations from degrees to radians
        rot1 = np.deg2rad(Gamma_fixed)
        rot2 = np.deg2rad(gamma_fixed)
        rot3 = 0.0  # Assuming no rotation around the sample's z-axis

        # Use the wavelength from the initial integrator
        wavelength = ai_initial._wavelength
        
        ai = AzimuthalIntegrator(
            dist=corto_detector_val,
            poni1=poni1,
            poni2=poni2,
            pixel1=pixel_size,
            pixel2=pixel_size,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            wavelength=wavelength
        )
        # Define Regions of Interest
        regions_of_interest = [
            {'theta_min': 7, 'theta_max': 12, 'phi_min': -20, 'phi_max': 20, 'name': '003'},
            {'theta_min': 16, 'theta_max': 19, 'phi_min': -10, 'phi_max': 10, 'name': '006'},
            #{'theta_min': 27, 'theta_max': 30, 'phi_min': -10, 'phi_max': 10, 'name': '009'},
            #{'theta_min': 37, 'theta_max': 40, 'phi_min': -10, 'phi_max': 10, 'name': '012'},
            #{'theta_min': 47, 'theta_max': 51, 'phi_min': -10, 'phi_max': 10, 'name': '015'},
        ]

        # Ensure that 'regions_of_interest' is defined
        # Example: regions_of_interest = [{'name': 'peak1', 'theta_min': 10, 'theta_max': 12, 'phi_min': -20, 'phi_max': 20}, ...]
        # Provide your actual regions_of_interest dict here.
        
        integrated_data = process_data(ai, simulated_image, regions_of_interest)
        cost = compute_cost(integrated_data, reference_profiles)

        # Return negative cost for maximization in Bayesian Optimization
        return -cost
    except:
        return 1e9


    
from bayes_opt import BayesianOptimization
def run_optimization():
    try:
        fit_button.config(state=tk.DISABLED)
        progress_label.config(text="Optimization in progress...")

        # Define parameter bounds
        pbounds = {
            'chi': (chi_scale.cget('from'), chi_scale.cget('to')),
            'zs': (zs_scale.cget('from'), zs_scale.cget('to')),
            'zb': (zb_scale.cget('from'), zb_scale.cget('to')),
            'eta': (0.0, 1.0),
            'sigma_mosaic_deg': (0.0, 5.0),
            'gamma_mosaic_deg': (0.0, 5.0)
        }

        # Initial parameters
        initial_params = {
            'chi': chi_var.get(),
            'zs': zs_var.get(),
            'zb': zb_var.get(),
            'eta': eta_var.get(),
            'sigma_mosaic_deg': sigma_mosaic_var.get(),
            'gamma_mosaic_deg': gamma_mosaic_var.get()
        }

        # Create optimizer here
        optimizer = BayesianOptimization(
            f=lambda chi, zs, zb, eta, sigma_mosaic_deg, gamma_mosaic_deg:
                objective_function_bayesian(None, None, chi, zs, zb, eta, sigma_mosaic_deg, gamma_mosaic_deg),
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        # Now we can safely probe and maximize
        optimizer.probe(params=initial_params, lazy=True)
        optimizer.maximize(init_points=5, n_iter=50)

        # Proceed after optimization
        best_params = optimizer.max['params']
        chi_var.set(best_params['chi'])
        zs_var.set(best_params['zs'])
        zb_var.set(best_params['zb'])
        update()

        progress_label.config(text=f"Optimization complete.\n"
                                   f"Best Loss: {-optimizer.max['target']:.2e}\n"
                                   f"Optimal Parameters:\n"
                                   f"Chi: {best_params['chi']:.4f}\n"
                                   f"Zs: {best_params['zs']:.6f}\n"
                                   f"Zb: {best_params['zb']:.6f}")

    except Exception as e:
        optimization_failed(e)
    finally:
        fit_button.config(state=tk.NORMAL)


# Function to handle when optimization fails
def optimization_failed(exception):
    fit_button.config(state=tk.NORMAL)
    traceback_str = ''.join(traceback.format_exception(None, exception, exception.__traceback__))
    progress_label.config(text=f"Optimization failed with exception:\n{traceback_str}")

def perform_fit():
    optimization_thread = threading.Thread(target=run_optimization, daemon=True)
    optimization_thread.start()


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

optimizer = BayesianOptimization(
    f=lambda chi, zs, zb, eta, sigma_mosaic_deg, gamma_mosaic_deg: 
        objective_function_bayesian(None, None, chi, zs, zb, eta, sigma_mosaic_deg, gamma_mosaic_deg),
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
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

def save_parameters():
    print("Saving parameters to parameters.npy...")
    params = {
        'theta_initial': theta_initial_var.get(),
        'gamma': gamma_var.get(),
        'Gamma': Gamma_var.get(),
        'chi': chi_var.get(),
        'zs': zs_var.get(),
        'zb': zb_var.get(),
        'debye_x': debye_x_var.get(),
        'debye_y': debye_y_var.get(),
        'corto_detector': corto_detector_var.get(),
        'sigma_mosaic': sigma_mosaic_var.get(),
        'gamma_mosaic': gamma_mosaic_var.get(),
        'eta': eta_var.get(),
    }
    np.save(r"C:\users\kenpo\OneDrive\Research\Uniaxially Twisted Project\UniTwist2D\subroutines\parameters.npy", params)
    progress_label.config(text="Parameters saved to parameters.npy")

# --- Add Buttons for Background Management ---

# Create a Scale widget for vmax
vmax_slider = ttk.Scale(vmax_frame, from_=1e4, to=1e8, orient=tk.HORIZONTAL, variable=vmax_var, command=vmax_slider_command)
vmax_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

# Create the Fit button
fit_button = ttk.Button(slider_frame, text="Perform Fit", command=run_optimization)
fit_button.pack(pady=10)

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

from skimage.metrics import mean_squared_error

def update():
    global inset_ax, legend_handle, colorbar_handle
    global bragg_positions, bragg_miller_indices, bragg_pixel_positions
    global updated_image_global

    # Get current slider values for optimized parameters
    gamma_updated = round(gamma_var.get(), 3)
    Gamma_updated = round(Gamma_var.get(), 3)
    chi_updated = round(chi_var.get(), 3)
    zs_updated = round(zs_var.get(), 6)
    zb_updated = round(zb_var.get(), 6)

    # Fixed parameters
    theta_initial_updated = round(theta_initial_var.get(), 2)
    debye_x_updated = round(debye_x_var.get(), 3)
    debye_y_updated = round(debye_y_var.get(), 3)
    corto_detector_updated = round(corto_detector_var.get(), 2)

    # current sigma_mosaic and gamma_mosaic
    current_sigma_mosaic = (sigma_mosaic_var.get())
    current_gamma_mosaic = (gamma_mosaic_var.get())
    current_eta = eta_var.get()
    # Regenerate the random profiles with updated parameters
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

    # Define unit_x and n_detector
    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])

    # Proceed with updating the image
    updated_image = process_peaks_parallel(
        miller, intensities, image_size, av, cv, lambda_, np.zeros((image_size, image_size)),
        Distance_CoR_to_Detector, gamma_updated, Gamma_updated, chi_updated, psi,
        zs_updated, zb_updated, n2,
        beam_x_array, beam_y_array, beam_intensity_array,
        beta_array, kappa_array, mosaic_intensity_array,
        theta_array, phi_array, divergence_intensity_array,
        debye_x_updated, debye_y_updated, center,
        theta_initial_updated, theta_initial_updated + 0.1, 0.1,
        unit_x, n_detector
    )
    
    # Update the displayed image
    image_display.set_data(updated_image)
    
    ax.set_ylim(1600,0)
        # Set the new vmax from vmax_var
    current_vmax = vmax_var.get()
    image_display.set_clim(vmax=current_vmax)
    # Remove the previous colorbar if it exists
    if colorbar_handle:
        colorbar_handle.remove()
    
    # Remove previous legend if it exists
    if legend_handle:
        legend_handle.remove()
    
    # Add a new legend (replace this with your actual labels)
    legend_handle = ax.legend(['Simulated Diffraction Pattern'], loc='upper left', fontsize='small')
    
    # Compute χ² between the simulated image and the reference image
    try:
        # Normalize the images for better comparison (prevent division by zero)
        if np.max(updated_image) > 0:
            updated_image_normalized = updated_image / np.max(updated_image)
        else:
            updated_image_normalized = updated_image
            
        if np.max(current_background_image) > 0:
            reference_image_normalized = current_background_image / np.max(current_background_image)
        else:
            reference_image_normalized = current_background_image

        # Crop the images to only include pixels above y = 1600
        updated_image_cropped = updated_image_normalized[0:1600, :]
        reference_image_cropped = reference_image_normalized[0:1600, :]
        
        # Compute the chi-squared value
        chi_squared_value = mean_squared_error(reference_image_normalized, updated_image_normalized) * updated_image.size

        # Update the chi-squared label in the GUI
        chi_square_label.config(text=f"Chi-Squared: {chi_squared_value:.2e}")

    except Exception as e:
        # Handle any exceptions in chi-squared calculation
        chi_square_label.config(text=f"Chi-Squared: Error - {e}")
    
    # Now, compute Bragg peak positions
    bragg_positions = []
    bragg_miller_indices = []

    # Precompute necessary parameters (same as in the process_peaks_parallel function)
    gamma_rad = np.radians(gamma_updated)
    Gamma_rad = np.radians(Gamma_updated)
    chi_rad = np.radians(chi_updated)
    psi_rad = np.radians(psi)

    k = 2 * np.pi / lambda_

    # Precompute detector geometry
    R_x_detector = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(gamma_rad), np.sin(gamma_rad)],
        [0.0, -np.sin(gamma_rad), np.cos(gamma_rad)]
    ])
    R_z_detector = np.array([
        [np.cos(Gamma_rad), np.sin(Gamma_rad), 0.0],
        [-np.sin(Gamma_rad), np.cos(Gamma_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    n_detector = np.array([0.0, 1.0, 0.0])
    n_det_rot = R_z_detector @ (R_x_detector @ n_detector)
    n_det_rot /= np.linalg.norm(n_det_rot)
    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0])
    unit_x = np.array([1.0, 0.0, 0.0])
    e1_det = unit_x - np.dot(unit_x, n_det_rot) * n_det_rot
    e1_det /= np.linalg.norm(e1_det)
    e2_det = -np.cross(n_det_rot, e1_det)
    e2_det /= np.linalg.norm(e2_det)

    # Precompute sample rotation matrices
    R_y = np.array([
        [np.cos(chi_rad), 0.0, np.sin(chi_rad)],
        [0.0, 1.0, 0.0],
        [-np.sin(chi_rad), 0.0, np.cos(chi_rad)]
    ])
    R_z = np.array([
        [np.cos(psi_rad), np.sin(psi_rad), 0.0],
        [-np.sin(psi_rad), np.cos(psi_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    R_z_R_y = R_z @ R_y
    n1 = np.array([0.0, 0.0, 1.0])
    R_ZY_n = R_z_R_y @ n1
    R_ZY_n /= np.linalg.norm(R_ZY_n)

    # Precompute sample position
    P0 = np.array([0.0, 0.0, -zs_updated])

    bragg_pixel_positions = []

    for i in range(len(miller)):
        H, K, L = miller[i]
        x_det, y_det = compute_bragg_peak_position(
            H, K, L, av, cv, lambda_,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs_updated, zb_updated, n2, debye_x_updated, debye_y_updated, center,
            theta_initial_updated, k, R_x_detector, R_z_detector, n_det_rot, Detector_Pos, e1_det, e2_det,
            R_z_R_y, R_ZY_n, P0
        )
        if x_det is not None and y_det is not None:
            bragg_positions.append((x_det, y_det))
            bragg_miller_indices.append((H, K, L))
            # Compute pixel positions
            pixel_x_b = int(round(x_det / (100e-6) + center[1]))
            pixel_y_b = 3000 - int(round(y_det / (100e-6) + center[0]))
            bragg_pixel_positions.append((pixel_x_b, pixel_y_b))

    updated_image_global = updated_image
    image_display.set_data(updated_image)
    image_display.set_clim(vmax=vmax_var.get())
    # refresh canvas
    canvas.draw_idle()
def view_azimuthal_radial():
    import pyFAI
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    # Retrieve current parameter values (exactly as used in the objective_function_bayesian / simulate_diffraction routines)
    theta_initial_val = theta_initial_var.get()
    gamma_fixed = gamma_var.get()
    Gamma_fixed = Gamma_var.get()
    chi_val = chi_var.get()
    zs_val = zs_var.get()
    zb_val = zb_var.get()
    debye_x_val = debye_x_var.get()
    debye_y_val = debye_y_var.get()
    corto_detector_val = corto_detector_var.get()

    # Update the pseudo-Voigt parameters
    eta_val = eta_var.get()
    sigma_mosaic_deg_val = sigma_mosaic_var.get()
    gamma_mosaic_deg_val = gamma_mosaic_var.get()

    # Regenerate the random profiles with updated parameters
    (beam_x_array, beam_y_array, beam_intensity_array,
     beta_array, kappa_array, mosaic_intensity_array,
     theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
         num_samples,
         divergence_sigma,
         bw_sigma,
         sigma_mosaic_deg_val,
         gamma_mosaic_deg_val,
         eta_val
    )

    # Simulate the diffraction pattern using the same code as the fitting routine
    simulated_image = simulate_diffraction(
        theta_initial_val, gamma_fixed, Gamma_fixed, chi_val, zs_val, zb_val,
        debye_x_val, debye_y_val, corto_detector_val
    )
    # rotate the image ccw by 90 degrees
    simulated_image = np.rot90(simulated_image, k=1)
    
    # Set up the AzimuthalIntegrator using the current geometry parameters, identical to how the fitting routine does it
    pixel_size = 100e-6  # 100 micrometers per pixel
    center_x = center[0]
    center_y = center[1]
    poni1 = (3000 - center_x) * pixel_size
    poni2 = (3000 - center_y) * pixel_size

    rot1 = np.deg2rad(Gamma_fixed)
    rot2 = np.deg2rad(gamma_fixed)
    rot3 = 0.0  # Assuming no rotation around sample z-axis
    wavelength = ai_initial._wavelength

    ai = AzimuthalIntegrator(
        dist=corto_detector_val,
        poni1=poni1,
        poni2=poni2,
        pixel1=pixel_size,
        pixel2=pixel_size,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        wavelength=wavelength
    )

    # Perform the azimuthal integration with the same parameters
    res2 = ai.integrate2d(
        simulated_image,
        npt_rad=2000,
        npt_azim=1000,
        unit="2th_deg",
    )

    # Extract the intensity map, radial, and azimuthal arrays
    intensity = res2.intensity  # shape: (npt_azim, npt_rad)
    radial = res2.radial
    azimuthal = res2.azimuthal

    # Adjust azimuthal values to (-90, 90) range
    azimuthal_adjusted = np.where(azimuthal < 0, azimuthal + 180, azimuthal - 180)
    sort_indices = np.argsort(azimuthal_adjusted)
    azimuthal_adjusted_sorted = azimuthal_adjusted[sort_indices]
    intensity_sorted = intensity[sort_indices, :]

    mask = (azimuthal_adjusted_sorted > -90) & (azimuthal_adjusted_sorted < 90)
    azimuthal_adjusted_sorted = azimuthal_adjusted_sorted[mask]
    intensity_sorted = intensity_sorted[mask, :]

    extent = [radial.min(), radial.max(), azimuthal_adjusted_sorted.min(), azimuthal_adjusted_sorted.max()]

    # Create a new figure to display Azimuthal vs Radial map
    plt.figure(figsize=(10, 8))
    plt.imshow(
        intensity_sorted,
        extent=extent,
        cmap='turbo',
        vmin=0,
        vmax=5e6,
        aspect='auto',
        origin='lower'
    )
    plt.title('Azimuthal vs Radial View (Current Parameters)')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Azimuthal angle φ (degrees)')
    plt.colorbar(label='Intensity')
    plt.show()

    
def load_parameters():
    print("Attempting to load parameters...")
    path = r"C:\users\kenpo\OneDrive\Research\Uniaxially Twisted Project\UniTwist2D\subroutines\parameters.npy"
    if os.path.exists(path):
        params = np.load(path, allow_pickle=True).item()
        theta_initial_var.set(params['theta_initial'])
        gamma_var.set(params['gamma'])
        Gamma_var.set(params['Gamma'])
        chi_var.set(params['chi'])
        zs_var.set(params['zs'])
        zb_var.set(params['zb'])
        debye_x_var.set(params['debye_x'])
        debye_y_var.set(params['debye_y'])
        corto_detector_var.set(params['corto_detector'])
        sigma_mosaic_var.set(params['sigma_mosaic'])
        gamma_mosaic_var.set(params['gamma_mosaic'])
        eta_var.set(params['eta'])
        update()
        progress_label.config(text="Parameters loaded from parameters.npy")
    else:
        progress_label.config(text="No parameters.npy file found to load.")


# Change the slider creation function to smaller font and shorter length:
def create_slider(label, min_val, max_val, initial_val, step_size, length=150):
    frame = ttk.Frame(control_frame)
    frame.pack(pady=2, fill=tk.X)
    label_widget = ttk.Label(frame, text=label, font=("Helvetica", 8))
    label_widget.pack(anchor=tk.W)
    slider_var = tk.DoubleVar(value=initial_val)

    def slider_command(val):
        precise_value = round(float(val) / step_size) * step_size
        slider_var.set(precise_value)
        update()

    slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, variable=slider_var,
                       command=slider_command, length=length)
    slider.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

    entry = ttk.Entry(frame, textvariable=slider_var, width=8)
    entry.pack(side=tk.RIGHT, padx=2)

    return slider_var, slider
top_frame = ttk.Frame(root, padding=5)
top_frame.pack(side=tk.TOP, fill=tk.X)

# Buttons on top_frame
fit_button = ttk.Button(top_frame, text="Perform Fit", command=run_optimization)
fit_button.pack(side=tk.LEFT, padx=5)

toggle_button = ttk.Button(top_frame, text="Toggle Background", command=toggle_background)
toggle_button.pack(side=tk.LEFT, padx=5)

switch_button = ttk.Button(top_frame, text="Switch Background", command=switch_background)
switch_button.pack(side=tk.LEFT, padx=5)

reset_button = ttk.Button(top_frame, text="Reset to Defaults", command=reset_to_defaults)
reset_button.pack(side=tk.LEFT, padx=5)

azimuthal_button = ttk.Button(top_frame, text="Azimuthal vs Radial", command=view_azimuthal_radial)
azimuthal_button.pack(side=tk.LEFT, padx=5)

save_button = ttk.Button(top_frame, text="Save Params", command=save_parameters)
save_button.pack(side=tk.LEFT, padx=5)

load_button = ttk.Button(top_frame, text="Load Params", command=load_parameters)
load_button.pack(side=tk.LEFT, padx=5)

progress_label = ttk.Label(top_frame, text="", font=("Helvetica", 8))
progress_label.pack(side=tk.LEFT, padx=5)

chi_square_label = ttk.Label(top_frame, text="Chi-Squared: ", font=("Helvetica", 8))
chi_square_label.pack(side=tk.LEFT, padx=5)

info_label = ttk.Label(top_frame, text="", wraplength=200, font=("Helvetica", 8))
info_label.pack(side=tk.LEFT, padx=5)


# Now the sliders and figure frames:
fig_frame = ttk.Frame(root)
fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control_frame = ttk.Frame(root, padding=10, width=200)
control_frame.pack(side=tk.RIGHT, fill=tk.Y)
control_frame.pack_propagate(False)


def on_click(event):
    global bragg_positions, bragg_miller_indices, bragg_pixel_positions
    x_pixel = event.xdata
    y_pixel = event.ydata
    if x_pixel is None or y_pixel is None:
        return  # Clicked outside the axes

    x_pixel = int(round(x_pixel))
    y_pixel = int(round(y_pixel))

    # Compute distances to Bragg peaks in pixel units
    distances = []
    for (pixel_x_b, pixel_y_b) in bragg_pixel_positions:
        dx = x_pixel - pixel_x_b
        dy = y_pixel - pixel_y_b
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(dist)

    distances = np.array(distances)
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    H, K, L = bragg_miller_indices[min_index]
    # Display the Miller indices
    info_text = f"Nearest Bragg peak:\nH={H}, K={K}, L={L}\nDistance: {min_distance:.2f} pixels"
    print(info_text)
    # Update the info_label in the GUI
    info_label.config(text=info_text)



# Connect the click event to the figure
fig.canvas.mpl_connect('button_press_event', on_click)

# --- Start the Tkinter Main Loop ---
# Call update function when sliders are adjusted
update()
root.mainloop()