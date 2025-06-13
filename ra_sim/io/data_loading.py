"""Functions for persisting and retrieving configuration data."""

import numpy as np
import os

def load_parameters(
    path,
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
    center_x_var,      # <--- ADDED
    center_y_var       # <--- ADDED
):
    """
    Load slider parameters from a .npy file (dictionary). If the file does not exist,
    return a message. This now includes center_x and center_y.
    """
    if os.path.exists(path):
        params = np.load(path, allow_pickle=True).item()
        
        # Set all the old parameters
        theta_initial_var.set(params.get('theta_initial', theta_initial_var.get()))
        gamma_var.set(params.get('gamma', gamma_var.get()))
        Gamma_var.set(params.get('Gamma', Gamma_var.get()))
        chi_var.set(params.get('chi', chi_var.get()))
        zs_var.set(params.get('zs', zs_var.get()))
        zb_var.set(params.get('zb', zb_var.get()))
        debye_x_var.set(params.get('debye_x', debye_x_var.get()))
        debye_y_var.set(params.get('debye_y', debye_y_var.get()))
        corto_detector_var.set(params.get('corto_detector', corto_detector_var.get()))
        sigma_mosaic_var.set(params.get('sigma_mosaic', sigma_mosaic_var.get()))
        gamma_mosaic_var.set(params.get('gamma_mosaic', gamma_mosaic_var.get()))
        eta_var.set(params.get('eta', eta_var.get()))
        a_var.set(params.get('a', a_var.get()))
        c_var.set(params.get('c', c_var.get()))
        
        # Set the new beam center parameters
        center_x_var.set(params.get('center_x', center_x_var.get()))
        center_y_var.set(params.get('center_y', center_y_var.get()))

        return "Parameters loaded from parameters.npy"
    else:
        return "No parameters.npy file found to load."

def save_all_parameters(
    filepath,
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
    center_x_var,    # <--- ADDED
    center_y_var     # <--- ADDED
):
    """
    Save all slider parameters into a .npy file as a dictionary. This now
    includes beam center (center_x, center_y).
    """
    parameters = {
        'theta_initial':  theta_initial_var.get(),
        'gamma':          gamma_var.get(),
        'Gamma':          Gamma_var.get(),
        'chi':            chi_var.get(),
        'zs':             zs_var.get(),
        'zb':             zb_var.get(),
        'debye_x':        debye_x_var.get(),
        'debye_y':        debye_y_var.get(),
        'corto_detector': corto_detector_var.get(),
        'sigma_mosaic':   sigma_mosaic_var.get(),
        'gamma_mosaic':   gamma_mosaic_var.get(),
        'eta':            eta_var.get(),
        'a':              a_var.get(),
        'c':              c_var.get(),
        # Beam center
        'center_x':       center_x_var.get(),
        'center_y':       center_y_var.get()
    }
    np.save(filepath, parameters)
    print(f"Parameters saved successfully to {filepath}")


def load_background_image(file_path):
    """
    Example function to load an ASCII file (with 6 header lines, 3000x3000 pixel data).
    If needed, adapt for your real background file structure or use other formats.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        pixel_lines = lines[6:]
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        image = flattened_pixels.reshape((3000, 3000))
    return image


def load_and_format_reference_profiles(input_filename):
    """
    Loads a .npy file produced by some preprocessing. Returns a dict:
      {
        region_name: {
          "Radial (2θ)": np.array(...),
          "Radial Intensity": np.array(...),
          "Azimuthal φ": np.array(...),
          "Azimuthal Intensity": np.array(...),
          "FittedParams": {...}
        },
        ...
      }
    """
    raw_data = np.load(input_filename, allow_pickle=True).item()
    reference_profiles = {}

    # "Regions" dictionary from raw_data
    regions_dict = raw_data.get("Regions", {})
    for region_name, region_info in regions_dict.items():
        radial_dict = region_info.get("Radial", {})
        azimuthal_dict = region_info.get("Azimuthal", {})
        fitted_params = region_info.get("FittedParams", {})

        radial_2theta = np.array(radial_dict.get("2θ", []), dtype=float)
        radial_intensity = np.array(radial_dict.get("Intensity", []), dtype=float)
        azimuthal_phi = np.array(azimuthal_dict.get("φ", []), dtype=float)
        azimuthal_intensity = np.array(azimuthal_dict.get("Intensity", []), dtype=float)

        reference_profiles[region_name] = {
            "Radial (2θ)": radial_2theta,
            "Radial Intensity": radial_intensity,
            "Azimuthal φ": azimuthal_phi,
            "Azimuthal Intensity": azimuthal_intensity,
            "FittedParams": fitted_params
        }

    return reference_profiles
