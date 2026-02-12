"""Functions for persisting and retrieving configuration data."""

import numpy as np
import os


def _normalize_optics_mode(value, fallback="fast"):
    """Return optics-mode label as ``'fast'`` or ``'exact'``."""

    if value is None:
        return fallback

    # Support legacy numeric storage and bool-like flags.
    if isinstance(value, (int, np.integer)):
        return "exact" if int(value) == 1 else "fast"
    if isinstance(value, (float, np.floating)):
        return "exact" if int(round(float(value))) == 1 else "fast"

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

    return fallback


def load_parameters(
    path,
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
    center_x_var,      # <--- ADDED
    center_y_var,      # <--- ADDED
    resolution_var=None,
    custom_samples_var=None,
    optics_mode_var=None,
):
    """
    Load slider parameters from a .npy file (dictionary). If the file does not exist,
    return a message. This now includes center_x and center_y.
    """
    if os.path.exists(path):
        params = np.load(path, allow_pickle=True).item()
        
        # Set all the old parameters
        theta_initial_var.set(params.get('theta_initial', theta_initial_var.get()))
        cor_angle_var.set(params.get('cor_angle', cor_angle_var.get()))
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
        if custom_samples_var is not None:
            stored_custom_count = params.get(
                'sampling_custom_count',
                params.get('sampling_count'),
            )
            if stored_custom_count is not None:
                try:
                    parsed_custom_count = int(round(float(stored_custom_count)))
                except (TypeError, ValueError):
                    parsed_custom_count = None
                if parsed_custom_count is not None and parsed_custom_count > 0:
                    custom_samples_var.set(str(parsed_custom_count))
        if resolution_var is not None:
            stored_resolution = params.get('sampling_resolution')
            if stored_resolution:
                resolution_var.set(stored_resolution)
        if optics_mode_var is not None:
            current_mode = _normalize_optics_mode(optics_mode_var.get(), fallback="fast")
            stored_mode = _normalize_optics_mode(
                params.get('optics_mode', current_mode),
                fallback=current_mode,
            )
            optics_mode_var.set(stored_mode)

        return "Parameters loaded from parameters.npy"
    else:
        return "No parameters.npy file found to load."

def save_all_parameters(
    filepath,
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
    center_x_var,    # <--- ADDED
    center_y_var,    # <--- ADDED
    resolution_var=None,
    custom_samples_var=None,
    optics_mode_var=None,
):
    """
    Save all slider parameters into a .npy file as a dictionary. This now
    includes beam center (center_x, center_y).
    """
    parameters = {
        'theta_initial':  theta_initial_var.get(),
        'cor_angle':      cor_angle_var.get(),
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
        'center_y':       center_y_var.get(),
    }
    if resolution_var is not None:
        resolution_value = resolution_var.get()
        parameters['sampling_resolution'] = resolution_value
    else:
        resolution_value = None

    if custom_samples_var is not None:
        try:
            custom_sample_count = int(round(float(custom_samples_var.get())))
        except (TypeError, ValueError):
            custom_sample_count = None
        if custom_sample_count is not None and custom_sample_count > 0:
            parameters['sampling_custom_count'] = custom_sample_count
            if resolution_value == "Custom":
                parameters['sampling_count'] = custom_sample_count

    if optics_mode_var is not None:
        parameters['optics_mode'] = _normalize_optics_mode(
            optics_mode_var.get(),
            fallback="fast",
        )
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
