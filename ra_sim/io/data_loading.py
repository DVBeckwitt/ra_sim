import numpy as np
import numpy as np
import os

def load_parameters(path,
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
                    eta_var):
    """
    Load parameters from a .npy file and update the given Tkinter variables.
    """
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
    eta_var
):
    """
    Gathers all parameters and saves them to a file.

    Args:
        filepath (str): The file path where the parameters will be saved.
        theta_initial_var, gamma_var, Gamma_var, etc.: Tkinter DoubleVar objects for the parameters.
    """
    try:
        # Gather parameters
        parameters = {
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
        # Save to file
        np.save(filepath, parameters)
        print(f"Parameters saved successfully to {filepath}")
    except Exception as e:
        print(f"Failed to save parameters: {e}")

def load_background_image(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        pixel_lines = lines[6:]
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        image = flattened_pixels.reshape((3000, 3000))
    return image


def load_and_format_reference_profiles(input_filename):
    # Load the data from the .npy file
    data = np.load(input_filename, allow_pickle=True).item()

    # Extract relevant arrays
    radial_region = np.array(data['Radial']['Region'])
    radial_theta = np.array(data['Radial']['Radial (2θ)'])
    radial_intensity = np.array(data['Radial']['Intensity'])

    azimuthal_region = np.array(data['Azimuthal']['Region'])
    azimuthal_phi = np.array(data['Azimuthal']['Azimuthal Angle (φ)'])
    azimuthal_intensity = np.array(data['Azimuthal']['Azimuthal Intensity'])

    # Identify unique regions
    unique_regions = np.unique(radial_region)

    reference_profiles = {}

    # For each unique region, gather the radial and azimuthal data
    for region in unique_regions:
        # Radial data for this region
        mask_rad = (radial_region == region)
        region_radial_theta = radial_theta[mask_rad]
        region_radial_intensity = radial_intensity[mask_rad]

        # Sort by 2θ if needed
        sort_idx_rad = np.argsort(region_radial_theta)
        region_radial_theta = region_radial_theta[sort_idx_rad]
        region_radial_intensity = region_radial_intensity[sort_idx_rad]

        # Azimuthal data for this region
        mask_az = (azimuthal_region == region)
        region_azimuthal_phi = azimuthal_phi[mask_az]
        region_azimuthal_int = azimuthal_intensity[mask_az]

        # Sort by φ if needed
        sort_idx_az = np.argsort(region_azimuthal_phi)
        region_azimuthal_phi = region_azimuthal_phi[sort_idx_az]
        region_azimuthal_int = region_azimuthal_int[sort_idx_az]

        # Store in the desired dictionary format
        reference_profiles[region] = {
            'Radial (2θ)': region_radial_theta,
            'Intensity': region_radial_intensity,
            'Azimuthal Angle (φ)': region_azimuthal_phi,
            'Azimuthal Intensity': region_azimuthal_int
        }

    return reference_profiles