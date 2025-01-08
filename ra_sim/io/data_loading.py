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
                    eta_var,
                    a_var,              # new
                    c_var):             # new
    if os.path.exists(path):
        params = np.load(path, allow_pickle=True).item()
        
        # Set all the old parameters
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

        # Set the new lattice parameters
        a_var.set(params['a'])    # or 'a_lattice' if you prefer
        c_var.set(params['c'])

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
    c_var
):
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
        'c':              c_var.get()
    }
    np.save(filepath, parameters)
    print(f"Parameters saved successfully to {filepath}")



def load_background_image(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        pixel_lines = lines[6:]
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        image = flattened_pixels.reshape((3000, 3000))
    return image

def load_and_format_reference_profiles(input_filename):
    """
    Loads the new .npy file produced by process_data(...) and
    creates a simpler dictionary for each region:
      {
        region_name: {
          "Radial (2θ)": np.array(...),
          "Radial Intensity": np.array(...),
          "Azimuthal φ": np.array(...),
          "Azimuthal Intensity": np.array(...),
          "FittedParams": { ... }  # optional
        },
        ...
      }
    """

    raw_data = np.load(input_filename, allow_pickle=True).item()
    # raw_data should have keys: "Regions", "MainData", "Regions_of_Interest"

    # A dictionary to store the reformatted results
    reference_profiles = {}

    # 1) Pull out the "Regions" dictionary from raw_data
    regions_dict = raw_data.get("Regions", {})

    for region_name, region_info in regions_dict.items():
        # region_info should have "Radial", "Azimuthal", and "FittedParams"
        radial_dict = region_info.get("Radial", {})
        azimuthal_dict = region_info.get("Azimuthal", {})
        fitted_params = region_info.get("FittedParams", {})

        # Extract arrays from radial_dict
        # "2θ" is a list, convert to np.array
        radial_2theta = np.array(radial_dict.get("2θ", []), dtype=float)
        radial_intensity = np.array(radial_dict.get("Intensity", []), dtype=float)

        # Extract arrays from azimuthal_dict
        azimuthal_phi = np.array(azimuthal_dict.get("φ", []), dtype=float)
        azimuthal_intensity = np.array(azimuthal_dict.get("Intensity", []), dtype=float)

        # Build your consolidated entry
        reference_profiles[region_name] = {
            "Radial (2θ)": radial_2theta,
            "Radial Intensity": radial_intensity,
            "Azimuthal φ": azimuthal_phi,
            "Azimuthal Intensity": azimuthal_intensity,
            # Optionally store the fitted parameters too
            "FittedParams": fitted_params
        }

    return reference_profiles
