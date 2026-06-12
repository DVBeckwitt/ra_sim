import numpy as np

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