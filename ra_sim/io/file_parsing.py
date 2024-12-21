import json
import numpy as np
from ra_sim.utils.tools import detect_blobs

def parse_poni_file(file_path):
    # Dictionary to hold the values
    parameters = {}
    # Read the file and extract parameters
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split(':', 1)  # Split only on the first colon
                value = value.strip()
                try:
                    parameters[key.strip()] = float(value)  # Try converting to float
                except ValueError:
                    parameters[key.strip()] = value  # Store as string if conversion fails

    return parameters

def Open_ASC(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        pixel_lines = lines[6:]
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        background_image = flattened_pixels.reshape((3000, 3000))
        
    return np.rot90(background_image, k=3)

