import json


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