import json

def parse_poni_file(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split(':', 1)
                value = value.strip()
                try:
                    parameters[key.strip()] = float(value)
                except ValueError:
                    parameters[key.strip()] = value
    return parameters
