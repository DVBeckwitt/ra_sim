import numpy as np
import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator   
import json
from ra_sim.utils.calculations import d_spacing, two_theta
from ra_sim.StructureFactor.StructureFactor  import calculate_structure_factor

import numpy as np
import itertools

def setup_azimuthal_integrator(parameters):
    # Parse the detector configuration from the JSON-like structure
    detector_config = json.loads(parameters['Detector_config'])
    pixel1 = float(detector_config['pixel1'])
    pixel2 = float(detector_config['pixel2'])
    max_shape = list(map(int, detector_config['max_shape']))  # Convert max_shape elements to integers

    # Initialize the AzimuthalIntegrator
    ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(
        dist=parameters['Distance'],
        poni1=parameters['Poni1'],
        poni2=  parameters['Poni2'],
        pixel1= pixel1,
        pixel2= pixel2,
        rot1= -parameters['Rot1']* 180/np.pi,
        rot2= -parameters['Rot2']*180/np.pi,
        rot3= parameters['Rot3'],
        wavelength=parameters['Wavelength']
    )
    return ai


def miller_generator(max_miller, av, cv, lambda_, atoms, data, occ):
    # Generate Miller Indices
    Raw_Miller = [p for p in itertools.product(range(max_miller + 1), repeat=3)]  # Now includes 0 to mx, inclusive
    # Filter Miller indices based on 2theta condition and structure factor intensity
    miller = []
    intensities = []
    for h, k, l in Raw_Miller:
        D = d_spacing(h, k, l, av, cv)
        two_theta_value = two_theta(D, lambda_)
        if two_theta_value is not None and 0 <= two_theta_value <= 70:
            intensity = calculate_structure_factor(h, k, l, atoms, data, occ)
            if intensity > 1:  # Arbitrary threshold for small intensity
                miller.append((h, k, l))
                intensities.append(intensity)
    miller = np.array(miller, dtype = np.int32)
    intensities = np.array(intensities, dtype = np.float64)

    return miller, intensities

