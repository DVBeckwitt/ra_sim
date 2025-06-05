"""Geometry helpers for configuring pyFAI integrators."""

import json
import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

def setup_azimuthal_integrator(parameters):
    detector_config = json.loads(parameters['Detector_config'])
    pixel1 = float(detector_config['pixel1'])
    pixel2 = float(detector_config['pixel2'])
    # max_shape not necessarily needed directly, but available if required

    ai = AzimuthalIntegrator(
        dist=parameters['Distance'],
        poni1=parameters['Poni1'],
        poni2=parameters['Poni2'],
        pixel1=pixel1,
        pixel2=pixel2,
        rot1=-parameters['Rot1'] * 180/np.pi,
        rot2=-parameters['Rot2'] * 180/np.pi,
        rot3=parameters['Rot3'],
        wavelength=parameters['Wavelength']
    )
    return ai
