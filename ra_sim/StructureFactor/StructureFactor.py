import numpy as np
from numba import njit

# Function to calculate the structure factor
def calculate_structure_factor(h, k, l, atoms, data, occ):
    F_hkl = 0 + 0j  # Initialize complex number for structure factor

    # Substitute q into equation data[0] and data[1]
    f_Bi = data[0](np.sqrt(h**2 + k**2 + l**2))
    f_Se = data[1](np.sqrt(h**2 + k**2 + l**2))

    # Precompute 2j * np.pi
    two_pi_j = 2j * np.pi

    # Precompute the occupancies multipliers
    f_Bi *= occ[0]
    f_Se *= occ[1]
    f_Se2 = f_Se * occ[2]

    # Calculate structure factor for each atom
    for name, x, y, z in atoms:
        if name == 'Bi':
            f = f_Bi
        elif name == 'Se1':
            f = f_Se
        elif name == 'Se2':
            f = f_Se2

        # Calculate phase_in more efficiently
        phase_in = two_pi_j * (h * x + k * y + l * z)
        F_hkl += f * np.exp(phase_in)

    return np.abs(F_hkl)**2

@njit
def attenuation(c_nm, qz_real, qz_imag):
    """
    Compute the finite-layer interference factor with an additional surface roughness damping.
    
    Parameters:
      c_nm    : layer spacing in nm
      qz_real : real part of qz (in nm^-1)
      qz_imag : imaginary part of qz (in nm^-1)
      sigma   : roughness (nm)
      
    Returns:
      The attenuated interference factor (float).
      
    Note:
      Thickness is given in Ångströms. Make sure c_nm is in nm.
    """
    Thickness = 500.0  # Example thickness in Ångströms
    # Compute number of layers (N)
    N = np.round(Thickness / c_nm)
    
    # Create complex qz using complex(0,1)
    qz = (qz_real + complex(0,1)*qz_imag)
    # Compute phase: phi = qz * c_nm
    phi = qz * c_nm
    half_phi = 0.5 * phi

    # Avoid division by zero: if |sin(half_phi)| is too small, set ratio to N directly.
    sin_half_phi = np.sin(half_phi)
    if abs(sin_half_phi) < 1e-14:
        ratio = N
    else:
        ratio = np.sin(N * half_phi) / sin_half_phi

    F = (abs(ratio))**2

    return F 