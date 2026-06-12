import numpy as np

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