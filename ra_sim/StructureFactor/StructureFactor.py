"""Legacy Bi2Se3-specific X-ray structure-factor helpers."""

import numpy as np
from numba import njit

_SUPPORTED_BI2SE3_LABELS = ("Bi", "Se1", "Se2")


def calculate_bi2se3_structure_factor(h, k, l, atoms, data, occ):
    """Return one legacy Bi2Se3 structure-factor intensity.

    This helper is intentionally material-specific. It preserves the older
    ``sqrt(h^2 + k^2 + l^2)`` Q surrogate only for the legacy Bi2Se3 path.
    """

    if len(occ) != 3:
        raise ValueError(
            f"Bi2Se3 structure factor requires exactly 3 occupancies for "
            f"{_SUPPORTED_BI2SE3_LABELS}; got {len(occ)}."
        )
    if len(data) != 2:
        raise ValueError(
            "Bi2Se3 structure factor requires exactly 2 scattering-factor "
            "callables: [Bi, Se]."
        )
    if not all(callable(fn) for fn in data):
        raise ValueError(
            "Bi2Se3 structure factor requires scattering-factor callables for [Bi, Se]."
        )

    atoms = tuple(atoms)
    atom_labels = [name for name, *_coords in atoms]
    unsupported_labels = sorted(
        {str(label) for label in atom_labels if label not in _SUPPORTED_BI2SE3_LABELS}
    )
    if unsupported_labels:
        raise ValueError(
            "Unsupported Bi2Se3 atom labels: "
            f"{unsupported_labels}. Supported labels: {_SUPPORTED_BI2SE3_LABELS}."
        )

    F_hkl = 0 + 0j  # Initialize complex number for structure factor

    q_magnitude = np.sqrt(h**2 + k**2 + l**2)

    # Substitute legacy q surrogate into Bi and Se scattering-factor callables.
    f_bi = data[0](q_magnitude) - 4.23706 + 8.83640j
    f_se = data[1](q_magnitude) - 0.787865 + 1.13462j

    # Precompute 2j * np.pi
    two_pi_j = 2j * np.pi

    label_to_factor = {
        "Bi": f_bi * occ[0],
        "Se1": f_se * occ[1],
        "Se2": f_se * occ[2],
    }

    # Calculate structure factor for each atom
    for name, x, y, z in atoms:
        f = label_to_factor[name]
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
