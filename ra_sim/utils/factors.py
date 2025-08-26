"""Form factor utilities."""

from __future__ import annotations

import re
from typing import Dict

import numpy as np
from Dans_Diffraction.functions_crystallography import (
    xray_scattering_factor,
    xray_dispersion_corrections,
)

# Default ionic labels for a few common elements. Extend as needed.
_DEFAULT_ION_MAP = {
    "Pb": "Pb2+",
    "I": "I1-",
}


def _parse_formula(formula: str) -> list[tuple[str, int]]:
    """Return list of ``(element, count)`` pairs from ``formula``."""
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    if not tokens:
        raise ValueError(f"Unable to parse formula '{formula}'")
    out = []
    for sym, num in tokens:
        out.append((sym, int(num) if num else 1))
    return out


def F_comp(el: str, Q: np.ndarray, energy_kev: float, ion_map: Dict[str, str] | None = None) -> np.ndarray:
    """Return complex scattering factor for one element.

    Parameters
    ----------
    el:
        Chemical symbol, e.g. ``"Pb"`` or ``"I"``.
    Q:
        Scattering vector magnitude in Å⁻¹.
    energy_kev:
        Photon energy in keV.
    ion_map:
        Optional mapping of element symbols to ionic labels used for
        ``xray_scattering_factor``. If not provided, ``_DEFAULT_ION_MAP`` is
        consulted and falls back to the neutral symbol.
    """
    ion_map = ion_map or _DEFAULT_ION_MAP
    ion_label = ion_map.get(el, el)
    q = np.asarray(Q, dtype=float).reshape(-1)
    f0 = xray_scattering_factor([ion_label], q)[:, 0]
    f1, f2 = xray_dispersion_corrections([el], energy_kev=[energy_kev])
    f1 = float(f1[0, 0])
    f2 = float(f2[0, 0])
    f = f0 + f1 + 1j * f2
    return f.reshape(Q.shape)


def ionic_atomic_form_factors(formula: str, Q: np.ndarray, energy_kev: float, ion_map: Dict[str, str] | None = None) -> Dict[str, np.ndarray]:
    """Compute ionic atomic form factors for every element in ``formula``.

    The return value is a dictionary mapping each element symbol to the total
    scattering factor ``f = f₀ + f′ + i f″`` evaluated at all ``Q`` values.

    Parameters
    ----------
    formula:
        Chemical formula such as ``"PbI2"``.
    Q:
        Scattering vector magnitude(s) in Å⁻¹.
    energy_kev:
        Photon energy in keV used for the dispersion corrections.
    ion_map:
        Optional mapping of element symbols to ionic labels. By default
        ``_DEFAULT_ION_MAP`` is used and falls back to the neutral symbol if
        an element is not present.

    Returns
    -------
    dict
        ``{element: form_factor_array}`` where ``form_factor_array`` has the
        same shape as ``Q``.

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.linspace(0, 2, 5)
    >>> ff = ionic_atomic_form_factors("PbI2", Q, 8.047)
    >>> list(ff)
    ['Pb', 'I']
    >>> ff['Pb'].shape
    (5,)
    """
    factors: Dict[str, np.ndarray] = {}
    for sym, _ in _parse_formula(formula):
        factors[sym] = F_comp(sym, Q, energy_kev, ion_map=ion_map)
    return factors

__all__ = ["ionic_atomic_form_factors", "F_comp"]

