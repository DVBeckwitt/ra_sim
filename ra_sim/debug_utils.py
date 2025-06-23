"""Helper utilities for debugging RA_SIM execution."""

import os
import numpy as np


def is_debug_enabled() -> bool:
    """Return ``True`` if ``RA_SIM_DEBUG`` is set to a truthy value."""
    val = os.environ.get("RA_SIM_DEBUG", "")
    return bool(val) and val.lower() not in {"0", "false", "no"}


DEBUG_ENABLED = is_debug_enabled()


def debug_print(*args, **kwargs) -> None:
    """Print only when ``RA_SIM_DEBUG`` is enabled."""
    if DEBUG_ENABLED:
        print(*args, **kwargs)


def check_ht_arrays(miller1: np.ndarray, intens1: np.ndarray) -> None:
    """Print diagnostics for Hendricks–Teller arrays when debugging is enabled."""
    debug_print("miller1 dtype:", miller1.dtype, "shape:", miller1.shape)
    if miller1.size:
        l_min = float(miller1[:, 2].min())
        l_max = float(miller1[:, 2].max())
        debug_print("L range:", l_min, l_max)
        if np.allclose(miller1[:, 2], np.round(miller1[:, 2])):
            debug_print("L values are integer-only")
        else:
            frac_count = np.count_nonzero(~np.isclose(miller1[:, 2], np.round(miller1[:, 2])))
            debug_print(f"L values contain {frac_count} fractional entries")
    else:
        debug_print("L range: array empty")
    if intens1.size:
        i_min = float(intens1.min())
        i_max = float(intens1.max())
    else:
        i_min = i_max = float('nan')
    debug_print("intens1 dtype:", intens1.dtype, "min:", i_min, "max:", i_max)
    debug_print("miller1 contiguous:", miller1.flags['C_CONTIGUOUS'])
    debug_print("intens1 contiguous:", intens1.flags['C_CONTIGUOUS'])
