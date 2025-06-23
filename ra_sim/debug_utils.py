import numpy as np


def check_ht_arrays(miller1: np.ndarray, intens1: np.ndarray) -> None:
    """Print diagnostics for Hendricks–Teller arrays when debugging is enabled."""
    print("miller1 dtype:", miller1.dtype, "shape:", miller1.shape)
    if miller1.size:
        l_min = float(miller1[:, 2].min())
        l_max = float(miller1[:, 2].max())
        print("L range:", l_min, l_max)
    else:
        print("L range: array empty")
    if intens1.size:
        i_min = float(intens1.min())
        i_max = float(intens1.max())
    else:
        i_min = i_max = float('nan')
    print("intens1 dtype:", intens1.dtype, "min:", i_min, "max:", i_max)
    print("miller1 contiguous:", miller1.flags['C_CONTIGUOUS'])
    print("intens1 contiguous:", intens1.flags['C_CONTIGUOUS'])
