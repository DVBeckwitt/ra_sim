import numpy as np
from ra_sim.utils.stacking_fault import (
    ht_integrated_area,
    ht_numeric_area,
    p_from_ht_area,
    _cell_c_from_cif,
)

def test_ht_area_and_p_inversion():
    c = _cell_c_from_cif('tests/local_test.cif')
    p_true = 0.2
    h, k, ell = 2, 1, 3
    analytic = ht_integrated_area(p_true, h, k, ell, c)
    numeric = ht_numeric_area(p_true, h, k, ell, c, nphi=4001)
    assert np.isclose(analytic, numeric, rtol=1e-5)
    p_est = p_from_ht_area(analytic, h, k, ell, c)
    assert np.isclose(p_est, p_true, rtol=1e-5)
