import numpy as np
from ra_sim.utils.tools import miller_generator, intensities_for_hkls


def test_intensities_for_hkls_matches_miller_generator():
    cif = 'tests/local_test.cif'
    hkls = [(1, 0, 1), (0, 0, 1), (0, 0, 2)]
    lam = 1.0
    occ = [1.0]
    ints = intensities_for_hkls(hkls, cif, occ, lam, energy=8.047,
                                intensity_threshold=0, two_theta_range=(0, 180))
    _, _, _, details = miller_generator(3, cif, occ, lam, 8.047,
                                        intensity_threshold=0,
                                        two_theta_range=(0, 180))
    mapping = {}
    for group in details:
        for hkl, val in group:
            mapping[tuple(hkl)] = val
    scale = ints[0] / mapping[hkls[0]] if mapping.get(hkls[0]) else 1.0
    expected = np.array([mapping.get(hkl, 0.0) * scale for hkl in hkls])
    assert np.allclose(ints, expected, atol=1e-2)
