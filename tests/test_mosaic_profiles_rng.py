from __future__ import annotations

import numpy as np

from ra_sim.simulation.mosaic_profiles import generate_random_profiles


def test_generate_random_profiles_reproducible_with_seeded_rng() -> None:
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(12345)

    out_a = generate_random_profiles(
        128,
        divergence_sigma=0.01,
        bw_sigma=0.005,
        lambda0=1.54,
        bandwidth=0.007,
        rng=rng_a,
    )
    out_b = generate_random_profiles(
        128,
        divergence_sigma=0.01,
        bw_sigma=0.005,
        lambda0=1.54,
        bandwidth=0.007,
        rng=rng_b,
    )

    for arr_a, arr_b in zip(out_a, out_b):
        assert np.allclose(arr_a, arr_b)


def test_generate_random_profiles_default_rng_shapes() -> None:
    beam_x, beam_y, theta, phi, wavelength = generate_random_profiles(
        64,
        divergence_sigma=0.02,
        bw_sigma=0.004,
        lambda0=1.54,
        bandwidth=0.006,
    )
    assert beam_x.shape == (64,)
    assert beam_y.shape == (64,)
    assert theta.shape == (64,)
    assert phi.shape == (64,)
    assert wavelength.shape == (64,)
