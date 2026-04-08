from __future__ import annotations

import numpy as np

from ra_sim.simulation.mosaic_profiles import (
    cluster_beam_profiles,
    generate_random_profiles,
)


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


def test_generate_random_profiles_uses_antithetic_pairs_and_center_sample() -> None:
    lambda0 = 1.54
    beam_x, beam_y, theta, phi, wavelength = generate_random_profiles(
        5,
        divergence_sigma=0.02,
        bw_sigma=0.004,
        lambda0=lambda0,
        bandwidth=0.006,
        rng=np.random.default_rng(7),
    )

    np.testing.assert_allclose(theta[0] + theta[1], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(phi[0] + phi[1], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(beam_x[0] + beam_x[1], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(beam_y[0] + beam_y[1], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(wavelength[0] + wavelength[1], 2.0 * lambda0, atol=1e-12, rtol=0.0)

    np.testing.assert_allclose(theta[-1], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(phi[-1], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(beam_x[-1], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(beam_y[-1], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(wavelength[-1], lambda0, atol=0.0, rtol=0.0)

def test_cluster_beam_profiles_preserves_total_weight_and_mappings() -> None:
    beam_x, beam_y, theta, phi, wavelength = generate_random_profiles(
        128,
        divergence_sigma=0.02,
        bw_sigma=0.004,
        lambda0=1.54,
        bandwidth=0.006,
        rng=np.random.default_rng(11),
    )

    (
        clustered_beam_x,
        clustered_beam_y,
        clustered_theta,
        clustered_phi,
        clustered_wavelength,
        sample_weights,
        raw_to_cluster,
        cluster_to_rep,
    ) = cluster_beam_profiles(beam_x, beam_y, theta, phi, wavelength)

    cluster_count = clustered_beam_x.shape[0]
    assert cluster_count < beam_x.shape[0]
    assert clustered_beam_y.shape == (cluster_count,)
    assert clustered_theta.shape == (cluster_count,)
    assert clustered_phi.shape == (cluster_count,)
    assert clustered_wavelength.shape == (cluster_count,)
    np.testing.assert_allclose(np.sum(sample_weights), float(beam_x.shape[0]))
    assert raw_to_cluster.shape == (beam_x.shape[0],)
    assert cluster_to_rep.shape == (cluster_count,)
    assert np.all(raw_to_cluster >= 0)
    assert np.all(raw_to_cluster < cluster_count)
    assert np.all(cluster_to_rep >= 0)
    assert np.all(cluster_to_rep < beam_x.shape[0])
