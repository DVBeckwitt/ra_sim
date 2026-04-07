from __future__ import annotations

import numpy as np

from ra_sim.simulation.mosaic_profiles import (
    cluster_beam_profiles,
    generate_random_profiles,
    generate_stratified_profiles,
    sample_stratified_gaussian_1d,
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


def test_sample_stratified_gaussian_1d_is_reproducible() -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    samples_a = sample_stratified_gaussian_1d(
        16,
        mean=1.5,
        sigma=0.2,
        rng=rng_a,
    )
    samples_b = sample_stratified_gaussian_1d(
        16,
        mean=1.5,
        sigma=0.2,
        rng=rng_b,
    )

    np.testing.assert_allclose(samples_a, samples_b)


def test_generate_stratified_profiles_returns_equal_weight_cartesian_product() -> None:
    (
        beam_x,
        beam_y,
        theta,
        phi,
        wavelength,
        sample_weights,
    ) = generate_stratified_profiles(
        x_mean=0.0,
        x_sigma=0.01,
        x_samples=2,
        y_mean=0.0,
        y_sigma=0.02,
        y_samples=3,
        dx_mean=0.0,
        dx_sigma=0.03,
        dx_samples=2,
        dz_mean=0.0,
        dz_sigma=0.04,
        dz_samples=2,
        lambda_mean=1.54,
        lambda_sigma=0.005,
        lambda_samples=4,
        rng=np.random.default_rng(7),
    )

    assert beam_x.shape == (96,)
    assert beam_y.shape == (96,)
    assert theta.shape == (96,)
    assert phi.shape == (96,)
    assert wavelength.shape == (96,)
    assert sample_weights.shape == (96,)
    np.testing.assert_allclose(sample_weights, np.full(96, 1.0 / 96.0))
    np.testing.assert_allclose(np.sum(sample_weights), 1.0)

    _, x_counts = np.unique(beam_x, return_counts=True)
    _, y_counts = np.unique(beam_y, return_counts=True)
    _, theta_counts = np.unique(theta, return_counts=True)
    _, phi_counts = np.unique(phi, return_counts=True)
    _, wavelength_counts = np.unique(wavelength, return_counts=True)
    np.testing.assert_array_equal(x_counts, np.full(2, 48))
    np.testing.assert_array_equal(y_counts, np.full(3, 32))
    np.testing.assert_array_equal(theta_counts, np.full(2, 48))
    np.testing.assert_array_equal(phi_counts, np.full(2, 48))
    np.testing.assert_array_equal(wavelength_counts, np.full(4, 24))


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
