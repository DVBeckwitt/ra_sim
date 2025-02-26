import numpy as np
import matplotlib.pyplot as plt

def sample_2d_gaussian(n, sigma):
    return np.random.normal(0, sigma, (n, 2))

def sample_2d_cauchy(n, gamma):
    # For a 1D Cauchy: x = gamma * tan(pi*(u-0.5)), where u is uniform in (0,1).
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)
    x = gamma * np.tan(np.pi * (u1 - 0.5))
    y = gamma * np.tan(np.pi * (u2 - 0.5))
    return np.column_stack((x, y))

def sample_mosaic_uniform(n):
    # Sample beta and kappa uniformly in degrees over [-180, 180]
    beta = np.random.uniform(-180, 180, n)
    kappa = np.random.uniform(-180, 180, n)
    return np.column_stack((beta, kappa))

def generate_random_profiles(num_samples, divergence_sigma, bw_sigma, sigma_mosaic_deg, gamma_mosaic_deg, eta, lambda0, Bandwidth):
    # Mosaic profile: now sampled uniformly in beta and kappa from -180 to 180 degrees.
    Mosaic_samples = sample_mosaic_uniform(num_samples)
    beta_array = Mosaic_samples[:, 0]
    kappa_array = Mosaic_samples[:, 1]
    mosaic_intensity_array = np.ones(num_samples)
    
    # Generate divergence profile (Gaussian)
    Divergence_samples = sample_2d_gaussian(num_samples, divergence_sigma)
    theta_array = Divergence_samples[:, 0]
    phi_array = Divergence_samples[:, 1]
    divergence_intensity_array = np.ones(num_samples)
    
    # Generate beam profile (Gaussian)
    Beam_samples = sample_2d_gaussian(num_samples, bw_sigma)
    beam_x_array = Beam_samples[:, 0]
    beam_y_array = Beam_samples[:, 1]
    beam_intensity_array = np.ones(num_samples)
    
    # Generate bandpass profile (wavelength distribution)
    bandpass_sigma = Bandwidth * lambda0  # assuming Bandwidth is relative
    wavelength_array = np.random.normal(lambda0, bandpass_sigma, num_samples)
    
    return (beam_x_array, beam_y_array, beam_intensity_array,
            beta_array, kappa_array, mosaic_intensity_array,
            theta_array, phi_array, divergence_intensity_array,
            wavelength_array)
