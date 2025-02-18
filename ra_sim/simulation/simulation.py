import numpy as np

from ra_sim.simulation.mosaic_profiles import generate_random_profiles

from ra_sim.simulation.diffraction import process_peaks_parallel
from ra_sim.utils.calculations import IndexofRefraction, fresnel_transmission

# Load background images
def simulate_diffraction(theta_initial, gamma, Gamma, chi, zs, zb, debye_x_value, debye_y_value, corto_detector_value, miller, intensities, image_size, 
                         av, cv, lambda_, psi, n2, center, num_samples, divergence_sigma, bw_sigma, sigma_mosaic_var, gamma_mosaic_var,eta_var):
    # Get updated parameters for mosaic and eta
    current_sigma_mosaic = np.radians(sigma_mosaic_var.get())
    current_gamma_mosaic = np.radians(gamma_mosaic_var.get())
    current_eta = eta_var.get()

    # Regenerate the random profiles with updated parameters
    (beam_x_array, beam_y_array, beam_intensity_array,
     beta_array, kappa_array, mosaic_intensity_array,
     theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
         num_samples,
         divergence_sigma,
         bw_sigma,
         current_sigma_mosaic,
         current_gamma_mosaic,
         current_eta
    )

    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])
    
    simulated_image = process_peaks_parallel(
        miller, intensities, image_size, av, cv, lambda_, np.zeros((image_size, image_size)),
        corto_detector_value, gamma, Gamma, chi, psi, zs, zb, n2,
        beam_x_array, beam_y_array, beam_intensity_array,
        beta_array, kappa_array, mosaic_intensity_array,
        theta_array, phi_array, divergence_intensity_array,
        debye_x_value, debye_y_value, center,
        theta_initial, theta_initial + 0.1, 0.1,
        unit_x, n_detector,
        save_flag = 0 
    )
    
    return simulated_image