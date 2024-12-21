import numpy as np

# objective_functions.py

import numpy as np
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from numpy import deg2rad
from numpy import rot90

from bayes_opt import BayesianOptimization
import traceback

def run_optimization(
    fit_button, progress_label,
    chi_scale, zs_scale, zb_scale,
    eta_var, sigma_mosaic_var, gamma_mosaic_var,
    chi_var, zs_var, zb_var,
    gamma_var, Gamma_var, theta_initial_var, debye_x_var, debye_y_var, corto_detector_var,
    simulate_diffraction, miller, intensities, image_size, av, cv, lambda_, psi, n2, center,
    num_samples, divergence_sigma, bw_sigma, ai_initial, process_data, compute_cost, reference_profiles,
    objective_function_bayesian
):
    try:
        fit_button.config(state='disabled')
        progress_label.config(text="Optimization in progress...")

        pbounds = {
            'chi': (float(chi_scale.cget('from')), float(chi_scale.cget('to'))),
            'zs': (float(zs_scale.cget('from')), float(zs_scale.cget('to'))),
            'zb': (float(zb_scale.cget('from')), float(zb_scale.cget('to'))),
            'eta': (0.0, 1.0),
            'sigma_mosaic_deg': (0.0, 5.0),
            'gamma_mosaic_deg': (0.0, 5.0)
        }

        initial_params = {
            'chi': chi_var.get(),
            'zs': zs_var.get(),
            'zb': zb_var.get(),
            'eta': eta_var.get(),
            'sigma_mosaic_deg': sigma_mosaic_var.get(),
            'gamma_mosaic_deg': gamma_mosaic_var.get()
        }

        def bayes_objective(chi, zs, zb, eta, sigma_mosaic_deg, gamma_mosaic_deg):
            return objective_function_bayesian(
                chi, zs, zb, eta, sigma_mosaic_deg, gamma_mosaic_deg,
                eta_var, sigma_mosaic_var, gamma_mosaic_var, chi_var, zs_var, zb_var,
                gamma_var, Gamma_var, theta_initial_var, debye_x_var, debye_y_var, corto_detector_var,
                simulate_diffraction, miller, intensities, image_size, av, cv, lambda_, psi, n2, center,
                num_samples, divergence_sigma, bw_sigma, ai_initial, process_data, compute_cost, reference_profiles
            )

        optimizer = BayesianOptimization(
            f=bayes_objective,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        optimizer.probe(params=initial_params, lazy=True)
        optimizer.maximize(init_points=5, n_iter=50)

        best_params = optimizer.max['params']
        chi_var.set(best_params['chi'])
        zs_var.set(best_params['zs'])
        zb_var.set(best_params['zb'])

        progress_label.config(text=f"Optimization complete.\n"
                                   f"Best Loss: {-optimizer.max['target']:.2e}\n"
                                   f"Optimal Parameters:\n"
                                   f"Chi: {best_params['chi']:.4f}\n"
                                   f"Zs: {best_params['zs']:.6f}\n"
                                   f"Zb: {best_params['zb']:.6f}")
    except Exception as e:
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        progress_label.config(text=f"Optimization failed with exception:\n{traceback_str}")
    finally:
        fit_button.config(state='normal')


def compute_cost(integrated_data, reference_profiles):
    # We'll accumulate squared differences for all regions and then take the mean
    squared_errors = []
    for region_data in integrated_data:
        region_name = region_data['Region']
        
        # Extract simulated profiles
        sim_theta = region_data['Radial (2θ)']
        sim_intensity = region_data['Intensity']
        
        # Extract reference profiles
        ref_theta = reference_profiles[region_name]['Radial (2θ)']
        ref_intensity = reference_profiles[region_name]['Intensity']
        
        # Ensure both arrays are aligned. If they are not the same length or grid, 
        # you may need interpolation. For simplicity, let's assume they match in length and order.
        if len(sim_theta) == len(ref_theta):
            # Compute squared error for this region
            diff = sim_intensity - ref_intensity
            squared_errors.extend(diff**2)
        else:
            # If lengths differ, consider interpolation or a different approach
            # For now, just skip or handle gracefully:
            continue
    
    if len(squared_errors) == 0:
        # No data to compare, return a large cost
        return 1e9
    
    # Mean Squared Error
    mse = np.mean(squared_errors)
    return mse


def objective_function_bayesian(
    chi_val, zs_val, zb_val, eta_val, sigma_mosaic_deg_val, gamma_mosaic_deg_val,
    eta_var, sigma_mosaic_var, gamma_mosaic_var, chi_var, zs_var, zb_var,
    gamma_var, Gamma_var, theta_initial_var, debye_x_var, debye_y_var, corto_detector_var,
    simulate_diffraction, miller, intensities, image_size, av, cv, lambda_, psi, n2, center,
    num_samples, divergence_sigma, bw_sigma, ai_initial, process_data, compute_cost, reference_profiles
):
    # Set the sliders to these trial values
    eta_var.set(eta_val)
    sigma_mosaic_var.set(sigma_mosaic_deg_val)
    gamma_mosaic_var.set(gamma_mosaic_deg_val)
    chi_var.set(chi_val)
    zs_var.set(zs_val)
    zb_var.set(zb_val)

    # Retrieve current slider values
    gamma_fixed = gamma_var.get()
    Gamma_fixed = Gamma_var.get()
    theta_initial = theta_initial_var.get()
    debye_x_val = debye_x_var.get()
    debye_y_val = debye_y_var.get()
    corto_detector_val = corto_detector_var.get()

    try:
        # Simulate diffraction
        simulated_image = simulate_diffraction(
            theta_initial, gamma_fixed, Gamma_fixed, chi_val, zs_val, zb_val, debye_x_val,
            debye_y_val, corto_detector_val, miller, intensities, image_size, av, cv, lambda_,
            psi, n2, center, num_samples, divergence_sigma, bw_sigma, sigma_mosaic_var, gamma_mosaic_var, eta_var
        )


        # Prepare AzimuthalIntegrator with updated parameters
        pixel_size = 100e-6
        center_x = center[0]
        center_y = center[1]
        poni1 = (3000 - center_x) * pixel_size
        poni2 = (3000 - center_y) * pixel_size

        rot1 = deg2rad(Gamma_fixed)
        rot2 = deg2rad(gamma_fixed)
        rot3 = 0.0

        wavelength = ai_initial._wavelength

        ai = AzimuthalIntegrator(
            dist=corto_detector_val,
            poni1=poni1,
            poni2=poni2,
            pixel1=pixel_size,
            pixel2=pixel_size,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            wavelength=wavelength
        )

        # Define Regions of Interest
        regions_of_interest = [
            {'theta_min': 7, 'theta_max': 12, 'phi_min': -20, 'phi_max': 20, 'name': '003'},
            {'theta_min': 16, 'theta_max': 19, 'phi_min': -10, 'phi_max': 10, 'name': '006'},
        ]

        # Process data and compute cost
        integrated_data = process_data(ai, simulated_image, regions_of_interest)
        cost = compute_cost(integrated_data, reference_profiles)

        # Return negative cost for Bayesian Optimization (which typically maximizes)
        return -cost
    except:
        return 1e9


def process_data(ai, data, regions_of_interest):
    """
    Processes diffraction data for specified regions of interest.

    Args:
        ai (AzimuthalIntegrator): Azimuthal integrator object for 2D integration.
        data (numpy.ndarray): 2D diffraction pattern data.
        regions_of_interest (list of dict): List of dictionaries defining regions of interest.

    Returns:
        list: Processed data for each region of interest.
    """
    # Perform 2D integration
    res2 = ai.integrate2d(
        data,
        npt_rad=3000,
        npt_azim=1000,
        unit="2th_deg",
    )

    # Extract arrays
    intensity = res2.intensity  # shape: (npt_azim, npt_rad)
    radial = res2.radial
    azimuthal = res2.azimuthal

    # Adjust azimuthal values
    azimuthal_adjusted = np.where(azimuthal < 0, azimuthal + 180, azimuthal - 180)

    # Sort by adjusted azimuthal angle
    sort_indices = np.argsort(azimuthal_adjusted)
    azimuthal_adjusted_sorted = azimuthal_adjusted[sort_indices]
    intensity_sorted = intensity[sort_indices, :]

    # Restrict azimuthal range to -90 to 90
    mask = (azimuthal_adjusted_sorted > -90) & (azimuthal_adjusted_sorted < 90)
    azimuthal_adjusted_sorted = azimuthal_adjusted_sorted[mask]
    intensity_sorted = intensity_sorted[mask, :]

    peak_data = []

    # Iterate over each region of interest
    for region in regions_of_interest:
        theta_min, theta_max = region['theta_min'], region['theta_max']
        phi_min, phi_max = region['phi_min'], region['phi_max']

        # Filter azimuthal
        mask_az = (azimuthal_adjusted_sorted >= phi_min) & (azimuthal_adjusted_sorted <= phi_max)
        intensity_filtered_azimuth = intensity_sorted[mask_az, :]
        azimuthal_filtered = azimuthal_adjusted_sorted[mask_az]

        # Filter radial
        mask_rad = (radial >= theta_min) & (radial <= theta_max)
        intensity_filtered = intensity_filtered_azimuth[:, mask_rad]
        radial_filtered = radial[mask_rad]

        if intensity_filtered.size == 0:
            continue

        # Sum along azimuthal to get I(2θ)
        intensity_1d = np.sum(intensity_filtered, axis=0)

        # Sum along radial to get I(φ)
        intensity_1d_phi = np.sum(intensity_filtered, axis=1)

        peak_data.append({
            'Region': region['name'],
            'Radial (2θ)': radial_filtered,
            'Intensity': intensity_1d,
            'Azimuthal Angle (φ)': azimuthal_filtered,
            'Azimuthal Intensity': intensity_1d_phi
        })

    return peak_data

def optimization_complete(result, fit_button, progress_label, gamma_var, Gamma_var, chi_var, zs_var, zb_var, update):
    """
    Handles the completion of the optimization process.

    Args:
        result (OptimizeResult): The result of the optimization process.
        fit_button (ttk.Button): The button to re-enable after optimization.
        progress_label (ttk.Label): The label to update with optimization results.
        gamma_var, Gamma_var, chi_var, zs_var, zb_var (tk.DoubleVar): Variables to update with optimized values.
        update (function): Function to update the GUI with new values.
    """
    fit_button.config(state="normal")  # Re-enable the fit button
    if result.success:
        optimal_params = result.x  # [gamma, Gamma, chi, zs, zb]
        gamma_var.set(optimal_params[0])
        Gamma_var.set(optimal_params[1])
        chi_var.set(optimal_params[2])
        zs_var.set(optimal_params[3])
        zb_var.set(optimal_params[4])

        # Update the GUI with new values
        update()

        # Calculate the cost from the residuals
        cost = sum(result.fun ** 2)

        progress_label.config(text=f"Optimization complete.\n"
                                   f"Best Loss: {cost:.2e}\n"
                                   f"Optimal Parameters:\n"
                                   f"Gamma: {optimal_params[0]:.4f}\n"
                                   f"Detector Rotation Gamma: {optimal_params[1]:.4f}\n"
                                   f"Chi: {optimal_params[2]:.4f}\n"
                                   f"Zs: {optimal_params[3]:.6f}\n"
                                   f"Zb: {optimal_params[4]:.6f}")
    else:
        progress_label.config(text="Optimization failed.")
