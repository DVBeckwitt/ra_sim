##########################################
# optimization.py (Local Optimization) - with DEBUG logs
##########################################

import numpy as np
import traceback
from collections import defaultdict
from scipy.optimize import minimize, least_squares
import pandas as pd
import scipy.optimize as opt  # if not already imported

###############################################################################
# DEBUG LOGGING
###############################################################################
def debug_log(msg):
    """Append debug messages to debug.log."""
    with open('debug.log', 'a') as f:
        f.write(msg + '\n')

###############################################################################
# UTILITY FUNCTIONS: build_measured_dict, filter_reflections
###############################################################################

def build_measured_dict(measured_peaks):
    """
    Builds a dictionary mapping Miller indices (h, k, l) to measured peak info.
    Each entry in measured_peaks is expected to have:
      {
        'label': 'h,k,l',
        'x': float,
        'y': float,
        'sigma': float or None
      }
    """
    debug_log("[build_measured_dict] Starting with measured_peaks length = {}".format(len(measured_peaks)))
    measured_dict = {}
    for idx, peak in enumerate(measured_peaks):
        # Expect 'label' like "0,1,5"
        if 'label' not in peak:
            debug_log("  Peak #{} missing 'label'; skipping.".format(idx))
            continue
        label_str = peak['label']
        try:
            hkl = tuple(map(int, label_str.split(',')))
        except Exception as e:
            debug_log("  Peak #{} label='{}' parsing error: {}".format(idx, label_str, e))
            continue

        # Store X,Y
        if hkl not in measured_dict:
            measured_dict[hkl] = []
        measured_dict[hkl].append({
            'x': peak.get('x', np.nan),
            'y': peak.get('y', np.nan),
            'sigma': peak.get('sigma', None)
        })
    debug_log("[build_measured_dict] Finished. Created measured_dict with {} HKL entries.".format(len(measured_dict)))
    return measured_dict


def filter_reflections(miller, intensities, measured_dict):
    """
    Keep only those Miller indices from 'miller' that appear in 'measured_dict'.
    Return (miller_sub, intensities_sub) as arrays.
    """
    debug_log("[filter_reflections] Attempting to match miller with measured_dict keys.")
    miller_sub = []
    intensities_sub = []
    for i, (H, K, L) in enumerate(miller):
        if (H, K, L) in measured_dict:
            miller_sub.append((H, K, L))
            intensities_sub.append(intensities[i])

    debug_log("  miller_sub length after filtering = {}".format(len(miller_sub)))

    if len(miller_sub) == 0:
        debug_log("  WARNING: No overlapping (H,K,L) => cost function will be 1e9.")
        return np.array([]), np.array([])
    miller_sub = np.array(miller_sub, dtype=int)
    intensities_sub = np.array(intensities_sub, dtype=float)
    return miller_sub, intensities_sub


###############################################################################
# LOCAL SIMULATION FUNCTION (geometry-based)
###############################################################################

def simulate_diffraction_positions(
    theta_i,
    gamma,
    Gamma,
    dist,
    zs,
    zb,
    chi,
    mosaic_sigma,
    mosaic_gamma,
    mosaic_eta,
    miller,
    intensities,
    image_size,
    a_val,
    c_val,
    lambda_,
    psi,
    n2,
    center,
    num_samples=5000,
    divergence_sigma=0.0,
    bw_sigma=0.0
):
    """
    A helper to simulate the diffraction pattern for geometry-fitting.
    """
    from ra_sim.simulation.mosaic_profiles import generate_random_profiles
    from ra_sim.simulation.diffraction import process_peaks_parallel

    # Convert mosaic angles from deg to rad
    sigma_mosaic_rad = np.radians(mosaic_sigma)
    gamma_mosaic_rad = np.radians(mosaic_gamma)

    # Generate random mosaic + divergence + beam profiles
    (beam_x_array, beam_y_array, beam_intensity_array,
     beta_array, kappa_array, mosaic_intensity_array,
     theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
         num_samples,
         divergence_sigma,
         bw_sigma,
         sigma_mosaic_rad,
         gamma_mosaic_rad,
         mosaic_eta
    )

    empty_image = np.zeros((image_size, image_size), dtype=np.float64)

    try:
        # This returns (image, max_positions)
        simulated_image, max_positions = process_peaks_parallel(
            miller, intensities,
            image_size,
            a_val, c_val, lambda_,
            empty_image,
            dist,
            gamma,
            Gamma,
            chi,
            psi,
            zs,
            zb,
            n2,
            beam_x_array,
            beam_y_array,
            beam_intensity_array,
            beta_array,
            kappa_array,
            mosaic_intensity_array,
            theta_array,
            phi_array,
            divergence_intensity_array,
            debye_x=0.0,
            debye_y=0.0,
            center=center,
            theta_initial=theta_i,
            theta_final=(theta_i + 0.1),
            step=0.1,
            unit_x=np.array([1.0, 0.0, 0.0]),
            n_detector=np.array([0.0, 1.0, 0.0])
        )
    except Exception as e:
        # If there's any numerical error, log and re-raise
        debug_log(f"[simulate_diffraction_positions] Exception in process_peaks_parallel: {e}")
        raise

    return simulated_image, max_positions


###############################################################################
# COST FUNCTION: geometry, returning a single scalar
###############################################################################

def compute_peak_position_error_geometry_local(
    gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val, chi_val,
    mosaic_sigma_const, mosaic_gamma_const, mosaic_eta_const,
    miller, intensities, image_size, a_val, c_val, lambda_, psi, n2,
    center,
    measured_peaks
):
    """
    Cost = sum of squared distances to measured peaks, or 1e9 if no overlap/fail.
    """
    from math import sqrt, isnan

    debug_log(f"compute_peak_position_error_geometry_local called with gamma={gamma_val:.4f}, "
              f"Gamma={Gamma_val:.4f}, dist={dist_val:.4f}, theta_i={theta_i_val:.4f}, "
              f"zs={zs_val:.6g}, zb={zb_val:.6g}, chi={chi_val:.4f}, a={a_val:.4f}, c={c_val:.4f}, "
              f"center=({center[0]:.2f}, {center[1]:.2f})")

    # 1) Build measured dict
    measured_dict = build_measured_dict(measured_peaks)
    # 2) Filter reflections
    miller_sub, intensities_sub = filter_reflections(miller, intensities, measured_dict)
    if len(miller_sub) == 0:
        debug_log("  => No valid reflections => cost=1e9")
        return 1e9

    # 3) Simulate
    try:
        _, max_positions = simulate_diffraction_positions(
            theta_i=theta_i_val,
            gamma=gamma_val,
            Gamma=Gamma_val,
            dist=dist_val,
            zs=zs_val,
            zb=zb_val,
            chi=chi_val,
            mosaic_sigma=mosaic_sigma_const,
            mosaic_gamma=mosaic_gamma_const,
            mosaic_eta=mosaic_eta_const,
            miller=miller_sub,
            intensities=intensities_sub,
            image_size=image_size,
            a_val=a_val,
            c_val=c_val,
            lambda_=lambda_,
            psi=psi,
            n2=n2,
            center=center
        )
    except Exception as e:
        debug_log(f"  => Simulation failed with error: {e} => cost=1e9")
        return 1e9

    sum_weighted_sq = 0.0
    total_points = 0

    # 4) Evaluate cost
    for i, (H, K, L) in enumerate(miller_sub):
        mx0, my0, mv0, mx1, my1, mv1 = max_positions[i]
        meas_list = measured_dict[(H, K, L)]
        for meas in meas_list:
            x_m = meas['x']
            y_m = meas['y']

            # Weight = 1/(y_m^2) fallback if y_m=0
            if abs(y_m) < 1e-12:
                w = 1.0
            else:
                w = 1.0 / (y_m**2)

            dx0 = mx0 - x_m if not isnan(mx0) else 1e6
            dy0 = my0 - y_m if not isnan(my0) else 1e6
            dist_sq_0 = dx0*dx0 + dy0*dy0

            dx1 = mx1 - x_m if not isnan(mx1) else 1e6
            dy1 = my1 - y_m if not isnan(my1) else 1e6
            dist_sq_1 = dx1*dx1 + dy1*dy1

            best_dist_sq = dist_sq_0 if dist_sq_0 < dist_sq_1 else dist_sq_1
            sum_weighted_sq += best_dist_sq * w
            total_points += 1

    if total_points == 0:
        debug_log("  => total_points=0 => cost=1e9")
        return 1e9

    cost_val = sum_weighted_sq / float(total_points)
    debug_log(f"  => Computed cost: {cost_val:.6g} for total_points={total_points}")
    return cost_val


###############################################################################
# LEVENBERG-MARQUARDT Residual Function (geometry)
###############################################################################

def geometry_residuals_local(
    params,
    mosaic_sigma_const,
    mosaic_gamma_const,
    mosaic_eta_const,
    miller,
    intensities,
    image_size,
    lambda_,
    psi,
    n2,
    measured_peaks
):
    """
    Return a 1D array of residuals => the per-peak XY distance.
    """
    import math

    (gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val,
     chi_val, a_val, c_val, cx_val, cy_val) = params

    measured_dict = build_measured_dict(measured_peaks)
    miller_sub, intensities_sub = filter_reflections(miller, intensities, measured_dict)
    if len(miller_sub) == 0:
        # Return a single big residual
        return np.array([1e6])

    try:
        _, max_positions = simulate_diffraction_positions(
            theta_i=theta_i_val,
            gamma=gamma_val,
            Gamma=Gamma_val,
            dist=dist_val,
            zs=zs_val,
            zb=zb_val,
            chi=chi_val,
            mosaic_sigma=mosaic_sigma_const,
            mosaic_gamma=mosaic_gamma_const,
            mosaic_eta=mosaic_eta_const,
            miller=miller_sub,
            intensities=intensities_sub,
            image_size=image_size,
            a_val=a_val,
            c_val=c_val,
            lambda_=lambda_,
            psi=psi,
            n2=n2,
            center=(cx_val, cy_val)
        )
    except Exception as e:
        debug_log(f"[geometry_residuals_local] Simulation error => returning large residual. {e}")
        return np.array([1e6])

    residuals = []
    for i, (H, K, L) in enumerate(miller_sub):
        mx0, my0, mv0, mx1, my1, mv1 = max_positions[i]
        meas_list = measured_dict[(H, K, L)]
        for meas in meas_list:
            x_m = meas['x']
            y_m = meas['y']

            dist_sq_0 = (mx0 - x_m)**2 + (my0 - y_m)**2 if not math.isnan(mx0) else 1e12
            dist_sq_1 = (mx1 - x_m)**2 + (my1 - y_m)**2 if not math.isnan(mx1) else 1e12
            best_dist = np.sqrt(min(dist_sq_0, dist_sq_1))
            residuals.append(best_dist)

    if len(residuals) == 0:
        return np.array([1e6])
    return np.array(residuals)


###############################################################################
# LOCAL GEOMETRY OPTIMIZATION
###############################################################################
def run_optimization_positions_geometry_local(
    fit_button,
    progress_label,
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    psi,
    n2,
    center,  # (cx, cy)
    measured_peaks,
    mosaic_sigma_const,
    mosaic_gamma_const,
    mosaic_eta_const,
    gamma_var,
    Gamma_var,
    dist_var,
    theta_i_var,
    zs_var,
    zb_var,
    chi_var,
    a_var,
    c_var,
    center_x_var,
    center_y_var,
    update_gui
):
    import traceback
    from scipy.optimize import minimize, differential_evolution, least_squares

    try:
        fit_button.config(state='disabled')
        progress_label.config(text="Local geometry optimization in progress...")

        debug_log("\n[run_optimization_positions_geometry_local] Called.")
        debug_log(f"Initial guess: gamma={gamma_var.get()}, Gamma={Gamma_var.get()}, "
                  f"dist={dist_var.get()}, theta_i={theta_i_var.get()}, "
                  f"zs={zs_var.get()}, zb={zb_var.get()}, chi={chi_var.get()}, a={a_var.get()}, c={c_var.get()}, "
                  f"center=({center_x_var.get()}, {center_y_var.get()})")

        # STEP 0: Define the starting point and bounds.
        x0 = [
            gamma_var.get(),
            Gamma_var.get(),
            dist_var.get(),
            theta_i_var.get(),
            zs_var.get(),
            zb_var.get(),
            chi_var.get(),
            a_var.get(),
            c_var.get(),
            center_x_var.get(),
            center_y_var.get()
        ]

        bounds = [
            (-4.0, 4.0),    # gamma
            (-4.0, 4.0),    # Gamma
            (72e-3, 78e-3), # dist
            (5.0, 7.0),     # theta_i
            (-2e-3, 2e-3),  # zs
            (-2e-3, 2e-3),  # zb
            (-0.1, 0.1),    # chi
            (4.0, 4.5),     # a
            (27.0, 29.0),   # c
            (center[0], center[0]),  # center x (fixed)
            (center[1], center[1])   # center y (fixed)
        ]

        # Define the cost function (wrapper around compute_peak_position_error_geometry_local)
        def cost_function(x):
            (gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val,
             chi_val, a_val, c_val, cx_val, cy_val) = x

            cst = compute_peak_position_error_geometry_local(
                gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val, chi_val,
                mosaic_sigma_const, mosaic_gamma_const, mosaic_eta_const,
                miller, intensities, image_size, a_val, c_val, lambda_, psi, n2,
                center=(cx_val, cy_val),
                measured_peaks=measured_peaks
            )
            return cst

        # STEP 1: Global search using differential evolution.
        debug_log("[run_optimization_positions_geometry_local] Starting global optimization (Differential Evolution).")
        global_result = differential_evolution(
            cost_function,
            bounds,
            strategy='best1bin',
            maxiter=200,
            popsize=15,
            disp=False
        )
        debug_log(f"  Differential Evolution result: success={global_result.success}, message='{global_result.message}'")
        # Use global_result.x as the new starting point if successful.
        if global_result.success:
            x0_global = global_result.x
        else:
            x0_global = x0

        # STEP 2: Local optimization using L-BFGS-B.
        debug_log("[run_optimization_positions_geometry_local] Starting local optimization (L-BFGS-B) from global optimum.")

        # Define the minimizer kwargs (still using L-BFGS-B)
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {"maxiter": 1000}
        }

        # Use basin hopping to perturb around x0.
        res_basinhopping = opt.basinhopping(
            cost_function,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            stepsize=0.1,   # small step size to stay local
            niter=50        # number of iterations for the hopping
        )

        # Get the best result from basin hopping.
        res_lbfgs = res_basinhopping.lowest_optimization_result

        debug_log(f"  L-BFGS-B result: success={res_lbfgs.success}, message='{res_lbfgs.message}'")
        debug_log(f"  L-BFGS-B final cost={res_lbfgs.fun:.6g}, final x={res_lbfgs.x}")

        if not res_lbfgs.success:
            progress_label.config(text=f"[Geometry Fit] L-BFGS-B failed: {res_lbfgs.message}")
            best_params = x0
        else:
            best_params = res_lbfgs.x
            # Update the GUI variables
            gamma_var.set(best_params[0])
            Gamma_var.set(best_params[1])
            dist_var.set(best_params[2])
            theta_i_var.set(best_params[3])
            zs_var.set(best_params[4])
            zb_var.set(best_params[5])
            chi_var.set(best_params[6])
            a_var.set(best_params[7])
            c_var.set(best_params[8])
            center_x_var.set(best_params[9])
            center_y_var.set(best_params[10])
            update_gui()

        msg_lbfgs = (
            "[Geometry Fit] L-BFGS-B done.\n"
            f"Status={res_lbfgs.message}, success={res_lbfgs.success}\n"
            f"Best Loss (L-BFGS-B) = {res_lbfgs.fun:.6g}"
        )

        # STEP 3: Further refinement with Levenberg–Marquardt.
        debug_log("[run_optimization_positions_geometry_local] Starting Levenberg–Marquardt from L-BFGS-B result.")
        def marquardt_residuals(x):
            return geometry_residuals_local(
                x,
                mosaic_sigma_const,
                mosaic_gamma_const,
                mosaic_eta_const,
                miller,
                intensities,
                image_size,
                lambda_,
                psi,
                n2,
                measured_peaks
            )

        res_marq = least_squares(
            marquardt_residuals,
            best_params,
            method='dogbox'
        )

        if not res_marq.success:
            msg_mq = f"[Marquardt] Not success: {res_marq.message}"
            debug_log(f"  Marquardt => Not success. {res_marq.message}")
        else:
            marq_params = res_marq.x
            # Update the GUI with refined parameters
            gamma_var.set(marq_params[0])
            Gamma_var.set(marq_params[1])
            dist_var.set(marq_params[2])
            theta_i_var.set(marq_params[3])
            zs_var.set(marq_params[4])
            zb_var.set(marq_params[5])
            chi_var.set(marq_params[6])
            a_var.set(marq_params[7])
            c_var.set(marq_params[8])
            center_x_var.set(marq_params[9])
            center_y_var.set(marq_params[10])
            update_gui()

            # Compute final cost
            final_residuals = marquardt_residuals(marq_params)
            final_cost = 0.5 * np.sum(final_residuals**2)
            msg_mq = (
                "[Marquardt] Done.\n"
                f"Marquardt final cost = {final_cost:.6g}\n"
                f"Status=Success, message={res_marq.message}"
            )
            debug_log(f"  Marquardt => success. Final param={marq_params} final cost={final_cost:.6g}")

        progress_label.config(text=msg_lbfgs + "\n" + msg_mq)

    except Exception as e:
        import traceback
        tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        progress_label.config(text=f"Geometry optimization failed:\n{tb_str}")
        debug_log(f"[run_optimization_positions_geometry_local] EXCEPTION: {tb_str}")
    finally:
        fit_button.config(state='normal')



def mosaic_1d_cost_function(
    mosaic_sigma_val,
    mosaic_gamma_val,
    mosaic_eta_val,
    # fixed geometry
    gamma_const,
    Gamma_const,
    dist_const,
    theta_i_const,
    zs_const,
    zb_const,
    chi_const,
    a_const,
    c_const,
    center_xy,
    # global settings
    miller,
    intensities,
    image_size,
    lambda_,
    psi,
    n2,
    # 1D integration sliders
    rmin,
    rmax,
    amin_deg,
    amax_deg,
    # background image
    background_image,
    # mosaic profiles cache or generation parameters
    num_samples,
    divergence_sigma,
    bw_sigma,
    # JIT integration methods
    radial_integration_func,
    azimuthal_integration_func
):
    """
    Builds a simulated 2D pattern using *fixed* geometry and the
    given mosaic parameters (sigma, gamma, eta). Then integrates
    that pattern in the radial & azimuthal range indicated by
    [rmin, rmax] and [amin_deg, amax_deg].
    
    Compares with the same integrated slice of background_image
    and returns a scalar cost (sum of squared differences).
    """
    import numpy as np
    from ra_sim.simulation.mosaic_profiles import generate_random_profiles
    from ra_sim.simulation.diffraction import process_peaks_parallel

    # 1) Generate the mosaic + beam profiles
    (beam_x_array, beam_y_array, beam_intensity_array,
     beta_array, kappa_array, mosaic_intensity_array,
     theta_array, phi_array, divergence_intensity_array) = generate_random_profiles(
         num_samples=num_samples,
         divergence_sigma=divergence_sigma,
         bw_sigma=bw_sigma,
         sigma_mosaic_deg=mosaic_sigma_val,
         gamma_mosaic_deg=mosaic_gamma_val,
         eta=mosaic_eta_val
     )

    # 2) Simulate 2D diffraction image (fixed geometry)
    simulated_image = np.zeros((image_size, image_size), dtype=np.float64)
    try:
        process_peaks_parallel(
            miller,
            intensities,
            image_size,
            a_const,
            c_const,
            lambda_,
            simulated_image,          # fill in
            dist_const,
            gamma_const,
            Gamma_const,
            chi_const,
            psi,
            zs_const,
            zb_const,
            n2,
            beam_x_array,
            beam_y_array,
            beam_intensity_array,
            beta_array,
            kappa_array,
            mosaic_intensity_array,
            theta_array,
            phi_array,
            divergence_intensity_array,
            debye_x=0.0,      # or your slider if you prefer
            debye_y=0.0,
            center=center_xy,
            theta_initial=theta_i_const,
            theta_final=theta_i_const + 0.1,
            step=0.1,
            unit_x=np.array([1.0, 0.0, 0.0]),
            n_detector=np.array([0.0, 1.0, 0.0])
        )
    except Exception:
        # On error, return a large cost
        return 1e9

    # 3) Get radial/azimuthal integration of simulated image
    amin_rad = np.deg2rad(amin_deg)
    amax_rad = np.deg2rad(amax_deg)

    # Radial integration
    sim_r, sim_radial = radial_integration_func(
        simulated_image,
        center_xy[0],
        center_xy[1],
        rmin,
        rmax,
        amin_rad,
        amax_rad
    )
    # Azimuthal integration
    sim_ang, sim_az = azimuthal_integration_func(
        simulated_image,
        center_xy[0],
        center_xy[1],
        rmin,
        rmax,
        amin_rad,
        amax_rad,
        nbins=360  # or 1000, etc.
    )

    # 4) Get radial/azimuthal integration of background image (same range)
    if background_image is None or background_image.shape != simulated_image.shape:
        # If no valid background, large cost
        return 1e9

    bg_r, bg_radial = radial_integration_func(
        background_image,
        center_xy[0],
        center_xy[1],
        rmin,
        rmax,
        amin_rad,
        amax_rad
    )
    bg_ang, bg_az = azimuthal_integration_func(
        background_image,
        center_xy[0],
        center_xy[1],
        rmin,
        rmax,
        amin_rad,
        amax_rad,
        nbins=360
    )

    # 5) Match up arrays for difference (naive approach: assume same bin edges)
    #    If their shapes differ, you should handle interpolation or smaller arrays
    if len(sim_radial) != len(bg_radial) or len(sim_az) != len(bg_az):
        return 1e9  # or handle differently

    # 6) Compute cost: sum of squared differences (radial + az)
    #    You can weight them differently or do just radial, etc.
    #    This example uses a simple combined sum of squares.
    diff_radial = sim_radial - bg_radial
    diff_az = sim_az - bg_az

    cost_val = np.sum(diff_radial**2) + np.sum(diff_az**2)
    return cost_val

def run_optimization_mosaic_1d_local(
    fit_button,
    progress_label,
    # fixed geometry
    gamma_const,
    Gamma_const,
    dist_const,
    theta_i_const,
    zs_const,
    zb_const,
    chi_const,
    a_const,
    c_const,
    center_xy,
    # global
    miller,
    intensities,
    image_size,
    lambda_,
    psi,
    n2,
    # radial/azimuthal sliders
    rmin,
    rmax,
    amin_deg,
    amax_deg,
    # background
    background_image,
    # mosaic variables (tkinter Var)
    mosaic_sigma_var,
    mosaic_gamma_var,
    mosaic_eta_var,
    # other
    num_samples,
    divergence_sigma,
    bw_sigma,
    radial_integration_func,
    azimuthal_integration_func,
    update_gui
):
    """
    Similar to your geometry fits: fix geometry, then do a local optimization
    of mosaic_sigma, mosaic_gamma, and eta by matching the 1D integrated 
    simulation to the 1D integrated region of background_image.
    """
    try:
        fit_button.config(state='disabled')
        progress_label.config(text="1D Mosaic Optimization in progress...")

        import numpy as np
        from scipy.optimize import minimize

        def cost_wrapper(x):
            # x = [sigma_mosaic, gamma_mosaic, eta]
            return mosaic_1d_cost_function(
                mosaic_sigma_val=x[0],
                mosaic_gamma_val=x[1],
                mosaic_eta_val=x[2],
                gamma_const=gamma_const,
                Gamma_const=Gamma_const,
                dist_const=dist_const,
                theta_i_const=theta_i_const,
                zs_const=zs_const,
                zb_const=zb_const,
                chi_const=chi_const,
                a_const=a_const,
                c_const=c_const,
                center_xy=center_xy,
                miller=miller,
                intensities=intensities,
                image_size=image_size,
                lambda_=lambda_,
                psi=psi,
                n2=n2,
                rmin=rmin,
                rmax=rmax,
                amin_deg=amin_deg,
                amax_deg=amax_deg,
                background_image=background_image,
                num_samples=num_samples,
                divergence_sigma=divergence_sigma,
                bw_sigma=bw_sigma,
                radial_integration_func=radial_integration_func,
                azimuthal_integration_func=azimuthal_integration_func
            )
        
        # Get starting guess from the sliders
        x0 = [mosaic_sigma_var.get(), mosaic_gamma_var.get(), mosaic_eta_var.get()]

        # Bounds: you can tweak these
        bounds = [(0.0, 5.0),  # sigma deg
                  (0.0, 5.0),  # gamma deg
                  (0.0, 1.0)]  # eta fraction

        res = minimize(cost_wrapper, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200})
        
        if res.success:
            best_sigma, best_gamma, best_eta = res.x
            mosaic_sigma_var.set(best_sigma)
            mosaic_gamma_var.set(best_gamma)
            mosaic_eta_var.set(best_eta)
            update_gui()
            msg = (f"1D Mosaic Optimization DONE\n"
                   f"status={res.message}, success={res.success}\n"
                   f"sigma={best_sigma:.3f}, gamma={best_gamma:.3f}, eta={best_eta:.3f}\n"
                   f"final cost={res.fun:.5g}")
        else:
            msg = f"1D Mosaic Optimization FAILED: {res.message}"
        
        progress_label.config(text=msg)

    except Exception as e:
        import traceback
        tb_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        progress_label.config(text=f"[1D Mosaic Fit] Exception:\n{tb_str}")
    finally:
        fit_button.config(state='normal')
