##########################################
# optimization.py
##########################################

import numpy as np
import traceback
from bayes_opt import BayesianOptimization
from collections import defaultdict
from bayes_opt.acquisition import ExpectedImprovement



# ------------------------------------------------------------------------
# 1) Utility functions:
#    - build_measured_dict: convert measured_peaks list into a dict
#    - filter_reflections : keep only the Miller indices you actually measured
# ------------------------------------------------------------------------
# File: ra_sim/fitting/optimization.py

def build_measured_dict(measured_peaks):
    """
    Builds a dictionary mapping Miller indices (h, k, l) to a list of measured peaks.

    Parameters:
    - measured_peaks: Iterable of dictionaries, each containing:
        - 'label': str, format 'h,k,l'
        - 'x': float, x-coordinate
        - 'y': float, y-coordinate
        - 'sigma': float, uncertainty or other parameter

    Returns:
    - measured_dict: Dict with keys as (h, k, l) tuples and values as lists of peak dictionaries.
    """
    measured_dict = {}
    for idx, peak in enumerate(measured_peaks):
        try:
            # Extract the 'label' and parse it into h, k, l
            label_str = peak['label']  # e.g., '0,1,5'
            hkl = tuple(map(int, label_str.split(',')))  # Converts to (0, 1, 5)
        except KeyError:
            print(f">>> KeyError: 'label' key missing in peak {idx}: {peak}")
            continue  # Skip this peak
        except ValueError:
            print(f">>> ValueError: Unable to parse 'label' in peak {idx}: {peak}")
            continue  # Skip this peak

        # Initialize the list for this hkl if not already present
        if hkl not in measured_dict:
            measured_dict[hkl] = []

        # Append the peak information (you can include 'sigma' if needed)
        measured_dict[hkl].append({
            'x': peak['x'],
            'y': peak['y'],
            'sigma': peak['sigma']
        })

    return measured_dict


def filter_reflections(miller, intensities, measured_dict):
    """
    Keep only those Miller indices from 'miller' that appear in 'measured_dict'.
    Return (miller_sub, intensities_sub) as arrays.

    miller : array of shape (N,3) -> each row is (H,K,L)
    intensities: length N
    measured_dict: dict keyed by (H,K,L)
    """
    miller_sub = []
    intensities_sub = []
    for i, (H, K, L) in enumerate(miller):
        if (H, K, L) in measured_dict:
            miller_sub.append((H, K, L))
            intensities_sub.append(intensities[i])

    if len(miller_sub) == 0:
        return np.array([]), np.array([])

    miller_sub = np.array(miller_sub, dtype=int)
    intensities_sub = np.array(intensities_sub, dtype=float)
    return miller_sub, intensities_sub

# ------------------------------------------------------------------------
# 2) Shared simulation function (geometry + mosaic)
#    This is used in both geometry and mosaic optimization.
# ------------------------------------------------------------------------
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
    num_samples=1000,
    divergence_sigma=0.0,
    bw_sigma=0.0
):
    """
    Simulates a 2D diffraction pattern for the given (miller, intensities),
    using geometry + mosaic parameters.

    Returns:
      (simulated_image, max_positions), where:
        simulated_image -> 2D array of shape (image_size, image_size)
        max_positions   -> array of shape (len(miller), 6), with
                           (mx0, my0, mv0, mx1, my1, mv1) for each reflection
    """
    from ra_sim.simulation.mosaic_profiles import generate_random_profiles
    from ra_sim.simulation.diffraction import process_peaks_parallel

    # Convert mosaic angles from deg to rad
    sigma_mosaic_rad = np.radians(mosaic_sigma)
    gamma_mosaic_rad = np.radians(mosaic_gamma)

    # Generate random beam/mosaic/divergence profiles
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

    # Prepare an empty image
    empty_image = np.zeros((image_size, image_size), dtype=np.float64)

    # Run diffraction simulation
    simulated_image, max_positions = process_peaks_parallel(
        miller, intensities,
        image_size,
        a_val, c_val, lambda_,
        empty_image,
        dist,
        gamma,
        Gamma,
        chi,  # sample tilt
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
        # Debye approximations:
        debye_x=0.0,
        debye_y=0.0,
        center=center,
        theta_initial=theta_i,
        theta_final=(theta_i + 0.5),  # small range
        step=0.1,
        unit_x=np.array([1.0, 0.0, 0.0]),
        n_detector=np.array([0.0, 1.0, 0.0])
    )

    return simulated_image, max_positions

# ------------------------------------------------------------------------
# 3) Geometry-Only Optimization (Mosaic is constant)
# ------------------------------------------------------------------------
import os
import os
import os
import numpy as np
def compute_peak_position_error_geometry(
    # geometry parameters
    gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val, chi_val,
    # mosaic (constant)
    mosaic_sigma_const, mosaic_gamma_const, mosaic_eta_const,
    # data
    miller,
    intensities,
    image_size,
    a_val,
    c_val,
    lambda_,
    psi,
    n2,
    center,
    measured_peaks,  # a list of dicts with 'label': 'h,k,l', 'x':..., 'y':..., 'sigma':...
    # debugging
    debug=True,
    debug_log_path=r"C:\Users\Kenpo\Downloads\geometry_debug.txt"
):
    """
    Steps:
      1) Build a dictionary of measured reflections (H,K,L) -> list of measured peaks.
      2) Filter (miller, intensities) so we only keep measured reflections.
      3) Simulate diffraction for that subset.
      4) For each reflection row i in max_positions, we have up to two sign solutions:
         (mx0, my0) and (mx1, my1). For each valid sign solution, find the measured point
         in the same reflection that yields the smallest pixel distance^2, store that error.
      5) Return the average distance^2 over all valid sign solutions.

    If debug=True, writes detailed logs to debug_log_path.
    """
    debug = True
    if debug:
        print(">>> Entered compute_peak_position_error_geometry!")

    debug_lines = []

    # ------------------------------------------------
    # 1) Build measured_dict from measured_peaks
    # ------------------------------------------------
    if debug:
        print(">>> Step 1: Attempting to build_measured_dict(measured_peaks)...")
    try:
        if len(measured_peaks) > 0 and debug:
            print(">>> First few measured_peaks entries:", measured_peaks[:3])
        elif len(measured_peaks) == 0 and debug:
            print(">>> measured_peaks is empty.")

        measured_dict = build_measured_dict(measured_peaks)  # parse 'label' -> (H,K,L)
        if debug:
            print(f">>> measured_dict built. length: {len(measured_dict)}")
    except Exception as e:
        if debug:
            print(f">>> EXCEPTION building measured_dict: {e}")
        return 1e9  # Can't proceed, return large penalty

    # ------------------------------------------------
    # 2) Filter reflections
    # ------------------------------------------------
    if debug:
        print(">>> Step 2: Attempting filter_reflections(...)")
    try:
        miller_sub, intensities_sub = filter_reflections(miller, intensities, measured_dict)
        if debug:
            print(f">>> Filtered reflections count: {len(miller_sub)}")

        if len(miller_sub) == 0:
            if debug:
                print(">>> Early return: no measured reflections => cost=1e9")
            return 1e9
    except Exception as e:
        if debug:
            print(f">>> EXCEPTION filtering reflections: {e}")
        return 1e9

    # ------------------------------------------------
    # 3) Simulate the diffraction for these reflections
    # ------------------------------------------------
    if debug:
        print(">>> Step 3: Attempting simulate_diffraction_positions(...)")
    try:
        _, max_positions = simulate_diffraction_positions(
            theta_i_val,
            gamma_val,
            Gamma_val,
            dist_val,
            zs_val,
            zb_val,
            chi_val,
            mosaic_sigma_const,
            mosaic_gamma_const,
            mosaic_eta_const,
            miller_sub,
            intensities_sub,
            image_size,
            a_val,
            c_val,
            lambda_,
            psi,
            n2,
            center
        )
        if debug:
            print(">>> Simulation done. About to compute errors.")
    except Exception as e:
        if debug:
            print(f">>> EXCEPTION in simulate_diffraction_positions: {e}")
        return 1e9

    # ------------------------------------------------
    # 4) For each reflection, handle up to two sign solutions
    # ------------------------------------------------
    all_errors = []
    if debug:
        print(">>> Step 4: Matching each sign solution to the closest measured peak.")
        debug_lines.append("===== Debug Log for Geometry Cost Function =====")
        debug_lines.append(
            f"gamma={gamma_val}, Gamma={Gamma_val}, dist={dist_val}, theta_i={theta_i_val}, "
            f"zs={zs_val}, zb={zb_val}, chi={chi_val}, "
            f"mosaic_sigma={mosaic_sigma_const}, mosaic_gamma={mosaic_gamma_const}, mosaic_eta={mosaic_eta_const}"
        )
        debug_lines.append(f"Filtered reflections count: {len(miller_sub)}\n")

    import math
    for i, (H, K, L) in enumerate(miller_sub):
        try:
            mx0, my0, mv0, mx1, my1, mv1 = max_positions[i]
        except IndexError:
            if debug:
                print(f">>> IndexError: max_positions does not have entry {i}")
            continue

        # Retrieve measured data for reflection (H,K,L)
        meas_list = measured_dict.get((H, K, L), [])

        if debug:
            debug_lines.append(f"** Reflection (H,K,L)=({H},{K},{L}) **")
            debug_lines.append(f"  #Measured peaks for this reflection: {len(meas_list)}")

        # If no measured peaks => can't compute error => large penalty for each valid sign
        if len(meas_list) == 0:
            if debug:
                debug_lines.append("  -> No measured peaks => large penalty for each valid sign.\n")

            # If sign0 is valid
            if not math.isnan(mx0) and not math.isnan(my0):
                all_errors.append(1e9)
            # If sign1 is valid
            if not math.isnan(mx1) and not math.isnan(my1):
                all_errors.append(1e9)
            continue  # Move to next reflection

        # sign0 => if valid, find the best measured peak
        if not math.isnan(mx0) and not math.isnan(my0):
            best_err_0 = 1e15
            best_pair_0 = None
            for meas in meas_list:
                x_meas, y_meas = meas['x'], meas['y']
                d_sq = (mx0 - x_meas)**2 + (my0 - y_meas)**2
                if d_sq < best_err_0:
                    best_err_0 = d_sq
                    best_pair_0 = (x_meas, y_meas, d_sq)
            all_errors.append(best_err_0)

            if debug and best_pair_0 is not None:
                (x_meas, y_meas, d_sq) = best_pair_0
                debug_lines.append(
                    f"  sign0=({mx0:.1f},{my0:.1f}), bestMeas=({x_meas:.1f},{y_meas:.1f}), dist^2={d_sq:.2f}"
                )

        # sign1 => if valid, find the best measured peak
        if not math.isnan(mx1) and not math.isnan(my1):
            best_err_1 = 1e15
            best_pair_1 = None
            for meas in meas_list:
                x_meas, y_meas = meas['x'], meas['y']
                d_sq = (mx1 - x_meas)**2 + (my1 - y_meas)**2
                if d_sq < best_err_1:
                    best_err_1 = d_sq
                    best_pair_1 = (x_meas, y_meas, d_sq)
            all_errors.append(best_err_1)

            if debug and best_pair_1 is not None:
                (x_meas, y_meas, d_sq) = best_pair_1
                debug_lines.append(
                    f"  sign1=({mx1:.1f},{my1:.1f}), bestMeas=({x_meas:.1f},{y_meas:.1f}), dist^2={d_sq:.2f}"
                )

        if debug:
            debug_lines.append("")  # blank line after reflection

    # If no sign solutions at all => cost=1e9
    if len(all_errors) == 0:
        cost = 1e9
        if debug:
            debug_lines.append("No valid sign solutions => cost=1e9.\n")
    else:
        cost = np.mean(all_errors)

    if debug:
        print(f">>> Done computing errors. cost={cost:.4f}")

    # ------------------------------------------------
    # 5) If debug, write the debug file
    # ------------------------------------------------
    if debug:
        debug_lines.append(f"Final cost (mean dist^2) = {cost:.4f}\n")

        print("-------- DEBUG FILE WRITE ATTEMPT --------")
        print(f"  Writing debug info to {debug_log_path} ...")
        try:
            with open(debug_log_path, "w", encoding="utf-8") as f:
                for line in debug_lines:
                    f.write(line + "\n")
            print(f"  SUCCESS: Debug log written to {debug_log_path}")
        except Exception as e:
            print(f"  [ERROR] Could not write debug file at {debug_log_path}: {e}")
        print("-------- END DEBUG FILE WRITE ATTEMPT --------\n")

        print(">>> compute_peak_position_error_geometry finished. Returning cost.")

    return cost

def objective_function_bayesian_geometry(
    gamma_val,
    Gamma_val,
    dist_val,
    theta_i_val,
    zs_val,
    zb_val,
    chi_val,
    # fixed mosaic
    mosaic_sigma_const,
    mosaic_gamma_const,
    mosaic_eta_const,
    miller,
    intensities,
    image_size,
    a_val,
    c_val,
    lambda_,
    psi,
    n2,
    center,
    measured_peaks
):
    """
    Returns -cost for BayesianOptimization (which is a maximizer).
    """
    try:
        cost = compute_peak_position_error_geometry(
            gamma_val,
            Gamma_val,
            dist_val,
            theta_i_val,
            zs_val,
            zb_val,
            chi_val,
            mosaic_sigma_const,
            mosaic_gamma_const,
            mosaic_eta_const,
            miller,
            intensities,
            image_size,
            a_val,
            c_val,
            lambda_,
            psi,
            n2,
            center,
            measured_peaks
        )
        return -cost
    except:
        return -1e9
from bayes_opt import BayesianOptimization

def run_optimization_positions_geometry(
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
    center,
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
    update_gui
):
    """
     Bayesian optimization for geometry (gamma, Gamma, dist, etc.),
    with mosaic parameters fixed.
    """
    try:
        fit_button.config(state='disabled')
        progress_label.config(text="2D Peak (Geometry) Optimization in progress...")

        # We import here to keep function-based scope tidy
        from bayes_opt import BayesianOptimization

        pbounds = {
            'gamma_val': (-2.0, 2.0),
            'Gamma_val': (-2.0, 2.0),
            'dist_val': (70e-3, 80e-3),
            'theta_i_val': (4.0, 7.0),
            'zs_val': (0, 2e-3),
            'zb_val': (0, 2e-3),
            'chi_val': (-1.0, 1.0),
            'a_val': (4, 5.0),  #  range in Å
            'c_val': (27.0, 30.0)
        }


        # 2) Define your objective function in terms of the geometry parameters
        #    and the function that calculates cost (objective_function_bayesian_geometry).
        #    We keep mosaic parameters fixed from the GUI.
        def objective_function(
            gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val, chi_val, a_val, c_val 

        ):
            cost = objective_function_bayesian_geometry(
                gamma_val,
                Gamma_val,
                dist_val,
                theta_i_val,
                zs_val,
                zb_val,
                chi_val,
                mosaic_sigma_const,
                mosaic_gamma_const,
                mosaic_eta_const,
                miller,
                intensities,
                image_size,
                a_val,
                c_val,
                lambda_,
                psi,
                n2,
                center,
                measured_peaks
            )

            return cost 
        
        acquisition_function = ExpectedImprovement(xi = 0.01)
        # 3) Instantiate the BayesianOptimizer.
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=41,
            verbose=2,  # 2 = Status prints, 1 = minimal prints, 0 = silent
            acquisition_function= acquisition_function  
        )

        # Let's show an example with more random inits and more total iterations:
        optimizer.maximize(
            init_points=20,    # Increased random points to explore space
            n_iter=80         # More iterations for deeper optimization
        )
        # 5) Retrieve the best parameters found
        best_params = optimizer.max['params']
        best_loss = -optimizer.max['target']

        # 6) Update slider variables with the best geometry
        gamma_var.set(best_params['gamma_val'])
        Gamma_var.set(best_params['Gamma_val'])
        dist_var.set(best_params['dist_val'])
        theta_i_var.set(best_params['theta_i_val'])
        zs_var.set(best_params['zs_val'])
        zb_var.set(best_params['zb_val'])
        chi_var.set(best_params['chi_val'])

        # Lattice constants
        a_var.set(best_params['a_val'])
        c_var.set(best_params['c_val'])

        update_gui()

        # 7) Display results
        msg = (
            f"[Geometry-Only] Optimization complete.\n"
            f"Best Loss: {best_loss:.3f}\n"
            f"gamma= {best_params['gamma_val']:.3f}, "
            f"Gamma= {best_params['Gamma_val']:.3f}, "
            f"dist= {best_params['dist_val']:.4f}\n"
            f"theta_i= {best_params['theta_i_val']:.3f}, "
            f"zs= {best_params['zs_val']:.6f}, "
            f"zb= {best_params['zb_val']:.6f}, "
            f"chi= {best_params['chi_val']:.3f},"
            f"a= {best_params['a_val']:.3f},"
            f"c= {best_params['c_val']:.3f}"
        )
        progress_label.config(text=msg)

    except Exception as e:
        import traceback
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        progress_label.config(text=f"Optimization failed with exception:\n{traceback_str}")

    finally:
        fit_button.config(state='normal')


# ------------------------------------------------------------------------
# 4) Mosaic-Only Optimization (Geometry is constant)
# ------------------------------------------------------------------------
def compute_peak_position_error_mosaic(
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
    # data
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    psi,
    n2,
    center,
    measured_peaks
):
    """
    Similar approach: Filter out unmeasured reflections, then simulate only those,
    and compare sign0/sign1 to measured data for each reflection.
    """
    measured_dict = build_measured_dict(measured_peaks)
    miller_sub, intensities_sub = filter_reflections(miller, intensities, measured_dict)
    if len(miller_sub) == 0:
        return 1e9

    _, max_positions = simulate_diffraction_positions(
        theta_i_const,
        gamma_const,
        Gamma_const,
        dist_const,
        zs_const,
        zb_const,
        mosaic_sigma_val,
        mosaic_gamma_val,
        mosaic_eta_val,
        miller_sub,
        intensities_sub,
        image_size,
        av,
        cv,
        lambda_,
        psi,
        n2,
        center
    )

    errors = []
    for i, (H, K, L) in enumerate(miller_sub):
        mx0, my0, mv0, mx1, my1, mv1 = max_positions[i]
        # Grab measured data for (H,K,L)
        meas_list = measured_dict.get((H, K, L), [])
        for meas in meas_list:
            x_meas, y_meas = meas['x'], meas['y']

            dist_sq_0 = (mx0 - x_meas)**2 + (my0 - y_meas)**2 if not np.isnan(mx0) else 1e12
            dist_sq_1 = (mx1 - x_meas)**2 + (my1 - y_meas)**2 if not np.isnan(mx1) else 1e12
            best_dist_sq = min(dist_sq_0, dist_sq_1)
            errors.append(best_dist_sq)

    if len(errors) == 0:
        return 1e9
    return np.mean(errors)

def objective_function_bayesian_mosaic(
    mosaic_sigma_val,
    mosaic_gamma_val,
    mosaic_eta_val,
    # fixed geometry:
    gamma_const,
    Gamma_const,
    dist_const,
    theta_i_const,
    zs_const,
    zb_const,
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    psi,
    n2,
    center,
    measured_peaks
):
    """
    Returns -cost for BayesianOptimization.
    """
    try:
        cost = compute_peak_position_error_mosaic(
            mosaic_sigma_val,
            mosaic_gamma_val,
            mosaic_eta_val,
            gamma_const,
            Gamma_const,
            dist_const,
            theta_i_const,
            zs_const,
            zb_const,
            miller,
            intensities,
            image_size,
            av,
            cv,
            lambda_,
            psi,
            n2,
            center,
            measured_peaks
        )
        return -cost
    except:
        return -1e9

def run_optimization_mosaic(
    fit_button,
    progress_label,
    # fixed geometry
    gamma_const,
    Gamma_const,
    dist_const,
    theta_i_const,
    zs_const,
    zb_const,
    # data
    miller,
    intensities,
    image_size,
    a_val,
    c_val,
    lambda_,
    psi,
    n2,
    center,
    measured_peaks
):
    """
    Optimizes mosaic parameters (mosaic_sigma, mosaic_gamma, mosaic_eta)
    while geometry is held constant.
    """
    try:
        fit_button.config(state='disabled')
        progress_label.config(text="2D Peak (Mosaic) Optimization in progress...")

        pbounds = {
            'mosaic_sigma_val': (0.0, 5.0),   # deg
            'mosaic_gamma_val': (0.0, 5.0),   # deg
            'mosaic_eta_val':   (0.0, 1.0)
        }

        optimizer = BayesianOptimization(
            f=lambda mosaic_sigma_val, mosaic_gamma_val, mosaic_eta_val:
                objective_function_bayesian_mosaic(
                    mosaic_sigma_val,
                    mosaic_gamma_val,
                    mosaic_eta_val,
                    gamma_const,
                    Gamma_const,
                    dist_const,
                    theta_i_const,
                    zs_const,
                    zb_const,
                    miller,
                    intensities,
                    image_size,
                    av,
                    cv,
                    lambda_,
                    psi,
                    n2,
                    center,
                    measured_peaks
                ),
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        optimizer.maximize(init_points=5, n_iter=30)
        best_params = optimizer.max['params']
        best_loss = -optimizer.max['target']

        msg = (f"[Mosaic-Only] Optimization complete.\n"
               f"Best Loss: {best_loss:.3f}\n"
               f"mosaic_sigma= {best_params['mosaic_sigma_val']:.3f} deg\n"
               f"mosaic_gamma= {best_params['mosaic_gamma_val']:.3f} deg\n"
               f"mosaic_eta= {best_params['mosaic_eta_val']:.3f}")
        progress_label.config(text=msg)

    except Exception as e:
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        progress_label.config(text=f"Optimization failed:\n{traceback_str}")
    finally:
        fit_button.config(state='normal')
