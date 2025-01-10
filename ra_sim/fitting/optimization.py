##########################################
# optimization.py
##########################################

import numpy as np
import traceback
from bayes_opt import BayesianOptimization
from collections import defaultdict
from bayes_opt.acquisition import ExpectedImprovement


#######################################################################
# 1) Utility functions:
#    - build_measured_dict: convert measured_peaks list into a dict
#    - filter_reflections : keep only the Miller indices you actually measured
#######################################################################
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
            label_str = peak['label']  # e.g., "0,1,5"
            hkl = tuple(map(int, label_str.split(',')))
        except KeyError:
            print(f">>> KeyError: 'label' key missing in peak {idx}: {peak}")
            continue
        except ValueError:
            print(f">>> ValueError: Unable to parse 'label' in peak {idx}: {peak}")
            continue

        if hkl not in measured_dict:
            measured_dict[hkl] = []
        measured_dict[hkl].append({
            'x': peak['x'],
            'y': peak['y'],
            'sigma': peak.get('sigma', None)
        })
    return measured_dict


def filter_reflections(miller, intensities, measured_dict):
    """
    Keep only those Miller indices from 'miller' that appear in 'measured_dict'.
    Return (miller_sub, intensities_sub) as arrays.
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


#######################################################################
# 2) Shared simulation function (geometry + mosaic)
#######################################################################
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
    center,            # (cx, cy)
    num_samples=1000,
    divergence_sigma=0.0,
    bw_sigma=0.0
):
    """
    Simulates a 2D diffraction pattern for the given (miller, intensities),
    using geometry + mosaic parameters, plus a beam center.

    Returns:
      (simulated_image, max_positions)
    """
    from ra_sim.simulation.mosaic_profiles import generate_random_profiles
    from ra_sim.simulation.diffraction import process_peaks_parallel

    sigma_mosaic_rad = np.radians(mosaic_sigma)
    gamma_mosaic_rad = np.radians(mosaic_gamma)

    # Random beam/mosaic/divergence profiles
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
        center=center,       # pass (cx, cy)
        theta_initial=theta_i,
        theta_final=(theta_i + 0.5),
        step=0.1,
        unit_x=np.array([1.0, 0.0, 0.0]),
        n_detector=np.array([0.0, 1.0, 0.0])
    )

    return simulated_image, max_positions


#######################################################################
# 3) Geometry-Only Optimization (Mosaic is constant)
#######################################################################
def compute_peak_position_error_geometry(
    gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val, chi_val,
    mosaic_sigma_const, mosaic_gamma_const, mosaic_eta_const,
    miller,
    intensities,
    image_size,
    a_val,
    c_val,
    lambda_,
    psi,
    n2,
    center,  
    measured_peaks,
    debug=True,
    debug_log_path="geometry_debug.txt"
):
    """
    Steps:
      1) Filter the reflections to only those with measured data.
      2) Simulate diffraction for that subset (including new center).
      3) Compare simulated positions (mx, my) to measured (x, y).
      4) Return average distance^2 across sign solutions & reflections.
    """
    import math

    measured_dict = build_measured_dict(measured_peaks)
    miller_sub, intensities_sub = filter_reflections(miller, intensities, measured_dict)
    if len(miller_sub) == 0:
        return 1e9

    try:
        sim_image, max_positions = simulate_diffraction_positions(
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
        if debug:
            print(f"Sim error: {e}")
        return 1e9

    all_errors = []
    for i, (H, K, L) in enumerate(miller_sub):
        mx0, my0, mv0, mx1, my1, mv1 = max_positions[i]
        meas_list = measured_dict.get((H, K, L), [])
        if len(meas_list) == 0:
            if not math.isnan(mx0) and not math.isnan(my0):
                all_errors.append(1e9)
            if not math.isnan(mx1) and not math.isnan(my1):
                all_errors.append(1e9)
            continue

        # sign0 => if valid
        if not math.isnan(mx0) and not math.isnan(my0):
            best_err_0 = 1e15
            for meas in meas_list:
                d_sq = (mx0 - meas['x'])**2 + (my0 - meas['y'])**2
                if d_sq < best_err_0:
                    best_err_0 = d_sq
            all_errors.append(best_err_0)

        # sign1 => if valid
        if not math.isnan(mx1) and not math.isnan(my1):
            best_err_1 = 1e15
            for meas in meas_list:
                d_sq = (mx1 - meas['x'])**2 + (my1 - meas['y'])**2
                if d_sq < best_err_1:
                    best_err_1 = d_sq
            all_errors.append(best_err_1)

    if len(all_errors) == 0:
        return 1e9
    return np.mean(all_errors)


def objective_function_bayesian_geometry(
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
    center,  # (cx, cy)
    measured_peaks
):
    """We compute cost, then return negative for the BayesianOptimizer to maximize."""
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
    center,  # This is the center from sliders
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
    """
    Bayesian optimization for geometry + beam center, with mosaic params fixed.
    We want the center to vary ±100 from the slider values.
    """
    try:
        fit_button.config(state='disabled')
        progress_label.config(text="2D Peak (Geometry) Optimization in progress...")

        from bayes_opt import BayesianOptimization

        # The user wants the pbounds for beam center to be ±100 of current slider values:
        cx_init = center[0]  # from slider
        cy_init = center[1]  # from slider

        # Now define your pbounds for beam center dynamically:
        pbounds = {
            'gamma_val': (-2.0, 2.0),
            'Gamma_val': (-2.0, 2.0),
            'dist_val': (74e-3, 78e-3),
            'theta_i_val': (5.5, 6.5),
            'zs_val': (0, 1.5e-3),
            'zb_val': (0, 1e-3),
            'chi_val': (-0.3, 0.3),
            'a_val': (4.1, 4.6),  #  range in Å
            'c_val': (27.5, 29.5),
            #'cx_val':     (cx_init - 100, cx_init + 100),
            #'cy_val':     (cy_init - 100, cy_init + 100),
        }

        def objective_function(
            gamma_val, Gamma_val, dist_val, theta_i_val, zs_val, zb_val,
            chi_val, a_val, c_val
            #, cx_val, cy_val
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
                center, #=(cx_val, cy_val),
                measured_peaks=measured_peaks
            )
            return cost  # already negative inside

        acquisition_function = ExpectedImprovement(xi=0.1)
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2,
            acquisition_function=acquisition_function
        )

        optimizer.maximize(init_points=100, n_iter=100)

        best_params = optimizer.max['params']
        best_loss = -optimizer.max['target']

        gamma_var.set(best_params['gamma_val'])
        Gamma_var.set(best_params['Gamma_val'])
        dist_var.set(best_params['dist_val'])
        theta_i_var.set(best_params['theta_i_val'])
        zs_var.set(best_params['zs_val'])
        zb_var.set(best_params['zb_val'])
        chi_var.set(best_params['chi_val'])
        a_var.set(best_params['a_val'])
        c_var.set(best_params['c_val'])

        # Also update center_x, center_y
        #center_x_var.set(best_params['cx_val'])
        #center_y_var.set(best_params['cy_val'])

        update_gui()

        msg = (
            f"[Geometry-Only] Optimization complete.\n"
            f"Best Loss: {best_loss:.3f}\n"
            f"gamma= {best_params['gamma_val']:.3f},  "
            f"Gamma= {best_params['Gamma_val']:.3f}, "
            f"dist= {best_params['dist_val']:.5f}\n"
            f"theta_i= {best_params['theta_i_val']:.3f}, "
            f"zs= {best_params['zs_val']:.6f},   zb= {best_params['zb_val']:.6f}, "
            f"chi= {best_params['chi_val']:.3f}\n"
            f"a= {best_params['a_val']:.3f},   c= {best_params['c_val']:.3f}\n"
            #f"center_x= {best_params['cx_val']:.1f}, center_y= {best_params['cy_val']:.1f}"
        )
        progress_label.config(text=msg)

    except Exception as e:
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        progress_label.config(text=f"Optimization failed:\n{traceback_str}")
    finally:
        fit_button.config(state='normal')


#######################################################################
# 4) Mosaic-Only Optimization
#######################################################################
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
    center,
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    psi,
    n2,
    measured_peaks
):
    """Filter out unmeasured reflections, then simulate only those, and compare sign0/sign1."""
    measured_dict = build_measured_dict(measured_peaks)
    miller_sub, intensities_sub = filter_reflections(miller, intensities, measured_dict)
    if len(miller_sub) == 0:
        return 1e9

    sim_image, max_positions = simulate_diffraction_positions(
        theta_i=theta_i_const,
        gamma=gamma_const,
        Gamma=Gamma_const,
        dist=dist_const,
        zs=zs_const,
        zb=zb_const,
        chi=0.0,
        mosaic_sigma=mosaic_sigma_val,
        mosaic_gamma=mosaic_gamma_val,
        mosaic_eta=mosaic_eta_val,
        miller=miller_sub,
        intensities=intensities_sub,
        image_size=image_size,
        a_val=av,
        c_val=cv,
        lambda_=lambda_,
        psi=psi,
        n2=n2,
        center=center
    )

    errors = []
    for i, (H, K, L) in enumerate(miller_sub):
        mx0, my0, mv0, mx1, my1, mv1 = max_positions[i]
        meas_list = measured_dict.get((H, K, L), [])
        if len(meas_list) == 0:
            if not np.isnan(mx0) and not np.isnan(my0):
                errors.append(1e9)
            if not np.isnan(mx1) and not np.isnan(my1):
                errors.append(1e9)
            continue
        for meas in meas_list:
            x_meas, y_meas = meas['x'], meas['y']
            dist_sq_0 = (mx0 - x_meas)**2 + (my0 - y_meas)**2 if not np.isnan(mx0) else 1e12
            dist_sq_1 = (mx1 - x_meas)**2 + (my1 - y_meas)**2 if not np.isnan(mx1) else 1e12
            errors.append(min(dist_sq_0, dist_sq_1))

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
    center,
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    psi,
    n2,
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
            center,
            miller,
            intensities,
            image_size,
            av,
            cv,
            lambda_,
            psi,
            n2,
            measured_peaks
        )
        return -cost
    except:
        return -1e9


def run_optimization_mosaic(
    fit_button,
    progress_label,
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
    """Optimizes mosaic parameters while geometry is held constant."""
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
                    center,
                    miller,
                    intensities,
                    image_size,
                    av,
                    cv,
                    lambda_,
                    psi,
                    n2,
                    measured_peaks
                ),
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        optimizer.maximize(init_points=5, n_iter=30)
        best_params = optimizer.max['params']
        best_loss = -optimizer.max['target']

        msg = (
            f"[Mosaic-Only] Optimization complete.\n"
            f"Best Loss: {best_loss:.3f}\n"
            f"mosaic_sigma= {best_params['mosaic_sigma_val']:.3f} deg\n"
            f"mosaic_gamma= {best_params['mosaic_gamma_val']:.3f} deg\n"
            f"mosaic_eta= {best_params['mosaic_eta_val']:.3f}"
        )
        progress_label.config(text=msg)

    except Exception as e:
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        progress_label.config(text=f"Optimization failed:\n{traceback_str}")
    finally:
        fit_button.config(state='normal')
