# optimizer.py

import numpy as np
from scipy.optimize import least_squares, differential_evolution

from ra_sim.simulation.mosaic_profiles import generate_random_profiles
from ra_sim.simulation.diffraction import process_peaks_parallel

def build_measured_dict(measured_peaks):
    """
    Convert a list of measured‐peak dicts into a mapping
    from (h,k,l) → list of (x,y) positions.
    """
    measured_dict = {}
    for p in measured_peaks:
        # support both {'label': 'h,k,l', 'x':…, 'y':…} and direct tuple formats
        if isinstance(p, dict) and 'label' in p:
            h, k, l = map(int, p['label'].split(','))
            x, y = float(p['x']), float(p['y'])
        else:
            # assume p is tuple/list like (h,k,l,x,y)
            h, k, l, x, y = p
        measured_dict.setdefault((h, k, l), []).append((x, y))
    return measured_dict

def simulate_and_compare_hkl(
    miller, intensities, image_size,
    params, measured_peaks, pixel_tol=np.inf
):
    """
    Simulate all HKLs with process_peaks_parallel, then for each
    reflection match the two possible spots to the measured positions
    of the same HKL, returning arrays of distances and coords.
    """
    measured_dict = build_measured_dict(measured_peaks)

    # Prepare the simulation buffer
    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    # Unpack geometry & mosaic parameters from params dict
    a = params['a']; c = params['c']

    dist = params['corto_detector']
    gamma = params['gamma']; Gamma = params['Gamma']
    chi   = params['chi']; psi = params.get('psi', 0.0)
    zs    = params['zs']; zb    = params['zb']
    debye_x = params['debye_x']; debye_y = params['debye_y']
    n2    = params['n2']
    center = params['center']
    theta_initial = params['theta_initial']

    mosaic = params['mosaic_params']
    wavelength = (mosaic.get('wavelength_array') or
               mosaic.get('wavelength_i_array'))
    
    # Full‐pattern simulation
    updated_image, maxpos, _, _ = process_peaks_parallel(
        miller, intensities, image_size,
        a, c, wavelength,
        sim_buffer, dist,
        gamma, Gamma, chi, psi,
        zs, zb, n2,
        mosaic['beam_x_array'],
        mosaic['beam_y_array'],
        mosaic['theta_array'],
        mosaic['phi_array'],
        mosaic['sigma_mosaic_deg'],
        mosaic['gamma_mosaic_deg'],
        mosaic['eta'],
        mosaic['wavelength_array'],
        debye_x, debye_y,
        center, theta_initial,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0
    )

    distances = []
    sim_coords = []
    meas_coords = []

    # For each HKL, test both peaks
    for i, (H, K, L) in enumerate(miller):
        if (H, K, L) not in measured_dict:
            continue
        candidates = measured_dict[(H, K, L)]
        I0, x0, y0, I1, x1, y1 = maxpos[i]

        for x, y in ((x0, y0), (x1, y1)):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            # find nearest measured
            ds = [np.hypot(x - mx, y - my) for mx, my in candidates]
            idx = int(np.argmin(ds))
            d = ds[idx]
            if d <= pixel_tol:
                sim_coords.append((x, y))
                meas_coords.append(candidates[idx])
                distances.append(d)

    return np.array(distances), sim_coords, meas_coords

def compute_peak_position_error_geometry_local(
    gamma, Gamma, dist, theta_initial, zs, zb, chi, a, c,
    center_x, center_y, measured_peaks,
    miller, intensities, image_size, mosaic_params, n2,
    psi, debye_x, debye_y, wavelength, pixel_tol=np.inf
):
    """
    Objective function for differential evolution: returns
    the 1D array of distances for all matched peaks.
    """
    
    params = {
        'gamma': gamma,
        'Gamma': Gamma,
        'corto_detector': dist,
        'theta_initial': theta_initial,
        'zs': zs,
        'zb': zb,
        'chi': chi,
        'a': a,
        'c': c,
        'center': (center_x, center_y),
        'lambda': wavelength,
        'n2': n2,
        'psi': psi,
        'debye_x': debye_x,
        'debye_y': debye_y,
        'mosaic_params': mosaic_params
    }

    D, _, _ = simulate_and_compare_hkl(
        miller, intensities, image_size,
        params, measured_peaks, pixel_tol
    )
    return D

def fit_geometry_parameters(
    miller, intensities, image_size,
    params, measured_peaks, var_names, pixel_tol=np.inf
):
    """
    Least‐squares fit for a subset of geometry parameters.
    var_names is a list of keys in `params` to optimize.
    """

    # Pack everything else needed into `common_args`
    common_args = {
        'miller': miller,
        'intensities': intensities,
        'image_size': image_size,
        'mosaic_params': params['mosaic_params'],
        'n2': params['n2'],
        'psi': params.get('psi', 0.0),
        'debye_x': params['debye_x'],
        'debye_y': params['debye_y'],
        'wavelength': params['lambda'],
        'pixel_tol': pixel_tol,
        'measured_peaks': measured_peaks
    }

    def cost_fn(x):
        # update only the var_names in params
        local = params.copy()
        for name, v in zip(var_names, x):
            local[name] = v

        # unpack args for compute_peak_position_error
        args = [
            local['gamma'],
            local['Gamma'],
            local['corto_detector'],
            local['theta_initial'],
            local['zs'],
            local['zb'],
            local['chi'],
            local['a'],
            local['c'],
            local['center'][0],
            local['center'][1],
        ]
        D = compute_peak_position_error_geometry_local(
            *args, **common_args
        )
        return D

    x0 = [params[name] for name in var_names]
    assert set(var_names) <= set(params), f"Unknown parameter(s): {set(var_names)-set(params)}"

    res = least_squares(cost_fn, x0)
    return res

def run_optimization_positions_geometry_local(
    miller, intensities, image_size,
    initial_params, bounds, measured_peaks
):
    """
    Global optimization (Differential Evolution) over all geometry
    parameters plus beam center. `bounds` is a list of (min, max)
    for each of [gamma, Gamma, dist, theta_initial,
    zs, zb, chi, a, c, center_x, center_y].
    """
    common_args = {
        'miller': miller,
        'intensities': intensities,
        'image_size': image_size,
        'mosaic_params': initial_params['mosaic_params'],
        'n2': initial_params['n2'],
        'psi': initial_params.get('psi', 0.0),
        'debye_x': initial_params['debye_x'],
        'debye_y': initial_params['debye_y'],
        'wavelength': initial_params['lambda'],
        'pixel_tol': np.inf,
        'measured_peaks': measured_peaks
    }

    def obj_glob(x):
        # x = [gamma, Gamma, dist, theta_init, zs, zb, chi, a, c, cx, cy]
        return np.sum(compute_peak_position_error_geometry_local(*x, **common_args))

    res = differential_evolution(obj_glob, bounds, maxiter=200, popsize=15)
    return res
