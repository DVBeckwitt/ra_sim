import numpy as np
from numba import njit, prange
from math import sin, cos, sqrt, pi, exp

@njit
def binary_search(cdf, target):
    """
    Returns the index where target <= cdf[index] using binary search.
    """
    lo = 0
    hi = cdf.size - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if cdf[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return lo

@njit
def sample_mosaic_angles_combined(eta, sigma_rad, gamma_rad, N=100, grid_points=1000):
    """
    Combined function that computes mosaic constants, evaluates the 2D pseudo-Voigt
    profile on a grid of angles, and samples N mosaic angles. All angles are in radians.

    Parameters:
      eta       : weight for the Lorentzian part.
      sigma_rad : Gaussian sigma in radians.
      gamma_rad : Lorentzian half-width in radians.
      N         : number of mosaic angles to sample.
      grid_points: number of points for discretizing the distribution.

    Returns:
      samples : array of N sampled mosaic angles in radians.
    """
    # Calculate pseudo-Voigt constants
    G2D_const = 1.0 / (2.0 * pi * sigma_rad * sigma_rad)
    G2D_exp   = 1.0 / (2.0 * sigma_rad * sigma_rad)
    L2D_const = 1.0 / (2.0 * pi)
    
    # Determine the grid maximum (10× the larger of sigma_rad or gamma_rad)
    max_alpha = 10.0 * (sigma_rad if sigma_rad > gamma_rad else gamma_rad)
    grid = np.linspace(0.0, max_alpha, grid_points)
    
    # Evaluate the pseudo-Voigt PDF on the grid
    pdf = np.empty(grid_points, dtype=np.float64)
    total = 0.0
    for i in range(grid_points):
        alpha = grid[i]
        r = sqrt(2.0) * abs(alpha)
        g_val = G2D_const * exp(-r*r * G2D_exp)
        denom = (r*r + gamma_rad*gamma_rad)**1.5
        l_val = L2D_const * (gamma_rad / denom)
        val = (1.0 - eta) * g_val + eta * l_val
        pdf[i] = val
        total += val

    # Normalize the PDF so that the sum equals 1
    for i in range(grid_points):
        pdf[i] /= total

    # Compute the cumulative distribution function (CDF)
    cdf = np.empty(grid_points, dtype=np.float64)
    cdf[0] = pdf[0]
    for i in range(1, grid_points):
        cdf[i] = cdf[i-1] + pdf[i]

    # Sample N angles using the CDF and binary search
    samples = np.empty(N, dtype=np.float64)
    for i in range(N):
        rnd = np.random.random()
        idx = binary_search(cdf, rnd)
        sample_val = grid[idx]
        # Randomly assign a positive or negative sign for symmetry about zero
        if np.random.random() > 0.5:
            samples[i] = sample_val
        else:
            samples[i] = -sample_val

    return samples


@njit
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect a parametric line (P0 + t*k_vec) with plane (P_plane, n_plane).
    Returns (ix, iy, iz, valid_bool).

    If no intersection or t<0 => valid_bool=False and we put NaNs for ix,iy,iz.
    Otherwise, returns the intersection coords plus valid_bool=True.
    """
    denom = k_vec[0]*n_plane[0] + k_vec[1]*n_plane[1] + k_vec[2]*n_plane[2]
    if abs(denom) < 1e-14:
        return (np.nan, np.nan, np.nan, False)

    num = ((P_plane[0] - P0[0]) * n_plane[0] +
           (P_plane[1] - P0[1]) * n_plane[1] +
           (P_plane[2] - P0[2]) * n_plane[2])
    t = num / denom

    if t < 0.0:
        return (np.nan, np.nan, np.nan, False)

    ix = P0[0] + t*k_vec[0]
    iy = P0[1] + t*k_vec[1]
    iz = P0[2] + t*k_vec[2]
    return (ix, iy, iz, True)

@njit
def intersect_line_plane_batch(start_pt, directions, plane_pt, plane_n):
    """
    Batch version: for each direction directions[i,:], compute intersection
    with plane. Returns (intersects, valid_mask).

    intersects.shape = (N, 3), valid_mask.shape = (N,).
    If no intersection or t<0 => mask=False and that row in intersects=NaN.
    """
    N = directions.shape[0]
    intersects = np.full((N, 3), np.nan, dtype=np.float64)
    valid = np.zeros(N, dtype=np.bool_)

    for i in range(N):
        kx = directions[i, 0]
        ky = directions[i, 1]
        kz = directions[i, 2]

        dot_dn = kx*plane_n[0] + ky*plane_n[1] + kz*plane_n[2]
        if abs(dot_dn) < 1e-14:
            continue

        num = ((plane_pt[0] - start_pt[0]) * plane_n[0] +
               (plane_pt[1] - start_pt[1]) * plane_n[1] +
               (plane_pt[2] - start_pt[2]) * plane_n[2])
        t = num / dot_dn
        if t < 0.0:
            continue

        ix = start_pt[0] + t*kx
        iy = start_pt[1] + t*ky
        iz = start_pt[2] + t*kz

        intersects[i, 0] = ix
        intersects[i, 1] = iy
        intersects[i, 2] = iz
        valid[i] = True

    return intersects, valid

import numpy as np
from numba import njit

@njit
def solve_q(k_in, k, gz0, gr0, G,mos_sampling, eps=1e-14):
    """
    Returns an array of shape (num_steps, 2, 3) with
    positive and negative (qx, qy, qz) solutions for each gz.

    Parameters
    ----------
    k_in : tuple or list of float
        (k_x, k_y, k_z) wavevector components.
    k : float
        Magnitude of the wavevector.
    gz0 : float
        Initial/maximum gz value for stepping.
    gr0 : float
        Some radial parameter gr0 used in solution calculation.
    G : float
        The magnitude sqrt(gr0^2 + gz0^2).
    num_steps : int, optional
        Number of discretized steps from 0 to gz0. Default 100.
    eps : float, optional
        Tolerance for ignoring negative or near-zero discriminant. Default 1e-14.

    Returns
    -------
    solutions : numpy.ndarray
        Shape (num_steps, 2, 3), where for each step `idx`:
        - solutions[idx, 0, :] = (qx_positive, qy_positive, gz)
        - solutions[idx, 1, :] = (qx_negative, qy_negative, gz)
    """
    k_x, k_y, k_z = k_in
    
    # Compute arrays for gz and gr from mosaic sampling
    gz_arr = gr0 * np.cos(mos_sampling) - gz0 * np.sin(mos_sampling)
    gr_arr = gr0 * np.sin(mos_sampling) + gz0 * np.cos(mos_sampling)
    
    solutions = np.zeros((gz_arr.size, 2, 3), dtype=np.float64)
    
    for idx in range(gz_arr.size):
        gz = gz_arr[idx]
        gr = gr_arr[idx]
        qx_positive =  (-k_x*(gr**2 + gz**2 + 2*gz*k_z - k**2 + k_x**2 + k_y**2 + k_z**2) - k_y*sqrt(-gr**4 - 2*gr**2*gz**2 - 4*gr**2*gz*k_z + 2*gr**2*k**2 + 2*gr**2*k_x**2 + 2*gr**2*k_y**2 - 2*gr**2*k_z**2 - gz**4 - 4*gz**3*k_z + 2*gz**2*k**2 - 2*gz**2*k_x**2 - 2*gz**2*k_y**2 - 6*gz**2*k_z**2 + 4*gz*k**2*k_z - 4*gz*k_x**2*k_z - 4*gz*k_y**2*k_z - 4*gz*k_z**3 - k**4 + 2*k**2*k_x**2 + 2*k**2*k_y**2 + 2*k**2*k_z**2 - k_x**4 - 2*k_x**2*k_y**2 - 2*k_x**2*k_z**2 - k_y**4 - 2*k_y**2*k_z**2 - k_z**4))/(2*(k_x**2 + k_y**2))
        qy_positive =  (-gr**2 - gz**2 - 2*gz*k_z + k**2 - k_x**2 - 2*k_x*qx_positive - k_y**2 - k_z**2)/(2*k_y)
        qx_negative =  (-k_x*(gr**2 + gz**2 + 2*gz*k_z - k**2 + k_x**2 + k_y**2 + k_z**2) + k_y*sqrt(-gr**4 - 2*gr**2*gz**2 - 4*gr**2*gz*k_z + 2*gr**2*k**2 + 2*gr**2*k_x**2 + 2*gr**2*k_y**2 - 2*gr**2*k_z**2 - gz**4 - 4*gz**3*k_z + 2*gz**2*k**2 - 2*gz**2*k_x**2 - 2*gz**2*k_y**2 - 6*gz**2*k_z**2 + 4*gz*k**2*k_z - 4*gz*k_x**2*k_z - 4*gz*k_y**2*k_z - 4*gz*k_z**3 - k**4 + 2*k**2*k_x**2 + 2*k**2*k_y**2 + 2*k**2*k_z**2 - k_x**4 - 2*k_x**2*k_y**2 - 2*k_x**2*k_z**2 - k_y**4 - 2*k_y**2*k_z**2 - k_z**4))/(2*(k_x**2 + k_y**2))
        qy_negative =  (-gr**2 - gz**2 - 2*gz*k_z + k**2 - k_x**2 - 2*k_x*qx_negative - k_y**2 - k_z**2)/(2*k_y)

        solutions[idx, 0, 0] = qx_positive
        solutions[idx, 0, 1] = qy_positive
        solutions[idx, 0, 2] = gz
        solutions[idx, 1, 0] = qx_negative
        solutions[idx, 1, 1] = qy_negative
        solutions[idx, 1, 2] = gz

    return solutions

@njit
def calculate_phi(
    H, K, L, av, cv,
    wavelength_array,
    image, image_size,
    gamma_rad, Gamma_rad, chi_rad, psi_rad,
    zs, zb, n2,
    beam_x_array, beam_y_array,
    theta_array, phi_array,
    reflection_intensity,
    mos_sampling,
    debye_x, debye_y,
    center,
    theta_initial_deg,
    R_x_detector, R_z_detector, n_det_rot, Detector_Pos,
    e1_det, e2_det,
    R_z_R_y,         # combined rotation for sample
    R_ZY_n,          # sample-plane normal in sample coords
    P0, unit_x,
    save_flag, q_data, q_count, i_peaks_index
):
    """
    For a single reflection (H,K,L), we do:
      1) Build G in crystal coords => (gr0, 0, gz0).
      2) Rotate the sample by theta_initial => R_x @ R_z_R_y => R_sample.
      3) For each beam sample, compute wavevector with partial refraction.
      4) Solve q from solve_q => for each solution i_sol, compute scattered wave,
         intersect with the detector plane, and accumulate intensity.

    Returns (max_I_sign0, max_x_sign0, max_y_sign0,
             max_I_sign1, max_x_sign1, max_y_sign1).
    """

    gz0 = 2.0*np.pi * (L/cv)
    gr0 = 4.0*np.pi/av * sqrt((H*H + H*K + K*K)/3.0)
    G = np.sqrt(gr0**2 + gz0**2)
    # Track maximum intensities
    max_I_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan
    max_I_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan
    
    # Create rotation for the chosen theta_initial
    rad_theta_i = theta_initial_deg*(pi/180.0)
    
    
    R_x = np.array([
        [1.0, 0.0,            0.0],
        [0.0, cos(rad_theta_i), -sin(rad_theta_i)],
        [0.0, sin(rad_theta_i),  cos(rad_theta_i)]
    ])
    R_sample = R_x @ R_z_R_y

    # Sample-plane normal in lab
    n_surf = R_x @ R_ZY_n
    n_surf /= np.linalg.norm(n_surf)

    # Rotated sample-plane offset
    P0_rot = R_sample @ P0
    P0_rot[0] = 0 
    
    row_center = center[0]
    col_center = center[1]
    n_samp = beam_x_array.size

    # Build coordinate axes e1_temp, e2_temp on sample surface
    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_norm = np.linalg.norm(e1_temp)
    if e1_norm < 1e-12:
        alt_refs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        success = False
        for alt_ref in alt_refs:
            cross_tmp = np.cross(n_surf, alt_ref)
            cross_norm_tmp = np.linalg.norm(cross_tmp)
            if cross_norm_tmp > 1e-12:
                e1_temp = cross_tmp / cross_norm_tmp
                success = True
                break
        if not success:
            return (max_x_sign0, max_y_sign0, max_I_sign0,
                    max_x_sign1, max_y_sign1, max_I_sign1)
    else:
        e1_temp /= e1_norm

    e2_temp = np.cross(n_surf, e1_temp)

    # Main beam-sampling loop
    for i_samp in prange(n_samp):
        lam_samp = wavelength_array[i_samp]
        k_mag = 2.0 * np.pi / lam_samp

        bx = beam_x_array[i_samp]
        by = beam_y_array[i_samp]
        beam_start = np.array([bx, -20e-3, -zb + by], dtype=np.float64)

        dtheta = theta_array[i_samp]
        dphi   = phi_array[i_samp]

        # Incoming wave in lab
        k_in = np.array([
            cos(dtheta)*sin(dphi),
            cos(dtheta)*cos(dphi),
            sin(dtheta)
        ], dtype=np.float64)

        # Intersect the beam with the sample plane
        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if not valid_int:
            # No valid intersection with sample plane
            continue

        I_plane = np.array([ix, iy, iz])

        # Incident angle relative to sample normal
        kn_dot = k_in[0]*n_surf[0] + k_in[1]*n_surf[1] + k_in[2]*n_surf[2]
        th_i_prime = (pi/2.0) - np.arccos(kn_dot)

        # Project onto sample surface to get azimuth
        projected_incident = k_in - kn_dot*n_surf
        proj_len = np.linalg.norm(projected_incident)
        if proj_len > 1e-12:
            projected_incident /= proj_len
        else:
            projected_incident[:] = 0.0

        p1 = (projected_incident[0]*e1_temp[0]
              + projected_incident[1]*e1_temp[1]
              + projected_incident[2]*e1_temp[2])
        p2 = (projected_incident[0]*e2_temp[0]
              + projected_incident[1]*e2_temp[1]
              + projected_incident[2]*e2_temp[2])

        phi_i_prime = (pi/2.0) - np.arctan2(p2, p1)
        th_t = np.arccos(cos(th_i_prime)/np.real(n2))*np.sign(th_i_prime)

        # Build the scattered wave entering the crystal
        k_scat = k_mag * sqrt(np.real(n2**2))
        k_x_scat = k_scat*cos(th_t)*sin(phi_i_prime)
        k_y_scat = k_scat*cos(th_t)*cos(phi_i_prime)
        k_z_scat = k_scat*sin(th_t)

        # Solve for q
        k_in_crystal = np.array([k_x_scat, k_y_scat, k_z_scat])
        All_Q = solve_q(k_in_crystal, k_scat, gz0, gr0, G,mos_sampling)
        All_Q_flat = All_Q.reshape((-1, 3))  # => shape (2*num_steps,3)

        # For each solution i_sol, compute final wave, intersect with detector
        for i_sol in range(All_Q_flat.shape[0]):
            Qx = All_Q_flat[i_sol, 0]
            Qy = All_Q_flat[i_sol, 1]
            Qz = All_Q_flat[i_sol, 2]

            # Create the total scattered wave in lab
            k_tx_prime = Qx + k_x_scat
            k_ty_prime = Qy + k_y_scat
            k_tz_prime = Qz + k_z_scat

            # Check scattering angles
            kr = sqrt(k_tx_prime**2 + k_ty_prime**2)
            if abs(kr) < 1e-12:
                twotheta_t = 0.0
            else:
                twotheta_t = np.arctan(k_tz_prime / kr)

            # Build final scattered wave
            phi_f = np.arctan2(k_tx_prime, k_ty_prime)
            k_tx_f = k_scat*cos(twotheta_t)*sin(phi_f)
            k_ty_f = k_scat*cos(twotheta_t)*cos(phi_f)
            real_k_tz_f = k_scat*sin(twotheta_t)

            kf = np.array([k_tx_f, k_ty_f, real_k_tz_f])
            kf_prime = R_sample @ kf

            # Now intersect from I_plane in direction kf_prime to the detector plane
            dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
            if not valid_det:
                continue

            plane_to_det = np.array([dx - Detector_Pos[0],
                                     dy - Detector_Pos[1],
                                     dz - Detector_Pos[2]], dtype=np.float64)

            x_det = (plane_to_det[0]*e1_det[0]
                     + plane_to_det[1]*e1_det[1]
                     + plane_to_det[2]*e1_det[2])
            y_det = (plane_to_det[0]*e2_det[0]
                     + plane_to_det[1]*e2_det[1]
                     + plane_to_det[2]*e2_det[2])

            rpx = int(round(center[0] - y_det / 100e-6))
            cpx = int(round(center[1] + x_det / 100e-6))
            if not (0 <= rpx < image_size and 0 <= cpx < image_size):
                continue

            val = (
                reflection_intensity 
                * np.exp(-Qz**2 * debye_x**2)
                * np.exp(-(Qx**2 + Qy**2)*debye_y**2)
            )
            image[rpx, cpx] += val

            # Track maxima
            if i_sol % 2 == 0:
                if image[rpx, cpx] > max_I_sign0:
                    max_I_sign0 = image[rpx, cpx]
                    max_x_sign0 = cpx
                    max_y_sign0 = rpx
            else:
                if image[rpx, cpx] > max_I_sign1:
                    max_I_sign1 = image[rpx, cpx]
                    max_x_sign1 = cpx
                    max_y_sign1 = rpx

            # Optionally store Q-data
            if save_flag == 1 and q_count[i_peaks_index] < q_data.shape[1]:
                idx = q_count[i_peaks_index]
                q_data[i_peaks_index, idx, 0] = Qx
                q_data[i_peaks_index, idx, 1] = Qy
                q_data[i_peaks_index, idx, 2] = Qz
                q_data[i_peaks_index, idx, 3] = val
                # mosaic angle => i_sol//2
                q_data[i_peaks_index, idx, 4] = mos_sampling[i_sol // 2]
                q_count[i_peaks_index] += 1

    return (
        max_I_sign0, max_x_sign0, max_y_sign0,
        max_I_sign1, max_x_sign1, max_y_sign1
    )

@njit(parallel=True, fastmath=True)
def process_peaks_parallel(
    miller, intensities, image_size,
    av, cv, lambda_, image,
    Distance_CoR_to_Detector, gamma_deg, Gamma_deg, chi_deg, psi_deg,
    zs, zb, n2,
    beam_x_array, beam_y_array,
    theta_array, phi_array,
    sigma_pv_deg, gamma_pv_deg, eta_pv,
    wavelength_array,
    debye_x, debye_y, center,
    theta_initial_deg,
    unit_x, n_detector,
    save_flag
):
    """
    1) Convert angles => gamma_rad, Gamma_rad, chi_rad, psi_rad
    2) Build mosaic distribution => mosaic_angles, mosaic_intens
    3) Build detector transformations => R_x_det, R_z_det => n_det_rot, etc.
    4) Build sample orientation => R_y, R_z => R_z_R_y
    5) Loop over reflections => call calculate_phi
    6) Return final image, plus q_data if save_flag=1
    """
    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)

    # Build mosaic distribution
    sigma_rad = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)


    mos_sampling = sample_mosaic_angles_combined(eta_pv, sigma_rad, gamma_rad, N=100, grid_points=1000)

    # Detector-plane transformations
    cg = cos(gamma_rad); sg = sin(gamma_rad)
    cG = cos(Gamma_rad); sG = sin(Gamma_rad)

    R_x_det = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cg,  sg],
        [0.0,-sg,  cg]
    ])
    R_z_det = np.array([
        [ cG, sG, 0.0],
        [-sG, cG, 0.0],
        [ 0.0, 0.0,1.0]
    ])

    nd_temp = R_x_det @ n_detector
    n_det_rot = R_z_det @ nd_temp
    nd_len = sqrt(n_det_rot[0]*n_det_rot[0]
                  + n_det_rot[1]*n_det_rot[1]
                  + n_det_rot[2]*n_det_rot[2])
    n_det_rot /= nd_len

    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0], dtype=np.float64)

    # define e1_det => projection of unit_x onto plane orthonormal to n_det_rot
    dot_e1 = (unit_x[0]*n_det_rot[0]
              + unit_x[1]*n_det_rot[1]
              + unit_x[2]*n_det_rot[2])
    e1_det = unit_x - dot_e1*n_det_rot
    e1_len = sqrt(e1_det[0]**2 + e1_det[1]**2 + e1_det[2]**2)
    if e1_len < 1e-14:
        e1_det = np.array([1.0,0.0,0.0])
    else:
        e1_det /= e1_len

    # define e2_det => cross(-n_det_rot, e1_det)
    tmpx = n_det_rot[1]* e1_det[2] - n_det_rot[2]* e1_det[1]
    tmpy = n_det_rot[2]* e1_det[0] - n_det_rot[0]* e1_det[2]
    tmpz = n_det_rot[0]* e1_det[1] - n_det_rot[1]* e1_det[0]
    e2_det = np.array([-tmpx, -tmpy, -tmpz], dtype=np.float64)
    e2_len= sqrt(e2_det[0]**2 + e2_det[1]**2 + e2_det[2]**2)
    if e2_len<1e-14:
        e2_det= np.array([0.0,1.0,0.0])
    else:
        e2_det/= e2_len

    # Sample orientation => build R_y, R_z => R_z_R_y
    c_chi = cos(chi_rad); s_chi = sin(chi_rad)
    R_y = np.array([
        [ c_chi, 0.0,   s_chi],
        [ 0.0,   1.0,   0.0],
        [-s_chi, 0.0, c_chi]
    ])
    c_psi= cos(psi_rad); s_psi= sin(psi_rad)
    R_z= np.array([
        [ c_psi, s_psi, 0.0],
        [-s_psi, c_psi, 0.0],
        [ 0.0,   0.0,   1.0]
    ])
    R_z_R_y= R_z @ R_y

    # sample-plane normal in sample coords => R_ZY_n
    n1 = np.array([0.0,0.0,1.0], dtype=np.float64)
    R_ZY_n = R_z_R_y @ n1
    nzy_len = sqrt(R_ZY_n[0]**2 + R_ZY_n[1]**2 + R_ZY_n[2]**2)
    R_ZY_n /= nzy_len

    # sample plane offset => P0
    P0 = np.array([0.0,0.0,-zs], dtype=np.float64)

    num_peaks= miller.shape[0]
    max_solutions= 2000000

    if save_flag==1:
        q_data= np.full((num_peaks,max_solutions,5), np.nan, dtype= np.float64)
        q_count= np.zeros(num_peaks, dtype= np.int64)
    else:
        q_data= np.zeros((1,1,5), dtype= np.float64)
        q_count= np.zeros(1, dtype= np.int64)

    max_positions= np.empty((num_peaks,6), dtype= np.float64)

    # Main reflection loop
    for i_pk in prange(num_peaks):
        H= miller[i_pk,0]
        K= miller[i_pk,1]
        L= miller[i_pk,2]
        reflI= intensities[i_pk]

        # call calculate_phi for this reflection
        (mx0,my0,mv0,
         mx1,my1,mv1) = calculate_phi(
            H, K, L,
            av, cv,
            wavelength_array,
            image, image_size,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs, zb, n2,
            beam_x_array, beam_y_array,
            theta_array, phi_array,
            reflI,mos_sampling,
            debye_x, debye_y,
            center,
            theta_initial_deg,
            R_x_det, R_z_det, n_det_rot, Detector_Pos,
            e1_det, e2_det,
            R_z_R_y,
            R_ZY_n,
            P0, unit_x,
            save_flag, q_data, q_count, i_pk
        )

        # store maximum positions
        max_positions[i_pk,0] = mx0
        max_positions[i_pk,1] = my0
        max_positions[i_pk,2] = mv0
        max_positions[i_pk,3] = mx1
        max_positions[i_pk,4] = my1
        max_positions[i_pk,5] = mv1

    if save_flag==1:
        return image, max_positions, q_data, q_count
    else:
        return image, max_positions, None, None
