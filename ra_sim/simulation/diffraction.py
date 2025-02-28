import numpy as np
from numba import njit, prange
from math import sin, cos, sqrt, pi

# If you have an external Fresnel:
# from ra_sim.utils.calculations import fresnel_transmission

@njit
def find_mosaic_constants(sigma_rad, gamma_rad):
    """
    Returns constants for 2D pseudo-Voigt:
      G(r)=1/(2πσ²)*exp[-r²/(2σ²)]
      L(r)=1/(2π)*(γ/(r²+γ²)^(3/2)).
    """
    G2D_const = 1.0/(2.0*np.pi * sigma_rad*sigma_rad)
    G2D_exp   = 1.0/(2.0*sigma_rad*sigma_rad)
    L2D_const = 1.0/(2.0*np.pi)
    return (G2D_const, G2D_exp, L2D_const, gamma_rad)


@njit
def radial_pseudo_voigt_2d(mosaic_angles, eta, mosaic_consts):
    """
    radial 2D pseudo-Voigt => shape(n,).
    For angle α => r = sqrt(2)*|α|.
    mosaic_consts= (G2D_const, G2D_exp, L2D_const, gamma).
    """
    G2D_const, G2D_exp, L2D_const, gamma = mosaic_consts
    n = mosaic_angles.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        alpha = mosaic_angles[i]
        r = sqrt(2.0)*abs(alpha)
        # Gaussian part
        g_val = G2D_const * np.exp(- (r*r)* G2D_exp)
        # Lorentz part
        denom = (r*r + gamma*gamma)**1.5
        l_val = L2D_const*(gamma / denom)
        out[i] = (1.0 - eta)*g_val + eta*l_val
    return out


@njit
def G_rotation_array(gz0, gr0, angles):
    """
    Rotate (gz0,gr0) => (gz,gr) for each angle:
      gz = gr0*sin(a) + gz0*cos(a)
      gr = gr0*cos(a) - gz0*sin(a)
    """
    n = angles.size
    gz_arr = np.empty(n, dtype=np.float64)
    gr_arr = np.empty(n, dtype=np.float64)
    for i in range(n):
        a = angles[i]
        gz_arr[i] = gr0*sin(a) + gz0*cos(a)
        gr_arr[i] = gr0*cos(a) - gz0*sin(a)
    return gz_arr, gr_arr


@njit
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect a ray parameterized by P0 + t*k_vec with
    the plane defined by point P_plane and normal n_plane.
    Returns (point, t) if intersection is valid and t>=0.
    Otherwise returns (None, None).
    """
    denom = k_vec[0]*n_plane[0] + k_vec[1]*n_plane[1] + k_vec[2]*n_plane[2]
    if abs(denom) < 1e-14:
        return None, None
    num = ((P_plane[0]-P0[0])*n_plane[0] +
           (P_plane[1]-P0[1])*n_plane[1] +
           (P_plane[2]-P0[2])*n_plane[2])
    t = num / denom
    if t < 0.0:
        return None, None
    pt = np.array([
        P0[0] + t*k_vec[0],
        P0[1] + t*k_vec[1],
        P0[2] + t*k_vec[2]
    ], dtype=np.float64)
    return pt, t


@njit
def intersect_line_plane_batch(start_pt, directions, plane_pt, plane_n):
    """
    Batch version: for an array of directions, find intersection
    with the plane. Return (intersect_points, valid_mask).
    """
    N = directions.shape[0]
    intersects = np.full((N,3), np.nan, dtype=np.float64)
    valid = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        kx = directions[i,0]
        ky = directions[i,1]
        kz = directions[i,2]
        dot_dn = kx*plane_n[0] + ky*plane_n[1] + kz*plane_n[2]
        if abs(dot_dn) < 1e-14:
            continue
        num = ((plane_pt[0]-start_pt[0])*plane_n[0] +
               (plane_pt[1]-start_pt[1])*plane_n[1] +
               (plane_pt[2]-start_pt[2])*plane_n[2])
        t = num / dot_dn
        if t < 0.0:
            continue
        ix = start_pt[0] + t*kx
        iy = start_pt[1] + t*ky
        iz = start_pt[2] + t*kz
        intersects[i,0] = ix
        intersects[i,1] = iy
        intersects[i,2] = iz
        valid[i] = True
    return intersects, valid

@njit
def solve_q(k_in, k, gz_values, gr_values, G, eps=1e-14):
    """
    Uses the explicit '1/k_y style' solutions:

    Branch 1:
      q_y = (...big expression...)
      q_x = (...big expression...)

    Branch 2:
      q_y = (...big expression...)
      q_x = (...big expression...)

    storing them along with Qz = gz in a (n_mos, 2, 3) result array:

        solutions[i, 0, :] = [qx_positive, qy_positive, gz]
        solutions[i, 1, :] = [qx_negative, qy_negative, gz]
    """

    k_x, k_y, k_z = k_in
    n_mos = gz_values.size

    # Prepare output array: shape = (n_mos, 2, 3)
    #   i => mosaic index
    #   sign => 0 for "branch 1" (positive), 1 for "branch 2" (negative)
    #   coordinate => (qx, qy, gz)
    solutions = np.full((n_mos, 2, 3), np.nan, dtype=np.float64)

    for i in range(n_mos):
        gz = gz_values[i]
        gr = gr_values[i]

        #--------------------------------------------------------------------
        # The big expression under the square root from your symbolic solution:
        # (Same for branches 1 and 2)
        #--------------------------------------------------------------------
        radical = (
            - gr**4
            - 2.0*gr**2*gz**2
            - 4.0*gr**2*gz*k_z
            + 2.0*gr**2*k**2
            + 2.0*gr**2*k_x**2
            + 2.0*gr**2*k_y**2
            - 2.0*gr**2*k_z**2
            - gz**4
            - 4.0*gz**3*k_z
            + 2.0*gz**2*k**2
            - 2.0*gz**2*k_x**2
            - 2.0*gz**2*k_y**2
            - 6.0*gz**2*k_z**2
            + 4.0*gz*k**2*k_z
            - 4.0*gz*k_x**2*k_z
            - 4.0*gz*k_y**2*k_z
            - 4.0*gz*k_z**3
            - k**4
            + 2.0*k**2*k_x**2
            + 2.0*k**2*k_y**2
            + 2.0*k**2*k_z**2
            - k_x**4
            - 2.0*k_x**2*k_y**2
            - 2.0*k_x**2*k_z**2
            - k_y**4
            - 2.0*k_y**2*k_z**2
            - k_z**4
        )

        # If negative, no real solution => skip
        if radical < 0.0:
            continue

        # Compute the sqrt(...) portion:
        sqrt_val = sqrt(radical) / (2.0 * (k_x**2 + k_y**2))

        #--------------------------------------------------------------------
        # "Solution branch 1"
        #  q_y = [the big expression from your snippet for branch 1]
        #  q_x = [the big expression from your snippet for branch 1]
        #
        #  We'll call them (q_y_positive, q_x_positive).
        #--------------------------------------------------------------------
        qy_positive = (
            - gr**2
            - gz**2
            - 2.0*gz*k_z
            + k**2
            - k_x**2
            - 2.0*k_x * (
                (
                    k_x * (
                        - gr**2 - gz**2 - 2.0*gz*k_z
                        + k**2 - k_x**2 - k_y**2 - k_z**2
                    )
                    / (2.0*(k_x**2 + k_y**2))
                )
                - (
                    k_y* sqrt_val*(2.0*(k_x**2 + k_y**2))
                    # careful: your original snippet divides by the same denom
                    # We'll keep the direct version matching your text below
                )
            )
            - k_y**2
            - k_z**2
        ) / (2.0*k_y)

        # The above has an inline sqrt(...) in your snippet:
        #   - k_y * sqrt( ... ) / (2*(k_x^2 + k_y^2))
        # so we replicate that carefully below:
        # Here is a more direct approach that matches your snippet exactly:

        # Actually let's define it exactly as your snippet "q_y = ..."
        # We'll do that in one line:
        # (But we have to embed sqrt_val carefully.)
        # The snippet is extremely long, so for clarity, let's keep it broken up:

        # Overwrite with a direct copy of "Solution branch 1" expression:
        qy_positive = (
            - gr**2
            - gz**2
            - 2*gz*k_z
            + k**2
            - k_x**2
            - 2*k_x * (
                k_x*(- gr**2 - gz**2 - 2*gz*k_z + k**2
                     - k_x**2 - k_y**2 - k_z**2)
                /(2*(k_x**2 + k_y**2))
                # minus the sqrt(...) portion:
                - k_y* sqrt(radical) /(2*(k_x**2 + k_y**2))
            )
            - k_y**2
            - k_z**2
        ) /(2*k_y)

        qx_positive = (
            k_x*(
                - gr**2 - gz**2 - 2*gz*k_z
                + k**2 - k_x**2 - k_y**2 - k_z**2
            )
            /(2*(k_x**2 + k_y**2))
            - k_y* sqrt(radical)
            /(2*(k_x**2 + k_y**2))
        )

        #--------------------------------------------------------------------
        # "Solution branch 2"
        #  q_y = [the big expression from your snippet for branch 2]
        #  q_x = [the big expression from your snippet for branch 2]
        #--------------------------------------------------------------------
        qy_negative = (
            - gr**2
            - gz**2
            - 2*gz*k_z
            + k**2
            - k_x**2
            - 2*k_x * (
                k_x*(- gr**2 - gz**2 - 2*gz*k_z
                     + k**2 - k_x**2 - k_y**2 - k_z**2)
                /(2*(k_x**2 + k_y**2))
                + k_y* sqrt(radical)/(2*(k_x**2 + k_y**2))
            )
            - k_y**2
            - k_z**2
        ) /(2*k_y)

        qx_negative = (
            k_x*(
                - gr**2 - gz**2 - 2*gz*k_z
                + k**2 - k_x**2 - k_y**2 - k_z**2
            )
            /(2*(k_x**2 + k_y**2))
            + k_y* sqrt(radical)
            /(2*(k_x**2 + k_y**2))
        )

        # Store them in the solutions array
        # sign=0 => "branch 1" (q_x_positive, q_y_positive)
        # sign=1 => "branch 2" (q_x_negative, q_y_negative)
        solutions[i, 0, 0] = qx_positive
        solutions[i, 0, 1] = qy_positive
        solutions[i, 0, 2] = gz

        solutions[i, 1, 0] = qx_negative
        solutions[i, 1, 1] = qy_negative
        solutions[i, 1, 2] = gz

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
    mosaic_angles,
    mosaic_intensities,
    debye_x, debye_y,
    center,
    theta_initial_deg,
    R_x_detector, R_z_detector, n_det_rot, Detector_Pos,
    e1_det, e2_det, R_z_R_y, R_ZY_n, P0, unit_x,
    save_flag, q_data, q_count, i_peaks_index
):
    """
    For a single reflection (H,K,L):
      1) Build G in crystal coords.
      2) Rotate once to sample coords (R_sample).
      3) For each mosaic angle => (gz, gr).
      4) Solve q from explicit formulas => fill image, optionally q_data.
    """
    # 1) Build G in crystal coords
    gz0 = 2.0*np.pi * (L/cv)
    gr0 = 4.0*np.pi / av * sqrt((H*H + H*K + K*K)/3.0)
    G0  = sqrt(gz0*gz0 + gr0*gr0)

    # 2) Sample rotation for the chosen theta_initial
    rad_theta_i = theta_initial_deg*(pi/180.0)
    R_x = np.array([
        [1.0, 0.0,               0.0],
        [0.0, cos(rad_theta_i), -sin(rad_theta_i)],
        [0.0, sin(rad_theta_i),  cos(rad_theta_i)]
    ])
    R_sample = R_x @ R_z_R_y

    # 3) Mosaic angle variation in the crystal frame
    gz_arr, gr_arr = G_rotation_array(gz0[2], gr0[1], mosaic_angles)
    # Then rotate each mosaic G-vector to the lab
    mosaic_vectors = np.column_stack((gr_arr, np.zeros_like(gr_arr), gz_arr))
    mosaic_vectors_lab = mosaic_vectors @ R_sample.T
    gr_values = mosaic_vectors_lab[:, 0]
    gz_values = mosaic_vectors_lab[:, 2]

    # Repetition arrays for weighting
    gz_rep = np.repeat(gz_values, 2)
    gr_rep = np.repeat(gr_values, 2)
    wt_rep = np.repeat(mosaic_intensities, 2)

    max_I_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan
    max_I_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan

    row_center = center[0]
    col_center = center[1]

    n_samp = beam_x_array.size
    for i_samp in prange(n_samp):
        lam_samp = wavelength_array[i_samp]
        if lam_samp < 1e-14:
            continue
        k_scat = 2.0*np.pi / lam_samp

        bx = beam_x_array[i_samp]
        by = beam_y_array[i_samp]
        beam_start = np.array([bx, -20e-3, -zb + by], dtype=np.float64)

        dtheta = theta_array[i_samp]
        dphi   = phi_array[i_samp]
        k_xi = cos(dtheta)* sin(dphi)
        k_yi = cos(dtheta)* cos(dphi)
        k_zi = sin(dtheta)
        k_in = np.array([k_xi, k_yi, k_zi], dtype=np.float64)

        n_surf = R_x @ R_ZY_n
        P0_rot = R_x @ P0
        P0_rot[0] = 0

        # Intersect incoming beam with sample plane
        I_plane, t_plane = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if I_plane is None or t_plane < 0.0:
            continue

        norm_in = sqrt(k_xi*k_xi + k_yi*k_yi + k_zi*k_zi)
        if norm_in < 1e-14:
            continue
        k_in_scat = k_in * (k_scat / norm_in)

        # 4) Solve for q using direct formula
        all_sol = solve_q(k_in_scat, k_scat, gz_values, gr_values, G0)
        n_mos = mosaic_angles.size
        sol_flat = all_sol.reshape((2*n_mos,3))

        N_sol= 2*n_mos
        k_out_det = np.empty((N_sol,3), dtype= np.float64)
        for i_sol in range(N_sol):
            Qx = sol_flat[i_sol,0]
            if np.isnan(Qx):
                k_out_det[i_sol,:] = np.nan
                continue
            Qy = sol_flat[i_sol,1]
            Qz = sol_flat[i_sol,2]
            k_out_det[i_sol,0] = k_in_scat[0] + Qx
            k_out_det[i_sol,1] = k_in_scat[1] + Qy
            k_out_det[i_sol,2] = k_in_scat[2] + Qz

        intersects, valid_mask= intersect_line_plane_batch(I_plane, k_out_det, Detector_Pos, n_det_rot)

        px_size= 100e-6
        for i_sol in range(N_sol):
            if not valid_mask[i_sol]:
                continue
            plane_to_det = intersects[i_sol,:] - Detector_Pos
            x_det = (plane_to_det[0]* e1_det[0]
                   + plane_to_det[1]* e1_det[1]
                   + plane_to_det[2]* e1_det[2])
            y_det = (plane_to_det[0]* e2_det[0]
                   + plane_to_det[1]* e2_det[1]
                   + plane_to_det[2]* e2_det[2])
            rpx= int(round(row_center - y_det/ px_size))
            cpx= int(round(col_center + x_det/ px_size))
            if rpx<0 or rpx>= image_size or cpx<0 or cpx>= image_size:
                continue

            gz_v= gz_rep[i_sol]
            gr_v= gr_rep[i_sol]
            wt_v= wt_rep[i_sol]
            val= (reflection_intensity *
                  wt_v *
                  np.exp(- (gz_v*gz_v)*(debye_x*debye_x)) *
                  np.exp(- (gr_v*gr_v)*(debye_y*debye_y)))
            image[rpx,cpx] += val

            sIdx = i_sol % 2
            if sIdx==0:
                if image[rpx,cpx] > max_I_sign0:
                    max_I_sign0= image[rpx,cpx]
                    max_x_sign0= cpx
                    max_y_sign0= rpx
            else:
                if image[rpx,cpx] > max_I_sign1:
                    max_I_sign1= image[rpx,cpx]
                    max_x_sign1= cpx
                    max_y_sign1= rpx

            if save_flag==1:
                ccount= q_count[i_peaks_index]
                if ccount< q_data.shape[1]:
                    Qx= sol_flat[i_sol,0]
                    Qy= sol_flat[i_sol,1]
                    Qz= sol_flat[i_sol,2]
                    q_data[i_peaks_index, ccount,0] = Qx
                    q_data[i_peaks_index, ccount,1] = Qy
                    q_data[i_peaks_index, ccount,2] = Qz
                    q_data[i_peaks_index, ccount,3] = val
                    q_data[i_peaks_index, ccount,4] = mosaic_angles[i_sol//2]
                    q_count[i_peaks_index]+=1

    return (max_I_sign0, max_x_sign0, max_y_sign0,
            max_I_sign1, max_x_sign1, max_y_sign1)


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
    Top-level driver:
      - Prepares rotation/detector geometry
      - Builds mosaic angles
      - Calls calculate_phi for each reflection
      - Returns final image and summary
    """
    # Convert angles to radians
    gamma_rad= gamma_deg*(pi/180.0)
    Gamma_rad= Gamma_deg*(pi/180.0)
    chi_rad= chi_deg*(pi/180.0)
    psi_rad= psi_deg*(pi/180.0)

    # Mosaic distribution parameters
    sigma_rad= sigma_pv_deg*(pi/180.0)
    gamma_rad_m= gamma_pv_deg*(pi/180.0)

    # Typically 99.9% coverage
    r_cut_gauss = sigma_rad* sqrt(2.0* np.log(1000.0))
    r_cut_lor   = 1000.0* gamma_rad_m
    r_cut= (1.0- eta_pv)* r_cut_gauss + eta_pv* r_cut_lor
    angle_cut_rad= r_cut/ sqrt(2.0)
    angle_cut_deg= angle_cut_rad*(180.0/ pi)

    # Step in degrees for mosaic sampling
    step_deg= 0.1
    temp= (2.0* angle_cut_deg)/ step_deg
    num_steps= int(temp) + 1
    mosaic_angles_deg= np.linspace(-angle_cut_deg, angle_cut_deg, num_steps)
    mosaic_angles= mosaic_angles_deg*(pi/180.0)

    # Build mosaic intensities
    mos_consts= find_mosaic_constants(sigma_rad, gamma_rad_m)
    mosaic_intens= radial_pseudo_voigt_2d(mosaic_angles, eta_pv, mos_consts)

    # Detector-plane transformations
    cg= cos(gamma_rad); sg= sin(gamma_rad)
    cG= cos(Gamma_rad); sG= sin(Gamma_rad)
    R_x_det= np.array([
        [1.0, 0.0, 0.0],
        [0.0, cg,  sg],
        [0.0,-sg,  cg]
    ])
    R_z_det= np.array([
        [ cG, sG, 0.0],
        [-sG, cG, 0.0],
        [ 0.0,0.0,1.0]
    ])
    nd_temp= R_x_det @ n_detector
    n_det_rot= R_z_det @ nd_temp
    nd_len= sqrt(n_det_rot[0]**2+ n_det_rot[1]**2+ n_det_rot[2]**2)
    n_det_rot/= nd_len

    Detector_Pos= np.array([0.0, Distance_CoR_to_Detector, 0.0], dtype= np.float64)
    dot_e1= unit_x[0]* n_det_rot[0] + unit_x[1]* n_det_rot[1] + unit_x[2]* n_det_rot[2]
    e1_det= unit_x - dot_e1* n_det_rot
    e1_len= sqrt(e1_det[0]**2 + e1_det[1]**2 + e1_det[2]**2)
    if e1_len<1e-14:
        e1_det= np.array([1.0,0.0,0.0])
    else:
        e1_det/= e1_len

    tmpx= n_det_rot[1]* e1_det[2] - n_det_rot[2]* e1_det[1]
    tmpy= n_det_rot[2]* e1_det[0] - n_det_rot[0]* e1_det[2]
    tmpz= n_det_rot[0]* e1_det[1] - n_det_rot[1]* e1_det[0]
    e2_det= np.array([-tmpx, -tmpy, -tmpz], dtype= np.float64)
    e2_len= sqrt(e2_det[0]**2+ e2_det[1]**2+ e2_det[2]**2)
    if e2_len<1e-14:
        e2_det= np.array([0.0,1.0,0.0])
    else:
        e2_det/= e2_len

    # Sample orientation wrt chi,psi
    c_chi= cos(chi_rad); s_chi= sin(chi_rad)
    R_y= np.array([
        [ c_chi, 0.0, s_chi],
        [ 0.0,   1.0, 0.0],
        [-s_chi, 0.0, c_chi]
    ])
    c_psi= cos(psi_rad); s_psi= sin(psi_rad)
    R_z= np.array([
        [ c_psi, s_psi, 0.0],
        [-s_psi, c_psi, 0.0],
        [ 0.0,   0.0,   1.0]
    ])
    R_z_R_y= R_z @ R_y
    n1= np.array([0.0,0.0,1.0], dtype= np.float64)
    R_ZY_n= R_z_R_y @ n1
    nzy_len= sqrt(R_ZY_n[0]**2+ R_ZY_n[1]**2+ R_ZY_n[2]**2)
    R_ZY_n/= nzy_len

    # sample plane in lab => used for intersection
    P0= np.array([0.0,0.0,-zs], dtype= np.float64)

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

        (mx0,my0,mv0,
         mx1,my1,mv1)= calculate_phi(
            H,K,L, av, cv,
            wavelength_array,
            image, image_size,
            gamma_rad, Gamma_rad, c_chi, s_psi,
            zs, zb, n2,
            beam_x_array, beam_y_array,
            theta_array, phi_array,
            reflI,
            mosaic_angles, mosaic_intens,
            debye_x, debye_y,
            center,
            theta_initial_deg,
            R_x_det, R_z_det, n_det_rot, Detector_Pos,
            e1_det, e2_det, R_z_R_y, R_ZY_n, P0, unit_x,
            save_flag, q_data, q_count, i_pk
        )

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
