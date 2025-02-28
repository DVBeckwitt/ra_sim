import numpy as np
from numba import njit, prange
from math import sin, cos, sqrt, pi

@njit
def find_mosaic_constants(sigma_rad, gamma_rad):
    """
    Returns constants for 2D pseudo-Voigt:
      G(r)=1/(2πσ²)*exp[-r²/(2σ²)]
      L(r)=1/(2π)*(γ/(r²+γ²)^(3/2)).

    sigma_rad: Gaussian sigma in radians
    gamma_rad: Lorentzian half-width in radians
    """
    # Gaussian prefactor => 1/(2πσ²)
    G2D_const = 1.0/(2.0 * np.pi * sigma_rad * sigma_rad)
    # In the exponent => r²/(2σ²)
    G2D_exp   = 1.0/(2.0 * sigma_rad * sigma_rad)

    # Lorentzian => 1/(2π)*(γ/(r²+γ²)^(3/2))
    L2D_const = 1.0/(2.0 * np.pi)

    return (G2D_const, G2D_exp, L2D_const, gamma_rad)


@njit
def radial_pseudo_voigt_2d(mosaic_angles, eta, mosaic_consts):
    """
    radial 2D pseudo-Voigt => shape(n,).
    For each angle α => radial distance r = sqrt(2)*|α| (2D).
    mosaic_consts = (G2D_const, G2D_exp, L2D_const, gamma).

    We combine the Gaussian and Lorentzian parts with weight (1-eta) and eta.
    """
    G2D_const, G2D_exp, L2D_const, gamma = mosaic_consts
    n = mosaic_angles.size
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        alpha = mosaic_angles[i]
        # radial distance in 2D => r = sqrt(2) * |alpha|
        r = sqrt(2.0) * abs(alpha)

        # Gaussian part => G2D_const * exp( - (r^2) * G2D_exp )
        g_val = G2D_const * np.exp(-(r*r) * G2D_exp)

        # Lorentz part => L2D_const * (gamma / (r^2 + gamma^2)^(3/2))
        denom = (r*r + gamma*gamma)**1.5
        l_val = L2D_const * (gamma / denom)

        # Weighted sum
        out[i] = (1.0 - eta)*g_val + eta*l_val

    return out


@njit
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect a parametric line (P0 + t*k_vec) with plane defined by (P_plane, n_plane).
    If denom ~ 0 => no intersection or line parallel. If t<0 => behind P0 => invalid.

    Returns (pt, t) or (None, None).
    """
    denom = (k_vec[0]*n_plane[0] +
             k_vec[1]*n_plane[1] +
             k_vec[2]*n_plane[2])
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
    Batch version: for each direction directions[i,:], compute intersection
    with plane. Return (intersect_points, valid_mask).
    If no intersection or t<0 => mask=False and that row in intersects=NaN.
    """
    N = directions.shape[0]
    intersects = np.full((N,3), np.nan, dtype=np.float64)
    valid = np.zeros(N, dtype=np.bool_)

    for i in range(N):
        kx = directions[i,0]
        ky = directions[i,1]
        kz = directions[i,2]

        dot_dn = (kx*plane_n[0] + ky*plane_n[1] + kz*plane_n[2])
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

from numba import njit
import numpy as np

@njit
def solve_q(k_in, k, gz_values, gr_values, G, eps=1e-14):
    k_x, k_y, k_z = k_in
    n_mos = gz_values.size

    solutions = np.full((n_mos, 2, 3), np.nan, dtype=np.float64)

    kxy2 = k_x**2 + k_y**2
    kxyz2 = kxy2 + k_z**2
    k2 = k**2

    for i in range(n_mos):
        gz = gz_values[i]
        gr = gr_values[i]

        gz2 = gz**2
        gr2 = gr**2
        common_term = gr2 + gz2 + 2*gz*k_z - k2 + kxyz2

        radical = (
            -gr2**2 - 2*gr2*gz2 - 4*gr2*gz*k_z + 2*gr2*(k2 + kxy2 - k_z**2)
            - gz2**2 - 4*gz2*gz*k_z + 2*gz2*(k2 - kxy2 - 3*k_z**2)
            + 4*gz*k2*k_z - 4*gz*k_z*(kxy2 + k_z**2)
            - k2**2 + 2*k2*kxyz2 - k_x**4 - 2*k_x**2*(k_y**2 + k_z**2) - k_y**4 - 2*k_y**2*k_z**2 - k_z**4
        )

        if radical < 0.0:
            continue

        sqrt_radical = np.sqrt(radical)

        # Positive branch
        qx_positive = (-k_x*common_term - k_y*sqrt_radical) / (2*kxy2)
        qy_positive = (-common_term - 2*k_x*qx_positive) / (2*k_y)

        # Negative branch
        qx_negative = (-k_x*common_term + k_y*sqrt_radical) / (2*kxy2)
        qy_negative = (-common_term - 2*k_x*qx_negative) / (2*k_y)

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
    e1_det, e2_det,
    R_z_R_y,       # <--- consistent name for combined rotation
    R_ZY_n,        # sample-plane normal in sample coords
    P0, unit_x,
    save_flag, q_data, q_count, i_peaks_index
):
    """
    For a single reflection (H,K,L), we do:
      1) Build G in crystal coords => (grC,0,gzC).
      2) Rotate the sample by theta_initial => R_x @ R_z_R_y => R_sample.
      3) For each beam sample, compute wavevector with partial refraction.
      4) Solve q from solve_q => store in image or q_data if requested.
    """
    # 1) Build G in crystal coords
    gzC = 2.0*np.pi * (L/cv)
    grC = 4.0*np.pi/av * sqrt((H*H + H*K + K*K)/3.0)
    G0  = sqrt(gzC*gzC + grC*grC)
    n2 = 1
    # 2) Create rotation for the chosen theta_initial
    rad_theta_i = theta_initial_deg*(pi/180.0)
    R_x = np.array([
        [1.0, 0.0,               0.0],
        [0.0, cos(rad_theta_i), -sin(rad_theta_i)],
        [0.0, sin(rad_theta_i),  cos(rad_theta_i)]
    ])
    R_sample = R_x @ R_z_R_y

    # sample-plane normal in lab
    n_surf = R_x @ R_ZY_n

    # We'll do a trivial mosaic in GZ,GR => 0.0, but keep mosaic weighting
    n_mos = mosaic_angles.size
    gz_values = np.zeros(n_mos, dtype=np.float64)
    gr_values = np.zeros(n_mos, dtype=np.float64)
    wt_rep = np.repeat(mosaic_intensities, 2)

    # track maximum intensities
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

        # Starting beam point in lab
        bx = beam_x_array[i_samp]
        by = beam_y_array[i_samp]
        beam_start = np.array([bx, -20e-3, -zb + by], dtype=np.float64)

        # build direction from (theta, phi)
        dtheta = theta_array[i_samp]
        dphi   = phi_array[i_samp]
        k_xi = cos(dtheta)* sin(dphi)
        k_yi = cos(dtheta)* cos(dphi)
        k_zi = sin(dtheta)
        k_in = np.array([k_xi, k_yi, k_zi], dtype=np.float64)

        # 3) intersect incoming beam with sample plane
        P0_rot = R_x @ P0
        P0_rot[0] = 0.0
        I_plane, t_plane = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if I_plane is None or t_plane < 0.0:
            continue

        # wavevector magnitude in vacuum => 2π / λ
        k_scat = 2.0*np.pi / lam_samp

        # Normalize incoming to length k_scat
        norm_in = sqrt(k_in[0]**2 + k_in[1]**2 + k_in[2]**2)
        if norm_in<1e-14:
            continue
        k_in_scat = k_in*(k_scat / norm_in)

        # partial refraction logic
        # dot with normal axis from R_sample => let's assume that's R_sample[:,2]
        kn_dot = (k_in_scat[0]*R_sample[0,2] +
                  k_in_scat[1]*R_sample[1,2] +
                  k_in_scat[2]*R_sample[2,2])

        th_i_prime = (pi/2.0) - np.arccos(kn_dot)

        # project incident onto plane => define p1,p2 for azimuth
        projected_incident = k_in_scat - kn_dot*R_sample[:,2]
        
        proj_len = sqrt(projected_incident[0]**2 +
                        projected_incident[1]**2 +
                        projected_incident[2]**2)
        if proj_len>1e-14:
            projected_incident /= proj_len
        else:
            projected_incident[:] = 0.0

        # define p1,p2 in R_sample basis => one approach
        # you might do R_sample.T @ projected_incident, but let's do manual:
        # for simplicity, let's treat R_sample[:,0],R_sample[:,1],R_sample[:,2]
        p1 = (projected_incident[0]*R_sample[0,0] +
              projected_incident[1]*R_sample[1,0] +
              projected_incident[2]*R_sample[2,0])
        p2 = (projected_incident[0]*R_sample[0,1] +
              projected_incident[1]*R_sample[1,1] +
              projected_incident[2]*R_sample[2,1])

        phi_i_prime = (pi/2.0) - np.arctan2(p2, p1)

        # index of refraction => n2
        c_i = cos(th_i_prime)
        n_real = np.real(n2)
        if abs(n_real)<1e-14:
            continue
        ratio = c_i / n_real
        if abs(ratio)>1.0:
            # total external reflection => skip
            continue

        # transmitted angle => th_t
        th_t = np.arccos(ratio)*np.sign(th_i_prime)

        # define k_re, k_im as from your snippet => e.g.:
        # k_re = k_in**2 * np.real(n2**2)
        # but let's do something consistent with your code:
        mod_k_in = sqrt(k_in[0]**2 + k_in[1]**2 + k_in[2]**2)
        k_re = (mod_k_in**2)* (np.real(n2*n2))
        k_im = (mod_k_in**2)* (np.imag(n2*n2))

        k_x_scatter = k_scat*cos(th_t)* sin(phi_i_prime)
        k_y_scatter = k_scat*cos(th_t)* cos(phi_i_prime)

        # define some partial approach to imaginary k_z
        af  = k_re*(sin(th_t)**2)
        b   = k_im
        # example from snippet => imaginary_k_z_scatter
        imaginary_k_z_scatter = sqrt((sqrt(af*af + b*b) - af)/2)

        # you might define k_z_scatter => or do a second approach:
        k_z_scatter = k_scat*sin(th_t)
        k_in_scat = np.array([k_x_scatter, k_y_scatter, k_z_scatter], dtype=np.float64)
        # 4) Solve for q using direct formula => solve_q
        # note we pass (k_in_scat, k_scat, gz_values=0, gr_values=0, G0)
        all_sol = solve_q(k_in_scat, k_scat, gz_values, gr_values, G0)
        sol_flat = all_sol.reshape((2*n_mos, 3))
        N_sol = 2*n_mos

        # Build scattered wave => k_out_det
        k_out_det = np.empty((N_sol,3), dtype=np.float64)
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

        # 5) Intersect scattered with detector plane
        intersects, valid_mask = intersect_line_plane_batch(I_plane, k_out_det,
                                                            Detector_Pos, n_det_rot)

        px_size= 100e-6
        for i_sol in range(N_sol):
            if not valid_mask[i_sol]:
                continue

            # final offset from detector pos
            plane_to_det = intersects[i_sol,:] - Detector_Pos
            x_det = (plane_to_det[0]* e1_det[0] +
                     plane_to_det[1]* e1_det[1] +
                     plane_to_det[2]* e1_det[2])
            y_det = (plane_to_det[0]* e2_det[0] +
                     plane_to_det[1]* e2_det[1] +
                     plane_to_det[2]* e2_det[2])

            rpx= int(round(row_center - y_det/ px_size))
            cpx= int(round(col_center + x_det/ px_size))
            if (rpx<0 or rpx>= image_size or
                cpx<0 or cpx>= image_size):
                continue

            sIdx = i_sol % 2
            mos_idx = i_sol // 2

            Qx = sol_flat[i_sol,0]
            Qy = sol_flat[i_sol,1]
            Qz = sol_flat[i_sol,2]

            # mosaic weighting
            wt_v = wt_rep[i_sol]

            # example intensity with  Debye factors if you want
            val = (reflection_intensity * wt_v *
                   np.exp(-(Qz*Qz)*(debye_x*debye_x)) *
                   np.exp(-(Qx*Qx + Qy*Qy)*(debye_y*debye_y)))

            image[rpx,cpx] += val

            # track maximum intensities for each sign
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

            # optional save
            if save_flag==1:
                ccount= q_count[i_peaks_index]
                if ccount< q_data.shape[1]:
                    q_data[i_peaks_index, ccount,0] = Qx
                    q_data[i_peaks_index, ccount,1] = Qy
                    q_data[i_peaks_index, ccount,2] = Qz
                    q_data[i_peaks_index, ccount,3] = val
                    q_data[i_peaks_index, ccount,4] = mosaic_angles[mos_idx]
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
    1) Convert angles => gamma_rad, Gamma_rad, chi_rad, psi_rad
    2) Build mosaic distribution => mosaic_angles, mosaic_intens
    3) Build detector transformations => R_x_det, R_z_det => n_det_rot, etc.
    4) Build sample orientation => R_z_R_y, normal => R_ZY_n
    5) Loop over reflections => call calculate_phi
    6) Return final image, plus q_data if save_flag=1
    """
    gamma_rad= gamma_deg*(pi/180.0)
    Gamma_rad= Gamma_deg*(pi/180.0)
    chi_rad= chi_deg*(pi/180.0)
    psi_rad= psi_deg*(pi/180.0)

    # Build mosaic distribution
    sigma_rad= sigma_pv_deg*(pi/180.0)
    gamma_rad_m= gamma_pv_deg*(pi/180.0)
    # coverage ~ 99.9% => combine r_cut_gauss, r_cut_lor
    r_cut_gauss = sigma_rad* sqrt(2.0* np.log(1000.0))
    r_cut_lor   = 1000.0* gamma_rad_m
    r_cut= (1.0 - eta_pv)*r_cut_gauss + eta_pv*r_cut_lor
    angle_cut_rad= r_cut/ sqrt(2.0)
    angle_cut_deg= angle_cut_rad*(180.0/pi)
    step_deg= 0.1
    temp= (2.0* angle_cut_deg)/ step_deg
    num_steps= int(temp) + 1
    mosaic_angles_deg= np.linspace(-angle_cut_deg, angle_cut_deg, num_steps)
    mosaic_angles= mosaic_angles_deg*(pi/180.0)

    mos_consts= find_mosaic_constants(sigma_rad, gamma_rad_m)
    mosaic_intens= radial_pseudo_voigt_2d(mosaic_angles, eta_pv, mos_consts)

    # Detector-plane transformations => R_x_det, R_z_det, etc.
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
        [ 0.0, 0.0,1.0]
    ])

    nd_temp= R_x_det @ n_detector
    n_det_rot= R_z_det @ nd_temp
    nd_len= sqrt(n_det_rot[0]*n_det_rot[0] +
                 n_det_rot[1]*n_det_rot[1] +
                 n_det_rot[2]*n_det_rot[2])
    n_det_rot /= nd_len

    Detector_Pos= np.array([0.0, Distance_CoR_to_Detector, 0.0], dtype=np.float64)

    # define e1_det => projection of unit_x onto plane orthonormal to n_det_rot
    dot_e1= (unit_x[0]*n_det_rot[0] +
             unit_x[1]*n_det_rot[1] +
             unit_x[2]*n_det_rot[2])
    e1_det= unit_x - dot_e1*n_det_rot
    e1_len= sqrt(e1_det[0]**2 + e1_det[1]**2 + e1_det[2]**2)
    if e1_len<1e-14:
        e1_det= np.array([1.0,0.0,0.0])
    else:
        e1_det/= e1_len

    # define e2_det => cross( -n_det_rot, e1_det ), or some approach
    tmpx= n_det_rot[1]* e1_det[2] - n_det_rot[2]* e1_det[1]
    tmpy= n_det_rot[2]* e1_det[0] - n_det_rot[0]* e1_det[2]
    tmpz= n_det_rot[0]* e1_det[1] - n_det_rot[1]* e1_det[0]
    e2_det= np.array([-tmpx, -tmpy, -tmpz], dtype=np.float64)

    e2_len= sqrt(e2_det[0]**2 + e2_det[1]**2 + e2_det[2]**2)
    if e2_len<1e-14:
        e2_det= np.array([0.0,1.0,0.0])
    else:
        e2_det/= e2_len

    # Sample orientation => build R_y, R_z => R_z_R_y
    c_chi= cos(chi_rad); s_chi= sin(chi_rad)
    R_y= np.array([
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
    n1= np.array([0.0,0.0,1.0], dtype= np.float64)
    R_ZY_n= R_z_R_y @ n1
    nzy_len= sqrt(R_ZY_n[0]**2 + R_ZY_n[1]**2 + R_ZY_n[2]**2)
    R_ZY_n/= nzy_len

    # sample plane offset => P0
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
            reflI,
            mosaic_angles, mosaic_intens,
            debye_x, debye_y,
            center,
            theta_initial_deg,
            R_x_det, R_z_det, n_det_rot, Detector_Pos,
            e1_det, e2_det,
            R_z_R_y,       # <--- consistent name
            R_ZY_n,
            P0, unit_x,
            save_flag, q_data, q_count, i_pk
        )

        # store maximum positions for each reflection
        max_positions[i_pk,0] = mx0
        max_positions[i_pk,1] = my0
        max_positions[i_pk,2] = mv0
        max_positions[i_pk,3] = mx1
        max_positions[i_pk,4] = my1
        max_positions[i_pk,5] = mv1

    # final return
    if save_flag==1:
        return image, max_positions, q_data, q_count
    else:
        return image, max_positions, None, None
