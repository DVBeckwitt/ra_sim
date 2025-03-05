import numpy as np
from numba import njit, prange
from math import sin, cos, sqrt, pi, exp, acos
import numpy as np
from numba import njit
from math import sin, cos, sqrt, pi, exp

@njit
def rotation_matrix_y_numba(theta):
    """Return the 3×3 rotation matrix about the y-axis (Numba-friendly)."""
    return np.array([
        [cos(theta),  0.0, sin(theta)],
        [0.0,         1.0, 0.0       ],
        [-sin(theta), 0.0, cos(theta)]
    ])

@njit
def rotation_matrix_z_numba(phi):
    """Return the 3×3 rotation matrix about the z-axis (Numba-friendly)."""
    return np.array([
        [ cos(phi), -sin(phi), 0.0],
        [ sin(phi),  cos(phi), 0.0],
        [ 0.0,       0.0,      1.0]
    ])

@njit
def rotate_G(G, sigma, N=100, M=100, n_phi=100):
    """
    Reproduces the exact logic of the old code to generate a big array of all
    ring-points, base-points, original ring, and G itself.

    Returns an array of shape:
        (N*M*n_phi + N*M + n_phi + 1, 3)

    Steps matching the old logic:
      1) Sample theta ~ N(0, sigma)  [N total].
      2) Sample phi_offsets ~ uniform in [0,2π)  [M total].
      3) Define a fine phi_range ~ [0,2π]  [n_phi points, with endpoint=True].
      4) Build rotation matrices, ring basis, etc.
      5) Compute ring_points and base_points.
      6) Flatten + concatenate with original_ring and single G.
    """
    # ------------------------------------------------------
    # 1) Sample thetas from Gaussian, 2) phi_offsets uniform
    # ------------------------------------------------------
    theta_samples = np.random.normal(0.0, sigma, N)

    # Manually build M offsets in [0,2π), ignoring endpoint:
    phi_offsets = np.empty(M, dtype=np.float64)
    for i in range(M):
        phi_offsets[i] = 2.0*pi*i / M

    # For the "fine ring" we do want to include endpoint => n_phi points from 0..2π:
    # e.g. if n_phi=100, it includes the last point 2π exactly.
    phi_range = np.empty(n_phi, dtype=np.float64)
    if n_phi > 1:
        step = 2.0*pi / (n_phi - 1)
        for i in range(n_phi):
            phi_range[i] = step * i
    else:
        # trivial case, just 1 point at 0.0
        phi_range[0] = 0.0

    # -------------------------------------
    # 3) Precompute rotation matrices
    # -------------------------------------
    # R_y_all => shape (N, 3, 3)
    R_y_all = np.empty((N, 3, 3), dtype=np.float64)
    for i in range(N):
        R_y_all[i] = rotation_matrix_y_numba(theta_samples[i])

    # Rz_offset_all => shape (M, 3, 3)
    Rz_offset_all = np.empty((M, 3, 3), dtype=np.float64)
    for j in range(M):
        Rz_offset_all[j] = rotation_matrix_z_numba(phi_offsets[j])

    # ring_basis => shape (n_phi, 3)
    ring_basis = np.empty((n_phi, 3), dtype=np.float64)
    for k in range(n_phi):
        Rz_k = rotation_matrix_z_numba(phi_range[k])
        ring_basis[k] = Rz_k @ G

    # ---------------------------------------------------------
    # 4) Build combined transformations & compute ring points
    #    T[i,j] = Rz_offset_all[j] @ R_y_all[i]
    # ---------------------------------------------------------
    # ring_points => shape (N, M, n_phi, 3)
    ring_points = np.empty((N, M, n_phi, 3), dtype=np.float64)
    # base_points => shape (N, M, 3)
    base_points = np.empty((N, M, 3), dtype=np.float64)

    for i in range(N):
        for j in range(M):
            # T_ij = Rz_offset_all[j] @ R_y_all[i]
            # We'll apply T_ij to ring_basis and to G.
            # Because we can't store or multiply big arrays inline with einsum under Numba,
            # we do explicit loops:
            T_ij = Rz_offset_all[j] @ R_y_all[i]

            # ring_points for each phi
            for k in range(n_phi):
                ring_points[i, j, k, :] = T_ij @ ring_basis[k]
            # base_point for (i,j)
            base_points[i, j, :] = T_ij @ G

    # -----------------------------------------------------
    # 5) Flatten & combine [ring_points, base_points, original_ring, G]
    # -----------------------------------------------------
    # ring_points_flat => (N*M*n_phi, 3)
    ring_points_flat = np.empty((N*M*n_phi, 3), dtype=np.float64)
    idx = 0
    for i in range(N):
        for j in range(M):
            for k in range(n_phi):
                ring_points_flat[idx, :] = ring_points[i, j, k, :]
                idx += 1

    # base_points_flat => (N*M, 3)
    base_points_flat = np.empty((N*M, 3), dtype=np.float64)
    idx = 0
    for i in range(N):
        for j in range(M):
            base_points_flat[idx, :] = base_points[i, j, :]
            idx += 1

    # original_ring => shape (n_phi, 3)
    original_ring = np.empty((n_phi, 3), dtype=np.float64)
    for k in range(n_phi):
        Rz_k = rotation_matrix_z_numba(phi_range[k])
        original_ring[k] = Rz_k @ G

    # Single G => shape (1, 3)
    G_array = G.reshape(1, 3)

    # total # of rows in final big_array
    total_rows = (N*M*n_phi) + (N*M) + n_phi + 1
    big_array = np.empty((total_rows, 3), dtype=np.float64)

    # Fill big_array
    idx = 0
    # 1) ring_points_flat
    for r in range(N*M*n_phi):
        big_array[idx, :] = ring_points_flat[r, :]
        idx += 1
    # 2) base_points_flat
    for r in range(N*M):
        big_array[idx, :] = base_points_flat[r, :]
        idx += 1
    # 3) original_ring
    for r in range(n_phi):
        big_array[idx, :] = original_ring[r, :]
        idx += 1
    # 4) single G
    big_array[idx, :] = G_array[0, :]

    return big_array



##############################################################################
# 3) Intersect line-plane (unchanged)
##############################################################################
@njit
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    denom = k_vec[0]*n_plane[0] + k_vec[1]*n_plane[1] + k_vec[2]*n_plane[2]
    if abs(denom) < 1e-14:
        return (np.nan, np.nan, np.nan, False)

    num = ((P_plane[0] - P0[0]) * n_plane[0]
         + (P_plane[1] - P0[1]) * n_plane[1]
         + (P_plane[2] - P0[2]) * n_plane[2])
    t = num/denom
    if t < 0.0:
        return (np.nan, np.nan, np.nan, False)

    ix = P0[0] + t*k_vec[0]
    iy = P0[1] + t*k_vec[1]
    iz = P0[2] + t*k_vec[2]
    return (ix, iy, iz, True)

@njit
def intersect_line_plane_batch(start_pt, directions, plane_pt, plane_n):
    Ndir = directions.shape[0]
    intersects = np.full((Ndir,3), np.nan, dtype=np.float64)
    valid = np.zeros(Ndir, dtype=np.bool_)
    for i in range(Ndir):
        kx = directions[i,0]
        ky = directions[i,1]
        kz = directions[i,2]
        dot_dn = kx*plane_n[0] + ky*plane_n[1] + kz*plane_n[2]
        if abs(dot_dn) < 1e-14:
            continue

        num = ((plane_pt[0]-start_pt[0])*plane_n[0]
             + (plane_pt[1]-start_pt[1])*plane_n[1]
             + (plane_pt[2]-start_pt[2])*plane_n[2])
        t = num/dot_dn
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

##############################################################################
# 4) solve_q
##############################################################################
@njit
def solve_q(k_in, k, g, G_rotated, eps=1e-14):
    """
    For each mosaic sample i (each row in G_rotated):
    returns shape (N,2,3).
    """
    kx, ky, kz = k_in
    Gx = G_rotated[:,0]
    Gy = G_rotated[:,1]
    qz = G_rotated[:,2]

    N_samples = G_rotated.shape[0]
    solutions = np.empty((N_samples,2,3), dtype=np.float64)

    G_ideal_sq = Gx*Gx + Gy*Gy + qz*qz

    kx2 = kx*kx
    ky2 = ky*ky
    kz2 = kz*kz

    alpha0 = 0.5*(k*k - G_ideal_sq - (kx2+ky2+kz2))
    L = kx2+ky2

    if abs(ky)<eps:
        for i in range(N_samples):
            R_sq = G_ideal_sq[i] - qz[i]*qz[i]
            if R_sq<eps:
                solutions[i,0,:] = np.nan
                solutions[i,1,:] = np.nan
                continue
            alpha = alpha0[i] - qz[i]*kz
            if abs(kx)<eps:
                solutions[i,0,:] = np.nan
                solutions[i,1,:] = np.nan
                continue
            qx0 = alpha/kx
            tmp = R_sq - qx0*qx0
            if tmp<0:
                solutions[i,0,:] = np.nan
                solutions[i,1,:] = np.nan
                continue
            sqrt_tmp = sqrt(tmp)
            solutions[i,0,0] = qx0
            solutions[i,0,1] = +sqrt_tmp
            solutions[i,0,2] = qz[i]
            solutions[i,1,0] = qx0
            solutions[i,1,1] = -sqrt_tmp
            solutions[i,1,2] = qz[i]
        return solutions

    r = kx/ky
    for i in range(N_samples):
        R_sq = G_ideal_sq[i] - qz[i]*qz[i]
        if R_sq<eps:
            solutions[i,0,:] = np.nan
            solutions[i,1,:] = np.nan
            continue
        alpha = alpha0[i] - qz[i]*kz
        disc = R_sq*L - alpha*alpha
        if disc<0:
            solutions[i,0,:] = np.nan
            solutions[i,1,:] = np.nan
            continue
        sqrt_disc = sqrt(disc)
        denom = ky*(r*r +1.0)
        qxp = (alpha*r + sqrt_disc)/denom
        qxm = (alpha*r - sqrt_disc)/denom
        qyp = (alpha - kx*qxp)/ky
        qym = (alpha - kx*qxm)/ky

        solutions[i,0,0] = qxp
        solutions[i,0,1] = qyp
        solutions[i,0,2] = qz[i]
        solutions[i,1,0] = qxm
        solutions[i,1,1] = qym
        solutions[i,1,2] = qz[i]

    return solutions

##############################################################################
# 5) calculate_phi
##############################################################################
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
    Mosaic_Rotation,  # shape (N*M, 3)
    debye_x, debye_y,
    center,
    theta_initial_deg,
    R_x_detector, R_z_detector, n_det_rot, Detector_Pos,
    e1_det, e2_det,
    R_z_R_y,
    R_ZY_n,
    P0, unit_x,
    save_flag, q_data, q_count, i_peaks_index
):
    gz0 = 2.0*pi*(L/cv)
    gr0 = 4.0*pi/av * sqrt((H*H + H*K + K*K)/3.0)
    G   = sqrt(gr0*gr0 + gz0*gz0)
    n2  = 1

    max_I_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan
    max_I_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan

    rad_theta_i = theta_initial_deg*(pi/180.0)
    R_x = np.array([
        [1.0,              0.0,                0.0],
        [0.0,  cos(rad_theta_i), -sin(rad_theta_i)],
        [0.0,  sin(rad_theta_i),  cos(rad_theta_i)]
    ])
    R_sample = R_x @ R_z_R_y

    n_surf = R_x @ R_ZY_n
    n_surf /= sqrt(n_surf[0]*n_surf[0] + n_surf[1]*n_surf[1] + n_surf[2]*n_surf[2])

    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0

    n_samp = beam_x_array.size

    u_ref = np.array([0.0,0.0,-1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_norm = sqrt(e1_temp[0]*e1_temp[0] + e1_temp[1]*e1_temp[1] + e1_temp[2]*e1_temp[2])
    if e1_norm<1e-12:
        alt_refs= [
            np.array([1.0,0.0,0.0]),
            np.array([0.0,1.0,0.0]),
            np.array([0.0,0.0,1.0])
        ]
        success=False
        for ar in alt_refs:
            cross_tmp = np.cross(n_surf, ar)
            cross_norm_tmp = sqrt(cross_tmp[0]*cross_tmp[0] + cross_tmp[1]*cross_tmp[1] + cross_tmp[2]*cross_tmp[2])
            if cross_norm_tmp>1e-12:
                e1_temp = cross_tmp/cross_norm_tmp
                success=True
                break
        if not success:
            return (max_x_sign0, max_y_sign0, max_I_sign0,
                    max_x_sign1, max_y_sign1, max_I_sign1)
    else:
        e1_temp /= e1_norm

    e2_temp = np.cross(n_surf, e1_temp)

    for i_samp in prange(n_samp):
        lam_samp = wavelength_array[i_samp]
        k_mag = 2.0*pi / lam_samp

        bx = beam_x_array[i_samp]
        by = beam_y_array[i_samp]
        beam_start = np.array([bx, -20e-3, -zb + by], dtype=np.float64)

        dtheta = theta_array[i_samp]
        dphi   = phi_array[i_samp]
        k_in = np.array([
            cos(dtheta)*sin(dphi),
            cos(dtheta)*cos(dphi),
            sin(dtheta)
        ], dtype=np.float64)

        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if not valid_int:
            continue

        I_plane = np.array([ix, iy, iz])
        kn_dot = k_in[0]*n_surf[0] + k_in[1]*n_surf[1] + k_in[2]*n_surf[2]
        th_i_prime = (pi/2.0) - acos(kn_dot)

        projected_incident = k_in - kn_dot*n_surf
        pln = sqrt(projected_incident[0]**2 + projected_incident[1]**2 + projected_incident[2]**2)
        if pln>1e-12:
            projected_incident/=pln
        else:
            projected_incident[:]=0.0

        p1 = projected_incident[0]*e1_temp[0] + projected_incident[1]*e1_temp[1] + projected_incident[2]*e1_temp[2]
        p2 = projected_incident[0]*e2_temp[0] + projected_incident[1]*e2_temp[1] + projected_incident[2]*e2_temp[2]
        phi_i_prime = (pi/2.0) - np.arctan2(p2, p1)
        th_t = acos(cos(th_i_prime)/np.real(n2))*np.sign(th_i_prime)

        k_scat = k_mag*sqrt(np.real(n2)*np.real(n2))
        k_x_scat = k_scat*cos(th_t)*sin(phi_i_prime)
        k_y_scat = k_scat*cos(th_t)*cos(phi_i_prime)
        k_z_scat = k_scat*sin(th_t)
        k_in_crystal= np.array([k_x_scat, k_y_scat, k_z_scat])

        All_Q = solve_q(k_in_crystal, k_scat, G, Mosaic_Rotation)
        All_Q_flat = All_Q.reshape((-1,3))

        for i_sol in range(All_Q_flat.shape[0]):
            Qx, Qy, Qz = All_Q_flat[i_sol]
            if np.isnan(Qx) or np.isnan(Qy) or np.isnan(Qz):
                continue
            k_tx_prime = Qx + k_x_scat
            k_ty_prime = Qy + k_y_scat
            k_tz_prime = Qz + k_z_scat

            kr = sqrt(k_tx_prime*k_tx_prime + k_ty_prime*k_ty_prime)
            if kr<1e-12:
                twotheta_t=0.0
            else:
                twotheta_t= np.arctan(k_tz_prime/kr)

            phi_f= np.arctan2(k_tx_prime, k_ty_prime)
            k_tx_f = k_scat*cos(twotheta_t)*sin(phi_f)
            k_ty_f = k_scat*cos(twotheta_t)*cos(phi_f)
            real_k_tz_f= k_scat*sin(twotheta_t)
            kf= np.array([k_tx_f, k_ty_f, real_k_tz_f])
            kf_prime = R_sample @ kf

            dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
            if not valid_det:
                continue

            plane_to_det = np.array([dx - Detector_Pos[0],
                                     dy - Detector_Pos[1],
                                     dz - Detector_Pos[2]], dtype=np.float64)
            x_det = plane_to_det[0]*e1_det[0] + plane_to_det[1]*e1_det[1] + plane_to_det[2]*e1_det[2]
            y_det = plane_to_det[0]*e2_det[0] + plane_to_det[1]*e2_det[1] + plane_to_det[2]*e2_det[2]

            rpx = int(round(center[0] - y_det/100e-6))
            cpx = int(round(center[1] + x_det/100e-6))
            if not(0<=rpx<image_size and 0<=cpx<image_size):
                continue

            val = reflection_intensity * \
                  exp(-Qz*Qz * debye_x*debye_x)* \
                  exp(-(Qx*Qx + Qy*Qy)* debye_y*debye_y)
            image[rpx, cpx]+= val

            if i_sol%2==0:
                if image[rpx,cpx]>max_I_sign0:
                    max_I_sign0= image[rpx,cpx]
                    max_x_sign0= cpx
                    max_y_sign0= rpx
            else:
                if image[rpx,cpx]>max_I_sign1:
                    max_I_sign1= image[rpx,cpx]
                    max_x_sign1= cpx
                    max_y_sign1= rpx

            if save_flag==1 and q_count[i_peaks_index]< q_data.shape[1]:
                idx = q_count[i_peaks_index]
                q_data[i_peaks_index, idx,0] = Qx
                q_data[i_peaks_index, idx,1] = Qy
                q_data[i_peaks_index, idx,2] = Qz
                q_data[i_peaks_index, idx,3] = val
                q_count[i_peaks_index]+=1

    return (max_I_sign0, max_x_sign0, max_y_sign0,
            max_I_sign1, max_x_sign1, max_y_sign1)

##############################################################################
# 6) process_peaks_parallel
##############################################################################
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
    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)

    sigma_rad   = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)  # not used in example
    # (eta_pv is also not used in rotate_G but could be used if wanted)

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
        [ 0.0, 0.0, 1.0]
    ])
    nd_temp   = R_x_det @ n_detector
    n_det_rot = R_z_det @ nd_temp
    nd_len    = sqrt(n_det_rot[0]*n_det_rot[0] + n_det_rot[1]*n_det_rot[1] + n_det_rot[2]*n_det_rot[2])
    n_det_rot/= nd_len

    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0], dtype=np.float64)

    dot_e1 = unit_x[0]*n_det_rot[0] + unit_x[1]*n_det_rot[1] + unit_x[2]*n_det_rot[2]
    e1_det = unit_x - dot_e1*n_det_rot
    e1_len = sqrt(e1_det[0]**2 + e1_det[1]**2 + e1_det[2]**2)
    if e1_len<1e-14:
        e1_det= np.array([1.0,0.0,0.0])
    else:
        e1_det/= e1_len

    tmpx= n_det_rot[1]* e1_det[2] - n_det_rot[2]* e1_det[1]
    tmpy= n_det_rot[2]* e1_det[0] - n_det_rot[0]* e1_det[2]
    tmpz= n_det_rot[0]* e1_det[1] - n_det_rot[1]* e1_det[0]
    e2_det= np.array([-tmpx, -tmpy, -tmpz], dtype=np.float64)
    e2_len= sqrt(e2_det[0]**2 + e2_det[1]**2 + e2_det[2]**2)
    if e2_len<1e-14:
        e2_det= np.array([0.0,1.0,0.0])
    else:
        e2_det/= e2_len

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

    n1= np.array([0.0,0.0,1.0], dtype=np.float64)
    R_ZY_n= R_z_R_y @ n1
    nzy_len= sqrt(R_ZY_n[0]*R_ZY_n[0]+ R_ZY_n[1]*R_ZY_n[1]+ R_ZY_n[2]*R_ZY_n[2])
    R_ZY_n/= nzy_len

    P0= np.array([0.0,0.0,-zs], dtype=np.float64)

    num_peaks= miller.shape[0]
    max_solutions=2000000

    if save_flag==1:
        q_data= np.full((num_peaks,max_solutions,5), np.nan, dtype=np.float64)
        q_count= np.zeros(num_peaks, dtype=np.int64)
    else:
        q_data= np.zeros((1,1,5), dtype=np.float64)
        q_count= np.zeros(1, dtype=np.int64)

    max_positions= np.empty((num_peaks,6), dtype=np.float64)

    for i_pk in prange(num_peaks):
        H= miller[i_pk,0]
        K= miller[i_pk,1]
        L= miller[i_pk,2]
        reflI= intensities[i_pk]

        gz0 = 2.0*pi*(L/cv)
        gr0 = 4.0*pi/av * sqrt((H*H + H*K + K*K)/3.0)
        G_ideal = np.array([0.0, gr0, gz0], dtype=np.float64)

        # Build mosaic distribution => shape (N*M, 3)
        Mosaic_Rotation = rotate_G(G_ideal, sigma_rad, N=20, M=20, n_phi=20)

        (mx0,my0,mv0, mx1,my1,mv1) = calculate_phi(
            H, K, L, av, cv,
            wavelength_array,
            image, image_size,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs, zb, n2,
            beam_x_array, beam_y_array,
            theta_array, phi_array,
            reflI, Mosaic_Rotation,
            debye_x, debye_y,
            center,
            theta_initial_deg,
            R_x_det, R_z_det, n_det_rot, Detector_Pos,
            e1_det, e2_det,
            R_z_R_y,
            R_ZY_n,
            P0, 
            unit_x,
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
