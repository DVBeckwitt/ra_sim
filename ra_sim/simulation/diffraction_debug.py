import numpy as np
import os
import csv
import datetime
from math import sin, cos, sqrt, pi, exp

##############################################################################
# Debug Logging
##############################################################################

DEBUG_LOG = []

def dump_debug_log():
    """
    Writes the global debug log to ~/Downloads/mosaic_full_debug_log.csv,
    ensuring columns like 'IntersectionDetector', 'EventType', 'Qx', 'Qy', etc. exist.
    """
    filename = os.path.expanduser("~/Downloads/mosaic_full_debug_log.csv")
    now_str = datetime.datetime.now().isoformat()

    fieldnames = [
        "Timestamp", 
        "EventType", 
        "IntersectionDetector",
        "H", "K", "L",
        "Qx", "Qy", "Qz", 
        "Val"
    ]
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in DEBUG_LOG:
            row = dict.fromkeys(fieldnames, "")
            row["Timestamp"] = now_str

            # record is a tuple, e.g. ("intersection-detector", True, H, K, L, Qx, Qy, Qz, Val)
            # or some other pattern depending on the event type

            event_type = record[0]
            row["EventType"] = event_type

            if event_type == "intersection-detector":
                # Expected record structure:
                # ("intersection-detector", True/False, H, K, L, Qx, Qy, Qz, Val)
                # or any other relevant columns
                row["IntersectionDetector"] = str(record[1])  # "True" or "False"
                row["H"] = str(record[2])
                row["K"] = str(record[3])
                row["L"] = str(record[4])
                row["Qx"] = str(record[5])
                row["Qy"] = str(record[6])
                row["Qz"] = str(record[7])
                row["Val"] = str(record[8])

            elif event_type == "intersection-sample":
                # e.g. ("intersection-sample", H, K, L, Qx, Qy, Qz, val, etc.)
                # Fill in as needed, or skip
                row["IntersectionDetector"] = "False"
                row["H"] = str(record[1])
                row["K"] = str(record[2])
                row["L"] = str(record[3])

            elif event_type == "Ray Missed it":
                # Possibly record partial info
                row["IntersectionDetector"] = "False"
                # Could store reason in "Val", for instance
                # row["Val"] = str(record[1])  # or skip

            else:
                # Catch-all
                row["IntersectionDetector"] = "False"

            writer.writerow(row)

    print(f"Debug log saved to: {filename}")
    DEBUG_LOG.clear()

##############################################################################
# Functions (no @njit, no parallelization)
##############################################################################

def pseudo_voigt_1d(r, eta, sigma, gamma):
    """
    Returns the value of the 1D pseudo-Voigt function at radius r:
        f(r) = (1 - eta)*Gauss(r) + eta*Lorentz(r)

    where:
      Gauss(r) = (1 / (sqrt(2*pi)*sigma)) * exp(-r^2/(2*sigma^2))
      Lorentz(r) = (1/pi) * [ gamma / (r^2 + gamma^2) ]

    Note: 'sigma' and 'gamma' are interpreted as the 'width' parameters.
    """
    gauss = (1.0 / (sqrt(2.0*pi)*sigma)) * exp(-0.5*(r*r)/(sigma*sigma))
    lorentz = (1.0/pi) * (gamma / (r*r + gamma*gamma))
    return (1.0 - eta)*gauss + eta*lorentz

def sample_mosaic_angles_separable(eta, sigma_rad, gamma_rad, N=10000, grid_points=100000):
    DEBUG_LOG.append(("sample_mosaic_angles_separable-begin", eta, sigma_rad, gamma_rad, N, grid_points))

    r_max = 5.0 * max(sigma_rad, gamma_rad)
    dr = r_max / (grid_points - 1)
    r_vals = np.linspace(0.0, r_max, grid_points)
    pdf = np.empty(grid_points, dtype=np.float64)
    for i in range(grid_points):
        r = r_vals[i]
        f_r = pseudo_voigt_1d(r, eta, sigma_rad, gamma_rad)
        pdf[i] = 2.0*pi*r*f_r
    cdf = np.empty(grid_points, dtype=np.float64)
    cdf[0] = pdf[0]*dr
    for i in range(1, grid_points):
        cdf[i] = cdf[i-1] + 0.5*(pdf[i]+pdf[i-1])*dr
    total_area = cdf[-1]
    cdf /= total_area

    R_out = np.empty((N, 3, 3), dtype=np.float64)
    for i in range(N):
        u = np.random.random()
        left = 0
        right = grid_points-1
        while right - left > 1:
            mid = (left+right)//2
            if cdf[mid] >= u:
                right = mid
            else:
                left = mid
        if left == right:
            r_sample = r_vals[left]
        else:
            denom = cdf[right] - cdf[left]
            if denom < 1e-14:
                r_sample = r_vals[left]
            else:
                ratio = (u - cdf[left])/denom
                r_sample = r_vals[left] + ratio*(r_vals[right]-r_vals[left])

        phi = 2.0*pi*np.random.random()
        beta = r_sample*cos(phi)
        kappa = r_sample*sin(phi)

        cosb = cos(beta)
        sinb = sin(beta)
        R_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cosb, -sinb],
            [0.0, sinb,  cosb]
        ], dtype=np.float64)

        cosk = cos(kappa)
        sink = sin(kappa)
        R_y = np.array([
            [cosk, 0.0, sink],
            [0.0,  1.0,  0.0],
            [-sink,0.0, cosk]
        ], dtype=np.float64)

        R = R_y @ R_x
        R_out[i,:,:] = R

    DEBUG_LOG.append(("sample_mosaic_angles_separable-end", N))
    return R_out

def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    dot_val = np.dot(k_vec, n_plane)
    if abs(dot_val) < 1e-14:
        return (np.nan, np.nan, np.nan, False)
    num = np.dot((P_plane - P0), n_plane)
    t = num / dot_val
    if t < 0.0:
        return (np.nan, np.nan, np.nan, False)
    ix = P0[0] + t*k_vec[0]
    iy = P0[1] + t*k_vec[1]
    iz = P0[2] + t*k_vec[2]
    return (ix, iy, iz, True)

def solve_q(k_in, k, gz0, gr0, g, R_mats, eps=1e-14):
    DEBUG_LOG.append(("solve_q-begin", len(R_mats)))

    kx, ky, kz = k_in
    N_samples = R_mats.shape[0]
    solutions = np.full((N_samples, 2, 3), np.nan, dtype=np.float64)

    G_ideal = np.array([0.0, gr0, gz0], dtype=np.float64)
    G_ideal_sq = gr0*gr0 + gz0*gz0

    kx2 = kx*kx
    ky2 = ky*ky
    kz2 = kz*kz
    alpha0 = 0.5*(k*k - G_ideal_sq - (kx2 + ky2 + kz2))
    L = kx2 + ky2

    small_ky = (abs(ky) < eps)

    for i in range(N_samples):
        G_rot = R_mats[i] @ G_ideal
        qz = G_rot[2]
        R_sq = G_ideal_sq - qz*qz
        if R_sq < eps:
            continue

        alpha = alpha0 - qz*kz
        if small_ky:
            if abs(kx) < eps:
                continue
            qx0 = alpha/kx
            tmp = R_sq - qx0*qx0
            if tmp < 0:
                continue
            rt = sqrt(tmp)
            solutions[i,0,0] = qx0
            solutions[i,0,1] = +rt
            solutions[i,0,2] = qz
            solutions[i,1,0] = qx0
            solutions[i,1,1] = -rt
            solutions[i,1,2] = qz
        else:
            r = kx/ky
            disc = R_sq*L - alpha*alpha
            if disc < 0:
                continue
            rt = sqrt(disc)
            denom = ky*(r*r + 1.0)

            qxp = (alpha*r + rt)/denom
            qxm = (alpha*r - rt)/denom
            qyp = (alpha - kx*qxp)/ky
            qym = (alpha - kx*qxm)/ky

            solutions[i,0,0] = qxp
            solutions[i,0,1] = qyp
            solutions[i,0,2] = qz
            solutions[i,1,0] = qxm
            solutions[i,1,1] = qym
            solutions[i,1,2] = qz

    DEBUG_LOG.append(("solve_q-end", len(R_mats)))
    return solutions

def calculate_phi(
    H, K, L, av, cv,
    wavelength_array,
    image, image_size,
    gamma_rad, Gamma_rad, chi_rad, psi_rad,
    zs, zb, n2,
    beam_x_array, beam_y_array,
    theta_array, phi_array,
    reflection_intensity,
    Mosaic_Rotation,
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
    DEBUG_LOG.append(("calculate_phi-begin", H, K, L))

    gz0 = 2.0*np.pi * (L/cv)
    gr0 = 4.0*np.pi/av * sqrt((H*H + H*K + K*K)/3.0)
    G = sqrt(gr0**2 + gz0**2)

    max_I_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan
    max_I_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan

    rad_theta_i = theta_initial_deg*(pi/180.0)
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos(rad_theta_i), -sin(rad_theta_i)],
        [0.0, sin(rad_theta_i),  cos(rad_theta_i)]
    ], dtype=np.float64)
    R_sample = R_x @ R_z_R_y

    # Sample-plane normal in lab
    n_surf = R_x @ R_ZY_n
    n_surf /= np.linalg.norm(n_surf)

    # Shift in sample plane
    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0

    n_samp = beam_x_array.size

    # Build e1_temp, e2_temp
    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_norm = np.linalg.norm(e1_temp)
    if e1_norm < 1e-12:
        alt_refs = [
            np.array([1.0,0.0,0.0]),
            np.array([0.0,1.0,0.0]),
            np.array([0.0,0.0,1.0])
        ]
        success = False
        for alt_ref in alt_refs:
            cross_tmp = np.cross(n_surf, alt_ref)
            cross_len = np.linalg.norm(cross_tmp)
            if cross_len > 1e-12:
                e1_temp = cross_tmp / cross_len
                success = True
                break
        if not success:
            return (max_I_sign0, max_x_sign0, max_y_sign0,
                    max_I_sign1, max_x_sign1, max_y_sign1)
    else:
        e1_temp /= e1_norm

    e2_temp = np.cross(n_surf, e1_temp)

    for i_samp in range(n_samp):
        lam_samp = wavelength_array[i_samp]
        k_mag = 2.0*np.pi / lam_samp

        bx = beam_x_array[i_samp]
        by = beam_y_array[i_samp]
        beam_start = np.array([bx, -20e-3, -zb + by], dtype=np.float64)

        dtheta = theta_array[i_samp]
        dphi = phi_array[i_samp]
        k_in = np.array([
            cos(dtheta)*sin(dphi),
            cos(dtheta)*cos(dphi),
            sin(dtheta)
        ], dtype=np.float64)

        # Intersect with sample plane
        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if not valid_int:
            # Could log a "Ray Missed it" event here if desired
            continue

        I_plane = np.array([ix, iy, iz], dtype=np.float64)
        kn_dot = np.dot(k_in, n_surf)
        th_i_prime = (pi/2.0) - np.arccos(kn_dot)

        projected_incident = k_in - kn_dot*n_surf
        proj_len = np.linalg.norm(projected_incident)
        if proj_len > 1e-12:
            projected_incident /= proj_len
        else:
            projected_incident[:] = 0.0

        p1 = np.dot(projected_incident, e1_temp)
        p2 = np.dot(projected_incident, e2_temp)
        phi_i_prime = (pi/2.0) - np.arctan2(p2, p1)
        th_t = np.arccos(cos(th_i_prime))*np.sign(th_i_prime)

        k_scat = k_mag
        k_x_scat = k_scat*cos(th_t)*sin(phi_i_prime)
        k_y_scat = k_scat*cos(th_t)*cos(phi_i_prime)
        k_z_scat = k_scat*sin(th_t)

        k_in_crystal = np.array([k_x_scat, k_y_scat, k_z_scat], dtype=np.float64)
        All_Q = solve_q(k_in_crystal, k_scat, gz0, gr0, G, Mosaic_Rotation)
        All_Q_flat = All_Q.reshape((-1, 3))

        for i_sol in range(All_Q_flat.shape[0]):
            Qx = All_Q_flat[i_sol, 0]
            Qy = All_Q_flat[i_sol, 1]
            Qz = All_Q_flat[i_sol, 2]
            if np.isnan(Qx) or np.isnan(Qy) or np.isnan(Qz):
                continue

            k_tx_prime = Qx + k_x_scat
            k_ty_prime = Qy + k_y_scat
            k_tz_prime = Qz + k_z_scat

            kr = sqrt(k_tx_prime**2 + k_ty_prime**2)
            if abs(kr) < 1e-12:
                twotheta_t = 0.0
            else:
                twotheta_t = np.arctan(k_tz_prime / kr)

            phi_f = np.arctan2(k_tx_prime, k_ty_prime)
            k_tx_f = k_scat*cos(twotheta_t)*sin(phi_f)
            k_ty_f = k_scat*cos(twotheta_t)*cos(phi_f)
            real_k_tz_f = k_scat*sin(twotheta_t)

            kf = np.array([k_tx_f, k_ty_f, real_k_tz_f], dtype=np.float64)
            kf_prime = R_sample @ kf

            dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
            if not valid_det:
                continue

            plane_to_det = np.array([
                dx - Detector_Pos[0],
                dy - Detector_Pos[1],
                dz - Detector_Pos[2]
            ], dtype=np.float64)

            x_det = np.dot(plane_to_det, e1_det)
            y_det = np.dot(plane_to_det, e2_det)

            rpx = int(round(center[0] - y_det/1e-4))  
            cpx = int(round(center[1] + x_det/1e-4))
            if not (0 <= rpx < image_size and 0 <= cpx < image_size):
                continue

            val = reflection_intensity * exp(-Qz*Qz*debye_x*debye_x) * exp(-(Qx*Qx + Qy*Qy)*debye_y*debye_y)
            image[rpx, cpx] += val

            # Log the intersection with the detector => "intersection-detector"
            DEBUG_LOG.append((
                "intersection-detector",
                True,           # IntersectionDetector
                H,              # H
                K,              # K
                L,              # L
                Qx,             # Qx
                Qy,             # Qy
                Qz,             # Qz
                val             # Val (intensity)
            ))

            # Track maxima
            if (i_sol % 2) == 0:
                if image[rpx, cpx] > max_I_sign0:
                    max_I_sign0 = image[rpx, cpx]
                    max_x_sign0 = cpx
                    max_y_sign0 = rpx
            else:
                if image[rpx, cpx] > max_I_sign1:
                    max_I_sign1 = image[rpx, cpx]
                    max_x_sign1 = cpx
                    max_y_sign1 = rpx

            if save_flag == 1 and q_count[i_peaks_index] < q_data.shape[1]:
                idx = q_count[i_peaks_index]
                q_data[i_peaks_index, idx, 0] = Qx
                q_data[i_peaks_index, idx, 1] = Qy
                q_data[i_peaks_index, idx, 2] = Qz
                q_data[i_peaks_index, idx, 3] = val
                q_count[i_peaks_index] += 1

    DEBUG_LOG.append(("calculate_phi-end", H, K, L))
    return (max_I_sign0, max_x_sign0, max_y_sign0,
            max_I_sign1, max_x_sign1, max_y_sign1)

def process_peaks_parallel_debug(
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
    DEBUG_LOG.append(("process_peaks_parallel-begin", miller.shape[0]))

    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)

    sigma_rad = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)

    Mosaic_Rotation = sample_mosaic_angles_separable(eta_pv, sigma_rad, gamma_rad_m, N=1000, grid_points=100000)

    cg = cos(gamma_rad); sg = sin(gamma_rad)
    cG = cos(Gamma_rad); sG = sin(Gamma_rad)

    R_x_det = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cg,  sg],
        [0.0,-sg,  cg]
    ], dtype=np.float64)
    R_z_det = np.array([
        [ cG, sG, 0.0],
        [-sG, cG, 0.0],
        [ 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    nd_temp = R_x_det @ n_detector
    n_det_rot = R_z_det @ nd_temp
    nd_len = sqrt((n_det_rot**2).sum())
    n_det_rot /= nd_len

    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0], dtype=np.float64)

    dot_e1 = np.dot(unit_x, n_det_rot)
    e1_det = unit_x - dot_e1*n_det_rot
    e1_len = sqrt((e1_det**2).sum())
    if e1_len < 1e-14:
        e1_det = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1_det /= e1_len

    tmpx = n_det_rot[1]*e1_det[2] - n_det_rot[2]*e1_det[1]
    tmpy = n_det_rot[2]*e1_det[0] - n_det_rot[0]*e1_det[2]
    tmpz = n_det_rot[0]*e1_det[1] - n_det_rot[1]*e1_det[0]
    e2_det = np.array([-tmpx, -tmpy, -tmpz], dtype=np.float64)
    e2_len = sqrt((e2_det**2).sum())
    if e2_len < 1e-14:
        e2_det = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        e2_det /= e2_len

    c_chi = cos(chi_rad); s_chi = sin(chi_rad)
    R_y = np.array([
        [ c_chi, 0.0,   s_chi],
        [ 0.0,   1.0,   0.0],
        [-s_chi, 0.0, c_chi]
    ], dtype=np.float64)
    c_psi = cos(psi_rad); s_psi = sin(psi_rad)
    R_z = np.array([
        [ c_psi, s_psi, 0.0],
        [-s_psi, c_psi, 0.0],
        [ 0.0,   0.0,   1.0]
    ], dtype=np.float64)
    R_z_R_y = R_z @ R_y

    n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R_ZY_n = R_z_R_y @ n1
    R_ZY_n /= np.linalg.norm(R_ZY_n)

    P0 = np.array([0.0, 0.0, -zs], dtype=np.float64)

    num_peaks = miller.shape[0]
    max_solutions = 2000000

    if save_flag == 1:
        q_data = np.full((num_peaks, max_solutions, 5), np.nan, dtype=np.float64)
        q_count = np.zeros(num_peaks, dtype=np.int64)
    else:
        q_data = np.zeros((1,1,5), dtype=np.float64)
        q_count = np.zeros(1, dtype=np.int64)

    max_positions = np.empty((num_peaks, 6), dtype=np.float64)

    for i_pk in range(num_peaks):
        H = miller[i_pk, 0]
        K = miller[i_pk, 1]
        L = miller[i_pk, 2]
        reflI = intensities[i_pk]

        (mx0, my0, mv0,
         mx1, my1, mv1) = calculate_phi(
            H, K, L,
            av, cv,
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
            P0, unit_x,
            save_flag, q_data, q_count, i_pk
        )

        max_positions[i_pk, 0] = mx0
        max_positions[i_pk, 1] = my0
        max_positions[i_pk, 2] = mv0
        max_positions[i_pk, 3] = mx1
        max_positions[i_pk, 4] = my1
        max_positions[i_pk, 5] = mv1

    DEBUG_LOG.append(("process_peaks_parallel-end", num_peaks))

    if save_flag == 1:
        return image, max_positions, q_data, q_count
    else:
        return image, max_positions, None, None
