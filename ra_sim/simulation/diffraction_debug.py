import os
import csv
import datetime
import numpy as np
from math import sin, cos, sqrt, pi, exp, acos

# -----------------------------------------------------------
# Global debug log
# -----------------------------------------------------------
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
            # Initialize a row with defaults
            row = dict.fromkeys(fieldnames, "")
            row["Timestamp"] = now_str

            # Example record structures:
            # ("intersection-detector", True, H, K, L, Qx, Qy, Qz, val)
            # ("intersection-detector", False, H, K, L, Qx, Qy, Qz, val)
            # ("intersection-sample", H, K, L, reason)
            # ("Ray Missed it", H, K, L, reason)
            # etc.

            event_type = record[0]
            row["EventType"] = event_type

            if event_type == "intersection-detector":
                # Expected record structure:
                # ("intersection-detector", True/False, H, K, L, Qx, Qy, Qz, Val)
                row["IntersectionDetector"] = str(record[1])
                row["H"] = str(record[2])
                row["K"] = str(record[3])
                row["L"] = str(record[4])
                row["Qx"] = str(record[5])
                row["Qy"] = str(record[6])
                row["Qz"] = str(record[7])
                row["Val"] = str(record[8])

            elif event_type == "intersection-sample":
                # Example record structure:
                # ("intersection-sample", H, K, L, info...)
                # Fill out as needed
                row["IntersectionDetector"] = "False"
                row["H"] = str(record[1])
                row["K"] = str(record[2])
                row["L"] = str(record[3])
            
            elif event_type == "Ray Missed it":
                # Possibly record reason or partial info
                row["IntersectionDetector"] = "False"
                row["H"] = str(record[1])
                row["K"] = str(record[2])
                row["L"] = str(record[3])
                row["Val"] = str(record[4])  # e.g. reason

            writer.writerow(row)

    print(f"Debug log saved to: {filename}")
    DEBUG_LOG.clear()


# -----------------------------------------------------------
# 1) compute_intensity (was compute_intensity_jit)
# -----------------------------------------------------------
def compute_intensity(Q, G_vec, sigma):
    """
    A debugging-friendly version of compute_intensity (no numba).
    (Assumes sigma is already in radians, or else adapt accordingly.)
    """
    # Norm of G
    R = sqrt(G_vec[0]*G_vec[0] + G_vec[1]*G_vec[1] + G_vec[2]*G_vec[2])
    
    eps = 1e-8
    # Check if G is "vertical" in X-Y plane
    if abs(G_vec[0])<eps and abs(G_vec[1])<eps:
        # CAP METHOD
        Gnorm = 1.0 / R if R>eps else 1.0
        Qlen = sqrt(Q[0]*Q[0] + Q[1]*Q[1] + Q[2]*Q[2])
        Qnorm = 1.0 / Qlen if Qlen>eps else 1.0
        dot_ = (G_vec[0]*Gnorm)*(Q[0]*Qnorm) + \
               (G_vec[1]*Gnorm)*(Q[1]*Qnorm) + \
               (G_vec[2]*Gnorm)*(Q[2]*Qnorm)
        if dot_>1.0: 
            dot_=1.0
        elif dot_<-1.0:
            dot_=-1.0
        alpha = acos(dot_)
        intensity = np.exp( - (alpha*alpha)/(2.0*sigma*sigma) )
    else:
        # HORIZONTAL BAND
        R_inv = 1.0 / R if R>eps else 1.0
        Qz = Q[2]
        Gz = G_vec[2]
        ratioQ = Qz*R_inv
        ratioQ = max(-1.0, min(1.0, ratioQ))
        v_prime = acos(ratioQ)

        ratioG = Gz*R_inv
        ratioG = max(-1.0, min(1.0, ratioG))
        v_center = acos(ratioG)

        dv = v_prime - v_center
        intensity = np.exp( - (dv*dv)/(2.0*sigma*sigma) )

    return intensity


# -----------------------------------------------------------
# 2) intersect_line_plane (unchanged except removing njit)
# -----------------------------------------------------------
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


def intersect_line_plane_batch(start_pt, directions, plane_pt, plane_n):
    Ndir = directions.shape[0]
    intersects = np.full((Ndir,3), np.nan, dtype=np.float64)
    valid = np.zeros(Ndir, dtype=bool)
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


# -----------------------------------------------------------
# 3) solve_q (no njit)
# -----------------------------------------------------------
def solve_q(k_in_crystal, k_scat, G_vec, sigma, N_steps=1000):
    """
    Directly sample the intersection circle in Q-space for points with Qz > 0.
    Returns an array of shape (n,4) with each row [Qx, Qy, Qz, intensity].
    """
    bragg_rad_sq = G_vec[0]*G_vec[0] + G_vec[1]*G_vec[1] + G_vec[2]*G_vec[2]
    if bragg_rad_sq < 1e-14:
        return np.zeros((0,4), dtype=np.float64)
    bragg_rad = sqrt(bragg_rad_sq)
    
    Ax = -k_in_crystal[0]
    Ay = -k_in_crystal[1]
    Az = -k_in_crystal[2]
    rA = k_scat
    A_sq = Ax*Ax + Ay*Ay + Az*Az
    if A_sq < 1e-14:
        return np.zeros((0,4), dtype=np.float64)
    A_len = sqrt(A_sq)

    c = (bragg_rad_sq + A_sq - rA*rA) / (2.0*A_len)

    circle_r_sq = bragg_rad_sq - c*c
    if circle_r_sq < 0.0:
        return np.zeros((0,4), dtype=np.float64)
    circle_r = sqrt(circle_r_sq)

    Ax_hat = Ax / A_len
    Ay_hat = Ay / A_len
    Az_hat = Az / A_len

    Ox = c * Ax_hat
    Oy = c * Ay_hat
    Oz = c * Az_hat

    # Build e1, e2
    ax, ay, az = 1.0, 0.0, 0.0
    dot_aA = ax*Ax_hat + ay*Ay_hat + az*Az_hat
    if abs(dot_aA) > 0.9999:
        ax, ay, az = 0.0, 1.0, 0.0
        dot_aA = ax*Ax_hat + ay*Ay_hat + az*Az_hat
    aox = ax - dot_aA*Ax_hat
    aoy = ay - dot_aA*Ay_hat
    aoz = az - dot_aA*Az_hat
    ao_len = sqrt(aox*aox + aoy*aoy + aoz*aoz)
    if ao_len < 1e-14:
        return np.zeros((0,4), dtype=np.float64)
    e1x = aox / ao_len
    e1y = aoy / ao_len
    e1z = aoz / ao_len

    e2x = Az_hat*e1y - Ay_hat*e1z
    e2y = Ax_hat*e1z - Az_hat*e1x
    e2z = Ay_hat*e1x - Ax_hat*e1y
    e2_len = sqrt(e2x*e2x + e2y*e2y + e2z*e2z)
    if e2_len < 1e-14:
        return np.zeros((0,4), dtype=np.float64)
    e2x /= e2_len
    e2y /= e2_len
    e2z /= e2_len

    # Count how many Qz > 0
    count = 0
    for i in range(N_steps):
        theta = 2.0*pi*(i/float(N_steps))
        cth = cos(theta)
        sth = sin(theta)
        Qz = Oz + circle_r*(cth*e1z + sth*e2z)
        if Qz > 0.0:
            count += 1

    out = np.zeros((count, 4), dtype=np.float64)
    
    idx = 0
    for i in range(N_steps):
        theta = 2.0*pi*(i/float(N_steps))
        cth = cos(theta)
        sth = sin(theta)
        Qx = Ox + circle_r*(cth*e1x + sth*e2x)
        Qy = Oy + circle_r*(cth*e1y + sth*e2y)
        Qz = Oz + circle_r*(cth*e1z + sth*e2z)
        if Qz > 0.0:
            Ival = compute_intensity(np.array([Qx, Qy, Qz]), G_vec, sigma)
            out[idx, 0] = Qx
            out[idx, 1] = Qy
            out[idx, 2] = Qz
            out[idx, 3] = Ival
            idx += 1
    return out


# -----------------------------------------------------------
# 4) calculate_phi (no njit)
# -----------------------------------------------------------
def calculate_phi(
    H, K, L, av, cv,
    wavelength_array,
    image, image_size,
    gamma_rad, Gamma_rad, chi_rad, psi_rad,
    zs, zb, n2,
    beam_x_array, beam_y_array,
    theta_array, phi_array,
    reflection_intensity,
    sigma_rad, gamma_rad_m, eta_pv,
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
    G_vec = np.array([0.0, gr0, gz0], dtype=np.float64)
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

    for i_samp in range(n_samp):
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
            # Optionally record debug for missed sample intersection
            DEBUG_LOG.append(("intersection-sample", H, K, L, "No valid intersection with sample"))
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

        All_Q = solve_q(k_in_crystal, k_scat, G_vec, sigma_rad)
        for i_sol in range(All_Q.shape[0]):
            Qx = All_Q[i_sol, 0]
            Qy = All_Q[i_sol, 1]
            Qz = All_Q[i_sol, 2]
            I_Q = All_Q[i_sol, 3]  # intensity from Q

            k_tx_prime = Qx + k_x_scat
            k_ty_prime = Qy + k_y_scat
            k_tz_prime = Qz + k_z_scat

            kr = sqrt(k_tx_prime*k_tx_prime + k_ty_prime*k_ty_prime)
            if kr < 1e-12:
                twotheta_t = 0.0
            else:
                twotheta_t = np.arctan(k_tz_prime/kr)

            phi_f = np.arctan2(k_tx_prime, k_ty_prime)
            k_tx_f = k_scat*cos(twotheta_t)*sin(phi_f)
            k_ty_f = k_scat*cos(twotheta_t)*cos(phi_f)
            real_k_tz_f = k_scat*sin(twotheta_t)
            kf = np.array([k_tx_f, k_ty_f, real_k_tz_f])
            kf_prime = R_sample @ kf

            dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
            if not valid_det:
                # Optionally record a missed detector intersection
                DEBUG_LOG.append(("intersection-detector", False, H, K, L, Qx, Qy, Qz, 0.0))
                continue

            plane_to_det = np.array([dx - Detector_Pos[0],
                                     dy - Detector_Pos[1],
                                     dz - Detector_Pos[2]], dtype=np.float64)
            x_det = plane_to_det[0]*e1_det[0] + plane_to_det[1]*e1_det[1] + plane_to_det[2]*e1_det[2]
            y_det = plane_to_det[0]*e2_det[0] + plane_to_det[1]*e2_det[1] + plane_to_det[2]*e2_det[2]

            rpx = int(round(center[0] - y_det/100e-6))
            cpx = int(round(center[1] + x_det/100e-6))
            if not (0 <= rpx < image_size and 0 <= cpx < image_size):
                # Ray goes off the detector boundary
                DEBUG_LOG.append(("intersection-detector", False, H, K, L, Qx, Qy, Qz, 0.0))
                continue

            # Multiply reflection_intensity by I_Q and Debye-Waller
            val = reflection_intensity * I_Q * \
                  exp(-Qz*Qz * debye_x*debye_x) * \
                  exp(-(Qx*Qx + Qy*Qy) * debye_y*debye_y)
            image[rpx, cpx] += val

            # Record a successful intersection on the detector in the debug log
            DEBUG_LOG.append(("intersection-detector", True, H, K, L, Qx, Qy, Qz, val))

            # Track maximum intensities
            if i_sol%2==0:
                if image[rpx,cpx]>max_I_sign0:
                    max_I_sign0 = image[rpx,cpx]
                    max_x_sign0 = cpx
                    max_y_sign0 = rpx
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


# -----------------------------------------------------------
# 5) process_peaks_parallel (no parallel, just a normal loop)
# -----------------------------------------------------------
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
    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)

    sigma_rad   = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)  # not used in example

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

    for i_pk in range(num_peaks):
        H= miller[i_pk,0]
        K= miller[i_pk,1]
        L= miller[i_pk,2]
        reflI= intensities[i_pk]

        (mx0,my0,mv0, mx1,my1,mv1) = calculate_phi(
            H, K, L, av, cv,
            wavelength_array,
            image, image_size,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs, zb, n2,
            beam_x_array, beam_y_array,
            theta_array, phi_array,
            reflI, sigma_rad, gamma_rad_m, eta_pv,
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
