import os
import csv
import datetime
import numpy as np
from math import sin, cos, sqrt, pi, exp, atan2

##############################################################################
# GLOBAL DEBUG LOG
##############################################################################
DEBUG_LOG = []
DEBUG_ENABLED = True

def dump_debug_log():
    """
    Write all debug-log records to ~/Downloads/mosaic_debug_log.csv in CSV format.
    You can adjust columns or path as needed.
    """
    global DEBUG_LOG

    filename = os.path.expanduser("~/Downloads/mosaic_debug_log.csv")
    now_str = datetime.datetime.now().isoformat()

    # Example header adjusted for mosaic-based methodology
    header = [
        "Timestamp",
        "EventType",
        "H", "K", "L",
        "MosaicAngle_rad",
        "BeamStartX", "BeamStartY",
        "IntersectionSample",      # True/False
        "IntersectionDetector",    # True/False
        "Qx", "Qy", "Qz",
        "Val",                     # The final intensity or weighting
        "DetectorX", "DetectorY",  # x_det, y_det
        "Extra"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for record in DEBUG_LOG:
            row = dict.fromkeys(header, "")
            row["Timestamp"] = now_str
            row["EventType"] = record[0]

            if record[0] == "process_peaks_parallel":
                # e.g. ("process_peaks_parallel", num_miller)
                row["Extra"] = f"num_miller={record[1]}"

            elif record[0] == "mosaic-sampling":
                # e.g. ("mosaic-sampling", eta, sigma_rad, gamma_rad, N, grid_points)
                row["Extra"] = (f"eta={record[1]}, sigma_rad={record[2]}, "
                                f"gamma_rad={record[3]}, N={record[4]}, "
                                f"grid_points={record[5]}")

            elif record[0] == "calc_phi-begin":
                # e.g. ("calc_phi-begin", H, K, L)
                row["H"] = record[1]
                row["K"] = record[2]
                row["L"] = record[3]

            elif record[0] == "intersection-sample":
                # e.g. ("intersection-sample", H, K, L, bx, by, valid_bool)
                row["H"] = record[1]
                row["K"] = record[2]
                row["L"] = record[3]
                row["BeamStartX"] = record[4]
                row["BeamStartY"] = record[5]
                row["IntersectionSample"] = record[6]

            elif record[0] == "intersection-detector":
                # e.g. ("intersection-detector", H, K, L, mosaic_angle, valid_bool, x_det, y_det, Qx, Qy, Qz, val)
                row["H"]               = record[1]
                row["K"]               = record[2]
                row["L"]               = record[3]
                row["MosaicAngle_rad"] = record[4]
                row["IntersectionDetector"] = record[5]
                row["DetectorX"]       = record[6]
                row["DetectorY"]       = record[7]
                row["Qx"]              = record[8]
                row["Qy"]              = record[9]
                row["Qz"]              = record[10]
                row["Val"]             = record[11]

            elif record[0] == "solve_q-solution":
                # e.g. ("solve_q-solution", idx, gz, gr, Qx_pos, Qy_pos, Qz, Qx_neg, Qy_neg)
                row["Extra"] = (f"mosaic_idx={record[1]}, gz={record[2]}, gr={record[3]}, "
                                f"Qx_pos={record[4]}, Qy_pos={record[5]}, Qz={record[6]}, "
                                f"Qx_neg={record[7]}, Qy_neg={record[8]}")

            writer.writerow(row)

    print(f"Debug log saved to: {filename}")
    DEBUG_LOG.clear()


##############################################################################
# 1) Mosaic Distribution
##############################################################################
def sample_mosaic_angles_combined(eta, sigma_rad, gamma_rad, N=100, grid_points=1000):
    """
    Sample mosaic angles in radians from pseudo-Voigt distribution.
    """
    if DEBUG_ENABLED:
        DEBUG_LOG.append(("mosaic-sampling", eta, sigma_rad, gamma_rad, N, grid_points))

    # same code as before
    import math

    G2D_const = 1.0 / (2.0*math.pi*sigma_rad*sigma_rad)
    G2D_exp   = 1.0 / (2.0*sigma_rad*sigma_rad)
    L2D_const = 1.0 / (2.0*math.pi)

    max_alpha = 10.0 * (sigma_rad if sigma_rad>gamma_rad else gamma_rad)
    grid = np.linspace(0.0, max_alpha, grid_points)
    pdf = np.empty(grid_points, dtype=np.float64)
    total = 0.0

    for i in range(grid_points):
        alpha = grid[i]
        r = math.sqrt(2.0)*abs(alpha)
        g_val = G2D_const*math.exp(-r*r*G2D_exp)
        denom = (r*r+gamma_rad*gamma_rad)**1.5
        l_val = L2D_const*(gamma_rad/denom)
        val = (1.0-eta)*g_val + eta*l_val
        pdf[i] = val
        total += val

    pdf /= total

    cdf = np.empty(grid_points, dtype=np.float64)
    cdf[0] = pdf[0]
    for i in range(1, grid_points):
        cdf[i] = cdf[i-1] + pdf[i]

    samples = np.empty(N, dtype=np.float64)
    for i in range(N):
        rnd = np.random.random() * 0.999
        # binary search
        lo=0
        hi=grid_points-1
        while lo<=hi:
            mid=(lo+hi)//2
            if cdf[mid]<rnd:
                lo=mid+1
            else:
                hi=mid-1
        idx=lo
        sample_val=grid[idx]
        if np.random.random()>0.5:
            samples[i]=sample_val
        else:
            samples[i]=-sample_val

    return samples


##############################################################################
# 2) Intersection
##############################################################################
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    denom = k_vec[0]*n_plane[0] + k_vec[1]*n_plane[1] + k_vec[2]*n_plane[2]
    if abs(denom)<1e-14:
        return np.nan, np.nan, np.nan, False
    num = ((P_plane[0]-P0[0])*n_plane[0]
         + (P_plane[1]-P0[1])*n_plane[1]
         + (P_plane[2]-P0[2])*n_plane[2])
    t= num/denom
    if t<0.0:
        return np.nan, np.nan, np.nan, False
    ix=P0[0]+t*k_vec[0]
    iy=P0[1]+t*k_vec[1]
    iz=P0[2]+t*k_vec[2]
    return ix, iy, iz, True


##############################################################################
# 3) Solve_q with mosaic
##############################################################################
def solve_q(k_in, k, gz0, gr0, G, mos_sampling, eps=1e-14):
    """
    Build (gz, gr) by rotating (gz0, gr0) by mosaic angles, then solve for Q+/-.
    We'll log solutions here for debugging.
    """
    k_x, k_y, k_z = k_in
    # For each mosaic angle => rotated gz, gr
    gz_arr = gr0*np.cos(mos_sampling) - gz0*np.sin(mos_sampling)
    gr_arr = gr0*np.sin(mos_sampling) + gz0*np.cos(mos_sampling)

    # shape => (N,2,3)
    solutions = np.zeros((gz_arr.size,2,3), dtype=np.float64)

    for idx in range(gz_arr.size):
        gz= gz_arr[idx]
        gr= gr_arr[idx]
        # Positive root
        try:
            disc_part = -gr**4 - 2*gr**2*gz**2 - 4*gr**2*gz*k_z + 2*gr**2*k**2 + \
                        2*gr**2*k_x**2 + 2*gr**2*k_y**2 - 2*gr**2*k_z**2 - gz**4 - \
                        4*gz**3*k_z + 2*gz**2*k**2 - 2*gz**2*k_x**2 - \
                        2*gz**2*k_y**2 - 6*gz**2*k_z**2 + 4*gz*k**2*k_z - \
                        4*gz*k_x**2*k_z - 4*gz*k_y**2*k_z - 4*gz*k_z**3 - k**4 + \
                        2*k**2*k_x**2 + 2*k**2*k_y**2 + 2*k**2*k_z**2 - k_x**4 - \
                        2*k_x**2*k_y**2 - 2*k_x**2*k_z**2 - k_y**4 - 2*k_y**2*k_z**2 - k_z**4

            # if disc_part < 0 => no real solution => skip
            # but let's just proceed if disc_part >= eps
            if disc_part < -eps:
                # We'll log that we have negative discriminant
                if DEBUG_ENABLED:
                    DEBUG_LOG.append((
                        "solve_q-solution",
                        idx, float(gz), float(gr),
                        np.nan, np.nan, np.nan,  # Qx_pos, Qy_pos, Qz
                        np.nan, np.nan           # Qx_neg, Qy_neg
                    ))
                continue

            sqrt_disc = sqrt(max(0.0, disc_part))
            # qx+
            qx_pos = (
                -k_x*(gr**2+gz**2+2*gz*k_z - k**2 + k_x**2 + k_y**2 + k_z**2)
                - k_y* sqrt_disc
            )/(2*(k_x**2 + k_y**2 + 1e-30))

            # qy+
            qy_pos = (
                -gr**2 - gz**2 -2*gz*k_z + k**2 - k_x**2
                -2*k_x*qx_pos - k_y**2 - k_z**2
            ) / (2*(k_y+1e-30))

            # qx-
            qx_neg = (
                -k_x*(gr**2+gz**2+2*gz*k_z - k**2 + k_x**2 + k_y**2 + k_z**2)
                + k_y* sqrt_disc
            )/(2*(k_x**2 + k_y**2 + 1e-30))

            # qy-
            qy_neg = (
                -gr**2 - gz**2 - 2*gz*k_z + k**2 - k_x**2
                -2*k_x*qx_neg - k_y**2 - k_z**2
            )/(2*(k_y+1e-30))

            solutions[idx,0,0] = qx_pos
            solutions[idx,0,1] = qy_pos
            solutions[idx,0,2] = gz
            solutions[idx,1,0] = qx_neg
            solutions[idx,1,1] = qy_neg
            solutions[idx,1,2] = gz

            if DEBUG_ENABLED:
                DEBUG_LOG.append((
                    "solve_q-solution",
                    idx, float(gz), float(gr),
                    float(qx_pos), float(qy_pos), float(gz),
                    float(qx_neg), float(qy_neg)
                ))

        except:
            # in case of math error => log
            if DEBUG_ENABLED:
                DEBUG_LOG.append((
                    "solve_q-solution",
                    idx, float(gz), float(gr),
                    np.nan, np.nan, np.nan,
                    np.nan, np.nan
                ))
    return solutions


##############################################################################
# 4) calculate_phi
##############################################################################
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
    R_z_R_y, R_ZY_n, P0, unit_x,
    save_flag, q_data, q_count, i_peaks_index
):
    # Log we started
    if DEBUG_ENABLED:
        DEBUG_LOG.append(("calc_phi-begin", float(H), float(K), float(L)))

    # Build G in crystal coords => (gr0, 0, gz0)
    gz0 = 2.0*pi * (L/cv)
    gr0 = 4.0*pi/av * sqrt((H*H + H*K + K*K)/3.0)
    G   = sqrt(gr0**2 + gz0**2)

    # Track maxima
    max_I_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan
    max_I_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan

    # Build the sample rotation
    rad_theta_i = theta_initial_deg*(pi/180.0)
    R_x = np.array([
        [1.0, 0.0,             0.0],
        [0.0, cos(rad_theta_i), -sin(rad_theta_i)],
        [0.0, sin(rad_theta_i),  cos(rad_theta_i)]
    ])
    R_sample = R_x @ R_z_R_y

    # Sample-plane normal
    n_surf = R_x @ R_ZY_n
    norm_n_surf = sqrt(n_surf[0]**2 + n_surf[1]**2 + n_surf[2]**2)
    if norm_n_surf<1e-14:  # degenerate => return
        return (max_x_sign0, max_y_sign0, max_I_sign0,
                max_x_sign1, max_y_sign1, max_I_sign1)
    n_surf /= norm_n_surf
    P0_rot = R_sample @ P0

    # define e1_temp,e2_temp on sample plane
    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_len = sqrt(e1_temp[0]**2 + e1_temp[1]**2 + e1_temp[2]**2)
    if e1_len<1e-12:
        # fallback
        e1_temp = np.array([1.0, 0.0, 0.0])
    else:
        e1_temp /= e1_len
    e2_temp = np.cross(n_surf, e1_temp)

    # For each beam sample
    n_samp = beam_x_array.size
    for i_samp in range(n_samp):
        lam_samp = wavelength_array[i_samp]
        k_mag = 2.0*pi/lam_samp
        bx = beam_x_array[i_samp]
        by = beam_y_array[i_samp]
        beam_start = np.array([bx, -20e-3, -zb+by], dtype=np.float64)

        dtheta = theta_array[i_samp]
        dphi   = phi_array[i_samp]

        # Build incoming wave
        k_in = np.array([
            cos(dtheta)*sin(dphi),
            cos(dtheta)*cos(dphi),
            sin(dtheta)
        ], dtype=np.float64)

        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        # Log sample intersection
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "intersection-sample",
                float(H), float(K), float(L),
                float(bx), float(by),
                bool(valid_int)
            ))
        if not valid_int:
            continue
        I_plane = np.array([ix, iy, iz], dtype=np.float64)

        # Dot with sample normal => incidence angle
        kn_dot = (k_in[0]*n_surf[0] + k_in[1]*n_surf[1] + k_in[2]*n_surf[2])
        th_i_prime = (pi/2.0)-np.arccos(kn_dot)
        # project
        projected_incident = k_in - kn_dot*n_surf
        proj_len = sqrt(projected_incident[0]**2 + projected_incident[1]**2 + projected_incident[2]**2)
        if proj_len>1e-12:
            projected_incident /= proj_len
        p1 = (projected_incident[0]*e1_temp[0]
              + projected_incident[1]*e1_temp[1]
              + projected_incident[2]*e1_temp[2])
        p2 = (projected_incident[0]*e2_temp[0]
              + projected_incident[1]*e2_temp[1]
              + projected_incident[2]*e2_temp[2])
        phi_i_prime = (pi/2.0)-atan2(p2,p1)
        th_t = np.arccos(cos(th_i_prime)/(1.0))*np.sign(th_i_prime)  # n2=1 => no refraction

        # Build scattered wave inside
        k_scat = k_mag
        k_x_scat = k_scat*cos(th_t)*sin(phi_i_prime)
        k_y_scat = k_scat*cos(th_t)*cos(phi_i_prime)
        k_z_scat = k_scat*sin(th_t)

        k_in_crystal = np.array([k_x_scat, k_y_scat, k_z_scat])
        # Solve for Q
        All_Q = solve_q(k_in_crystal, k_scat, gz0, gr0, G, mos_sampling)
        All_Q_flat = All_Q.reshape((-1,3))

        # For each Q => final wave => detector intersection
        for i_sol in range(All_Q_flat.shape[0]):
            Qx = All_Q_flat[i_sol,0]
            Qy = All_Q_flat[i_sol,1]
            Qz = All_Q_flat[i_sol,2]

            k_tx_prime = Qx + k_x_scat
            k_ty_prime = Qy + k_y_scat
            k_tz_prime = Qz + k_z_scat

            kr= sqrt(k_tx_prime**2 + k_ty_prime**2)
            if abs(kr)<1e-12:
                twotheta_t= 0.0
            else:
                twotheta_t= np.arctan(k_tz_prime/kr)

            phi_f= atan2(k_tx_prime, k_ty_prime)
            k_tx_f= k_scat*cos(twotheta_t)*sin(phi_f)
            k_ty_f= k_scat*cos(twotheta_t)*cos(phi_f)
            real_k_tz_f= k_scat*sin(twotheta_t)

            kf = np.array([k_tx_f, k_ty_f, real_k_tz_f])
            kf_prime= R_sample@kf

            dx, dy, dz, valid_det= intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)

            # Log intersection with detector
            mosaic_idx= i_sol//2  # each mosaic angle yields 2 solutions => approximate index
            if DEBUG_ENABLED:
                DEBUG_LOG.append((
                    "intersection-detector",
                    float(H), float(K), float(L),
                    float(mos_sampling[mosaic_idx]) if mosaic_idx<mos_sampling.size else np.nan,
                    bool(valid_det),
                    np.nan, np.nan,  # we fill after check
                    float(Qx), float(Qy), float(Qz),
                    0.0  # final intensity placeholder
                ))

            if not valid_det:
                continue

            plane_to_det= np.array([dx-Detector_Pos[0],
                                    dy-Detector_Pos[1],
                                    dz-Detector_Pos[2]], dtype=np.float64)
            x_det= plane_to_det[0]*e1_det[0] + plane_to_det[1]*e1_det[1] + plane_to_det[2]*e1_det[2]
            y_det= plane_to_det[0]*e2_det[0] + plane_to_det[1]*e2_det[1] + plane_to_det[2]*e2_det[2]

            rpx= int(round(center[0]-y_det/100e-6))
            cpx= int(round(center[1]+x_det/100e-6))
            if not(0<=rpx<image_size and 0<=cpx<image_size):
                continue

            val= (
                reflection_intensity *
                exp(-Qz**2*debye_x**2) *
                exp(-(Qx**2+Qy**2)*debye_y**2)
            )
            image[rpx,cpx]+= val

            # Update max
            if (i_sol%2)==0:
                if image[rpx,cpx]>max_I_sign0:
                    max_I_sign0= image[rpx,cpx]
                    max_x_sign0= cpx
                    max_y_sign0= rpx
            else:
                if image[rpx,cpx]>max_I_sign1:
                    max_I_sign1= image[rpx,cpx]
                    max_x_sign1= cpx
                    max_y_sign1= rpx

            # Update debug log with actual val + x_det,y_det
            if DEBUG_ENABLED:
                DEBUG_LOG[-1]= (
                    "intersection-detector",
                    float(H), float(K), float(L),
                    float(mos_sampling[mosaic_idx]) if mosaic_idx<mos_sampling.size else np.nan,
                    True,     # IntersectionDetector
                    float(x_det),
                    float(y_det),
                    float(Qx),
                    float(Qy),
                    float(Qz),
                    float(val)
                )

            # Optionally store Q data
            if save_flag==1 and q_count[i_peaks_index]<q_data.shape[1]:
                idx= q_count[i_peaks_index]
                q_data[i_peaks_index, idx, 0]= Qx
                q_data[i_peaks_index, idx, 1]= Qy
                q_data[i_peaks_index, idx, 2]= Qz
                q_data[i_peaks_index, idx, 3]= val
                q_data[i_peaks_index, idx, 4]= mos_sampling[mosaic_idx] if mosaic_idx<mos_sampling.size else np.nan
                q_count[i_peaks_index]+=1

    return (
        max_x_sign0, max_y_sign0, max_I_sign0,
        max_x_sign1, max_y_sign1, max_I_sign1
    )


##############################################################################
# 5) process_peaks_parallel
##############################################################################
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
    if DEBUG_ENABLED:
        DEBUG_LOG.append(("process_peaks_parallel", float(len(miller))))

    # Convert angles => rad
    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)

    # Build mosaic distribution
    sigma_rad = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)
    mos_sampling = sample_mosaic_angles_combined(eta_pv, sigma_rad, gamma_rad_m, N=1000, grid_points=100000)

    # Detector-plane transformations
    cg= cos(gamma_rad); sg= sin(gamma_rad)
    cG= cos(Gamma_rad); sG= sin(Gamma_rad)
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
    nd_temp= R_x_det @ n_detector
    n_det_rot= R_z_det@ nd_temp
    nd_len= sqrt(n_det_rot[0]**2 + n_det_rot[1]**2 + n_det_rot[2]**2)
    n_det_rot/= nd_len

    Detector_Pos= np.array([0.0, Distance_CoR_to_Detector, 0.0], dtype=np.float64)

    # e1_det
    dot_e1= (unit_x[0]*n_det_rot[0] + unit_x[1]*n_det_rot[1] + unit_x[2]*n_det_rot[2])
    e1_det= unit_x - dot_e1*n_det_rot
    e1_len= sqrt(e1_det[0]**2 + e1_det[1]**2 + e1_det[2]**2)
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

    # Sample orientation => R_z_R_y
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
    R_z_R_y= R_z@R_y

    # sample-plane normal => R_ZY_n
    n1= np.array([0.0,0.0,1.0], dtype= np.float64)
    R_ZY_n= R_z_R_y@ n1
    norm_nzy= sqrt(R_ZY_n[0]**2 + R_ZY_n[1]**2 + R_ZY_n[2]**2)
    R_ZY_n/= norm_nzy

    # offset
    P0= np.array([0.0,0.0,-zs], dtype= np.float64)

    num_peaks= miller.shape[0]
    max_solutions= 2000000

    if save_flag==1:
        q_data= np.full((num_peaks, max_solutions,5), np.nan, dtype= np.float64)
        q_count= np.zeros(num_peaks, dtype= np.int64)
    else:
        q_data= np.zeros((1,1,5), dtype= np.float64)
        q_count= np.zeros(1, dtype= np.int64)

    max_positions= np.empty((num_peaks,6), dtype= np.float64)

    for i_pk in range(num_peaks):
        H= miller[i_pk,0]
        K= miller[i_pk,1]
        L= miller[i_pk,2]
        reflI= intensities[i_pk]

        (mx0, my0, mv0,
         mx1, my1, mv1) = calculate_phi(
            H, K, L, av, cv,
            wavelength_array,
            image, image_size,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs, zb, n2,
            beam_x_array, beam_y_array,
            theta_array, phi_array,
            reflI,
            mos_sampling,
            debye_x, debye_y,
            center,
            theta_initial_deg,
            R_x_det, R_z_det, n_det_rot, Detector_Pos,
            e1_det, e2_det,
            R_z_R_y, R_ZY_n, P0, unit_x,
            save_flag, q_data, q_count, i_pk
        )

        max_positions[i_pk,0]= mx0
        max_positions[i_pk,1]= my0
        max_positions[i_pk,2]= mv0
        max_positions[i_pk,3]= mx1
        max_positions[i_pk,4]= my1
        max_positions[i_pk,5]= mv1

    return image, max_positions, (q_data if save_flag==1 else None), (q_count if save_flag==1 else None)

