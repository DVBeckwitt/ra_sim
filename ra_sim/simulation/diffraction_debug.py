import os
import csv
import datetime
import numpy as np
from math import sin, cos, sqrt, pi, exp

# --------------------------------------------------------------------
# Global debug log
# --------------------------------------------------------------------
DEBUG_LOG = []

def dump_debug_log():
    """
    Writes the global debug log to ~/Downloads/mosaic_full_debug_log.csv.

    The header includes separate columns for k_x_in, k_y_in, k_z_in,
    and k_x_out, k_y_out, k_z_out for the scattered direction.
    """
    global DEBUG_LOG

    filename = os.path.expanduser("~/Downloads/mosaic_full_debug_log.csv")
    now_str = datetime.datetime.now().isoformat()

    header = [
        "Timestamp",
        "EventType",
        "Reason",
        "H", "K", "L",
        "Reflection_Intensity",
        "Beam_Intensity",
        "Mosaic_Intensity",
        "Divergence_Intensity",
        "MosaicAngle_rad",
        "theta_i",
        "k_x_in", "k_y_in", "k_z_in",
        "k_x_out","k_y_out","k_z_out",
        "gz", "gr", "G",
        "Qx", "Qy", "Qz",
        "Val",
        "BeamStartX", "BeamStartY",
        "IntersectionSample",
        "IntersectionDetector",
        "Intersection_t",
        "Detector_t",
        "DetectorX", "DetectorY",
        "Extra"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for record in DEBUG_LOG:
            row = dict.fromkeys(header, "")
            row["Timestamp"] = now_str

            if len(record) == 0:
                continue

            event_type = record[0]
            row["EventType"] = event_type

            #
            # Match each event type to its expected data pattern
            #
            if event_type == "process_peaks_parallel":
                # e.g. ("process_peaks_parallel", num_miller, sigma_pv_deg, gamma_pv_deg, eta_pv)
                if len(record) >= 5:
                    row["Extra"] = (
                        f"num_miller={record[1]}, "
                        f"sigma_pv_deg={record[2]}, "
                        f"gamma_pv_deg={record[3]}, "
                        f"eta_pv={record[4]}"
                    )

            elif event_type == "mosaic-sampling-begin":
                # e.g. ("mosaic-sampling-begin", sigma_rad, gamma_rad, step_deg, ...)
                if len(record) >= 4:
                    row["Extra"] = (
                        f"sigma_rad={record[1]}, gamma_rad={record[2]}, "
                        f"step_deg={record[3]}"
                    )

            elif event_type == "mosaic-sampling-end":
                # e.g. ("mosaic-sampling-end", count)
                if len(record) > 1:
                    row["Extra"] = str(record[1])

            elif event_type == "calc_phi-begin":
                # e.g. ("calc_phi-begin", H, K, L, reflection_intensity, G)
                if len(record) >= 6:
                    row["H"]                    = record[1]
                    row["K"]                    = record[2]
                    row["L"]                    = record[3]
                    row["Reflection_Intensity"] = record[4]
                    row["G"]                    = record[5]

            elif event_type == "intersection-sample":
                # e.g. ("intersection-sample", H, K, L, bx, by, valid_int, kx_in, ky_in, kz_in)
                if len(record) >= 10:
                    row["H"]                  = record[1]
                    row["K"]                  = record[2]
                    row["L"]                  = record[3]
                    row["BeamStartX"]         = record[4]
                    row["BeamStartY"]         = record[5]
                    row["IntersectionSample"] = record[6]  # True/False
                    row["k_x_in"]             = record[7]
                    row["k_y_in"]             = record[8]
                    row["k_z_in"]             = record[9]

            elif event_type == "intersection-detector":
                # e.g. ("intersection-detector", H, K, L, mosaic_idx, valid_det, x_det, y_det,
                #      Qx, Qy, Qz, val, kx_out, ky_out, kz_out)
                if len(record) >= 15:
                    row["H"]                = record[1]
                    row["K"]                = record[2]
                    row["L"]                = record[3]
                    row["MosaicAngle_rad"]  = record[4]
                    row["IntersectionDetector"] = record[5]  # True/False
                    row["DetectorX"]        = record[6]
                    row["DetectorY"]        = record[7]
                    row["Qx"]               = record[8]
                    row["Qy"]               = record[9]
                    row["Qz"]               = record[10]
                    row["Val"]              = record[11]
                    row["k_x_out"]          = record[12]
                    row["k_y_out"]          = record[13]
                    row["k_z_out"]          = record[14]

            elif event_type == "solve_q-solution":
                # e.g. ("solve_q-solution", mosaic_idx, gz, gr, qx_pos, qy_pos, qz, qx_neg, qy_neg)
                if len(record) >= 9:
                    row["gz"] = record[2]
                    row["gr"] = record[3]
                    row["Extra"] = (
                        f"mosaic_idx={record[1]}, qx_pos={record[4]}, "
                        f"qy_pos={record[5]}, Qz={record[6]}, "
                        f"qx_neg={record[7]}, qy_neg={record[8]}"
                    )

            elif event_type == "Ray Missed it":
                # e.g. ("Ray Missed it", reason, additional_info, ...)
                if len(record) > 1:
                    row["Reason"] = str(record[1])
                if len(record) > 2:
                    row["Extra"] = str(record[2:])

            elif event_type == "Ray Made it":
                row["Extra"] = str(record[1:])

            else:
                # Catch-all
                if len(record) > 1:
                    row["Extra"] = str(record[1:])

            writer.writerow(row)

    print(f"Debug log saved to: {filename}")
    DEBUG_LOG.clear()


# --------------------------------------------------------------------
# Pseudo-Voigt
# --------------------------------------------------------------------
def pseudo_voigt_1d(r, eta, sigma, gamma):
    """
    Returns the value of the 1D pseudo-Voigt function at radius r:
        f(r) = (1 - eta)*Gauss(r) + eta*Lorentz(r)
    where:
      Gauss(r) = (1 / (sqrt(2*pi)*sigma)) * exp(-r^2/(2*sigma^2))
      Lorentz(r) = (1/pi) * (gamma / (r^2 + gamma^2))
    """
    gauss = (1.0 / (sqrt(2.0*pi)*sigma)) * exp(-0.5*(r*r)/(sigma*sigma))
    lorentz = (1.0/pi) * (gamma / (r*r + gamma*gamma))
    return (1.0 - eta)*gauss + eta*lorentz

# --------------------------------------------------------------------
# Mosaic angles grid
# --------------------------------------------------------------------
def grid_mosaic_rotations(sigma_rad, gamma_rad, step_deg=0.1):
    """
    Creates a grid of (beta, kappa) in degrees up to 5*sigma_rad or 5*gamma_rad (whichever is max).
    Returns the rotation matrices for each valid (beta, kappa).
    """
    # Optional: debug log begin
    DEBUG_LOG.append(("mosaic-sampling-begin", sigma_rad, gamma_rad, step_deg))

    r_max = 5.0 * max(sigma_rad, gamma_rad)
    r_max_deg = r_max * 180.0 / pi

    beta_vals_deg = np.arange(-r_max_deg, r_max_deg + step_deg, step_deg)
    kappa_vals_deg = np.arange(-r_max_deg, r_max_deg + step_deg, step_deg)
    n_beta = beta_vals_deg.shape[0]
    n_kappa = kappa_vals_deg.shape[0]

    valid_pts = []
    for i in range(n_beta):
        beta = beta_vals_deg[i] * pi / 180.0
        for j in range(n_kappa):
            kappa = kappa_vals_deg[j] * pi / 180.0
            if sqrt(beta*beta + kappa*kappa) <= r_max:
                valid_pts.append((beta, kappa))

    count = len(valid_pts)
    beta_grid = np.empty(count, dtype=np.float64)
    kappa_grid = np.empty(count, dtype=np.float64)
    rotations = np.empty((count, 3, 3), dtype=np.float64)

    idx = 0
    for (b, k) in valid_pts:
        beta_grid[idx] = b
        kappa_grid[idx] = k

        cosb = cos(b)
        sinb = sin(b)
        R_x = np.array([
            [1.0,  0.0,   0.0],
            [0.0, cosb,  -sinb],
            [0.0, sinb,   cosb]
        ], dtype=np.float64)

        cosk = cos(k)
        sink = sin(k)
        R_y = np.array([
            [ cosk, 0.0, sink],
            [ 0.0,  1.0, 0.0 ],
            [-sink, 0.0, cosk]
        ], dtype=np.float64)

        rotations[idx,:,:] = R_y @ R_x
        idx += 1

    DEBUG_LOG.append(("mosaic-sampling-end", count))
    return rotations

# --------------------------------------------------------------------
# Intersect line and plane
# --------------------------------------------------------------------
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect a parametric line (P0 + t*k_vec) with plane (P_plane, n_plane).
    Returns (ix, iy, iz, valid_bool).
    If t<0 or no intersection => valid_bool=False, coords=NaN.
    """
    denom = k_vec[0]*n_plane[0] + k_vec[1]*n_plane[1] + k_vec[2]*n_plane[2]
    if abs(denom) < 1e-14:
        return (np.nan, np.nan, np.nan, False)

    num = ((P_plane[0] - P0[0]) * n_plane[0]
         + (P_plane[1] - P0[1]) * n_plane[1]
         + (P_plane[2] - P0[2]) * n_plane[2])
    t = num / denom

    if t < 0.0:
        return (np.nan, np.nan, np.nan, False)

    ix = P0[0] + t*k_vec[0]
    iy = P0[1] + t*k_vec[1]
    iz = P0[2] + t*k_vec[2]
    return (ix, iy, iz, True)

# --------------------------------------------------------------------
# Solve Q
# --------------------------------------------------------------------
def solve_q(k_in, k, gz0, gr0, G, R, eps=1e-14):
    """
    For each mosaic rotation R[idx], we rotate the ideal G_ideal=(0,gr0,gz0),
    then solve for Q. Returns array shape (N,2,3). N=R.shape[0].
    """
    N_samples = R.shape[0]
    solutions = np.zeros((N_samples, 2, 3), dtype=np.float64)

    k_x, k_y, k_z = k_in
    G_ideal = np.array([0.0, gr0, gz0])
    if k_y <= 0:
        solutions[:] = np.nan
        return solutions

    k_x_sq = k_x*k_x
    k_y_sq = k_y*k_y
    k_z_sq = k_z*k_z
    k_sq   = k*k
    k_r_sq = (k_x_sq + k_y_sq)
    if k_r_sq <= 0:
        solutions[:] = np.nan
        return solutions

    for idx in range(N_samples):
        # Rotate G_ideal by mosaic rotation
        G_rot0 = R[idx, 0,0]*G_ideal[0] + R[idx, 0,1]*G_ideal[1] + R[idx, 0,2]*G_ideal[2]
        G_rot1 = R[idx, 1,0]*G_ideal[0] + R[idx, 1,1]*G_ideal[1] + R[idx, 1,2]*G_ideal[2]
        G_rot2 = R[idx, 2,0]*G_ideal[0] + R[idx, 2,1]*G_ideal[1] + R[idx, 2,2]*G_ideal[2]

        gr = sqrt(G_rot0*G_rot0 + G_rot1*G_rot1)
        gz = G_rot2
        gz_sq = gz*gz
        gr_sq = gr*gr

        # The main radical that must be >= 0
        sqrt_term = (
            -gr**4 - 2*gr_sq*gz_sq - 4*gr_sq*gz*k_z + 2*gr_sq*k_sq + 2*gr_sq*k_x_sq
            + 2*gr_sq*k_y_sq - 2*gr_sq*k_z_sq - gz**4 - 4*gz**3*k_z + 2*gz_sq*k_sq
            - 2*gz_sq*k_x_sq - 2*gz_sq*k_y_sq - 6*gz_sq*k_z_sq + 4*gz*k_sq*k_z
            - 4*gz*k_x_sq*k_z - 4*gz*k_y_sq*k_z - 4*gz*k_z**3 - k**4 + 2*k_sq*k_x_sq
            + 2*k_sq*k_y_sq + 2*k_sq*k_z_sq - k_x**4 - 2*k_x_sq*k_y_sq
            - 2*k_x_sq*k_z_sq - k_y**4 - 2*k_y_sq*k_z_sq - k_z**4
        )

        if sqrt_term < 0:
            # No real solution => log and skip
            solutions[idx, 0, :] = np.nan
            solutions[idx, 1, :] = np.nan
            continue

        # Otherwise, compute positive/negative solutions
        term1 = -k_x*(gr_sq + gz_sq + 2*gz*k_z - k_sq + k_x_sq + k_y_sq + k_z_sq)
        term2 = k_y*sqrt(sqrt_term)

        qy_term1 = - (gr_sq + gz_sq + 2*gz*k_z - k_sq + k_x_sq + k_y_sq + k_z_sq)
        qx_bottom = 2.0 * (k_r_sq)
        qy_bottom = 2.0 * k_y

        qx_positive = (term1 - term2) / qx_bottom
        qx_negative = (term1 + term2) / qx_bottom

        qy_positive = (qy_term1 - 2*k_x*qx_positive) / qy_bottom
        qy_negative = (qy_term1 - 2*k_x*qx_negative) / qy_bottom

        solutions[idx, 0, 0] = qx_positive
        solutions[idx, 0, 1] = qy_positive
        solutions[idx, 0, 2] = gz
        solutions[idx, 1, 0] = qx_negative
        solutions[idx, 1, 1] = qy_negative
        solutions[idx, 1, 2] = gz

        # Debug log for solutions
        DEBUG_LOG.append((
            "solve_q-solution",
            idx,         # mosaic index
            gz,          # gz
            gr,          # gr
            qx_positive, # qx_pos
            qy_positive, # qy_pos
            gz,          # Qz is same as gz in the final line
            qx_negative, # qx_neg
            qy_negative  # qy_neg
        ))

    return solutions

# --------------------------------------------------------------------
# Calculate phi for a single reflection
# --------------------------------------------------------------------
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
    R_z_R_y,   # combined rotation for sample
    R_ZY_n,    # sample-plane normal in sample coords
    P0, unit_x,
    save_flag, q_data, q_count, i_peaks_index
):
    """
    For a single reflection (H,K,L), samples multiple incoming rays, solves Q,
    and intersects with the detector. Accumulates intensities in `image`.
    Returns maximum intensities and pixel coords for the two solutions.
    """
    # Debug log: reflection begin
    G = None
    # Calculate G from (H,K,L) (just for logging)
    gz0 = 2.0*np.pi * (L/cv)
    gr0 = 4.0*np.pi/av * sqrt((H*H + H*K + K*K)/3.0)
    G   = sqrt(gr0**2 + gz0**2)

    DEBUG_LOG.append(("calc_phi-begin", H, K, L, reflection_intensity, G))

    gz0 = 2.0*np.pi * (L/cv)
    gr0 = 4.0*np.pi/av * sqrt((H*H + H*K + K*K)/3.0)

    max_I_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan
    max_I_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan

    # Build sample orientation for this reflection
    rad_theta_i = theta_initial_deg*(pi/180.0)
    R_x = np.array([
        [1.0,           0.0,           0.0 ],
        [0.0, cos(rad_theta_i), -sin(rad_theta_i)],
        [0.0, sin(rad_theta_i),  cos(rad_theta_i)]
    ])
    R_sample = R_x @ R_z_R_y

    # Normal in lab
    n_surf = R_sample @ R_ZY_n
    n_surf /= np.linalg.norm(n_surf)
    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0  # enforce x=0 ?

    # Build e1_temp, e2_temp on sample surface
    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_len = np.linalg.norm(e1_temp)
    if e1_len < 1e-12:
        # fallback
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
            return (max_I_sign0, max_x_sign0, max_y_sign0,
                    max_I_sign1, max_x_sign1, max_y_sign1)
    else:
        e1_temp /= e1_len

    e2_temp = np.cross(n_surf, e1_temp)

    n_samp = beam_x_array.size
    for i_samp in range(n_samp):
        lam_samp = wavelength_array[i_samp]
        k_mag = 2.0*pi / lam_samp

        bx = beam_x_array[i_samp]
        by = beam_y_array[i_samp]
        beam_start = np.array([bx, -20e-3, -zb + by], dtype=np.float64)

        dtheta = theta_array[i_samp]
        dphi   = phi_array[i_samp]

        # Incident wave in lab
        k_in = np.array([
            cos(dtheta)*sin(dphi),
            cos(dtheta)*cos(dphi),
            sin(dtheta)
        ], dtype=np.float64)

        # Intersection with sample
        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        # Log intersection
        DEBUG_LOG.append((
            "intersection-sample",
            H, K, L,
            bx, by,
            valid_int,
            k_in[0], k_in[1], k_in[2]
        ))
        if not valid_int:
            DEBUG_LOG.append((
                "Ray Missed it",
                "No valid intersection with sample plane",
                (H, K, L, bx, by, k_in[0], k_in[1], k_in[2])
            ))
            continue

        I_plane = np.array([ix, iy, iz])

        # Incident angle wrt sample normal
        kn_dot = k_in[0]*n_surf[0] + k_in[1]*n_surf[1] + k_in[2]*n_surf[2]
        th_i_prime = (pi/2.0) - np.arccos(kn_dot)

        # Project onto sample to get azimuth
        projected_incident = k_in - kn_dot*n_surf
        proj_len = np.linalg.norm(projected_incident)
        if proj_len > 1e-12:
            projected_incident /= proj_len
        else:
            projected_incident[:] = 0.0

        p1 = projected_incident[0]*e1_temp[0] + projected_incident[1]*e1_temp[1] + projected_incident[2]*e1_temp[2]
        p2 = projected_incident[0]*e2_temp[0] + projected_incident[1]*e2_temp[1] + projected_incident[2]*e2_temp[2]
        phi_i_prime = (pi/2.0) - np.arctan2(p2, p1)

        # Refracted wave (simple model)
        th_t = np.arccos(cos(th_i_prime)/np.real(n2))*np.sign(th_i_prime)
        k_scat = k_mag * sqrt((np.real(n2))**2)
        k_x_scat = k_scat*cos(th_t)*sin(phi_i_prime)
        k_y_scat = k_scat*cos(th_t)*cos(phi_i_prime)
        k_z_scat = k_scat*sin(th_t)

        k_in_crystal = np.array([k_x_scat, k_y_scat, k_z_scat], dtype=np.float64)

        # Solve for Q
        All_Q = solve_q(k_in_crystal, k_scat, gz0, gr0, G, Mosaic_Rotation)
        All_Q_flat = All_Q.reshape((-1, 3))  # shape (2*N_mosaic, 3)

        # For each Q-solution
        for i_sol in range(All_Q_flat.shape[0]):
            Qx = All_Q_flat[i_sol, 0]
            Qy = All_Q_flat[i_sol, 1]
            Qz = All_Q_flat[i_sol, 2]
            if np.isnan(Qx) or np.isnan(Qy) or np.isnan(Qz):
                continue

            # Final scattered wave in crystal
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

            # Rotate scattered wave back to lab (sample orientation):
            kf = np.array([k_tx_f, k_ty_f, real_k_tz_f])
            kf_prime = R_sample @ kf

            # Intersect with detector plane
            dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
            if not valid_det:
                DEBUG_LOG.append((
                    "Ray Missed it",
                    "No valid intersection with detector plane",
                    (H, K, L, i_sol, Qx, Qy, Qz, k_tx_f, k_ty_f, k_tz_prime)
                ))
                continue

            plane_to_det = np.array([dx - Detector_Pos[0],
                                     dy - Detector_Pos[1],
                                     dz - Detector_Pos[2]], dtype=np.float64)

            x_det = plane_to_det[0]*e1_det[0] + plane_to_det[1]*e1_det[1] + plane_to_det[2]*e1_det[2]
            y_det = plane_to_det[0]*e2_det[0] + plane_to_det[1]*e2_det[1] + plane_to_det[2]*e2_det[2]

            rpx = int(round(center[0] - y_det / 100e-6))
            cpx = int(round(center[1] + x_det / 100e-6))

            # Log intersection with detector
            val = 0.0
            mosaic_idx = i_sol // 2  # just a guess for "mosaic angle" index
            DEBUG_LOG.append((
                "intersection-detector",
                H, K, L,
                mosaic_idx,      # mosaic angle (approx)
                valid_det,
                cpx, rpx,        # DetectorX, DetectorY
                Qx, Qy, Qz,
                None,            # placeholder for intensity we compute below
                k_tx_f, k_ty_f, real_k_tz_f
            ))

            if not (0 <= rpx < image_size and 0 <= cpx < image_size):
                continue

            val = (
                reflection_intensity
                * np.exp(-Qz**2 * debye_x**2)
                * np.exp(-(Qx**2 + Qy**2)*debye_y**2)
            )
            image[rpx, cpx] += val

            # Update the last debug log entry with 'Val'
            DEBUG_LOG[-1] = (
                "intersection-detector",
                H, K, L,
                mosaic_idx,
                valid_det,
                cpx, rpx,
                Qx, Qy, Qz,
                val,
                k_tx_f, k_ty_f, real_k_tz_f
            )

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

            # Optionally store Q-data
            if save_flag == 1 and q_count[i_peaks_index] < q_data.shape[1]:
                idx_q = q_count[i_peaks_index]
                q_data[i_peaks_index, idx_q, 0] = Qx
                q_data[i_peaks_index, idx_q, 1] = Qy
                q_data[i_peaks_index, idx_q, 2] = Qz
                q_data[i_peaks_index, idx_q, 3] = val
                q_count[i_peaks_index] += 1

    return (
        max_I_sign0, max_x_sign0, max_y_sign0,
        max_I_sign1, max_x_sign1, max_y_sign1
)

# --------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------
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
    """
    Replacement for the original parallel code. Loops are purely Pythonic.
    Builds mosaic rotations, loops reflections, calls calculate_phi, etc.
    """
    # Log that we are starting
    DEBUG_LOG.append((
        "process_peaks_parallel",
        miller.shape[0],
        sigma_pv_deg,
        gamma_pv_deg,
        eta_pv
    ))

    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)

    sigma_rad = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)

    # Build mosaic distribution
    Mosaic_Rotation = grid_mosaic_rotations(sigma_rad, gamma_rad_m)

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
    nd_len = np.linalg.norm(n_det_rot)
    if nd_len < 1e-14:
        n_det_rot = np.array([0.0, 1.0, 0.0])
    else:
        n_det_rot /= nd_len

    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0], dtype=np.float64)

    # e1_det => projection of unit_x onto plane orthonormal to n_det_rot
    dot_e1 = unit_x[0]*n_det_rot[0] + unit_x[1]*n_det_rot[1] + unit_x[2]*n_det_rot[2]
    e1_det = unit_x - dot_e1*n_det_rot
    e1_len = np.linalg.norm(e1_det)
    if e1_len < 1e-14:
        e1_det = np.array([1.0, 0.0, 0.0])
    else:
        e1_det /= e1_len

    tmpx = n_det_rot[1]*e1_det[2] - n_det_rot[2]*e1_det[1]
    tmpy = n_det_rot[2]*e1_det[0] - n_det_rot[0]*e1_det[2]
    tmpz = n_det_rot[0]*e1_det[1] - n_det_rot[1]*e1_det[0]
    e2_det = np.array([-tmpx, -tmpy, -tmpz], dtype=np.float64)
    e2_len = np.linalg.norm(e2_det)
    if e2_len<1e-14:
        e2_det= np.array([0.0, 1.0, 0.0])
    else:
        e2_det /= e2_len

    # Sample orientation => R_z, R_y
    c_chi = cos(chi_rad); s_chi = sin(chi_rad)
    R_y = np.array([
        [ c_chi, 0.0,   s_chi],
        [ 0.0,   1.0,   0.0 ],
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
    R_ZY_n /= np.linalg.norm(R_ZY_n)

    # sample plane offset => P0
    P0 = np.array([0.0, 0.0, -zs], dtype=np.float64)

    num_peaks= miller.shape[0]
    max_solutions= 2000000

    if save_flag==1:
        q_data= np.full((num_peaks, max_solutions, 5), np.nan, dtype= np.float64)
        q_count= np.zeros(num_peaks, dtype= np.int64)
    else:
        q_data= np.zeros((1, 1, 5), dtype= np.float64)
        q_count= np.zeros(1, dtype= np.int64)

    max_positions= np.empty((num_peaks, 6), dtype= np.float64)

    # Main reflection loop
    for i_pk in range(num_peaks):
        H= miller[i_pk,0]
        K= miller[i_pk,1]
        L= miller[i_pk,2]
        reflI= intensities[i_pk]

        (
          mx0, my0, mv0,
          mx1, my1, mv1
        ) = calculate_phi(
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
