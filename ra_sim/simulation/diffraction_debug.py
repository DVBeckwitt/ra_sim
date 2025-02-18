import os
import matplotlib
matplotlib.use('TkAgg')  # Use a backend suitable for script execution
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy import cos, sin, sqrt
from ra_sim.StructureFactor.StructureFactor import attenuation
from ra_sim.utils.calculations import fresnel_transmission, IndexofRefraction

##############################################################################
# GLOBAL DEBUG LIST
# We'll store tuples of debug data and write them to a file after the run.
##############################################################################
DEBUG_LOG = []
DEBUG_ENABLED = True  # Set False if you want to skip logging

def G(H, K, L, av, cv, beta, kappa):
    """
    Compute the components of the reciprocal lattice vector G.
    Gz and Gr are combined to get final gz, gr after rotation by beta, kappa.
    """
    Gz = 2 * np.pi * L / cv
    Gr = 4 * np.pi / av * sqrt((H**2 + H*K + K**2) / 3.0)

    # Rotation by (kappa - beta)
    gr = Gr * cos(kappa - beta) - Gz * sin(kappa - beta)
    gz = Gr * sin(kappa - beta) + Gz * cos(kappa - beta)
    return gz, gr

def compute_d(gz, gr):
    """
    Example function to compute d-spacing from gz, gr.
    Not essential to this demonstration.
    """
    d = (gz**2 + gr**2)**(-0.5) * (2 * np.pi)
    return d

def solve_q(k_in, k, G, gz):
    """
    Solve the system:
      1) Q_x^2 + Q_y^2 + gz^2 = G^2
      2) (k_in + Q)^2 = k^2

    For a given gz, where Q = (Q_x, Q_y, gz).
    Log additional data (RHS, disc, reason) when returning.
    """
    global DEBUG_LOG, DEBUG_ENABLED

    R2 = G*G - gz*gz
    if R2 < 0:
        reason = "R2 < 0 => no real solutions for Q."
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "solve_q-return:R2<0",
                float(R2),
                float(G),
                float(gz),
                float(k_in[0]),
                float(k_in[1]),
                float(k_in[2]),
                reason
            ))
        return np.array([[np.nan, np.nan],
                         [np.nan, np.nan]], dtype=np.float64)

    R = np.sqrt(R2)
    RHS = 0.5 * (k*k - (np.dot(k_in, k_in) + G*G)) - k_in[2]*gz
    kx, ky, _ = k_in
    denom_sq = kx*kx + ky*ky

    if denom_sq < 1e-14:
        if abs(RHS) > 1e-14:
            reason = "Near-normal incidence and RHS != 0 => no Q solution."
            if DEBUG_ENABLED:
                DEBUG_LOG.append((
                    "solve_q-return:denom_sq<1e-14,RHS>1e-14",
                    float(R2),
                    float(denom_sq),
                    float(G),
                    float(gz),
                    float(k_in[0]),
                    float(k_in[1]),
                    float(k_in[2]),
                    float(RHS),
                    reason
                ))
            return np.array([[np.nan, np.nan],
                             [np.nan, np.nan]], dtype=np.float64)
        else:
            # Infinite solutions along a circle
            reason = "Near-normal incidence and RHS=0 => infinite solutions."
            if DEBUG_ENABLED:
                DEBUG_LOG.append((
                    "solve_q-return:infinite_solutions",
                    float(R2),
                    float(denom_sq),
                    float(G),
                    float(gz),
                    float(k_in[0]),
                    float(k_in[1]),
                    float(k_in[2]),
                    float(RHS),
                    reason
                ))
            sol = np.zeros((2, 2), dtype=np.float64)
            sol[0, 0] = R  # Qx=+R, Qy=0
            sol[0, 1] = 0
            sol[1, 0] = -R # Qx=-R, Qy=0
            sol[1, 1] =  0
            return sol

    a = kx
    b = ky
    A = (b*b)/(a*a) + 1.0
    B = -2.0*b*RHS/(a*a)
    C = (RHS*RHS)/(a*a) - R*R
    disc = B*B - 4.0*A*C

    if disc < -1e-14:
        reason = "Discriminant < 0 => no real Q solutions."
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "solve_q-return:disc<-1e-14",
                float(disc),
                float(R2),
                float(denom_sq),
                float(G),
                float(gz),
                float(k_in[0]),
                float(k_in[1]),
                float(k_in[2]),
                float(RHS),
                reason
            ))
        return np.array([[np.nan, np.nan],
                         [np.nan, np.nan]], dtype=np.float64)
    if disc < 0.0:
        # Tangential
        reason = "Tangential intersection => disc=0."
        disc = 0.0
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "solve_q:tangent->disc=0.0",
                float(disc),
                float(R2),
                float(denom_sq),
                float(G),
                float(gz),
                float(k_in[0]),
                float(k_in[1]),
                float(k_in[2]),
                float(RHS),
                reason
            ))

    sqrt_disc = np.sqrt(disc)
    sol = np.zeros((2, 2), dtype=np.float64)

    Qy1 = (-B + sqrt_disc)/(2.0*A)
    Qx1 = (RHS - b*Qy1)/a
    sol[0, 0] = Qx1
    sol[0, 1] = Qy1

    Qy2 = (-B - sqrt_disc)/(2.0*A)
    Qx2 = (RHS - b*Qy2)/a
    sol[1, 0] = Qx2
    sol[1, 1] = Qy2

    return sol

def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Compute intersection of a line and a plane:
      P0 + t * k_vec = Intersection_Point
    Returns (Intersection_Point, t). If no intersection, returns (None, None).
    """
    denominator = np.dot(k_vec, n_plane)
    if denominator == 0:
        return None, None
    t = np.dot(P_plane - P0, n_plane) / denominator
    Intersection_Point = P0 + t * k_vec
    return Intersection_Point, t

def calculate_phi(
    H, K, L, av, cv, lambda_, image, image_size,
    gamma_rad, Gamma_rad, chi_rad, psi_rad,
    zs, zb, n2,
    beam_x_array, beam_y_array, beam_intensity_array,
    beta_array, kappa_array, mosaic_intensity_array,
    theta_array, phi_array, divergence_intensity_array,
    debye_x, debye_y, center,
    theta_initial, theta_final, step, intensity,
    k, R_x_detector, R_z_detector, n_det_rot, Detector_Pos, e1_det, e2_det,
    R_z_R_y, R_ZY_n, P0, unit_x,
    save_flag,        # 0 or 1 to indicate if we save Q data with intensity
    q_data,           # shape: (num_peaks, max_solutions, 5) [Qx, Qy, Qz, gz, Mv]
    q_count,          # shape: (num_peaks,)
    i_peaks_index,
    DEBUG_ENABLED=False
):
    if DEBUG_ENABLED:
        DEBUG_LOG.append(("calc_phi-begin", int(H), int(K), int(L)))

    gz0, gr0 = G(H, K, L, av, cv, 0.0, 0.0)
    theta_range = np.arange(theta_initial, theta_final, step)

    max_intensity_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan

    max_intensity_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan

    for theta_i in theta_range:
        rad_theta_i = np.radians(theta_i)
        Ti = np.abs(fresnel_transmission(np.deg2rad(theta_i), n2))**2

        R_x = np.array([
            [1.0,               0.0,               0.0],
            [0.0, cos(rad_theta_i), -sin(rad_theta_i)],
            [0.0, sin(rad_theta_i),  cos(rad_theta_i)]
        ])
        R_sample = R_x @ R_z_R_y

        n = R_x @ R_ZY_n
        n /= np.linalg.norm(n)
        P0_rot = R_x @ P0
        P0_rot[0] = 0.0

        u_ref = np.array([0.0, 0.0, -1.0])
        e1 = np.cross(n, u_ref)
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-12:
            alt_refs = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0])
            ]
            success = False
            for alt_ref in alt_refs:
                cross_tmp = np.cross(n, alt_ref)
                cross_norm_tmp = np.linalg.norm(cross_tmp)
                if cross_norm_tmp > 1e-12:
                    e1 = cross_tmp / cross_norm_tmp
                    success = True
                    break
            if not success:
                continue
        else:
            e1 /= e1_norm

        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2)

        for idx in range(len(beam_x_array)):
            x = beam_x_array[idx]
            y = beam_y_array[idx]
            beam_intensity = beam_intensity_array[idx]

            beta = beta_array[idx]
            kappa = kappa_array[idx]
            mosaic_intensity = mosaic_intensity_array[idx]

            delta_theta_i_spot = theta_array[idx]
            delta_phi_i_spot   = phi_array[idx]
            divergence_intensity = divergence_intensity_array[idx]

            gz, gr = G(H, K, L, av, cv, beta, kappa)
            beam_start = np.array([x, -20e-3, -zb + y])

            k_xi = cos(delta_theta_i_spot) * sin(delta_phi_i_spot)
            k_yi = cos(delta_theta_i_spot) * cos(delta_phi_i_spot)
            k_zi = sin(delta_theta_i_spot)
            k_vec = np.array([k_xi, k_yi, k_zi])
            k_vec /= np.linalg.norm(k_vec)

            kn_dot = np.dot(k_vec, n)
            Intersection_Point_Plane, t = intersect_line_plane(beam_start, k_vec, P0_rot, n)
            if Intersection_Point_Plane is None or t < 0:
                continue

            th_i_prime = (np.pi / 2) - np.arccos(kn_dot)
            projected_incident = k_vec - kn_dot * n
            proj_norm = np.linalg.norm(projected_incident)
            if proj_norm > 0.0:
                projected_incident /= proj_norm
            else:
                projected_incident = np.array([0.0, 0.0, 0.0])

            p1 = np.dot(projected_incident, e1)
            p2 = np.dot(projected_incident, e2)
            phi_i_prime = (np.pi / 2) - np.arctan2(p2, p1)

            th_t = np.arccos(cos(th_i_prime)/ np.real(n2)) * np.sign(th_i_prime)
            k_0 = 2*np.pi/lambda_
            k_scat = k_0 * sqrt(np.real(n2**2))

            k_re = k_0**2 * np.real(n2**2)
            k_im = k_0**2 * np.imag(n2**2)

            k_x_scatter = k_scat * cos(th_t) * sin(phi_i_prime)
            k_y_scatter = k_scat * cos(th_t) * cos(phi_i_prime)
            at = k_re * sin(th_t)**2
            b  = k_im
            g = np.sqrt(gr**2 + gz**2)
            k_z_scatter = np.sign(th_t) * sqrt((sqrt(at**2 + b**2) + at)/2)
            imaginary_k_z_scatter = sqrt((sqrt(at**2 + b**2) - at)/2)
            k_in = np.array([k_x_scatter, k_y_scatter, k_z_scatter])

            # The main solve for Q
            All_Q = solve_q(k_in, k, g, gz)

            for sign in range(2):
                Qx, Qy = All_Q[sign]
                if np.isnan(Qx) or np.isnan(Qy):
                    continue

                k_tx_prime = Qx + k_x_scatter
                k_ty_prime = Qy + k_y_scatter
                k_tz_prime = gz + k_z_scatter
                k_tz_prime2 = gz0 + k_z_scatter

                kr = sqrt(k_tx_prime**2 + k_ty_prime**2)
                if abs(kr) < 1e-12:
                    twotheta_t = 0.0
                else:
                    twotheta_t  = np.arctan(k_tz_prime/ kr)

                if abs(gr0) > 1e-12:
                    twotheta_t2 = np.arctan(k_tz_prime2 / gr0)
                else:
                    twotheta_t2 = 0.0

                phi_f = np.arctan2(k_tx_prime, k_ty_prime)
                k_tx_f = k_scat * cos(twotheta_t) * sin(phi_f)
                k_ty_f = k_scat * cos(twotheta_t) * cos(phi_f)
                af  = k_re * sin(twotheta_t)**2
                af2 = k_re * sin(twotheta_t2)**2

                real_k_tz_f  = sqrt((sqrt(af**2 + b**2) + af)/2) * np.sign(twotheta_t)
                real_k_tz_f2 = sqrt((sqrt(af2**2+ b**2)+ af2)/2)* np.sign(twotheta_t2)
                imaginary_k_tz_f2 = sqrt((sqrt(af2**2+ b**2)- af2)/2)

                kf = np.array([k_tx_f, k_ty_f, real_k_tz_f])
                kf_prime = R_sample @ kf

                k_r = sqrt(kf_prime[0]**2 + kf_prime[1]**2)
                if k_r < 1e-12:
                    tth = 0.0
                else:
                    tth = abs(np.arctan(kf_prime[2]/ k_r))

                Tf = abs(fresnel_transmission(tth, n2))**2
                qz_real = real_k_tz_f2 - k_z_scatter
                qz_imag = imaginary_k_z_scatter - imaginary_k_z_scatter  # effectively 0

                Intersection_Point_Detector, s_ = intersect_line_plane(
                    Intersection_Point_Plane, kf_prime, Detector_Pos, n_det_rot
                )
                if Intersection_Point_Detector is None or s_ < 0:
                    continue

                Plane_to_Detector = Intersection_Point_Detector - Detector_Pos
                x_det = np.dot(Plane_to_Detector, e1_det)
                y_det = np.dot(Plane_to_Detector, e2_det)

                Mv = (
                    intensity
                    * beam_intensity
                    * mosaic_intensity
                    * divergence_intensity
                    * Tf
                    * Ti
                    * np.exp(-gz**2 * debye_x**2)
                    * np.exp(-gr**2 * debye_y**2)
                )

                if DEBUG_ENABLED:
                    DEBUG_LOG.append((
                        "Ray Made it",
                        float(H), float(K), float(L),
                        float(beta), float(kappa),
                        float(theta_i), float(Mv),
                        float(x_det), float(y_det),
                        float(Qx), float(Qy), float(gz), float(gr)
                    ))

                if not np.isnan(Mv):
                    row_center = center[0]
                    col_center = center[1]
                    px_size = 100e-6

                    pixel_row = int(round(row_center - y_det / px_size))
                    pixel_col = int(round(col_center + x_det / px_size))

                    if (0 <= pixel_row < image_size) and (0 <= pixel_col < image_size):
                        image[pixel_row, pixel_col] += Mv
                        if sign == 0:
                            if image[pixel_row, pixel_col] > max_intensity_sign0:
                                max_intensity_sign0 = image[pixel_row, pixel_col]
                                max_x_sign0 = pixel_col
                                max_y_sign0 = pixel_row
                        else:
                            if image[pixel_row, pixel_col] > max_intensity_sign1:
                                max_intensity_sign1 = image[pixel_row, pixel_col]
                                max_x_sign1 = pixel_col
                                max_y_sign1 = pixel_row

                # Save Q data if requested
                if save_flag == 1:
                    current_count = q_count[i_peaks_index]
                    if current_count < q_data.shape[1]:
                        q_data[i_peaks_index, current_count, 0] = Qx
                        q_data[i_peaks_index, current_count, 1] = Qy
                        q_data[i_peaks_index, current_count, 2] = (gz + k_z_scatter)
                        q_data[i_peaks_index, current_count, 3] = gz
                        q_data[i_peaks_index, current_count, 4] = Mv
                        q_count[i_peaks_index] += 1

    return (
        max_x_sign0, max_y_sign0, max_intensity_sign0,
        max_x_sign1, max_y_sign1, max_intensity_sign1
    )

def process_peaks_parallel_debug(
    miller, intensities, image_size,
    av, cv, lambda_, image,
    Distance_CoR_to_Detector, gamma, Gamma, chi, psi, zs, zb, n2,
    beam_x_array, beam_y_array, beam_intensity_array,
    beta_array, kappa_array, mosaic_intensity_array,
    theta_array, phi_array, divergence_intensity_array,
    debye_x, debye_y, center, theta_initial, theta_final, step,
    unit_x, n_detector,
    save_flag
):
    DEBUG_ENABLED = True
    if DEBUG_ENABLED:
        DEBUG_LOG.append(("process_peaks_parallel", float(len(miller))))

    gamma_rad = np.radians(gamma)
    Gamma_rad = np.radians(Gamma)
    chi_rad = np.radians(chi)
    psi_rad = np.radians(psi)
    k = 2 * np.pi / lambda_

    R_x_detector = np.array([
        [1.0,             0.0,            0.0],
        [0.0, cos(gamma_rad),  sin(gamma_rad)],
        [0.0, -sin(gamma_rad), cos(gamma_rad)]
    ])
    R_z_detector = np.array([
        [ cos(Gamma_rad), sin(Gamma_rad), 0.0],
        [-sin(Gamma_rad), cos(Gamma_rad), 0.0],
        [ 0.0,            0.0,            1.0]
    ])
    n_det_rot = R_z_detector @ (R_x_detector @ n_detector)
    n_det_rot /= np.linalg.norm(n_det_rot)
    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0])

    e1_det = unit_x - np.dot(unit_x, n_det_rot) * n_det_rot
    e1_det /= np.linalg.norm(e1_det)
    e2_det = -np.cross(n_det_rot, e1_det)
    e2_det /= np.linalg.norm(e2_det)

    R_y = np.array([
        [ cos(chi_rad), 0.0, sin(chi_rad)],
        [ 0.0,          1.0, 0.0         ],
        [-sin(chi_rad), 0.0, cos(chi_rad)]
    ])
    R_z = np.array([
        [ cos(psi_rad), sin(psi_rad), 0.0],
        [-sin(psi_rad), cos(psi_rad), 0.0],
        [ 0.0,          0.0,          1.0]
    ])
    R_z_R_y = R_z @ R_y
    n1 = np.array([0.0, 0.0, 1.0])
    R_ZY_n = R_z_R_y @ n1
    R_ZY_n /= np.linalg.norm(R_ZY_n)

    P0 = np.array([0.0, 0.0, -zs])
    num_peaks = len(miller)
    num_theta_steps = int((theta_final - theta_initial) // step)
    if num_theta_steps < 1:
        num_theta_steps = 1
    max_solutions = num_theta_steps * len(beam_x_array) * 2

    if save_flag == 1:
        q_data = np.full((num_peaks, max_solutions, 5), np.nan, dtype=np.float64)
        q_count = np.zeros(num_peaks, dtype=np.int64)
    else:
        q_data = np.zeros((1, 1, 5), dtype=np.float64)
        q_count = np.zeros(1, dtype=np.int64)

    max_positions = np.empty((num_peaks, 6), dtype=np.float64)

    for i in range(num_peaks):
        H, K, L = miller[i]
        reflection_intensity = intensities[i]

        (mx0, my0, mv0,
         mx1, my1, mv1) = calculate_phi(
            H, K, L, av, cv, lambda_, image, image_size,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs, zb, n2,
            beam_x_array, beam_y_array, beam_intensity_array,
            beta_array, kappa_array, mosaic_intensity_array,
            theta_array, phi_array, divergence_intensity_array,
            debye_x, debye_y, center,
            theta_initial, theta_final, step, reflection_intensity,
            k, R_x_detector, R_z_detector, n_det_rot, Detector_Pos,
            e1_det, e2_det, R_z_R_y, R_ZY_n, P0, unit_x,
            save_flag,
            q_data,
            q_count,
            i, DEBUG_ENABLED=True
        )

        max_positions[i, 0] = mx0
        max_positions[i, 1] = my0
        max_positions[i, 2] = mv0
        max_positions[i, 3] = mx1
        max_positions[i, 4] = my1
        max_positions[i, 5] = mv1

    dump_debug_log()
    if save_flag == 1:
        return image, max_positions, q_data, q_count
    else:
        return image, max_positions, None, None

import os
import csv
import datetime

def dump_debug_log():
    """
    Write all debug-log records to ~/Downloads/ewald_debug_log.csv with a structured format.
    The CSV will have fixed headers for common quantities. Extra fields go under 'Extra' or 'Reason'.
    """
    filename = os.path.expanduser("~/Downloads/ewald_debug_log.csv")
    now_str = datetime.datetime.now().isoformat()

    # Expanded header to include new fields like Reason, RHS, etc.
    header = [
        "Timestamp",    
        "EventType",    
        "Reason",       # Why a return or condition triggered
        "H", "K", "L",
        "beta", "kappa",
        "theta_i", "Mv",
        "x_det", "y_det",
        "Qx", "Qy",
        "gz", "gr",
        "R2", "denom_sq", "disc", "RHS",  # new numeric fields
        "G", "k_in_0", "k_in_1", "k_in_2",
        "Extra"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for record in DEBUG_LOG:
            row = {
                "Timestamp": now_str,
                "EventType": record[0],
                "Reason": "",
                "H": "",
                "K": "",
                "L": "",
                "beta": "",
                "kappa": "",
                "theta_i": "",
                "Mv": "",
                "x_det": "",
                "y_det": "",
                "Qx": "",
                "Qy": "",
                "gz": "",
                "gr": "",
                "R2": "",
                "denom_sq": "",
                "disc": "",
                "RHS": "",
                "G": "",
                "k_in_0": "",
                "k_in_1": "",
                "k_in_2": "",
                "Extra": ""
            }

            event = record[0]

            # Examples:
            if event == "calc_phi-begin":
                # record: ("calc_phi-begin", H, K, L)
                row["H"] = record[1]
                row["K"] = record[2]
                row["L"] = record[3]

            elif event == "Ray Made it":
                # ("Ray Made it", H,K,L, beta,kappa, theta_i, Mv, x_det, y_det, Qx, Qy, gz, gr)
                row["H"] = record[1]
                row["K"] = record[2]
                row["L"] = record[3]
                row["beta"] = record[4]
                row["kappa"] = record[5]
                row["theta_i"] = record[6]
                row["Mv"] = record[7]
                row["x_det"] = record[8]
                row["y_det"] = record[9]
                row["Qx"] = record[10]
                row["Qy"] = record[11]
                row["gz"] = record[12]
                row["gr"] = record[13]

            elif event == "process_peaks_parallel":
                # record: ("process_peaks_parallel", num_miller)
                row["Extra"] = f"num_miller: {record[1]}"

            elif event.startswith("solve_q-return:") or event.startswith("solve_q:tangent->"):
                # We store reason as the last entry in these records
                # Example: ("solve_q-return:R2<0", R2, G, gz, k_in0, k_in1, k_in2, reason)
                if "R2<0" in event:
                    row["EventType"] = "solve_q-return:R2<0"
                    row["R2"] = record[1]
                    row["G"]  = record[2]
                    row["gz"] = record[3]
                    row["k_in_0"] = record[4]
                    row["k_in_1"] = record[5]
                    row["k_in_2"] = record[6]
                    row["Reason"] = record[7]

                elif "denom_sq<1e-14,RHS>1e-14" in event:
                    # record: (..., R2, denom_sq, G, gz, k_in0, k_in1, k_in2, RHS, reason)
                    row["R2"] = record[1]
                    row["denom_sq"] = record[2]
                    row["G"]  = record[3]
                    row["gz"] = record[4]
                    row["k_in_0"] = record[5]
                    row["k_in_1"] = record[6]
                    row["k_in_2"] = record[7]
                    row["RHS"]     = record[8]
                    row["Reason"]  = record[9]

                elif "infinite_solutions" in event:
                    # record: (..., R2, denom_sq, G, gz, k_in0, k_in1, k_in2, RHS, reason)
                    row["R2"] = record[1]
                    row["denom_sq"] = record[2]
                    row["G"]  = record[3]
                    row["gz"] = record[4]
                    row["k_in_0"] = record[5]
                    row["k_in_1"] = record[6]
                    row["k_in_2"] = record[7]
                    row["RHS"]     = record[8]
                    row["Reason"]  = record[9]

                elif "disc<-1e-14" in event:
                    # record: (..., disc, R2, denom_sq, G, gz, k_in[0], k_in[1], k_in[2], RHS, reason)
                    row["disc"]     = record[1]
                    row["R2"]       = record[2]
                    row["denom_sq"] = record[3]
                    row["G"]        = record[4]
                    row["gz"]       = record[5]
                    row["k_in_0"]   = record[6]
                    row["k_in_1"]   = record[7]
                    row["k_in_2"]   = record[8]
                    row["RHS"]      = record[9]
                    row["Reason"]   = record[10]

                elif "tangent->disc=0.0" in event:
                    # record: (..., disc, R2, denom_sq, G, gz, k_in0, k_in1, k_in2, RHS, reason)
                    row["disc"]     = record[1]
                    row["R2"]       = record[2]
                    row["denom_sq"] = record[3]
                    row["G"]        = record[4]
                    row["gz"]       = record[5]
                    row["k_in_0"]   = record[6]
                    row["k_in_1"]   = record[7]
                    row["k_in_2"]   = record[8]
                    row["RHS"]      = record[9]
                    row["Reason"]   = record[10]

                else:
                    # For an unknown variant, store remaining info in Extra
                    row["Extra"] = str(record[1:])

            else:
                # Dump all unknown fields into Extra
                row["Extra"] = str(record[1:])

            writer.writerow(row)

    print("Debug log saved to:", filename)
