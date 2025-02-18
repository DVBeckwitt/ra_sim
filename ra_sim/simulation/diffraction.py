import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from numba import njit, prange, int64, float64
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy import cos, sin, sqrt
from ra_sim.StructureFactor.StructureFactor import attenuation
from ra_sim.utils.calculations import fresnel_transmission, IndexofRefraction

@njit
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

@njit
def compute_d(gz, gr):
    """
    Example function to compute d-spacing from gz, gr.
    Not essential to this demonstration.
    """
    d = (gz**2 + gr**2)**(-0.5) * (2 * np.pi)
    return d

@njit
def solve_q(k_in, k, G, gz):
    """
    Parameterizes the intersection of:
       1) Q_x^2 + Q_y^2 + gz^2 = G^2
       2) (k_in + Q)^2 = k^2
    as a function of z = gz.
    
    Returns exactly two solutions for (Q_x, Q_y) in a list.
    In a tangent case, the second tuple is (np.nan, np.nan).
    If no solution exists, both tuples are (np.nan, np.nan).
    """
    # 1) Determine the circle radius in the z = gz plane.
    R2 = G*G - gz*gz
    if R2 < 0:
        return [(np.nan, np.nan), (np.nan, np.nan)]
    R = np.sqrt(R2)
    
    # 2) Compute the dot product constant.
    kx, ky, kz = k_in
    kin_sq = kx*kx + ky*ky + kz*kz
    c = 0.5 * (k*k - (kin_sq + G*G))
    cprime = c - kz*gz

    tol = 1e-10

    # 3) Handle cases where one of the in-plane components is nearly zero.
    #    This avoids instabilities in the polar parameterization.
    if abs(kx) < tol and abs(ky) >= tol:
        # Here, the plane equation is: ky * Q_y = cprime  =>  Q_y = cprime/ky.
        Qy_sol = cprime / ky
        disc = R2 - Qy_sol*Qy_sol
        if disc < 0:
            return [(np.nan, np.nan), (np.nan, np.nan)]
        sqrt_disc = np.sqrt(disc)
        return [( sqrt_disc, Qy_sol), (-sqrt_disc, Qy_sol)]
    elif abs(ky) < tol and abs(kx) >= tol:
        # Here, the plane equation is: kx * Q_x = cprime  =>  Q_x = cprime/kx.
        Qx_sol = cprime / kx
        disc = R2 - Qx_sol*Qx_sol
        if disc < 0:
            return [(np.nan, np.nan), (np.nan, np.nan)]
        sqrt_disc = np.sqrt(disc)
        return [(Qx_sol,  sqrt_disc), (Qx_sol, -sqrt_disc)]
    elif abs(kx) < tol and abs(ky) < tol:
        # k_in is nearly vertical.
        if abs(cprime) < tol:
            return [(R, 0.0), (-R, 0.0)]
        else:
            return [(np.nan, np.nan), (np.nan, np.nan)]
    
    # 4) For the general case, use the polar parameterization.
    k_in_xy = np.sqrt(kx*kx + ky*ky)
    alpha = np.arctan2(ky, kx)
    denom = R * k_in_xy
    ratio = cprime / denom

    if abs(ratio) > 1.0:
        return [(np.nan, np.nan), (np.nan, np.nan)]
    elif abs(ratio - 1.0) < tol or abs(ratio + 1.0) < tol:
        theta = alpha if ratio > 0 else alpha + np.pi
        Qx = R * np.cos(theta)
        Qy = R * np.sin(theta)
        return [(Qx, Qy), (np.nan, np.nan)]
    else:
        acos_val = np.arccos(ratio)
        theta1 = alpha + acos_val
        theta2 = alpha - acos_val
        Qx1 = R * np.cos(theta1)
        Qy1 = R * np.sin(theta1)
        Qx2 = R * np.cos(theta2)
        Qy2 = R * np.sin(theta2)
        return [(Qx1, Qy1), (Qx2, Qy2)]




@njit
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

@njit
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
    save_flag,
    q_data,
    q_count,
    i_peaks_index
):
    gz0, gr0 = G(H, K, L, av, cv, 0.0, 0.0)
    theta_range = np.arange(theta_initial, theta_final, step)

    max_intensity_sign0 = -1.0
    max_x_sign0 = np.nan
    max_y_sign0 = np.nan

    max_intensity_sign1 = -1.0
    max_x_sign1 = np.nan
    max_y_sign1 = np.nan
    n2 = 1 
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

        for idx in prange(len(beam_x_array)):
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

            k_scat = k * sqrt(np.real(n2**2))

            k_re = k**2 * np.real(n2**2)
            k_im = k**2 * np.imag(n2**2)

            k_x_scatter = k_scat * cos(th_t) * sin(phi_i_prime)
            k_y_scatter = k_scat * cos(th_t) * cos(phi_i_prime)
            at = k_re * sin(th_t)**2 
            b  = k_im
            g = np.sqrt(gr**2 + gz**2)
            #k_z_scatter = np.sign(th_t) * sqrt((sqrt(at**2 + b**2) + at)/2)
            imaginary_k_z_scatter = sqrt((sqrt(at**2 + b**2) - at)/2)
            k_z_scatter = k_scat * sin(th_t) 
            
            k_in = np.array([k_x_scatter, k_y_scatter, k_z_scatter])
            
            All_Q = solve_q(k_in, k_scat, g, gz)

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
                    * np.exp(-gz**2 * debye_x**2)
                    * np.exp(-gr**2 * debye_y**2)
                )

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

                if save_flag == 1:
                    current_count = q_count[i_peaks_index]
                    if current_count < q_data.shape[1]:
                        q_data[i_peaks_index, current_count, 0] = Qx
                        q_data[i_peaks_index, current_count, 1] = Qy
                        q_data[i_peaks_index, current_count, 2] = (gz + k_z_scatter)
                        q_data[i_peaks_index, current_count, 3] = gz
                        q_data[i_peaks_index, current_count, 4] = Mv
                        q_count[i_peaks_index] += 1

    return (max_x_sign0, max_y_sign0, max_intensity_sign0,
            max_x_sign1, max_y_sign1, max_intensity_sign1)

@njit(parallel=True, fastmath=True)
def process_peaks_parallel(
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
    """
    Loop over all reflections in miller and compute intensities onto 'image'.
    If save_flag==1, store all valid Q data (with intensity) in the returned arrays.

    Returns:
      image, max_positions, q_data, q_count
    If save_flag==0, then q_data and q_count are None.
    """
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

    for i in prange(num_peaks):
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
            i
        )

        max_positions[i, 0] = mx0
        max_positions[i, 1] = my0
        max_positions[i, 2] = mv0
        max_positions[i, 3] = mx1
        max_positions[i, 4] = my1
        max_positions[i, 5] = mv1

    if save_flag == 1:
        return image, max_positions, q_data, q_count
    else:
        return image, max_positions, None, None
