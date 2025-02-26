#!/usr/bin/env python3
import os
import csv
import datetime
import matplotlib
matplotlib.use('TkAgg')  # Use a backend suitable for script execution
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy import cos, sin, sqrt

# If your local code references these, keep them; otherwise remove.
from ra_sim.StructureFactor.StructureFactor import attenuation
from ra_sim.utils.calculations import fresnel_transmission, IndexofRefraction

##############################################################################
# GLOBAL DEBUG LOG
# We'll store tuples of debug data and write them to a file after the run.
##############################################################################
DEBUG_LOG = []
DEBUG_ENABLED = True  # Set to False if you want to skip logging entirely.

def G(H, K, L, av, cv, beta, kappa):
    """
    Compute the components of the reciprocal lattice vector G.
    Gz and Gr are combined to get final gz, gr after rotation by (kappa - beta).
    Logs the output if DEBUG_ENABLED.
    """
    global DEBUG_LOG, DEBUG_ENABLED

    # Base magnitudes before rotation
    Gz = 2 * np.pi * L / cv
    Gr = 4 * np.pi / av * sqrt((H**2 + H*K + K**2) / 3.0)

    # Rotation by (kappa - beta)
    gr = Gr * cos(kappa - beta) - Gz * sin(kappa - beta)
    gz = Gr * sin(kappa - beta) + Gz * cos(kappa - beta)

    if DEBUG_ENABLED:
        DEBUG_LOG.append((
            "G-calculation",
            float(H), float(K), float(L),
            float(av), float(cv),
            float(beta), float(kappa),
            float(gz), float(gr)
        ))

    return gz, gr


def compute_d(gz, gr):
    """
    Compute d-spacing from gz, gr. (Utility function, not crucial to debugging).
    """
    return (gz**2 + gr**2)**(-0.5) * (2 * np.pi)


#!/usr/bin/env python3
import os
import csv
import numpy as np

# Global debug log (if needed)
DEBUG_LOG = []
DEBUG_ENABLED = True

def safe_float(s):
    try:
        return float(s)
    except:
        return np.nan

def G(H, K, L, av, cv, beta, kappa):
    """
    Compute the components of the reciprocal lattice vector G.
    After rotation by (kappa - beta), returns (gz, gr).
    """
    Gz = 2 * np.pi * L / cv
    Gr = 4 * np.pi / av * np.sqrt((H**2 + H*K + K**2) / 3.0)
    gr = Gr * np.cos(kappa - beta) - Gz * np.sin(kappa - beta)
    gz = Gr * np.sin(kappa - beta) + Gz * np.cos(kappa - beta)
    if DEBUG_ENABLED:
        DEBUG_LOG.append((
            "G-calculation",
            float(H), float(K), float(L),
            float(av), float(cv),
            float(beta), float(kappa),
            float(gz), float(gr)
        ))
    return gz, gr

#!/usr/bin/env python3
import os
import csv
import numpy as np

# Global debug log and flag (if you want to log more details)
DEBUG_LOG = []
DEBUG_ENABLED = True

#!/usr/bin/env python3
import os
import csv
import numpy as np

# Global debug log and flag (if you want to log more details)
DEBUG_LOG = []
DEBUG_ENABLED = True

#!/usr/bin/env python3
import os
import csv
import numpy as np

# Global debug log and flag (if needed)
DEBUG_LOG = []
DEBUG_ENABLED = True

def solve_q(k_in, k, G, g_z, H, K, L, eps=1e-14, disc_tol=1e-8):
    """
    Find the two solutions (Qx, Qy) of the constrained system:
    
      (i)  Qx^2 + Qy^2 = R^2,  where R^2 = G^2 - g_z^2,
      (ii) kx*Qx + ky*Qy = LHS_line, where 
           LHS_line = 0.5*(k^2 - (|k_in|^2 + G^2)) - kz_in*g_z.
    
    We solve by projecting onto the direction of (kx,ky). This is numerically more robust
    near Qx=0. When the discriminant is very small (or slightly negative due to rounding),
    we clamp it to zero.
    
    Returns an array of shape (2,2) with the two solutions.
    """
    global DEBUG_LOG, DEBUG_ENABLED

    # Compute R^2 = G^2 - g_z^2.
    R2 = G*G - g_z*g_z
    if R2 < 0.0:
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "solve_q-return:R2<0",
                float(H), float(K), float(L),
                float(R2), float(G), float(g_z),
                "No real in-plane circle => [NaN,NaN]"
            ))
        return np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
    R = np.sqrt(R2)

    kx, ky, kz_in = k_in
    kin_sq = kx*kx + ky*ky + kz_in*kz_in

    RHS_half = 0.5 * (k*k - (kin_sq + G*G))
    LHS_line = RHS_half - kz_in * g_z

    denom_sq = kx*kx + ky*ky
    if denom_sq < eps*eps:
        # near-normal incidence
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "solve_q-return:near_normal",
                float(H), float(K), float(L),
                float(kx), float(ky), float(kz_in),
                float(k), float(G), float(g_z),
                "kx^2+ky^2 < eps^2 => no real Q"
            ))
        return np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)

    # Project onto the (kx, ky) direction.
    d = np.sqrt(denom_sq)
    u = np.array([kx, ky]) / d       # unit vector along (kx,ky)
    a = LHS_line / d                # scalar projection along u

    # Now, we must have Q^2 = R^2, and the projection of Q onto u is a.
    # The remaining (perpendicular) magnitude is b = sqrt(R^2 - a^2).
    delta = R2 - a*a
    if abs(delta) < disc_tol:
        # If delta is within tolerance of zero, we treat it as exactly tangent.
        b = 0.0
    elif delta < 0:
        # If delta is negative by more than the tolerance, then no real solutions.
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "solve_q-return:NoRealSolutions",
                float(H), float(K), float(L),
                float(a*a), float(R2),
                "a^2 > R2 by more than tolerance => no real solutions"
            ))
        return np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64)
    else:
        b = np.sqrt(delta)

    # Find a perpendicular unit vector v (rotate u by 90°)
    v = np.array([-u[1], u[0]])

    sol1 = a * u + b * v
    sol2 = a * u - b * v

    if DEBUG_ENABLED:
        DEBUG_LOG.append((
            "solve_q:projection_solutions",
            float(H), float(K), float(L),
            float(kx), float(ky), float(kz_in),
            float(sol1[0]), float(sol1[1]),
            float(sol2[0]), float(sol2[1]),
            "Projection method: solutions via u and v with clamped delta"
        ))
    return np.array([sol1, sol2], dtype=np.float64)


# Example main function for testing:
if __name__ == "__main__":
    # Test with sample values (modify as needed)
    k_in = np.array([0.001, 0.002, 0.003])
    k = 1.0
    G = 0.5
    g_z = 0.1
    H, K, L = 0.0, 0.0, 3.0
    solutions = solve_q(k_in, k, G, g_z, H, K, L)
    print("Solutions (Qx, Qy):")
    print(solutions)


# For testing, you might want to add a main() to call solve_q with sample values.
if __name__ == "__main__":
    # Sample test: use arbitrary k_in, k, G, g_z, H, K, L values.
    k_in = np.array([0.001, 0.002, 0.003])
    k = 1.0
    G = 0.5
    g_z = 0.1
    H, K, L = 0.0, 0.0, 3.0
    sols = solve_q(k_in, k, G, g_z, H, K, L)
    print("Solutions (Qx, Qy):")
    print(sols)


# For testing, you might want to add a main() to call solve_q with sample values.
if __name__ == "__main__":
    # Sample test: use arbitrary k_in, k, G, g_z, H, K, L values.
    k_in = np.array([0.001, 0.002, 0.003])
    k = 1.0
    G = 0.5
    g_z = 0.1
    H, K, L = 0.0, 0.0, 3.0
    sols = solve_q(k_in, k, G, g_z, H, K, L)
    print("Solutions (Qx, Qy):")
    print(sols)

# (Other functions remain unchanged. For brevity, I'm not repeating intersect_line_plane,
# calculate_phi, process_peaks_parallel_debug, etc. They can be kept as before.)
# ...
if __name__ == "__main__":
    # This module is intended to be imported into your simulation.
    # When run directly, you might run tests or print a message.
    print("Updated solve_q function with projection method implemented.")



def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Compute the intersection of a line and plane:
      P0 + t * k_vec = Intersection_Point
    """
    global DEBUG_LOG, DEBUG_ENABLED
    denominator = np.dot(k_vec, n_plane)
    if abs(denominator) < 1e-14:
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "Ray Missed it",
                "Line-plane denominator=0 => no intersection",
                f"k_vec={k_vec}, n_plane={n_plane}"
            ))
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
    wavelength_array,  # per-sample bandpass
    debye_x, debye_y, center,
    theta_initial, theta_final, step, reflection_intensity,
    k_unused,
    R_x_detector, R_z_detector, n_det_rot, Detector_Pos, e1_det, e2_det,
    R_z_R_y, R_ZY_n, P0, unit_x,
    save_flag,
    q_data,     # shape: (num_peaks, max_solutions, 5) [Qx, Qy, Qz, gz, Mv]
    q_count,    # shape: (num_peaks,)
    i_peaks_index,
    DEBUG_ENABLED=False
):
    """
    Calculate & fill diffraction for a single (H,K,L).
    Logs geometry steps for debugging. Also logs disc<0 cases in solve_q.
    """
    global DEBUG_LOG

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

        # Surface normal in this orientation
        n = R_x @ R_ZY_n
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-14:
            if DEBUG_ENABLED:
                DEBUG_LOG.append((
                    "Ray Missed it",
                    f"Surface normal degenerate (theta_i={theta_i})",
                    f"H={H},K={K},L={L}"
                ))
            continue
        n /= n_norm

        P0_rot = R_x @ P0
        u_ref = np.array([0.0, 0.0, -1.0])
        e1 = np.cross(n, u_ref)
        e1_norm = np.linalg.norm(e1)
        if e1_norm < 1e-12:
            if DEBUG_ENABLED:
                DEBUG_LOG.append((
                    "Ray Missed it",
                    f"Cross(n, ref) ~0 => skip reflection (theta_i={theta_i})",
                    f"H={H},K={K},L={L}"
                ))
            continue
        e1 /= e1_norm
        e2 = np.cross(n, e1)
        e2 /= np.linalg.norm(e2)

        # Loop over beam spots
        for idx in range(len(beam_x_array)):
            x = beam_x_array[idx]
            y = beam_y_array[idx]
            beam_intensity = beam_intensity_array[idx]
            beta  = beta_array[idx]
            kappa = kappa_array[idx]
            mosaic_int = mosaic_intensity_array[idx]
            delta_theta_i_spot = theta_array[idx]
            delta_phi_i_spot   = phi_array[idx]
            div_int = divergence_intensity_array[idx]

            lam_sample = wavelength_array[idx]   # in Å
            k_0 = 2.0 * np.pi / lam_sample       # in reciprocal Å

            gz, gr = G(H, K, L, av, cv, beta, kappa)
            beam_start = np.array([x, -20e-3, -zb + y])

            # Incident direction
            k_xi = cos(delta_theta_i_spot) * sin(delta_phi_i_spot)
            k_yi = cos(delta_theta_i_spot) * cos(delta_phi_i_spot)
            k_zi = sin(delta_theta_i_spot)
            k_vec = np.array([k_xi, k_yi, k_zi])

            norm_kvec = np.linalg.norm(k_vec)
            if norm_kvec < 1e-14:
                if DEBUG_ENABLED:
                    DEBUG_LOG.append((
                        "Ray Missed it",
                        "k_vec norm < 1e-14 => invalid direction",
                        f"H={H},K={K},L={L}, X={x}, Y={y}"
                    ))
                continue
            k_vec /= norm_kvec

            kn_dot = np.dot(k_vec, n)
            Intersection_Point_Plane, t = intersect_line_plane(beam_start, k_vec, P0_rot, n)
            if Intersection_Point_Plane is None:
                continue
            if t < 0:
                if DEBUG_ENABLED:
                    DEBUG_LOG.append((
                        "Ray Missed it",
                        f"Intersection behind plane (t={t})",
                        f"H={H},K={K},L={L}, X={x}, Y={y}"
                    ))
                continue

            th_i_prime = (np.pi / 2) - np.arccos(kn_dot)
            projected_incident = k_vec - kn_dot * n
            proj_norm = np.linalg.norm(projected_incident)
            if proj_norm <= 1e-14:
                if DEBUG_ENABLED:
                    DEBUG_LOG.append((
                        "Ray Missed it",
                        "Projected incident ~ zero vector",
                        f"H={H},K={K},L={L}, index={idx}"
                    ))
                continue
            projected_incident /= proj_norm

            p1 = np.dot(projected_incident, e1)
            p2 = np.dot(projected_incident, e2)
            phi_i_prime = (np.pi / 2) - np.arctan2(p2, p1)
            th_t = np.arccos(cos(th_i_prime) / np.real(n2)) * np.sign(th_i_prime)

            k_scat = k_0 * sqrt(np.real(n2**2))
            k_re = k_0**2 * np.real(n2**2)
            k_im = k_0**2 * np.imag(n2**2)

            k_x_scatter = k_scat * cos(th_t) * sin(phi_i_prime)
            k_y_scatter = k_scat * cos(th_t) * cos(phi_i_prime)
            at = k_re * sin(th_t)**2
            b  = k_im
            g  = np.sqrt(gr**2 + gz**2)

            # scatter z
            k_z_scatter = np.sign(th_t) * sqrt((np.sqrt(at**2 + b**2) + at)/2)
            k_in = np.array([k_x_scatter, k_y_scatter, k_z_scatter])

            # Solve for reciprocal-lattice intersection
            All_Q = solve_q(k_in, k_scat, g, gz, H, K, L)

            for sign_idx in range(2):
                Qx, Qy = All_Q[sign_idx]
                if np.isnan(Qx) or np.isnan(Qy):
                    if DEBUG_ENABLED:
                        DEBUG_LOG.append((
                            "Ray Missed it",
                            "No valid Q => skip",
                            f"H={H},K={K},L={L}, X={x}, Y={y}"
                        ))
                    continue

                k_tx_prime = Qx + k_x_scatter
                k_ty_prime = Qy + k_y_scatter
                k_tz_prime = gz + k_z_scatter

                # For reference
                k_tz_prime2 = gz0 + k_z_scatter
                kr = sqrt(k_tx_prime**2 + k_ty_prime**2)
                if abs(kr) < 1e-12:
                    twotheta_t = 0.0
                else:
                    twotheta_t = abs(np.arctan(k_tz_prime / kr))

                if abs(gr0) > 1e-12:
                    twotheta_t2 = np.arctan(k_tz_prime2 / gr0)
                else:
                    twotheta_t2 = 0.0

                phi_f = np.arctan2(k_tx_prime, k_ty_prime)
                k_tx_f = k_scat * cos(twotheta_t) * sin(phi_f)
                k_ty_f = k_scat * cos(twotheta_t) * cos(phi_f)
                af  = k_re * sin(twotheta_t)**2
                real_k_tz_f = sqrt((np.sqrt(af**2 + b**2) + af)/2) * np.sign(twotheta_t)
                kf = np.array([k_tx_f, k_ty_f, real_k_tz_f])
                kf_prime = R_sample @ kf

                k_r = sqrt(kf_prime[0]**2 + kf_prime[1]**2)
                if k_r < 1e-12:
                    tth = 0.0
                else:
                    tth = abs(np.arctan(kf_prime[2] / k_r))

                Tf = abs(fresnel_transmission(tth, n2))**2

                Intersection_Point_Detector, s_ = intersect_line_plane(
                    Intersection_Point_Plane, kf_prime, Detector_Pos, n_det_rot
                )
                if Intersection_Point_Detector is None:
                    continue
                if s_ < 0:
                    if DEBUG_ENABLED:
                        DEBUG_LOG.append((
                            "Ray Missed it",
                            f"Detector intersection behind plane (s={s_})",
                            f"H={H},K={K},L={L}, Qx={Qx}, Qy={Qy}"
                        ))
                    continue

                Plane_to_Detector = Intersection_Point_Detector - Detector_Pos
                x_det = np.dot(Plane_to_Detector, e1_det)
                y_det = np.dot(Plane_to_Detector, e2_det)

                # Final intensity
                Mv = (
                    reflection_intensity
                    * beam_intensity
                    * mosaic_int
                    * div_int
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
                        float(theta_i),
                        float(reflection_intensity),
                        float(beam_intensity),
                        float(mosaic_int),
                        float(div_int),
                        float(Mv),
                        float(x_det),
                        float(y_det),
                        float(Qx),
                        float(Qy),
                        float(gz),
                        float(gr),
                        float(kf_prime[0]),
                        float(kf_prime[1]),
                        float(kf_prime[2]),
                        float(x),
                        float(y),
                        float(t),
                        float(s_),
                        float(k_xi),
                        float(k_yi),
                        float(k_zi),
                        float(delta_theta_i_spot),
                        float(delta_phi_i_spot)
                    ))

                # Update the image buffer
                if not np.isnan(Mv):
                    row_center = center[0]
                    col_center = center[1]
                    px_size = 100e-6
                    pixel_row = int(round(row_center - y_det / px_size))
                    pixel_col = int(round(col_center + x_det / px_size))

                    if (0 <= pixel_row < image_size) and (0 <= pixel_col < image_size):
                        image[pixel_row, pixel_col] += Mv
                        if sign_idx == 0:
                            if image[pixel_row, pixel_col] > max_intensity_sign0:
                                max_intensity_sign0 = image[pixel_row, pixel_col]
                                max_x_sign0 = pixel_col
                                max_y_sign0 = pixel_row
                        else:
                            if image[pixel_row, pixel_col] > max_intensity_sign1:
                                max_intensity_sign1 = image[pixel_row, pixel_col]
                                max_x_sign1 = pixel_col
                                max_y_sign1 = pixel_row

                # Optionally store Q data
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
    wavelength_array,        # per-sample bandpass
    debye_x, debye_y, center, theta_initial, theta_final, step,
    unit_x, n_detector,
    save_flag
):
    """
    Debug version of parallel peak processing.
    Calls calculate_phi for each reflection and logs debug info
    (including rays that missed it).
    """
    global DEBUG_LOG, DEBUG_ENABLED
    DEBUG_ENABLED = True  # Turn on logging

    DEBUG_LOG.append(("process_peaks_parallel", float(len(miller))))

    gamma_rad = np.radians(gamma)
    Gamma_rad = np.radians(Gamma)
    chi_rad   = np.radians(chi)
    psi_rad   = np.radians(psi)

    k = 2 * np.pi / lambda_

    R_x_detector = np.array([
        [1.0,              0.0,               0.0],
        [0.0, cos(gamma_rad),  sin(gamma_rad)],
        [0.0, -sin(gamma_rad), cos(gamma_rad)]
    ])
    R_z_detector = np.array([
        [ cos(Gamma_rad), sin(Gamma_rad), 0.0],
        [-sin(Gamma_rad), cos(Gamma_rad), 0.0],
        [ 0.0,            0.0,            1.0]
    ])
    n_det_rot = R_z_detector @ (R_x_detector @ n_detector)
    norm_n_det = np.linalg.norm(n_det_rot)
    if norm_n_det < 1e-14:
        if DEBUG_ENABLED:
            DEBUG_LOG.append((
                "Ray Missed it",
                "Detector normal degenerate => skip entire simulation",
                f"num_miller={len(miller)}"
            ))
        return image, None, None, None
    n_det_rot /= norm_n_det

    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0])
    e1_det = unit_x - np.dot(unit_x, n_det_rot)*n_det_rot
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
    num_theta_steps = int((theta_final - theta_initial)//step)
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
        refl_intensity = intensities[i]

        (mx0, my0, mv0,
         mx1, my1, mv1) = calculate_phi(
            H, K, L, av, cv, lambda_, image, image_size,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs, zb, n2,
            beam_x_array, beam_y_array, beam_intensity_array,
            beta_array, kappa_array, mosaic_intensity_array,
            theta_array, phi_array, divergence_intensity_array,
            wavelength_array,
            debye_x, debye_y, center,
            theta_initial, theta_final, step,
            refl_intensity,
            k, R_x_detector, R_z_detector, n_det_rot, Detector_Pos,
            e1_det, e2_det, R_z_R_y, R_ZY_n, P0, unit_x,
            save_flag,
            q_data,
            q_count,
            i,
            DEBUG_ENABLED=True
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


def dump_debug_log():
    """
    Write all debug-log records to ~/Downloads/ewald_debug_log.csv in CSV format.
    Includes additional columns (Reflection_Intensity, Beam_Intensity, etc.).
    """
    global DEBUG_LOG

    filename = os.path.expanduser("~/Downloads/ewald_debug_log.csv")
    now_str = datetime.datetime.now().isoformat()

    # Updated header with new columns
    header = [
        "Timestamp",
        "EventType",
        "Reason",
        "H", "K", "L",
        "beta", "kappa",
        "theta_i",
        "Reflection_Intensity",
        "Beam_Intensity",
        "Mosaic_Intensity",
        "Divergence_Intensity",
        "Mv",
        "x_det", "y_det",
        "Qx", "Qy",
        "gz", "gr",
        "kfX", "kfY", "kfZ",
        "BeamStartX", "BeamStartY",
        "Intersection_t", "Detector_t",
        "k_xi", "k_yi", "k_zi",
        "Delta_Theta_i_spot",
        "Delta_Phi_i_spot",
        "Extra"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for record in DEBUG_LOG:
            row = dict.fromkeys(header, "")
            row["Timestamp"] = now_str
            row["EventType"] = record[0]

            event = record[0]

            if event == "process_peaks_parallel":
                # e.g. ("process_peaks_parallel", float(len(miller)))
                row["Extra"] = f"num_miller={record[1]}"

            elif event == "calc_phi-begin":
                # e.g. ("calc_phi-begin", H, K, L)
                row["H"] = record[1]
                row["K"] = record[2]
                row["L"] = record[3]

            elif event == "Ray Made it":
                # e.g. ("Ray Made it", H, K, L, beta, kappa, theta_i, reflection_intensity, ...)
                row["H"]                      = record[1]
                row["K"]                      = record[2]
                row["L"]                      = record[3]
                row["beta"]                   = record[4]
                row["kappa"]                  = record[5]
                row["theta_i"]                = record[6]
                row["Reflection_Intensity"]   = record[7]
                row["Beam_Intensity"]         = record[8]
                row["Mosaic_Intensity"]       = record[9]
                row["Divergence_Intensity"]   = record[10]
                row["Mv"]                     = record[11]
                row["x_det"]                  = record[12]
                row["y_det"]                  = record[13]
                row["Qx"]                     = record[14]
                row["Qy"]                     = record[15]
                row["gz"]                     = record[16]
                row["gr"]                     = record[17]
                row["kfX"]                    = record[18]
                row["kfY"]                    = record[19]
                row["kfZ"]                    = record[20]
                row["BeamStartX"]             = record[21]
                row["BeamStartY"]             = record[22]
                row["Intersection_t"]         = record[23]
                row["Detector_t"]             = record[24]
                row["k_xi"]                   = record[25]
                row["k_yi"]                   = record[26]
                row["k_zi"]                   = record[27]
                row["Delta_Theta_i_spot"]     = record[28]
                row["Delta_Phi_i_spot"]       = record[29]

            elif event == "Ray Missed it":
                # e.g. ("Ray Missed it", reason_str, maybe extra info)
                row["Reason"] = str(record[1])
                if len(record) > 2:
                    row["Extra"] = str(record[2])

            elif event.startswith("solve_q-return:") or event.startswith("solve_q:case"):
                # e.g. ("solve_q-return:disc<0_in_case3", H, K, L, disc, A, B, C, ...)
                # or   ("solve_q:case2_solutions", H, K, L, kx, ky, kz_in, Qx1, Qy1, Qx2, Qy2, ...)
                if len(record) >= 4:
                    row["H"] = record[1]
                    row["K"] = record[2]
                    row["L"] = record[3]
                # Everything else jammed into "Extra" for reference
                row["Extra"] = str(record[4:])

            elif event == "G-calculation":
                # e.g. ("G-calculation", H, K, L, av, cv, beta, kappa, gz, gr)
                row["H"]     = record[1]
                row["K"]     = record[2]
                row["L"]     = record[3]
                av           = record[4]
                cv           = record[5]
                beta_        = record[6]
                kappa_       = record[7]
                gz_          = record[8]
                gr_          = record[9]
                row["Extra"] = f"av={av}, cv={cv}, beta={beta_}, kappa={kappa_}, gz={gz_}, gr={gr_}"

            else:
                # geometry-skip, or unrecognized
                if len(record) > 1:
                    row["Reason"] = str(record[1])
                if len(record) > 2:
                    row["Extra"] = str(record[2:])

            writer.writerow(row)

    print(f"Debug log saved to: {filename}")
    DEBUG_LOG.clear()


if __name__ == "__main__":
    print("Corrected debug script now logs disc<0 in case2/case3 with H,K,L etc.")
