
# Define your functions with @njit (assuming numba is used)
import matplotlib
matplotlib.use('TkAgg')  # Use a backend that is suitable for script execution
import numpy as np
from numba import njit, prange, int64, float64
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy import sqrt, cos, sin, pi, isnan, arctan2, radians, arccos, arctan2, cross, dot
# Define your functions with @njit
@njit
def G(H, K, L, av, cv, beta, kappa):
    Gz = 2 * np.pi * L / cv
    Gr = 4 * np.pi / av * sqrt(
        (H ** 2 + H * K + K ** 2) / 3.0
    )
    gr = Gr * cos(kappa - beta) - Gz * sin(kappa - beta)
    gz = Gr * sin(kappa - beta) + Gz * cos(kappa - beta)
    return gz, gr

@njit
def compute_d(gz, gr):
    d = (gz ** 2 + gr ** 2) ** (-0.5) * (2 * np.pi)
    return d

@njit
def mapping(image, image_size, is_real, Mv, center, x_det, y_det):
    if is_real and not np.isnan(Mv):
        if not np.isnan(x_det) and not np.isnan(y_det):
            pixel_x = int(round(x_det / (100e-6) + center[1]))
            pixel_y = 3000-int(round(y_det / (100e-6) + center[0]))
            if 0 <= pixel_x < image_size and 0 <= pixel_y < image_size:
                image[pixel_y, pixel_x] += Mv  # Use Mv directly

@njit
def solve_q(k_x, k_y, k_z, gz, k, gr):
    # Always return a 2x2 NumPy array: [[Qx1, Qy1], [Qx2, Qy2]]
    # If no solution is found, fill with np.nan.

    top_sphere = k - k_z
    if gz > top_sphere:
        # No solution
        return np.array([[np.nan, np.nan],
                         [np.nan, np.nan]], dtype=np.float64)

    val = k**2 - (gz + k_z)**2
    if val < 0:
        # No solution, return nan
        return np.array([[np.nan, np.nan],
                         [np.nan, np.nan]], dtype=np.float64)

    R1 = gr
    R2 = np.sqrt(val)
    Cx = -k_x
    Cy = -k_y
    d = np.sqrt(Cx**2 + Cy**2)

    # Avoid division by zero
    if d == 0:
        return np.array([[np.nan, np.nan],
                         [np.nan, np.nan]], dtype=np.float64)

    a = (R1**2 - R2**2 + d**2) / (2*d)
    Px = (a/d)*Cx
    Py = (a/d)*Cy

    h_sq = R1**2 - a**2
    min_val = -1e-3
    if h_sq < 0 and h_sq > min_val:
        h = 0.0
    elif h_sq < min_val:
        # No solution
        return np.array([[np.nan, np.nan],
                         [np.nan, np.nan]], dtype=np.float64)
    else:
        h = np.sqrt(h_sq)

    rx = (-Cy)*(h/d)
    ry = ( Cx)*(h/d)

    Q1 = (Px + rx, Py + ry)
    Q2 = (Px - rx, Py - ry)

    return np.array([[Q1[0], Q1[1]],
                     [Q2[0], Q2[1]]], dtype=np.float64)


@njit
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Computes the intersection of a line and a plane.

    Parameters:
    P0 : ndarray
        Starting point of the line (numpy array of shape (3,))
    k_vec : ndarray
        Direction vector of the line (numpy array of shape (3,))
    P_plane : ndarray
        A point on the plane (numpy array of shape (3,))
    n_plane : ndarray
        Normal vector of the plane (numpy array of shape (3,))

    Returns:
    Intersection_Point : ndarray
        The intersection point (numpy array of shape (3,))
    t : float
        The parameter along the line at which the intersection occurs

    Note:
    Returns (None, None) if the line is parallel to the plane.
    """
    denominator = np.dot(k_vec, n_plane)
    if denominator == 0:
        # Line is parallel to the plane
        return None, None
    t = np.dot((P_plane - P0), n_plane) / denominator
    Intersection_Point = P0 + t * k_vec
    return Intersection_Point, t

from numba import njit, prange
import numpy as np

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
    R_z_R_y, R_ZY_n, P0, unit_x  # Include unit_x here
):
    
    u_ref = np.array([0.0, 0.0, -1.0])

    theta_range = np.arange(theta_initial, theta_final, step)
    
    for theta_i in theta_range:
        
        rad_theta_i = np.radians(theta_i)
        

        R_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rad_theta_i), -np.sin(rad_theta_i)],
            [0.0, np.sin(rad_theta_i), np.cos(rad_theta_i)]
        ])
        R_sample = R_x @ R_z_R_y
        n = R_x @ R_ZY_n
        n /= np.linalg.norm(n)
        P0_rot = R_x @ P0
        P0_rot[0] = 0.0

        e1 = np.cross(n, u_ref)
        e1_norm = np.linalg.norm(e1)
        if e1_norm != 0.0:
            e1 /= e1_norm
        else:
            continue  # Skip if e1 cannot be computed

        e2 = np.cross(n, e1)

        # Loop over beam permutations
        for idx in prange(len(beam_x_array)):
            # Extract beam profile variables
            x = beam_x_array[idx]
            y = beam_y_array[idx]
            beam_intensity = beam_intensity_array[idx]

            # Extract mosaicity profile variables
            beta = beta_array[idx]
            kappa = kappa_array[idx]
            mosaic_intensity = mosaic_intensity_array[idx]

            # Extract divergence profile variables
            delta_theta_i_spot = theta_array[idx]
            delta_phi_i_spot = phi_array[idx]
            divergence_intensity = divergence_intensity_array[idx]

            # Compute G vector components
            gz, gr = G(H, K, L, av, cv, beta, kappa)

            # Beam start position
            beam_start = np.array([x, -20e-3, -zb + y])

            # Compute incident beam vector components
            k_xi = np.cos(delta_theta_i_spot) * np.sin(delta_phi_i_spot)
            k_yi = np.cos(delta_theta_i_spot) * np.cos(delta_phi_i_spot)
            k_zi = np.sin(delta_theta_i_spot)

            # Incident beam vector
            k_vec = np.array([k_xi, k_yi, k_zi])
            k_vec /= np.linalg.norm(k_vec)

            # Compute dot product with normal vector
            kn_dot = np.dot(k_vec, n)

            # Intersection with the sample plane
            Intersection_Point_Plane, t = intersect_line_plane(beam_start, k_vec, P0_rot, n)
            if Intersection_Point_Plane is None:
                continue  # Beam does not intersect the sample plane

            # Compute incident angles
            th_i_prime = (np.pi / 2) - np.arccos(kn_dot)
            projected_incident_beam = k_vec - kn_dot * n
            projected_incident_beam_norm = np.linalg.norm(projected_incident_beam)

            if projected_incident_beam_norm != 0.0:
                projected_incident_beam /= projected_incident_beam_norm
            else:
                projected_incident_beam = np.array([0.0, 0.0, 0.0])

            # Compute phi_i_prime
            p1 = np.dot(projected_incident_beam, e1)
            p2 = np.dot(projected_incident_beam, e2)
            phi_i_prime = (np.pi / 2) - np.arctan2(p2, p1)

            # Compute scattered wavevector components
            k_x = k * np.cos(th_i_prime) * np.sin(phi_i_prime)
            k_y = k * np.cos(th_i_prime) * np.cos(phi_i_prime)
            k_z = k * np.sin(th_i_prime)

            if k_x == 0.0:
                k_x += 1e-10  # Avoid division by zero

            # Solve for Q vectors
            All_Q = solve_q(k_x, k_y, k_z, gz, k, gr)

            # Loop over possible solutions
            for sign in range(2):
                Qx, Qy = All_Q[sign]
                if np.isnan(All_Q[0,0]):
                    # No solution found, skip
                    continue
                if np.iscomplex(Qx) or np.iscomplex(Qy):
                    continue  # Skip complex solutions

                # Compute transmitted wavevector components
                k_tx_prime = Qx + k_x
                k_ty_prime = Qy + k_y
                k_tz_prime = gz + k_z

                # Compute scattering angle
                k_r = np.sqrt(k_tx_prime**2 + k_ty_prime**2)
                tth = np.arctan2(k_tz_prime, k_r)
                if tth < 0.0:
                    continue  # Negative scattering angle

                # Compute final wavevector
                kf = np.array([k_tx_prime, k_ty_prime, k_tz_prime])
                kf /= np.linalg.norm(kf)

                # Rotate final wavevector to sample coordinates
                kf_prime = R_sample @ kf
                kf_prime /= np.linalg.norm(kf_prime)

                # Intersection with the detector plane
                Intersection_Point_Detector, s = intersect_line_plane(
                    Intersection_Point_Plane, kf_prime, Detector_Pos, n_det_rot
                )
                if Intersection_Point_Detector is None:
                    continue  # Beam does not intersect the detector plane

                # Compute detector coordinates
                Plane_to_Detector = Intersection_Point_Detector - Detector_Pos
                x_det = np.dot(Plane_to_Detector, e1_det)
                y_det = np.dot(Plane_to_Detector, e2_det) + zb

                # Check if the scattered beam is in the forward direction
                is_real = kf_prime[1] >= 0.0

                # Compute intensity value
                Mv = (
                    intensity
                    * beam_intensity
                    * mosaic_intensity
                    * divergence_intensity
                    * np.exp(-gz**2 * debye_x**2)
                    * np.exp(-gr**2 * debye_y**2)
                )

                # Map the intensity to the image
                mapping(image, image_size, is_real, Mv, center, x_det, y_det)


@njit(parallel=True, fastmath=True)
def process_peaks_parallel(
    miller, intensities, image_size, av, cv, lambda_, image,
    Distance_CoR_to_Detector, gamma, Gamma, chi, psi, zs, zb, n2,
    beam_x_array, beam_y_array, beam_intensity_array,
    beta_array, kappa_array, mosaic_intensity_array,
    theta_array, phi_array, divergence_intensity_array,
    debye_x, debye_y, center, theta_initial, theta_final, step,
    unit_x, n_detector
):
    gamma_rad = np.radians(gamma)
    Gamma_rad = np.radians(Gamma)
    chi_rad = np.radians(chi)
    psi_rad = np.radians(psi)

    # Compute wave number
    k = 2 * np.pi / lambda_

    # Precompute detector geometry
    R_x_detector = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(gamma_rad), np.sin(gamma_rad)],
        [0.0, -np.sin(gamma_rad), np.cos(gamma_rad)]
    ])
    R_z_detector = np.array([
        [np.cos(Gamma_rad), np.sin(Gamma_rad), 0.0],
        [-np.sin(Gamma_rad), np.cos(Gamma_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    n_det_rot = R_z_detector @ (R_x_detector @ n_detector)
    n_det_rot /= np.linalg.norm(n_det_rot)
    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0])
    e1_det = unit_x - np.dot(unit_x, n_det_rot) * n_det_rot
    e1_det /= np.linalg.norm(e1_det)
    e2_det = -np.cross(n_det_rot, e1_det)
    e2_det /= np.linalg.norm(e2_det)

    # Precompute sample rotation matrices
    R_y = np.array([
        [np.cos(chi_rad), 0.0, np.sin(chi_rad)],
        [0.0, 1.0, 0.0],
        [-np.sin(chi_rad), 0.0, np.cos(chi_rad)]
    ])
    R_z = np.array([
        [np.cos(psi_rad), np.sin(psi_rad), 0.0],
        [-np.sin(psi_rad), np.cos(psi_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    R_z_R_y = R_z @ R_y
    n1 = np.array([0.0, 0.0, 1.0])
    R_ZY_n = R_z_R_y @ n1
    R_ZY_n /= np.linalg.norm(R_ZY_n)

    # Precompute sample position
    P0 = np.array([0.0, 0.0, -zs])

    # Loop over all Miller indices
    for i in prange(len(miller)):
        H, K, L = miller[i]
        intensity = intensities[i]
        calculate_phi(
            H, K, L, av, cv, lambda_, image, image_size,
            gamma_rad, Gamma_rad, chi_rad, psi_rad,
            zs, zb, n2,
            beam_x_array, beam_y_array, beam_intensity_array,
            beta_array, kappa_array, mosaic_intensity_array,
            theta_array, phi_array, divergence_intensity_array,
            debye_x, debye_y, center,
            theta_initial, theta_final, step, intensity,
            k, R_x_detector, R_z_detector, n_det_rot, Detector_Pos, e1_det, e2_det,
            R_z_R_y, R_ZY_n, P0, unit_x
        )
    return image  # Return the full image


def simulate_diffraction_pattern(miller, intensities, parameters, beam_arrays, mosaic_arrays, divergence_arrays,
                                av, cv, lambda_, center, debye_x, debye_y, 
                                theta_initial, theta_range, step, geometry_params):
    """
    A convenience function that calls process_peaks_parallel with all required parameters.
    """
    (Distance_CoR_to_Detector, gamma, Gamma, chi, psi, zs, zb, n2) = geometry_params
    (beam_x_array, beam_y_array, beam_intensity_array) = beam_arrays
    (beta_array, kappa_array, mosaic_intensity_array) = mosaic_arrays
    (theta_array, phi_array, divergence_intensity_array) = divergence_arrays
    
    image_size = 3000
    image = np.zeros((image_size, image_size))
    unit_x = np.array([1.0, 0.0, 0.0])
    n_detector = np.array([0.0, 1.0, 0.0])
    
    image = process_peaks_parallel(
        miller, intensities, image_size, av, cv, lambda_, image,
        Distance_CoR_to_Detector, gamma, Gamma, chi, psi, zs, zb, n2,
        beam_x_array, beam_y_array, beam_intensity_array,
        beta_array, kappa_array, mosaic_intensity_array,
        theta_array, phi_array, divergence_intensity_array,
        debye_x, debye_y, center, theta_initial, theta_initial+theta_range, step,
        unit_x, n_detector
    )
    return image