"""Core diffraction routines used by the simulator."""

import numpy as np
from numba import njit, prange
from math import sin, cos, sqrt, pi, exp, acos
from ra_sim.utils.calculations import fresnel_transmission
from numba import types, float64
from numba.typed import List       #  only List lives here
from numba import types            #  <--  this is the right module
# =============================================================================
# 1) FINITE-STACK INTERFERENCE FOR N LAYERS
# =============================================================================

@njit
def attenuation(N, Qz, c):
    """
    Compute the coherent interference (Kiessig-like) from a stack of N layers,
    each of thickness 'c'. Qz is a complex wavevector transfer along z:
        Qz = Qz.real + i * Qz.imag

    Physically:
      - Qz.imag encodes absorption or evanescent decay,
      - Qz.real encodes the wave's momentum transfer in z.

    The formula is:
        |S|^2 = [1 - 2*exp(-2*N*A)*cos(2*N*B) + exp(-4*N*A)]
                 / [1 - 2*exp(-2*A)*cos(2*B) + exp(-4*A)]
      where A = Qz.imag * c / 2, B = Qz.real * c / 2.

    Returns the intensity factor for the finite stack, capturing thickness fringes.
    """
    A = (Qz.imag * c) / 2.0
    B = (Qz.real * c) / 2.0
    num = 1.0 - 2.0 * np.exp(-2.0 * N * A) * np.cos(2.0 * N * B) + np.exp(-4.0 * N * A)
    den = 1.0 - 2.0 * np.exp(-2.0 * A)     * np.cos(2.0 * B)     + np.exp(-4.0 * A)
    if abs(den) < 1e-12:
        # If near zero in the denominator, we take the limit ~ N^2
        return N**2
    return num / den

@njit(parallel=True)
def attenuation_array(N, Qz_array, c):
    """
    Vectorized version of attenuation(N, Qz, c) for an entire array Qz_array.
    Runs in parallel using prange for efficiency.

    Parameters
    ----------
    N : int
        Number of layers in the stack.
    Qz_array : ndarray of complex128
        Complex wavevector transfers for which to compute the finite-stack
        interference factor.
    c : float
        Single-layer thickness (Å).

    Returns
    -------
    out : ndarray of float64
        The computed intensities from the finite stack for each Qz in Qz_array.
    """
    n = Qz_array.size
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        out[i] = attenuation(N, Qz_array[i], c)
    return out


# =============================================================================
# 2) PARALLEL CUSTOM MESHGRID
# =============================================================================

@njit(parallel=True)
def custom_meshgrid(qx_vals, qy_vals, qz_vals):
    """
    Parallel replacement for np.meshgrid in 3D. Builds Qx, Qy, Qz arrays of
    shape (len(qx_vals), len(qy_vals), len(qz_vals)).

    Physically:
      - We are building a reciprocal-space grid around a nominal G_vec or region.
      - Each point (Qx, Qy, Qz) is one point in the 3D mosaic distribution.

    Returns
    -------
    Qx, Qy, Qz : 3D arrays
        The coordinate grids for Qx, Qy, Qz.
    """
    nx, ny, nz = len(qx_vals), len(qy_vals), len(qz_vals)
    Qx = np.empty((nx, ny, nz), dtype=qx_vals.dtype)
    Qy = np.empty((nx, ny, nz), dtype=qy_vals.dtype)
    Qz = np.empty((nx, ny, nz), dtype=qz_vals.dtype)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                Qx[i, j, k] = qx_vals[i]
                Qy[i, j, k] = qy_vals[j]
                Qz[i, j, k] = qz_vals[k]
    return Qx, Qy, Qz


# =============================================================================
# 3) PARALLEL SAMPLE_FROM_PDF
# =============================================================================

@njit(parallel=True)
def sample_from_pdf(Qx_grid, Qy_grid, Qz_grid, pdf_3d, n_samples):
    """
    Sample from a 3D probability density function (pdf_3d). This is a parallel
    version that uses prange. Each iteration draws a random number and picks a
    location from the cumulative distribution.

    Physically:
      - pdf_3d is the mosaic orientation distribution in reciprocal space.
      - We are generating random Qx, Qy, Qz from that distribution, each representing
        a possible microcrystallite orientation or partial scattering vector.

    Parameters
    ----------
    Qx_grid, Qy_grid, Qz_grid : 3D arrays
        The coordinate grids from custom_meshgrid.
    pdf_3d : 3D array
        Probability density at each (Qx, Qy, Qz).
    n_samples : int
        Number of random samples to draw.

    Returns
    -------
    (out_Qx, out_Qy, out_Qz) : tuple of 1D arrays
        The random draws from pdf_3d in flattened form.
    """
    Nx, Ny, Nz = Qx_grid.shape
    pdf_flat = pdf_3d.ravel()

    sum_pdf = np.sum(pdf_flat)
    if sum_pdf < 1e-14:
        # No distribution means everything is near zero => return empty arrays
        return (np.zeros(0), np.zeros(0), np.zeros(0))

    # Normalize
    pdf_flat /= sum_pdf

    # Build a cumulative distribution
    cdf_flat = np.cumsum(pdf_flat)

    out_Qx = np.empty(n_samples, dtype=np.float64)
    out_Qy = np.empty(n_samples, dtype=np.float64)
    out_Qz = np.empty(n_samples, dtype=np.float64)

    # Parallel sampling loop
    for i in prange(n_samples):
        r = np.random.rand()  # random uniform in [0,1)
        idx = np.searchsorted(cdf_flat, r)
        if idx == pdf_flat.size:
            idx = pdf_flat.size - 1

        iz = idx % Nz
        iy = (idx // Nz) % Ny
        ix = (idx // (Ny*Nz)) % Nx

        out_Qx[i] = Qx_grid[ix, iy, iz]
        out_Qy[i] = Qy_grid[ix, iy, iz]
        out_Qz[i] = Qz_grid[ix, iy, iz]

    return (out_Qx, out_Qy, out_Qz)

from numba import njit
import numpy as np

@njit
def compute_intensity_array(Qx, Qy, Qz,
                            G_vec,
                            sigma,
                            gamma_pv,
                            eta_pv,
                            H, K,
                            dphi, dtheta):
    """
    Compute the pseudo-Voigt mosaic distribution around G_vec for each (Qx, Qy, Qz).
    Gaussian and Lorentzian components are individually normalized, so their
    mixture preserves unit total area without additional normalization.

    Parameters
    ----------
    Qx, Qy, Qz : array-like
        Coordinates of Q vectors.
    G_vec : length-3 array
        The reciprocal-space vector for the reflection.
    sigma : float
        Gaussian width (rad).
    gamma_pv : float
        Lorentzian half-width at half-maximum (rad).
    eta_pv : float
        Mixing parameter (0=Gaussian, 1=Lorentzian).
    H, K : int
        Miller indices to choose the cap vs band method.
    dphi, dtheta : float
        Angular steps (unused here since profiles are normalized).

    Returns
    -------
    intensities : array-like
        Pseudo-Voigt intensities, same shape as Qx.
    """
    # Unpack G and compute magnitudes
    Gx, Gy, Gz = G_vec[0], G_vec[1], G_vec[2]
    G_mag = np.sqrt(Gx*Gx + Gy*Gy + Gz*Gz)
    if G_mag < 1e-14:
        return np.zeros_like(Qx)

    Q_mag = np.sqrt(Qx*Qx + Qy*Qy + Qz*Qz)
    Q_mag_safe = np.maximum(Q_mag, 1e-14)
    Qr = np.sqrt(Qx*Qx + Qy*Qy)

    # Amplitude factors for normalized 1D profiles
    A_gauss = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    A_lor   = 1.0 / (np.pi * gamma_pv)

    intensities = np.zeros_like(Qx)

    if H == 0 and K == 0:
        # --- CAP method (full-cone around G) ---
        Gnorm = 1.0 / G_mag
        Qnorm = 1.0 / Q_mag_safe
        dot_ = (Gx*Gnorm)*(Qx*Qnorm) + (Gy*Gnorm)*(Qy*Qnorm) + (Gz*Gnorm)*(Qz*Qnorm)
        dot_clamped = np.minimum(np.maximum(dot_, -1.0), 1.0)
        alpha = np.arccos(dot_clamped)

        # Gaussian and Lorentzian contributions
        gauss_val = A_gauss * np.exp(-0.5 * (alpha / sigma)**2)
        lor_val   = A_lor   / (1.0 + (alpha / gamma_pv)**2)

        # Weighted sum: preserves total = 1
        intensities = (1.0 - eta_pv) * gauss_val + eta_pv * lor_val

    else:
        # --- BAND method (ring around G_z axis) ---
        ratioQ = Qz / Q_mag_safe
        ratioQ_clamped = np.minimum(np.maximum(ratioQ, -1.0), 1.0)
        v_prime = np.arccos(ratioQ_clamped)

        ratioG = Gz / G_mag
        ratioG_clamped = np.minimum(np.maximum(ratioG, -1.0), 1.0)
        v_center = np.arccos(ratioG_clamped)

        dv = np.abs(v_prime - v_center)

        # Gaussian and Lorentzian contributions
        gauss_val = A_gauss * np.exp(-0.5 * (dv / sigma)**2)
        lor_val   = A_lor   / (1.0 + (dv / gamma_pv)**2)
        intensities = (1.0 - eta_pv) * gauss_val + eta_pv * lor_val

        # Heuristic weighting along the ring
        R = np.max(Q_mag_safe)
        f = Qr / Q_mag_safe
        t = (1.0 - np.cos(np.pi * f)) / 2.0
        weight = 1.0 + (2.0 * np.pi * R - 1.0) * t
        intensities = intensities / weight

    return intensities


@njit
def Generate_PDF_Grid(
    G_vec,
    sigma, gamma_pv, eta_pv, H,K ,
    Qrange=0.1,   # default value; will be overridden if dynamic_Qrange is True
    n_grid=51,    # grid resolution
    n_samples=10000,  # number of samples
    dynamic_Qrange=True,  # new flag to control dynamic Qrange selection
    multiplier=3.0       # factor to extend the half-width (adjust as needed)
):
    # Compute effective FWHM in angle (radians)
    # For a Gaussian, FWHM = 2.35482*sigma
    # For a Lorentzian, FWHM = 2*gamma_pv
    fwhm_gauss = 2.35482 * sigma
    fwhm_lorentz = 2.0 * gamma_pv
    effective_fwhm = (1 - eta_pv) * fwhm_gauss + eta_pv * fwhm_lorentz
    
    # Use half of the effective FWHM as the angular half-width
    effective_angle_half_width = effective_fwhm / 2.0
    
    # Convert the angular half-width into Q-space using |G_vec|
    G_mag = sqrt(G_vec[0]**2 + G_vec[1]**2 + G_vec[2]**2)
    effective_Q_half_width = G_mag * effective_angle_half_width
    
    # If dynamic Qrange is enabled, override the Qrange value
    if dynamic_Qrange:
        Qrange = multiplier * effective_Q_half_width

    # Build grid in reciprocal space centered around G_vec
    Gx, Gy, Gz = G_vec
    qx_vals = np.linspace(Gx - Qrange, Gx + Qrange, n_grid)
    qy_vals = np.linspace(Gy - Qrange, Gy + Qrange, n_grid)
    qz_vals = np.linspace(Gz - Qrange, Gz + Qrange, n_grid)

    # 1) Build the meshgrid in parallel
    Qx_grid, Qy_grid, Qz_grid = custom_meshgrid(qx_vals, qy_vals, qz_vals)

    # 2) Evaluate mosaic distribution => pdf_3d
    pdf_3d = compute_intensity_array(Qx_grid, Qy_grid, Qz_grid, G_vec, sigma, gamma_pv, eta_pv, H,K)

    # 3) Sample from this 3D pdf
    Qx_s, Qy_s, Qz_s = sample_from_pdf(Qx_grid, Qy_grid, Qz_grid, pdf_3d, n_samples)

    Q_grid = np.stack((Qx_s, Qy_s, Qz_s), axis=1)
    return Q_grid


# =============================================================================
# 5) VECTORIZED INCOHERENT AVERAGING
# =============================================================================

@njit
def incoherent_averaging(Q_grid, N, c, thickness, re_k_z, im_k_z, k_in_crystal, k_mag, n2, bt):
    """
    For a mosaic-sampled Q_grid, compute the average finite-stack interference
    by building Qz_complex for each sample and calling attenuation.

    Physical steps:
      1) For each mosaic point (Qx, Qy, Qz), offset by the incident wave in
         crystal (k_in_crystal) to get the scattered wavevector in the crystal.
      2) Convert to angles, build final Qz_complex with real_k_tz_f, im_k_tz_f.
      3) Summation of attenuation(...) across all mosaic samples (incoherent sum).

    Parameters
    ----------
    Q_grid : ndarray (n_samples, 3)
        Random mosaic points in Q. Q_grid[i] = (Qx_i, Qy_i, Qz_i).
    N : int
        Number of layers in the finite stack.
    c : float
        Single-layer thickness.
    thickness : float
        Full film thickness (not always used explicitly if N*c is thickness).
    re_k_z, im_k_z : float
        Real and imaginary parts of the incident wavevector's z-component.
    k_in_crystal : array_like (3,)
        The in-plane scattering wavevector for the crystal domain.
    k_mag : float
        Magnitude of the wavevector in vacuum 2π/λ.
    n2 : complex
        Refractive index factor used to compute scattering angles.
    bt : float
        A leftover factor from the absorption imaginary part: (Im(n2))^2 * k_mag^2.

    Returns
    -------
    average_int : float
        The average (incoherent) intensity factor from the mosaic sum.
    """
    Qx_s = Q_grid[:,0]
    Qy_s = Q_grid[:,1]
    Qz_s = Q_grid[:,2]

    # Add the crystal wavevector offset
    kx = Qx_s + k_in_crystal[0]
    ky = Qy_s + k_in_crystal[1]
    kz = Qz_s + re_k_z

    # Magnitude in-plane => twotheta
    kr = np.sqrt(kx*kx + ky*ky)
    twotheta = np.arctan2(kz, kr)

    af = k_mag**2 * (np.real(n2)*np.real(n2)) * np.sin(twotheta)**2

    real_k_tz_f = np.sqrt((np.sqrt(af*af + bt*bt) + af)/2.0)
    im_k_tz_f   = np.sqrt((np.sqrt(af*af + bt*bt) - af)/2.0)

    n_samples = Qx_s.size
    Qz_complex_arr = np.empty(n_samples, dtype=np.complex128)

    # Build array of Qz for each sample
    for i in range(n_samples):
        Qz_complex_arr[i] = (real_k_tz_f[i] - re_k_z) - 1j*(im_k_z + im_k_tz_f[i])

    # Evaluate finite-stack interference for all samples
    att_vals = attenuation_array(N, Qz_complex_arr, c)
    total_int = np.sum(att_vals)

    # Return the average (incoherent) intensity
    return total_int / n_samples



# =============================================================================
# 3) INTERSECT_LINE_PLANE, BATCH
# =============================================================================

@njit
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect a single ray (start=P0, direction=k_vec) with a plane
    defined by (P_plane, n_plane). Returns the intersection point (ix, iy, iz)
    and a boolean if valid.

    Physical meaning:
      - Used to find where the scattered beam intersects e.g. the sample plane
        or a detector plane in real space.
    """
    denom = k_vec[0]*n_plane[0] + k_vec[1]*n_plane[1] + k_vec[2]*n_plane[2]
    if abs(denom) < 1e-14:
        # The ray is parallel to the plane. If the starting point already lies
        # on the plane (within a tolerance) we treat it as the intersection
        # point so that grazing rays are not discarded.
        dist = ((P0[0] - P_plane[0]) * n_plane[0]
              + (P0[1] - P_plane[1]) * n_plane[1]
              + (P0[2] - P_plane[2]) * n_plane[2])
        if abs(dist) < 1e-6:
            return (P0[0], P0[1], P0[2], True)
        return (np.nan, np.nan, np.nan, False)
    num = ((P_plane[0] - P0[0]) * n_plane[0]
         + (P_plane[1] - P0[1]) * n_plane[1]
         + (P_plane[2] - P0[2]) * n_plane[2])
    t = num / denom
    # Numerical precision can yield tiny negative values for *t* when the ray
    # should intersect exactly on the plane.  Allow a small tolerance so these
    # near-zero cases are not discarded which previously produced missing bands
    # on the detector when the beam was almost parallel to the sample plane.
    if t < -1e-9:
        return (np.nan, np.nan, np.nan, False)
    if t < 0.0:
        t = 0.0
    ix = P0[0] + t*k_vec[0]
    iy = P0[1] + t*k_vec[1]
    iz = P0[2] + t*k_vec[2]
    return (ix, iy, iz, True)

@njit
def intersect_line_plane_batch(start_pt, directions, plane_pt, plane_n):
    """
    Batch version: intersect multiple directions with a plane.

    directions : shape (N,3)
        Each row is a vector to test.

    Returns
    -------
    intersects : shape (N,3)
        Intersection points (ix, iy, iz).
    valid : boolean array of length N
        True if the intersection is valid (t>=0, not parallel).
    """
    Ndir = directions.shape[0]
    intersects = np.full((Ndir,3), np.nan, dtype=np.float64)
    valid = np.zeros(Ndir, dtype=np.bool_)

    dist = ((start_pt[0] - plane_pt[0]) * plane_n[0]
          + (start_pt[1] - plane_pt[1]) * plane_n[1]
          + (start_pt[2] - plane_pt[2]) * plane_n[2])

    for i in range(Ndir):
        kx = directions[i,0]
        ky = directions[i,1]
        kz = directions[i,2]
        dot_dn = kx*plane_n[0] + ky*plane_n[1] + kz*plane_n[2]
        if abs(dot_dn) < 1e-14:
            if abs(dist) < 1e-6:
                intersects[i,0] = start_pt[0]
                intersects[i,1] = start_pt[1]
                intersects[i,2] = start_pt[2]
                valid[i] = True
            continue
        num = ((plane_pt[0]-start_pt[0])*plane_n[0]
             + (plane_pt[1]-start_pt[1])*plane_n[1]
             + (plane_pt[2]-start_pt[2])*plane_n[2])
        t = num/dot_dn
        if t < -1e-9:
            continue
        if t < 0.0:
            t = 0.0
        ix = start_pt[0] + t*kx
        iy = start_pt[1] + t*ky
        iz = start_pt[2] + t*kz
        intersects[i,0] = ix
        intersects[i,1] = iy
        intersects[i,2] = iz
        valid[i] = True
    return intersects, valid


# =============================================================================
# 4) solve_q
# =============================================================================

@njit
def solve_q(
    k_in_crystal, k_scat, G_vec, sigma, gamma_pv, eta_pv, H, K, L,
    N_steps=1000
):
    """
    Build a 'circle' in reciprocal space for the reflection G_vec, i.e. the
    set of Q that satisfies |Q|=|G| or an intersection with Ewald sphere, then
    filter by mosaic distribution compute_intensity_array.

    Physically: 
      - We param by angle from 0..2π,
      - Circle radius circle_r,
      - Then for each Q on that circle, compute mosaic weighting all_int.

    Returns
    -------
    out : ndarray of shape (M,4)
        For the valid points, columns = (Qx, Qy, Qz, mosaic_intensity).
    status : int
        0 for success or a negative code indicating the failure reason.
    """
    status = 0
    G_sq = G_vec[0]*G_vec[0] + G_vec[1]*G_vec[1] + G_vec[2]*G_vec[2]
    if G_sq < 1e-14:
        status = -1
        return np.zeros((0, 4), dtype=np.float64), status

    Ax = -k_in_crystal[0]
    Ay = -k_in_crystal[1]
    Az = -k_in_crystal[2]
    rA = k_scat
    A_sq = Ax*Ax + Ay*Ay + Az*Az
    if A_sq < 1e-14:
        status = -2
        return np.zeros((0, 4), dtype=np.float64), status
    A_len = sqrt(A_sq)

    c = (G_sq + A_sq - rA*rA) / (2.0*A_len)
    # Compute circle parameters
    circle_r_sq = G_sq - c*c
    if circle_r_sq < 0.0:
        status = -3
        return np.zeros((0, 4), dtype=np.float64), status
    circle_r = np.sqrt(circle_r_sq)

    Ax_hat = Ax / A_len
    Ay_hat = Ay / A_len
    Az_hat = Az / A_len

    Ox = c * Ax_hat
    Oy = c * Ay_hat
    Oz = c * Az_hat

    # Build two orthonormal vectors (e1, e2) in the plane perpendicular to Ax_hat.
    ax, ay, az = 1.0, 0.0, 0.0
    dot_aA = ax*Ax_hat + ay*Ay_hat + az*Az_hat
    if abs(dot_aA) > 0.9999:
        ax, ay, az = 0.0, 1.0, 0.0
        dot_aA = ax*Ax_hat + ay*Ay_hat + az*Az_hat
    aox = ax - dot_aA*Ax_hat
    aoy = ay - dot_aA*Ay_hat
    aoz = az - dot_aA*Az_hat
    ao_len = np.sqrt(aox*aox + aoy*aoy + aoz*aoz)
    if ao_len < 1e-14:
        status = -4
        return np.zeros((0, 4), dtype=np.float64), status
    e1x = aox / ao_len
    e1y = aoy / ao_len
    e1z = aoz / ao_len

    e2x = Az_hat*e1y - Ay_hat*e1z
    e2y = Ax_hat*e1z - Az_hat*e1x
    e2z = Ay_hat*e1x - Ax_hat*e1y
    e2_len = np.sqrt(e2x*e2x + e2y*e2y + e2z*e2z)
    if e2_len < 1e-14:
        status = -5
        return np.zeros((0, 4), dtype=np.float64), status
    e2x /= e2_len
    e2y /= e2_len
    e2z /= e2_len

    # Parameterize the circle
    dtheta = 2.0*np.pi / N_steps
    theta_arr = dtheta * np.arange(N_steps)
    cth = np.cos(theta_arr)
    sth = np.sin(theta_arr)
    Qx_arr = Ox + circle_r*(cth*e1x + sth*e2x)
    Qy_arr = Oy + circle_r*(cth*e1y + sth*e2y)
    Qz_arr = Oz + circle_r*(cth*e1z + sth*e2z)

    # Compute dtheta from the parametrization
    dtheta = 2.0 * np.pi / N_steps

    # Estimate an effective φ from the Q points.
    Q_mag_sample = np.sqrt(Qx_arr[0]*Qx_arr[0] + Qy_arr[0]*Qy_arr[0] + Qz_arr[0]*Qz_arr[0])
    phi_vals = np.empty(N_steps, dtype=np.float64)
    for i in range(N_steps):
        # Clamp Qz/Q_mag to [-1,1] to avoid numerical issues.
        ratio = Qz_arr[i] / Q_mag_sample
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0
        phi_vals[i] = np.arccos(ratio)
    phi0 = 0.0
    for i in range(N_steps):
        phi0 += phi_vals[i]
    phi0 /= N_steps

    # Now, equate the area element sin(φ0) dφ_eff with the circle arc length (circle_r * dθ)
    if np.sin(phi0) < 1e-12:
        dphi_eff = 0.0  # or some fallback value
    else:
        dphi_eff = circle_r * dtheta / np.sin(phi0)

    # Now call compute_intensity_array with the effective dphi and dtheta.
    all_int = compute_intensity_array(Qx_arr, Qy_arr, Qz_arr, G_vec, sigma, gamma_pv, eta_pv, H, K, dphi_eff, dtheta)

    # Apply any intensity cutoff and construct the output.
    intensity_cutoff = np.exp(-100.0)
    mask = mask = (all_int > intensity_cutoff)
    valid_idx = np.nonzero(mask)[0]

    out = np.zeros((valid_idx.size, 4), dtype=np.float64)
    for i in range(valid_idx.size):
        idx = valid_idx[i]
        out[i, 0] = Qx_arr[idx]
        out[i, 1] = Qy_arr[idx]
        out[i, 2] = Qz_arr[idx]
        out[i, 3] = all_int[idx]

    status = 0
    return out, status

# =============================================================================
# 5) CALCULATE_PHI
# =============================================================================

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
    sigma_rad, gamma_pv, eta_pv,
    debye_x, debye_y,
    center,
    theta_initial_deg,
    R_x_detector, R_z_detector, n_det_rot, Detector_Pos,
    e1_det, e2_det,
    R_z_R_y,
    R_ZY_n,
    P0, unit_x,
    save_flag, q_data, q_count, i_peaks_index,
    record_status=False
):
    """
    For a single reflection (H,K,L), build a mosaic Q_grid around G_vec.
    Then for each sample beam (beam_x_array, etc.), compute the intersection
    with the sample plane, wavevectors in the crystal, do "incoherent_averaging"
    + "solve_q", and deposit intensities into the final 2D 'image'.

    Physical meaning:
      - This function is the "core" for each reflection, merging mosaic 
        sampling (Q_grid) with geometry (plane intersection, wavevectors) 
        and the final detection step.

    Returns
    -------
    pixel_hits : ndarray
        Record of peak intensities and pixel positions for the best beam sample.
    statuses : ndarray
        ``solve_q`` status codes when ``record_status`` is True.
    missed_kf : ndarray
        Outgoing wavevectors that failed to intersect the detector plane.
    sol_debug : ndarray
        Debug table recording ``[i_samp, i_sol, hit_flag, cpx, rpx, Qx, Qy, Qz]``
        for every solution from ``solve_q``.
    """
    gz0 = 2.0*pi*(L/cv)
    gr0 = 4.0*pi/av * sqrt((H*H + H*K + K*K)/3.0)
    G_vec = np.array([0.0, gr0, gz0], dtype=np.float64)

    # # Build a random mosaic distribution around G_vec
    # Q_grid = Generate_PDF_Grid(
    #     G_vec,
    #     sigma_rad, gamma_pv, eta_pv,
    #     Qrange=1,   # half-width in Q around G_vec
    #     n_grid=51,  # grid resolution
    #     n_samples=2000
    # )
    n_samp = beam_x_array.size

    max_hits = n_samp * 2          # 2 solutions per beam sample ≫ safe
    pixel_hits = np.empty((max_hits, 7), dtype=np.float64)
    missed_kf = np.empty((max_hits, 3), dtype=np.float64)
    sol_debug = np.empty((max_hits, 8), dtype=np.float64)
    n_hits = 0                     # running counter
    n_missed = 0
    n_sol_dbg = 0
    if record_status:
        statuses = np.zeros(n_samp, dtype=np.int64)

    #thickness = 500.0  # film thickness in Å

    #N = int(thickness/cv)  # Number of layers (approx)

    # Build a sample rotation from "theta_initial_deg"
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

    best_idx   = 0
    best_angle = theta_array[0]**2 + phi_array[0]**2
    for ii in range(1, n_samp):
        metric = theta_array[ii]**2 + phi_array[ii]**2
        if metric < best_angle:
            best_angle = metric
            best_idx   = ii


    # Build a local reference for the beam incidence
    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_norm = sqrt(e1_temp[0]*e1_temp[0] + e1_temp[1]*e1_temp[1] + e1_temp[2]*e1_temp[2])
    if e1_norm < 1e-12:
        # fallback if cross is degenerate
        alt_refs = [
            np.array([1.0,0.0,0.0]),
            np.array([0.0,1.0,0.0]),
            np.array([0.0,0.0,1.0])
        ]
        success = False
        for ar in alt_refs:
            cross_tmp = np.cross(n_surf, ar)
            cross_norm_tmp = sqrt(cross_tmp[0]*cross_tmp[0] + cross_tmp[1]*cross_tmp[1] + cross_tmp[2]*cross_tmp[2])
            if cross_norm_tmp > 1e-12:
                e1_temp = cross_tmp / cross_norm_tmp
                success = True
                break

    else:
        e1_temp /= e1_norm

    e2_temp = np.cross(n_surf, e1_temp)

    # Main loop over each beam sample in wave + mosaic
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

        # Intersect the beam with the sample plane
        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if not valid_int:
            if record_status:
                statuses[i_samp] = -10
            continue

        I_plane = np.array([ix, iy, iz])
        kn_dot = k_in[0]*n_surf[0] + k_in[1]*n_surf[1] + k_in[2]*n_surf[2]
        th_i_prime = (pi/2.0) - acos(kn_dot)

        projected_incident = k_in - kn_dot*n_surf
        pln = sqrt(projected_incident[0]*projected_incident[0]
                 + projected_incident[1]*projected_incident[1]
                 + projected_incident[2]*projected_incident[2])
        if pln > 1e-12:
            projected_incident /= pln
        else:
            projected_incident[:] = 0.0

        p1 = projected_incident[0]*e1_temp[0] + projected_incident[1]*e1_temp[1] + projected_incident[2]*e1_temp[2]
        p2 = projected_incident[0]*e2_temp[0] + projected_incident[1]*e2_temp[1] + projected_incident[2]*e2_temp[2]
        phi_i_prime = (pi/2.0) - np.arctan2(p2, p1)

        th_t = acos(cos(th_i_prime)/np.real(n2))*np.sign(th_i_prime)
        
        # k_scat is magnitude of the scattering wave in the crystal
        k_scat = k_mag*sqrt(np.real(n2)*np.real(n2))

        k_x_scat = k_scat*cos(th_t)*sin(phi_i_prime)
        k_y_scat = k_scat*cos(th_t)*cos(phi_i_prime)

        # Compute partial absorption factors
        at = k_mag**2 * np.real(n2)*np.real(n2) * np.sin(th_t)**2
        bt = np.imag(n2)**2 * k_mag**2

        re_k_z = - np.sqrt((np.sqrt(at*at + bt*bt) + at)/2.0)
        #im_k_z =   np.sqrt((np.sqrt(at*at + bt*bt) - at)/2.0)

        k_in_crystal = np.array([k_x_scat, k_y_scat, re_k_z])

        # Incoherent mosaic average
        #incoherent = incoherent_averaging(Q_grid, N, cv, thickness,
        #                                  re_k_z, im_k_z, k_in_crystal,
        #                                  k_mag, n2, bt)
        #Ti = fresnel_transmission(th_t, n2)
        
        # solve_q approach for reflection geometry
        All_Q, stat = solve_q(k_in_crystal, k_scat, G_vec, sigma_rad, gamma_pv, eta_pv, H, K, L, 1000)
        if record_status:
            statuses[i_samp] = stat
        for i_sol in range(All_Q.shape[0]):
            Qx = All_Q[i_sol, 0]
            Qy = All_Q[i_sol, 1]
            Qz = All_Q[i_sol, 2]
            I_Q = All_Q[i_sol, 3]
            if I_Q < 1e-5:
                sol_debug[n_sol_dbg, 0] = i_samp
                sol_debug[n_sol_dbg, 1] = i_sol
                sol_debug[n_sol_dbg, 2] = 0.0
                sol_debug[n_sol_dbg, 3] = np.nan
                sol_debug[n_sol_dbg, 4] = np.nan
                sol_debug[n_sol_dbg, 5] = Qx
                sol_debug[n_sol_dbg, 6] = Qy
                sol_debug[n_sol_dbg, 7] = Qz
                n_sol_dbg += 1
                continue
            
            k_tx_prime = Qx + k_x_scat
            k_ty_prime = Qy + k_y_scat
            k_tz_prime = Qz + re_k_z

            kr = sqrt(k_tx_prime*k_tx_prime + k_ty_prime*k_ty_prime)
            if kr < 1e-12:
                twotheta_t = 0.0
            else:
                twotheta_t_prime = np.arctan(k_tz_prime/kr)
                twotheta_t = acos(cos(twotheta_t_prime)* np.real(n2))*np.sign(twotheta_t_prime)

            phi_f = np.arctan2(k_tx_prime, k_ty_prime)
            k_tx_f = k_scat*cos(twotheta_t)*sin(phi_f)
            k_ty_f = k_scat*cos(twotheta_t)*cos(phi_f)
            k_tz_f = k_scat*sin(twotheta_t)
            #Tf = fresnel_transmission(th_t, n2, direction="out")

            kf = np.array([k_tx_f, k_ty_f, k_tz_f])
            kf_prime = R_sample @ kf

            dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
            if not valid_det:
                if n_missed < max_hits:
                    missed_kf[n_missed, 0] = kf_prime[0]
                    missed_kf[n_missed, 1] = kf_prime[1]
                    missed_kf[n_missed, 2] = kf_prime[2]
                    n_missed += 1
                sol_debug[n_sol_dbg, 0] = i_samp
                sol_debug[n_sol_dbg, 1] = i_sol
                sol_debug[n_sol_dbg, 2] = 0.0
                sol_debug[n_sol_dbg, 3] = np.nan
                sol_debug[n_sol_dbg, 4] = np.nan
                sol_debug[n_sol_dbg, 5] = Qx
                sol_debug[n_sol_dbg, 6] = Qy
                sol_debug[n_sol_dbg, 7] = Qz
                n_sol_dbg += 1
                continue

            plane_to_det = np.array([dx - Detector_Pos[0],
                                     dy - Detector_Pos[1],
                                     dz - Detector_Pos[2]], dtype=np.float64)
            x_det = plane_to_det[0]*e1_det[0] + plane_to_det[1]*e1_det[1] + plane_to_det[2]*e1_det[2]
            y_det = plane_to_det[0]*e2_det[0] + plane_to_det[1]*e2_det[1] + plane_to_det[2]*e2_det[2]

            rpx = int(round(center[0] - y_det/100e-6))
            cpx = int(round(center[1] + x_det/100e-6))
            if not (0 <= rpx < image_size and 0 <= cpx < image_size):
                sol_debug[n_sol_dbg, 0] = i_samp
                sol_debug[n_sol_dbg, 1] = i_sol
                sol_debug[n_sol_dbg, 2] = 0.0
                sol_debug[n_sol_dbg, 3] = np.nan
                sol_debug[n_sol_dbg, 4] = np.nan
                sol_debug[n_sol_dbg, 5] = Qx
                sol_debug[n_sol_dbg, 6] = Qy
                sol_debug[n_sol_dbg, 7] = Qz
                n_sol_dbg += 1
                continue

            # Combine:
            #  1) reflection_intensity -> structure/basis factor
            #  2) I_Q -> partial mosaic weighting from solve_q circle
            #  3) incoherent -> the mosaic average from Q_grid
            #  4) exponent dampings -> Debye or extra broadening
            val = (reflection_intensity * I_Q #* incoherent * (abs(Ti)**2) + (abs(Tf)**2)
                * exp(-Qz*Qz * debye_x*debye_x)
                * exp(-(Qx*Qx + Qy*Qy) * debye_y*debye_y))
            image[rpx, cpx] += val
            if i_samp == best_idx:
                
                if valid_det and 0 <= rpx < image_size and 0 <= cpx < image_size:
                    # save       I_Q⋅extras        x-pix     y-pix      φf
                    pixel_hits[n_hits, 0] = val
                    pixel_hits[n_hits, 1] = cpx
                    pixel_hits[n_hits, 2] = rpx
                    pixel_hits[n_hits, 3] = phi_f
                    pixel_hits[n_hits, 4] = H
                    pixel_hits[n_hits, 5] = K
                    pixel_hits[n_hits, 6] = L

                    n_hits += 1

            sol_debug[n_sol_dbg, 0] = i_samp
            sol_debug[n_sol_dbg, 1] = i_sol
            sol_debug[n_sol_dbg, 2] = 1.0
            sol_debug[n_sol_dbg, 3] = cpx
            sol_debug[n_sol_dbg, 4] = rpx
            sol_debug[n_sol_dbg, 5] = Qx
            sol_debug[n_sol_dbg, 6] = Qy
            sol_debug[n_sol_dbg, 7] = Qz
            n_sol_dbg += 1
                
            # Optionally store Q-data
            if save_flag==1 and q_count[i_peaks_index]< q_data.shape[1]:
                idx = q_count[i_peaks_index]
                q_data[i_peaks_index, idx,0] = Qx
                q_data[i_peaks_index, idx,1] = Qy
                q_data[i_peaks_index, idx,2] = Qz
                q_data[i_peaks_index, idx,3] = val
                q_count[i_peaks_index]+=1

    if record_status:
        return pixel_hits[:n_hits], statuses, missed_kf[:n_missed], sol_debug[:n_sol_dbg]
    else:
        return pixel_hits[:n_hits], np.empty(0, dtype=np.int64), missed_kf[:n_missed], sol_debug[:n_sol_dbg]



# =============================================================================
# 6) PROCESS_PEAKS_PARALLEL
# =============================================================================

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
    save_flag,
    record_status=False
):
    """
    High-level loop over multiple reflections from 'miller', each with an intensity
    from 'intensities'. For each reflection, call 'calculate_phi(...).'

    parallel=True: We do a prange over each reflection. Each reflection is processed
    independently, building the mosaic, computing geometry, and depositing
    intensities in the final 'image'.

    Physically:
      - This simulates multiple Bragg peaks in a single run,
      - Summing up the resulting scattered intensities for each reflection.

    Returns
    -------
    image : ndarray
        The simulated diffraction image.
    hit_tables : List of ndarrays
        Per-reflection list of pixel hits for the best beam sample.
    q_data, q_count : ndarrays
        Detailed Q sampling information when ``save_flag`` is 1, otherwise
        placeholder arrays.
    all_status : ndarray
        ``solve_q`` status codes for every beam sample when ``record_status`` is
        True.
    miss_tables : List of ndarrays
        Outgoing wavevectors that failed to intersect the detector.
    sol_tables : List of ndarrays
        For debugging, each entry records ``[i_samp, i_sol, hit_flag, cpx, rpx,
        Qx, Qy, Qz]`` for every solution of ``solve_q``.
    """
    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)

    sigma_rad   = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)

    # Build transforms for the detector
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
    e1_len = sqrt(e1_det[0]*e1_det[0] + e1_det[1]*e1_det[1] + e1_det[2]*e1_det[2])
    if e1_len < 1e-14:
        e1_det = np.array([1.0, 0.0, 0.0])
    else:
        e1_det /= e1_len

    tmpx = n_det_rot[1]* e1_det[2] - n_det_rot[2]* e1_det[1]
    tmpy = n_det_rot[2]* e1_det[0] - n_det_rot[0]* e1_det[2]
    tmpz = n_det_rot[0]* e1_det[1] - n_det_rot[1]* e1_det[0]
    e2_det = np.array([-tmpx, -tmpy, -tmpz], dtype=np.float64)
    e2_len = sqrt(e2_det[0]*e2_det[0] + e2_det[1]*e2_det[1] + e2_det[2]*e2_det[2])
    if e2_len < 1e-14:
        e2_det = np.array([0.0,1.0,0.0])
    else:
        e2_det /= e2_len

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

    n1= np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R_ZY_n= R_z_R_y @ n1
    nzy_len= sqrt(R_ZY_n[0]*R_ZY_n[0] + R_ZY_n[1]*R_ZY_n[1] + R_ZY_n[2]*R_ZY_n[2])
    R_ZY_n/= nzy_len

    P0= np.array([0.0, 0.0, -zs], dtype=np.float64)
    num_peaks= miller.shape[0]

    max_solutions= 2000000
    if save_flag==1:
        q_data= np.full((num_peaks, max_solutions, 5), np.nan, dtype=np.float64)
        q_count= np.zeros(num_peaks, dtype=np.int64)
    else:
        q_data= np.zeros((1,1,5), dtype=np.float64)
        q_count= np.zeros(1, dtype=np.int64)
    hit_tables = List.empty_list(types.float64[:, ::1])
    miss_tables = List.empty_list(types.float64[:, ::1])
    sol_tables  = List.empty_list(types.float64[:, ::1])
    all_status = np.zeros((num_peaks, beam_x_array.size), dtype=np.int64)

    # prange over each reflection
    for i_pk in prange(num_peaks):
        H= miller[i_pk,0]
        K= miller[i_pk,1]
        L= miller[i_pk,2]
        reflI= intensities[i_pk]

        # We'll do a reflection-level call to calculate_phi
        pixel_hits, status_arr, missed_arr, sol_dbg = calculate_phi(
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
            save_flag, q_data, q_count, i_pk,
            record_status
        )
        if record_status:
            all_status[i_pk, :] = status_arr
        hit_tables.append(pixel_hits)
        miss_tables.append(missed_arr)
        sol_tables.append(sol_dbg)
    return image, hit_tables, q_data, q_count, all_status, miss_tables, sol_tables


def debug_detector_paths(
    beam_x_array, beam_y_array, theta_array, phi_array,
    theta_initial_deg, chi_deg, psi_deg,
    zb, zs,
    Distance_CoR_to_Detector, gamma_deg, Gamma_deg,
    n_detector=np.array([0.0, 1.0, 0.0]),
    unit_x=np.array([1.0, 0.0, 0.0])
):
    """Trace specular reflection paths for debugging.

    For each beam sample, this returns whether the incoming ray intersects the
    sample plane and whether its specular reflection would intersect the
    detector plane.  The function is purely for debugging geometry issues and is
    not JIT compiled.

    Parameters
    ----------
    beam_x_array, beam_y_array, theta_array, phi_array : array-like
        Sampled beam parameters as used in :func:`process_peaks_parallel`.
    theta_initial_deg : float
        Sample tilt around the x-axis.
    chi_deg, psi_deg : float
        Additional sample rotations around y and z.
    zb, zs : float
        Beam and sample offsets used in the main simulation.
    Distance_CoR_to_Detector, gamma_deg, Gamma_deg : float
        Detector geometry parameters.

    Returns
    -------
    out : ndarray of shape (N,4)
        Columns contain (theta, phi, hit_sample, hit_detector).
    """
    gamma_rad = np.radians(gamma_deg)
    Gamma_rad = np.radians(Gamma_deg)
    chi_rad   = np.radians(chi_deg)
    psi_rad   = np.radians(psi_deg)
    rad_theta_i = np.radians(theta_initial_deg)

    cg = np.cos(gamma_rad); sg = np.sin(gamma_rad)
    cG = np.cos(Gamma_rad); sG = np.sin(Gamma_rad)
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
    n_det_rot /= np.linalg.norm(n_det_rot)
    Detector_Pos = np.array([0.0, Distance_CoR_to_Detector, 0.0])

    c_chi = np.cos(chi_rad); s_chi = np.sin(chi_rad)
    R_y = np.array([
        [ c_chi, 0.0,   s_chi],
        [ 0.0,   1.0,   0.0],
        [-s_chi, 0.0, c_chi]
    ])
    c_psi= np.cos(psi_rad); s_psi= np.sin(psi_rad)
    R_z = np.array([
        [ c_psi, s_psi, 0.0],
        [-s_psi, c_psi, 0.0],
        [ 0.0,   0.0,   1.0]
    ])
    R_z_R_y = R_z @ R_y

    ct = np.cos(rad_theta_i); st = np.sin(rad_theta_i)
    R_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, ct, -st],
        [0.0, st,  ct]
    ])
    R_sample = R_x @ R_z_R_y

    n_surf = R_x @ (R_z_R_y @ np.array([0.0, 0.0, 1.0]))
    n_surf /= np.linalg.norm(n_surf)

    P0 = np.array([0.0, 0.0, -zs])
    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0

    N = len(theta_array)
    out = np.zeros((N, 4), dtype=np.float64)

    for i in range(N):
        dtheta = theta_array[i]
        dphi   = phi_array[i]
        bx = beam_x_array[i]
        by = beam_y_array[i]

        k_in = np.array([
            np.cos(dtheta)*np.sin(dphi),
            np.cos(dtheta)*np.cos(dphi),
            np.sin(dtheta)
        ])

        beam_start = np.array([bx, -20e-3, -zb + by])
        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)

        hit_sample = 1.0 if valid_int else 0.0
        if not valid_int:
            out[i] = [dtheta, dphi, 0.0, 0.0]
            continue

        I_plane = np.array([ix, iy, iz])
        dot_kn = k_in[0]*n_surf[0] + k_in[1]*n_surf[1] + k_in[2]*n_surf[2]
        k_spec = k_in - 2.0*dot_kn*n_surf

        dx, dy, dz, valid_det = intersect_line_plane(I_plane, k_spec, Detector_Pos, n_det_rot)
        hit_det = 1.0 if valid_det else 0.0

        out[i] = [dtheta, dphi, hit_sample, hit_det]

    return out


