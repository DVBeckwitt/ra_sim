"""Core diffraction routines used by the simulator."""

import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange
from math import sin, cos, sqrt, pi, exp, acos
from ra_sim.utils.calculations import complex_sqrt, fresnel_transmission
from numba import types
from numba.typed import Dict, List

_SOLVE_Q_CACHE_KEY_TYPE = types.UniTuple(types.int64, 13)
_SOLVE_Q_CACHE_VAL_TYPE = types.float64[:, ::1]
_DET_PROJ_CACHE_KEY_TYPE = types.UniTuple(types.int64, 10)
_DET_PROJ_CACHE_VAL_TYPE = types.float64[::1]

# Optical transport modes
# Canonical names:
#   - FRESNEL_CTR_DAMPING: fast Fresnel-weighted CTR damping path
#   - COMPLEX_K_DWBA_SLAB: precise complex-k slab optics path
FRESNEL_CTR_DAMPING = 0
COMPLEX_K_DWBA_SLAB = 1

# Backward-compatible aliases used throughout the existing codebase.
OPTICS_MODE_FAST = FRESNEL_CTR_DAMPING
OPTICS_MODE_EXACT = COMPLEX_K_DWBA_SLAB

# solve_q hot-loop constants
DEFAULT_SOLVE_Q_STEPS = 1000
_DEFAULT_SOLVE_Q_DTHETA = (2.0 * np.pi) / DEFAULT_SOLVE_Q_STEPS
_DEFAULT_SOLVE_Q_COS = np.cos(
    _DEFAULT_SOLVE_Q_DTHETA * np.arange(DEFAULT_SOLVE_Q_STEPS, dtype=np.float64)
)
_DEFAULT_SOLVE_Q_SIN = np.sin(
    _DEFAULT_SOLVE_Q_DTHETA * np.arange(DEFAULT_SOLVE_Q_STEPS, dtype=np.float64)
)
_INTENSITY_CUTOFF = float(np.exp(-100.0))
_QUANTIZATION_SIGMA_PIXELS = 0.6
_QUANTIZATION_MIN_MARGIN_PIXELS = 0.02
_NEAR_CRITICAL_KZ_RATIO = 2e-3
_NEAR_SOLVE_Q_BRANCH_EPS = 1e-10
_DETECTOR_GRAZING_COS_EPS = 2e-3
_CENTROID_SHIFT_BUDGET_PX = 0.02
_FWHM_SHIFT_BUDGET_FRAC = 5e-3
_INTEGRATED_INTENSITY_SHIFT_BUDGET_FRAC = 5e-3
_FWHM_REFERENCE_PIXELS = 4.0
_DET_PROJ_GRAZING_COS_EPS = 5e-3
_DET_PROJ_MIN_BOUNDARY_MARGIN_PX = 0.12
# Keep thread-local image buffers bounded to avoid runaway allocations.
_THREAD_LOCAL_IMAGE_MAX_BYTES = 768 * 1024 * 1024
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


@njit
def wrap_to_pi(x):
    while x <= -pi:
        x += 2.0 * pi
    while x > pi:
        x -= 2.0 * pi
    return x


@njit
def compute_intensity_array(Qx, Qy, Qz,
                            G_vec,
                            sigma,
                            gamma_pv,
                            eta_pv):
    """
    Compute the mosaic surface density sigma(theta) on the Bragg sphere for
    each (Qx, Qy, Qz). Uses a pseudo-Voigt in the grazing-angle offset.

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

    Returns
    -------
    intensities : array-like
        Surface density sigma(theta), same shape as Qx.
    """
    # Unpack G and compute magnitudes
    Gx, Gy, Gz = G_vec[0], G_vec[1], G_vec[2]
    G_mag = np.sqrt(Gx*Gx + Gy*Gy + Gz*Gz)
    if G_mag < 1e-14:
        return np.zeros_like(Qx)

    Qr = np.sqrt(Qx*Qx + Qy*Qy)

    # Amplitude factors for normalized 1D profiles
    A_gauss = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    A_lor   = 1.0 / (np.pi * gamma_pv)

    # Reference grazing angle for the reflection
    Gr = np.sqrt(Gx*Gx + Gy*Gy)
    theta0 = np.arctan2(Gz, Gr)

    denom_base = 2.0 * np.pi * G_mag * G_mag

    intensities = np.empty_like(Qx)
    Qx_flat = Qx.ravel()
    Qy_flat = Qy.ravel()
    Qz_flat = Qz.ravel()
    Qr_flat = Qr.ravel()
    out_flat = intensities.ravel()

    for i in range(Qx_flat.size):
        theta = np.arctan2(Qz_flat[i], Qr_flat[i])
        dtheta = wrap_to_pi(theta - theta0)

        gauss_val = A_gauss * np.exp(-0.5 * (dtheta / sigma)**2)
        lor_val   = A_lor   / (1.0 + (dtheta / gamma_pv)**2)
        omega = (1.0 - eta_pv) * gauss_val + eta_pv * lor_val

        # Keep a geometry normalization that is stable for pseudo-Voigt tails.
        # The previous 1/cos(theta) factor caused pole amplification when eta>0
        # (Lorentzian component), which collapsed Bragg-sphere color scales.
        out_flat[i] = omega / denom_base

    return intensities


@njit
def Generate_PDF_Grid(
    G_vec,
    sigma, gamma_pv, eta_pv,
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
    pdf_3d = compute_intensity_array(Qx_grid, Qy_grid, Qz_grid, G_vec, sigma, gamma_pv, eta_pv)

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
def intersect_infinite_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect an infinite line (start=P0, direction=k_vec) with a plane.
    Unlike ``intersect_line_plane`` this does not reject negative t values.
    """
    denom = k_vec[0]*n_plane[0] + k_vec[1]*n_plane[1] + k_vec[2]*n_plane[2]
    if abs(denom) < 1e-14:
        # Parallel direction: project start point orthogonally onto the plane
        # instead of declaring "no hit", avoiding artificial horizon cutoffs.
        dist = ((P0[0] - P_plane[0]) * n_plane[0]
              + (P0[1] - P_plane[1]) * n_plane[1]
              + (P0[2] - P_plane[2]) * n_plane[2])
        n_sq = (
            n_plane[0]*n_plane[0]
            + n_plane[1]*n_plane[1]
            + n_plane[2]*n_plane[2]
        )
        if n_sq < 1e-20:
            return (np.nan, np.nan, np.nan, False)
        scale = dist / n_sq
        ix = P0[0] - scale * n_plane[0]
        iy = P0[1] - scale * n_plane[1]
        iz = P0[2] - scale * n_plane[2]
        return (ix, iy, iz, True)

    num = ((P_plane[0] - P0[0]) * n_plane[0]
         + (P_plane[1] - P0[1]) * n_plane[1]
         + (P_plane[2] - P0[2]) * n_plane[2])
    t = num / denom
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


# ---------- NEW JIT-SAFE HELPERS ----------
@njit
def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit
def _kz_branch_decay(arg):
    """Return sqrt(arg) with the physically decaying branch (Im(kz) >= 0)."""
    kz = complex_sqrt(arg)
    if kz.imag < 0.0:
        kz = -kz
    elif abs(kz.imag) < 1e-15 and kz.real < 0.0:
        kz = -kz
    return kz


@njit
def _fresnel_t_exact(kz_i, kz_j, eps_i, eps_j, s_polarization):
    """Exact Fresnel transmission amplitude in kz/epsilon form."""
    if s_polarization:
        den = kz_i + kz_j
        if abs(den) < 1e-30:
            return 0j
        return (2.0 * kz_i) / den

    den = eps_j * kz_i + eps_i * kz_j
    if abs(den) < 1e-30:
        return 0j
    return (2.0 * eps_j * kz_i) / den


@njit
def _fresnel_power_t_exact(t_amp, kz_i, kz_j, eps_i, eps_j, s_polarization):
    """Convert exact transmission amplitude to power transmission."""
    abs_t2 = t_amp.real * t_amp.real + t_amp.imag * t_amp.imag
    if abs_t2 <= 0.0:
        return 0.0

    if s_polarization:
        denom = kz_i.real
        if abs(denom) < 1e-30:
            return 0.0
        ratio = kz_j.real / denom
    else:
        den = kz_i / eps_i
        if abs(den) < 1e-30:
            return 0.0
        ratio = ((kz_j / eps_j) / den).real

    out = ratio * abs_t2
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    return out


@njit
def _thickness_to_angstrom(depth):
    """Interpret sub-mm values as meters and convert to angstrom."""
    if depth <= 0.0:
        return 0.0
    # Existing defaults pass thickness in meters (e.g., 50e-9).
    if depth < 1e-3:
        return depth * 1.0e10
    # Larger values are assumed to already be in angstrom.
    return depth


@njit
def _ctr_discrete_finite_intensity(qz, v_sum, c_ang, n_layers):
    """
    Finite discrete CTR:
      |sum_{l=0}^{N-1} exp(i (qz + iV) c l)|^2
    with V >= 0 and c in angstrom.
    """
    if n_layers <= 0 or c_ang <= 0.0:
        return 0.0

    v_eff = np.maximum(v_sum, 0.0)
    phase = qz * c_ang
    rho = np.exp(-v_eff * c_ang)
    rho_n = rho ** n_layers

    num = 1.0 - 2.0 * rho_n * np.cos(n_layers * phase) + rho_n * rho_n
    den = 1.0 - 2.0 * rho * np.cos(phase) + rho * rho
    if den < 1e-30:
        # Bragg-pole limit of the finite geometric series.
        return float(n_layers * n_layers)

    out = num / den
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    return out


@njit
def _ctr_discrete_infinite_intensity(qz, v_sum, c_ang):
    """
    Infinite discrete CTR:
      |sum_{l=0}^{inf} exp(i (qz + iV) c l)|^2
      = 1 / |1 - exp(i (qz + iV) c)|^2
    where V > 0 for convergence.
    """
    if c_ang <= 0.0:
        return 1.0

    # Keep convergence explicit even if v_sum is numerically zero.
    v_eff = np.maximum(v_sum, 1e-12 / np.maximum(c_ang, 1e-30))
    phase = qz * c_ang
    rho = np.exp(-v_eff * c_ang)

    den = 1.0 - 2.0 * rho * np.cos(phase) + rho * rho
    if den < 1e-30:
        den = 1e-30

    out = 1.0 / den
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    return out


@njit
def _ctr_effective_layers_from_absorption(v_sum, c_ang):
    """
    Semi-infinite fallback depth in layer units.

    For n_layers<=0, use an absorption-limited effective finite depth to avoid
    the undamped infinite-crystal pole in I(q,0), which otherwise drives the
    attenuation ratio to zero near q=0.
    """
    if c_ang <= 0.0:
        return 1

    vc = np.maximum(v_sum, 0.0) * c_ang
    if vc <= 1e-6:
        n_eff = 1000000
    else:
        n_eff = int(np.ceil(1.0 / vc))

    if n_eff < 1:
        return 1
    if n_eff > 1000000:
        return 1000000
    return n_eff


@njit
def _ctr_attenuation_factor(qz, v_in, v_out, c_ang, n_layers):
    """
    CTR-only absorption correction for fast mode.

    Use the same discrete CTR model with and without damping at the same qz:

      finite:    f(qz) = I_N(qz, V)   / I_N(qz, 0)
      semi-infinite fallback (n_layers<=0):
                 f(qz) = I_Neff(qz, V) / I_Neff(qz, 0)
                 with Neff set by absorption depth

    where V = v_in + v_out >= 0.
    """
    if c_ang <= 0.0:
        return 1.0

    v_sum = np.maximum(v_in + v_out, 0.0)
    if n_layers > 0:
        i_v = _ctr_discrete_finite_intensity(qz, v_sum, c_ang, n_layers)
        i_0 = _ctr_discrete_finite_intensity(qz, 0.0, c_ang, n_layers)
    else:
        # Avoid pole-driven collapse at qz~0 for an undamped infinite
        # denominator by using an absorption-limited effective finite depth.
        n_eff = _ctr_effective_layers_from_absorption(v_sum, c_ang)
        i_v = _ctr_discrete_finite_intensity(qz, v_sum, c_ang, n_eff)
        i_0 = _ctr_discrete_finite_intensity(qz, 0.0, c_ang, n_eff)

    if i_0 < 1e-30:
        i_0 = 1e-30

    out = i_v / i_0
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    return out


@njit
def transmit_angle_grazing(theta_i_plane, n2):
    """
    Grazing angle form of Snell: cos(theta_t) = cos(theta_i) / Re(n2)
    Inputs/outputs are grazing angles with respect to the surface plane.
    """
    c = np.cos(theta_i_plane) / np.maximum(np.real(n2), 1e-12)
    c = _clamp(c, -1.0, 1.0)
    # preserve sign of the incident grazing angle
    return np.arccos(c) * np.sign(theta_i_plane)


@njit
def ktz_components(k0, n2, theta_t_plane):
    """
    Decompose K_tz for a complex refractive index.
    Uses: a = Re(n^2) k0^2 sin^2(theta_t), b = Im(n^2) k0^2
    Returns positive magnitudes (sign handled at call site).
    """
    n2_sq = n2 * n2
    a = np.real(n2_sq) * (k0 * k0) * (np.sin(theta_t_plane) ** 2)
    b = np.abs(np.imag(n2_sq)) * (k0 * k0)     # magnitude for stability
    root = np.sqrt(a * a + b * b)
    re_kz = np.sqrt(0.5 * (root + a))
    im_kz = np.sqrt(0.5 * (root - a))
    return re_kz, im_kz


@njit
def safe_path_length(thickness_m, theta_plane):
    """
    Path length through a slab of thickness_m for a ray with grazing angle
    theta_plane (with respect to the plane). For semi-infinite thickness_m<=0,
    returns 0 to signal use of penetration depth.
    """
    if thickness_m <= 0.0:
        return 0.0
    s = np.abs(np.sin(theta_plane))
    if s < 1e-12:
        s = 1e-12
    return thickness_m / s


# =============================================================================
# 4) solve_q
# =============================================================================

@njit(fastmath=True)
def solve_q(
    k_in_crystal, k_scat, G_vec, sigma, gamma_pv, eta_pv, H, K, L,
    N_steps=1000,
    adaptive=True,
):
    """
    Build a 'circle' in reciprocal space for the reflection G_vec, i.e. the
    set of Q that satisfies |Q|=|G| or an intersection with Ewald sphere, then
    filter by mosaic surface density compute_intensity_array.

    Physically: 
      - We param by angle from 0..2π,
      - Circle radius circle_r,
      - Then for each Q on that circle, compute sigma(theta) and apply arc-length weighting.

    Returns
    -------
    out : ndarray of shape (M,4)
        For the valid points, columns = (Qx, Qy, Qz, mosaic_intensity).
    status : int
        0 for success or a negative code indicating the failure reason.
    """
    status = 0
    if N_steps <= 0:
        N_steps = DEFAULT_SOLVE_Q_STEPS
    if N_steps < 16:
        N_steps = 16

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

    # Adaptive path: start coarse and refine only arc bins that carry weight.
    use_adaptive = bool(adaptive) and (N_steps >= 128)
    if use_adaptive:
        coarse_steps = max(64, N_steps // 4)
        if coarse_steps > N_steps:
            coarse_steps = N_steps
    else:
        coarse_steps = N_steps

    if coarse_steps == DEFAULT_SOLVE_Q_STEPS:
        dtheta_coarse = _DEFAULT_SOLVE_Q_DTHETA
        cth_coarse = _DEFAULT_SOLVE_Q_COS
        sth_coarse = _DEFAULT_SOLVE_Q_SIN
    else:
        dtheta_coarse = 2.0 * np.pi / coarse_steps
        theta_arr_coarse = dtheta_coarse * np.arange(coarse_steps)
        cth_coarse = np.cos(theta_arr_coarse)
        sth_coarse = np.sin(theta_arr_coarse)

    Qx_coarse = Ox + circle_r * (cth_coarse * e1x + sth_coarse * e2x)
    Qy_coarse = Oy + circle_r * (cth_coarse * e1y + sth_coarse * e2y)
    Qz_coarse = Oz + circle_r * (cth_coarse * e1z + sth_coarse * e2z)

    sigma_coarse = compute_intensity_array(
        Qx_coarse,
        Qy_coarse,
        Qz_coarse,
        G_vec,
        sigma,
        gamma_pv,
        eta_pv,
    )
    all_int_coarse = sigma_coarse * (circle_r * dtheta_coarse)

    if not use_adaptive:
        Qx_arr = Qx_coarse
        Qy_arr = Qy_coarse
        Qz_arr = Qz_coarse
        all_int = all_int_coarse
    else:
        max_coarse_int = 0.0
        for i in range(all_int_coarse.size):
            if all_int_coarse[i] > max_coarse_int:
                max_coarse_int = all_int_coarse[i]
        adaptive_gate = max(_INTENSITY_CUTOFF, 0.02 * max_coarse_int)

        active_bins = np.zeros(coarse_steps, dtype=np.uint8)
        for i in range(coarse_steps):
            if all_int_coarse[i] > adaptive_gate:
                active_bins[i] = 1
                active_bins[(i - 1) % coarse_steps] = 1
                active_bins[(i + 1) % coarse_steps] = 1

        refine_factor = max(2, N_steps // coarse_steps)
        max_refined_pts = coarse_steps * refine_factor
        Qx_ref = np.empty(max_refined_pts, dtype=np.float64)
        Qy_ref = np.empty(max_refined_pts, dtype=np.float64)
        Qz_ref = np.empty(max_refined_pts, dtype=np.float64)
        dtheta_ref = np.empty(max_refined_pts, dtype=np.float64)

        n_ref = 0
        for i in range(coarse_steps):
            if active_bins[i] == 1:
                local_dtheta = dtheta_coarse / refine_factor
                for j in range(refine_factor):
                    theta_val = (i + (j / refine_factor)) * dtheta_coarse
                    ct = np.cos(theta_val)
                    st = np.sin(theta_val)
                    Qx_ref[n_ref] = Ox + circle_r * (ct * e1x + st * e2x)
                    Qy_ref[n_ref] = Oy + circle_r * (ct * e1y + st * e2y)
                    Qz_ref[n_ref] = Oz + circle_r * (ct * e1z + st * e2z)
                    dtheta_ref[n_ref] = local_dtheta
                    n_ref += 1
            else:
                theta_val = i * dtheta_coarse
                ct = np.cos(theta_val)
                st = np.sin(theta_val)
                Qx_ref[n_ref] = Ox + circle_r * (ct * e1x + st * e2x)
                Qy_ref[n_ref] = Oy + circle_r * (ct * e1y + st * e2y)
                Qz_ref[n_ref] = Oz + circle_r * (ct * e1z + st * e2z)
                dtheta_ref[n_ref] = dtheta_coarse
                n_ref += 1

        Qx_arr = Qx_ref[:n_ref]
        Qy_arr = Qy_ref[:n_ref]
        Qz_arr = Qz_ref[:n_ref]

        sigma_ref = compute_intensity_array(
            Qx_arr,
            Qy_arr,
            Qz_arr,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
        )
        all_int = sigma_ref * (circle_r * dtheta_ref[:n_ref])

    # Apply any intensity cutoff and construct the output.
    valid_idx = np.nonzero(all_int > _INTENSITY_CUTOFF)[0]
    out = np.zeros((valid_idx.size, 4), dtype=np.float64)
    for i in range(valid_idx.size):
        idx = valid_idx[i]
        out[i, 0] = Qx_arr[idx]
        out[i, 1] = Qy_arr[idx]
        out[i, 2] = Qz_arr[idx]
        out[i, 3] = all_int[idx]

    return out, status


@njit(fastmath=True)
def _solve_q_feasibility_status(k_in_crystal, k_scat, G_vec):
    """Cheap exact feasibility checks mirroring ``solve_q`` preconditions."""

    G_sq = G_vec[0]*G_vec[0] + G_vec[1]*G_vec[1] + G_vec[2]*G_vec[2]
    if G_sq < 1e-14:
        return -1

    Ax = -k_in_crystal[0]
    Ay = -k_in_crystal[1]
    Az = -k_in_crystal[2]
    A_sq = Ax*Ax + Ay*Ay + Az*Az
    if A_sq < 1e-14:
        return -2

    A_len = sqrt(A_sq)
    c = (G_sq + A_sq - k_scat*k_scat) / (2.0*A_len)
    circle_r_sq = G_sq - c*c
    if circle_r_sq < 0.0:
        return -3

    return 0


@njit(fastmath=True)
def _solve_q_circle_radius_sq(k_in_crystal, k_scat, G_vec):
    """Return the exact circle-radius-squared term used by solve_q geometry."""

    G_sq = G_vec[0]*G_vec[0] + G_vec[1]*G_vec[1] + G_vec[2]*G_vec[2]
    Ax = -k_in_crystal[0]
    Ay = -k_in_crystal[1]
    Az = -k_in_crystal[2]
    A_sq = Ax*Ax + Ay*Ay + Az*Az
    A_len = sqrt(np.maximum(A_sq, 1e-30))
    c = (G_sq + A_sq - k_scat*k_scat) / (2.0*A_len)
    return G_sq - c*c


@njit(fastmath=True)
def _quantize_component(value, inv_step):
    return int(np.rint(value * inv_step))


@njit(fastmath=True)
def _coarse_detector_coords(i_plane, kf_dir, detector_pos, n_det, det_e1, det_e2):
    dx, dy, dz, ok = intersect_line_plane(i_plane, kf_dir, detector_pos, n_det)
    if not ok:
        return 0.0, 0.0, False

    rx = dx - detector_pos[0]
    ry = dy - detector_pos[1]
    rz = dz - detector_pos[2]
    x_det = rx * det_e1[0] + ry * det_e1[1] + rz * det_e1[2]
    y_det = rx * det_e2[0] + ry * det_e2[1] + rz * det_e2[2]
    return x_det, y_det, True


@njit(fastmath=True)
def _nearest_pixel_boundary_margin(x_det, y_det, pixel_scale):
    cpx_f = x_det * pixel_scale
    rpx_f = -y_det * pixel_scale
    frac_c = np.abs(cpx_f - np.rint(cpx_f))
    frac_r = np.abs(rpx_f - np.rint(rpx_f))
    return 0.5 - max(frac_c, frac_r)

@njit(fastmath=True)
def _prepare_reflection_invariant_geometry(
    theta_initial_deg,
    cor_angle_deg,
    R_z_R_y,
    R_ZY_n,
    P0,
    theta_array,
    phi_array,
):
    """Build sample-geometry quantities reused by every reflection."""

    rad_theta_i = theta_initial_deg * (pi / 180.0)
    cor_axis_rad = cor_angle_deg * (pi / 180.0)

    ax = cos(cor_axis_rad)
    ay = 0.0
    az = sin(cor_axis_rad)
    axis_norm = sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-12:
        axis_norm = 1.0
    ax /= axis_norm
    ay /= axis_norm
    az /= axis_norm

    ct = cos(rad_theta_i)
    st = sin(rad_theta_i)
    one_ct = 1.0 - ct
    R_cor = np.array(
        [
            [ct + ax * ax * one_ct, ax * ay * one_ct - az * st, ax * az * one_ct + ay * st],
            [ay * ax * one_ct + az * st, ct + ay * ay * one_ct, ay * az * one_ct - ax * st],
            [az * ax * one_ct - ay * st, az * ay * one_ct + ax * st, ct + az * az * one_ct],
        ]
    )
    R_sample = R_cor @ R_z_R_y

    n_surf = R_cor @ R_ZY_n
    n_surf /= sqrt(n_surf[0] * n_surf[0] + n_surf[1] * n_surf[1] + n_surf[2] * n_surf[2])

    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0

    n_samp = theta_array.size
    best_idx = 0
    if n_samp > 0:
        best_angle = theta_array[0] * theta_array[0] + phi_array[0] * phi_array[0]
        for ii in range(1, n_samp):
            metric = theta_array[ii] * theta_array[ii] + phi_array[ii] * phi_array[ii]
            if metric < best_angle:
                best_angle = metric
                best_idx = ii

    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_norm = sqrt(e1_temp[0] * e1_temp[0] + e1_temp[1] * e1_temp[1] + e1_temp[2] * e1_temp[2])
    if e1_norm < 1e-12:
        alt_refs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        for ar in alt_refs:
            cross_tmp = np.cross(n_surf, ar)
            cross_norm_tmp = sqrt(
                cross_tmp[0] * cross_tmp[0]
                + cross_tmp[1] * cross_tmp[1]
                + cross_tmp[2] * cross_tmp[2]
            )
            if cross_norm_tmp > 1e-12:
                e1_temp = cross_tmp / cross_norm_tmp
                break
    else:
        e1_temp /= e1_norm

    e2_temp = np.cross(n_surf, e1_temp)
    return R_sample, n_surf, P0_rot, e1_temp, e2_temp, best_idx


# =============================================================================
# 5) CALCULATE_PHI
# =============================================================================

@njit(fastmath=True)
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
    R_x_detector, R_z_detector, n_det_rot, Detector_Pos,
    e1_det, e2_det,
    R_sample, n_surf, P0_rot, e1_temp, e2_temp,
    best_idx,
    use_exact_optics,
    n2_sq,
    n2_sq_real,
    unit_x,
    save_flag, q_data, q_count, i_peaks_index,
    record_status=False,
    thickness=0.0,
    optics_mode=OPTICS_MODE_FAST,
    forced_sample_idx=-1,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_adaptive=True,
    quantization_sigma_pixels=_QUANTIZATION_SIGMA_PIXELS,
    quantization_min_margin_pixels=_QUANTIZATION_MIN_MARGIN_PIXELS,
    centroid_shift_budget_px=_CENTROID_SHIFT_BUDGET_PX,
    fwhm_shift_budget_frac=_FWHM_SHIFT_BUDGET_FRAC,
    integrated_intensity_shift_budget_frac=_INTEGRATED_INTENSITY_SHIFT_BUDGET_FRAC,
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
    """
    hkl_inplane = H*H + H*K + K*K
    gz0 = 2.0*pi*(L/cv)
    gr0 = 4.0*pi/av * sqrt(hkl_inplane/3.0)
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

    beam_z_offsets = beam_y_array - zb
    cos_theta = np.cos(theta_array)
    sin_theta = np.sin(theta_array)
    cos_phi = np.cos(phi_array)
    sin_phi = np.sin(phi_array)
    k_in_x_arr = cos_theta * sin_phi
    k_in_y_arr = cos_theta * cos_phi
    k_in_z_arr = sin_theta
    k_mag_arr = (2.0 * pi) / wavelength_array
    k0_sq_arr = k_mag_arr * k_mag_arr
    debye_x_sq = debye_x * debye_x
    debye_y_sq = debye_y * debye_y
    pixel_scale = 1.0 / 100e-6
    n2_sq = n2 * n2
    n2_real = np.real(n2)
    inv_sqrt_eps = 1e-24

    # Geometry invariants for this pose (hoisted out of beam / Q loops).
    beam_start_y = -20e-3
    n_surf_x = n_surf[0]
    n_surf_y = n_surf[1]
    n_surf_z = n_surf[2]
    e1x = e1_temp[0]
    e1y = e1_temp[1]
    e1z = e1_temp[2]
    e2x = e2_temp[0]
    e2y = e2_temp[1]
    e2z = e2_temp[2]
    det_pos_x = Detector_Pos[0]
    det_pos_y = Detector_Pos[1]
    det_pos_z = Detector_Pos[2]
    det_e1x = e1_det[0]
    det_e1y = e1_det[1]
    det_e1z = e1_det[2]
    det_e2x = e2_det[0]
    det_e2y = e2_det[1]
    det_e2z = e2_det[2]
    det_nx = n_det_rot[0]
    det_ny = n_det_rot[1]
    det_nz = n_det_rot[2]
    r00 = R_sample[0, 0]
    r01 = R_sample[0, 1]
    r02 = R_sample[0, 2]
    r10 = R_sample[1, 0]
    r11 = R_sample[1, 1]
    r12 = R_sample[1, 2]
    r20 = R_sample[2, 0]
    r21 = R_sample[2, 1]
    r22 = R_sample[2, 2]

    max_hits = max(n_samp * 2, 16)  # Ensure capacity even for very small samples
    pixel_hits = np.empty((max_hits, 7), dtype=np.float64)
    missed_kf = np.empty((max_hits, 3), dtype=np.float64)
    n_hits = 0                     # running counter
    n_missed = 0

    # Track the strongest valid hit so we can fall back when the nominal
    # best beam sample does not intersect the detector.  This avoids losing
    # reflections such as 00L where only off-axis beam samples produce a hit.
    best_candidate = np.zeros(7, dtype=np.float64)
    best_candidate_val = -1.0
    have_candidate = False
    recorded_nominal_hit = False
    if record_status:
        statuses = np.zeros(n_samp, dtype=np.int64)

    # Preallocate small arrays used inside the loop to avoid dynamic
    # allocations when parallelizing with ``prange``.
    beam_start = np.empty(3, dtype=np.float64)
    k_in = np.empty(3, dtype=np.float64)
    I_plane = np.empty(3, dtype=np.float64)
    proj_incident = np.empty(3, dtype=np.float64)
    k_in_crystal = np.empty(3, dtype=np.float64)
    canonical_k_in = np.empty(3, dtype=np.float64)
    kf = np.empty(3, dtype=np.float64)
    kf_prime = np.empty(3, dtype=np.float64)
    plane_to_det = np.empty(3, dtype=np.float64)

    eps1 = 1.0 + 0.0j
    eps2 = n2_sq
    eps3 = 1.0 + 0.0j
    reflection_scale = reflection_intensity

    # Quantized detector-aware solve_q cache (stores only solve_q outputs + status).
    solve_q_cache = Dict.empty(key_type=_SOLVE_Q_CACHE_KEY_TYPE, value_type=_SOLVE_Q_CACHE_VAL_TYPE)
    solve_q_status_cache = Dict.empty(key_type=_SOLVE_Q_CACHE_KEY_TYPE, value_type=types.int64)
    det_proj_cache = Dict.empty(key_type=_DET_PROJ_CACHE_KEY_TYPE, value_type=_DET_PROJ_CACHE_VAL_TYPE)

    # Signature changes when detector/sample geometry changes; cache is local to
    # this reflection call, so this primarily documents detector-aware coupling.
    geometry_signature = (
        det_pos_x + det_pos_y + det_pos_z
        + n_surf_x + n_surf_y + n_surf_z
        + det_nx + det_ny + det_nz
        + det_e1x + det_e1y + det_e1z
        + det_e2x + det_e2y + det_e2z
        + r00 + r11 + r22
    )

    # Main loop over each beam sample in wave + mosaic. During fitting we can
    # optionally force a single preselected sample index per reflection.
    loop_start = 0
    loop_stop = n_samp
    if 0 <= forced_sample_idx < n_samp:
        loop_start = forced_sample_idx
        loop_stop = forced_sample_idx + 1

    best_candidate_sample_idx = -1

    # Quantized-cache diagnostics (logged when record_status=True).
    fallback_count = 0
    fallback_critical_count = 0
    fallback_branch_count = 0
    fallback_grazing_count = 0
    fallback_boundary_count = 0
    quantized_reuse_count = 0
    quantized_miss_count = 0
    quantized_store_count = 0
    quantized_candidate_count = 0
    sum_pred_centroid_shift = 0.0
    max_pred_centroid_shift = 0.0
    centroid_budget_fail_count = 0
    fwhm_budget_fail_count = 0
    intensity_budget_fail_count = 0
    det_proj_cache_hit_count = 0
    det_proj_cache_store_count = 0
    det_proj_cache_bypass_count = 0

    for i_samp in range(loop_start, loop_stop):
        k_mag = k_mag_arr[i_samp]
        k0_sq = k0_sq_arr[i_samp]

        beam_start[0] = beam_x_array[i_samp]
        beam_start[1] = beam_start_y
        beam_start[2] = beam_z_offsets[i_samp]

        k_in[0] = k_in_x_arr[i_samp]
        k_in[1] = k_in_y_arr[i_samp]
        k_in[2] = k_in_z_arr[i_samp]

        # Intersect the beam with the sample plane
        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if not valid_int:
            if record_status:
                statuses[i_samp] = -10
            continue

        I_plane[0] = ix
        I_plane[1] = iy
        I_plane[2] = iz
        kn_dot = k_in[0]*n_surf_x + k_in[1]*n_surf_y + k_in[2]*n_surf_z
        if kn_dot <= 1e-12:
            if record_status:
                statuses[i_samp] = -11
            continue
        th_i_prime = (pi/2.0) - acos(kn_dot)

        proj_incident[0] = k_in[0] - kn_dot*n_surf_x
        proj_incident[1] = k_in[1] - kn_dot*n_surf_y
        proj_incident[2] = k_in[2] - kn_dot*n_surf_z
        pln = sqrt(proj_incident[0]*proj_incident[0]
                 + proj_incident[1]*proj_incident[1]
                 + proj_incident[2]*proj_incident[2])
        if pln > 1e-12:
            proj_incident[0] /= pln
            proj_incident[1] /= pln
            proj_incident[2] /= pln
        else:
            proj_incident[0] = 0.0
            proj_incident[1] = 0.0
            proj_incident[2] = 0.0

        p1 = proj_incident[0]*e1x + proj_incident[1]*e1y + proj_incident[2]*e1z
        p2 = proj_incident[0]*e2x + proj_incident[1]*e2y + proj_incident[2]*e2z
        phi_i_prime = (pi/2.0) - np.arctan2(p2, p1)

        # ---------- ENTRY REFRACTION AND kz ----------
        k0 = k_mag

        if use_exact_optics:
            # Exact interface phase-matching uses conserved k_parallel.
            k_par_i = k0 * np.abs(np.cos(th_i_prime))
            k_par_i_sq = k_par_i * k_par_i

            kz1_i = _kz_branch_decay((k0_sq - k_par_i_sq) + 0.0j)
            kz2_i = _kz_branch_decay((eps2 * k0_sq) - k_par_i_sq)

            k_x_scat = k_par_i * np.sin(phi_i_prime)
            k_y_scat = k_par_i * np.cos(phi_i_prime)
            re_k_z = -np.abs(kz2_i.real)  # into sample
            im_k_z = np.abs(kz2_i.imag)
            k_scat = np.sqrt(np.maximum(k_par_i_sq + kz2_i.real * kz2_i.real, 0.0))

            Ti_s = _fresnel_t_exact(kz1_i, kz2_i, eps1, eps2, True)
            Ti_p = _fresnel_t_exact(kz1_i, kz2_i, eps1, eps2, False)
            Ti2 = 0.5 * (
                _fresnel_power_t_exact(Ti_s, kz1_i, kz2_i, eps1, eps2, True)
                + _fresnel_power_t_exact(Ti_p, kz1_i, kz2_i, eps1, eps2, False)
            )
            th_t = np.arctan2(np.abs(kz2_i.real), np.maximum(k_par_i, 1e-30))
        else:
            th_t = transmit_angle_grazing(th_i_prime, n2)     # grazing angle in medium
            # magnitude for in-plane internal wavevector (use Re(n^2))
            k_scat = k0 * np.sqrt(np.maximum(n2_sq_real, 0.0))
            k_x_scat = k_scat*np.cos(th_t)*np.sin(phi_i_prime)
            k_y_scat = k_scat*np.cos(th_t)*np.cos(phi_i_prime)

            # kz decomposition for the incident leg (sign convention: into sample)
            re_k_z, im_k_z = ktz_components(k0, n2, th_t)
            re_k_z = -re_k_z  # into the sample

            # Fresnel transmission at entry (amplitude -> intensity).
            # Use both s- and p-polarizations and average for an unpolarized beam.
            # fresnel_transmission signature: (grazing_angle, refractive_index,
            #                                s_polarization=True, incoming=True)
            Ti_s = fresnel_transmission(th_i_prime, n2, True, True)
            Ti_p = fresnel_transmission(th_i_prime, n2, False, True)
            Ti2 = 0.5 * (
                (np.real(Ti_s)*np.real(Ti_s) + np.imag(Ti_s)*np.imag(Ti_s))
                + (np.real(Ti_p)*np.real(Ti_p) + np.imag(Ti_p)*np.imag(Ti_p))
            )

        k_in_crystal[0] = k_x_scat
        k_in_crystal[1] = k_y_scat
        k_in_crystal[2] = re_k_z

        # Deterministic early rejection before entering solve_q.
        # These checks are exact mirror preconditions from solve_q.
        stat = _solve_q_feasibility_status(k_in_crystal, k_scat, G_vec)
        if stat != 0:
            if record_status:
                statuses[i_samp] = stat
            continue

        # ---------- Solve allowed Q on the circle (detector-aware quantized cache) ----------
        mirror_x = k_in_crystal[0] < 0.0
        mirror_y = k_in_crystal[1] < 0.0
        canonical_k_in[0] = np.abs(k_in_crystal[0])
        canonical_k_in[1] = np.abs(k_in_crystal[1])
        canonical_k_in[2] = k_in_crystal[2]

        # Local detector-aware Jacobian for k_in -> detector (u,v in pixel units).
        # This makes quantization anisotropic and orientation-sensitive.
        kf_cx = canonical_k_in[0] + G_vec[0]
        kf_cy = canonical_k_in[1] + G_vec[1]
        kf_cz = canonical_k_in[2] + G_vec[2]
        kf_prime[0] = r00*kf_cx + r01*kf_cy + r02*kf_cz
        kf_prime[1] = r10*kf_cx + r11*kf_cy + r12*kf_cz
        kf_prime[2] = r20*kf_cx + r21*kf_cy + r22*kf_cz

        x0_det, y0_det, coarse_ok = _coarse_detector_coords(
            I_plane, kf_prime, Detector_Pos, n_det_rot, e1_det, e2_det
        )

        delta_k = np.maximum(1e-6, 1e-4 * np.maximum(k_scat, 1.0))
        sensitivity_kx = 1e-12
        sensitivity_ky = 1e-12
        boundary_margin = 0.5

        # Automatic exact fallback triggers in sensitive regions.
        # 1) near critical-angle optics (internal normal component nearly zero)
        critical_kz_limit = _NEAR_CRITICAL_KZ_RATIO * np.maximum(k_scat, 1.0)
        near_critical_optics = np.abs(re_k_z) < critical_kz_limit

        # 2) near solve_q branch transition (circle radius tends to zero)
        circle_r_sq = _solve_q_circle_radius_sq(k_in_crystal, k_scat, G_vec)
        scale_sq = np.maximum(
            G_vec[0]*G_vec[0] + G_vec[1]*G_vec[1] + G_vec[2]*G_vec[2],
            k_scat*k_scat,
        )
        near_solve_q_branch = np.abs(circle_r_sq) < (_NEAR_SOLVE_Q_BRANCH_EPS * np.maximum(scale_sq, 1.0))

        # 3) near detector grazing intersections
        kf_norm = sqrt(kf_prime[0]*kf_prime[0] + kf_prime[1]*kf_prime[1] + kf_prime[2]*kf_prime[2])
        det_dot = np.abs(kf_prime[0]*det_nx + kf_prime[1]*det_ny + kf_prime[2]*det_nz)
        near_detector_grazing = det_dot < (_DETECTOR_GRAZING_COS_EPS * np.maximum(kf_norm, 1e-30))

        if coarse_ok:
            # Direction perturbation columns in the lab frame.
            kf[0] = kf_prime[0] + delta_k * r00
            kf[1] = kf_prime[1] + delta_k * r10
            kf[2] = kf_prime[2] + delta_k * r20
            x1_det, y1_det, ok_x = _coarse_detector_coords(
                I_plane, kf, Detector_Pos, n_det_rot, e1_det, e2_det
            )
            if ok_x:
                du_dkx = ((x1_det - x0_det) * pixel_scale) / delta_k
                dv_dkx = ((-y1_det + y0_det) * pixel_scale) / delta_k
                sensitivity_kx = sqrt(du_dkx*du_dkx + dv_dkx*dv_dkx)

            kf[0] = kf_prime[0] + delta_k * r01
            kf[1] = kf_prime[1] + delta_k * r11
            kf[2] = kf_prime[2] + delta_k * r21
            x2_det, y2_det, ok_y = _coarse_detector_coords(
                I_plane, kf, Detector_Pos, n_det_rot, e1_det, e2_det
            )
            if ok_y:
                du_dky = ((x2_det - x0_det) * pixel_scale) / delta_k
                dv_dky = ((-y2_det + y0_det) * pixel_scale) / delta_k
                sensitivity_ky = sqrt(du_dky*du_dky + dv_dky*dv_dky)

            boundary_margin = _nearest_pixel_boundary_margin(x0_det, y0_det, pixel_scale)

        step_kx = quantization_sigma_pixels / np.maximum(sensitivity_kx, 1e-12)
        step_ky = quantization_sigma_pixels / np.maximum(sensitivity_ky, 1e-12)
        q_idx_x = _quantize_component(canonical_k_in[0], 1.0 / np.maximum(step_kx, 1e-18))
        q_idx_y = _quantize_component(canonical_k_in[1], 1.0 / np.maximum(step_ky, 1e-18))
        q_idx_z = _quantize_component(canonical_k_in[2], 1e6)
        q_idx_ks = _quantize_component(k_scat, 1e6)
        h_i = int(np.rint(H))
        k_i = int(np.rint(K))
        l_i = int(np.rint(L))
        sigma_i = _quantize_component(sigma_rad, 1e9)
        gamma_i = _quantize_component(gamma_pv, 1e9)
        eta_i = _quantize_component(eta_pv, 1e9)
        step_i = int(solve_q_steps)
        adapt_i = 1 if solve_q_adaptive else 0
        geo_i = _quantize_component(geometry_signature, 1e6)

        solve_q_key = (
            q_idx_x,
            q_idx_y,
            q_idx_z,
            q_idx_ks,
            h_i,
            k_i,
            l_i,
            sigma_i,
            gamma_i,
            eta_i,
            step_i,
            adapt_i,
            geo_i,
        )

        can_reuse_quantized = False
        predicted_shift = 1e12
        predicted_fwhm_shift_frac = 1e12
        predicted_intensity_shift_frac = 1e12
        near_pixel_boundary = True
        half_step_kx = 0.5 * step_kx
        half_step_ky = 0.5 * step_ky
        if coarse_ok:
            predicted_shift = sqrt(
                (sensitivity_kx * half_step_kx) * (sensitivity_kx * half_step_kx)
                + (sensitivity_ky * half_step_ky) * (sensitivity_ky * half_step_ky)
            )
            near_pixel_boundary = predicted_shift >= (boundary_margin - quantization_min_margin_pixels)
            predicted_fwhm_shift_frac = predicted_shift / np.maximum(_FWHM_REFERENCE_PIXELS, 1e-12)
            predicted_intensity_shift_frac = predicted_fwhm_shift_frac

        over_centroid_budget = predicted_shift > centroid_shift_budget_px
        over_fwhm_budget = predicted_fwhm_shift_frac > fwhm_shift_budget_frac
        over_intensity_budget = predicted_intensity_shift_frac > integrated_intensity_shift_budget_frac

        sensitive_region = (
            near_critical_optics
            or near_solve_q_branch
            or near_detector_grazing
            or near_pixel_boundary
            or over_centroid_budget
            or over_fwhm_budget
            or over_intensity_budget
        )
        if coarse_ok and not sensitive_region:
            can_reuse_quantized = True

        if coarse_ok:
            quantized_candidate_count += 1
            sum_pred_centroid_shift += predicted_shift
            if predicted_shift > max_pred_centroid_shift:
                max_pred_centroid_shift = predicted_shift
            if over_centroid_budget:
                centroid_budget_fail_count += 1
            if over_fwhm_budget:
                fwhm_budget_fail_count += 1
            if over_intensity_budget:
                intensity_budget_fail_count += 1

        if can_reuse_quantized and solve_q_key in solve_q_cache:
            quantized_reuse_count += 1
            all_q_canonical = solve_q_cache[solve_q_key]
            stat = solve_q_status_cache[solve_q_key]
        else:
            if can_reuse_quantized:
                quantized_miss_count += 1
            all_q_canonical, stat = solve_q(
                canonical_k_in,
                k_scat,
                G_vec,
                sigma_rad,
                gamma_pv,
                eta_pv,
                H,
                K,
                L,
                solve_q_steps,
                solve_q_adaptive,
            )
            if can_reuse_quantized:
                solve_q_cache[solve_q_key] = all_q_canonical
                solve_q_status_cache[solve_q_key] = stat
                quantized_store_count += 1
            else:
                fallback_count += 1
                if near_critical_optics:
                    fallback_critical_count += 1
                if near_solve_q_branch:
                    fallback_branch_count += 1
                if near_detector_grazing:
                    fallback_grazing_count += 1
                if near_pixel_boundary:
                    fallback_boundary_count += 1

        if mirror_x or mirror_y:
            All_Q = all_q_canonical.copy()
            if mirror_x:
                All_Q[:, 0] *= -1.0
            if mirror_y:
                All_Q[:, 1] *= -1.0
        else:
            All_Q = all_q_canonical

        if record_status:
            statuses[i_samp] = stat
        if stat != 0 or All_Q.shape[0] == 0:
            continue

        # Precompute entrance attenuation depth.
        if use_exact_optics:
            if thickness > 0.0:
                L_in = thickness
            else:
                L_in = 1.0 / np.maximum(2.0 * im_k_z, 1e-30)
        else:
            L_in = safe_path_length(thickness, th_t)
            if L_in <= 0.0:
                # Semi-infinite: use effective penetration depth
                L_in = 1.0 / np.maximum(2.0*im_k_z, 1e-30)

        # ---------- Loop over each Q solution ----------
        if use_exact_optics:
            k_out_mag = k0
        else:
            k_out_mag = k_scat

        for i_sol in range(All_Q.shape[0]):
            Qx = All_Q[i_sol, 0]
            Qy = All_Q[i_sol, 1]
            Qz = All_Q[i_sol, 2]
            I_Q = All_Q[i_sol, 3]
            if I_Q < 1e-5:
                continue

            # internal scattered direction before exiting
            k_tx_prime = Qx + k_x_scat
            k_ty_prime = Qy + k_y_scat
            k_tz_prime = Qz + re_k_z

            kr_sq = k_tx_prime*k_tx_prime + k_ty_prime*k_ty_prime
            sign_t = 1.0
            if k_tz_prime < 0.0:
                sign_t = -1.0

            # refract out to air: convert internal grazing angle to external
            if use_exact_optics:
                kz2_f = _kz_branch_decay((eps2 * k0_sq) - kr_sq)
                kz3_f = _kz_branch_decay((k0_sq - kr_sq) + 0.0j)

                Tf_s = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, True)
                Tf_p = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, False)
                Tf2 = 0.5 * (
                    _fresnel_power_t_exact(Tf_s, kz2_f, kz3_f, eps2, eps3, True)
                    + _fresnel_power_t_exact(Tf_p, kz2_f, kz3_f, eps2, eps3, False)
                )

                im_k_z_f = np.abs(kz2_f.imag)
                if thickness > 0.0:
                    L_out = thickness
                else:
                    L_out = 1.0 / np.maximum(2.0 * im_k_z_f, 1e-30)

                if kr_sq <= inv_sqrt_eps:
                    k_tx_f = 0.0
                    k_ty_f = 0.0
                else:
                    k_tx_f = k_tx_prime
                    k_ty_f = k_ty_prime
                kz_out_sq = np.maximum(k0_sq - kr_sq, 0.0)
                k_tz_f = sign_t * sqrt(kz_out_sq)
            else:
                if kr_sq <= inv_sqrt_eps:
                    th_t_out = pi * 0.5
                    cos_in = 0.0
                else:
                    kr = sqrt(kr_sq)
                    inv_k_t = 1.0 / sqrt(kr_sq + k_tz_prime*k_tz_prime)
                    cos_in = kr * inv_k_t
                    th_t_out = np.arctan(np.abs(k_tz_prime) / np.maximum(kr, 1e-30))

                # Exit transmission amplitude and kz for exit leg.
                # Use both polarizations and average for an unpolarized beam.
                Tf_s = fresnel_transmission(th_t_out, n2, True, False)
                Tf_p = fresnel_transmission(th_t_out, n2, False, False)
                Tf2 = 0.5 * (
                    (np.real(Tf_s)*np.real(Tf_s) + np.imag(Tf_s)*np.imag(Tf_s))
                    + (np.real(Tf_p)*np.real(Tf_p) + np.imag(Tf_p)*np.imag(Tf_p))
                )

                re_k_z_f, im_k_z_f = ktz_components(k0, n2, th_t_out)

                # path length for exit leg
                L_out = safe_path_length(thickness, th_t_out)
                if L_out <= 0.0:
                    L_out = 1.0 / np.maximum(2.0*im_k_z_f, 1e-30)

                cos_out = _clamp(cos_in * n2_real, -1.0, 1.0)
                sin_out = sign_t * sqrt(np.maximum(1.0 - cos_out*cos_out, 0.0))
                if kr_sq <= inv_sqrt_eps:
                    k_tx_f = 0.0
                    k_ty_f = 0.0
                else:
                    kr = sqrt(kr_sq)
                    out_xy_scale = (k_out_mag * cos_out) / np.maximum(kr, 1e-30)
                    k_tx_f = out_xy_scale * k_tx_prime
                    k_ty_f = out_xy_scale * k_ty_prime
                k_tz_f = k_out_mag * sin_out

            # apply propagation and interface factors
            prop_att = (
                np.exp(-2.0 * im_k_z * L_in)
                * np.exp(-2.0 * im_k_z_f * L_out)
            )
            prop_fac = Ti2 * Tf2 * prop_att
            sample_scale = reflection_scale * prop_fac
            kf[0] = k_tx_f
            kf[1] = k_ty_f
            kf[2] = k_tz_f
            kf_prime[0] = r00*kf[0] + r01*kf[1] + r02*kf[2]
            kf_prime[1] = r10*kf[0] + r11*kf[1] + r12*kf[2]
            kf_prime[2] = r20*kf[0] + r21*kf[1] + r22*kf[2]

            # Detector-projection cache (separate from solve_q cache) with stricter
            # geometry-sensitive reuse guards.
            proj_x, proj_y, proj_ok = _coarse_detector_coords(
                I_plane, kf_prime, Detector_Pos, n_det_rot, e1_det, e2_det
            )
            kf_norm = sqrt(kf_prime[0]*kf_prime[0] + kf_prime[1]*kf_prime[1] + kf_prime[2]*kf_prime[2])
            det_cos = np.abs(kf_prime[0]*det_nx + kf_prime[1]*det_ny + kf_prime[2]*det_nz) / np.maximum(kf_norm, 1e-30)
            coarse_margin = 0.0
            if proj_ok:
                coarse_margin = _nearest_pixel_boundary_margin(proj_x, proj_y, pixel_scale)
            allow_det_proj_reuse = (
                proj_ok
                and det_cos > _DET_PROJ_GRAZING_COS_EPS
                and coarse_margin > _DET_PROJ_MIN_BOUNDARY_MARGIN_PX
            )

            if allow_det_proj_reuse:
                det_proj_key = (
                    _quantize_component(I_plane[0], 1e7),
                    _quantize_component(I_plane[1], 1e7),
                    _quantize_component(I_plane[2], 1e7),
                    _quantize_component(kf_prime[0], 1e8),
                    _quantize_component(kf_prime[1], 1e8),
                    _quantize_component(kf_prime[2], 1e8),
                    _quantize_component(det_pos_x + det_pos_y + det_pos_z, 1e6),
                    _quantize_component(det_nx + det_ny + det_nz, 1e6),
                    _quantize_component(det_e1x + det_e1y + det_e1z, 1e6),
                    _quantize_component(det_e2x + det_e2y + det_e2z, 1e6),
                )
                if det_proj_key in det_proj_cache:
                    det_proj_cache_hit_count += 1
                    det_entry = det_proj_cache[det_proj_key]
                    valid_det = det_entry[0] > 0.5
                    x_det = det_entry[1]
                    y_det = det_entry[2]
                else:
                    dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
                    if valid_det:
                        plane_to_det[0] = dx - det_pos_x
                        plane_to_det[1] = dy - det_pos_y
                        plane_to_det[2] = dz - det_pos_z
                        x_det = plane_to_det[0]*det_e1x + plane_to_det[1]*det_e1y + plane_to_det[2]*det_e1z
                        y_det = plane_to_det[0]*det_e2x + plane_to_det[1]*det_e2y + plane_to_det[2]*det_e2z
                    else:
                        x_det = 0.0
                        y_det = 0.0
                    det_proj_cache[det_proj_key] = np.array([
                        1.0 if valid_det else 0.0,
                        x_det,
                        y_det,
                    ], dtype=np.float64)
                    det_proj_cache_store_count += 1
            else:
                det_proj_cache_bypass_count += 1
                dx, dy, dz, valid_det = intersect_line_plane(I_plane, kf_prime, Detector_Pos, n_det_rot)
                if valid_det:
                    plane_to_det[0] = dx - det_pos_x
                    plane_to_det[1] = dy - det_pos_y
                    plane_to_det[2] = dz - det_pos_z
                    x_det = plane_to_det[0]*det_e1x + plane_to_det[1]*det_e1y + plane_to_det[2]*det_e1z
                    y_det = plane_to_det[0]*det_e2x + plane_to_det[1]*det_e2y + plane_to_det[2]*det_e2z
                else:
                    x_det = 0.0
                    y_det = 0.0

            if not valid_det:
                if n_missed < max_hits:
                    missed_kf[n_missed, 0] = kf_prime[0]
                    missed_kf[n_missed, 1] = kf_prime[1]
                    missed_kf[n_missed, 2] = kf_prime[2]
                    n_missed += 1
                continue

            rpx = int(round(center[0] - y_det * pixel_scale))
            cpx = int(round(center[1] + x_det * pixel_scale))
            if not (0 <= rpx < image_size and 0 <= cpx < image_size):
                continue

            # Combine:
            #  1) reflection_intensity -> structure/basis factor
            #  2) I_Q -> partial mosaic weighting from solve_q circle
            #  3) incoherent -> the mosaic average from Q_grid
            #  4) exponent dampings -> Debye or extra broadening
            val = (sample_scale * I_Q
                * exp(-Qz * Qz * debye_x_sq)
                * exp(-(Qx * Qx + Qy * Qy) * debye_y_sq))
            image[rpx, cpx] += val

            phi_f = np.arctan2(k_tx_prime, k_ty_prime)

            if val > best_candidate_val:
                best_candidate_val = val
                best_candidate[0] = val
                best_candidate[1] = cpx
                best_candidate[2] = rpx
                best_candidate[3] = phi_f
                best_candidate[4] = H
                best_candidate[5] = K
                best_candidate[6] = L
                have_candidate = True
                best_candidate_sample_idx = i_samp

            if i_samp == best_idx:
                if n_hits < max_hits:
                    # save       I_Q⋅extras        x-pix     y-pix      φf
                    pixel_hits[n_hits, 0] = val
                    pixel_hits[n_hits, 1] = cpx
                    pixel_hits[n_hits, 2] = rpx
                    pixel_hits[n_hits, 3] = phi_f
                    pixel_hits[n_hits, 4] = H
                    pixel_hits[n_hits, 5] = K
                    pixel_hits[n_hits, 6] = L

                    n_hits += 1
                recorded_nominal_hit = True
                
            # Optionally store Q-data
            if save_flag==1 and q_count[i_peaks_index]< q_data.shape[1]:
                idx = q_count[i_peaks_index]
                q_data[i_peaks_index, idx,0] = Qx
                q_data[i_peaks_index, idx,1] = Qy
                q_data[i_peaks_index, idx,2] = Qz
                q_data[i_peaks_index, idx,3] = val
                q_count[i_peaks_index]+=1

    add_candidate = False
    if have_candidate:
        add_candidate = not recorded_nominal_hit
        if not add_candidate:
            duplicate = False
            for idx in range(n_hits):
                if (
                    abs(pixel_hits[idx, 1] - best_candidate[1]) < 0.5
                    and abs(pixel_hits[idx, 2] - best_candidate[2]) < 0.5
                ):
                    duplicate = True
                    break
            if not duplicate:
                add_candidate = True
    if add_candidate and n_hits < max_hits:
        pixel_hits[n_hits, :] = best_candidate
        n_hits += 1

    best_sample_idx = best_idx
    if best_candidate_sample_idx >= 0:
        best_sample_idx = best_candidate_sample_idx

    if record_status:
        mean_pred_centroid_shift = 0.0
        if quantized_candidate_count > 0:
            mean_pred_centroid_shift = sum_pred_centroid_shift / quantized_candidate_count

        print(
            "quantized_cache_stats",
            H,
            K,
            L,
            quantized_reuse_count,
            quantized_miss_count,
            quantized_store_count,
            fallback_count,
            fallback_critical_count,
            fallback_branch_count,
            fallback_grazing_count,
            fallback_boundary_count,
            mean_pred_centroid_shift,
            max_pred_centroid_shift,
            centroid_budget_fail_count,
            fwhm_budget_fail_count,
            intensity_budget_fail_count,
            det_proj_cache_hit_count,
            det_proj_cache_store_count,
            det_proj_cache_bypass_count,
        )
        return pixel_hits[:n_hits], statuses, missed_kf[:n_missed], best_sample_idx
    else:
        return (
            pixel_hits[:n_hits],
            np.empty(0, dtype=np.int64),
            missed_kf[:n_missed],
            best_sample_idx,
        )



# =============================================================================
# 6) PROCESS_PEAKS_PARALLEL
# =============================================================================

@njit(parallel=True, fastmath=True)
def process_peaks_parallel(
    miller, intensities, image_size,
    av, cv, lambda_, image,
    Distance_CoR_to_Detector, gamma_deg, Gamma_deg, chi_deg, psi_deg, psi_z_deg,
    zs, zb, n2,
    beam_x_array, beam_y_array,
    theta_array, phi_array,
    sigma_pv_deg, gamma_pv_deg, eta_pv,
    wavelength_array,
    debye_x, debye_y, center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x, n_detector,
    save_flag,
    record_status=False,
    thickness=50e-9,
    optics_mode=OPTICS_MODE_FAST,
    single_sample_indices=None,
    best_sample_indices_out=None,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_adaptive=True,
    q_data_max_solutions=0,
    quantization_sigma_pixels=_QUANTIZATION_SIGMA_PIXELS,
    quantization_min_margin_pixels=_QUANTIZATION_MIN_MARGIN_PIXELS,
    centroid_shift_budget_px=_CENTROID_SHIFT_BUDGET_PX,
    fwhm_shift_budget_frac=_FWHM_SHIFT_BUDGET_FRAC,
    integrated_intensity_shift_budget_frac=_INTEGRATED_INTENSITY_SHIFT_BUDGET_FRAC,
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
    If save_flag==1, also returns q_data, q_count with detailed Q sampling info.
    Otherwise, returns just the updated image and max_positions for each reflection.
    """
    gamma_rad = gamma_deg*(pi/180.0)
    Gamma_rad = Gamma_deg*(pi/180.0)
    chi_rad   = chi_deg*(pi/180.0)
    psi_rad   = psi_deg*(pi/180.0)
    psi_z_rad = psi_z_deg*(pi/180.0)

    sigma_rad   = sigma_pv_deg*(pi/180.0)
    gamma_rad_m = gamma_pv_deg*(pi/180.0)

    # Build transforms for the detector
    cg = cos(gamma_rad)
    sg = sin(gamma_rad)
    cG = cos(Gamma_rad)
    sG = sin(Gamma_rad)
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

    c_chi = cos(chi_rad)
    s_chi = sin(chi_rad)
    R_y = np.array([
        [ c_chi, 0.0,   s_chi],
        [ 0.0,   1.0,   0.0],
        [-s_chi, 0.0, c_chi]
    ])
    c_psi= cos(psi_rad)
    s_psi= sin(psi_rad)
    R_z = np.array([
        [ c_psi, s_psi, 0.0],
        [-s_psi, c_psi, 0.0],
        [ 0.0,   0.0,   1.0]
    ])
    c_psi_z = cos(psi_z_rad)
    s_psi_z = sin(psi_z_rad)
    R_z_gonio = np.array([
        [ c_psi_z, s_psi_z, 0.0],
        [-s_psi_z, c_psi_z, 0.0],
        [ 0.0,     0.0,     1.0]
    ])
    R_z_R_y = (R_z_gonio @ R_z) @ R_y

    n1= np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R_ZY_n= R_z_R_y @ n1
    nzy_len= sqrt(R_ZY_n[0]*R_ZY_n[0] + R_ZY_n[1]*R_ZY_n[1] + R_ZY_n[2]*R_ZY_n[2])
    R_ZY_n/= nzy_len

    P0= np.array([0.0, 0.0, -zs], dtype=np.float64)
    num_peaks= miller.shape[0]

    max_solutions = int(q_data_max_solutions)
    if max_solutions <= 0:
        est_steps = int(max(16, solve_q_steps))
        est_total = est_steps * max(1, beam_x_array.size)
        if solve_q_adaptive:
            est_total = max(est_steps, est_total // 2)
        max_solutions = min(max(est_total, 1024), 200000)
    if save_flag==1:
        q_data= np.full((num_peaks, max_solutions, 5), np.nan, dtype=np.float64)
        q_count= np.zeros(num_peaks, dtype=np.int64)
    else:
        q_data= np.zeros((1,1,5), dtype=np.float64)
        q_count= np.zeros(1, dtype=np.int64)
    hit_tables = List.empty_list(types.float64[:, ::1])
    miss_tables = List.empty_list(types.float64[:, ::1])
    for _ in range(num_peaks):
        hit_tables.append(np.empty((0, 7), dtype=np.float64))
        miss_tables.append(np.empty((0, 3), dtype=np.float64))
    all_status = np.zeros((num_peaks, beam_x_array.size), dtype=np.int64)

    # Use per-thread image accumulation when affordable, then reduce once.
    # This avoids concurrent scatter-add contention on the shared detector image.
    n_threads = get_num_threads()
    use_thread_local_image = False
    image_partials = np.zeros((1, 1, 1), dtype=np.float64)
    if n_threads > 1 and num_peaks > 1 and image_size > 0:
        bytes_per_image = image_size * image_size * 8
        total_bytes = n_threads * bytes_per_image
        if total_bytes <= _THREAD_LOCAL_IMAGE_MAX_BYTES:
            image_partials = np.zeros((n_threads, image_size, image_size), dtype=np.float64)
            use_thread_local_image = True

    R_sample, n_surf, P0_rot, e1_temp, e2_temp, best_idx = _prepare_reflection_invariant_geometry(
        theta_initial_deg,
        cor_angle_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
        theta_array,
        phi_array,
    )
    use_exact_optics = optics_mode == OPTICS_MODE_EXACT
    n2_sq = n2 * n2
    n2_sq_real = np.real(n2_sq)

    # prange over each reflection
    for i_pk in prange(num_peaks):
        # Ensure HKL values remain floating point to allow fractional indices
        H = float(miller[i_pk, 0])
        K = float(miller[i_pk, 1])
        L = float(miller[i_pk, 2])
        if L < 0.0:
            continue
        reflI= intensities[i_pk]

        # We'll do a reflection-level call to calculate_phi
        forced_idx = -1
        if single_sample_indices is not None:
            if i_pk < single_sample_indices.shape[0]:
                forced_idx = int(single_sample_indices[i_pk])

        if use_thread_local_image:
            tid = get_thread_id()
            if 0 <= tid < image_partials.shape[0]:
                pixel_hits, status_arr, missed_arr, best_sample_idx_out = calculate_phi(
                    H, K, L, av, cv,
                    wavelength_array,
                    image_partials[tid], image_size,
                    gamma_rad, Gamma_rad, chi_rad, psi_rad,
                    zs, zb, n2,
                    beam_x_array, beam_y_array,
                    theta_array, phi_array,
                    reflI, sigma_rad, gamma_rad_m, eta_pv,
                    debye_x, debye_y,
                    center,
                    R_x_det, R_z_det, n_det_rot, Detector_Pos,
                    e1_det, e2_det,
                    R_sample, n_surf, P0_rot, e1_temp, e2_temp,
                    best_idx,
                    use_exact_optics,
                    n2_sq,
                    n2_sq_real,
                    unit_x,
                    save_flag, q_data, q_count, i_pk,
                    record_status,
                    thickness,
                    optics_mode,
                    forced_idx,
                    solve_q_steps,
                    solve_q_adaptive,
                    quantization_sigma_pixels,
                    quantization_min_margin_pixels,
                    centroid_shift_budget_px,
                    fwhm_shift_budget_frac,
                    integrated_intensity_shift_budget_frac,
                )
            else:
                pixel_hits, status_arr, missed_arr, best_sample_idx_out = calculate_phi(
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
                    R_x_det, R_z_det, n_det_rot, Detector_Pos,
                    e1_det, e2_det,
                    R_sample, n_surf, P0_rot, e1_temp, e2_temp,
                    best_idx,
                    use_exact_optics,
                    n2_sq,
                    n2_sq_real,
                    unit_x,
                    save_flag, q_data, q_count, i_pk,
                    record_status,
                    thickness,
                    optics_mode,
                    forced_idx,
                    solve_q_steps,
                    solve_q_adaptive,
                    quantization_sigma_pixels,
                    quantization_min_margin_pixels,
                    centroid_shift_budget_px,
                    fwhm_shift_budget_frac,
                    integrated_intensity_shift_budget_frac,
                )
        else:
            pixel_hits, status_arr, missed_arr, best_sample_idx_out = calculate_phi(
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
                R_x_det, R_z_det, n_det_rot, Detector_Pos,
                e1_det, e2_det,
                R_sample, n_surf, P0_rot, e1_temp, e2_temp,
                best_idx,
                use_exact_optics,
                n2_sq,
                n2_sq_real,
                unit_x,
                save_flag, q_data, q_count, i_pk,
                record_status,
                thickness,
                optics_mode,
                forced_idx,
                solve_q_steps,
                solve_q_adaptive,
                quantization_sigma_pixels,
                quantization_min_margin_pixels,
                centroid_shift_budget_px,
                fwhm_shift_budget_frac,
                integrated_intensity_shift_budget_frac,
            )
        if record_status:
            all_status[i_pk, :] = status_arr
        hit_tables[i_pk] = pixel_hits
        miss_tables[i_pk] = missed_arr
        if best_sample_indices_out is not None:
            if i_pk < best_sample_indices_out.shape[0]:
                best_sample_indices_out[i_pk] = best_sample_idx_out

    if use_thread_local_image:
        for tid in range(image_partials.shape[0]):
            image += image_partials[tid]
    return image, hit_tables, q_data, q_count, all_status, miss_tables


def process_peaks_parallel_safe(*args, **kwargs):
    """Run ``process_peaks_parallel`` with Python fallback if JIT execution fails."""

    try:
        return process_peaks_parallel(*args, **kwargs)
    except Exception:
        py_runner = getattr(process_peaks_parallel, "py_func", None)
        if callable(py_runner):
            return py_runner(*args, **kwargs)
        raise


def hit_tables_to_max_positions(hit_tables):
    """Extract top-2 peak locations per reflection from ``hit_tables``.

    ``process_peaks_parallel`` returns a list of pixel-hit tables, each with
    columns ``[intensity, col, row, phi, H, K, L]``.  Older callers expect a
    ``max_positions`` array shaped ``(N, 6)`` containing the brightest two hit
    locations per reflection: ``(I0, x0, y0, I1, x1, y1)``.  This helper
    restores that structure so downstream code can remain unchanged.
    """

    num_peaks = len(hit_tables)
    max_positions = np.zeros((num_peaks, 6), dtype=np.float64)

    for i, hits in enumerate(hit_tables):
        hits_arr = np.asarray(hits)
        if hits_arr.size == 0:
            continue

        # Sort by intensity descending and take the top two
        sorted_hits = hits_arr[np.argsort(hits_arr[:, 0])[::-1]]
        primary = sorted_hits[0]
        max_positions[i, 0:3] = primary[0:3]

        if sorted_hits.shape[0] > 1:
            secondary = sorted_hits[1]
            max_positions[i, 3:6] = secondary[0:3]

    return max_positions


def process_qr_rods_parallel(
    qr_dict,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    record_status=False,
    thickness=0.0,
    optics_mode=OPTICS_MODE_FAST,
    single_sample_indices=None,
    best_sample_indices_out=None,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_adaptive=True,
    q_data_max_solutions=0,
):
    """Wrapper to process Hendricks–Teller rods instead of individual reflections.

    The Hendricks–Teller preprocessing groups symmetry-related in-plane peaks
    into ``Qr`` rods and records how many peaks contributed to each rod in the
    ``deg`` field.  ``qr_dict_to_arrays`` already returns rod intensities as the
    total summed intensity over the grouped HK pairs, so we forward that array
    unchanged to avoid double-counting.  The degeneracy array is returned so
    downstream code can still track how many symmetry-equivalent HK pairs
    contributed to each rod.
    """
    from ra_sim.utils.stacking_fault import qr_dict_to_arrays

    miller, intensities, degeneracy, _ = qr_dict_to_arrays(qr_dict)

    result = process_peaks_parallel(
        miller,
        intensities,
        image_size,
        av,
        cv,
        lambda_,
        image,
        Distance_CoR_to_Detector,
        gamma_deg,
        Gamma_deg,
        chi_deg,
        psi_deg,
        psi_z_deg,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_pv_deg,
        gamma_pv_deg,
        eta_pv,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        unit_x,
        n_detector,
        save_flag,
        record_status,
        thickness,
        optics_mode,
        single_sample_indices,
        best_sample_indices_out,
        solve_q_steps,
        solve_q_adaptive,
        q_data_max_solutions,
    )

    return (*result, degeneracy)


def process_qr_rods_parallel_safe(*args, **kwargs):
    """Run ``process_qr_rods_parallel`` with Python fallback if needed."""

    try:
        return process_qr_rods_parallel(*args, **kwargs)
    except Exception:
        py_runner = getattr(process_qr_rods_parallel, "py_func", None)
        if callable(py_runner):
            return py_runner(*args, **kwargs)
        raise


def debug_detector_paths(
    beam_x_array, beam_y_array, theta_array, phi_array,
    theta_initial_deg, cor_angle_deg, chi_deg, psi_deg, psi_z_deg,
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
        Sample tilt around the CoR axis.
    cor_angle_deg : float
        Angle of the CoR axis relative to the +x axis.
    chi_deg, psi_deg, psi_z_deg : float
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
    psi_z_rad = np.radians(psi_z_deg)
    rad_theta_i = np.radians(theta_initial_deg)
    cor_axis_rad = np.radians(cor_angle_deg)

    cg = np.cos(gamma_rad)
    sg = np.sin(gamma_rad)
    cG = np.cos(Gamma_rad)
    sG = np.sin(Gamma_rad)
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

    c_chi = np.cos(chi_rad)
    s_chi = np.sin(chi_rad)
    R_y = np.array([
        [ c_chi, 0.0,   s_chi],
        [ 0.0,   1.0,   0.0],
        [-s_chi, 0.0, c_chi]
    ])
    c_psi = np.cos(psi_rad)
    s_psi = np.sin(psi_rad)
    R_z = np.array([
        [ c_psi, s_psi, 0.0],
        [-s_psi, c_psi, 0.0],
        [ 0.0,   0.0,   1.0]
    ])
    c_psi_z = np.cos(psi_z_rad)
    s_psi_z = np.sin(psi_z_rad)
    R_z_gonio = np.array([
        [ c_psi_z, s_psi_z, 0.0],
        [-s_psi_z, c_psi_z, 0.0],
        [ 0.0,     0.0,     1.0]
    ])
    R_z_R_y = (R_z_gonio @ R_z) @ R_y

    # Construct the pitched CoR axis in x–z and rotate with Rodrigues' formula;
    # see docs/cor_rotation_math.md for the math details.
    ax = np.cos(cor_axis_rad)
    ay = 0.0
    az = np.sin(cor_axis_rad)
    axis_norm = np.sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-12:
        axis_norm = 1.0
    ax /= axis_norm
    ay /= axis_norm
    az /= axis_norm

    ct = np.cos(rad_theta_i)
    st = np.sin(rad_theta_i)
    one_ct = 1.0 - ct
    R_cor = np.array([
        [ct + ax * ax * one_ct, ax * ay * one_ct - az * st, ax * az * one_ct + ay * st],
        [ay * ax * one_ct + az * st, ct + ay * ay * one_ct, ay * az * one_ct - ax * st],
        [az * ax * one_ct - ay * st, az * ay * one_ct + ax * st, ct + az * az * one_ct],
    ])
    R_sample = R_cor @ R_z_R_y

    n_surf = R_cor @ (R_z_R_y @ np.array([0.0, 0.0, 1.0]))
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
