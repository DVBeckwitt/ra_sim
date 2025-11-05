"""Optimization routines for fitting simulated data to experiments."""

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, differential_evolution, OptimizeResult
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, zoom
from scipy.spatial import cKDTree

from ra_sim.simulation.diffraction import process_peaks_parallel
from ra_sim.utils.calculations import d_spacing, two_theta

RNG = np.random.default_rng(42)


@dataclass
class TubeROI:
    """Representation of a physics-motivated tube ROI around a reflection."""

    reflection: Tuple[int, int, int]
    centerline: np.ndarray
    width: float
    bounds: Tuple[int, int, int, int]
    mask: np.ndarray
    off_tube_mask: np.ndarray
    sampling_probability: float = 1.0
    weights: Optional[np.ndarray] = None
    active_pixels: Optional[np.ndarray] = None
    full_weight_map: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None
    centerline_mask: Optional[np.ndarray] = None
    tile_size: int = 8
    tile_probabilities: Optional[np.ndarray] = None
    identifier: int = 0


@dataclass
class PeakROI:
    """Lightweight container describing a square ROI around a simulated peak."""

    reflection_index: int
    hkl: Tuple[int, int, int]
    center: Tuple[float, float]
    row_indices: np.ndarray
    col_indices: np.ndarray
    flat_indices: np.ndarray
    observed: np.ndarray
    observed_sum: float
    observed_mean: float
    num_pixels: int
    simulated_intensity: float


@dataclass
class SimulationCache:
    """Simple cache for simulated detector images keyed by parameter vectors."""

    keys: Sequence[str]
    images: Dict[Tuple[float, ...], np.ndarray] = field(default_factory=dict)
    max_positions: Dict[Tuple[float, ...], np.ndarray] = field(default_factory=dict)

    def _flatten_value(self, value: np.ndarray) -> Iterable[float]:
        if isinstance(value, np.ndarray):
            return value.ravel()
        if isinstance(value, (list, tuple)):
            return np.asarray(value, dtype=float).ravel()
        return (float(value),)

    def key_for(self, params: Dict[str, float]) -> Tuple[float, ...]:
        parts: List[float] = []
        for key in self.keys:
            value = params[key]
            parts.extend(float(f"{v:.8f}") for v in self._flatten_value(value))
        return tuple(parts)

    def get(
        self,
        params: Dict[str, float],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        key = self.key_for(params)
        if key in self.images:
            return self.images[key], self.max_positions[key]
        return None

    def store(
        self,
        params: Dict[str, float],
        image: np.ndarray,
        max_positions: np.ndarray,
    ) -> None:
        key = self.key_for(params)
        self.images[key] = image
        self.max_positions[key] = max_positions


@dataclass
class IterativeRefinementResult:
    """Container mimicking scipy's ``OptimizeResult`` with extra context."""

    x: np.ndarray
    fun: np.ndarray
    success: bool
    message: str
    best_params: Dict[str, float]
    history: List[Dict[str, float]]
    stage_summaries: List[Dict[str, float]]

    def __iter__(self):
        yield from (
            ("x", self.x),
            ("fun", self.fun),
            ("success", self.success),
            ("message", self.message),
        )


def _downsample_with_antialiasing(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample *image* by *factor* using Gaussian pre-filtering."""

    if factor <= 1:
        return image
    sigma = 0.5 * factor
    blurred = gaussian_filter(image, sigma=sigma, mode="reflect")
    return zoom(blurred, 1.0 / factor, order=1, prefilter=False)


def _compute_ridge_map(image: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Return a binary ridge map derived from gradient magnitude."""

    grad_x = sobel(image, axis=1, mode="reflect")
    grad_y = sobel(image, axis=0, mode="reflect")
    magnitude = np.hypot(grad_x, grad_y)
    threshold = np.percentile(magnitude, percentile)
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(np.mean(magnitude))
    return magnitude > threshold


def _update_params(
    params: Dict[str, float],
    var_names: Sequence[str],
    values: Sequence[float],
) -> Dict[str, float]:
    updated = dict(params)
    for name, val in zip(var_names, values):
        updated[name] = float(val)
    return updated


def _allowed_reflection_mask(miller: np.ndarray) -> np.ndarray:
    """Return a mask selecting reflections with ``2h + k`` divisible by 3."""

    if miller.ndim != 2 or miller.shape[1] < 2:
        raise ValueError("miller array must have shape (N, >=3)")
    hk = np.rint(miller[:, :2]).astype(np.int64, copy=False)
    return (2 * hk[:, 0] + hk[:, 1]) % 3 == 0


def _simulate_with_cache(
    params: Dict[str, float],
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    cache: SimulationCache,
) -> Tuple[np.ndarray, np.ndarray]:
    cached = cache.get(params)
    if cached is not None:
        return cached

    buffer = np.zeros((image_size, image_size), dtype=np.float64)

    mosaic = params['mosaic_params']
    wavelength_array = mosaic.get('wavelength_array')
    if wavelength_array is None:
        wavelength_array = mosaic.get('wavelength_i_array')
    if wavelength_array is None:
        wavelength_array = params.get('lambda')

    image, maxpos, _, _, _ = process_peaks_parallel(
        miller, intensities, image_size,
        params['a'], params['c'], wavelength_array,
        buffer, params['corto_detector'],
        params['gamma'], params['Gamma'], params['chi'], params.get('psi', 0.0),
        params['zs'], params['zb'], params['n2'],
        mosaic['beam_x_array'],
        mosaic['beam_y_array'],
        mosaic['theta_array'],
        mosaic['phi_array'],
        mosaic['sigma_mosaic_deg'],
        mosaic['gamma_mosaic_deg'],
        mosaic['eta'],
        wavelength_array,
        params['debye_x'], params['debye_y'],
        params['center'], params['theta_initial'],
        params.get('uv1', np.array([1.0, 0.0, 0.0])),
        params.get('uv2', np.array([0.0, 1.0, 0.0])),
        save_flag=0,
    )

    image = np.asarray(image, dtype=np.float64)
    maxpos = np.asarray(maxpos)

    cache.store(params, image, maxpos)
    return image, maxpos


def fit_mosaic_widths_separable(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    *,
    num_peaks: int = 40,
    roi_half_width: int = 8,
    min_peak_separation: Optional[float] = None,
    stratify: Optional[str] = None,
    stratify_bins: int = 5,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    max_nfev: int = 50,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> OptimizeResult:
    r"""Estimate mosaic pseudo-Voigt widths using separable non-linear least squares.

    Geometry, detector center, and per-peak locations are assumed to be fixed. Only
    the pseudo-Voigt mosaic widths (:math:`\sigma`, :math:`\gamma`) and mixing
    parameter :math:`\eta` are refined.  Peak amplitudes and a constant
    background are eliminated analytically within each ROI so that intensity
    outliers do not bias the width estimates.

    Parameters
    ----------
    experimental_image:
        Measured detector image at full resolution.
    miller, intensities:
        Arrays describing the simulated reflections.  These should match the
        arrays used to generate ``experimental_image``.
    image_size:
        Detector dimension (assumed square).
    params:
        Dictionary of simulation parameters.  ``params['mosaic_params']`` must
        contain the beam sample arrays plus initial values for
        ``sigma_mosaic_deg``, ``gamma_mosaic_deg`` and ``eta``.
    num_peaks:
        Number of bright peaks to include in the separable fit.
    roi_half_width:
        Half-width (in pixels) of the square ROI around each selected peak.
    min_peak_separation:
        Minimum Euclidean distance between ROI centres.  Defaults to ``2.5``
        times ``roi_half_width`` when ``None``.
    stratify:
        ``None`` (default) selects the globally brightest peaks.  ``"L"``
        enforces round-robin selection across distinct :math:`L` values while
        ``"twotheta"`` stratifies by equal-width 2θ bins.
    stratify_bins:
        Maximum number of bins when ``stratify='twotheta'``.
    loss, f_scale, max_nfev, bounds:
        Directly forwarded to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    OptimizeResult
        ``x`` holds the refined ``[sigma_deg, gamma_deg, eta]`` vector.  The
        ``best_params`` attribute mirrors ``params`` with the optimized mosaic
        values inserted, and ``selected_rois`` enumerates the ROIs used during
        the fit.
    """

    experimental_image = np.asarray(experimental_image, dtype=np.float64)
    if experimental_image.shape != (image_size, image_size):
        raise ValueError(
            "experimental_image shape must match the provided image_size"
        )

    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("miller must be an array of shape (N, 3)")
    if intensities.shape[0] != miller.shape[0]:
        raise ValueError("intensities and miller must have matching lengths")

    allowed_mask = _allowed_reflection_mask(miller)
    allowed_indices = np.flatnonzero(allowed_mask)
    if allowed_indices.size == 0:
        raise RuntimeError(
            "No reflections satisfy 2h + k ≡ 0 (mod 3) for mosaic-width fitting"
        )

    mosaic_params = dict(params.get("mosaic_params", {}))
    if not mosaic_params:
        raise ValueError("params['mosaic_params'] is required")

    beam_x = np.asarray(mosaic_params.get("beam_x_array"), dtype=np.float64)
    beam_y = np.asarray(mosaic_params.get("beam_y_array"), dtype=np.float64)
    theta_array = np.asarray(mosaic_params.get("theta_array"), dtype=np.float64)
    phi_array = np.asarray(mosaic_params.get("phi_array"), dtype=np.float64)
    if not (beam_x.size and beam_y.size and theta_array.size and phi_array.size):
        raise ValueError("mosaic_params must include beam and divergence samples")

    wavelength_array = mosaic_params.get("wavelength_array")
    if wavelength_array is None:
        wavelength_array = mosaic_params.get("wavelength_i_array")
    if wavelength_array is None:
        base_lambda = float(params.get("lambda", 1.0))
        wavelength_array = np.full(beam_x.shape, base_lambda, dtype=np.float64)
    else:
        wavelength_array = np.asarray(wavelength_array, dtype=np.float64)

    sigma0 = float(mosaic_params.get("sigma_mosaic_deg", 0.5))
    gamma0 = float(mosaic_params.get("gamma_mosaic_deg", 0.5))
    eta0 = float(mosaic_params.get("eta", 0.05))

    roi_half_width = int(roi_half_width)
    if roi_half_width <= 0:
        raise ValueError("roi_half_width must be a positive integer")
    if min_peak_separation is None:
        min_peak_separation = 2.5 * float(roi_half_width)
    min_peak_separation = float(min_peak_separation)
    min_peak_separation_sq = min_peak_separation * min_peak_separation

    num_peaks = int(num_peaks)
    if num_peaks <= 0:
        raise ValueError("num_peaks must be positive")

    a_lattice = float(params.get("a", 1.0))
    c_lattice = float(params.get("c", 1.0))
    lambda_scalar = float(params.get("lambda", float(np.mean(wavelength_array))))

    gamma_deg = float(params.get("gamma", 0.0))
    Gamma_deg = float(params.get("Gamma", 0.0))
    chi_deg = float(params.get("chi", 0.0))
    psi_deg = float(params.get("psi", 0.0))
    zs = float(params.get("zs", 0.0))
    zb = float(params.get("zb", 0.0))
    n2 = params.get("n2")
    if n2 is None:
        raise ValueError("params['n2'] (complex index of refraction) is required")
    debye_x = float(params.get("debye_x", 0.0))
    debye_y = float(params.get("debye_y", 0.0))
    theta_initial = float(params.get("theta_initial", 0.0))
    corto_detector = float(params.get("corto_detector"))
    center = tuple(params.get("center", (image_size / 2.0, image_size / 2.0)))
    unit_x = np.asarray(params.get("uv1", np.array([1.0, 0.0, 0.0])), dtype=np.float64)
    n_detector = np.asarray(params.get("uv2", np.array([0.0, 1.0, 0.0])), dtype=np.float64)

    full_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    def _simulate(
        miller_subset: np.ndarray,
        intens_subset: np.ndarray,
        sigma_deg: float,
        gamma_deg_: float,
        eta_: float,
        buffer: np.ndarray,
        *,
        record_hits: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        buffer.fill(0.0)
        image, hit_tables, *_ = process_peaks_parallel(
            miller_subset,
            intens_subset,
            image_size,
            a_lattice,
            c_lattice,
            wavelength_array,
            buffer,
            corto_detector,
            gamma_deg,
            Gamma_deg,
            chi_deg,
            psi_deg,
            zs,
            zb,
            n2,
            beam_x,
            beam_y,
            theta_array,
            phi_array,
            sigma_deg,
            gamma_deg_,
            eta_,
            wavelength_array,
            debye_x,
            debye_y,
            center,
            theta_initial,
            unit_x,
            n_detector,
            0,
        )
        image = np.asarray(image, dtype=np.float64)
        if not record_hits:
            return image, None
        hits_py = [np.asarray(tbl) for tbl in hit_tables]
        return image, hits_py

    allowed_miller = np.ascontiguousarray(miller[allowed_indices], dtype=np.float64)
    allowed_intensities = np.ascontiguousarray(
        intensities[allowed_indices], dtype=np.float64
    )

    two_theta_limit = 65.0
    kept_rows: List[int] = []
    allowed_two_theta: List[float] = []
    for local_idx, hkl_row in enumerate(allowed_miller):
        hkl_int = tuple(int(round(val)) for val in hkl_row)
        if all(v == 0 for v in hkl_int):
            continue
        d_hkl = d_spacing(hkl_int[0], hkl_int[1], hkl_int[2], a_lattice, c_lattice)
        tth_val = two_theta(d_hkl, lambda_scalar)
        if tth_val is None or not np.isfinite(tth_val):
            continue
        h_zero = hkl_int[0] == 0 and hkl_int[1] == 0
        if (not h_zero) and tth_val > two_theta_limit:
            continue
        kept_rows.append(local_idx)
        allowed_two_theta.append(float(tth_val) if tth_val is not None else float("nan"))

    if not kept_rows:
        raise RuntimeError(
            "No reflections satisfy the 2h + k ≡ 0 constraint within 65° 2θ"
        )

    allowed_indices = allowed_indices[kept_rows]
    allowed_miller = allowed_miller[kept_rows]
    allowed_intensities = allowed_intensities[kept_rows]
    allowed_two_theta = np.asarray(allowed_two_theta, dtype=np.float64)

    _, hit_tables = _simulate(
        allowed_miller,
        allowed_intensities,
        sigma0,
        gamma0,
        eta0,
        full_buffer,
        record_hits=True,
    )

    if not hit_tables:
        raise RuntimeError("Initial simulation produced no peak information")

    candidates: List[Dict[str, float]] = []
    for local_idx, tbl in enumerate(hit_tables):
        arr = np.asarray(tbl, dtype=np.float64)
        if arr.size == 0:
            continue
        for row in arr:
            intensity = float(row[0])
            col = float(row[1])
            row_pix = float(row[2])
            if not (np.isfinite(intensity) and np.isfinite(col) and np.isfinite(row_pix)):
                continue
            hkl = (int(round(row[4])), int(round(row[5])), int(round(row[6])))
            if all(v == 0 for v in hkl):
                continue
            h, k, _ = hkl
            if (2 * h + k) % 3 != 0:
                continue
            tth = float(allowed_two_theta[local_idx])
            candidates.append(
                {
                    "reflection_index": int(allowed_indices[local_idx]),
                    "intensity": intensity,
                    "row": row_pix,
                    "col": col,
                    "hkl": hkl,
                    "L": hkl[2],
                    "two_theta": tth,
                }
            )

    if not candidates:
        raise RuntimeError("No simulated peaks available for ROI selection")

    def _candidate_iter() -> Iterable[Dict[str, float]]:
        key = (stratify or "none").lower()
        if key not in {"none", "l", "twotheta"}:
            raise ValueError("stratify must be None, 'L', or 'twotheta'")
        sorted_candidates = sorted(
            candidates, key=lambda c: float(c["intensity"]), reverse=True
        )
        if key == "none" or len(sorted_candidates) <= 1:
            return sorted_candidates

        if key == "l":
            groups: Dict[int, List[Dict[str, float]]] = {}
            for cand in sorted_candidates:
                groups.setdefault(cand["L"], []).append(cand)
            for group in groups.values():
                group.sort(key=lambda c: float(c["intensity"]), reverse=True)
            order = sorted(groups.keys(), key=lambda val: (abs(val), val))
            round_robin: List[Dict[str, float]] = []
            while True:
                progressed = False
                for val in list(order):
                    group = groups.get(val)
                    if not group:
                        continue
                    round_robin.append(group.pop(0))
                    progressed = True
                    if not group:
                        groups.pop(val)
                if not progressed:
                    break
            return round_robin

        # stratify == 'twotheta'
        tth_values = [c["two_theta"] for c in sorted_candidates if c["two_theta"] is not None]
        if not tth_values or np.allclose(tth_values, tth_values[0]):
            return sorted_candidates
        num_bins = max(1, min(int(stratify_bins), len(tth_values)))
        if num_bins == 1:
            return sorted_candidates
        edges = np.linspace(min(tth_values), max(tth_values), num_bins + 1)
        bins: Dict[int, List[Dict[str, float]]] = {i: [] for i in range(num_bins)}
        for cand in sorted_candidates:
            tth = cand["two_theta"]
            if tth is None or not np.isfinite(tth):
                bin_idx = 0
            else:
                bin_idx = int(np.searchsorted(edges, tth, side="right") - 1)
                bin_idx = max(0, min(num_bins - 1, bin_idx))
            bins[bin_idx].append(cand)
        for group in bins.values():
            group.sort(key=lambda c: float(c["intensity"]), reverse=True)
        round_robin: List[Dict[str, float]] = []
        active_bins = [idx for idx in range(num_bins) if bins[idx]]
        while active_bins:
            progressed = False
            for idx in list(active_bins):
                group = bins[idx]
                if not group:
                    active_bins.remove(idx)
                    continue
                round_robin.append(group.pop(0))
                progressed = True
                if not group:
                    active_bins.remove(idx)
            if not progressed:
                break
        return round_robin if round_robin else sorted_candidates

    ordered_candidates = list(_candidate_iter())

    rois: List[PeakROI] = []
    selected_centres: List[Tuple[float, float]] = []
    for cand in ordered_candidates:
        if len(rois) >= num_peaks:
            break
        row_c = cand["row"]
        col_c = cand["col"]
        row_idx = int(round(row_c))
        col_idx = int(round(col_c))
        if (
            row_idx - roi_half_width < 0
            or row_idx + roi_half_width >= image_size
            or col_idx - roi_half_width < 0
            or col_idx + roi_half_width >= image_size
        ):
            continue
        if selected_centres:
            if any(
                (row_c - r) ** 2 + (col_c - c) ** 2 < min_peak_separation_sq
                for r, c in selected_centres
            ):
                continue

        rows = np.arange(row_idx - roi_half_width, row_idx + roi_half_width + 1, dtype=int)
        cols = np.arange(col_idx - roi_half_width, col_idx + roi_half_width + 1, dtype=int)
        patch = experimental_image[np.ix_(rows, cols)]
        flat_patch = patch.ravel()
        valid_idx = np.flatnonzero(np.isfinite(flat_patch))
        if valid_idx.size == 0:
            continue
        observed = flat_patch.take(valid_idx)
        observed = observed.astype(np.float64, copy=False)
        observed_sum = float(observed.sum())
        num_valid = int(observed.size)
        observed_mean = observed_sum / num_valid
        rois.append(
            PeakROI(
                reflection_index=int(cand["reflection_index"]),
                hkl=cand["hkl"],
                center=(row_c, col_c),
                row_indices=rows,
                col_indices=cols,
                flat_indices=valid_idx,
                observed=observed,
                observed_sum=observed_sum,
                observed_mean=observed_mean,
                num_pixels=num_valid,
                simulated_intensity=float(cand["intensity"]),
            )
        )
        selected_centres.append((row_c, col_c))

    if not rois:
        raise RuntimeError("No valid ROIs were selected for mosaic fitting")

    unique_indices = sorted({roi.reflection_index for roi in rois})
    subset_miller = np.ascontiguousarray(miller[unique_indices], dtype=np.float64)
    subset_intensities = np.ones(subset_miller.shape[0], dtype=np.float64)

    subset_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    def residual(theta: np.ndarray) -> np.ndarray:
        sigma_deg, gamma_deg_local, eta_local = map(float, theta)
        sim_image, _ = _simulate(
            subset_miller,
            subset_intensities,
            sigma_deg,
            gamma_deg_local,
            eta_local,
            subset_buffer,
            record_hits=False,
        )

        residual_blocks: List[np.ndarray] = []
        for roi in rois:
            block = sim_image[np.ix_(roi.row_indices, roi.col_indices)]
            flat = block.ravel()
            template = flat.take(roi.flat_indices)
            if template.size == 0:
                residual_blocks.append(roi.observed - roi.observed_mean)
                continue
            tt = float(np.dot(template, template))
            if tt <= 1e-16:
                residual_blocks.append(roi.observed - roi.observed_mean)
                continue
            to = float(np.sum(template))
            ty = float(np.dot(template, roi.observed))
            oy = roi.observed_sum
            oo = float(roi.num_pixels)
            det = tt * oo - to * to
            if abs(det) <= 1e-14 * tt * oo:
                residual_blocks.append(roi.observed - roi.observed_mean)
                continue
            amp = (ty * oo - oy * to) / det
            bkg = (tt * oy - to * ty) / det
            model = amp * template + bkg
            residual_blocks.append(roi.observed - model)
        if not residual_blocks:
            return np.zeros(1, dtype=np.float64)
        return np.concatenate(residual_blocks)

    if bounds is None:
        bounds = (
            np.array([0.01, 0.01, 0.0], dtype=np.float64),
            np.array([5.0, 5.0, 1.0], dtype=np.float64),
        )

    x0 = np.array([sigma0, gamma0, eta0], dtype=np.float64)
    result = least_squares(
        residual,
        x0,
        bounds=bounds,
        loss=loss,
        f_scale=f_scale,
        max_nfev=int(max_nfev),
    )

    best_params = dict(params)
    best_mosaic = dict(best_params.get("mosaic_params", {}))
    best_mosaic["sigma_mosaic_deg"] = float(result.x[0])
    best_mosaic["gamma_mosaic_deg"] = float(result.x[1])
    best_mosaic["eta"] = float(result.x[2])
    best_params["mosaic_params"] = best_mosaic

    result.best_params = best_params
    result.selected_rois = rois
    result.reflection_indices = unique_indices
    return result


def _estimate_pixel_size(params: Dict[str, float]) -> float:
    pixel_size = params.get('pixel_size')
    if pixel_size is None:
        # Fall back to detector distance divided by nominal pixels for 4k detector
        pixel_size = params.get('corto_detector', 1.0) / 4096.0
    return max(float(pixel_size), 1e-6)


def _interpolate_line(points: List[Tuple[float, float]]) -> np.ndarray:
    if len(points) == 1:
        return np.asarray(points, dtype=float)

    dense_points: List[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        steps = max(int(length * 2), 1)
        xs = np.linspace(start[0], end[0], steps)
        ys = np.linspace(start[1], end[1], steps)
        dense_points.append(np.column_stack((xs, ys)))
    return np.vstack(dense_points)


def build_tube_rois(
    miller: np.ndarray,
    max_positions: np.ndarray,
    params: Dict[str, float],
    image_size: int,
    *,
    measured_dict: Optional[Dict[Tuple[int, int, int], List[Tuple[float, float]]]] = None,
    base_width: Optional[float] = None,
) -> List[TubeROI]:
    """Construct tube-shaped ROIs following the manuscript guidance."""

    pixel_size = _estimate_pixel_size(params)
    mosaic = params['mosaic_params']
    sigma_deg = float(mosaic.get('sigma_mosaic_deg', 0.3))
    mosaic_fwhm_rad = math.radians(sigma_deg) * 2.0 * math.sqrt(2.0 * math.log(2.0))
    divergence_rad = math.radians(float(mosaic.get('gamma_mosaic_deg', 0.3)))
    bandwidth_rad = float(params.get('bandwidth_rad', 0.0))
    detector_length = float(params.get('corto_detector', 1.0))

    nominal_width = mosaic_fwhm_rad * detector_length / pixel_size
    nominal_width += divergence_rad * detector_length / pixel_size
    nominal_width += bandwidth_rad * detector_length / pixel_size
    if base_width is not None:
        nominal_width = max(nominal_width, base_width)
    width_px = max(3.0, nominal_width)

    rois: List[TubeROI] = []
    measured_dict = measured_dict or {}

    for idx, reflection in enumerate(miller):
        I0, x0, y0, I1, x1, y1 = max_positions[idx]
        points: List[Tuple[float, float]] = []
        for x, y in ((x0, y0), (x1, y1)):
            if np.isfinite(x) and np.isfinite(y):
                points.append((float(x), float(y)))

        key = tuple(int(v) for v in reflection)
        for mx, my in measured_dict.get(key, []):
            points.append((float(mx), float(my)))

        if not points:
            continue

        # Sort points by polar angle around detector center to stabilise tubes
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] > 2:
            centroid = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            order = np.argsort(angles)
            pts = pts[order]
        dense_centerline = _interpolate_line(list(map(tuple, pts)))

        min_x = max(int(np.floor(np.min(pts[:, 0] - width_px - 2))), 0)
        max_x = min(int(np.ceil(np.max(pts[:, 0] + width_px + 2))), image_size - 1)
        min_y = max(int(np.floor(np.min(pts[:, 1] - width_px - 2))), 0)
        max_y = min(int(np.ceil(np.max(pts[:, 1] + width_px + 2))), image_size - 1)

        if min_x >= max_x or min_y >= max_y:
            continue

        bounds = (min_x, max_x, min_y, max_y)
        grid_y, grid_x = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
        query_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        tree = cKDTree(dense_centerline)
        distances, _ = tree.query(query_points, k=1)
        distance_map = distances.reshape(grid_x.shape)

        tube_mask = distance_map <= width_px
        centerline_mask = distance_map <= max(1.0, width_px * 0.25)
        off_mask = np.ones_like(tube_mask, dtype=bool)
        off_mask[tube_mask] = False

        roi = TubeROI(
            reflection=key,
            centerline=dense_centerline,
            width=float(width_px),
            bounds=bounds,
            mask=tube_mask,
            off_tube_mask=off_mask,
            identifier=idx,
        )
        roi.centerline_mask = centerline_mask
        rois.append(roi)

    return rois


def compute_sensitivity_weights(
    base_sim: np.ndarray,
    params: Dict[str, float],
    var_names: Sequence[str],
    rois: List[TubeROI],
    simulator,
    *,
    downsample_factor: int = 4,
    percentile: float = 90.0,
    huber_percentile: float = 97.0,
    per_reflection_quota: int = 200,
    off_tube_fraction: float = 0.05,
) -> None:
    """Populate each ROI with active pixels using a sensitivity-driven map."""

    base_down = _downsample_with_antialiasing(base_sim, downsample_factor)
    sensitivity = np.zeros_like(base_down, dtype=np.float64)

    for name in var_names:
        delta = max(abs(params.get(name, 1.0)) * 1e-3, 1e-3)
        shifted = dict(params)
        shifted[name] = params.get(name, 0.0) + delta
        sim_shift, _ = simulator(shifted)
        shift_down = _downsample_with_antialiasing(sim_shift, downsample_factor)
        grad = (shift_down - base_down) / delta
        sensitivity += grad * grad

    delta_s = np.sqrt(np.maximum(sensitivity, 0.0))
    safe_base = np.clip(base_down, 1e-6, None)
    importance_map = (delta_s * delta_s) / safe_base

    clip_level = np.percentile(importance_map, huber_percentile)
    if np.isfinite(clip_level) and clip_level > 0:
        importance_map = np.where(
            importance_map <= clip_level,
            importance_map,
            clip_level + 0.1 * (importance_map - clip_level),
        )

    upsample_factor = downsample_factor
    upsampled = zoom(importance_map, upsample_factor, order=1, prefilter=False)
    if upsampled.shape != base_sim.shape:
        pad_y = base_sim.shape[0] - upsampled.shape[0]
        pad_x = base_sim.shape[1] - upsampled.shape[1]
        upsampled = np.pad(
            upsampled,
            ((0, max(pad_y, 0)), (0, max(pad_x, 0))),
            mode="edge",
        )[: base_sim.shape[0], : base_sim.shape[1]]

    for roi in rois:
        min_x, max_x, min_y, max_y = roi.bounds
        weights_map = upsampled[min_y:max_y + 1, min_x:max_x + 1]
        weights_map = np.where(roi.mask, weights_map, 0.0)
        if not np.any(weights_map):
            weights_map = np.where(roi.mask, 1.0, 0.0)

        values = weights_map[roi.mask]
        if values.size == 0:
            continue

        threshold = np.percentile(values, percentile)
        active_mask = np.zeros_like(roi.mask, dtype=bool)
        active_mask[roi.mask] = weights_map[roi.mask] >= threshold

        # Ensure at least a quota of pixels per reflection
        if active_mask.sum() < per_reflection_quota:
            flat_weights = weights_map[roi.mask]
            order = np.argsort(flat_weights)[::-1]
            top = order[: min(per_reflection_quota, order.size)]
            template = np.zeros_like(flat_weights, dtype=bool)
            template[top] = True
            active_mask = np.zeros_like(roi.mask, dtype=bool)
            active_mask[roi.mask] = template

        # Add low-weight off-tube pixels
        off_candidates = np.argwhere(roi.off_tube_mask)
        if off_candidates.size:
            count = max(1, int(off_tube_fraction * off_candidates.shape[0]))
            chosen = RNG.choice(off_candidates.shape[0], size=count, replace=False)
            off_points = off_candidates[chosen]
            active_mask[off_points[:, 0], off_points[:, 1]] = True

        ys, xs = np.nonzero(active_mask)
        global_y = ys + min_y
        global_x = xs + min_x

        roi.active_pixels = np.stack((global_y, global_x), axis=1)
        roi.weights = weights_map[ys, xs]
        roi.full_weight_map = weights_map
        roi.active_mask = active_mask

        sampling_prob = roi.weights.size / max(1, roi.mask.sum())
        roi.sampling_probability = max(float(sampling_prob), 1e-3)


def sample_tiles(
    rois: Sequence[TubeROI],
    *,
    temperature: float,
    explore_fraction: float,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    """Sample weighted tiles from each ROI and return selected pixels."""

    selected: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}

    for roi in rois:
        if roi.active_pixels is None or roi.weights is None or roi.weights.size == 0:
            continue

        min_x, max_x, min_y, max_y = roi.bounds
        width = max_x - min_x + 1
        tiles_x = (width + roi.tile_size - 1) // roi.tile_size

        rel_y = roi.active_pixels[:, 0] - min_y
        rel_x = roi.active_pixels[:, 1] - min_x
        tile_y = rel_y // roi.tile_size
        tile_x = rel_x // roi.tile_size
        tile_ids = tile_y * tiles_x + tile_x
        tile_count = int(tile_ids.max()) + 1

        energies = np.bincount(tile_ids, weights=roi.weights, minlength=tile_count)
        if not np.any(energies):
            energies = np.ones_like(energies)

        scaled = energies.astype(np.float64)
        temperature = max(float(temperature), 1e-3)
        if temperature != 1.0:
            scaled = scaled ** (1.0 / temperature)
        probs = scaled / scaled.sum()

        base_tiles = max(1, int(round(roi.sampling_probability * tile_count)))
        base_tiles = min(base_tiles, tile_count)
        explore_tiles = max(1, int(round(explore_fraction * tile_count)))
        explore_tiles = min(explore_tiles, tile_count)

        base_selection = set(RNG.choice(tile_count, size=base_tiles, replace=False, p=probs))
        explore_selection = set(RNG.choice(tile_count, size=explore_tiles, replace=False))
        chosen_tiles = base_selection.union(explore_selection)

        mask = np.isin(tile_ids, list(chosen_tiles))
        pixels = roi.active_pixels[mask]
        weights = roi.weights[mask]
        if pixels.size == 0:
            continue

        sampling_prob = max(float(len(chosen_tiles)) / max(tile_count, 1), roi.sampling_probability)
        selected[roi.identifier] = (pixels, weights, sampling_prob)
        roi.tile_probabilities = probs

    return selected


def fit_local_background(
    experimental_image: np.ndarray,
    rois: Sequence[TubeROI],
    *,
    anchor_fraction: float = 0.01,
) -> np.ndarray:
    """Fit a smooth background inside ROI bounding boxes."""

    background = np.zeros_like(experimental_image, dtype=np.float64)
    coverage = np.zeros_like(experimental_image, dtype=np.float64)
    global_mask = np.zeros_like(experimental_image, dtype=bool)

    for roi in rois:
        min_x, max_x, min_y, max_y = roi.bounds
        patch = experimental_image[min_y:max_y + 1, min_x:max_x + 1]
        mask = roi.mask
        center_mask = roi.centerline_mask if roi.centerline_mask is not None else np.zeros_like(mask, dtype=bool)
        sample_mask = mask & ~center_mask
        if not np.any(sample_mask):
            sample_mask = mask

        ys, xs = np.nonzero(sample_mask)
        if ys.size < 6:
            ys, xs = np.nonzero(mask)

        if ys.size == 0:
            continue

        xx = xs.astype(np.float64)
        yy = ys.astype(np.float64)
        A = np.column_stack([
            np.ones_like(xx),
            xx,
            yy,
            xx * xx,
            xx * yy,
            yy * yy,
        ])
        b = patch[ys, xs]
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

        grid_x, grid_y = np.meshgrid(
            np.arange(patch.shape[1], dtype=np.float64),
            np.arange(patch.shape[0], dtype=np.float64),
            indexing="xy",
        )
        fitted = (
            coeffs[0]
            + coeffs[1] * grid_x
            + coeffs[2] * grid_y
            + coeffs[3] * grid_x * grid_x
            + coeffs[4] * grid_x * grid_y
            + coeffs[5] * grid_y * grid_y
        )

        if roi.centerline_mask is not None:
            fitted = np.where(roi.centerline_mask, np.minimum(fitted, patch), fitted)

        background[min_y:max_y + 1, min_x:max_x + 1] += fitted
        coverage[min_y:max_y + 1, min_x:max_x + 1] += 1.0
        global_mask[min_y:max_y + 1, min_x:max_x + 1] |= mask

    mask = coverage > 0
    if np.any(mask):
        background[mask] /= np.maximum(coverage[mask], 1.0)

    smooth = gaussian_filter(background, sigma=3.0, mode="reflect")
    background[mask] = 0.7 * background[mask] + 0.3 * smooth[mask]

    outside_mask = ~global_mask
    anchors = np.argwhere(outside_mask)
    if anchors.size:
        count = max(1, int(anchor_fraction * anchors.shape[0]))
        chosen = RNG.choice(anchors.shape[0], size=count, replace=False)
        anchor_points = anchors[chosen]
        global_background = gaussian_filter(experimental_image, sigma=30.0, mode="reflect")
        background[anchor_points[:, 0], anchor_points[:, 1]] = global_background[
            anchor_points[:, 0], anchor_points[:, 1]
        ]

    return background


def _poisson_deviance(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    obs = np.clip(obs, 1e-9, None)
    pred = np.clip(pred, 1e-9, None)
    return 2.0 * (pred - obs + obs * np.log(obs / pred))


def _anscombe(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.clip(x, 0.0, None) + 3.0 / 8.0)


def _huber(residual: np.ndarray, delta: float) -> np.ndarray:
    delta = max(delta, 1e-6)
    abs_res = np.abs(residual)
    return np.where(abs_res <= delta, residual, delta * np.sign(residual))


def robust_residuals(
    obs: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    sampling_probability: float,
    *,
    huber_delta: float,
    mixture: float = 0.1,
) -> np.ndarray:
    dev = _poisson_deviance(obs, pred)
    ans_res = _anscombe(obs) - _anscombe(pred)
    huber_res = _huber(ans_res, huber_delta)
    mixture = np.clip(mixture, 0.0, 1.0)
    mixture_weight = (1.0 - mixture) + mixture * np.exp(-0.5 * (ans_res / (huber_delta + 1e-6)) ** 2)
    combined = 0.5 * np.sqrt(np.abs(dev)) + 0.5 * huber_res
    scaling = np.sqrt(np.maximum(weights, 1e-8)) / max(sampling_probability, 1e-3)
    return combined * scaling * np.sqrt(mixture_weight)


def _select_active_reflections(
    rois: Sequence[TubeROI],
    *,
    max_reflections: int,
    random_fraction: float,
) -> List[TubeROI]:
    scored = [
        (float(np.sum(roi.weights)) if roi.weights is not None else 0.0, roi)
        for roi in rois
    ]
    scored.sort(key=lambda t: t[0], reverse=True)

    selected = [roi for _, roi in scored[:max_reflections]]
    remainder = [roi for _, roi in scored[max_reflections:]]
    if remainder and random_fraction > 0:
        count = max(1, int(round(random_fraction * len(remainder))))
        count = min(count, len(remainder))
        selected.extend(RNG.choice(remainder, size=count, replace=False))
    return selected


def _centerline_shift(
    old: Sequence[Tuple[float, float]],
    new: Sequence[Tuple[float, float]],
) -> float:
    if not len(old) or not len(new):
        return np.inf
    old_arr = np.asarray(old)
    new_arr = np.asarray(new)
    tree = cKDTree(old_arr)
    distances, _ = tree.query(new_arr, k=1)
    return float(np.max(distances))


def _refresh_rois_if_needed(
    rois: List[TubeROI],
    miller: np.ndarray,
    max_positions: np.ndarray,
    params: Dict[str, float],
    image_size: int,
    *,
    measured_dict: Optional[Dict[Tuple[int, int, int], List[Tuple[float, float]]]] = None,
    threshold: float = 1.0,
) -> List[TubeROI]:
    updated: List[TubeROI] = []
    for roi in rois:
        idx = roi.identifier
        if idx >= len(miller):
            continue
        I0, x0, y0, I1, x1, y1 = max_positions[idx]
        points = []
        for x, y in ((x0, y0), (x1, y1)):
            if np.isfinite(x) and np.isfinite(y):
                points.append((float(x), float(y)))
        shift = _centerline_shift(roi.centerline, points)
        if shift <= threshold:
            updated.append(roi)
        else:
            # rebuild ROI when shift significant
            rebuilt = build_tube_rois(
                miller[idx:idx + 1],
                max_positions[idx:idx + 1],
                params,
                image_size,
                measured_dict=measured_dict,
                base_width=roi.width,
            )
            if rebuilt:
                new_roi = rebuilt[0]
                new_roi.identifier = roi.identifier
                updated.append(new_roi)
            else:
                updated.append(roi)
    return updated


def _stage_one_initialize(
    experimental_image: np.ndarray,
    params: Dict[str, float],
    var_names: Sequence[str],
    simulator,
    *,
    downsample_factor: int,
    max_nfev: int,
) -> Tuple[Dict[str, float], OptimizeResult]:
    """Stage 1 coarse alignment using Chamfer distance on ridge maps."""

    downsample_factor = max(int(downsample_factor), 1)
    exp_down = _downsample_with_antialiasing(experimental_image, downsample_factor)
    ridge_exp = _compute_ridge_map(exp_down)
    distance_exp = distance_transform_edt(~ridge_exp)

    x0 = np.array([params[name] for name in var_names], dtype=float)

    def residual(x):
        trial = _update_params(params, var_names, x)
        sim_img, _ = simulator(trial)
        sim_down = _downsample_with_antialiasing(sim_img, downsample_factor)
        ridge_sim = _compute_ridge_map(sim_down)
        distance_sim = distance_transform_edt(~ridge_sim)
        fwd = distance_sim[ridge_exp]
        back = distance_exp[ridge_sim]
        return np.concatenate((fwd.ravel(), back.ravel()))

    result = least_squares(residual, x0, max_nfev=max_nfev)
    updated_params = _update_params(params, var_names, result.x)
    return updated_params, result


def _stage_two_refinement(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    var_names: Sequence[str],
    simulator,
    measured_dict: Dict[Tuple[int, int, int], List[Tuple[float, float]]],
    *,
    cfg: Dict[str, float],
) -> Tuple[Dict[str, float], OptimizeResult, List[TubeROI], np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Stage 2 refinement using Poisson deviance on active pixels."""

    base_sim, maxpos = simulator(params)
    rois = build_tube_rois(
        miller,
        maxpos,
        params,
        image_size,
        measured_dict=measured_dict,
    )

    compute_sensitivity_weights(
        base_sim,
        params,
        var_names,
        rois,
        simulator,
        downsample_factor=int(cfg.get('downsample_factor', 4)),
        percentile=float(cfg.get('percentile', 90.0)),
        huber_percentile=float(cfg.get('huber_percentile', 97.0)),
        per_reflection_quota=int(cfg.get('per_reflection_quota', 200)),
        off_tube_fraction=float(cfg.get('off_tube_fraction', 0.05)),
    )

    rois_state = rois

    def residual(x):
        nonlocal rois_state
        trial = _update_params(params, var_names, x)
        sim_img, trial_maxpos = simulator(trial)
        rois_state = _refresh_rois_if_needed(
            rois_state,
            miller,
            trial_maxpos,
            trial,
            image_size,
            measured_dict=measured_dict,
            threshold=float(cfg.get('roi_refresh_threshold', 1.0)),
        )

        missing = [roi for roi in rois_state if roi.weights is None]
        if missing:
            compute_sensitivity_weights(
                sim_img,
                trial,
                var_names,
                rois_state,
                simulator,
                downsample_factor=int(cfg.get('downsample_factor', 4)),
                percentile=float(cfg.get('percentile', 90.0)),
                huber_percentile=float(cfg.get('huber_percentile', 97.0)),
                per_reflection_quota=int(cfg.get('per_reflection_quota', 200)),
                off_tube_fraction=float(cfg.get('off_tube_fraction', 0.05)),
            )

        active = _select_active_reflections(
            rois_state,
            max_reflections=int(cfg.get('max_reflections', 12)),
            random_fraction=float(cfg.get('random_reflection_fraction', 0.15)),
        )

        background = fit_local_background(experimental_image, rois_state)
        sampled = sample_tiles(
            active,
            temperature=float(cfg.get('sampling_temperature', 1.0)),
            explore_fraction=float(cfg.get('explore_fraction', 0.15)),
        )

        residuals_list: List[np.ndarray] = []
        for roi in active:
            selection = sampled.get(roi.identifier)
            if selection is None:
                continue
            pixels, weights, sampling_prob = selection
            obs = experimental_image[pixels[:, 0], pixels[:, 1]]
            model = sim_img[pixels[:, 0], pixels[:, 1]] + background[pixels[:, 0], pixels[:, 1]]
            res = robust_residuals(
                obs,
                model,
                weights,
                sampling_prob,
                huber_delta=float(cfg.get('huber_delta', 2.5)),
                mixture=float(cfg.get('outlier_mixture', 0.1)),
            )
            residuals_list.append(res)

        if not residuals_list:
            return np.zeros(1, dtype=float)
        return np.concatenate(residuals_list)

    x0 = np.array([params[name] for name in var_names], dtype=float)
    result = least_squares(residual, x0, max_nfev=int(cfg.get('max_nfev', 25)))
    updated_params = _update_params(params, var_names, result.x)
    final_sim, _ = simulator(updated_params)
    rois_state = _refresh_rois_if_needed(
        rois_state,
        miller,
        maxpos,
        updated_params,
        image_size,
        measured_dict=measured_dict,
        threshold=float(cfg.get('roi_refresh_threshold', 1.0)),
    )
    return updated_params, result, rois_state, final_sim, residual


def _stage_three_refinement(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    var_names: Sequence[str],
    simulator,
    rois: List[TubeROI],
    measured_dict: Dict[Tuple[int, int, int], List[Tuple[float, float]]],
    *,
    cfg: Dict[str, float],
) -> Tuple[Dict[str, float], OptimizeResult, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Final stage at native resolution with expanded active sets."""

    compute_sensitivity_weights(
        simulator(params)[0],
        params,
        var_names,
        rois,
        simulator,
        downsample_factor=int(cfg.get('downsample_factor', 1)),
        percentile=float(cfg.get('percentile', 85.0)),
        huber_percentile=float(cfg.get('huber_percentile', 98.0)),
        per_reflection_quota=int(cfg.get('per_reflection_quota', 400)),
        off_tube_fraction=float(cfg.get('off_tube_fraction', 0.1)),
    )

    rois_state = rois

    def residual(x):
        nonlocal rois_state
        trial = _update_params(params, var_names, x)
        sim_img, trial_maxpos = simulator(trial)
        rois_state = _refresh_rois_if_needed(
            rois_state,
            miller,
            trial_maxpos,
            trial,
            image_size,
            measured_dict=measured_dict,
            threshold=float(cfg.get('roi_refresh_threshold', 0.5)),
        )

        compute_sensitivity_weights(
            sim_img,
            trial,
            var_names,
            rois_state,
            simulator,
            downsample_factor=int(cfg.get('downsample_factor', 1)),
            percentile=float(cfg.get('percentile', 85.0)),
            huber_percentile=float(cfg.get('huber_percentile', 98.0)),
            per_reflection_quota=int(cfg.get('per_reflection_quota', 400)),
            off_tube_fraction=float(cfg.get('off_tube_fraction', 0.1)),
        )

        active = _select_active_reflections(
            rois_state,
            max_reflections=int(cfg.get('max_reflections', len(rois_state))),
            random_fraction=float(cfg.get('random_reflection_fraction', 0.1)),
        )

        background = fit_local_background(experimental_image, rois_state)
        sampled = sample_tiles(
            active,
            temperature=float(cfg.get('sampling_temperature', 0.7)),
            explore_fraction=float(cfg.get('explore_fraction', 0.1)),
        )

        residuals_list: List[np.ndarray] = []
        for roi in active:
            selection = sampled.get(roi.identifier)
            if selection is None:
                continue
            pixels, weights, sampling_prob = selection
            obs = experimental_image[pixels[:, 0], pixels[:, 1]]
            model = sim_img[pixels[:, 0], pixels[:, 1]] + background[pixels[:, 0], pixels[:, 1]]
            res = robust_residuals(
                obs,
                model,
                weights,
                sampling_prob,
                huber_delta=float(cfg.get('huber_delta', 2.0)),
                mixture=float(cfg.get('outlier_mixture', 0.05)),
            )
            residuals_list.append(res)

        if not residuals_list:
            return np.zeros(1, dtype=float)
        return np.concatenate(residuals_list)

    x0 = np.array([params[name] for name in var_names], dtype=float)
    result = least_squares(residual, x0, max_nfev=int(cfg.get('max_nfev', 35)))
    updated_params = _update_params(params, var_names, result.x)
    final_sim, _ = simulator(updated_params)
    return updated_params, result, final_sim, residual


def iterative_refinement(
    experimental_image: np.ndarray,
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, float],
    *,
    var_names: Optional[Sequence[str]] = None,
    measured_peaks: Optional[Sequence[Dict[str, float]]] = None,
    config: Optional[Dict[str, Dict[str, float]]] = None,
) -> IterativeRefinementResult:
    """Run the multi-stage refinement described in the manuscript."""

    if var_names is None:
        var_names = ('zb', 'zs', 'theta_initial', 'chi')
    var_names = list(var_names)

    measured_dict = build_measured_dict(measured_peaks or [])

    cache_keys = [
        'gamma', 'Gamma', 'corto_detector', 'theta_initial',
        'zs', 'zb', 'chi', 'a', 'c', 'center'
    ]
    cache = SimulationCache(cache_keys)

    def simulator(local_params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        merged = dict(params)
        merged.update(local_params)
        return _simulate_with_cache(merged, miller, intensities, image_size, cache)

    config = config or {}
    stage1_cfg = {'downsample_factor': 8, 'max_nfev': 20}
    stage1_cfg.update(config.get('stage1', {}))
    stage2_cfg = {
        'downsample_factor': 4,
        'percentile': 90.0,
        'huber_percentile': 97.0,
        'per_reflection_quota': 200,
        'off_tube_fraction': 0.05,
        'max_reflections': 12,
        'random_reflection_fraction': 0.15,
        'sampling_temperature': 1.0,
        'explore_fraction': 0.15,
        'huber_delta': 2.5,
        'outlier_mixture': 0.1,
        'max_nfev': 25,
        'roi_refresh_threshold': 1.0,
    }
    stage2_cfg.update(config.get('stage2', {}))
    stage3_cfg = {
        'downsample_factor': 1,
        'percentile': 85.0,
        'huber_percentile': 98.0,
        'per_reflection_quota': 400,
        'off_tube_fraction': 0.1,
        'max_reflections': len(miller),
        'random_reflection_fraction': 0.1,
        'sampling_temperature': 0.7,
        'explore_fraction': 0.1,
        'huber_delta': 2.0,
        'outlier_mixture': 0.05,
        'max_nfev': 35,
        'roi_refresh_threshold': 0.5,
    }
    stage3_cfg.update(config.get('stage3', {}))

    history: List[Dict[str, float]] = []
    stage_summaries: List[Dict[str, float]] = []
    current_params = dict(params)

    current_params, stage1_result = _stage_one_initialize(
        experimental_image,
        current_params,
        var_names,
        simulator,
        downsample_factor=stage1_cfg['downsample_factor'],
        max_nfev=stage1_cfg['max_nfev'],
    )
    history.append({name: current_params[name] for name in var_names})
    stage_summaries.append({
        'stage': 'level1',
        'cost': float(np.sum(stage1_result.fun ** 2)),
        'nfev': stage1_result.nfev,
    })

    current_params, stage2_result, rois, stage2_sim, stage2_residual = _stage_two_refinement(
        experimental_image,
        miller,
        intensities,
        image_size,
        current_params,
        var_names,
        simulator,
        measured_dict,
        cfg=stage2_cfg,
    )
    history.append({name: current_params[name] for name in var_names})
    stage_summaries.append({
        'stage': 'level2',
        'cost': float(np.sum(stage2_result.fun ** 2)),
        'nfev': stage2_result.nfev,
    })

    current_params, stage3_result, stage3_sim, stage3_residual = _stage_three_refinement(
        experimental_image,
        miller,
        intensities,
        image_size,
        current_params,
        var_names,
        simulator,
        rois,
        measured_dict,
        cfg=stage3_cfg,
    )
    history.append({name: current_params[name] for name in var_names})
    stage_summaries.append({
        'stage': 'level3',
        'cost': float(np.sum(stage3_result.fun ** 2)),
        'nfev': stage3_result.nfev,
    })

    final_residual = stage3_residual(np.array([current_params[name] for name in var_names]))

    return IterativeRefinementResult(
        x=np.array([current_params[name] for name in var_names], dtype=float),
        fun=final_residual,
        success=bool(stage1_result.success and stage2_result.success and stage3_result.success),
        message='; '.join(filter(None, [stage1_result.message, stage2_result.message, stage3_result.message])),
        best_params=dict(current_params),
        history=history,
        stage_summaries=stage_summaries,
    )


def build_measured_dict(measured_peaks):
    """
    Convert a list of measured-peak dicts into a mapping
    from (h,k,l) -> list of (x,y) positions.
    """
    measured_dict = {}
    for p in measured_peaks:
        if isinstance(p, dict) and 'label' in p:
            h, k, l = map(int, p['label'].split(','))
            x, y = float(p['x']), float(p['y'])
        else:
            h, k, l, x, y = p
        measured_dict.setdefault((h, k, l), []).append((x, y))
    return measured_dict


def simulate_and_compare_hkl(
    miller,
    intensities,
    image_size,
    params,
    measured_peaks,
    pixel_tol=np.inf,
):
    """Simulate reflections and pair them with measured peak positions.

    The routine performs a full-pattern simulation using
    :func:`process_peaks_parallel`, then, for each Miller index present in
    ``measured_peaks``, finds the closest simulated peak positions.  Only
    matches within ``pixel_tol`` pixels are retained.

    Returns
    -------
    distances : :class:`numpy.ndarray`
        1D array with the distance (in pixels) between each matched
        simulated and measured peak.
    sim_coords : list[tuple[float, float]]
        The coordinates of the simulated peaks that were matched.
    meas_coords : list[tuple[float, float]]
        The coordinates of the measured peaks corresponding to
        ``sim_coords``.
    sim_millers : list[tuple[int, int, int]]
        Miller indices associated with each entry in ``sim_coords``.
    meas_millers : list[tuple[int, int, int]]
        Miller indices associated with each entry in ``meas_coords``.
    """
    measured_dict = build_measured_dict(measured_peaks)
    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    # Unpack geometry & mosaic parameters
    a = params['a']; c = params['c']
    dist = params['corto_detector']
    gamma = params['gamma']; Gamma = params['Gamma']
    chi   = params['chi']; psi = params.get('psi', 0.0)
    zs    = params['zs']; zb    = params['zb']
    debye_x = params['debye_x']; debye_y = params['debye_y']
    n2    = params['n2']
    center = params['center']
    theta_initial = params['theta_initial']

    mosaic = params['mosaic_params']
    wavelength_array = mosaic.get('wavelength_array')
    if wavelength_array is None:
        wavelength_array = mosaic.get('wavelength_i_array')

    # Full-pattern simulation
    updated_image, maxpos, _, _, _ = process_peaks_parallel(
        miller, intensities, image_size,
        a, c, wavelength_array,
        sim_buffer, dist,
        gamma, Gamma, chi, psi,
        zs, zb, n2,
        mosaic['beam_x_array'],
        mosaic['beam_y_array'],
        mosaic['theta_array'],
        mosaic['phi_array'],
        mosaic['sigma_mosaic_deg'],
        mosaic['gamma_mosaic_deg'],
        mosaic['eta'],
        wavelength_array,
        debye_x, debye_y,
        center, theta_initial,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=0
    )

    distances: list[float] = []
    sim_coords: list[tuple[float, float]] = []
    meas_coords: list[tuple[float, float]] = []
    sim_millers: list[tuple[int, int, int]] = []
    meas_millers: list[tuple[int, int, int]] = []

    for i, (H, K, L) in enumerate(miller):
        if (H, K, L) not in measured_dict:
            continue
        candidates = measured_dict[(H, K, L)]
        I0, x0, y0, I1, x1, y1 = maxpos[i]
        for x, y in ((x0, y0), (x1, y1)):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            ds = [np.hypot(x - mx, y - my) for mx, my in candidates]
            idx = int(np.argmin(ds))
            d = ds[idx]
            if d <= pixel_tol:
                sim_coords.append((x, y))
                meas_coords.append(candidates[idx])
                distances.append(d)
                hkl_int = (int(round(H)), int(round(K)), int(round(L)))
                sim_millers.append(hkl_int)
                meas_millers.append(hkl_int)

    return (
        np.array(distances, dtype=float),
        sim_coords,
        meas_coords,
        sim_millers,
        meas_millers,
    )


def compute_peak_position_error_geometry_local(
    gamma, Gamma, dist, theta_initial, zs, zb, chi, a, c,
    center_x, center_y, measured_peaks,
    miller, intensities, image_size, mosaic_params, n2,
    psi, debye_x, debye_y, wavelength, pixel_tol=np.inf
):
    """
    Objective for DE: returns the 1D array of distances for all matched peaks.
    """
    params = {
        'gamma': gamma,
        'Gamma': Gamma,
        'corto_detector': dist,
        'theta_initial': theta_initial,
        'zs': zs,
        'zb': zb,
        'chi': chi,
        'a': a,
        'c': c,
        'center': (center_x, center_y),
        'lambda': wavelength,
        'n2': n2,
        'psi': psi,
        'debye_x': debye_x,
        'debye_y': debye_y,
        'mosaic_params': mosaic_params
    }
    D, *_ = simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks,
        pixel_tol,
    )
    return D


def fit_geometry_parameters(
    miller, intensities, image_size,
    params, measured_peaks, var_names, pixel_tol=np.inf,
    *, experimental_image: Optional[np.ndarray] = None,
    refinement_config: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    Least-squares fit for a subset of geometry parameters.
    var_names is a list of keys in `params` to optimize.
    """

    if experimental_image is not None:
        return iterative_refinement(
            experimental_image,
            miller,
            intensities,
            image_size,
            params,
            var_names=var_names,
            measured_peaks=measured_peaks,
            config=refinement_config,
        )

    def cost_fn(x):
        local = params.copy()
        for name, v in zip(var_names, x):
            local[name] = v
        args = [
            local['gamma'], local['Gamma'], local['corto_detector'],
            local['theta_initial'], local['zs'], local['zb'],
            local['chi'], local['a'], local['c'],
            local['center'][0], local['center'][1]
        ]
        D = compute_peak_position_error_geometry_local(
            *args,
            measured_peaks=measured_peaks,
            miller=miller,
            intensities=intensities,
            image_size=image_size,
            mosaic_params=params['mosaic_params'],
            n2=params['n2'],
            psi=params.get('psi', 0.0),
            debye_x=params['debye_x'],
            debye_y=params['debye_y'],
            wavelength=params['lambda'],
            pixel_tol=pixel_tol
        )
        return D

    x0 = [params[name] for name in var_names]
    res = least_squares(cost_fn, x0)
    return res


def run_optimization_positions_geometry_local(
    miller, intensities, image_size,
    initial_params, bounds, measured_peaks
):
    """
    Global optimization (Differential Evolution) over geometry + beam center.
    bounds is list of (min,max) for [gamma, Gamma, dist, theta_i,
    zs, zb, chi, a, c, center_x, center_y].
    """
    def obj_glob(x):
        return np.sum(compute_peak_position_error_geometry_local(
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8],
            x[9], x[10],
            measured_peaks,
            miller, intensities, image_size,
            initial_params['mosaic_params'],
            initial_params['n2'],
            initial_params.get('psi', 0.0),
            initial_params['debye_x'],
            initial_params['debye_y'],
            initial_params['lambda'],
            pixel_tol=np.inf
        ))
    res = differential_evolution(obj_glob, bounds, maxiter=200, popsize=15)
    return res
