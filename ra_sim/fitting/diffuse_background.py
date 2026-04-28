"""Robust diffuse-background subtraction helpers.

The functions in this module are GUI-agnostic: callers provide detector images
and angular maps, and receive plain NumPy arrays plus JSON-friendly diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, gaussian_filter1d


_MODE_ALIASES = {
    "off": "off",
    "none": "off",
    "disabled": "off",
    "false": "off",
    "0": "off",
    "radial": "radial",
    "1d": "radial",
    "radial_1d": "radial",
    "radial-plus-caked-2d": "radial_plus_caked_2d",
    "radial_plus_caked_2d": "radial_plus_caked_2d",
    "radial+caked": "radial_plus_caked_2d",
    "radial_caked": "radial_plus_caked_2d",
    "radial plus caked 2d": "radial_plus_caked_2d",
    "radial_plus_phi_blocks": "radial_plus_phi_blocks",
    "radial-plus-phi-blocks": "radial_plus_phi_blocks",
    "radial+phi": "radial_plus_phi_blocks",
    "radial_phi_blocks": "radial_plus_phi_blocks",
    "phi_blocks": "radial_plus_phi_blocks",
    "radial_plus_phi_blocks_plus_caked_2d": "radial_plus_phi_blocks_plus_caked_2d",
    "radial-plus-phi-blocks-plus-caked-2d": "radial_plus_phi_blocks_plus_caked_2d",
    "radial+phi+caked": "radial_plus_phi_blocks_plus_caked_2d",
    "radial_phi_blocks_caked": "radial_plus_phi_blocks_plus_caked_2d",
}


@dataclass(frozen=True)
class DiffuseBackgroundConfig:
    enabled: bool = False
    mode: str = "radial_plus_caked_2d"

    apply_to_fit: bool = True
    apply_to_display: bool = False
    display_mode: str = "raw"

    scale: float = 1.0
    auto_scale: bool = False

    radial_bin_width_deg: float = 0.10
    radial_quantile: float = 0.35
    radial_smooth_sigma_deg: float = 0.50

    caked_theta_window_deg: float = 1.5
    caked_phi_window_deg: float = 15.0
    caked_quantile: float = 0.35

    phi_block_theta_bin_width_deg: float = 0.75
    phi_block_phi_bin_width_deg: float = 12.0
    phi_block_quantile: float = 0.50
    phi_block_min_pixels: int = 20
    phi_block_min_coverage: float = 0.05
    phi_block_smooth_theta_bins: float = 0.75
    phi_block_smooth_phi_bins: float = 0.50
    phi_block_outlier_sigma: float = 6.0
    phi_block_interpolation: str = "nearest"
    phi_block_scale: float = 1.0
    phi_block_preserve_block_edges: bool = True

    peak_mask_sigma: float = 4.0
    peak_mask_radius_px: float = 10.0
    direct_beam_mask_radius_px: float = 35.0

    valid_min: float | None = None
    valid_max: float | None = None

    clip_for_display: bool = True
    diagnostics: bool = True

    def __post_init__(self) -> None:
        mode = normalize_diffuse_background_mode(self.mode)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "display_mode", _normalize_display_mode(self.display_mode))
        object.__setattr__(self, "radial_bin_width_deg", max(float(self.radial_bin_width_deg), 1e-6))
        object.__setattr__(self, "radial_quantile", _clip01(self.radial_quantile))
        object.__setattr__(
            self,
            "radial_smooth_sigma_deg",
            max(float(self.radial_smooth_sigma_deg), 0.0),
        )
        object.__setattr__(
            self,
            "caked_theta_window_deg",
            max(float(self.caked_theta_window_deg), 1e-6),
        )
        object.__setattr__(
            self,
            "caked_phi_window_deg",
            max(float(self.caked_phi_window_deg), 1e-6),
        )
        object.__setattr__(self, "caked_quantile", _clip01(self.caked_quantile))
        object.__setattr__(
            self,
            "phi_block_theta_bin_width_deg",
            max(float(self.phi_block_theta_bin_width_deg), 1e-6),
        )
        object.__setattr__(
            self,
            "phi_block_phi_bin_width_deg",
            max(float(self.phi_block_phi_bin_width_deg), 1e-6),
        )
        object.__setattr__(self, "phi_block_quantile", _clip01(self.phi_block_quantile))
        object.__setattr__(
            self,
            "phi_block_min_pixels",
            max(1, int(round(float(self.phi_block_min_pixels)))),
        )
        object.__setattr__(
            self,
            "phi_block_min_coverage",
            _clip01(self.phi_block_min_coverage),
        )
        object.__setattr__(
            self,
            "phi_block_smooth_theta_bins",
            max(float(self.phi_block_smooth_theta_bins), 0.0),
        )
        object.__setattr__(
            self,
            "phi_block_smooth_phi_bins",
            max(float(self.phi_block_smooth_phi_bins), 0.0),
        )
        object.__setattr__(
            self,
            "phi_block_outlier_sigma",
            max(float(self.phi_block_outlier_sigma), 0.0),
        )
        interpolation = str(self.phi_block_interpolation).strip().lower().replace("-", "_")
        if interpolation not in {"nearest", "linear"}:
            interpolation = "nearest"
        object.__setattr__(self, "phi_block_interpolation", interpolation)
        object.__setattr__(self, "phi_block_scale", float(self.phi_block_scale))
        object.__setattr__(
            self,
            "phi_block_preserve_block_edges",
            bool(self.phi_block_preserve_block_edges),
        )
        object.__setattr__(self, "peak_mask_sigma", max(float(self.peak_mask_sigma), 0.0))
        object.__setattr__(
            self,
            "peak_mask_radius_px",
            max(float(self.peak_mask_radius_px), 0.0),
        )
        object.__setattr__(
            self,
            "direct_beam_mask_radius_px",
            max(float(self.direct_beam_mask_radius_px), 0.0),
        )
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "enabled", bool(self.enabled and mode != "off"))


def normalize_diffuse_background_mode(mode: object) -> str:
    """Return a canonical diffuse-background mode name."""

    text = str(mode if mode is not None else "").strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("-", "_")
    if text in _MODE_ALIASES:
        return _MODE_ALIASES[text]
    plus_text = text.replace("_plus_", "+").replace("_2d", "")
    if plus_text in _MODE_ALIASES:
        return _MODE_ALIASES[plus_text]
    return "off" if text in {"", "off"} else "radial_plus_caked_2d"


def diffuse_background_config_from_mapping(
    mapping: Mapping[str, object] | None,
) -> DiffuseBackgroundConfig:
    """Build a config from a loose mapping while preserving forward compatibility."""

    if not isinstance(mapping, Mapping):
        return DiffuseBackgroundConfig()
    defaults = DiffuseBackgroundConfig()
    values: dict[str, object] = {}
    field_map = {field.name: field for field in fields(DiffuseBackgroundConfig)}
    for key, field in field_map.items():
        if key not in mapping:
            continue
        raw_value = mapping.get(key)
        default_value = getattr(defaults, key)
        if isinstance(default_value, bool):
            values[key] = _coerce_bool(raw_value, default=default_value)
        elif isinstance(default_value, float) or key in {"valid_min", "valid_max"}:
            values[key] = _coerce_optional_float(raw_value, default=default_value)
        elif isinstance(default_value, str):
            values[key] = str(raw_value)
        else:
            values[key] = raw_value
    if "mode" in values:
        values["mode"] = normalize_diffuse_background_mode(values["mode"])
    if values.get("mode") == "off":
        values["enabled"] = False
    return DiffuseBackgroundConfig(**values)


def diffuse_background_config_to_mapping(
    config: DiffuseBackgroundConfig,
) -> dict[str, object]:
    """Return a stable primitive mapping for persistence and signatures."""

    cfg = config if isinstance(config, DiffuseBackgroundConfig) else DiffuseBackgroundConfig()
    return {field.name: getattr(cfg, field.name) for field in fields(DiffuseBackgroundConfig)}


def build_detector_valid_mask(
    image: np.ndarray,
    config: DiffuseBackgroundConfig,
) -> np.ndarray:
    """Return finite pixels that satisfy optional intensity bounds."""

    img = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(img)
    if config.valid_min is not None:
        valid &= img >= float(config.valid_min)
    if config.valid_max is not None:
        valid &= img <= float(config.valid_max)
    return np.asarray(valid, dtype=bool)


def build_peak_exclusion_mask(
    image: np.ndarray,
    valid_mask: np.ndarray,
    config: DiffuseBackgroundConfig,
    *,
    direct_beam_center_rc: tuple[float, float] | None = None,
    simulated_peaks_rc: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Return pixels excluded from background fitting."""

    img = np.asarray(image, dtype=np.float64)
    valid = _as_bool_mask(valid_mask, img.shape)
    exclusion = ~valid
    if img.ndim != 2 or img.size <= 0:
        return np.asarray(exclusion, dtype=bool)

    valid_values = img[valid & np.isfinite(img)]
    if valid_values.size > 0:
        baseline = float(np.median(valid_values))
    else:
        baseline = 0.0
    work = np.where(valid, img, baseline).astype(np.float64, copy=False)
    broad_sigma = max(float(config.peak_mask_radius_px), 6.0)
    broad = gaussian_filter(work, sigma=broad_sigma, mode="nearest")
    residual = work - broad
    residual_values = residual[valid & np.isfinite(residual)]
    if residual_values.size > 0:
        residual_center = float(np.median(residual_values))
        mad = float(np.median(np.abs(residual_values - residual_center)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = float(np.std(residual_values))
    else:
        residual_center = 0.0
        sigma = 0.0
    if np.isfinite(sigma) and sigma > 1e-12:
        candidate = valid & (residual > residual_center + float(config.peak_mask_sigma) * sigma)
    else:
        candidate = np.zeros(img.shape, dtype=bool)

    peak_radius = int(max(0, round(float(config.peak_mask_radius_px))))
    peak_struct = _disk_structure(peak_radius)
    if np.any(candidate):
        exclusion |= binary_dilation(candidate, structure=peak_struct, border_value=0)

    simulated_mask = np.zeros(img.shape, dtype=bool)
    for point in simulated_peaks_rc or ():
        try:
            row = int(round(float(point[0])))
            col = int(round(float(point[1])))
        except Exception:
            continue
        if 0 <= row < img.shape[0] and 0 <= col < img.shape[1]:
            simulated_mask[row, col] = True
    if np.any(simulated_mask):
        exclusion |= binary_dilation(simulated_mask, structure=peak_struct, border_value=0)

    if direct_beam_center_rc is not None:
        try:
            center_r = float(direct_beam_center_rc[0])
            center_c = float(direct_beam_center_rc[1])
        except Exception:
            center_r = center_c = float("nan")
        if np.isfinite(center_r) and np.isfinite(center_c):
            rr, cc = np.ogrid[: img.shape[0], : img.shape[1]]
            radius = max(float(config.direct_beam_mask_radius_px), 0.0)
            exclusion |= (rr - center_r) ** 2 + (cc - center_c) ** 2 <= radius * radius

    return np.asarray(exclusion, dtype=bool)


def estimate_radial_background_native(
    image: np.ndarray,
    two_theta_deg: np.ndarray,
    valid_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    config: DiffuseBackgroundConfig,
) -> dict[str, object]:
    """Fit a smooth robust radial profile and evaluate it on detector pixels."""

    img = np.asarray(image, dtype=np.float64)
    two_theta = np.asarray(two_theta_deg, dtype=np.float64)
    if img.shape != two_theta.shape:
        raise ValueError("image and two_theta_deg must have matching shapes")
    valid = _as_bool_mask(valid_mask, img.shape)
    exclusion = _as_bool_mask(exclusion_mask, img.shape)
    fit_mask = valid & ~exclusion & np.isfinite(img) & np.isfinite(two_theta)
    model = np.full(img.shape, np.nan, dtype=np.float64)

    if not np.any(fit_mask):
        radial_profile = np.array([0.0], dtype=np.float64)
        radial_centers = np.array([0.0], dtype=np.float64)
        model[valid] = 0.0
        return {
            "model": model,
            "radial_profile": radial_profile,
            "radial_bin_centers_deg": radial_centers,
            "radial_bin_counts": np.array([0], dtype=np.int64),
            "diagnostics": {"radial_bin_count": 1, "fit_pixel_count": 0},
        }

    theta_values = two_theta[fit_mask]
    image_values = img[fit_mask]
    width = max(float(config.radial_bin_width_deg), 1.0e-6)
    theta_min = math_floor(float(np.nanmin(theta_values)), width)
    theta_max = math_ceil(float(np.nanmax(theta_values)), width)
    if not np.isfinite(theta_min) or not np.isfinite(theta_max) or theta_max <= theta_min:
        theta_min = float(np.nanmin(theta_values))
        theta_max = theta_min + width
    edges = np.arange(theta_min, theta_max + width * 1.5, width, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([theta_min, theta_min + width], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.searchsorted(edges, theta_values, side="right") - 1
    in_range = (bin_index >= 0) & (bin_index < centers.size)
    bin_index = bin_index[in_range]
    image_values = image_values[in_range]

    profile = np.full(centers.shape, np.nan, dtype=np.float64)
    counts = np.zeros(centers.shape, dtype=np.int64)
    if bin_index.size > 0:
        order = np.argsort(bin_index, kind="stable")
        sorted_bins = bin_index[order]
        sorted_values = image_values[order]
        starts = np.r_[0, np.nonzero(np.diff(sorted_bins))[0] + 1]
        ends = np.r_[starts[1:], sorted_bins.size]
        q = _clip01(config.radial_quantile)
        for start, end in zip(starts, ends, strict=False):
            idx = int(sorted_bins[start])
            vals = sorted_values[start:end]
            vals = vals[np.isfinite(vals)]
            counts[idx] = int(vals.size)
            if vals.size > 0:
                profile[idx] = float(np.quantile(vals, q))

    finite_profile = np.isfinite(profile)
    if not np.any(finite_profile):
        fallback = float(np.median(image_values[np.isfinite(image_values)])) if image_values.size else 0.0
        profile[:] = fallback
    elif not np.all(finite_profile):
        profile = np.interp(
            centers,
            centers[finite_profile],
            profile[finite_profile],
            left=float(profile[finite_profile][0]),
            right=float(profile[finite_profile][-1]),
        )

    sigma_bins = float(config.radial_smooth_sigma_deg) / width
    smooth_profile = (
        gaussian_filter1d(profile, sigma=sigma_bins, mode="nearest")
        if sigma_bins > 1.0e-9
        else profile
    )
    theta_eval = np.asarray(two_theta, dtype=np.float64)
    eval_mask = valid & np.isfinite(theta_eval)
    if centers.size == 1:
        model[eval_mask] = float(smooth_profile[0])
    else:
        model[eval_mask] = np.interp(
            theta_eval[eval_mask],
            centers,
            smooth_profile,
            left=float(smooth_profile[0]),
            right=float(smooth_profile[-1]),
        )
    return {
        "model": model,
        "radial_profile": np.asarray(smooth_profile, dtype=np.float64),
        "radial_bin_centers_deg": np.asarray(centers, dtype=np.float64),
        "radial_bin_counts": counts,
        "diagnostics": {
            "radial_bin_count": int(centers.size),
            "fit_pixel_count": int(np.count_nonzero(fit_mask)),
            "radial_bin_width_deg": float(width),
        },
    }


def estimate_caked_residual_background(
    caked_residual: np.ndarray,
    radial_axis_deg: np.ndarray,
    azimuth_axis_deg: np.ndarray,
    valid_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    config: DiffuseBackgroundConfig,
) -> dict[str, object]:
    """Fit a slow robust residual background on a caked ``(phi, 2theta)`` grid."""

    residual = np.asarray(caked_residual, dtype=np.float64)
    radial_axis = np.asarray(radial_axis_deg, dtype=np.float64).reshape(-1)
    azimuth_axis = np.asarray(azimuth_axis_deg, dtype=np.float64).reshape(-1)
    if residual.ndim != 2:
        raise ValueError("caked_residual must be 2D")
    if residual.shape != (azimuth_axis.size, radial_axis.size):
        raise ValueError("caked_residual shape must match azimuth/radial axes")

    valid = _as_bool_mask(valid_mask, residual.shape)
    exclusion = _as_bool_mask(exclusion_mask, residual.shape)
    data_mask = valid & ~exclusion & np.isfinite(residual)
    if not np.any(data_mask):
        model = np.zeros_like(residual, dtype=np.float64)
        return {
            "caked_residual_model": model,
            "diagnostics": {
                "caked_valid_fraction": 0.0,
                "caked_residual_median": 0.0,
                "caked_residual_mad": 0.0,
            },
        }

    global_values = residual[data_mask]
    global_quantile = float(np.quantile(global_values, _clip01(config.caked_quantile)))
    theta_window = max(float(config.caked_theta_window_deg), _axis_step(radial_axis))
    phi_window = max(float(config.caked_phi_window_deg), _axis_step(azimuth_axis))
    theta_stride = max(1, int(round(0.5 * theta_window / max(_axis_step(radial_axis), 1e-9))))
    phi_stride = max(1, int(round(0.5 * phi_window / max(_axis_step(azimuth_axis), 1e-9))))
    theta_indices = _coarse_indices(radial_axis.size, theta_stride)
    phi_indices = _coarse_indices(azimuth_axis.size, phi_stride)
    coarse = np.full((phi_indices.size, theta_indices.size), np.nan, dtype=np.float64)
    half_theta = 0.5 * theta_window
    half_phi = 0.5 * phi_window
    q = _clip01(config.caked_quantile)

    for pi, row_idx in enumerate(phi_indices):
        phi_center = float(azimuth_axis[int(row_idx)])
        phi_mask = _axis_window_mask(azimuth_axis, phi_center, half_phi, periodic=True)
        for ti, col_idx in enumerate(theta_indices):
            theta_center = float(radial_axis[int(col_idx)])
            theta_mask = (radial_axis >= theta_center - half_theta) & (
                radial_axis <= theta_center + half_theta
            )
            local_mask = data_mask[np.ix_(phi_mask, theta_mask)]
            if not np.any(local_mask):
                continue
            vals = residual[np.ix_(phi_mask, theta_mask)][local_mask]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                coarse[pi, ti] = float(np.quantile(vals, q))

    finite = np.isfinite(coarse)
    if not np.any(finite):
        coarse[:] = global_quantile
    elif not np.all(finite):
        coarse = _fill_2d_missing_by_interp(
            coarse,
            azimuth_axis[phi_indices],
            radial_axis[theta_indices],
            fallback=global_quantile,
        )

    theta_full = np.empty((phi_indices.size, radial_axis.size), dtype=np.float64)
    coarse_theta_axis = radial_axis[theta_indices]
    for pi in range(phi_indices.size):
        theta_full[pi, :] = np.interp(
            radial_axis,
            coarse_theta_axis,
            coarse[pi, :],
            left=float(coarse[pi, 0]),
            right=float(coarse[pi, -1]),
        )
    model = np.empty_like(residual, dtype=np.float64)
    coarse_phi_axis = azimuth_axis[phi_indices]
    for ti in range(radial_axis.size):
        model[:, ti] = np.interp(
            azimuth_axis,
            coarse_phi_axis,
            theta_full[:, ti],
            left=float(theta_full[0, ti]),
            right=float(theta_full[-1, ti]),
        )
    sigma_theta_px = max(0.0, 0.25 * theta_window / max(_axis_step(radial_axis), 1e-9))
    sigma_phi_px = max(0.0, 0.25 * phi_window / max(_axis_step(azimuth_axis), 1e-9))
    if sigma_theta_px > 1.0e-9 or sigma_phi_px > 1.0e-9:
        model = gaussian_filter(model, sigma=(sigma_phi_px, sigma_theta_px), mode="nearest")
    return {
        "caked_residual_model": np.asarray(model, dtype=np.float64),
        "diagnostics": {
            "caked_valid_fraction": float(np.count_nonzero(data_mask) / data_mask.size),
            "caked_residual_median": float(np.median(global_values)),
            "caked_residual_mad": _mad(global_values),
        },
    }


def estimate_phi_block_residual_background(
    caked_residual: np.ndarray,
    radial_axis_deg: np.ndarray,
    azimuth_axis_deg: np.ndarray,
    valid_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    config: DiffuseBackgroundConfig,
) -> dict[str, object]:
    """Fit a coarse blocky phi-dependent residual model on a caked grid."""

    residual = np.asarray(caked_residual, dtype=np.float64)
    radial_axis = np.asarray(radial_axis_deg, dtype=np.float64).reshape(-1)
    azimuth_axis = np.asarray(azimuth_axis_deg, dtype=np.float64).reshape(-1)
    if residual.ndim != 2:
        raise ValueError("caked_residual must be 2D")
    if residual.shape != (azimuth_axis.size, radial_axis.size):
        raise ValueError("caked_residual shape must match azimuth/radial axes")

    valid = _as_bool_mask(valid_mask, residual.shape)
    exclusion = _as_bool_mask(exclusion_mask, residual.shape)
    data_mask = valid & ~exclusion & np.isfinite(residual)
    theta_edges = _regular_edges_for_axis(
        radial_axis,
        max(float(config.phi_block_theta_bin_width_deg), 1e-6),
    )
    phi_edges = _regular_edges_for_axis(
        azimuth_axis,
        max(float(config.phi_block_phi_bin_width_deg), 1e-6),
    )
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    block_grid = np.full((phi_centers.size, theta_centers.size), np.nan, dtype=np.float64)
    block_counts = np.zeros(block_grid.shape, dtype=np.int64)
    block_valid_fraction = np.zeros(block_grid.shape, dtype=np.float64)

    if theta_centers.size <= 0 or phi_centers.size <= 0:
        model = np.zeros_like(residual, dtype=np.float64)
        return {
            "phi_block_model": model,
            "phi_block_grid": block_grid,
            "phi_block_counts": block_counts,
            "phi_block_valid_fraction_grid": block_valid_fraction,
            "phi_block_theta_edges_deg": theta_edges,
            "phi_block_phi_edges_deg": phi_edges,
            "diagnostics": _phi_block_empty_diagnostics(data_mask, block_grid),
        }

    theta_bin = _axis_bin_indices(radial_axis, theta_edges)
    phi_bin = _axis_bin_indices(azimuth_axis, phi_edges)
    q = _clip01(config.phi_block_quantile)
    min_pixels = max(1, int(config.phi_block_min_pixels))
    min_coverage = _clip01(config.phi_block_min_coverage)
    outlier_sigma = max(float(config.phi_block_outlier_sigma), 0.0)

    before_values = residual[data_mask]
    global_fallback = (
        float(np.nanmedian(before_values[np.isfinite(before_values)]))
        if before_values.size
        else 0.0
    )

    for pi in range(phi_centers.size):
        phi_axis_mask = phi_bin == pi
        if not np.any(phi_axis_mask):
            continue
        for ti in range(theta_centers.size):
            theta_axis_mask = theta_bin == ti
            if not np.any(theta_axis_mask):
                continue
            cell_shape_count = int(np.count_nonzero(phi_axis_mask) * np.count_nonzero(theta_axis_mask))
            if cell_shape_count <= 0:
                continue
            local_mask = data_mask[np.ix_(phi_axis_mask, theta_axis_mask)]
            vals = residual[np.ix_(phi_axis_mask, theta_axis_mask)][local_mask]
            vals = vals[np.isfinite(vals)]
            if vals.size > 0 and outlier_sigma > 0.0:
                center = float(np.median(vals))
                spread = _mad(vals)
                if np.isfinite(spread) and spread > 1.0e-12:
                    keep = np.abs(vals - center) <= outlier_sigma * spread
                    vals = vals[keep]
            count = int(vals.size)
            coverage = float(count / cell_shape_count)
            block_counts[pi, ti] = count
            block_valid_fraction[pi, ti] = coverage
            if count >= min_pixels and coverage >= min_coverage:
                block_grid[pi, ti] = float(np.quantile(vals, q))

    finite_blocks = np.isfinite(block_grid)
    if not np.any(finite_blocks):
        filled_grid = np.full_like(block_grid, global_fallback, dtype=np.float64)
    elif np.all(finite_blocks):
        filled_grid = block_grid.copy()
    else:
        filled_grid = _fill_2d_missing_by_interp(
            block_grid,
            phi_centers,
            theta_centers,
            fallback=global_fallback,
        )

    smooth_grid = filled_grid
    sigma = (
        max(float(config.phi_block_smooth_phi_bins), 0.0),
        max(float(config.phi_block_smooth_theta_bins), 0.0),
    )
    if sigma[0] > 1.0e-9 or sigma[1] > 1.0e-9:
        smooth_grid = _weighted_gaussian_smooth_2d(
            filled_grid,
            np.where(finite_blocks, np.maximum(block_counts, 1), 0.0),
            sigma,
            mode="nearest",
        )
        if not np.all(np.isfinite(smooth_grid)):
            smooth_grid = _fill_2d_missing_by_interp(
                smooth_grid,
                phi_centers,
                theta_centers,
                fallback=global_fallback,
            )

    interpolation = (
        "nearest"
        if bool(config.phi_block_preserve_block_edges)
        else str(config.phi_block_interpolation)
    )
    model = _upsample_phi_block_grid(
        smooth_grid,
        phi_centers,
        theta_centers,
        azimuth_axis,
        radial_axis,
        interpolation=interpolation,
    )
    after_values = (residual - model)[data_mask & np.isfinite(model)]
    before_mad = _mad(before_values)
    after_mad = _mad(after_values)
    diagnostics = {
        "phi_block_valid_fraction": (
            float(np.count_nonzero(data_mask) / data_mask.size) if data_mask.size else 0.0
        ),
        "phi_block_cell_count": int(block_grid.size),
        "phi_block_filled_cell_count": int(np.count_nonzero(np.isfinite(smooth_grid))),
        "phi_block_empty_cell_count": int(block_grid.size - np.count_nonzero(finite_blocks)),
        "phi_block_median_abs_before": _median_abs(before_values),
        "phi_block_median_abs_after": _median_abs(after_values),
        "phi_block_mad_before": before_mad,
        "phi_block_mad_after": after_mad,
    }
    return {
        "phi_block_model": np.asarray(model, dtype=np.float64),
        "phi_block_grid": np.asarray(smooth_grid, dtype=np.float64),
        "phi_block_raw_grid": np.asarray(block_grid, dtype=np.float64),
        "phi_block_counts": block_counts,
        "phi_block_valid_fraction_grid": block_valid_fraction,
        "phi_block_theta_edges_deg": theta_edges,
        "phi_block_phi_edges_deg": phi_edges,
        "diagnostics": diagnostics,
    }


def evaluate_caked_background_on_detector(
    caked_background: np.ndarray,
    radial_axis_deg: np.ndarray,
    azimuth_axis_deg: np.ndarray,
    two_theta_deg: np.ndarray,
    phi_deg: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a caked background model onto detector pixels."""

    caked = np.asarray(caked_background, dtype=np.float64)
    radial_axis = np.asarray(radial_axis_deg, dtype=np.float64).reshape(-1)
    azimuth_axis = np.asarray(azimuth_axis_deg, dtype=np.float64).reshape(-1)
    two_theta = np.asarray(two_theta_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)
    if caked.shape != (azimuth_axis.size, radial_axis.size):
        raise ValueError("caked_background shape must match azimuth/radial axes")
    if two_theta.shape != phi.shape:
        raise ValueError("two_theta_deg and phi_deg must have matching shapes")
    if radial_axis.size < 1 or azimuth_axis.size < 1:
        return np.full(two_theta.shape, np.nan, dtype=np.float64)

    if radial_axis.size > 1 and radial_axis[0] > radial_axis[-1]:
        radial_axis = radial_axis[::-1]
        caked = caked[:, ::-1]
    if azimuth_axis.size > 1 and azimuth_axis[0] > azimuth_axis[-1]:
        azimuth_axis = azimuth_axis[::-1]
        caked = caked[::-1, :]

    theta_flat = two_theta.reshape(-1)
    phi_flat = _wrap_phi_to_axis(phi.reshape(-1), azimuth_axis)
    out = np.full(theta_flat.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(theta_flat) & np.isfinite(phi_flat)
    finite &= theta_flat >= radial_axis[0]
    finite &= theta_flat <= radial_axis[-1]
    finite &= phi_flat >= azimuth_axis[0]
    finite &= phi_flat <= azimuth_axis[-1]
    if not np.any(finite):
        return out.reshape(two_theta.shape)

    t = theta_flat[finite]
    p = phi_flat[finite]
    ti1 = np.searchsorted(radial_axis, t, side="right")
    pi1 = np.searchsorted(azimuth_axis, p, side="right")
    ti1 = np.clip(ti1, 1, radial_axis.size - 1 if radial_axis.size > 1 else 1)
    pi1 = np.clip(pi1, 1, azimuth_axis.size - 1 if azimuth_axis.size > 1 else 1)
    ti0 = ti1 - 1
    pi0 = pi1 - 1
    if radial_axis.size == 1:
        ti0 = ti1 = np.zeros_like(ti0)
        wt = np.zeros_like(t, dtype=np.float64)
    else:
        denom_t = radial_axis[ti1] - radial_axis[ti0]
        wt = np.divide(t - radial_axis[ti0], denom_t, out=np.zeros_like(t), where=denom_t != 0.0)
    if azimuth_axis.size == 1:
        pi0 = pi1 = np.zeros_like(pi0)
        wp = np.zeros_like(p, dtype=np.float64)
    else:
        denom_p = azimuth_axis[pi1] - azimuth_axis[pi0]
        wp = np.divide(p - azimuth_axis[pi0], denom_p, out=np.zeros_like(p), where=denom_p != 0.0)
    v00 = caked[pi0, ti0]
    v01 = caked[pi0, ti1]
    v10 = caked[pi1, ti0]
    v11 = caked[pi1, ti1]
    interp = (
        (1.0 - wp) * ((1.0 - wt) * v00 + wt * v01)
        + wp * ((1.0 - wt) * v10 + wt * v11)
    )
    out[finite] = interp
    return out.reshape(two_theta.shape)


def fit_diffuse_background_native(
    image: np.ndarray,
    *,
    two_theta_deg: np.ndarray,
    phi_deg: np.ndarray | None = None,
    caked_image: np.ndarray | None = None,
    caked_radial_axis_deg: np.ndarray | None = None,
    caked_azimuth_axis_deg: np.ndarray | None = None,
    config: DiffuseBackgroundConfig,
    direct_beam_center_rc: tuple[float, float] | None = None,
    simulated_peaks_rc: Sequence[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Fit and subtract a diffuse background model on a detector image."""

    cfg = config if isinstance(config, DiffuseBackgroundConfig) else DiffuseBackgroundConfig()
    img = np.asarray(image, dtype=np.float64)
    two_theta = np.asarray(two_theta_deg, dtype=np.float64)
    if img.shape != two_theta.shape:
        raise ValueError("image and two_theta_deg must have matching shapes")
    valid_mask = build_detector_valid_mask(img, cfg)
    exclusion_mask = build_peak_exclusion_mask(
        img,
        valid_mask,
        cfg,
        direct_beam_center_rc=direct_beam_center_rc,
        simulated_peaks_rc=simulated_peaks_rc,
    )

    if not cfg.enabled or cfg.mode == "off":
        model = np.zeros_like(img, dtype=np.float64)
        corrected = img.astype(np.float64, copy=True)
        return _assemble_result(
            cfg,
            img,
            model,
            corrected,
            valid_mask,
            exclusion_mask,
            radial_profile=np.array([], dtype=np.float64),
            radial_bin_centers=np.array([], dtype=np.float64),
            caked_residual_model=None,
            diagnostics={"enabled": False, "mode": "off"},
        )

    radial_result = estimate_radial_background_native(
        img,
        two_theta,
        valid_mask,
        exclusion_mask,
        cfg,
    )
    model = np.asarray(radial_result["model"], dtype=np.float64).copy()
    radial_detector_model = model.copy()
    background_components: dict[str, object] = {"radial": radial_detector_model.copy()}
    phi_block_model_caked = None
    phi_block_model_detector = None
    phi_block_grid = None
    phi_block_counts = None
    phi_block_valid_fraction_grid = None
    phi_block_theta_edges = None
    phi_block_phi_edges = None
    after_radial_before_phi_blocks_caked = None
    after_phi_blocks_caked = None
    caked_residual_model = None
    slow_caked_detector_model = None
    caked_diagnostics: dict[str, object] = {}
    phi_block_diagnostics: dict[str, object] = {}

    if _mode_includes_phi_blocks(cfg.mode) and phi_deg is not None:
        caked_fit = _prepare_caked_residual_for_fit(
            img,
            model,
            two_theta,
            np.asarray(phi_deg, dtype=np.float64),
            valid_mask,
            exclusion_mask,
            caked_image=caked_image,
            caked_radial_axis_deg=caked_radial_axis_deg,
            caked_azimuth_axis_deg=caked_azimuth_axis_deg,
        )
        if caked_fit is not None:
            caked_residual, radial_axis, azimuth_axis, caked_valid, caked_exclusion = caked_fit
            after_radial_before_phi_blocks_caked = np.asarray(caked_residual, dtype=np.float64)
            phi_block_result = estimate_phi_block_residual_background(
                caked_residual,
                radial_axis,
                azimuth_axis,
                caked_valid,
                caked_exclusion,
                cfg,
            )
            phi_block_model_caked = np.asarray(
                phi_block_result.get("phi_block_model"),
                dtype=np.float64,
            )
            detector_phi_block = evaluate_caked_background_on_detector(
                phi_block_model_caked,
                radial_axis,
                azimuth_axis,
                two_theta,
                np.asarray(phi_deg, dtype=np.float64),
            )
            scaled_detector_phi_block = np.asarray(detector_phi_block, dtype=np.float64) * float(
                cfg.phi_block_scale
            )
            scaled_caked_phi_block = phi_block_model_caked * float(cfg.phi_block_scale)
            model = np.where(
                np.isfinite(scaled_detector_phi_block),
                model + scaled_detector_phi_block,
                model,
            )
            phi_block_model_detector = scaled_detector_phi_block
            phi_block_model_caked = scaled_caked_phi_block
            after_phi_blocks_caked = np.asarray(caked_residual, dtype=np.float64) - np.asarray(
                phi_block_model_caked,
                dtype=np.float64,
            )
            phi_block_grid = phi_block_result.get("phi_block_grid")
            phi_block_counts = phi_block_result.get("phi_block_counts")
            phi_block_valid_fraction_grid = phi_block_result.get("phi_block_valid_fraction_grid")
            phi_block_theta_edges = phi_block_result.get("phi_block_theta_edges_deg")
            phi_block_phi_edges = phi_block_result.get("phi_block_phi_edges_deg")
            background_components["phi_blocks_detector"] = phi_block_model_detector
            background_components["phi_blocks_caked"] = phi_block_model_caked
            phi_block_diagnostics = dict(phi_block_result.get("diagnostics", {}) or {})

    if _mode_includes_caked_2d(cfg.mode) and phi_deg is not None:
        caked_fit = _prepare_caked_residual_for_fit(
            img,
            model,
            two_theta,
            np.asarray(phi_deg, dtype=np.float64),
            valid_mask,
            exclusion_mask,
            caked_image=(
                caked_image
                if cfg.mode == "radial_plus_caked_2d" and phi_block_model_detector is None
                else None
            ),
            caked_radial_axis_deg=caked_radial_axis_deg,
            caked_azimuth_axis_deg=caked_azimuth_axis_deg,
        )
        if caked_fit is not None:
            caked_residual, radial_axis, azimuth_axis, caked_valid, caked_exclusion = caked_fit
            caked_result = estimate_caked_residual_background(
                caked_residual,
                radial_axis,
                azimuth_axis,
                caked_valid,
                caked_exclusion,
                cfg,
            )
            caked_residual_model = np.asarray(
                caked_result.get("caked_residual_model"),
                dtype=np.float64,
            )
            detector_caked = evaluate_caked_background_on_detector(
                caked_residual_model,
                radial_axis,
                azimuth_axis,
                two_theta,
                np.asarray(phi_deg, dtype=np.float64),
            )
            slow_caked_detector_model = detector_caked
            model = np.where(np.isfinite(detector_caked), model + detector_caked, model)
            background_components["slow_caked_detector"] = slow_caked_detector_model
            background_components["slow_caked_caked"] = caked_residual_model
            caked_diagnostics = dict(caked_result.get("diagnostics", {}) or {})

    corrected = subtract_diffuse_background(img, model, cfg)
    diagnostics = _build_diagnostics(
        img,
        corrected,
        model,
        valid_mask,
        exclusion_mask,
        cfg,
        radial_result,
        caked_diagnostics,
        phi_block_diagnostics,
    )
    return _assemble_result(
        cfg,
        img,
        model,
        corrected,
        valid_mask,
        exclusion_mask,
        radial_profile=np.asarray(radial_result["radial_profile"], dtype=np.float64),
        radial_bin_centers=np.asarray(radial_result["radial_bin_centers_deg"], dtype=np.float64),
        radial_model=radial_detector_model,
        phi_block_model_caked=phi_block_model_caked,
        phi_block_model_detector=phi_block_model_detector,
        phi_block_grid=phi_block_grid,
        phi_block_counts=phi_block_counts,
        phi_block_valid_fraction_grid=phi_block_valid_fraction_grid,
        phi_block_theta_edges=phi_block_theta_edges,
        phi_block_phi_edges=phi_block_phi_edges,
        after_radial_before_phi_blocks_caked=after_radial_before_phi_blocks_caked,
        after_phi_blocks_caked=after_phi_blocks_caked,
        caked_residual_model=caked_residual_model,
        slow_caked_detector_model=slow_caked_detector_model,
        background_components=background_components,
        diagnostics=diagnostics,
    )


def subtract_diffuse_background(
    image: np.ndarray,
    model: np.ndarray,
    config: DiffuseBackgroundConfig,
) -> np.ndarray:
    """Return signed ``image - scale * model`` without numerical clipping."""

    img = np.asarray(image, dtype=np.float64)
    mdl = np.asarray(model, dtype=np.float64)
    if img.shape != mdl.shape:
        raise ValueError("image and model must have matching shapes")
    effective_scale = _effective_scale(img, mdl, config)
    return img - effective_scale * mdl


def _assemble_result(
    config: DiffuseBackgroundConfig,
    raw: np.ndarray,
    model: np.ndarray,
    corrected: np.ndarray,
    valid_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    radial_profile: np.ndarray,
    radial_bin_centers: np.ndarray,
    radial_model: np.ndarray | None = None,
    phi_block_model_caked: np.ndarray | None = None,
    phi_block_model_detector: np.ndarray | None = None,
    phi_block_grid: object | None = None,
    phi_block_counts: object | None = None,
    phi_block_valid_fraction_grid: object | None = None,
    phi_block_theta_edges: object | None = None,
    phi_block_phi_edges: object | None = None,
    after_radial_before_phi_blocks_caked: np.ndarray | None = None,
    after_phi_blocks_caked: np.ndarray | None = None,
    caked_residual_model: np.ndarray | None = None,
    slow_caked_detector_model: np.ndarray | None = None,
    background_components: Mapping[str, object] | None = None,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    components = dict(background_components or {})
    if radial_model is not None and "radial" not in components:
        components["radial"] = radial_model
    if phi_block_model_detector is not None and "phi_blocks_detector" not in components:
        components["phi_blocks_detector"] = phi_block_model_detector
    if phi_block_model_caked is not None and "phi_blocks_caked" not in components:
        components["phi_blocks_caked"] = phi_block_model_caked
    if slow_caked_detector_model is not None and "slow_caked_detector" not in components:
        components["slow_caked_detector"] = slow_caked_detector_model
    if caked_residual_model is not None and "slow_caked_caked" not in components:
        components["slow_caked_caked"] = caked_residual_model
    return {
        "config": config,
        "raw": raw,
        "model": model,
        "corrected": corrected,
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "exclusion_mask": np.asarray(exclusion_mask, dtype=bool),
        "radial_profile": np.asarray(radial_profile, dtype=np.float64),
        "radial_bin_centers_deg": np.asarray(radial_bin_centers, dtype=np.float64),
        "radial_model": radial_model,
        "phi_block_model_caked": phi_block_model_caked,
        "phi_block_model_detector": phi_block_model_detector,
        "phi_block_grid": phi_block_grid,
        "phi_block_counts": phi_block_counts,
        "phi_block_valid_fraction_grid": phi_block_valid_fraction_grid,
        "phi_block_theta_edges_deg": phi_block_theta_edges,
        "phi_block_phi_edges_deg": phi_block_phi_edges,
        "after_radial_before_phi_blocks_caked": after_radial_before_phi_blocks_caked,
        "after_phi_blocks_caked": after_phi_blocks_caked,
        "caked_residual_model": caked_residual_model,
        "slow_caked_model_detector": slow_caked_detector_model,
        "background_components": components,
        "diagnostics": dict(diagnostics),
    }


def _build_diagnostics(
    raw: np.ndarray,
    corrected: np.ndarray,
    model: np.ndarray,
    valid_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    config: DiffuseBackgroundConfig,
    radial_result: Mapping[str, object],
    caked_diagnostics: Mapping[str, object],
    phi_block_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    valid = np.asarray(valid_mask, dtype=bool)
    excluded = np.asarray(exclusion_mask, dtype=bool)
    raw_vals = np.asarray(raw, dtype=np.float64)[valid & np.isfinite(raw)]
    corr_vals = np.asarray(corrected, dtype=np.float64)[valid & np.isfinite(corrected)]
    profile = np.asarray(radial_result.get("radial_profile", []), dtype=np.float64)
    diagnostics = {
        "mode": str(config.mode),
        "enabled": bool(config.enabled),
        "scale": float(config.scale),
        "auto_scale": bool(config.auto_scale),
        "effective_scale": _effective_scale(raw, model, config),
        "valid_fraction": float(np.count_nonzero(valid) / valid.size) if valid.size else 0.0,
        "masked_fraction": float(np.count_nonzero(excluded) / excluded.size) if excluded.size else 0.0,
        "radial_bin_count": int(profile.size),
        "radial_profile_min": _finite_stat(profile, np.min),
        "radial_profile_median": _finite_stat(profile, np.median),
        "radial_profile_max": _finite_stat(profile, np.max),
        "raw_median": _finite_stat(raw_vals, np.median),
        "raw_mad": _mad(raw_vals),
        "corrected_median": _finite_stat(corr_vals, np.median),
        "corrected_mad": _mad(corr_vals),
        "negative_fraction": (
            float(np.count_nonzero(corr_vals < 0.0) / corr_vals.size) if corr_vals.size else 0.0
        ),
        "phi_block_enabled": bool(_mode_includes_phi_blocks(config.mode)),
        "phi_block_scale": float(config.phi_block_scale),
        "phi_block_theta_bin_width_deg": float(config.phi_block_theta_bin_width_deg),
        "phi_block_phi_bin_width_deg": float(config.phi_block_phi_bin_width_deg),
        "phi_block_quantile": float(config.phi_block_quantile),
    }
    diagnostics.update(dict(caked_diagnostics or {}))
    diagnostics.update(dict(phi_block_diagnostics or {}))
    before = float(diagnostics.get("phi_block_mad_before", 0.0) or 0.0)
    after = float(diagnostics.get("phi_block_mad_after", 0.0) or 0.0)
    tiny = 1.0e-12
    diagnostics["phi_block_reduction_fraction"] = (
        float(1.0 - after / max(before, tiny)) if before > tiny else 0.0
    )
    return diagnostics


def _prepare_caked_residual_for_fit(
    image: np.ndarray,
    radial_model: np.ndarray,
    two_theta: np.ndarray,
    phi: np.ndarray,
    valid_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    caked_image: np.ndarray | None,
    caked_radial_axis_deg: np.ndarray | None,
    caked_azimuth_axis_deg: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if caked_radial_axis_deg is None or caked_azimuth_axis_deg is None:
        return None
    radial_axis = np.asarray(caked_radial_axis_deg, dtype=np.float64).reshape(-1)
    azimuth_axis = np.asarray(caked_azimuth_axis_deg, dtype=np.float64).reshape(-1)
    if radial_axis.size <= 1 or azimuth_axis.size <= 1:
        return None
    if caked_image is not None:
        caked = np.asarray(caked_image, dtype=np.float64)
        if caked.shape != (azimuth_axis.size, radial_axis.size):
            return None
        centers = np.asarray(radial_axis, dtype=np.float64)
        finite_model = np.isfinite(radial_model) & np.isfinite(two_theta)
        if not np.any(finite_model):
            radial_on_cake = np.zeros_like(radial_axis)
        else:
            radial_on_cake = _radial_model_on_axis(two_theta, radial_model, radial_axis)
        residual = caked - radial_on_cake[None, :]
        caked_valid = np.isfinite(caked)
        caked_exclusion = ~caked_valid
        return residual, centers, azimuth_axis, caked_valid, caked_exclusion

    residual_detector = np.asarray(image, dtype=np.float64) - np.asarray(radial_model, dtype=np.float64)
    return _cake_detector_residual(
        residual_detector,
        two_theta,
        phi,
        radial_axis,
        azimuth_axis,
        valid_mask,
        exclusion_mask,
    )


def _cake_detector_residual(
    residual_detector: np.ndarray,
    two_theta: np.ndarray,
    phi: np.ndarray,
    radial_axis: np.ndarray,
    azimuth_axis: np.ndarray,
    valid_mask: np.ndarray,
    exclusion_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    mask = (
        np.asarray(valid_mask, dtype=bool)
        & ~np.asarray(exclusion_mask, dtype=bool)
        & np.isfinite(residual_detector)
        & np.isfinite(two_theta)
        & np.isfinite(phi)
    )
    if not np.any(mask):
        return None
    theta_vals = two_theta[mask]
    phi_vals = _wrap_phi_to_axis(phi[mask], azimuth_axis)
    vals = residual_detector[mask]
    theta_edges = _axis_edges(radial_axis)
    phi_edges = _axis_edges(azimuth_axis)
    ti = np.searchsorted(theta_edges, theta_vals, side="right") - 1
    pi = np.searchsorted(phi_edges, phi_vals, side="right") - 1
    ok = (ti >= 0) & (ti < radial_axis.size) & (pi >= 0) & (pi < azimuth_axis.size)
    if not np.any(ok):
        return None
    ti = ti[ok].astype(np.int64, copy=False)
    pi = pi[ok].astype(np.int64, copy=False)
    vals = vals[ok]
    flat = pi * radial_axis.size + ti
    sum_grid = np.bincount(flat, weights=vals, minlength=azimuth_axis.size * radial_axis.size)
    count_grid = np.bincount(flat, minlength=azimuth_axis.size * radial_axis.size)
    sum_grid = sum_grid.reshape((azimuth_axis.size, radial_axis.size))
    count_grid = count_grid.reshape((azimuth_axis.size, radial_axis.size))
    caked = np.divide(
        sum_grid,
        count_grid,
        out=np.full_like(sum_grid, np.nan, dtype=np.float64),
        where=count_grid > 0,
    )
    caked_valid = count_grid > 0
    caked_exclusion = ~caked_valid
    return caked, radial_axis, azimuth_axis, caked_valid, caked_exclusion


def _radial_model_on_axis(
    two_theta: np.ndarray,
    radial_model: np.ndarray,
    radial_axis: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(two_theta) & np.isfinite(radial_model)
    if not np.any(mask):
        return np.zeros_like(radial_axis, dtype=np.float64)
    theta = two_theta[mask].reshape(-1)
    model = radial_model[mask].reshape(-1)
    order = np.argsort(theta, kind="stable")
    theta = theta[order]
    model = model[order]
    unique_theta, start = np.unique(theta, return_index=True)
    unique_model = np.array(
        [np.median(model[start[i] : start[i + 1] if i + 1 < start.size else model.size]) for i in range(start.size)],
        dtype=np.float64,
    )
    return np.interp(radial_axis, unique_theta, unique_model, left=unique_model[0], right=unique_model[-1])


def _effective_scale(
    image: np.ndarray,
    model: np.ndarray,
    config: DiffuseBackgroundConfig,
) -> float:
    scale = float(config.scale)
    if not bool(config.auto_scale):
        return scale
    img = np.asarray(image, dtype=np.float64)
    mdl = np.asarray(model, dtype=np.float64)
    mask = np.isfinite(img) & np.isfinite(mdl) & (np.abs(mdl) > 1.0e-12)
    if not np.any(mask):
        return scale
    denom = float(np.sum(mdl[mask] * mdl[mask]))
    if not np.isfinite(denom) or denom <= 1.0e-12:
        return scale
    auto = float(np.sum(img[mask] * mdl[mask]) / denom)
    if not np.isfinite(auto) or auto <= 0.0:
        return scale
    return float(scale * auto)


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        try:
            return bool(int(round(float(value))))
        except Exception:
            return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(default)


def _coerce_optional_float(value: object, *, default: float | None) -> float | None:
    if value is None or value == "":
        return None if default is None else default
    try:
        parsed = float(value)
    except Exception:
        return default
    return float(parsed) if np.isfinite(parsed) else default


def _normalize_display_mode(value: object) -> str:
    text = str(value if value is not None else "raw").strip().lower().replace("-", "_")
    aliases = {
        "subtracted": "subtracted",
        "corrected": "subtracted",
        "mask": "mask",
        "phi_blocks_model": "phi_block_model",
        "phi_blocks": "phi_block_model",
        "radial_plus_phi_blocks_model": "radial_plus_phi_block_model",
        "slow_caked": "slow_caked_model",
    }
    allowed = {
        "raw",
        "model",
        "residual",
        "radial_model",
        "phi_block_model",
        "radial_plus_phi_block_model",
        "slow_caked_model",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in allowed or normalized in {"subtracted", "mask"} else "raw"


def _mode_includes_phi_blocks(mode: object) -> bool:
    return normalize_diffuse_background_mode(mode) in {
        "radial_plus_phi_blocks",
        "radial_plus_phi_blocks_plus_caked_2d",
    }


def _mode_includes_caked_2d(mode: object) -> bool:
    return normalize_diffuse_background_mode(mode) in {
        "radial_plus_caked_2d",
        "radial_plus_phi_blocks_plus_caked_2d",
    }


def _clip01(value: object) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except Exception:
        return 0.5


def _as_bool_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError("mask shape does not match image shape")
    return arr


def _disk_structure(radius: int) -> np.ndarray:
    radius = int(max(0, radius))
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (yy * yy + xx * xx) <= radius * radius


def _axis_step(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return 1.0
    diffs = np.diff(np.sort(values[np.isfinite(values)]))
    diffs = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1.0e-12)]
    if diffs.size <= 0:
        return 1.0
    return float(np.median(np.abs(diffs)))


def _coarse_indices(size: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    indices = np.arange(0, int(size), stride, dtype=np.int64)
    if indices.size == 0 or indices[-1] != int(size) - 1:
        indices = np.r_[indices, int(size) - 1]
    return np.unique(indices)


def _axis_window_mask(
    axis: np.ndarray,
    center: float,
    half_width: float,
    *,
    periodic: bool = False,
) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if periodic:
        delta = ((values - float(center) + 180.0) % 360.0) - 180.0
        return np.abs(delta) <= float(half_width)
    return (values >= float(center) - float(half_width)) & (
        values <= float(center) + float(half_width)
    )


def _fill_2d_missing_by_interp(
    grid: np.ndarray,
    row_axis: np.ndarray,
    col_axis: np.ndarray,
    *,
    fallback: float,
) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float64).copy()
    for row in range(arr.shape[0]):
        finite = np.isfinite(arr[row])
        if np.any(finite):
            arr[row] = np.interp(
                col_axis,
                col_axis[finite],
                arr[row, finite],
                left=float(arr[row, finite][0]),
                right=float(arr[row, finite][-1]),
            )
    for col in range(arr.shape[1]):
        finite = np.isfinite(arr[:, col])
        if np.any(finite):
            arr[:, col] = np.interp(
                row_axis,
                row_axis[finite],
                arr[finite, col],
                left=float(arr[finite, col][0]),
                right=float(arr[finite, col][-1]),
            )
    arr[~np.isfinite(arr)] = float(fallback)
    return arr


def _axis_edges(axis: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mids[0] - values[0])
    last = values[-1] + (values[-1] - mids[-1])
    return np.r_[first, mids, last].astype(np.float64)


def _regular_edges_for_axis(axis: np.ndarray, width: float) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    bin_width = max(float(width), 1.0e-6)
    if finite.size <= 0:
        return np.array([0.0, bin_width], dtype=np.float64)
    low = math_floor(float(np.nanmin(finite)), bin_width)
    high = math_ceil(float(np.nanmax(finite)), bin_width)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite))
        high = low + bin_width
    edges = np.arange(low, high + bin_width * 1.5, bin_width, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([low, low + bin_width], dtype=np.float64)
    return edges


def _axis_bin_indices(axis: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    edge_values = np.asarray(edges, dtype=np.float64).reshape(-1)
    if edge_values.size < 2:
        return np.full(values.shape, -1, dtype=np.int64)
    indices = np.searchsorted(edge_values, values, side="right") - 1
    indices = np.where(values == edge_values[-1], edge_values.size - 2, indices)
    finite = np.isfinite(values)
    ok = finite & (indices >= 0) & (indices < edge_values.size - 1)
    return np.where(ok, indices, -1).astype(np.int64, copy=False)


def _weighted_gaussian_smooth_2d(
    values: np.ndarray,
    weights: np.ndarray,
    sigma: tuple[float, float],
    *,
    mode: tuple[str, str] | str = "nearest",
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != weight_arr.shape:
        raise ValueError("values and weights must have matching shapes")
    sigma_tuple = (max(float(sigma[0]), 0.0), max(float(sigma[1]), 0.0))
    finite = np.isfinite(arr) & np.isfinite(weight_arr) & (weight_arr > 0.0)
    if not np.any(finite):
        return np.full(arr.shape, np.nan, dtype=np.float64)
    if sigma_tuple[0] <= 1.0e-12 and sigma_tuple[1] <= 1.0e-12:
        return np.where(finite, arr, np.nan).astype(np.float64, copy=False)
    numerator = gaussian_filter(
        np.where(finite, arr * weight_arr, 0.0),
        sigma=sigma_tuple,
        mode=mode,
    )
    denominator = gaussian_filter(
        np.where(finite, weight_arr, 0.0),
        sigma=sigma_tuple,
        mode=mode,
    )
    tiny = 1.0e-12
    out = np.divide(
        numerator,
        denominator,
        out=np.full(arr.shape, np.nan, dtype=np.float64),
        where=denominator > tiny,
    )
    out[denominator <= tiny] = np.nan
    return out


def _upsample_phi_block_grid(
    block_grid: np.ndarray,
    block_phi_centers: np.ndarray,
    block_theta_centers: np.ndarray,
    azimuth_axis: np.ndarray,
    radial_axis: np.ndarray,
    *,
    interpolation: str,
) -> np.ndarray:
    grid = np.asarray(block_grid, dtype=np.float64)
    phi_centers = np.asarray(block_phi_centers, dtype=np.float64).reshape(-1)
    theta_centers = np.asarray(block_theta_centers, dtype=np.float64).reshape(-1)
    target_phi = np.asarray(azimuth_axis, dtype=np.float64).reshape(-1)
    target_theta = np.asarray(radial_axis, dtype=np.float64).reshape(-1)
    if grid.shape != (phi_centers.size, theta_centers.size):
        raise ValueError("block_grid shape must match block center axes")
    if grid.size <= 0:
        return np.zeros((target_phi.size, target_theta.size), dtype=np.float64)

    phi_order = np.argsort(phi_centers, kind="stable")
    theta_order = np.argsort(theta_centers, kind="stable")
    sorted_phi = phi_centers[phi_order]
    sorted_theta = theta_centers[theta_order]
    sorted_grid = grid[np.ix_(phi_order, theta_order)]
    sorted_grid = np.where(np.isfinite(sorted_grid), sorted_grid, 0.0)

    method = str(interpolation).strip().lower()
    if method != "linear":
        phi_idx = _nearest_axis_indices(sorted_phi, target_phi)
        theta_idx = _nearest_axis_indices(sorted_theta, target_theta)
        return sorted_grid[np.ix_(phi_idx, theta_idx)].astype(np.float64, copy=False)

    theta_full = np.empty((sorted_phi.size, target_theta.size), dtype=np.float64)
    for pi in range(sorted_phi.size):
        theta_full[pi, :] = np.interp(
            target_theta,
            sorted_theta,
            sorted_grid[pi, :],
            left=float(sorted_grid[pi, 0]),
            right=float(sorted_grid[pi, -1]),
        )
    out = np.empty((target_phi.size, target_theta.size), dtype=np.float64)
    for ti in range(target_theta.size):
        out[:, ti] = np.interp(
            target_phi,
            sorted_phi,
            theta_full[:, ti],
            left=float(theta_full[0, ti]),
            right=float(theta_full[-1, ti]),
        )
    return out


def _nearest_axis_indices(axis: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    target_values = np.asarray(targets, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return np.zeros(target_values.shape, dtype=np.int64)
    right = np.searchsorted(values, target_values, side="left")
    right = np.clip(right, 0, values.size - 1)
    left = np.clip(right - 1, 0, values.size - 1)
    choose_right = np.abs(values[right] - target_values) < np.abs(target_values - values[left])
    return np.where(choose_right, right, left).astype(np.int64, copy=False)


def _wrap_phi_to_axis(phi: np.ndarray, axis: np.ndarray) -> np.ndarray:
    values = np.asarray(phi, dtype=np.float64)
    axis_values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if axis_values.size <= 0:
        return values
    amin = float(np.nanmin(axis_values))
    amax = float(np.nanmax(axis_values))
    wrapped = ((values - amin) % 360.0) + amin
    wrapped = np.where(wrapped > amax, wrapped - 360.0, wrapped)
    return wrapped


def _finite_stat(values: np.ndarray, fn: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return 0.0
    return float(fn(arr))


def _mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return 0.0
    center = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - center)))


def _median_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return 0.0
    return float(np.median(np.abs(arr)))


def _phi_block_empty_diagnostics(
    data_mask: np.ndarray,
    block_grid: np.ndarray,
) -> dict[str, object]:
    return {
        "phi_block_valid_fraction": (
            float(np.count_nonzero(data_mask) / data_mask.size) if data_mask.size else 0.0
        ),
        "phi_block_cell_count": int(np.asarray(block_grid).size),
        "phi_block_filled_cell_count": 0,
        "phi_block_empty_cell_count": int(np.asarray(block_grid).size),
        "phi_block_median_abs_before": 0.0,
        "phi_block_median_abs_after": 0.0,
        "phi_block_mad_before": 0.0,
        "phi_block_mad_after": 0.0,
    }


def math_floor(value: float, width: float) -> float:
    return float(np.floor(float(value) / float(width)) * float(width))


def math_ceil(value: float, width: float) -> float:
    return float(np.ceil(float(value) / float(width)) * float(width))


__all__ = [
    "DiffuseBackgroundConfig",
    "build_detector_valid_mask",
    "build_peak_exclusion_mask",
    "diffuse_background_config_from_mapping",
    "diffuse_background_config_to_mapping",
    "estimate_caked_residual_background",
    "estimate_phi_block_residual_background",
    "estimate_radial_background_native",
    "evaluate_caked_background_on_detector",
    "fit_diffuse_background_native",
    "normalize_diffuse_background_mode",
    "subtract_diffuse_background",
]
