"""Support-normalized rod/profile integration helpers."""

from __future__ import annotations

import numpy as np

from ra_sim.simulation.exact_cake_portable import raw_phi_to_gui_phi


def caked_field_to_gui_phi(
    field: object,
    raw_azimuth_deg: object,
    radial_deg: object,
    *,
    phi_min_deg: float = -180.0,
    phi_max_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``field`` reordered/filtered like ``prepare_gui_phi_display``."""

    field_array = np.asarray(field, dtype=np.float64)
    raw_phi = np.asarray(raw_azimuth_deg, dtype=np.float64).reshape(-1)
    radial = np.asarray(radial_deg, dtype=np.float64).reshape(-1)
    if field_array.ndim != 2:
        raise ValueError("field must be a 2D caked array.")
    if field_array.shape != (raw_phi.size, radial.size):
        raise ValueError("field shape must match raw azimuth and radial axes.")

    gui_phi = np.asarray(raw_phi_to_gui_phi(raw_phi), dtype=np.float64)
    order = np.argsort(gui_phi)
    gui_phi = gui_phi[order]
    sorted_field = field_array[order, :]

    phi_min = float(phi_min_deg)
    phi_max = float(phi_max_deg)
    if phi_min <= phi_max:
        phi_mask = (gui_phi >= phi_min) & (gui_phi <= phi_max)
    else:
        phi_mask = (gui_phi >= phi_min) | (gui_phi <= phi_max)
    return sorted_field[phi_mask, :], radial.copy(), gui_phi[phi_mask]


def _array_2d(name: str, value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return array


def _same_shape_optional(
    name: str,
    value: object | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} shape must match image.shape.")
    return array


def _bool_mask(value: object, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError("mask shape must match image.shape.")
    return array


def _coord_edges(value: object) -> np.ndarray:
    edges = np.asarray(value, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError(
            "coord_edges must be a strictly increasing 1D array with at least 2 values."
        )
    return edges


def _safe_sum(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nansum(values))


def _fill_nan(size: int) -> np.ndarray:
    return np.full(size, np.nan, dtype=np.float64)


def _nan_to_zero(values: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(values), 0.0, values)


def _bincount_float(
    bins: np.ndarray,
    values: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    if bins.size <= 0:
        return np.zeros(bin_count, dtype=np.float64)
    return np.bincount(
        bins.astype(np.intp, copy=False),
        weights=np.asarray(values, dtype=np.float64),
        minlength=bin_count,
    )[:bin_count].astype(np.float64, copy=False)


def _normalize_mode_source(
    *,
    signal_sum: np.ndarray | None,
    normalization_sum: np.ndarray | None,
    acceptance: np.ndarray | None,
) -> str:
    if (signal_sum is None) != (normalization_sum is None):
        raise ValueError("signal_sum and normalization_sum must be supplied together.")
    if signal_sum is not None and normalization_sum is not None:
        return "sum_normalization"
    if acceptance is not None:
        return "acceptance"
    return "pixel_count"


def binned_caked_mask_profile(
    *,
    image: object,
    coord_map: object,
    coord_edges: object,
    mask: object,
    model: object | None = None,
    signal_sum: object | None = None,
    normalization_sum: object | None = None,
    acceptance: object | None = None,
    theta_map: object | None = None,
    coord_name: str = "qz",
) -> dict[str, np.ndarray]:
    """Bin a caked mask profile and return support-normalized intensity densities."""

    image_array = _array_2d("image", image)
    shape = image_array.shape
    coord_array = _same_shape_optional("coord_map", coord_map, shape)
    if coord_array is None:
        raise ValueError("coord_map is required.")
    mask_array = _bool_mask(mask, shape)
    model_array = _same_shape_optional("model", model, shape)
    signal_array = _same_shape_optional("signal_sum", signal_sum, shape)
    normalization_array = _same_shape_optional("normalization_sum", normalization_sum, shape)
    acceptance_array = _same_shape_optional("acceptance", acceptance, shape)
    theta_array = _same_shape_optional("theta_map", theta_map, shape)
    edges = _coord_edges(coord_edges)

    source = _normalize_mode_source(
        signal_sum=signal_array,
        normalization_sum=normalization_array,
        acceptance=acceptance_array,
    )
    bin_count = int(edges.size - 1)
    pixel_count = np.zeros(bin_count, dtype=np.int64)
    acceptance_sum = _fill_nan(bin_count)
    background_sum = _fill_nan(bin_count)
    fit_sum = _fill_nan(bin_count)
    background_mean = _fill_nan(bin_count)
    fit_mean = _fill_nan(bin_count)
    background_weighted_sum = _fill_nan(bin_count)
    fit_weighted_sum = _fill_nan(bin_count)
    background_density = _fill_nan(bin_count)
    fit_density = _fill_nan(bin_count)
    two_theta_min = _fill_nan(bin_count)
    two_theta_max = _fill_nan(bin_count)
    two_theta_mean = _fill_nan(bin_count)

    coord_flat = coord_array.reshape(-1)
    mask_flat = mask_array.reshape(-1)
    bin_index = np.searchsorted(edges, coord_flat, side="right") - 1
    finite_coord = np.isfinite(coord_flat)
    last_edge = finite_coord & (coord_flat == edges[-1])
    if np.any(last_edge):
        bin_index[last_edge] = bin_count - 1
    support = (
        mask_flat
        & finite_coord
        & (bin_index >= 0)
        & (bin_index < bin_count)
    )
    support_bins = bin_index[support].astype(np.intp, copy=False)
    if support_bins.size > 0:
        pixel_count = np.bincount(support_bins, minlength=bin_count)[:bin_count].astype(
            np.int64,
            copy=False,
        )
    has_support = pixel_count > 0

    image_flat = image_array.reshape(-1)
    background_sum_all = _bincount_float(
        support_bins,
        _nan_to_zero(image_flat[support]),
        bin_count,
    )
    background_sum[has_support] = background_sum_all[has_support]
    background_mean[has_support] = (
        background_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
    )

    if model_array is not None:
        model_flat = model_array.reshape(-1)
        fit_sum_all = _bincount_float(
            support_bins,
            _nan_to_zero(model_flat[support]),
            bin_count,
        )
        fit_sum[has_support] = fit_sum_all[has_support]
        fit_mean[has_support] = (
            fit_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
        )
    else:
        model_flat = None

    if theta_array is not None:
        theta_flat = theta_array.reshape(-1)
        theta_support = support & np.isfinite(theta_flat)
        theta_bins = bin_index[theta_support].astype(np.intp, copy=False)
        theta_values = theta_flat[theta_support]
        if theta_bins.size > 0:
            theta_min = np.full(bin_count, np.inf, dtype=np.float64)
            theta_max = np.full(bin_count, -np.inf, dtype=np.float64)
            np.minimum.at(theta_min, theta_bins, theta_values)
            np.maximum.at(theta_max, theta_bins, theta_values)
            theta_count = np.bincount(theta_bins, minlength=bin_count)[:bin_count]
            theta_sum = _bincount_float(theta_bins, theta_values, bin_count)
            valid_theta = theta_count > 0
            two_theta_min[valid_theta] = theta_min[valid_theta]
            two_theta_max[valid_theta] = theta_max[valid_theta]
            two_theta_mean[valid_theta] = theta_sum[valid_theta] / theta_count[
                valid_theta
            ].astype(np.float64)

    if source == "sum_normalization":
        signal_flat = signal_array.reshape(-1)
        normalization_flat = normalization_array.reshape(-1)
        weighted = (
            support
            & np.isfinite(signal_flat)
            & np.isfinite(normalization_flat)
            & (normalization_flat > 0.0)
        )
        weighted_bins = bin_index[weighted].astype(np.intp, copy=False)
        acc_all = _bincount_float(weighted_bins, normalization_flat[weighted], bin_count)
        sig_all = _bincount_float(weighted_bins, signal_flat[weighted], bin_count)
        acceptance_sum[has_support] = acc_all[has_support]
        background_weighted_sum[has_support] = sig_all[has_support]
        positive_acceptance = has_support & (acc_all > 0.0)
        background_density[positive_acceptance] = (
            sig_all[positive_acceptance] / acc_all[positive_acceptance]
        )
        if model_flat is not None:
            fit_sig_all = _bincount_float(
                weighted_bins,
                _nan_to_zero(model_flat[weighted] * normalization_flat[weighted]),
                bin_count,
            )
            fit_weighted_sum[has_support] = fit_sig_all[has_support]
            fit_density[positive_acceptance] = (
                fit_sig_all[positive_acceptance] / acc_all[positive_acceptance]
            )
    elif source == "acceptance":
        acceptance_flat = acceptance_array.reshape(-1)
        weighted = support & np.isfinite(acceptance_flat) & (acceptance_flat > 0.0)
        weighted_bins = bin_index[weighted].astype(np.intp, copy=False)
        acc_all = _bincount_float(weighted_bins, acceptance_flat[weighted], bin_count)
        background_sig_all = _bincount_float(
            weighted_bins,
            _nan_to_zero(image_flat[weighted] * acceptance_flat[weighted]),
            bin_count,
        )
        acceptance_sum[has_support] = acc_all[has_support]
        background_weighted_sum[has_support] = background_sig_all[has_support]
        positive_acceptance = has_support & (acc_all > 0.0)
        background_density[positive_acceptance] = (
            background_sig_all[positive_acceptance] / acc_all[positive_acceptance]
        )
        if model_flat is not None:
            fit_sig_all = _bincount_float(
                weighted_bins,
                _nan_to_zero(model_flat[weighted] * acceptance_flat[weighted]),
                bin_count,
            )
            fit_weighted_sum[has_support] = fit_sig_all[has_support]
            fit_density[positive_acceptance] = (
                fit_sig_all[positive_acceptance] / acc_all[positive_acceptance]
            )
    else:
        acceptance_sum[~has_support] = 0.0
        acceptance_sum[has_support] = pixel_count[has_support].astype(np.float64)
        background_weighted_sum[has_support] = background_sum_all[has_support]
        background_density[has_support] = (
            background_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
        )
        if model_flat is not None:
            fit_weighted_sum[has_support] = fit_sum_all[has_support]
            fit_density[has_support] = (
                fit_sum_all[has_support] / pixel_count[has_support].astype(np.float64)
            )

    result: dict[str, np.ndarray] = {
        f"{coord_name}_bin": np.arange(1, bin_count + 1, dtype=np.int64),
        f"{coord_name}_min": edges[:-1].copy(),
        f"{coord_name}_max": edges[1:].copy(),
        f"{coord_name}_center": 0.5 * (edges[:-1] + edges[1:]),
        "pixel_count": pixel_count,
        "acceptance_sum": acceptance_sum,
        "acceptance_source": np.full(bin_count, source, dtype=object),
        "background_sum": background_sum,
        "fit_sum": fit_sum,
        "background_mean": background_mean,
        "fit_mean": fit_mean,
        "background_weighted_sum": background_weighted_sum,
        "fit_weighted_sum": fit_weighted_sum,
        "background_density": background_density,
        "fit_density": fit_density,
    }
    if theta_array is not None:
        result.update(
            {
                "two_theta_min": two_theta_min,
                "two_theta_max": two_theta_max,
                "two_theta_mean": two_theta_mean,
            }
        )
    return result


def qz_profile_from_caked_mask(
    *,
    image: object,
    qz_map: object,
    qz_edges: object,
    mask: object,
    model: object | None = None,
    signal_sum: object | None = None,
    normalization_sum: object | None = None,
    acceptance: object | None = None,
    theta_map: object | None = None,
) -> dict[str, np.ndarray]:
    """Qz-specialized wrapper for ``binned_caked_mask_profile``."""

    return binned_caked_mask_profile(
        image=image,
        coord_map=qz_map,
        coord_edges=qz_edges,
        mask=mask,
        model=model,
        signal_sum=signal_sum,
        normalization_sum=normalization_sum,
        acceptance=acceptance,
        theta_map=theta_map,
        coord_name="qz",
    )


__all__ = [
    "binned_caked_mask_profile",
    "caked_field_to_gui_phi",
    "qz_profile_from_caked_mask",
]
