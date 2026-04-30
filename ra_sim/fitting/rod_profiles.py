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

    for idx in range(bin_count):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == bin_count - 1:
            in_bin = (coord_array >= lo) & (coord_array <= hi)
        else:
            in_bin = (coord_array >= lo) & (coord_array < hi)
        support = mask_array & np.isfinite(coord_array) & in_bin
        support_count = int(np.count_nonzero(support))
        pixel_count[idx] = support_count
        if support_count <= 0:
            if source == "pixel_count":
                acceptance_sum[idx] = 0.0
            continue

        background_sum[idx] = _safe_sum(image_array[support])
        background_mean[idx] = background_sum[idx] / float(support_count)
        if model_array is not None:
            fit_sum[idx] = _safe_sum(model_array[support])
            fit_mean[idx] = fit_sum[idx] / float(support_count)

        if theta_array is not None:
            theta_values = theta_array[support & np.isfinite(theta_array)]
            if theta_values.size > 0:
                two_theta_min[idx] = float(np.nanmin(theta_values))
                two_theta_max[idx] = float(np.nanmax(theta_values))
                two_theta_mean[idx] = float(np.nanmean(theta_values))

        if source == "sum_normalization":
            weighted = (
                support
                & np.isfinite(signal_array)
                & np.isfinite(normalization_array)
                & (normalization_array > 0.0)
            )
            acc = float(np.nansum(normalization_array[weighted]))
            sig = float(np.nansum(signal_array[weighted]))
            acceptance_sum[idx] = acc
            background_weighted_sum[idx] = sig
            if acc > 0.0:
                background_density[idx] = sig / acc
            if model_array is not None:
                fit_sig = float(
                    np.nansum(model_array[weighted] * normalization_array[weighted])
                )
                fit_weighted_sum[idx] = fit_sig
                if acc > 0.0:
                    fit_density[idx] = fit_sig / acc
            continue

        if source == "acceptance":
            weighted = support & np.isfinite(acceptance_array) & (acceptance_array > 0.0)
            acc = float(np.nansum(acceptance_array[weighted]))
            acceptance_sum[idx] = acc
            background_sig = float(np.nansum(image_array[weighted] * acceptance_array[weighted]))
            background_weighted_sum[idx] = background_sig
            if acc > 0.0:
                background_density[idx] = background_sig / acc
            if model_array is not None:
                fit_sig = float(
                    np.nansum(model_array[weighted] * acceptance_array[weighted])
                )
                fit_weighted_sum[idx] = fit_sig
                if acc > 0.0:
                    fit_density[idx] = fit_sig / acc
            continue

        acceptance_sum[idx] = float(support_count)
        background_weighted_sum[idx] = background_sum[idx]
        background_density[idx] = background_sum[idx] / float(support_count)
        if model_array is not None:
            fit_weighted_sum[idx] = fit_sum[idx]
            fit_density[idx] = fit_sum[idx] / float(support_count)

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
