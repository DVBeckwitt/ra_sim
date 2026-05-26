"""Pure helpers for geometry-fit overlay rendering and diagnostics."""

from __future__ import annotations

import math
import logging
import os
from typing import Callable, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)
GEOMETRY_AUDIT_STRICT_ENV = "RA_SIM_GEOM_AUDIT_STRICT"
GEOMETRY_DISABLE_SIM_NATIVE_REBUILD_ENV = "RA_SIM_GEOM_DISABLE_SIM_NATIVE_REBUILD"


def _parse_optional_point(value: object) -> tuple[float, float] | None:
    try:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
        if len(value) < 2:
            return None
        point = (float(value[0]), float(value[1]))
    except Exception:
        return None
    if not (math.isfinite(point[0]) and math.isfinite(point[1])):
        return None
    return point


def compute_holistic_sim_residual(
    background_image: object,
    simulation_image: object,
    *,
    mask: object | None = None,
    scale_mode: str = "least_squares",
) -> dict[str, object]:
    """Return whole-image residual metrics for ``background - scale * sim``."""

    background = np.asarray(background_image, dtype=np.float64)
    simulation = np.asarray(simulation_image, dtype=np.float64)
    if background.shape != simulation.shape:
        raise ValueError("background_image and simulation_image must have the same shape")

    valid = np.isfinite(background) & np.isfinite(simulation)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != background.shape:
            raise ValueError("mask must have the same shape as background_image")
        valid &= mask_arr

    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {
            "status": "no_valid_pixels",
            "valid_px": 0,
            "scale": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "l1": float("nan"),
        }

    bg = background[valid]
    sim = simulation[valid]
    mode = str(scale_mode or "least_squares").strip().lower()
    if mode in {"none", "fixed", "identity"}:
        scale = 1.0
    elif mode in {"least_squares", "lsq", "auto"}:
        denom = float(np.dot(sim, sim))
        scale = float(np.dot(bg, sim) / denom) if denom > 0.0 else 1.0
    else:
        raise ValueError(f"Unsupported holistic residual scale_mode {scale_mode!r}")

    residual = bg - scale * sim
    abs_residual = np.abs(residual)
    return {
        "status": "ok",
        "valid_px": int(valid_count),
        "scale": float(scale),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mae": float(np.mean(abs_residual)),
        "l1": float(np.sum(abs_residual)),
    }


def compare_holistic_sim_residuals(
    initial_metrics: Mapping[str, object] | None,
    final_metrics: Mapping[str, object] | None,
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, object]:
    """Compare initial/final whole-image residuals and flag worse final fits."""

    initial = initial_metrics or {}
    final = final_metrics or {}
    try:
        initial_rmse = float(initial.get("rmse", np.nan))
        final_rmse = float(final.get("rmse", np.nan))
    except Exception:
        initial_rmse = float("nan")
        final_rmse = float("nan")
    delta = float(final_rmse - initial_rmse)
    suspicious = bool(
        np.isfinite(initial_rmse) and np.isfinite(final_rmse) and delta > float(tolerance)
    )
    return {
        "holistic_residual_initial_rmse": initial_rmse,
        "holistic_residual_final_rmse": final_rmse,
        "holistic_residual_delta_rmse": delta,
        "holistic_fit_suspicious": suspicious,
    }


def _default_image_extent_for_shape(shape: tuple[int, int]) -> tuple[float, float, float, float]:
    rows, cols = int(shape[0]), int(shape[1])
    return (-0.5, float(cols) - 0.5, -0.5, float(rows) - 0.5)


def _pixel_index_to_display_point(
    row_index: int,
    col_index: int,
    *,
    shape: tuple[int, int],
    extent: Sequence[object] | None,
) -> tuple[float, float]:
    rows, cols = int(shape[0]), int(shape[1])
    left, right, bottom, top = (
        tuple(float(value) for value in extent)
        if extent is not None
        else _default_image_extent_for_shape((rows, cols))
    )
    x_step = (right - left) / float(cols)
    y_step = (top - bottom) / float(rows)
    return (
        float(left + (float(col_index) + 0.5) * x_step),
        float(bottom + (float(row_index) + 0.5) * y_step),
    )


def _display_point_to_pixel_index(
    point: tuple[float, float],
    *,
    shape: tuple[int, int],
    extent: Sequence[object] | None,
) -> tuple[float, float] | None:
    rows, cols = int(shape[0]), int(shape[1])
    left, right, bottom, top = (
        tuple(float(value) for value in extent)
        if extent is not None
        else _default_image_extent_for_shape((rows, cols))
    )
    x_step = (right - left) / float(cols)
    y_step = (top - bottom) / float(rows)
    if x_step == 0.0 or y_step == 0.0:
        return None
    col_float = (float(point[0]) - left) / x_step - 0.5
    row_float = (float(point[1]) - bottom) / y_step - 0.5
    return (float(row_float), float(col_float))


def probe_display_image_peak_near_point(
    image: object,
    point: Sequence[object] | None,
    *,
    extent: Sequence[object] | None = None,
    search_radius_px: int = 8,
) -> dict[str, object]:
    """Find the strongest displayed-image pixel near one marker data point."""

    marker = _parse_optional_point(point)
    if marker is None:
        return {"status": "point_unavailable"}
    try:
        image_arr = np.asarray(image, dtype=np.float64)
    except Exception:
        return {"status": "image_unavailable"}
    if image_arr.ndim != 2 or image_arr.size <= 0:
        return {"status": "image_unavailable"}
    rows, cols = int(image_arr.shape[0]), int(image_arr.shape[1])
    pixel_float = _display_point_to_pixel_index(
        marker,
        shape=(rows, cols),
        extent=extent,
    )
    if pixel_float is None:
        return {"status": "invalid_extent"}
    center_row = int(round(pixel_float[0]))
    center_col = int(round(pixel_float[1]))
    radius = max(0, int(search_radius_px))
    row_min = max(0, center_row - radius)
    row_max = min(rows - 1, center_row + radius)
    col_min = max(0, center_col - radius)
    col_max = min(cols - 1, center_col + radius)
    if row_min > row_max or col_min > col_max:
        return {
            "status": "point_outside_image",
            "point_pixel_index": (float(pixel_float[0]), float(pixel_float[1])),
        }
    window = image_arr[row_min : row_max + 1, col_min : col_max + 1]
    finite = np.isfinite(window)
    if not bool(np.any(finite)):
        return {"status": "no_finite_pixels"}
    finite_values = np.where(finite, window, -np.inf)
    flat_index = int(np.argmax(finite_values))
    local_row, local_col = np.unravel_index(flat_index, finite_values.shape)
    peak_row = int(row_min + int(local_row))
    peak_col = int(col_min + int(local_col))
    peak_point = _pixel_index_to_display_point(
        peak_row,
        peak_col,
        shape=(rows, cols),
        extent=extent,
    )
    delta = float(math.hypot(float(marker[0]) - peak_point[0], float(marker[1]) - peak_point[1]))
    return {
        "status": "ok",
        "image_peak_point": peak_point,
        "image_peak_index": (peak_row, peak_col),
        "peak_value": float(image_arr[peak_row, peak_col]),
        "point_to_image_peak_delta": delta,
        "point_pixel_index": (float(pixel_float[0]), float(pixel_float[1])),
        "search_radius_px": int(radius),
        "search_window": (row_min, row_max, col_min, col_max),
    }


def _point_delta(
    point_a: Sequence[object] | None,
    point_b: Sequence[object] | None,
) -> float:
    parsed_a = _parse_optional_point(point_a)
    parsed_b = _parse_optional_point(point_b)
    if parsed_a is None or parsed_b is None:
        return float("nan")
    return float(math.hypot(parsed_a[0] - parsed_b[0], parsed_a[1] - parsed_b[1]))


def build_geometry_fit_visual_probe_records(
    draw_records: Sequence[Mapping[str, object]] | None,
    image: object,
    *,
    extent: Sequence[object] | None = None,
    search_radius_px: int = 8,
    image_source: str | None = None,
    axis_xlim: Sequence[object] | None = None,
    axis_ylim: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    """Compare drawn fit markers against the visible simulation image peak."""

    try:
        image_arr = np.asarray(image, dtype=np.float64)
    except Exception:
        image_arr = None
    image_shape = (
        (int(image_arr.shape[0]), int(image_arr.shape[1]))
        if isinstance(image_arr, np.ndarray) and image_arr.ndim == 2
        else None
    )
    probes: list[dict[str, object]] = []
    for entry in draw_records or ():
        if not isinstance(entry, Mapping):
            continue
        record_point = _parse_optional_point(entry.get("record_point"))
        artist_point = _parse_optional_point(entry.get("artist_point"))
        probe_point = artist_point if artist_point is not None else record_point
        peak_probe = probe_display_image_peak_near_point(
            image_arr,
            probe_point,
            extent=extent,
            search_radius_px=search_radius_px,
        )
        image_peak_point = _parse_optional_point(peak_probe.get("image_peak_point"))
        output = dict(entry)
        output.update(
            {
                "status": str(peak_probe.get("status", "unknown")),
                "image_peak_point": image_peak_point,
                "image_peak_index": peak_probe.get("image_peak_index"),
                "image_peak_value": peak_probe.get("peak_value"),
                "point_pixel_index": peak_probe.get("point_pixel_index"),
                "search_window": peak_probe.get("search_window"),
                "artist_to_record_delta": _point_delta(artist_point, record_point),
                "artist_to_image_peak_delta": _point_delta(artist_point, image_peak_point),
                "record_to_image_peak_delta": _point_delta(record_point, image_peak_point),
                "image_extent": tuple(float(value) for value in extent)
                if extent is not None
                else None,
                "image_source": str(image_source) if image_source is not None else "",
                "image_shape": image_shape,
                "axis_xlim": tuple(float(value) for value in axis_xlim)
                if axis_xlim is not None
                else None,
                "axis_ylim": tuple(float(value) for value in axis_ylim)
                if axis_ylim is not None
                else None,
                "search_radius_px": int(max(0, int(search_radius_px))),
            }
        )
        probes.append(output)
    return probes


def rotate_point_for_display(
    col: float,
    row: float,
    shape: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Rotate one ``(col, row)`` point using the same rule as ``np.rot90``."""

    height, width = int(shape[0]), int(shape[1])
    col_new = float(col)
    row_new = float(row)

    for _ in range(int(k) % 4):
        row_new, col_new, height, width = (
            width - 1.0 - col_new,
            row_new,
            width,
            height,
        )

    return float(col_new), float(row_new)


def rotated_shape_for_display(shape: tuple[int, ...], k: int) -> tuple[int, ...]:
    """Return the image shape after applying the same ``np.rot90`` rotation."""

    if len(shape) < 2:
        return tuple(int(v) for v in shape)
    normalized_shape = tuple(int(v) for v in shape)
    if int(k) % 2:
        return (normalized_shape[1], normalized_shape[0], *normalized_shape[2:])
    return normalized_shape


def rotated_image_shape(shape: tuple[int, ...], k: int) -> tuple[int, int]:
    """Return the 2D image shape after applying ``np.rot90(image, k)``."""

    rotated = rotated_shape_for_display(shape, int(k))
    return int(rotated[0]), int(rotated[1])


def display_point_to_native_for_rotation(
    col: float,
    row: float,
    native_shape: tuple[int, ...],
    k: int,
    *,
    rotate_point_for_display_fn: Callable[
        [float, float, tuple[int, ...], int], tuple[float, float]
    ] = rotate_point_for_display,
) -> tuple[float, float]:
    """Map a point from a ``np.rot90(..., k)`` display back to native pixels."""

    display_shape = rotated_shape_for_display(native_shape, int(k))
    return rotate_point_for_display_fn(
        float(col),
        float(row),
        display_shape,
        -int(k),
    )


def rotate_point_for_display_extent(
    col: float,
    row: float,
    shape: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Rotate a continuous image-coordinate point using image extents.

    Pixel-index transforms use ``width - 1 - x``. Beam-center geometry uses
    detector extents, so a 3000 px detector uses ``3000 - x``.
    """

    height, width = int(shape[0]), int(shape[1])
    col_new = float(col)
    row_new = float(row)

    for _ in range(int(k) % 4):
        row_new, col_new, height, width = (
            width - col_new,
            row_new,
            width,
            height,
        )

    return float(col_new), float(row_new)


def beam_center_row_col_from_poni(
    poni1_m: float,
    poni2_m: float,
    pixel_size_m: float,
) -> tuple[float, float]:
    """Return beam-center slider coordinates from pyFAI PONI values.

    ``poni1`` is the detector row-axis distance and ``poni2`` is the detector
    column-axis distance. GUI center sliders are labelled and consumed as
    ``(row, col)``, so no display rotation or image-size inversion belongs in
    this conversion.
    """

    pixel = float(pixel_size_m)
    if not math.isfinite(pixel) or pixel == 0.0:
        raise ValueError("pixel_size_m must be finite and non-zero")
    center_row = float(poni1_m) / pixel
    center_col = float(poni2_m) / pixel
    return float(center_row), float(center_col)


def detector_native_to_display_coords(
    col: float,
    row: float,
    native_shape: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Project native detector ``(col, row)`` into the rotated detector display."""

    return rotate_point_for_display(float(col), float(row), native_shape, int(k))


def detector_display_to_native_coords(
    col: float,
    row: float,
    native_shape: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Invert the detector display rotation back to native pixel coordinates."""

    display_shape = rotated_image_shape(native_shape, int(k))
    return rotate_point_for_display(float(col), float(row), display_shape, -int(k))


def beam_center_row_col_from_detector_display(
    display_col: float,
    display_row: float,
    detector_shape: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Return slider-order ``(row, col)`` for a picked detector-display point.

    For the current default clockwise detector display, GUI Row follows the
    clicked display row and GUI Col is mirrored across the displayed detector
    width.
    """

    display_shape = rotated_image_shape(detector_shape, int(k))
    if int(k) % 4 == 3:
        display_width = float(display_shape[1])
        center_row = float(display_row)
        center_col = display_width - float(display_col)
        return float(center_row), float(center_col)
    if int(k) % 4 == 0:
        return float(display_row), float(display_col)

    native_col, native_row = rotate_point_for_display_extent(
        float(display_col),
        float(display_row),
        display_shape,
        -int(k),
    )
    return float(native_row), float(native_col)


def beam_center_row_col_to_detector_display(
    center_row: float,
    center_col: float,
    detector_shape: tuple[int, ...],
    k: int,
) -> tuple[float, float]:
    """Project slider-order beam-center ``(row, col)`` into detector display.

    This is the inverse of :func:`beam_center_row_col_from_detector_display`
    for marker placement and diagnostics. For the current default clockwise
    detector display, ``display_col = detector_width - center_col`` and
    ``display_row = center_row``.
    """

    display_shape = rotated_image_shape(detector_shape, int(k))
    if int(k) % 4 == 3:
        display_width = float(display_shape[1])
        return display_width - float(center_col), float(center_row)
    if int(k) % 4 == 0:
        return float(center_col), float(center_row)

    display_col, display_row = rotate_point_for_display_extent(
        float(center_col),
        float(center_row),
        detector_shape,
        int(k),
    )
    return float(display_col), float(display_row)


def transform_points_orientation(
    points: Sequence[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
) -> list[tuple[float, float]]:
    """Apply the discrete fit orientation transform to point coordinates."""

    base_height, base_width = int(shape[0]), int(shape[1])
    mode = (indexing_mode or "xy").lower()
    if mode == "yx":
        height, width = base_width, base_height
    else:
        height, width = base_height, base_width

    order = (flip_order or "yx").lower()
    transformed: list[tuple[float, float]] = []

    def _flip_xy(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_x:
            col_t = width - 1.0 - col_t
        if flip_y:
            row_t = height - 1.0 - row_t
        return float(col_t), float(row_t)

    def _flip_yx(col_t: float, row_t: float) -> tuple[float, float]:
        if flip_y:
            row_t = height - 1.0 - row_t
        if flip_x:
            col_t = width - 1.0 - col_t
        return float(col_t), float(row_t)

    flipper = _flip_xy if order == "xy" else _flip_yx

    for col, row in points:
        col_t = float(col)
        row_t = float(row)
        if mode == "yx":
            col_t, row_t = row_t, col_t
        col_t, row_t = flipper(col_t, row_t)
        transformed.append(rotate_point_for_display(col_t, row_t, (height, width), int(k)))

    return transformed


def iter_orientation_transform_candidates():
    """Yield all discrete 90° rotation / flip transform candidates."""

    for indexing_mode in ("xy", "yx"):
        for flip_order in ("yx", "xy"):
            for k in range(4):
                for flip_x in (False, True):
                    for flip_y in (False, True):
                        yield {
                            "indexing_mode": indexing_mode,
                            "k": int(k),
                            "flip_x": bool(flip_x),
                            "flip_y": bool(flip_y),
                            "flip_order": flip_order,
                        }


def _identity_orientation_choice() -> dict[str, object]:
    return {
        "indexing_mode": "xy",
        "k": 0,
        "flip_x": False,
        "flip_y": False,
        "flip_order": "yx",
    }


def _normalize_orientation_choice(
    orientation_choice: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(orientation_choice, Mapping):
        return _identity_orientation_choice()
    return {
        "indexing_mode": str(orientation_choice.get("indexing_mode", "xy")),
        "k": int(orientation_choice.get("k", 0)),
        "flip_x": bool(orientation_choice.get("flip_x", False)),
        "flip_y": bool(orientation_choice.get("flip_y", False)),
        "flip_order": str(orientation_choice.get("flip_order", "yx")),
    }


def _orientation_frame_shape(
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
) -> tuple[int, int]:
    base_height, base_width = int(shape[0]), int(shape[1])
    if (indexing_mode or "xy").lower() == "yx":
        return base_width, base_height
    return base_height, base_width


def orientation_output_shape(
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
    k: int = 0,
) -> tuple[int, int]:
    """Return the detector-frame shape after one orientation transform."""

    height, width = _orientation_frame_shape(shape, indexing_mode=indexing_mode)
    if int(k) % 2:
        return width, height
    return height, width


def _orientation_reference_points(shape: tuple[int, int]) -> list[tuple[float, float]]:
    height, width = int(shape[0]), int(shape[1])
    return [
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (0.0, float(height - 1)),
        (float(width - 1), float(height - 1)),
        (0.5 * float(width - 1), 0.5 * float(height - 1)),
    ]


def inverse_transform_points_orientation(
    points: Sequence[tuple[float, float]],
    shape: tuple[int, int],
    orientation_choice: Mapping[str, object] | None,
) -> list[tuple[float, float]]:
    """Undo one discrete orientation transform on point coordinates exactly."""

    forward = _normalize_orientation_choice(orientation_choice)
    mode = str(forward["indexing_mode"])
    pre_rotation_shape = _orientation_frame_shape(shape, indexing_mode=mode)
    output_shape = orientation_output_shape(
        shape,
        indexing_mode=mode,
        k=int(forward["k"]),
    )
    height, width = pre_rotation_shape

    inverse_points: list[tuple[float, float]] = []
    for col, row in points:
        col_t, row_t = rotate_point_for_display(
            float(col),
            float(row),
            output_shape,
            -int(forward["k"]),
        )
        if bool(forward["flip_x"]):
            col_t = width - 1.0 - col_t
        if bool(forward["flip_y"]):
            row_t = height - 1.0 - row_t
        if mode == "yx":
            col_t, row_t = row_t, col_t
        inverse_points.append((float(col_t), float(row_t)))

    return inverse_points


def inverse_orientation_transform(
    shape: tuple[int, int],
    orientation_choice: dict[str, object] | None,
) -> dict[str, object]:
    """Return the inverse transform for the oriented output frame of ``shape``."""

    forward = _normalize_orientation_choice(orientation_choice)
    refs = _orientation_reference_points(shape)
    mapped = transform_points_orientation(refs, shape, **forward)
    output_shape = orientation_output_shape(
        shape,
        indexing_mode=str(forward["indexing_mode"]),
        k=int(forward["k"]),
    )

    best = None
    best_err = float("inf")
    for candidate in iter_orientation_transform_candidates():
        unmapped = transform_points_orientation(mapped, output_shape, **candidate)
        err = 0.0
        for (x_ref, y_ref), (x_back, y_back) in zip(refs, unmapped):
            err = max(err, float(math.hypot(x_back - x_ref, y_back - y_ref)))
        if err < best_err:
            best_err = err
            best = dict(candidate)
        if err <= 1e-6:
            break

    if best is None:
        return _identity_orientation_choice()
    return best


def compose_orientation_transforms(
    shape: tuple[int, int],
    first: dict[str, object] | None,
    second: dict[str, object] | None,
) -> dict[str, object]:
    """Return one discrete transform equivalent to applying ``first`` then ``second``."""

    first_norm = _normalize_orientation_choice(first)
    second_norm = _normalize_orientation_choice(second)

    refs = _orientation_reference_points(shape)
    mapped_once = transform_points_orientation(refs, shape, **first_norm)
    mapped_twice = transform_points_orientation(
        mapped_once,
        orientation_output_shape(
            shape,
            indexing_mode=str(first_norm["indexing_mode"]),
            k=int(first_norm["k"]),
        ),
        **second_norm,
    )

    best = None
    best_err = float("inf")
    for candidate in iter_orientation_transform_candidates():
        remapped = transform_points_orientation(refs, shape, **candidate)
        err = 0.0
        for (x_ref, y_ref), (x_back, y_back) in zip(mapped_twice, remapped):
            err = max(err, float(math.hypot(x_back - x_ref, y_back - y_ref)))
        if err < best_err:
            best_err = err
            best = dict(candidate)
        if err <= 1e-6:
            break

    if best is None:
        return _identity_orientation_choice()
    return best


def rotate_measured_peaks_for_display(
    measured,
    rotated_shape,
    *,
    display_rotate_k: int = -1,
):
    """Rotate measured-peak coordinates to match the displayed background."""

    if measured is None:
        return []

    rotated_entries = []
    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = rotate_point_for_display(
                    updated["x"],
                    updated["y"],
                    rotated_shape,
                    display_rotate_k,
                )
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = rotate_point_for_display(
                    updated["x_pix"],
                    updated["y_pix"],
                    rotated_shape,
                    display_rotate_k,
                )
            rotated_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = rotate_point_for_display(
                seq[3],
                seq[4],
                rotated_shape,
                display_rotate_k,
            )
            rotated_entries.append(type(entry)(seq))
        else:
            rotated_entries.append(entry)

    return rotated_entries


def unrotate_display_peaks(
    measured,
    rotated_shape,
    *,
    k: int | None = None,
    default_display_rotate_k: int = -1,
):
    """Undo a display rotation on peak coordinates."""

    if measured is None:
        return []

    rotate_k = default_display_rotate_k if k is None else k
    inv_k = -int(rotate_k)

    unrotated = []
    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = rotate_point_for_display(
                    updated["x"],
                    updated["y"],
                    rotated_shape,
                    inv_k,
                )
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = rotate_point_for_display(
                    updated["x_pix"],
                    updated["y_pix"],
                    rotated_shape,
                    inv_k,
                )
            unrotated.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = rotate_point_for_display(
                seq[3],
                seq[4],
                rotated_shape,
                inv_k,
            )
            unrotated.append(type(entry)(seq))
        else:
            unrotated.append(entry)

    return unrotated


def apply_indexing_mode_to_entries(
    measured,
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
):
    """Swap x/y coordinates when using alternate indexing modes."""

    if measured is None:
        return []

    _ = shape
    mode = (indexing_mode or "xy").lower()
    if mode == "xy":
        return list(measured)

    swapped_entries = []

    def _swap_pair(col: float, row: float) -> tuple[float, float]:
        return float(row), float(col)

    for entry in measured:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _swap_pair(updated["x"], updated["y"])
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _swap_pair(
                    updated["x_pix"],
                    updated["y_pix"],
                )
            swapped_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _swap_pair(seq[3], seq[4])
            swapped_entries.append(type(entry)(seq))
        else:
            swapped_entries.append(entry)

    return swapped_entries


def apply_orientation_to_entries(
    measured,
    rotated_shape,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Apply backend-only rotations/flips to measured peak entries."""

    if measured is None:
        return []

    indexed = apply_indexing_mode_to_entries(
        measured,
        rotated_shape,
        indexing_mode=indexing_mode,
    )

    k_mod = int(k) % 4
    if k_mod == 0 and not flip_x and not flip_y:
        return list(indexed)

    mode = (indexing_mode or "xy").lower()
    oriented_shape = rotated_shape if mode == "xy" else (rotated_shape[1], rotated_shape[0])

    def _apply_pair(x_val: float, y_val: float) -> tuple[float, float]:
        return transform_points_orientation(
            [(x_val, y_val)],
            oriented_shape,
            indexing_mode="xy",
            k=k_mod,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_order=flip_order,
        )[0]

    oriented_entries = []
    for entry in indexed:
        if isinstance(entry, dict):
            updated = dict(entry)
            if "x" in updated and "y" in updated:
                updated["x"], updated["y"] = _apply_pair(updated["x"], updated["y"])
            if "x_pix" in updated and "y_pix" in updated:
                updated["x_pix"], updated["y_pix"] = _apply_pair(
                    updated["x_pix"],
                    updated["y_pix"],
                )
            oriented_entries.append(updated)
            continue

        if isinstance(entry, (list, tuple)) and len(entry) >= 5:
            seq = list(entry)
            seq[3], seq[4] = _apply_pair(seq[3], seq[4])
            oriented_entries.append(type(entry)(seq))
        else:
            oriented_entries.append(entry)

    return oriented_entries


def orient_image_for_fit(
    image: np.ndarray | None,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Return a rotated/flipped copy of ``image`` for backend fitting only."""

    if image is None:
        return None

    oriented = np.asarray(image)
    mode = (indexing_mode or "xy").lower()
    if mode == "yx":
        oriented = np.swapaxes(oriented, 0, 1)
    order = (flip_order or "yx").lower()
    if order == "xy":
        if flip_x:
            oriented = np.flip(oriented, axis=1)
        if flip_y:
            oriented = np.flip(oriented, axis=0)
    else:
        if flip_y:
            oriented = np.flip(oriented, axis=0)
        if flip_x:
            oriented = np.flip(oriented, axis=1)
    k_mod = int(k) % 4
    if k_mod:
        oriented = np.rot90(oriented, k_mod)
    return oriented


def native_sim_to_display_coords(
    col: float,
    row: float,
    image_shape: tuple[int, ...],
    *,
    sim_display_rotate_k: int = 0,
) -> tuple[float, float]:
    """Rotate native simulation coordinates into the displayed frame."""

    return rotate_point_for_display(col, row, image_shape, sim_display_rotate_k)


def display_to_native_sim_coords(
    col: float,
    row: float,
    image_shape: tuple[int, ...],
    *,
    sim_display_rotate_k: int = 0,
) -> tuple[float, float]:
    """Map displayed simulation coordinates back into native simulation frame."""

    return display_point_to_native_for_rotation(
        col,
        row,
        image_shape,
        sim_display_rotate_k,
    )


def _geometry_overlay_env_flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def audit_detector_point_frames(
    *,
    pair_id: object,
    branch_index: object,
    native_shape: tuple[int, ...],
    background_display_rotate_k: int,
    sim_display_rotate_k: int,
    sim_display: Sequence[object] | None,
    sim_native: Sequence[object] | None,
    sim_native_source: object,
    bg_display: Sequence[object] | None = None,
    bg_native: Sequence[object] | None = None,
) -> dict[str, object]:
    """Classify whether a simulated point's native/display frames agree.

    The helper is intentionally diagnostic-only. It compares a stored
    ``sim_native`` point against two possible inversions of ``sim_display``:
    the background detector display rotation and the simulation display
    rotation. It does not rewrite either point.
    """

    native_frame_shape = tuple(int(value) for value in native_shape[:2])
    sim_display_point = _parse_optional_point(sim_display)
    sim_native_point = _parse_optional_point(sim_native)
    bg_display_point = _parse_optional_point(bg_display)
    bg_native_point = _parse_optional_point(bg_native)
    source_text = str(sim_native_source or "")

    expected_background_native: tuple[float, float] | None = None
    expected_sim_native: tuple[float, float] | None = None
    err_to_background_native = float("nan")
    err_to_sim_native = float("nan")
    raw_display_native_delta = float("nan")
    status = "missing"

    if (
        len(native_frame_shape) >= 2
        and sim_display_point is not None
        and sim_native_point is not None
    ):
        expected_background_native = display_point_to_native_for_rotation(
            float(sim_display_point[0]),
            float(sim_display_point[1]),
            native_frame_shape,
            int(background_display_rotate_k),
        )
        expected_sim_native = display_to_native_sim_coords(
            float(sim_display_point[0]),
            float(sim_display_point[1]),
            native_frame_shape,
            sim_display_rotate_k=int(sim_display_rotate_k),
        )
        err_to_background_native = float(
            math.hypot(
                float(sim_native_point[0]) - float(expected_background_native[0]),
                float(sim_native_point[1]) - float(expected_background_native[1]),
            )
        )
        err_to_sim_native = float(
            math.hypot(
                float(sim_native_point[0]) - float(expected_sim_native[0]),
                float(sim_native_point[1]) - float(expected_sim_native[1]),
            )
        )
        raw_display_native_delta = float(
            math.hypot(
                float(sim_native_point[0]) - float(sim_display_point[0]),
                float(sim_native_point[1]) - float(sim_display_point[1]),
            )
        )
        tolerance_px = 1.0e-6
        background_ok = bool(err_to_background_native <= tolerance_px)
        sim_ok = bool(err_to_sim_native <= tolerance_px)
        source_mentions_display = "display" in source_text.lower()
        display_native_alias_under_background_rotation = bool(
            raw_display_native_delta <= tolerance_px
            and not background_ok
            and int(background_display_rotate_k) % 4 != int(sim_display_rotate_k) % 4
        )
        source_mentions_overlay_or_saved = any(
            marker in source_text.lower() for marker in ("overlay", "saved")
        )
        display_labeled_native = bool(
            display_native_alias_under_background_rotation
            and source_mentions_display
        )
        if display_labeled_native:
            status = "mismatch_display_labeled_native"
        elif display_native_alias_under_background_rotation and source_mentions_overlay_or_saved:
            status = "mismatch_display_native_alias"
        elif background_ok and sim_ok:
            status = "ambiguous"
        elif background_ok:
            status = "ok_background_native"
        elif sim_ok:
            status = "ok_sim_native"
        else:
            status = "ambiguous"

    record: dict[str, object] = {
        "pair_id": pair_id,
        "branch_index": branch_index,
        "sim_display": sim_display_point,
        "sim_native": sim_native_point,
        "sim_native_source": source_text,
        "background_display_rotate_k": int(background_display_rotate_k),
        "sim_display_rotate_k": int(sim_display_rotate_k),
        "expected_native_from_background_display": expected_background_native,
        "expected_native_from_sim_display": expected_sim_native,
        "err_to_background_native_px": float(err_to_background_native),
        "err_to_sim_native_px": float(err_to_sim_native),
        "raw_sim_display_to_native_delta_px": float(raw_display_native_delta),
        "bg_display": bg_display_point,
        "bg_native": bg_native_point,
        "frame_status": status,
    }
    LOGGER.info("geometry_fit_frame_audit %s", record)
    if status == "mismatch_display_labeled_native" and _geometry_overlay_env_flag_enabled(
        GEOMETRY_AUDIT_STRICT_ENV
    ):
        raise ValueError("mismatch_display_labeled_native")
    return record


def best_orientation_alignment(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
):
    """Search over 90° rotations and axis flips to minimize RMS distance."""

    if not sim_coords or not meas_coords or len(sim_coords) != len(meas_coords):
        return None

    def _describe(
        k: int,
        flip_x: bool,
        flip_y: bool,
        flip_order: str,
        indexing_mode: str,
    ) -> str:
        parts: list[str] = []
        if k % 4:
            parts.append(f"rot{(k % 4) * 90}° CCW")
        if flip_x:
            parts.append("flip_x")
        if flip_y:
            parts.append("flip_y")
        parts.append(f"order={flip_order}")
        parts.append(f"indexing={indexing_mode}")
        return " + ".join(parts)

    best = None
    for candidate in iter_orientation_transform_candidates():
        transformed = transform_points_orientation(
            meas_coords,
            shape,
            indexing_mode=str(candidate["indexing_mode"]),
            k=int(candidate["k"]),
            flip_x=bool(candidate["flip_x"]),
            flip_y=bool(candidate["flip_y"]),
            flip_order=str(candidate["flip_order"]),
        )
        deltas = [
            math.hypot(sx - mx, sy - my) for (sx, sy), (mx, my) in zip(sim_coords, transformed)
        ]
        if not deltas:
            continue
        rms = math.sqrt(sum(delta * delta for delta in deltas) / len(deltas))
        mean = sum(deltas) / len(deltas)
        candidate_result = {
            "k": int(candidate["k"]),
            "flip_x": bool(candidate["flip_x"]),
            "flip_y": bool(candidate["flip_y"]),
            "flip_order": str(candidate["flip_order"]),
            "indexing_mode": str(candidate["indexing_mode"]),
            "rms": float(rms),
            "mean": float(mean),
            "label": _describe(
                int(candidate["k"]),
                bool(candidate["flip_x"]),
                bool(candidate["flip_y"]),
                str(candidate["flip_order"]),
                str(candidate["indexing_mode"]),
            ),
        }
        if best is None or candidate_result["rms"] < best["rms"]:
            best = candidate_result

    return best


def orientation_metrics(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str,
    k: int,
    flip_x: bool,
    flip_y: bool,
    flip_order: str,
):
    """Return RMS/mean/max distance after transforming measured coordinates."""

    transformed = transform_points_orientation(
        meas_coords,
        shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )
    deltas = [math.hypot(sx - mx, sy - my) for (sx, sy), (mx, my) in zip(sim_coords, transformed)]
    if not deltas:
        return {
            "rms": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    arr = np.asarray(deltas, dtype=float)
    return {
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.size),
    }


def select_fit_orientation(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    cfg: dict[str, object] | None = None,
):
    """Choose a measured-peak orientation transform that best aligns to simulation."""

    identity = {
        "k": 0,
        "flip_x": False,
        "flip_y": False,
        "flip_order": "yx",
        "indexing_mode": "xy",
        "label": "identity",
    }
    config = cfg if isinstance(cfg, dict) else {}
    enabled = bool(config.get("enabled", True))
    min_improvement = max(0.0, float(config.get("min_improvement_px", 0.25)))
    max_rms = float(config.get("max_rms_px", np.inf))

    diagnostics = {
        "enabled": bool(enabled),
        "pairs": int(min(len(sim_coords), len(meas_coords))),
        "identity_rms_px": float("nan"),
        "best_rms_px": float("nan"),
        "best_label": "identity",
        "chosen_label": "identity",
        "improvement_px": float("nan"),
        "reason": "identity_fallback",
    }

    if not sim_coords or not meas_coords or len(sim_coords) != len(meas_coords):
        diagnostics["reason"] = "insufficient_pairs"
        return identity, diagnostics

    identity_metrics = orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode="xy",
        k=0,
        flip_x=False,
        flip_y=False,
        flip_order="yx",
    )
    diagnostics["identity_rms_px"] = float(identity_metrics["rms"])

    best = best_orientation_alignment(sim_coords, meas_coords, shape)
    if best is None:
        diagnostics["reason"] = "no_candidate"
        return identity, diagnostics

    best_metrics = orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode=str(best.get("indexing_mode", "xy")),
        k=int(best.get("k", 0)),
        flip_x=bool(best.get("flip_x", False)),
        flip_y=bool(best.get("flip_y", False)),
        flip_order=str(best.get("flip_order", "yx")),
    )
    best_rms = float(best_metrics["rms"])
    identity_rms = float(identity_metrics["rms"])
    improvement = identity_rms - best_rms

    diagnostics.update(
        {
            "best_rms_px": best_rms,
            "best_label": str(best.get("label", "candidate")),
            "improvement_px": float(improvement),
        }
    )

    if not enabled:
        diagnostics["reason"] = "disabled_by_config"
        return identity, diagnostics

    if not np.isfinite(best_rms):
        diagnostics["reason"] = "best_rms_not_finite"
        return identity, diagnostics

    if np.isfinite(max_rms) and best_rms > max_rms:
        diagnostics["reason"] = "best_rms_above_threshold"
        return identity, diagnostics

    if not np.isfinite(improvement) or improvement < min_improvement:
        diagnostics["reason"] = "insufficient_improvement"
        return identity, diagnostics

    chosen = {
        "k": int(best.get("k", 0)),
        "flip_x": bool(best.get("flip_x", False)),
        "flip_y": bool(best.get("flip_y", False)),
        "flip_order": str(best.get("flip_order", "yx")),
        "indexing_mode": str(best.get("indexing_mode", "xy")),
        "label": str(best.get("label", "candidate")),
    }
    diagnostics["chosen_label"] = str(chosen["label"])
    diagnostics["reason"] = "selected_best"
    return chosen, diagnostics


def aggregate_match_centers(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    sim_millers: list[tuple[int, int, int]],
    meas_millers: list[tuple[int, int, int]],
):
    """Collapse matched peaks by HKL and return centroid pairs."""

    aggregated: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}
    for hkl_sim, hkl_meas, sim_xy, meas_xy in zip(
        sim_millers,
        meas_millers,
        sim_coords,
        meas_coords,
    ):
        hkl_key = tuple(int(v) for v in (hkl_sim or hkl_meas))
        entry = aggregated.setdefault(hkl_key, {"sim": [], "meas": []})
        entry["sim"].append(sim_xy)
        entry["meas"].append(meas_xy)

    agg_sim_coords: list[tuple[float, float]] = []
    agg_meas_coords: list[tuple[float, float]] = []
    agg_millers: list[tuple[int, int, int]] = []

    for hkl_key in sorted(aggregated):
        sim_arr = np.asarray(aggregated[hkl_key]["sim"], dtype=float)
        meas_arr = np.asarray(aggregated[hkl_key]["meas"], dtype=float)

        agg_sim_coords.append((float(sim_arr[:, 0].mean()), float(sim_arr[:, 1].mean())))
        agg_meas_coords.append((float(meas_arr[:, 0].mean()), float(meas_arr[:, 1].mean())))
        agg_millers.append(hkl_key)

    return agg_sim_coords, agg_meas_coords, agg_millers


def normalize_hkl_key(
    value: object,
) -> tuple[int, int, int] | None:
    """Return a rounded integer HKL tuple when *value* looks like one."""

    if isinstance(value, str):
        parts = value.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
        if len(parts) < 3:
            return None
        try:
            return tuple(int(np.rint(float(parts[i].strip()))) for i in range(3))
        except Exception:
            return None

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        try:
            return tuple(int(np.rint(float(value[i]))) for i in range(3))
        except Exception:
            return None

    return None


def aggregate_initial_geometry_display_pairs(
    initial_pairs_display: Sequence[dict[str, object]] | None,
) -> dict[tuple[int, int, int], dict[str, tuple[float, float]]]:
    """Aggregate initial display-frame picks by HKL."""

    grouped: dict[tuple[int, int, int], dict[str, list[tuple[float, float]]]] = {}

    def _parse_point(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    for entry in initial_pairs_display or []:
        if not isinstance(entry, dict):
            continue
        hkl_key = normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl_key is None:
            continue
        bucket = grouped.setdefault(hkl_key, {"sim": [], "bg": []})
        sim_pt = _parse_point(entry.get("sim_display"))
        bg_pt = _parse_point(entry.get("bg_display"))
        if sim_pt is not None:
            bucket["sim"].append(sim_pt)
        if bg_pt is not None:
            bucket["bg"].append(bg_pt)

    aggregated: dict[tuple[int, int, int], dict[str, tuple[float, float]]] = {}
    for hkl_key, bucket in grouped.items():
        item: dict[str, tuple[float, float]] = {}
        if bucket["sim"]:
            sim_arr = np.asarray(bucket["sim"], dtype=float)
            item["sim_display"] = (
                float(sim_arr[:, 0].mean()),
                float(sim_arr[:, 1].mean()),
            )
        if bucket["bg"]:
            bg_arr = np.asarray(bucket["bg"], dtype=float)
            item["bg_display"] = (
                float(bg_arr[:, 0].mean()),
                float(bg_arr[:, 1].mean()),
            )
        if item:
            aggregated[hkl_key] = item

    return aggregated


def normalize_overlay_match_index(value: object, fallback: int) -> int:
    """Return a non-negative per-match overlay index."""

    try:
        out = int(value)
    except Exception:
        out = int(fallback)
    if out < 0:
        return int(fallback)
    return int(out)


def normalize_initial_geometry_pairs_display(
    initial_pairs_display: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Normalize initial display-frame match records with stable overlay indices."""

    normalized: list[dict[str, object]] = []

    def _parse_point(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    for fallback_index, raw_entry in enumerate(initial_pairs_display or []):
        if not isinstance(raw_entry, dict):
            continue
        normalized_entry = dict(raw_entry)
        normalized_entry["overlay_match_index"] = normalize_overlay_match_index(
            raw_entry.get("overlay_match_index"),
            fallback_index,
        )
        sim_display = _parse_point(raw_entry.get("sim_display"))
        bg_display = _parse_point(raw_entry.get("bg_display"))
        if sim_display is not None:
            normalized_entry["sim_display"] = sim_display
        if bg_display is not None:
            normalized_entry["bg_display"] = bg_display
        sim_caked_display = _parse_point(raw_entry.get("sim_caked_display"))
        bg_caked_display = _parse_point(raw_entry.get("bg_caked_display"))
        if sim_caked_display is not None:
            normalized_entry["sim_caked_display"] = sim_caked_display
        if bg_caked_display is not None:
            normalized_entry["bg_caked_display"] = bg_caked_display
        raw_group_key = raw_entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            normalized_entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            normalized_entry["q_group_key"] = tuple(raw_group_key)
        normalized.append(normalized_entry)

    return normalized


def _parse_overlay_point(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        col = float(value[0])
        row = float(value[1])
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _parse_overlay_entry_pair(
    entry: Mapping[str, object],
    x_key: str,
    y_key: str,
) -> tuple[float, float] | None:
    try:
        col = float(entry.get(x_key, np.nan))
        row = float(entry.get(y_key, np.nan))
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _first_overlay_entry_point(
    entry: Mapping[str, object],
    *candidates: object,
) -> tuple[float, float] | None:
    for candidate in candidates:
        if (
            isinstance(candidate, tuple)
            and len(candidate) == 2
            and isinstance(candidate[0], str)
            and isinstance(candidate[1], str)
        ):
            point = _parse_overlay_entry_pair(entry, candidate[0], candidate[1])
        else:
            point = _parse_overlay_point(entry.get(str(candidate)))
        if point is not None:
            return point
    return None


def _resolve_overlay_display_point(
    entry: Mapping[str, object],
    *,
    display_key: str,
    native_key: str,
    show_caked_2d: bool = False,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None,
) -> tuple[float, float] | None:
    point, _source, _space_status = _resolve_overlay_display_point_with_source(
        entry,
        display_key=display_key,
        native_key=native_key,
        show_caked_2d=show_caked_2d,
        native_detector_coords_to_caked_display_coords=(
            native_detector_coords_to_caked_display_coords
        ),
    )
    return point


def _resolve_overlay_display_point_with_source(
    entry: Mapping[str, object],
    *,
    display_key: str,
    native_key: str,
    show_caked_2d: bool = False,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None,
) -> tuple[tuple[float, float] | None, str, str]:
    if show_caked_2d:
        caked_display_key = display_key.replace("_display", "_caked_display")
        caked_point = _parse_overlay_point(entry.get(caked_display_key))
        if caked_point is not None:
            return caked_point, caked_display_key, "caked_display"
        native_point = _parse_overlay_point(entry.get(native_key))
        if native_point is not None and native_detector_coords_to_caked_display_coords is not None:
            try:
                projected = native_detector_coords_to_caked_display_coords(
                    float(native_point[0]),
                    float(native_point[1]),
                )
            except Exception:
                projected = None
            projected_point = _parse_overlay_point(projected)
            if projected_point is not None:
                return projected_point, native_key, "native_projected_to_caked"
    display_point = _parse_overlay_point(entry.get(display_key))
    if display_point is not None:
        space_status = "detector_display_used_in_caked_mode" if show_caked_2d else "display"
        return display_point, display_key, space_status
    return None, "", "missing"


def audit_geometry_fit_overlay_visual_distance_inputs(
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    show_caked_2d: bool = False,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None,
) -> list[dict[str, object]]:
    """Return diagnostic-only source records for visual-distance inputs."""

    audit: list[dict[str, object]] = []
    for idx, raw_entry in enumerate(overlay_records or []):
        if not isinstance(raw_entry, Mapping):
            continue

        initial_sim = _resolve_overlay_display_point_with_source(
            raw_entry,
            display_key="initial_sim_display",
            native_key="initial_sim_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        initial_bg = _resolve_overlay_display_point_with_source(
            raw_entry,
            display_key="initial_bg_display",
            native_key="initial_bg_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        final_sim = _resolve_overlay_display_point_with_source(
            raw_entry,
            display_key="final_sim_display",
            native_key="final_sim_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        final_bg = _resolve_overlay_display_point_with_source(
            raw_entry,
            display_key="final_bg_display",
            native_key="final_bg_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )

        initial_distance = _point_delta(initial_sim[0], initial_bg[0])
        final_distance = (
            _point_delta(final_sim[0], final_bg[0])
            if str(raw_entry.get("match_status", "matched")).strip().lower() == "matched"
            else float("nan")
        )
        audit.append(
            {
                "overlay_match_index": raw_entry.get("overlay_match_index", idx),
                "pair_id": raw_entry.get("pair_id"),
                "match_status": raw_entry.get("match_status", "matched"),
                "mode": "caked_2d" if show_caked_2d else "detector_display",
                "units_claimed": "deg" if show_caked_2d else "px",
                "initial_sim_point": initial_sim[0],
                "initial_sim_source": initial_sim[1],
                "initial_sim_space_status": initial_sim[2],
                "initial_bg_point": initial_bg[0],
                "initial_bg_source": initial_bg[1],
                "initial_bg_space_status": initial_bg[2],
                "final_sim_point": final_sim[0],
                "final_sim_source": final_sim[1],
                "final_sim_space_status": final_sim[2],
                "final_bg_point": final_bg[0],
                "final_bg_source": final_bg[1],
                "final_bg_space_status": final_bg[2],
                "initial_distance": float(initial_distance),
                "final_distance": float(final_distance),
                "included_initial_in_median": bool(np.isfinite(initial_distance)),
                "included_final_in_median": bool(np.isfinite(final_distance)),
            }
        )
    return audit


def build_geometry_fit_overlay_records(
    initial_pairs_display: Sequence[dict[str, object]] | None,
    point_match_diagnostics: Sequence[dict[str, object]] | None,
    *,
    native_shape: tuple[int, int],
    orientation_choice: dict[str, object] | None = None,
    sim_display_rotate_k: int = 0,
    background_display_rotate_k: int = 0,
) -> list[dict[str, object]]:
    """Build one overlay record per matched peak from optimizer diagnostics.

    Coordinate-frame contract for ``point_match_diagnostics``:

    - ``simulated_x/y`` are simulation-native coordinates straight from the
      solver / hit-table path.
    - ``measured_x/y`` are fit-oriented coordinates already transformed to
      align with the simulation frame used by the solver. They must be inverse-
      oriented back to native background space before the background display
      rotation is applied.

    Overlay records are drawn on the current main axes, not on the simulator's
    private detector frame. When native detector coordinates are available,
    both the simulation and background points therefore need to be rebuilt in
    the current overlay display frame, which matches the background detector
    display rotation.
    """

    native_frame_shape = (int(native_shape[0]), int(native_shape[1]))
    sim_display_rotate_k = int(sim_display_rotate_k)
    background_display_rotate_k = int(background_display_rotate_k)
    overlay_display_rotate_k = int(background_display_rotate_k)

    def _parse_point(value: object) -> tuple[float, float] | None:
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _first_point_with_source(
        entry: Mapping[str, object],
        *candidates: object,
    ) -> tuple[tuple[float, float], str] | None:
        for candidate in candidates:
            if (
                isinstance(candidate, tuple)
                and len(candidate) == 3
                and isinstance(candidate[0], str)
                and isinstance(candidate[1], str)
                and isinstance(candidate[2], str)
            ):
                point = _parse_point((entry.get(candidate[0]), entry.get(candidate[1])))
                source = candidate[2]
            else:
                source = str(candidate)
                point = _parse_point(entry.get(source))
            if point is not None:
                return point, source
        return None

    def _fit_prediction_display_is_caked_space(entry: Mapping[str, object]) -> bool:
        objective_space = str(entry.get("objective_space", "") or "").strip().lower()
        objective_units = str(entry.get("objective_units", "") or "").strip().lower()
        cache_mode = str(entry.get("objective_cache_mode", "") or "").strip().lower()
        projector_kind = str(entry.get("fit_space_projector_kind", "") or "").strip().lower()
        return (
            objective_space in {"caked", "caked_deg", "fit_space", "fit-space"}
            or objective_units in {"deg", "degree", "degrees"}
            or cache_mode == "point_only_projection"
            or "caked" in projector_kind
        )

    initial_by_index = {
        int(entry["overlay_match_index"]): entry
        for entry in normalize_initial_geometry_pairs_display(initial_pairs_display)
    }

    diagnostics_by_index: dict[int, dict[str, object]] = {}
    diagnostic_order: list[int] = []
    for fallback_index, raw_entry in enumerate(point_match_diagnostics or []):
        if not isinstance(raw_entry, dict):
            continue
        overlay_match_index = normalize_overlay_match_index(
            raw_entry.get("overlay_match_index", raw_entry.get("match_input_index")),
            fallback_index,
        )
        status = str(raw_entry.get("match_status", "matched")).strip().lower()
        existing = diagnostics_by_index.get(overlay_match_index)
        if existing is None:
            diagnostics_by_index[overlay_match_index] = dict(raw_entry)
            diagnostic_order.append(int(overlay_match_index))
            continue
        existing_status = str(existing.get("match_status", "matched")).strip().lower()
        if existing_status != "matched" and status == "matched":
            diagnostics_by_index[overlay_match_index] = dict(raw_entry)

    ordered_indices: list[int] = []
    seen_indices: set[int] = set()
    for entry in normalize_initial_geometry_pairs_display(initial_pairs_display):
        index = int(entry["overlay_match_index"])
        if index in seen_indices:
            continue
        seen_indices.add(index)
        ordered_indices.append(index)
    for index in diagnostic_order:
        if index in seen_indices:
            continue
        seen_indices.add(index)
        ordered_indices.append(int(index))

    records: list[dict[str, object]] = []
    for fallback_index, overlay_match_index in enumerate(ordered_indices):
        initial_entry = initial_by_index.get(int(overlay_match_index), {})
        raw_entry = diagnostics_by_index.get(int(overlay_match_index), {})
        status = str(raw_entry.get("match_status", "missing_pair")).strip().lower()

        record = dict(raw_entry)
        record["overlay_match_index"] = int(overlay_match_index)
        if not status:
            status = "missing_pair"
        record["match_status"] = status

        initial_sim_native = _parse_point(initial_entry.get("sim_native"))
        initial_bg_native = _parse_point(initial_entry.get("bg_native"))
        initial_sim_display_raw = _parse_point(initial_entry.get("sim_display"))
        initial_bg_display_raw = _parse_point(initial_entry.get("bg_display"))
        initial_sim_caked_display = _parse_point(initial_entry.get("sim_caked_display"))
        initial_bg_caked_display = _parse_point(initial_entry.get("bg_caked_display"))
        # Legacy saved overlays may only have cached detector-view display
        # points. Recover native detector coordinates first so redraws can be
        # rebuilt in the current overlay frame instead of the stale snapshot
        # frame.
        if initial_sim_native is None and initial_sim_display_raw is not None:
            recovered_native = display_to_native_sim_coords(
                float(initial_sim_display_raw[0]),
                float(initial_sim_display_raw[1]),
                native_frame_shape,
                sim_display_rotate_k=sim_display_rotate_k,
            )
            if np.isfinite(float(recovered_native[0])) and np.isfinite(float(recovered_native[1])):
                initial_sim_native = (
                    float(recovered_native[0]),
                    float(recovered_native[1]),
                )
        if initial_bg_native is None and initial_bg_display_raw is not None:
            recovered_native = rotate_point_for_display(
                float(initial_bg_display_raw[0]),
                float(initial_bg_display_raw[1]),
                native_frame_shape,
                -background_display_rotate_k,
            )
            if np.isfinite(float(recovered_native[0])) and np.isfinite(float(recovered_native[1])):
                initial_bg_native = (
                    float(recovered_native[0]),
                    float(recovered_native[1]),
                )
        initial_sim_frame_audit = audit_detector_point_frames(
            pair_id=initial_entry.get(
                "pair_id",
                raw_entry.get("pair_id", initial_entry.get("overlay_match_index")),
            ),
            branch_index=initial_entry.get(
                "source_branch_index",
                raw_entry.get("source_branch_index", fallback_index),
            ),
            native_shape=native_frame_shape,
            background_display_rotate_k=background_display_rotate_k,
            sim_display_rotate_k=sim_display_rotate_k,
            sim_display=initial_sim_display_raw,
            sim_native=initial_sim_native,
            sim_native_source=initial_entry.get("sim_native_source", "initial_entry.sim_native"),
            bg_display=initial_bg_display_raw,
            bg_native=initial_bg_native,
        )
        record["initial_sim_native_frame_status"] = str(
            initial_sim_frame_audit.get("frame_status", "missing")
        )
        record["initial_sim_native_frame_audit"] = dict(initial_sim_frame_audit)
        record["initial_sim_native_source"] = str(
            initial_entry.get("sim_native_source", "initial_entry.sim_native")
        )
        if initial_sim_display_raw is not None:
            record["initial_sim_display_raw"] = (
                float(initial_sim_display_raw[0]),
                float(initial_sim_display_raw[1]),
            )

        initial_sim_display = None
        initial_bg_display = None
        initial_sim_display_rebuilt = None
        initial_sim_raw_vs_rebuilt_delta = float("nan")
        # Saved fits can be redrawn in a different view than the one that
        # produced the initial overlay snapshot, so prefer native coordinates
        # when available and rebuild the current overlay-display positions.
        if initial_sim_native is not None:
            rotated = rotate_point_for_display(
                float(initial_sim_native[0]),
                float(initial_sim_native[1]),
                native_frame_shape,
                overlay_display_rotate_k,
            )
            initial_sim_display_rebuilt = (float(rotated[0]), float(rotated[1]))
            if initial_sim_display_raw is not None:
                initial_sim_raw_vs_rebuilt_delta = _point_delta(
                    initial_sim_display_raw,
                    initial_sim_display_rebuilt,
                )
        record["initial_sim_display_rebuilt_from_native"] = initial_sim_display_rebuilt
        record["initial_sim_display_raw_vs_rebuilt_delta_px"] = float(
            initial_sim_raw_vs_rebuilt_delta
        )
        disable_sim_native_rebuild = bool(
            _geometry_overlay_env_flag_enabled(GEOMETRY_DISABLE_SIM_NATIVE_REBUILD_ENV)
        )
        if disable_sim_native_rebuild:
            record["diagnostic_flag_RA_SIM_GEOM_DISABLE_SIM_NATIVE_REBUILD"] = True
            LOGGER.info("%s active", GEOMETRY_DISABLE_SIM_NATIVE_REBUILD_ENV)
        initial_sim_native_frame_status = str(
            initial_sim_frame_audit.get("frame_status", "missing")
        )
        initial_sim_native_rebuild_valid = initial_sim_native_frame_status in {
            "ok_background_native",
            "ok_sim_native",
        }
        if (
            initial_sim_display_rebuilt is not None
            and initial_sim_display_raw is not None
            and initial_sim_native_frame_status == "ambiguous"
            and np.isfinite(float(initial_sim_raw_vs_rebuilt_delta))
            and float(initial_sim_raw_vs_rebuilt_delta) <= 1.0e-6
        ):
            initial_sim_native_rebuild_valid = True
        initial_sim_native_rebuild_rejected = bool(
            initial_sim_display_rebuilt is not None
            and initial_sim_display_raw is not None
            and initial_sim_native_frame_status
            in {"mismatch_display_labeled_native", "mismatch_display_native_alias"}
            and not initial_sim_native_rebuild_valid
        )
        if initial_sim_native_rebuild_rejected:
            record["initial_sim_native_rebuild_rejected_frame_status"] = (
                initial_sim_native_frame_status
            )
        if (
            initial_sim_display_rebuilt is not None
            and not (
                disable_sim_native_rebuild and initial_sim_display_raw is not None
            )
            and not initial_sim_native_rebuild_rejected
        ):
            initial_sim_display = initial_sim_display_rebuilt
            record["chosen_initial_sim_display_source"] = "recomputed_from_initial_sim_native"
        elif initial_sim_display_raw is not None:
            initial_sim_display = initial_sim_display_raw
            if disable_sim_native_rebuild and initial_sim_display_rebuilt is not None:
                source = "raw_initial_sim_display:diagnostic_rebuild_disabled"
            elif initial_sim_native_rebuild_rejected:
                source = "raw_initial_sim_display:rejected_initial_sim_native_frame"
            else:
                source = "raw_initial_sim_display"
            record["chosen_initial_sim_display_source"] = source
        else:
            record["chosen_initial_sim_display_source"] = "missing"
        if initial_bg_native is not None:
            rotated = rotate_point_for_display(
                float(initial_bg_native[0]),
                float(initial_bg_native[1]),
                native_frame_shape,
                background_display_rotate_k,
            )
            initial_bg_display = (float(rotated[0]), float(rotated[1]))
        else:
            initial_bg_display = initial_bg_display_raw
        record["initial_sim_display"] = initial_sim_display
        record["initial_bg_display"] = initial_bg_display
        if initial_sim_caked_display is not None:
            record["initial_sim_caked_display"] = (
                float(initial_sim_caked_display[0]),
                float(initial_sim_caked_display[1]),
            )
        if initial_bg_caked_display is not None:
            record["initial_bg_caked_display"] = (
                float(initial_bg_caked_display[0]),
                float(initial_bg_caked_display[1]),
            )
        if initial_sim_native is not None:
            record["initial_sim_native"] = (
                float(initial_sim_native[0]),
                float(initial_sim_native[1]),
            )
        if initial_bg_native is not None:
            record["initial_bg_native"] = (
                float(initial_bg_native[0]),
                float(initial_bg_native[1]),
            )
        if "hkl" not in record and initial_entry.get("hkl") is not None:
            record["hkl"] = initial_entry.get("hkl")
        if "label" not in record and initial_entry.get("hkl") is not None:
            record["label"] = str(initial_entry.get("hkl"))

        if status == "matched":
            try:
                legacy_simulated_point = (
                    float(raw_entry.get("simulated_x", np.nan)),
                    float(raw_entry.get("simulated_y", np.nan)),
                )
                measured_fit_oriented = (
                    float(raw_entry.get("measured_x", np.nan)),
                    float(raw_entry.get("measured_y", np.nan)),
                )
            except Exception:
                continue
            final_sim_display_payload = _first_point_with_source(
                raw_entry,
                "final_prediction_detector_display_px",
                "dynamic_final_detector_display_px",
                "fit_prediction_detector_display_px",
                "predicted_detector_display_px",
                "sim_refined_detector_display_px",
                "sim_nominal_detector_display_px",
                "sim_nominal_detector_px",
            )
            final_sim_native_payload = _first_point_with_source(
                raw_entry,
                "final_prediction_detector_native_px",
                "dynamic_final_detector_native_px",
                "fit_prediction_detector_native_px",
                "predicted_detector_native_px",
                "sim_refined_detector_native_px",
                "sim_nominal_native_px",
                "sim_visual_detector_canonical_native_px",
                ("simulated_native_col", "simulated_native_row", "simulated_native_col/row"),
            )
            if final_sim_native_payload is not None:
                simulated_native = final_sim_native_payload[0]
                final_sim_native_source = final_sim_native_payload[1]
            else:
                simulated_native = legacy_simulated_point
                final_sim_native_source = "simulated_x/y"
            if not all(np.isfinite(v) for v in (*simulated_native, *measured_fit_oriented)):
                continue
            if all(np.isfinite(v) for v in legacy_simulated_point):
                final_sim_fit = legacy_simulated_point
            else:
                final_sim_fit = simulated_native

            measured_native = inverse_transform_points_orientation(
                [measured_fit_oriented],
                native_frame_shape,
                orientation_choice,
            )[0]

            if final_sim_display_payload is not None:
                final_sim_display = final_sim_display_payload[0]
                final_sim_display_source = final_sim_display_payload[1]
            else:
                final_sim_display = rotate_point_for_display(
                    float(simulated_native[0]),
                    float(simulated_native[1]),
                    native_frame_shape,
                    overlay_display_rotate_k,
                )
                final_sim_display_source = f"{final_sim_native_source}->display"
            final_bg_display = rotate_point_for_display(
                float(measured_native[0]),
                float(measured_native[1]),
                native_frame_shape,
                background_display_rotate_k,
            )
            record["final_sim_fit"] = (
                float(final_sim_fit[0]),
                float(final_sim_fit[1]),
            )
            record["final_bg_fit"] = (
                float(measured_fit_oriented[0]),
                float(measured_fit_oriented[1]),
            )
            record["final_sim_native"] = (
                float(simulated_native[0]),
                float(simulated_native[1]),
            )
            record["final_bg_native"] = (
                float(measured_native[0]),
                float(measured_native[1]),
            )
            record["final_sim_display"] = (
                float(final_sim_display[0]),
                float(final_sim_display[1]),
            )
            record["final_sim_display_source"] = str(final_sim_display_source)
            record["final_sim_native_source"] = str(final_sim_native_source)
            record["final_bg_display"] = (
                float(final_bg_display[0]),
                float(final_bg_display[1]),
            )
            if initial_bg_caked_display is not None:
                record["final_bg_caked_display"] = (
                    float(initial_bg_caked_display[0]),
                    float(initial_bg_caked_display[1]),
                )
            # Caked point-only projection diagnostics historically reused the
            # detector-display field for the current caked fit-space prediction.
            fit_prediction_caked_fallbacks: tuple[object, ...] = (
                ("fit_prediction_detector_display_px", "predicted_detector_display_px")
                if _fit_prediction_display_is_caked_space(raw_entry)
                else ()
            )
            final_sim_caked_payload = _first_point_with_source(
                raw_entry,
                "final_prediction_caked_deg",
                "dynamic_final_caked_deg",
                "sim_visual_caked_deg",
                "predicted_refined_caked_deg",
                "sim_refined_caked_deg",
                "fit_prediction_caked_deg",
                *fit_prediction_caked_fallbacks,
                "predicted_caked_deg",
                (
                    "simulated_two_theta_deg",
                    "simulated_phi_deg",
                    "simulated_two_theta_deg/simulated_phi_deg",
                ),
                "manual_qr_fit_source_caked_deg",
                "projected_caked_deg",
            )
            if final_sim_caked_payload is not None:
                final_sim_caked_display = final_sim_caked_payload[0]
                final_sim_caked_source = final_sim_caked_payload[1]
                record["final_sim_caked_display"] = (
                    float(final_sim_caked_display[0]),
                    float(final_sim_caked_display[1]),
                )
                record["final_sim_caked_display_source"] = str(final_sim_caked_source)
                render_caked_payload = _first_point_with_source(
                    raw_entry,
                    "final_prediction_caked_deg",
                    "dynamic_final_caked_deg",
                    "sim_visual_caked_deg",
                    "sim_refined_caked_deg",
                    "fit_prediction_caked_deg",
                )
                if render_caked_payload is not None:
                    render_caked = render_caked_payload[0]
                    record["final_sim_render_caked_display"] = (
                        float(render_caked[0]),
                        float(render_caked[1]),
                    )
                    record["final_sim_render_caked_display_source"] = str(render_caked_payload[1])
                    record["fit_sim_render_caked_delta"] = float(
                        math.hypot(
                            final_sim_caked_display[0] - render_caked[0],
                            final_sim_caked_display[1] - render_caked[1],
                        )
                    )
            final_bg_caked_display = _first_overlay_entry_point(
                raw_entry,
                ("measured_two_theta_deg", "measured_phi_deg"),
                "observed_caked_deg",
                "manual_qr_fit_target_caked_deg",
            )
            if final_bg_caked_display is not None:
                record["final_bg_caked_display"] = (
                    float(final_bg_caked_display[0]),
                    float(final_bg_caked_display[1]),
                )
            record["simulated_frame"] = "sim_native"
            record["measured_frame"] = "fit_oriented"
            try:
                distance_px = float(record.get("distance_px", np.nan))
            except Exception:
                distance_px = float("nan")
            if not np.isfinite(distance_px):
                distance_px = float(
                    math.hypot(
                        final_sim_fit[0] - measured_fit_oriented[0],
                        final_sim_fit[1] - measured_fit_oriented[1],
                    )
                )
            record["overlay_distance_px"] = float(distance_px)

        records.append(record)

    return records


def _overlay_distance_stats(distances: Sequence[float]) -> dict[str, float]:
    values = np.asarray([float(value) for value in distances], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "max": float(np.max(values)),
    }


def summarize_geometry_fit_overlay_visual_distances(
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    show_caked_2d: bool = False,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None,
) -> dict[str, float]:
    """Summarize visible sim-to-background distances from overlay records."""

    initial_distances: list[float] = []
    final_distances: list[float] = []
    paired_delta_distances: list[float] = []
    improved_count = 0
    worsened_count = 0
    unchanged_count = 0
    paired_records = 0

    for raw_entry in overlay_records or []:
        if not isinstance(raw_entry, Mapping):
            continue
        status = str(raw_entry.get("match_status", "matched")).strip().lower()
        initial_sim = _resolve_overlay_display_point(
            raw_entry,
            display_key="initial_sim_display",
            native_key="initial_sim_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        initial_bg = _resolve_overlay_display_point(
            raw_entry,
            display_key="initial_bg_display",
            native_key="initial_bg_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        final_sim = _resolve_overlay_display_point(
            raw_entry,
            display_key="final_sim_display",
            native_key="final_sim_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        final_bg = _resolve_overlay_display_point(
            raw_entry,
            display_key="final_bg_display",
            native_key="final_bg_native",
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )
        initial_dist = _point_delta(initial_sim, initial_bg)
        if np.isfinite(initial_dist):
            initial_distances.append(initial_dist)
        final_dist = (
            _point_delta(final_sim, final_bg) if status == "matched" else float("nan")
        )
        if np.isfinite(final_dist):
            final_distances.append(final_dist)
        if np.isfinite(initial_dist) and np.isfinite(final_dist):
            paired_records += 1
            delta = float(final_dist - initial_dist)
            paired_delta_distances.append(delta)
            if delta < -1.0e-9:
                improved_count += 1
            elif delta > 1.0e-9:
                worsened_count += 1
            else:
                unchanged_count += 1

    initial_stats = _overlay_distance_stats(initial_distances)
    final_stats = _overlay_distance_stats(final_distances)
    delta_stats = _overlay_distance_stats(paired_delta_distances)
    return {
        "paired_visual_records": float(paired_records),
        "initial_distance_count": float(initial_stats["count"]),
        "initial_distance_mean": float(initial_stats["mean"]),
        "initial_distance_median": float(initial_stats["median"]),
        "initial_distance_p90": float(initial_stats["p90"]),
        "initial_distance_max": float(initial_stats["max"]),
        "final_distance_count": float(final_stats["count"]),
        "final_distance_mean": float(final_stats["mean"]),
        "final_distance_median": float(final_stats["median"]),
        "final_distance_p90": float(final_stats["p90"]),
        "final_distance_max": float(final_stats["max"]),
        "delta_distance_mean": float(delta_stats["mean"]),
        "delta_distance_median": float(delta_stats["median"]),
        "delta_distance_p90": float(delta_stats["p90"]),
        "improved_count": float(improved_count),
        "worsened_count": float(worsened_count),
        "unchanged_count": float(unchanged_count),
    }


def compute_geometry_overlay_frame_diagnostics(
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    show_caked_2d: bool = False,
    native_detector_coords_to_caked_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None,
) -> tuple[dict[str, float], str]:
    """Summarize per-match display-frame agreement for the final overlay."""

    def _resolve_display_point(
        entry: Mapping[str, object],
        *,
        display_key: str,
        native_key: str,
    ) -> tuple[float, float] | None:
        return _resolve_overlay_display_point(
            entry,
            display_key=display_key,
            native_key=native_key,
            show_caked_2d=show_caked_2d,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
        )

    sim_frame_dists: list[float] = []
    bg_frame_dists: list[float] = []
    fit_sim_render_caked_deltas: list[float] = []
    paired_records = 0

    for raw_entry in overlay_records or []:
        if not isinstance(raw_entry, dict):
            continue
        initial_sim = _resolve_display_point(
            raw_entry,
            display_key="initial_sim_display",
            native_key="initial_sim_native",
        )
        final_sim = _resolve_display_point(
            raw_entry,
            display_key="final_sim_display",
            native_key="final_sim_native",
        )
        initial_bg = _resolve_display_point(
            raw_entry,
            display_key="initial_bg_display",
            native_key="initial_bg_native",
        )
        final_bg = _resolve_display_point(
            raw_entry,
            display_key="final_bg_display",
            native_key="final_bg_native",
        )
        if (
            initial_sim is not None
            and final_sim is not None
            and initial_bg is not None
            and final_bg is not None
        ):
            paired_records += 1
        if initial_sim is not None and final_sim is not None:
            sim_frame_dists.append(_point_delta(final_sim, initial_sim))
        if initial_bg is not None and final_bg is not None:
            bg_frame_dists.append(_point_delta(final_bg, initial_bg))
        try:
            render_delta = float(raw_entry.get("fit_sim_render_caked_delta", np.nan))
        except Exception:
            render_delta = float("nan")
        if np.isfinite(render_delta):
            fit_sim_render_caked_deltas.append(float(abs(render_delta)))

    visual_stats = summarize_geometry_fit_overlay_visual_distances(
        overlay_records,
        show_caked_2d=show_caked_2d,
        native_detector_coords_to_caked_display_coords=(
            native_detector_coords_to_caked_display_coords
        ),
    )
    visual_input_audit = audit_geometry_fit_overlay_visual_distance_inputs(
        overlay_records,
        show_caked_2d=show_caked_2d,
        native_detector_coords_to_caked_display_coords=(
            native_detector_coords_to_caked_display_coords
        ),
    )
    caked_detector_display_input_count = 0
    if show_caked_2d:
        for audit_entry in visual_input_audit:
            for role in ("initial_sim", "initial_bg", "final_sim", "final_bg"):
                if (
                    audit_entry.get(f"{role}_space_status")
                    == "detector_display_used_in_caked_mode"
                ):
                    caked_detector_display_input_count += 1

    stats: dict[str, float] = {
        "overlay_record_count": float(len(list(overlay_records or []))),
        "paired_records": float(paired_records),
        "sim_display_med_px": float(np.median(sim_frame_dists))
        if sim_frame_dists
        else float("nan"),
        "bg_display_med_px": float(np.median(bg_frame_dists)) if bg_frame_dists else float("nan"),
        "sim_display_p90_px": float(np.percentile(sim_frame_dists, 90.0))
        if sim_frame_dists
        else float("nan"),
        "bg_display_p90_px": float(np.percentile(bg_frame_dists, 90.0))
        if bg_frame_dists
        else float("nan"),
        "fit_sim_render_caked_delta_count": float(len(fit_sim_render_caked_deltas)),
        "fit_sim_render_caked_delta_median": (
            float(np.median(fit_sim_render_caked_deltas))
            if fit_sim_render_caked_deltas
            else float("nan")
        ),
        "fit_sim_render_caked_delta_max": (
            float(np.max(fit_sim_render_caked_deltas))
            if fit_sim_render_caked_deltas
            else float("nan")
        ),
        "caked_visual_detector_display_input_count": float(caked_detector_display_input_count),
        **visual_stats,
    }

    warning = ""
    render_delta_max = float(stats["fit_sim_render_caked_delta_max"])
    if fit_sim_render_caked_deltas and np.isfinite(render_delta_max) and render_delta_max > 1.0e-6:
        warning = (
            "Fit-sim overlay mismatch suspect: fitted marker caked positions differ from "
            "the rendered simulation caked positions."
        )
    sim_med = float(stats["sim_display_med_px"])
    bg_med = float(stats["bg_display_med_px"])
    if (
        not warning
        and len(sim_frame_dists) >= 3
        and np.isfinite(sim_med)
        and np.isfinite(bg_med)
        and sim_med - bg_med > 40.0
        and bg_med <= 0.6 * sim_med
    ):
        warning = (
            "Frame mismatch suspect: fitted simulation overlay points do not land in the "
            "same display frame as the fixed background picks."
        )
    initial_visual_med = float(stats.get("initial_distance_median", np.nan))
    final_visual_med = float(stats.get("final_distance_median", np.nan))
    if (
        not warning
        and int(stats.get("paired_visual_records", 0.0)) >= 3
        and np.isfinite(initial_visual_med)
        and np.isfinite(final_visual_med)
        and final_visual_med > initial_visual_med + max(5.0, 0.1 * initial_visual_med)
    ):
        warning = (
            "Visual fit suspect: fitted simulation overlay points are farther from the "
            "saved background picks than the starting simulation points."
        )

    return stats, warning
