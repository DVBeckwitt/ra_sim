from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


MIN_DISPLAY_RASTER_SIZE = 1
MAX_DISPLAY_RASTER_SIZE = 3000
DISPLAY_PROJECTION_OVERSCAN_FRACTION = 1.0
DISPLAY_PROJECTION_VIEWPORT_OVERSAMPLE = 1.0
_PROJECTION_EPSILON = 1.0e-12


@dataclass(frozen=True)
class RasterProjection:
    """One display-ready raster plus the extent it should cover."""

    image: np.ndarray
    extent: tuple[float, float, float, float]


def normalize_display_raster_size_limit(
    value: Any,
    *,
    fallback: Any,
    minimum: int = MIN_DISPLAY_RASTER_SIZE,
    maximum: int = MAX_DISPLAY_RASTER_SIZE,
) -> int:
    """Clamp a display-only raster-size cap into the supported UI range."""

    lower = max(int(minimum), 1)
    upper = max(int(maximum), lower)
    try:
        candidate = int(round(float(value)))
    except Exception:
        try:
            candidate = int(round(float(fallback)))
        except Exception:
            candidate = upper
    return max(lower, min(candidate, upper))


def default_display_raster_size(image_size: Any) -> int:
    """Return the default display cap for one image size."""

    return normalize_display_raster_size_limit(
        image_size,
        fallback=MAX_DISPLAY_RASTER_SIZE,
    )


def _finite_pair(
    values: tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        return None
    try:
        first = float(values[0])
        second = float(values[1])
    except Exception:
        return None
    if not math.isfinite(first) or not math.isfinite(second):
        return None
    return (first, second)


def _positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _expanded_window(
    limits: tuple[float, float],
    *,
    axis_start: float,
    axis_end: float,
    overscan_fraction: float,
) -> tuple[float, float]:
    lo = min(float(limits[0]), float(limits[1]))
    hi = max(float(limits[0]), float(limits[1]))
    axis_lo = min(float(axis_start), float(axis_end))
    axis_hi = max(float(axis_start), float(axis_end))
    if axis_hi - axis_lo <= _PROJECTION_EPSILON:
        return (axis_lo, axis_hi)
    span = max(hi - lo, 0.0)
    if span <= _PROJECTION_EPSILON:
        span = axis_hi - axis_lo
    overscan = max(span * max(float(overscan_fraction), 0.0), 0.0)
    expanded_lo = max(axis_lo, lo - overscan)
    expanded_hi = min(axis_hi, hi + overscan)
    if expanded_hi - expanded_lo <= _PROJECTION_EPSILON:
        return (axis_lo, axis_hi)
    return (expanded_lo, expanded_hi)


def _slice_for_axis(
    start_edge: float,
    end_edge: float,
    count: int,
    *,
    view_limits: tuple[float, float],
) -> tuple[int, int, float, float]:
    if count <= 0:
        return (0, 0, float(start_edge), float(end_edge))

    start = float(start_edge)
    end = float(end_edge)
    limits = _finite_pair(view_limits)
    if limits is None or abs(end - start) <= _PROJECTION_EPSILON:
        return (0, int(count), start, end)

    lo = min(float(limits[0]), float(limits[1]))
    hi = max(float(limits[0]), float(limits[1]))
    step = (end - start) / float(count)
    if end >= start:
        start_idx = int(math.floor((lo - start) / step))
        stop_idx = int(math.ceil((hi - start) / step))
    else:
        step_abs = abs(step)
        start_idx = int(math.floor((start - hi) / step_abs))
        stop_idx = int(math.ceil((start - lo) / step_abs))

    start_idx = max(0, min(int(count) - 1, start_idx))
    stop_idx = max(start_idx + 1, min(int(count), stop_idx))
    projected_start = start + step * float(start_idx)
    projected_end = start + step * float(stop_idx)
    return (start_idx, stop_idx, float(projected_start), float(projected_end))


def _target_shape(
    *,
    rows: int,
    cols: int,
    max_size: Any,
    bbox_width_px: Any,
    bbox_height_px: Any,
) -> tuple[int, int]:
    if rows <= 0 or cols <= 0:
        return (max(rows, 0), max(cols, 0))

    limit = normalize_display_raster_size_limit(
        max_size,
        fallback=max(rows, cols),
    )
    longest = max(rows, cols)
    scale_limit = min(1.0, float(limit) / float(longest))
    target_rows = max(1, int(math.ceil(rows * scale_limit)))
    target_cols = max(1, int(math.ceil(cols * scale_limit)))

    bbox_width = _positive_float(bbox_width_px)
    bbox_height = _positive_float(bbox_height_px)
    if bbox_width is not None:
        target_cols = min(
            target_cols,
            max(
                1,
                int(
                    math.ceil(
                        min(
                            1.0,
                            (bbox_width * DISPLAY_PROJECTION_VIEWPORT_OVERSAMPLE)
                            / float(cols),
                        )
                        * cols
                    )
                ),
            ),
        )
    if bbox_height is not None:
        target_rows = min(
            target_rows,
            max(
                1,
                int(
                    math.ceil(
                        min(
                            1.0,
                            (bbox_height * DISPLAY_PROJECTION_VIEWPORT_OVERSAMPLE)
                            / float(rows),
                        )
                        * rows
                    )
                ),
            ),
        )

    return (target_rows, target_cols)


def _downsample_indices(length: int, target: int) -> np.ndarray | None:
    if int(length) <= int(target):
        return None
    target_count = max(1, int(target))
    sample_positions = (
        ((np.arange(target_count, dtype=np.float64) + 0.5) * float(length))
        / float(target_count)
    ) - 0.5
    indices = np.asarray(np.floor(sample_positions + 0.5), dtype=np.int64)
    return np.clip(indices, 0, int(length) - 1)


def _stride_downsample(
    image: np.ndarray,
    *,
    target_rows: int,
    target_cols: int,
) -> np.ndarray:
    rows = int(image.shape[0])
    cols = int(image.shape[1])
    if rows <= target_rows and cols <= target_cols:
        return image

    row_indices = _downsample_indices(rows, target_rows)
    col_indices = _downsample_indices(cols, target_cols)
    if row_indices is None and col_indices is None:
        return image
    if row_indices is None:
        if image.ndim == 2:
            return image[:, col_indices]
        return image[:, col_indices, ...]
    if col_indices is None:
        return image[row_indices, ...]
    if image.ndim == 2:
        return image[np.ix_(row_indices, col_indices)]
    return image[row_indices][:, col_indices, ...]


def project_raster_to_view(
    image: np.ndarray | None,
    *,
    extent: tuple[float, float, float, float] | list[float],
    axis_xlim: tuple[float, float] | list[float] | None,
    axis_ylim: tuple[float, float] | list[float] | None,
    max_size: Any,
    bbox_width_px: Any = None,
    bbox_height_px: Any = None,
    overscan_fraction: float = DISPLAY_PROJECTION_OVERSCAN_FRACTION,
) -> RasterProjection | None:
    """Crop the visible window plus overscan, then cap it to the screen budget."""

    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim < 2:
        extent_values = tuple(float(value) for value in extent)
        return RasterProjection(image=arr, extent=extent_values)

    try:
        full_extent = tuple(float(value) for value in extent)
    except Exception:
        full_extent = (0.0, float(arr.shape[1]), float(arr.shape[0]), 0.0)
    if len(full_extent) != 4:
        full_extent = (0.0, float(arr.shape[1]), float(arr.shape[0]), 0.0)

    xlim = _finite_pair(axis_xlim)
    ylim = _finite_pair(axis_ylim)
    if xlim is None:
        xlim = (full_extent[0], full_extent[1])
    if ylim is None:
        ylim = (full_extent[2], full_extent[3])

    x_window = _expanded_window(
        xlim,
        axis_start=full_extent[0],
        axis_end=full_extent[1],
        overscan_fraction=overscan_fraction,
    )
    y_window = _expanded_window(
        ylim,
        axis_start=full_extent[2],
        axis_end=full_extent[3],
        overscan_fraction=overscan_fraction,
    )

    col_start, col_stop, projected_x0, projected_x1 = _slice_for_axis(
        full_extent[0],
        full_extent[1],
        int(arr.shape[1]),
        view_limits=x_window,
    )
    row_start, row_stop, projected_y0, projected_y1 = _slice_for_axis(
        full_extent[2],
        full_extent[3],
        int(arr.shape[0]),
        view_limits=y_window,
    )

    if (
        row_start == 0
        and row_stop == int(arr.shape[0])
        and col_start == 0
        and col_stop == int(arr.shape[1])
    ):
        cropped = arr
    else:
        cropped = arr[row_start:row_stop, col_start:col_stop]
    target_rows, target_cols = _target_shape(
        rows=int(cropped.shape[0]),
        cols=int(cropped.shape[1]),
        max_size=max_size,
        bbox_width_px=bbox_width_px,
        bbox_height_px=bbox_height_px,
    )
    projected = _stride_downsample(
        cropped,
        target_rows=target_rows,
        target_cols=target_cols,
    )
    return RasterProjection(
        image=projected,
        extent=(
            float(projected_x0),
            float(projected_x1),
            float(projected_y0),
            float(projected_y1),
        ),
    )


def downsample_raster_for_display(
    image: np.ndarray | None,
    *,
    max_size: Any,
) -> np.ndarray | None:
    """Backwards-compatible whole-image sampled-down projection helper."""

    projection = project_raster_to_view(
        image,
        extent=(0.0, float(np.asarray(image).shape[1]), float(np.asarray(image).shape[0]), 0.0)
        if image is not None and np.asarray(image).ndim >= 2
        else (0.0, 1.0, 1.0, 0.0),
        axis_xlim=None,
        axis_ylim=None,
        max_size=max_size,
        bbox_width_px=None,
        bbox_height_px=None,
        overscan_fraction=0.0,
    )
    if projection is None:
        return None
    return projection.image
