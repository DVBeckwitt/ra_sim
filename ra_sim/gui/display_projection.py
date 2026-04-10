from __future__ import annotations

import math
from typing import Any

import numpy as np


MIN_DISPLAY_RASTER_SIZE = 1
MAX_DISPLAY_RASTER_SIZE = 3000


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


def downsample_raster_for_display(
    image: np.ndarray | None,
    *,
    max_size: Any,
) -> np.ndarray | None:
    """Return a cheap strided view capped to *max_size* for display only."""

    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim < 2:
        return arr

    limit = normalize_display_raster_size_limit(
        max_size,
        fallback=max(arr.shape[:2]),
    )
    max_dim = max(int(arr.shape[0]), int(arr.shape[1]))
    if max_dim <= limit:
        return arr

    stride = max(1, int(math.ceil(max_dim / float(limit))))
    if stride <= 1:
        return arr
    if arr.ndim == 2:
        return arr[::stride, ::stride]
    return arr[::stride, ::stride, ...]
