"""Import-safe helpers for detector-space peak-position preview images."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def build_peak_position_preview_image(
    hit_tables: Sequence[object] | None,
    *,
    image_size: int,
    hit_tables_to_max_positions: Callable[[Sequence[object]], Sequence[Sequence[float]]],
) -> np.ndarray:
    """Return a lightweight marker image from clustered peak-center positions."""

    size = max(0, int(image_size))
    image = np.zeros((size, size), dtype=np.float64)
    if size <= 0 or not hit_tables:
        return image

    try:
        max_positions = np.asarray(
            hit_tables_to_max_positions(hit_tables),
            dtype=np.float64,
        )
    except Exception:
        return image
    if max_positions.ndim != 2 or max_positions.shape[1] < 6:
        return image

    marker_kernel = (
        (0, 0, 1.0),
        (-1, 0, 0.35),
        (1, 0, 0.35),
        (0, -1, 0.35),
        (0, 1, 0.35),
        (-1, -1, 0.15),
        (-1, 1, 0.15),
        (1, -1, 0.15),
        (1, 1, 0.15),
    )

    for row in max_positions:
        for offset in (0, 3):
            intensity = float(row[offset])
            col = float(row[offset + 1])
            row_pix = float(row[offset + 2])
            if not (
                np.isfinite(intensity)
                and intensity > 0.0
                and np.isfinite(col)
                and np.isfinite(row_pix)
            ):
                continue
            col_idx = int(round(col))
            row_idx = int(round(row_pix))
            for d_row, d_col, value in marker_kernel:
                rr = row_idx + int(d_row)
                cc = col_idx + int(d_col)
                if 0 <= rr < size and 0 <= cc < size:
                    image[rr, cc] += float(value)

    return image
