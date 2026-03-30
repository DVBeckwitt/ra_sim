"""Minimal Rigaku RAXIS ``.osc`` reader used by RA-SIM runtime workflows."""

from __future__ import annotations

from os import PathLike

import numpy as np

RAXIS_SIGNATURE = b"RAXIS"
RAXIS_HEADER_BYTES = 6000


class ShapeError(Exception):
    """Raised when an OSC payload cannot be reshaped to its declared size."""


def _interpret(raw_bytes: np.ndarray) -> np.ndarray:
    version = raw_bytes[796:800].view(">u4")[0]
    endian = ">" if version < 20 else "<"

    width = int(raw_bytes[768:772].view(endian + "u4")[0])
    height = int(raw_bytes[772:776].view(endian + "u4")[0])
    pixel_count = width * height

    try:
        pixel_data = (
            raw_bytes[RAXIS_HEADER_BYTES:]
            .view(endian + "u2")[:pixel_count]
            .reshape((height, width))
        )
    except ValueError as exc:
        raise ShapeError(
            f"Could not reshape OSC pixel data to {height}x{width}."
        ) from exc

    int32_arr = pixel_data.astype(np.int32, copy=False)
    signed_mask = int32_arr >= 0x8000
    int32_arr[signed_mask] -= 0x10000
    int32_arr[signed_mask] += 0x8000
    int32_arr[signed_mask] *= 32
    return int32_arr


def read_osc(
    filename: str | PathLike[str],
    RAW: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Read one OSC file and return its interpreted detector image."""

    raw_bytes = np.fromfile(filename, dtype="u1")
    if raw_bytes[: len(RAXIS_SIGNATURE)].tobytes() != RAXIS_SIGNATURE:
        raise IOError(
            "This file does not start with the expected 'RAXIS' signature."
        )

    image = _interpret(raw_bytes)
    if RAW:
        return image, raw_bytes
    return image


__all__ = ["ShapeError", "read_osc"]
