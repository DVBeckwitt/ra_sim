"""Detector-center remap cache helpers.

Fast detector-center remaps are exact only when the cached hit set is not
bounded by the old detector support. Clipped hit tables are deliberately not
accepted as proof.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorCenterRemapCacheState:
    detector_relative_hit_tables: object = None
    unclipped_hit_tables: object = None
    clipped_hit_tables: object = None
    has_detector_relative_hit_tables: bool = False
    has_unclipped_hit_tables: bool = False


def _as_center_row_col(center: Sequence[object]) -> tuple[float, float]:
    """Return a detector-center sequence as native ``(row, col)`` pixels."""

    if len(center) < 2:
        raise ValueError("center must contain row and column")
    center_row = float(center[0])
    center_col = float(center[1])
    if not np.isfinite(center_row) or not np.isfinite(center_col):
        raise ValueError("center values must be finite")
    return center_row, center_col


def _copy_hit_tables(hit_tables: object) -> list[np.ndarray]:
    if hit_tables is None:
        return []
    if isinstance(hit_tables, np.ndarray):
        raw_tables = [hit_tables]
    elif isinstance(hit_tables, Mapping):
        raw_tables = list(hit_tables.values())
    elif isinstance(hit_tables, (str, bytes)):
        return []
    else:
        try:
            raw_tables = list(hit_tables)
        except TypeError:
            return []

    copied: list[np.ndarray] = []
    for table in raw_tables:
        arr = np.asarray(table, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("hit table must be a 2D array")
        if arr.shape[1] < 3:
            raise ValueError("hit table must include intensity, col, and row columns")
        copied.append(arr.copy())
    return copied


def translate_hit_tables_for_center_delta(
    hit_tables: object,
    delta_row: float,
    delta_col: float,
    image_size: int,
) -> list[np.ndarray]:
    """Translate absolute hit-table row/col coordinates without clipping."""

    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    translated = _copy_hit_tables(hit_tables)
    for table in translated:
        table[:, 1] += float(delta_col)
        table[:, 2] += float(delta_row)
    return translated


def make_relative_hit_tables_for_center(
    hit_tables: object,
    center: Sequence[object],
) -> list[np.ndarray]:
    """Store hit-table col/row coordinates relative to native ``(row, col)`` center."""

    center_row, center_col = _as_center_row_col(center)
    relative_tables = _copy_hit_tables(hit_tables)
    for table in relative_tables:
        table[:, 1] -= center_col
        table[:, 2] -= center_row
    return relative_tables


def materialize_absolute_hit_tables_from_relative(
    relative_tables: object,
    new_center: Sequence[object],
) -> list[np.ndarray]:
    """Convert detector-relative hit tables to absolute col/row detector coordinates."""

    center_row, center_col = _as_center_row_col(new_center)
    absolute_tables = _copy_hit_tables(relative_tables)
    for table in absolute_tables:
        table[:, 1] += center_col
        table[:, 2] += center_row
    return absolute_tables


def _cache_value(cache_state: object, name: str) -> object:
    if cache_state is None:
        return None
    if isinstance(cache_state, Mapping):
        return cache_state.get(name)
    return getattr(cache_state, name, None)


def _payload_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, Mapping):
        return len(value) > 0
    if isinstance(value, (str, bytes)):
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def _finite_center(center: Sequence[object]) -> bool:
    try:
        _as_center_row_col(center)
    except Exception:
        return False
    return True


def can_remap_detector_center_exactly(
    cache_state: object,
    old_center: Sequence[object],
    new_center: Sequence[object],
    image_size: int,
) -> bool:
    """Return true only for detector-relative or unclipped hit-table caches."""

    if cache_state is None or int(image_size) <= 0:
        return False
    if not (_finite_center(old_center) and _finite_center(new_center)):
        return False

    has_relative = bool(
        _cache_value(cache_state, "has_detector_relative_hit_tables")
    ) or _payload_present(_cache_value(cache_state, "detector_relative_hit_tables"))
    has_unclipped = bool(_cache_value(cache_state, "has_unclipped_hit_tables")) or (
        _payload_present(_cache_value(cache_state, "unclipped_hit_tables"))
    )
    return bool(has_relative or has_unclipped)
