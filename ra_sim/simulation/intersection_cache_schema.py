"""Shared helpers for supported intersection-cache table layouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEGACY_DETECTOR_CACHE_WIDTH = 14
CURRENT_DETECTOR_CACHE_WIDTH = 17
LEGACY_CAKED_CACHE_WIDTH = 16
CURRENT_CAKED_CACHE_WIDTH = 19

BASE_HIT_ROW_WIDTH = 7
HIT_ROW_WITH_PROVENANCE_WIDTH = 10

MIN_INTERSECTION_CACHE_WIDTH = 9

CACHE_COL_QR = 0
CACHE_COL_QZ = 1
CACHE_COL_DETECTOR_COL = 2
CACHE_COL_DETECTOR_ROW = 3
CACHE_COL_INTENSITY = 4
CACHE_COL_PHI = 5
CACHE_COL_H = 6
CACHE_COL_K = 7
CACHE_COL_L = 8
CACHE_COL_SOURCE_TABLE_INDEX = 14
CACHE_COL_SOURCE_ROW_INDEX = 15
CACHE_COL_BEST_SAMPLE_INDEX = 16
CACHE_COL_CAKED_TWO_THETA = 17
CACHE_COL_CAKED_PHI = 18

LEGACY_CAKED_COL_TWO_THETA = 14
LEGACY_CAKED_COL_PHI = 15
CURRENT_CAKED_COL_TWO_THETA = CACHE_COL_CAKED_TWO_THETA
CURRENT_CAKED_COL_PHI = CACHE_COL_CAKED_PHI

HIT_ROW_COL_INTENSITY = 0
HIT_ROW_COL_DETECTOR_COL = 1
HIT_ROW_COL_DETECTOR_ROW = 2
HIT_ROW_COL_PHI = 3
HIT_ROW_COL_H = 4
HIT_ROW_COL_K = 5
HIT_ROW_COL_L = 6
HIT_ROW_COL_SOURCE_TABLE_INDEX = 7
HIT_ROW_COL_SOURCE_ROW_INDEX = 8
HIT_ROW_COL_BEST_SAMPLE_INDEX = 9

_CACHE_KIND_INVALID = "invalid"
_CACHE_KIND_DETECTOR = "detector"
_CACHE_KIND_CAKED = "caked"


@dataclass(frozen=True)
class IntersectionCacheSchema:
    """Classification for one supported cache-table layout."""

    width: int
    cache_kind: str
    has_provenance: bool
    has_cached_caked_angles: bool
    is_valid: bool

    @property
    def is_detector_cache(self) -> bool:
        return self.is_valid and self.cache_kind == _CACHE_KIND_DETECTOR

    @property
    def is_caked_cache(self) -> bool:
        return self.is_valid and self.cache_kind == _CACHE_KIND_CAKED

    @property
    def hit_row_width(self) -> int:
        return (
            HIT_ROW_WITH_PROVENANCE_WIDTH
            if self.has_provenance
            else BASE_HIT_ROW_WIDTH
        )


def _schema_from_width(width: int) -> IntersectionCacheSchema:
    width = int(width)
    if width == LEGACY_DETECTOR_CACHE_WIDTH:
        return IntersectionCacheSchema(
            width=width,
            cache_kind=_CACHE_KIND_DETECTOR,
            has_provenance=False,
            has_cached_caked_angles=False,
            is_valid=True,
        )
    if width == CURRENT_DETECTOR_CACHE_WIDTH:
        return IntersectionCacheSchema(
            width=width,
            cache_kind=_CACHE_KIND_DETECTOR,
            has_provenance=True,
            has_cached_caked_angles=False,
            is_valid=True,
        )
    if width == LEGACY_CAKED_CACHE_WIDTH:
        return IntersectionCacheSchema(
            width=width,
            cache_kind=_CACHE_KIND_CAKED,
            has_provenance=False,
            has_cached_caked_angles=True,
            is_valid=True,
        )
    if width == CURRENT_CAKED_CACHE_WIDTH:
        return IntersectionCacheSchema(
            width=width,
            cache_kind=_CACHE_KIND_CAKED,
            has_provenance=True,
            has_cached_caked_angles=True,
            is_valid=True,
        )
    return IntersectionCacheSchema(
        width=max(0, width),
        cache_kind=_CACHE_KIND_INVALID,
        has_provenance=False,
        has_cached_caked_angles=False,
        is_valid=False,
    )


def empty_detector_cache_table(*, legacy: bool = False) -> np.ndarray:
    """Return canonical empty detector cache table."""

    width = LEGACY_DETECTOR_CACHE_WIDTH if legacy else CURRENT_DETECTOR_CACHE_WIDTH
    return np.empty((0, width), dtype=np.float64)


def empty_caked_cache_table(*, legacy: bool = False) -> np.ndarray:
    """Return canonical empty caked cache table."""

    width = LEGACY_CAKED_CACHE_WIDTH if legacy else CURRENT_CAKED_CACHE_WIDTH
    return np.empty((0, width), dtype=np.float64)


def empty_hit_table(*, with_provenance: bool = False) -> np.ndarray:
    """Return canonical empty hit-row table."""

    width = (
        HIT_ROW_WITH_PROVENANCE_WIDTH
        if with_provenance
        else BASE_HIT_ROW_WIDTH
    )
    return np.empty((0, width), dtype=np.float64)


def coerce_float64_table(
    table: object,
    *,
    empty_width: int = 0,
) -> np.ndarray:
    """Return detached float64 2D array or canonical empty table."""

    empty_width = max(0, int(empty_width))
    try:
        arr = np.asarray(table, dtype=np.float64)
    except Exception:
        return np.empty((0, empty_width), dtype=np.float64)

    if arr.ndim == 1:
        if arr.size == 0:
            return np.empty((0, empty_width), dtype=np.float64)
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return np.empty((0, empty_width), dtype=np.float64)
    return np.array(arr, dtype=np.float64, copy=True)


def classify_intersection_cache_table(table: object) -> IntersectionCacheSchema:
    """Classify one cache table among supported layouts."""

    arr = coerce_float64_table(table, empty_width=0)
    if arr.ndim != 2:
        return _schema_from_width(0)
    width = int(arr.shape[1])
    if width < MIN_INTERSECTION_CACHE_WIDTH:
        return _schema_from_width(width)
    return _schema_from_width(width)


def is_intersection_cache_table(table: object) -> bool:
    """Return whether one table matches supported cache layouts."""

    return bool(classify_intersection_cache_table(table).is_valid)


def _coerce_row(row: object) -> np.ndarray:
    try:
        return np.asarray(row, dtype=np.float64).reshape(-1)
    except Exception:
        return np.empty((0,), dtype=np.float64)


def _optional_int(value: object) -> int | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return int(np.rint(parsed))


def cache_table_to_hit_table(table: object) -> np.ndarray:
    """Convert one supported cache table into one hit-row table."""

    arr = coerce_float64_table(table, empty_width=0)
    schema = classify_intersection_cache_table(arr)
    if not schema.is_valid:
        return empty_hit_table()

    hit_table = np.full((arr.shape[0], schema.hit_row_width), np.nan, dtype=np.float64)
    hit_table[:, HIT_ROW_COL_INTENSITY] = arr[:, CACHE_COL_INTENSITY]
    hit_table[:, HIT_ROW_COL_DETECTOR_COL] = arr[:, CACHE_COL_DETECTOR_COL]
    hit_table[:, HIT_ROW_COL_DETECTOR_ROW] = arr[:, CACHE_COL_DETECTOR_ROW]
    hit_table[:, HIT_ROW_COL_PHI] = arr[:, CACHE_COL_PHI]
    hit_table[:, HIT_ROW_COL_H] = arr[:, CACHE_COL_H]
    hit_table[:, HIT_ROW_COL_K] = arr[:, CACHE_COL_K]
    hit_table[:, HIT_ROW_COL_L] = arr[:, CACHE_COL_L]
    if schema.has_provenance:
        hit_table[:, HIT_ROW_COL_SOURCE_TABLE_INDEX] = arr[:, CACHE_COL_SOURCE_TABLE_INDEX]
        hit_table[:, HIT_ROW_COL_SOURCE_ROW_INDEX] = arr[:, CACHE_COL_SOURCE_ROW_INDEX]
        hit_table[:, HIT_ROW_COL_BEST_SAMPLE_INDEX] = arr[:, CACHE_COL_BEST_SAMPLE_INDEX]

    finite_rows = np.isfinite(hit_table[:, HIT_ROW_COL_INTENSITY])
    finite_rows &= np.isfinite(hit_table[:, HIT_ROW_COL_DETECTOR_COL])
    finite_rows &= np.isfinite(hit_table[:, HIT_ROW_COL_DETECTOR_ROW])
    finite_rows &= np.isfinite(hit_table[:, HIT_ROW_COL_H])
    finite_rows &= np.isfinite(hit_table[:, HIT_ROW_COL_K])
    finite_rows &= np.isfinite(hit_table[:, HIT_ROW_COL_L])
    return np.array(hit_table[finite_rows], dtype=np.float64, copy=True)


def extract_cache_row_provenance(row: object) -> tuple[int | None, int | None, int | None]:
    """Return optional provenance from one cache row."""

    row_arr = _coerce_row(row)
    schema = _schema_from_width(int(row_arr.shape[0]))
    if not (schema.is_valid and schema.has_provenance):
        return (None, None, None)
    return (
        _optional_int(row_arr[CACHE_COL_SOURCE_TABLE_INDEX]),
        _optional_int(row_arr[CACHE_COL_SOURCE_ROW_INDEX]),
        _optional_int(row_arr[CACHE_COL_BEST_SAMPLE_INDEX]),
    )


def extract_hit_row_provenance(row: object) -> tuple[int | None, int | None, int | None]:
    """Return optional provenance from one hit row."""

    row_arr = _coerce_row(row)
    if row_arr.shape[0] < HIT_ROW_WITH_PROVENANCE_WIDTH:
        return (None, None, None)
    return (
        _optional_int(row_arr[HIT_ROW_COL_SOURCE_TABLE_INDEX]),
        _optional_int(row_arr[HIT_ROW_COL_SOURCE_ROW_INDEX]),
        _optional_int(row_arr[HIT_ROW_COL_BEST_SAMPLE_INDEX]),
    )


def extract_cached_caked_angles(row: object) -> tuple[float, float]:
    """Return cached ``(2theta, phi)`` when one cache row carries them."""

    row_arr = _coerce_row(row)
    schema = _schema_from_width(int(row_arr.shape[0]))
    if not (schema.is_valid and schema.has_cached_caked_angles):
        return (float("nan"), float("nan"))
    if schema.width == LEGACY_CAKED_CACHE_WIDTH:
        return (
            float(row_arr[LEGACY_CAKED_COL_TWO_THETA]),
            float(row_arr[LEGACY_CAKED_COL_PHI]),
        )
    return (
        float(row_arr[CURRENT_CAKED_COL_TWO_THETA]),
        float(row_arr[CURRENT_CAKED_COL_PHI]),
    )


__all__ = [
    "BASE_HIT_ROW_WIDTH",
    "CACHE_COL_BEST_SAMPLE_INDEX",
    "CACHE_COL_CAKED_PHI",
    "CACHE_COL_CAKED_TWO_THETA",
    "CACHE_COL_DETECTOR_COL",
    "CACHE_COL_DETECTOR_ROW",
    "CACHE_COL_H",
    "CACHE_COL_INTENSITY",
    "CACHE_COL_K",
    "CACHE_COL_L",
    "CACHE_COL_PHI",
    "CACHE_COL_QR",
    "CACHE_COL_QZ",
    "CACHE_COL_SOURCE_ROW_INDEX",
    "CACHE_COL_SOURCE_TABLE_INDEX",
    "CURRENT_CAKED_CACHE_WIDTH",
    "CURRENT_CAKED_COL_PHI",
    "CURRENT_CAKED_COL_TWO_THETA",
    "CURRENT_DETECTOR_CACHE_WIDTH",
    "HIT_ROW_COL_BEST_SAMPLE_INDEX",
    "HIT_ROW_COL_DETECTOR_COL",
    "HIT_ROW_COL_DETECTOR_ROW",
    "HIT_ROW_COL_H",
    "HIT_ROW_COL_INTENSITY",
    "HIT_ROW_COL_K",
    "HIT_ROW_COL_L",
    "HIT_ROW_COL_PHI",
    "HIT_ROW_COL_SOURCE_ROW_INDEX",
    "HIT_ROW_COL_SOURCE_TABLE_INDEX",
    "HIT_ROW_WITH_PROVENANCE_WIDTH",
    "IntersectionCacheSchema",
    "LEGACY_CAKED_COL_PHI",
    "LEGACY_CAKED_COL_TWO_THETA",
    "LEGACY_CAKED_CACHE_WIDTH",
    "LEGACY_DETECTOR_CACHE_WIDTH",
    "MIN_INTERSECTION_CACHE_WIDTH",
    "cache_table_to_hit_table",
    "classify_intersection_cache_table",
    "coerce_float64_table",
    "empty_caked_cache_table",
    "empty_detector_cache_table",
    "empty_hit_table",
    "extract_cache_row_provenance",
    "extract_cached_caked_angles",
    "extract_hit_row_provenance",
    "is_intersection_cache_table",
]
