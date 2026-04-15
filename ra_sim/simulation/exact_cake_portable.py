"""Portable flat-detector wrapper for the fast exact detector-to-cake splitter.

Example
-------
```python
import numpy as np

from ra_sim.simulation.exact_cake_portable import convert_image_to_angle_space

image = np.load("detector.npy")
result = convert_image_to_angle_space(
    image,
    pixel_size_m=1.0e-4,
    distance_m=0.075,
    poni1_m=0.1596422,
    poni2_m=0.1453120,
    npt_rad=1000,
    npt_azim=720,
    tth_min_deg=0.0,
    tth_max_deg=90.0,
    correct_solid_angle=True,
    engine="numba",
    workers=24,
)

cake = result.intensity
two_theta_deg = result.radial_deg
azimuth_deg = result.azimuthal_deg
```
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import threading

import numpy as np

from ra_sim.simulation.exact_cake import (
    DetectorCakeGeometry,
    DetectorCakeLUT,
    DetectorCakeResult,
    build_detector_to_cake_lut,
    integrate_detector_to_cake_exact,
    integrate_detector_to_cake_lut,
)


PHI_ZERO_OFFSET_DEGREES = -90.0
_PROCESS_DETECTOR_MAP_CACHE_LIMIT = 2
_PROCESS_CAKE_LUT_CACHE_LIMIT = 2
_GEOMETRY_WARMUP_COMPLETION_LIMIT = 16


_SharedDetectorMapKey = tuple["PortableGeometry", tuple[int, int]]
_SharedCakeLutKey = tuple["PortableGeometry", tuple[int, int], int, float, float, int, float, float]
_GeometryWarmupKey = tuple[int, tuple[int, int], int, int]

_PROCESS_DETECTOR_MAP_CACHE: OrderedDict[_SharedDetectorMapKey, tuple[np.ndarray, np.ndarray]] = OrderedDict()
_PROCESS_CAKE_LUT_CACHE: OrderedDict[_SharedCakeLutKey, DetectorCakeLUT] = OrderedDict()
_PROCESS_DETECTOR_MAP_CACHE_LOCK = threading.RLock()
_PROCESS_CAKE_LUT_CACHE_LOCK = threading.RLock()
_EXACT_CAKE_GEOMETRY_WARMUP_THREADS: dict[_GeometryWarmupKey, threading.Thread] = {}
_EXACT_CAKE_GEOMETRY_WARMUP_COMPLETED: OrderedDict[_GeometryWarmupKey, None] = OrderedDict()
_EXACT_CAKE_GEOMETRY_WARMUP_LOCK = threading.RLock()


@dataclass(frozen=True)
class PortableGeometry:
    pixel_size_m: float
    distance_m: float
    center_row_px: float
    center_col_px: float


@dataclass(frozen=True)
class CakeTransformBundle:
    detector_shape: tuple[int, int]
    radial_deg: np.ndarray
    raw_azimuth_deg: np.ndarray
    gui_azimuth_deg: np.ndarray
    lut: DetectorCakeLUT
    lut_t: object | None = None


def _clear_shared_exact_cake_caches() -> None:
    """Clear process-level detector-map and LUT caches."""

    with _PROCESS_DETECTOR_MAP_CACHE_LOCK:
        _PROCESS_DETECTOR_MAP_CACHE.clear()
    with _PROCESS_CAKE_LUT_CACHE_LOCK:
        _PROCESS_CAKE_LUT_CACHE.clear()
    with _EXACT_CAKE_GEOMETRY_WARMUP_LOCK:
        _EXACT_CAKE_GEOMETRY_WARMUP_THREADS.clear()
        _EXACT_CAKE_GEOMETRY_WARMUP_COMPLETED.clear()


def _shared_detector_map_cache_key(
    geometry: PortableGeometry,
    detector_shape: tuple[int, int],
) -> _SharedDetectorMapKey:
    return geometry, detector_shape


def _shared_cake_lut_cache_key(
    geometry: PortableGeometry,
    detector_shape: tuple[int, int],
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
) -> _SharedCakeLutKey:
    return (
        geometry,
        detector_shape,
        int(radial_deg.size),
        float(radial_deg[0]),
        float(radial_deg[-1]),
        int(azimuthal_deg.size),
        float(azimuthal_deg[0]),
        float(azimuthal_deg[-1]),
    )


def _shared_detector_maps_for_shape(
    detector_shape: tuple[int, int],
    geometry: PortableGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    key = _shared_detector_map_cache_key(geometry, detector_shape)
    with _PROCESS_DETECTOR_MAP_CACHE_LOCK:
        cached = _PROCESS_DETECTOR_MAP_CACHE.get(key)
        if cached is not None:
            _PROCESS_DETECTOR_MAP_CACHE.move_to_end(key)
            return cached

    two_theta_deg, raw_azimuth_deg = detector_pixel_angular_maps(detector_shape, geometry)
    two_theta_deg = np.asarray(two_theta_deg, dtype=np.float64)
    raw_azimuth_deg = np.asarray(raw_azimuth_deg, dtype=np.float64)
    two_theta_deg.setflags(write=False)
    raw_azimuth_deg.setflags(write=False)
    built = (two_theta_deg, raw_azimuth_deg)

    with _PROCESS_DETECTOR_MAP_CACHE_LOCK:
        cached = _PROCESS_DETECTOR_MAP_CACHE.get(key)
        if cached is not None:
            _PROCESS_DETECTOR_MAP_CACHE.move_to_end(key)
            return cached
        _PROCESS_DETECTOR_MAP_CACHE[key] = built
        while len(_PROCESS_DETECTOR_MAP_CACHE) > _PROCESS_DETECTOR_MAP_CACHE_LIMIT:
            _PROCESS_DETECTOR_MAP_CACHE.popitem(last=False)
    return built


def _shared_cake_lut_for_request(
    detector_shape: tuple[int, int],
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    geometry: DetectorCakeGeometry,
    portable_geometry: PortableGeometry,
    *,
    workers: int | str | None,
    strict: bool = False,
) -> DetectorCakeLUT | None:
    key = _shared_cake_lut_cache_key(portable_geometry, detector_shape, radial_deg, azimuthal_deg)
    with _PROCESS_CAKE_LUT_CACHE_LOCK:
        cached = _PROCESS_CAKE_LUT_CACHE.get(key)
        if cached is not None:
            _PROCESS_CAKE_LUT_CACHE.move_to_end(key)
            return cached

    try:
        built = build_detector_to_cake_lut(
            detector_shape,
            radial_deg,
            azimuthal_deg,
            geometry,
            workers=workers,
        )
    except (MemoryError, RuntimeError):
        if strict:
            raise
        return None
    if built is None:
        if strict:
            raise RuntimeError("Exact-cake LUT build returned no lookup table.")
        return None

    with _PROCESS_CAKE_LUT_CACHE_LOCK:
        cached = _PROCESS_CAKE_LUT_CACHE.get(key)
        if cached is not None:
            _PROCESS_CAKE_LUT_CACHE.move_to_end(key)
            return cached
        _PROCESS_CAKE_LUT_CACHE[key] = built
        while len(_PROCESS_CAKE_LUT_CACHE) > _PROCESS_CAKE_LUT_CACHE_LIMIT:
            _PROCESS_CAKE_LUT_CACHE.popitem(last=False)
    return built


def _normalize_detector_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, int] | None:
    if len(shape) < 2:
        return None
    detector_shape = tuple(int(v) for v in tuple(shape)[:2])
    if detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None
    return detector_shape


def _readonly_float64_vector(
    values: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    vector = np.array(np.asarray(values, dtype=np.float64).reshape(-1), copy=True)
    vector.setflags(write=False)
    return vector


def _normalize_lut_engine(engine: str) -> str:
    engine_name = str(engine).strip().lower()
    if engine_name not in {"", "auto", "python", "numba"}:
        raise ValueError("engine must be one of: auto, python, numba.")
    return engine_name


def _selection_exclusion_mask(
    image_shape: tuple[int, ...],
    rows: np.ndarray | None,
    cols: np.ndarray | None,
) -> np.ndarray | None:
    if (rows is None) != (cols is None):
        raise ValueError("rows and cols must both be provided or both be omitted.")
    if rows is None or cols is None:
        return None
    detector_shape = _normalize_detector_shape(tuple(image_shape))
    if detector_shape is None:
        raise ValueError("image must have two positive detector dimensions.")
    rows_array = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols_array = np.asarray(cols, dtype=np.int64).reshape(-1)
    if rows_array.shape != cols_array.shape:
        raise ValueError("rows and cols must have the same shape.")
    height, width = detector_shape
    if rows_array.size and (
        np.any(rows_array < 0)
        or np.any(rows_array >= height)
        or np.any(cols_array < 0)
        or np.any(cols_array >= width)
    ):
        raise ValueError("rows/cols contain indices outside the image bounds.")
    exclusion_mask = np.ones(detector_shape, dtype=bool)
    if rows_array.size:
        exclusion_mask[rows_array, cols_array] = False
    return exclusion_mask


def _merge_selection_into_mask(
    image_shape: tuple[int, ...],
    mask: np.ndarray | None,
    rows: np.ndarray | None,
    cols: np.ndarray | None,
) -> np.ndarray | None:
    selection_mask = _selection_exclusion_mask(image_shape, rows, cols)
    if selection_mask is None:
        return mask
    if mask is None:
        return selection_mask
    mask_array = np.asarray(mask)
    detector_shape = _normalize_detector_shape(tuple(image_shape))
    if detector_shape is None or mask_array.shape != detector_shape:
        raise ValueError("mask must match image shape.")
    return np.asarray(mask_array, dtype=bool) | selection_mask


def _integrate_detector_to_cake_lut_with_selection(
    image: np.ndarray,
    radial_deg: np.ndarray,
    azimuthal_deg: np.ndarray,
    lut: DetectorCakeLUT,
    *,
    normalization: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
) -> DetectorCakeResult:
    merged_mask = _merge_selection_into_mask(np.asarray(image).shape, mask, rows, cols)
    return integrate_detector_to_cake_lut(
        image,
        radial_deg,
        azimuthal_deg,
        lut,
        normalization=normalization,
        mask=merged_mask,
    )


def raw_phi_to_gui_phi(
    phi_values: np.ndarray | list[float] | tuple[float, ...] | float,
) -> np.ndarray:
    return ((PHI_ZERO_OFFSET_DEGREES - np.asarray(phi_values, dtype=np.float64) + 180.0) % 360.0) - 180.0


def gui_phi_to_raw_phi(
    phi_values: np.ndarray | list[float] | tuple[float, ...] | float,
) -> np.ndarray:
    return ((PHI_ZERO_OFFSET_DEGREES - np.asarray(phi_values, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _matrix_row_indices_and_weights(
    matrix: object,
    row_index: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if (
            hasattr(matrix, "indptr")
            and hasattr(matrix, "indices")
            and hasattr(matrix, "data")
        ):
            row_start = int(matrix.indptr[row_index])
            row_stop = int(matrix.indptr[row_index + 1])
            indices = np.asarray(matrix.indices[row_start:row_stop], dtype=np.int64)
            weights = np.asarray(matrix.data[row_start:row_stop], dtype=np.float64)
        else:
            row_weights = np.asarray(matrix[row_index], dtype=np.float64).reshape(-1)
            indices = np.flatnonzero(np.isfinite(row_weights) & (row_weights > 0.0))
            weights = row_weights[indices]
    except Exception:
        return None
    return indices, weights


def _weighted_wrapped_angle_mean_deg(
    angles_deg: np.ndarray,
    weights: np.ndarray,
) -> float:
    angle_values = np.asarray(angles_deg, dtype=np.float64).reshape(-1)
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    radians = np.deg2rad(angle_values)
    x = float(np.sum(np.cos(radians) * weight_values))
    y = float(np.sum(np.sin(radians) * weight_values))
    if not np.isfinite(x) or not np.isfinite(y) or (abs(x) <= 1.0e-15 and abs(y) <= 1.0e-15):
        return float(angle_values[int(np.argmax(weight_values))])
    return float(np.degrees(np.arctan2(y, x)))


def _bilinear_detector_pixel_weights(
    detector_shape: tuple[int, int],
    col: float,
    row: float,
) -> tuple[tuple[int, float], ...]:
    height, width = (int(detector_shape[0]), int(detector_shape[1]))
    if height <= 0 or width <= 0:
        return ()
    if not (
        np.isfinite(col)
        and np.isfinite(row)
        and 0.0 <= float(col) <= float(width - 1)
        and 0.0 <= float(row) <= float(height - 1)
    ):
        return ()

    col0 = int(math.floor(float(col)))
    row0 = int(math.floor(float(row)))
    col1 = min(col0 + 1, width - 1)
    row1 = min(row0 + 1, height - 1)
    tx = float(col) - float(col0)
    ty = float(row) - float(row0)

    weights_by_pixel: dict[int, float] = {}
    for pixel_col, pixel_row, weight in (
        (col0, row0, (1.0 - tx) * (1.0 - ty)),
        (col1, row0, tx * (1.0 - ty)),
        (col0, row1, (1.0 - tx) * ty),
        (col1, row1, tx * ty),
    ):
        weight_value = float(weight)
        if weight_value <= 0.0:
            continue
        pixel_index = int(int(pixel_row) * width + int(pixel_col))
        weights_by_pixel[pixel_index] = (
            float(weights_by_pixel.get(pixel_index, 0.0)) + weight_value
        )
    return tuple(
        (int(pixel_index), float(weight))
        for pixel_index, weight in weights_by_pixel.items()
        if weight > 0.0
    )


def cake_transform_bundle_lut_t(bundle: CakeTransformBundle) -> object:
    transposed = bundle.lut_t
    if transposed is not None:
        return transposed
    matrix = bundle.lut.matrix
    if hasattr(matrix, "transpose"):
        transposed = matrix.transpose()
        if hasattr(transposed, "tocsr"):
            transposed = transposed.tocsr()
    else:
        transposed = np.asarray(matrix, dtype=np.float64).T
    object.__setattr__(bundle, "lut_t", transposed)
    return transposed


def build_cake_transform_bundle(
    ai: FastAzimuthalIntegrator | None,
    detector_shape: tuple[int, ...] | list[int],
    radial_deg: np.ndarray | list[float] | tuple[float, ...],
    raw_azimuth_deg: np.ndarray | list[float] | tuple[float, ...],
    *,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> CakeTransformBundle | None:
    if not isinstance(ai, FastAzimuthalIntegrator):
        return None
    normalized_shape = _normalize_detector_shape(tuple(detector_shape))
    if normalized_shape is None:
        return None
    try:
        radial_axis = np.asarray(radial_deg, dtype=np.float64).reshape(-1)
        raw_axis = np.asarray(raw_azimuth_deg, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if radial_axis.size <= 0 or raw_axis.size <= 0:
        return None
    if not np.all(np.isfinite(radial_axis)) or not np.all(np.isfinite(raw_axis)):
        return None
    geometry = DetectorCakeGeometry(
        pixel_size_m=float(ai.geometry.pixel_size_m),
        distance_m=float(ai.geometry.distance_m),
        center_row_px=float(ai.geometry.center_row_px),
        center_col_px=float(ai.geometry.center_col_px),
    )
    try:
        lut = ai._cached_cake_lut(
            normalized_shape,
            radial_axis,
            raw_axis,
            geometry,
            engine=engine,
            workers=workers,
        )
        if lut is None:
            lut = _shared_cake_lut_for_request(
                normalized_shape,
                radial_axis,
                raw_axis,
                geometry,
                ai.geometry,
                workers=workers,
            )
    except Exception:
        return None
    if lut is None:
        return None
    radial_axis = _readonly_float64_vector(radial_axis)
    raw_axis = _readonly_float64_vector(raw_axis)
    gui_axis = _readonly_float64_vector(raw_phi_to_gui_phi(raw_axis))
    return CakeTransformBundle(
        detector_shape=normalized_shape,
        radial_deg=radial_axis,
        raw_azimuth_deg=raw_axis,
        gui_azimuth_deg=gui_axis,
        lut=lut,
    )


def build_cake_transform_bundle_from_result(
    ai: FastAzimuthalIntegrator | None,
    detector_shape: tuple[int, ...] | list[int],
    result: DetectorCakeResult | None,
    *,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> CakeTransformBundle | None:
    if not isinstance(result, DetectorCakeResult):
        return None
    return build_cake_transform_bundle(
        ai,
        detector_shape,
        result.radial_deg,
        result.azimuthal_deg,
        engine=engine,
        workers=workers,
    )


def detector_pixel_to_caked_bin(
    transform_bundle: CakeTransformBundle | None,
    col: float,
    row: float,
) -> tuple[float | None, float | None]:
    if not isinstance(transform_bundle, CakeTransformBundle):
        return None, None
    try:
        col_val = float(col)
        row_val = float(row)
    except Exception:
        return None, None
    if not np.isfinite(col_val) or not np.isfinite(row_val):
        return None, None
    height, width = transform_bundle.detector_shape
    if height <= 0 or width <= 0:
        return None, None
    matrix = cake_transform_bundle_lut_t(transform_bundle)
    pixel_weights = _bilinear_detector_pixel_weights(
        transform_bundle.detector_shape,
        col_val,
        row_val,
    )
    if len(pixel_weights) <= 0:
        return None, None
    n_rad = int(transform_bundle.radial_deg.size)
    n_az = int(transform_bundle.raw_azimuth_deg.size)
    if n_rad <= 0 or n_az <= 0:
        return None, None
    max_bin_index = int(n_rad * n_az)

    radial_numerator = 0.0
    total_weight = 0.0
    raw_angle_chunks: list[np.ndarray] = []
    raw_weight_chunks: list[np.ndarray] = []

    for pixel_index, detector_weight in pixel_weights:
        row_payload = _matrix_row_indices_and_weights(matrix, int(pixel_index))
        if row_payload is None:
            continue
        bin_indices, weights = row_payload
        valid = (
            np.isfinite(bin_indices)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        if not np.any(valid):
            continue
        bin_indices = np.asarray(bin_indices[valid], dtype=np.int64)
        valid_bins = (bin_indices >= 0) & (bin_indices < max_bin_index)
        if not np.any(valid_bins):
            continue
        bin_indices = np.asarray(bin_indices[valid_bins], dtype=np.int64)
        combined_weights = (
            np.asarray(weights[valid][valid_bins], dtype=np.float64)
            * float(detector_weight)
        )
        combined_weight_sum = float(np.sum(combined_weights))
        if not np.isfinite(combined_weight_sum) or combined_weight_sum <= 0.0:
            continue
        radial_indices = np.asarray(bin_indices % n_rad, dtype=np.int64)
        azimuth_indices = np.asarray(bin_indices // n_rad, dtype=np.int64)
        radial_numerator += float(
            np.sum(transform_bundle.radial_deg[radial_indices] * combined_weights)
        )
        total_weight += combined_weight_sum
        raw_angle_chunks.append(
            np.asarray(transform_bundle.raw_azimuth_deg[azimuth_indices], dtype=np.float64)
        )
        raw_weight_chunks.append(np.asarray(combined_weights, dtype=np.float64))

    if total_weight <= 0.0 or len(raw_angle_chunks) <= 0:
        return None, None

    radial_value = float(radial_numerator / total_weight)
    raw_phi_value = _weighted_wrapped_angle_mean_deg(
        np.concatenate(raw_angle_chunks),
        np.concatenate(raw_weight_chunks),
    )
    gui_phi_value = float(raw_phi_to_gui_phi(raw_phi_value))
    if not (np.isfinite(radial_value) and np.isfinite(gui_phi_value)):
        return None, None
    return radial_value, gui_phi_value

def _parse_poni_file_simple(poni_path: str | Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in Path(poni_path).expanduser().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            values[key] = float(value)
        except ValueError:
            continue
    return values


def build_geometry(
    *,
    pixel_size_m: float,
    distance_m: float | None = None,
    center_row_px: float | None = None,
    center_col_px: float | None = None,
    poni1_m: float | None = None,
    poni2_m: float | None = None,
    poni_path: str | Path | None = None,
) -> PortableGeometry:
    """Build portable detector geometry from either center pixels or PONI values.

    Zero-rotation, flat-detector convention:
    - ``center_row_px = poni1_m / pixel_size_m``
    - ``center_col_px = poni2_m / pixel_size_m``
    """

    pixel_size = float(pixel_size_m)
    if pixel_size <= 0.0:
        raise ValueError("pixel_size_m must be > 0.")
    parsed_poni: dict[str, float] = {}
    if poni_path is not None:
        parsed_poni = _parse_poni_file_simple(poni_path)
    dist = distance_m if distance_m is not None else parsed_poni.get("Dist")
    if dist is None:
        raise ValueError("distance_m or poni_path with Dist is required.")
    row_px = center_row_px
    col_px = center_col_px
    if row_px is None:
        if poni1_m is None:
            poni1_m = parsed_poni.get("Poni1")
        if poni1_m is None:
            raise ValueError("Provide center_row_px or poni1_m/poni_path.")
        row_px = float(poni1_m) / pixel_size
    if col_px is None:
        if poni2_m is None:
            poni2_m = parsed_poni.get("Poni2")
        if poni2_m is None:
            raise ValueError("Provide center_col_px or poni2_m/poni_path.")
        col_px = float(poni2_m) / pixel_size
    return PortableGeometry(
        pixel_size_m=pixel_size,
        distance_m=float(dist),
        center_row_px=float(row_px),
        center_col_px=float(col_px),
    )


def build_angle_axes(
    *,
    npt_rad: int = 1000,
    npt_azim: int = 720,
    tth_min_deg: float = 0.0,
    tth_max_deg: float = 90.0,
    azimuth_min_deg: float = -180.0,
    azimuth_max_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniformly spaced 2theta and canonical raw-azimuth bin centers."""

    radial_edges = np.linspace(float(tth_min_deg), float(tth_max_deg), int(max(2, npt_rad)) + 1, dtype=np.float64)
    azimuth_edges = np.linspace(
        float(azimuth_min_deg),
        float(azimuth_max_deg),
        int(max(2, npt_azim)) + 1,
        dtype=np.float64,
    )
    radial_deg = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    azimuthal_deg = 0.5 * (azimuth_edges[:-1] + azimuth_edges[1:])
    return radial_deg, azimuthal_deg


def flat_solid_angle_normalization(
    image_shape: tuple[int, int],
    geometry: PortableGeometry,
) -> np.ndarray:
    """Return flat-detector solid-angle normalization for the given geometry."""

    row_coords = ((np.arange(int(image_shape[0]), dtype=np.float64) + 0.5) - float(geometry.center_row_px)) * float(geometry.pixel_size_m)
    col_coords = ((np.arange(int(image_shape[1]), dtype=np.float64) + 0.5) - float(geometry.center_col_px)) * float(geometry.pixel_size_m)
    yy = row_coords[:, None]
    xx = col_coords[None, :]
    path = np.sqrt(xx * xx + yy * yy + float(geometry.distance_m) ** 2)
    return np.asarray((float(geometry.distance_m) / path) ** 3, dtype=np.float32)


def detector_pixel_angular_maps(
    image_shape: tuple[int, int],
    geometry: PortableGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel flat-detector ``(2theta_deg, raw_azimuth_deg)`` maps."""

    row_coords = (
        (np.arange(int(image_shape[0]), dtype=np.float64) + 0.5) - float(geometry.center_row_px)
    ) * float(geometry.pixel_size_m)
    col_coords = (
        (np.arange(int(image_shape[1]), dtype=np.float64) + 0.5) - float(geometry.center_col_px)
    ) * float(geometry.pixel_size_m)
    yy = row_coords[:, None]
    xx = col_coords[None, :]
    two_theta_deg = np.degrees(np.arctan2(np.hypot(xx, yy), float(geometry.distance_m)))
    raw_azimuth_deg = np.degrees(np.arctan2(yy, xx))
    return np.asarray(two_theta_deg, dtype=np.float64), np.asarray(raw_azimuth_deg, dtype=np.float64)


def detector_points_to_angles(
    cols: np.ndarray | list[float] | tuple[float, ...],
    rows: np.ndarray | list[float] | tuple[float, ...],
    geometry: PortableGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat-detector GUI-style ``(2theta_deg, phi_deg)`` for detector points."""

    cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
    rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
    two_theta = np.full(cols_arr.shape, np.nan, dtype=np.float64)
    phi = np.full(cols_arr.shape, np.nan, dtype=np.float64)

    if cols_arr.shape != rows_arr.shape:
        return two_theta, phi
    if not np.isfinite(float(geometry.distance_m)) or float(geometry.distance_m) <= 0.0:
        return two_theta, phi
    if not np.isfinite(float(geometry.pixel_size_m)) or float(geometry.pixel_size_m) <= 0.0:
        return two_theta, phi

    center_row = float(geometry.center_row_px)
    center_col = float(geometry.center_col_px)
    if not np.isfinite(center_row) or not np.isfinite(center_col):
        return two_theta, phi

    valid_mask = np.isfinite(cols_arr) & np.isfinite(rows_arr)
    if not np.any(valid_mask):
        return two_theta, phi

    dx = (cols_arr[valid_mask] - center_col) * float(geometry.pixel_size_m)
    raw_dy = (rows_arr[valid_mask] - center_row) * float(geometry.pixel_size_m)
    radius = np.hypot(dx, raw_dy)
    two_theta_valid = np.degrees(np.arctan2(radius, float(geometry.distance_m)))
    phi_valid = np.asarray(
        raw_phi_to_gui_phi(np.degrees(np.arctan2(raw_dy, dx))),
        dtype=np.float64,
    )

    two_theta[valid_mask] = two_theta_valid
    phi[valid_mask] = phi_valid
    return two_theta, phi


def caked_point_to_detector_pixel(
    ai: FastAzimuthalIntegrator | None,
    detector_shape: tuple[int, ...] | list[int],
    radial_deg: np.ndarray | list[float] | tuple[float, ...] | None,
    gui_phi_deg: np.ndarray | list[float] | tuple[float, ...] | None,
    two_theta_deg: float,
    phi_deg: float,
    *,
    transform_bundle: CakeTransformBundle | None = None,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> tuple[float | None, float | None]:
    """Invert one displayed caked bin back to a detector pixel via the exact-cake LUT."""

    if not isinstance(ai, FastAzimuthalIntegrator):
        return None, None
    normalized_shape = _normalize_detector_shape(tuple(detector_shape))
    if normalized_shape is None:
        return None, None

    if not (np.isfinite(two_theta_deg) and np.isfinite(phi_deg)):
        return None, None

    bundle = transform_bundle
    if not isinstance(bundle, CakeTransformBundle):
        live_bundle = getattr(ai, "_live_caked_transform_bundle", None)
        if isinstance(live_bundle, CakeTransformBundle) and live_bundle.detector_shape == normalized_shape:
            bundle = live_bundle
    if not isinstance(bundle, CakeTransformBundle):
        try:
            radial_axis = np.asarray(radial_deg, dtype=np.float64).reshape(-1)
            gui_phi_axis = np.asarray(gui_phi_deg, dtype=np.float64).reshape(-1)
        except Exception:
            return None, None
        if radial_axis.size <= 0 or gui_phi_axis.size <= 0:
            return None, None
        if not np.all(np.isfinite(radial_axis)) or not np.all(np.isfinite(gui_phi_axis)):
            return None, None
        raw_azimuth_axis = np.asarray(gui_phi_to_raw_phi(gui_phi_axis), dtype=np.float64).reshape(-1)
        raw_azimuth_axis = np.sort(raw_azimuth_axis)
        bundle = build_cake_transform_bundle(
            ai,
            normalized_shape,
            radial_axis,
            raw_azimuth_axis,
            engine=engine,
            workers=workers,
        )
    if not isinstance(bundle, CakeTransformBundle):
        return None, None
    radial_axis = bundle.radial_deg
    raw_azimuth_axis = bundle.raw_azimuth_deg
    lut = bundle.lut

    try:
        radial_idx = int(np.argmin(np.abs(radial_axis - float(two_theta_deg))))
        raw_phi_deg = float(gui_phi_to_raw_phi(float(phi_deg)))
        azimuth_idx = int(np.argmin(np.abs(raw_azimuth_axis - raw_phi_deg)))
    except Exception:
        return None, None
    if radial_idx < 0 or azimuth_idx < 0:
        return None, None

    flat_idx = int(azimuth_idx * int(radial_axis.size) + radial_idx)
    matrix = getattr(lut, "matrix", None)
    if matrix is None:
        return None, None

    row_payload = _matrix_row_indices_and_weights(matrix, flat_idx)
    if row_payload is None:
        return None, None
    pixel_indices, weights = row_payload

    valid = (
        np.isfinite(pixel_indices)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(valid):
        return None, None

    pixel_indices = np.asarray(pixel_indices[valid], dtype=np.int64)
    weights = np.asarray(weights[valid], dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return None, None

    width = int(normalized_shape[1])
    cols = np.asarray(pixel_indices % width, dtype=np.float64)
    rows = np.asarray(pixel_indices // width, dtype=np.float64)
    return (
        float(np.sum(cols * weights) / weight_sum),
        float(np.sum(rows * weights) / weight_sum),
    )


def detector_two_theta_max_deg(
    image_shape: tuple[int, int],
    geometry: PortableGeometry,
) -> float:
    """Return the largest detector-corner 2theta spanned by the flat detector."""

    height = int(image_shape[0])
    width = int(image_shape[1])
    row_edges = (
        np.arange(height + 1, dtype=np.float64) - float(geometry.center_row_px)
    ) * float(geometry.pixel_size_m)
    col_edges = (
        np.arange(width + 1, dtype=np.float64) - float(geometry.center_col_px)
    ) * float(geometry.pixel_size_m)
    max_two_theta = 0.0
    for row_edge in (float(row_edges[0]), float(row_edges[-1])):
        for col_edge in (float(col_edges[0]), float(col_edges[-1])):
            two_theta_deg = math.degrees(
                math.atan2(
                    math.hypot(float(col_edge), float(row_edge)),
                    float(geometry.distance_m),
                )
            )
            if two_theta_deg > max_two_theta:
                max_two_theta = float(two_theta_deg)
    return float(max_two_theta)


def _normalize_two_theta_unit(unit: str | None) -> str:
    if unit is None:
        return "rad"
    text = str(unit).strip().lower()
    if text in {"2th_deg", "deg", "degree", "degrees"}:
        return "deg"
    if text in {"2th_rad", "rad", "radian", "radians"}:
        return "rad"
    raise ValueError("Unsupported two-theta unit.")


def _normalize_angle_unit(unit: str | None) -> str:
    if unit is None:
        return "rad"
    text = str(unit).strip().lower()
    if text in {"deg", "degree", "degrees"}:
        return "deg"
    if text in {"rad", "radian", "radians"}:
        return "rad"
    raise ValueError("Unsupported angle unit.")


class FastAzimuthalIntegrator:
    """Small flat-detector integrator wrapper for the GUI exact-cake path."""

    def __init__(
        self,
        *,
        dist: float,
        poni1: float,
        poni2: float,
        pixel1: float,
        pixel2: float,
        rot1: float = 0.0,
        rot2: float = 0.0,
        rot3: float = 0.0,
        wavelength: float | None = None,
    ) -> None:
        del rot1, rot2, rot3, wavelength
        pixel_row = float(pixel1)
        pixel_col = float(pixel2)
        if not np.isfinite(pixel_row) or pixel_row <= 0.0:
            raise ValueError("pixel1 must be > 0.")
        if not np.isfinite(pixel_col) or pixel_col <= 0.0:
            raise ValueError("pixel2 must be > 0.")
        if not np.isclose(pixel_row, pixel_col, atol=1.0e-15, rtol=0.0):
            raise ValueError("FastAzimuthalIntegrator currently requires square pixels.")
        self.geometry = PortableGeometry(
            pixel_size_m=float(pixel_row),
            distance_m=float(dist),
            center_row_px=float(poni1) / float(pixel_row),
            center_col_px=float(poni2) / float(pixel_col),
        )
        self._solid_angle_cache: dict[tuple[int, int], np.ndarray] = {}
        self._solid_angle_cache_lock = threading.RLock()

    def twoThetaArray(self, *, shape: tuple[int, int], unit: str | None = None) -> np.ndarray:
        two_theta_deg, _raw_azimuth_deg = self._detector_maps_for_shape(shape)
        if _normalize_two_theta_unit(unit) == "deg":
            return two_theta_deg
        return np.deg2rad(two_theta_deg)

    def chiArray(self, *, shape: tuple[int, int], unit: str | None = None) -> np.ndarray:
        _two_theta_deg, raw_azimuth_deg = self._detector_maps_for_shape(shape)
        if _normalize_angle_unit(unit) == "deg":
            return raw_azimuth_deg
        return np.deg2rad(raw_azimuth_deg)

    def integrate2d(
        self,
        image: np.ndarray,
        *,
        npt_rad: int = 1000,
        npt_azim: int = 720,
        correctSolidAngle: bool = False,
        method: str | None = None,
        unit: str = "2th_deg",
        normalization: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        rows: np.ndarray | None = None,
        cols: np.ndarray | None = None,
        engine: str = "auto",
        workers: int | str | None = "auto",
    ) -> DetectorCakeResult:
        if _normalize_two_theta_unit(unit) != "deg":
            raise ValueError("FastAzimuthalIntegrator.integrate2d only supports degree output.")

        image_arr = np.asarray(image)
        radial_deg, azimuthal_deg = build_angle_axes(
            npt_rad=int(max(1, npt_rad)),
            npt_azim=int(max(1, npt_azim)),
            tth_min_deg=0.0,
            tth_max_deg=detector_two_theta_max_deg(image_arr.shape, self.geometry),
            azimuth_min_deg=-180.0,
            azimuth_max_deg=180.0,
        )
        norm = normalization
        if norm is None and bool(correctSolidAngle):
            norm = self._solid_angle_for_shape(image_arr.shape)
        exact_geometry = DetectorCakeGeometry(
            pixel_size_m=float(self.geometry.pixel_size_m),
            distance_m=float(self.geometry.distance_m),
            center_row_px=float(self.geometry.center_row_px),
            center_col_px=float(self.geometry.center_col_px),
        )
        method_name = "lut" if method is None else str(method).strip().lower()
        if method_name == "lut":
            lut = self._cached_cake_lut(
                image_arr.shape,
                radial_deg,
                azimuthal_deg,
                exact_geometry,
                engine=engine,
                workers=workers,
                strict=True,
            )
            if lut is None:
                raise RuntimeError("Exact-cake LUT unavailable for requested detector/cake transform.")
            return _integrate_detector_to_cake_lut_with_selection(
                image_arr,
                radial_deg,
                azimuthal_deg,
                lut,
                normalization=norm,
                mask=mask,
                rows=rows,
                cols=cols,
            )
        return integrate_detector_to_cake_exact(
            image_arr,
            radial_deg,
            azimuthal_deg,
            exact_geometry,
            normalization=norm,
            mask=mask,
            rows=rows,
            cols=cols,
            engine=engine,
            workers=workers,
        )

    def _detector_maps_for_shape(
        self,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        detector_shape = _normalize_detector_shape(tuple(shape))
        if detector_shape is None:
            raise ValueError("shape must have two positive detector dimensions.")
        return _shared_detector_maps_for_shape(detector_shape, self.geometry)

    def _solid_angle_for_shape(self, shape: tuple[int, ...]) -> np.ndarray:
        detector_shape = _normalize_detector_shape(tuple(shape))
        if detector_shape is None:
            raise ValueError("shape must have two positive detector dimensions.")
        with self._solid_angle_cache_lock:
            cached = self._solid_angle_cache.get(detector_shape)
            if cached is None:
                cached = np.asarray(
                    flat_solid_angle_normalization(detector_shape, self.geometry),
                    dtype=np.float32,
                )
                cached.setflags(write=False)
                self._solid_angle_cache[detector_shape] = cached
        return cached

    def warm_geometry_cache(
        self,
        detector_shape: tuple[int, ...] | list[int],
        *,
        npt_rad: int = 1000,
        npt_azim: int = 720,
        engine: str = "auto",
        workers: int | str | None = "auto",
    ) -> None:
        normalized_shape = _normalize_detector_shape(tuple(detector_shape))
        if normalized_shape is None:
            raise ValueError("detector_shape must have two positive dimensions.")
        self._detector_maps_for_shape(normalized_shape)
        self._solid_angle_for_shape(normalized_shape)
        radial_deg, azimuthal_deg = build_angle_axes(
            npt_rad=int(max(1, npt_rad)),
            npt_azim=int(max(1, npt_azim)),
            tth_min_deg=0.0,
            tth_max_deg=detector_two_theta_max_deg(normalized_shape, self.geometry),
            azimuth_min_deg=-180.0,
            azimuth_max_deg=180.0,
        )
        self._cached_cake_lut(
            normalized_shape,
            radial_deg,
            azimuthal_deg,
            DetectorCakeGeometry(
                pixel_size_m=float(self.geometry.pixel_size_m),
                distance_m=float(self.geometry.distance_m),
                center_row_px=float(self.geometry.center_row_px),
                center_col_px=float(self.geometry.center_col_px),
            ),
            engine=engine,
            workers=workers,
        )

    def _cached_cake_lut(
        self,
        image_shape: tuple[int, ...],
        radial_deg: np.ndarray,
        azimuthal_deg: np.ndarray,
        geometry: DetectorCakeGeometry,
        *,
        engine: str,
        workers: int | str | None,
        strict: bool = False,
    ) -> DetectorCakeLUT | None:
        _normalize_lut_engine(engine)
        detector_shape = tuple(int(v) for v in tuple(image_shape)[:2])
        return _shared_cake_lut_for_request(
            detector_shape,
            radial_deg,
            azimuthal_deg,
            geometry,
            self.geometry,
            workers=workers,
            strict=strict,
        )


def start_exact_cake_geometry_warmup_in_background(
    ai: FastAzimuthalIntegrator | None,
    detector_shape: tuple[int, ...] | list[int],
    *,
    npt_rad: int = 1000,
    npt_azim: int = 720,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> bool:
    """Warm detector maps and the default LUT for one live integrator in a daemon thread."""

    if not isinstance(ai, FastAzimuthalIntegrator):
        return False
    normalized_shape = _normalize_detector_shape(tuple(detector_shape))
    if normalized_shape is None:
        return False
    warm_key = (
        int(id(ai)),
        normalized_shape,
        int(max(1, npt_rad)),
        int(max(1, npt_azim)),
    )

    with _EXACT_CAKE_GEOMETRY_WARMUP_LOCK:
        if warm_key in _EXACT_CAKE_GEOMETRY_WARMUP_COMPLETED:
            return False
        existing = _EXACT_CAKE_GEOMETRY_WARMUP_THREADS.get(warm_key)
        if existing is not None and existing.is_alive():
            return False

        def _run() -> None:
            try:
                ai.warm_geometry_cache(
                    normalized_shape,
                    npt_rad=int(max(1, npt_rad)),
                    npt_azim=int(max(1, npt_azim)),
                    engine=engine,
                    workers=workers,
                )
            finally:
                with _EXACT_CAKE_GEOMETRY_WARMUP_LOCK:
                    _EXACT_CAKE_GEOMETRY_WARMUP_THREADS.pop(warm_key, None)
                    _EXACT_CAKE_GEOMETRY_WARMUP_COMPLETED[warm_key] = None
                    _EXACT_CAKE_GEOMETRY_WARMUP_COMPLETED.move_to_end(warm_key)
                    while (
                        len(_EXACT_CAKE_GEOMETRY_WARMUP_COMPLETED)
                        > _GEOMETRY_WARMUP_COMPLETION_LIMIT
                    ):
                        _EXACT_CAKE_GEOMETRY_WARMUP_COMPLETED.popitem(last=False)

        thread = threading.Thread(
            target=_run,
            name="exact-cake-geometry-warmup",
            daemon=True,
        )
        _EXACT_CAKE_GEOMETRY_WARMUP_THREADS[warm_key] = thread
        thread.start()
        return True


def convert_image_to_angle_space(
    image: np.ndarray,
    *,
    pixel_size_m: float,
    distance_m: float | None = None,
    center_row_px: float | None = None,
    center_col_px: float | None = None,
    poni1_m: float | None = None,
    poni2_m: float | None = None,
    poni_path: str | Path | None = None,
    npt_rad: int = 1000,
    npt_azim: int = 720,
    tth_min_deg: float = 0.0,
    tth_max_deg: float = 90.0,
    azimuth_min_deg: float = -180.0,
    azimuth_max_deg: float = 180.0,
    normalization: np.ndarray | None = None,
    correct_solid_angle: bool = False,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> DetectorCakeResult:
    """Convert one detector image directly into angle space.

    Inputs needed:
    - image
    - pixel size
    - distance
    - either center pixels or PONI values
    - optional ``rows``/``cols`` detector coordinates to limit integration
    """

    portable_geometry = build_geometry(
        pixel_size_m=pixel_size_m,
        distance_m=distance_m,
        center_row_px=center_row_px,
        center_col_px=center_col_px,
        poni1_m=poni1_m,
        poni2_m=poni2_m,
        poni_path=poni_path,
    )
    radial_deg, azimuthal_deg = build_angle_axes(
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        tth_min_deg=tth_min_deg,
        tth_max_deg=tth_max_deg,
        azimuth_min_deg=azimuth_min_deg,
        azimuth_max_deg=azimuth_max_deg,
    )
    norm = normalization
    if norm is None and correct_solid_angle:
        norm = flat_solid_angle_normalization(np.asarray(image).shape, portable_geometry)
    _normalize_lut_engine(engine)
    detector_shape = _normalize_detector_shape(tuple(np.asarray(image).shape))
    if detector_shape is None:
        raise ValueError("image must have two positive detector dimensions.")
    exact_geometry = DetectorCakeGeometry(
        pixel_size_m=float(portable_geometry.pixel_size_m),
        distance_m=float(portable_geometry.distance_m),
        center_row_px=float(portable_geometry.center_row_px),
        center_col_px=float(portable_geometry.center_col_px),
    )
    lut = _shared_cake_lut_for_request(
        detector_shape,
        radial_deg,
        azimuthal_deg,
        exact_geometry,
        portable_geometry,
        workers=workers,
        strict=True,
    )
    if lut is None:
        raise RuntimeError("Exact-cake LUT unavailable for requested detector/cake transform.")
    return _integrate_detector_to_cake_lut_with_selection(
        image,
        radial_deg,
        azimuthal_deg,
        lut,
        normalization=norm,
        mask=None,
        rows=rows,
        cols=cols,
    )


def prepare_gui_phi_display(
    result: DetectorCakeResult,
    *,
    phi_min_deg: float = -180.0,
    phi_max_deg: float = 180.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cake_image, radial_deg, gui_phi_deg)`` for GUI-style plotting."""

    gui_phi = np.asarray(raw_phi_to_gui_phi(result.azimuthal_deg), dtype=np.float64)
    order = np.argsort(gui_phi)
    gui_phi = gui_phi[order]
    cake = np.asarray(result.intensity, dtype=np.float64)[order, :]
    if float(phi_min_deg) <= float(phi_max_deg):
        mask = (gui_phi >= float(phi_min_deg)) & (gui_phi <= float(phi_max_deg))
    else:
        mask = (gui_phi >= float(phi_min_deg)) | (gui_phi <= float(phi_max_deg))
    return cake[mask, :], np.asarray(result.radial_deg, dtype=np.float64), gui_phi[mask]


__all__ = [
    "CakeTransformBundle",
    "FastAzimuthalIntegrator",
    "PortableGeometry",
    "build_angle_axes",
    "build_cake_transform_bundle",
    "build_cake_transform_bundle_from_result",
    "cake_transform_bundle_lut_t",
    "caked_point_to_detector_pixel",
    "build_geometry",
    "convert_image_to_angle_space",
    "detector_pixel_to_caked_bin",
    "detector_points_to_angles",
    "detector_pixel_angular_maps",
    "detector_two_theta_max_deg",
    "flat_solid_angle_normalization",
    "gui_phi_to_raw_phi",
    "prepare_gui_phi_display",
    "raw_phi_to_gui_phi",
]
