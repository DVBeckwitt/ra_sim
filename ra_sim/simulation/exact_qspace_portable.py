"""Portable direct detector-to-sample-frame Q-space remap helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import threading

import numpy as np

from ra_sim.simulation.exact_cake_portable import flat_solid_angle_normalization
from ra_sim.simulation.exact_qspace import (
    DetectorQSpaceGeometry,
    DetectorQSpaceLUT,
    DetectorQSpaceResult,
    _detector_corner_sample_q_maps,
    build_detector_to_qspace_lut,
    build_qspace_axes,
    detector_q_extents,
    integrate_detector_to_qspace_exact,
    integrate_detector_to_qspace_lut,
)


_PROCESS_DETECTOR_MAP_CACHE_LIMIT = 2
_PROCESS_QSPACE_LUT_CACHE_LIMIT = 2


_Float64ArrayCacheToken = tuple[tuple[int, ...], bytes]
_SharedDetectorMapKey = tuple["PortableQSpaceGeometry", tuple[int, int]]
_SharedQSpaceLutKey = tuple[
    "PortableQSpaceGeometry",
    tuple[int, int],
    _Float64ArrayCacheToken,
    _Float64ArrayCacheToken,
]


_PROCESS_DETECTOR_MAP_CACHE: OrderedDict[
    _SharedDetectorMapKey,
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = OrderedDict()
_PROCESS_QSPACE_LUT_CACHE: OrderedDict[_SharedQSpaceLutKey, DetectorQSpaceLUT] = OrderedDict()
_PROCESS_QSPACE_LUT_IN_FLIGHT: dict[_SharedQSpaceLutKey, "_SharedQSpaceLutInFlightState"] = {}
_PROCESS_DETECTOR_MAP_CACHE_LOCK = threading.RLock()
_PROCESS_QSPACE_LUT_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True)
class PortableQSpaceGeometry:
    pixel_size_m: float
    distance_m: float
    center_row_px: float
    center_col_px: float
    wavelength_m: float
    gamma_deg: float
    Gamma_deg: float
    chi_deg: float
    psi_deg: float
    psi_z_deg: float
    theta_initial_deg: float
    cor_angle_deg: float
    zs: float
    zb: float


@dataclass
class _SharedQSpaceLutInFlightState:
    event: threading.Event = field(default_factory=threading.Event)
    result: DetectorQSpaceLUT | None = None
    error: Exception | None = None


def _clear_shared_exact_qspace_caches() -> None:
    """Clear process-level direct-q detector-map and LUT caches."""

    with _PROCESS_DETECTOR_MAP_CACHE_LOCK:
        _PROCESS_DETECTOR_MAP_CACHE.clear()
    with _PROCESS_QSPACE_LUT_CACHE_LOCK:
        _PROCESS_QSPACE_LUT_CACHE.clear()
        _PROCESS_QSPACE_LUT_IN_FLIGHT.clear()


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


def _float64_array_cache_token(values: np.ndarray) -> _Float64ArrayCacheToken:
    contiguous = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.blake2b(contiguous.tobytes(order="C"), digest_size=16).digest()
    return tuple(int(v) for v in contiguous.shape), digest


def _shared_detector_map_cache_key(
    geometry: PortableQSpaceGeometry,
    detector_shape: tuple[int, int],
) -> _SharedDetectorMapKey:
    return geometry, detector_shape


def _shared_qspace_lut_cache_key(
    geometry: PortableQSpaceGeometry,
    detector_shape: tuple[int, int],
    qr: np.ndarray,
    qz: np.ndarray,
) -> _SharedQSpaceLutKey:
    return (
        geometry,
        detector_shape,
        _float64_array_cache_token(qr),
        _float64_array_cache_token(qz),
    )


def _geometry_to_exact(geometry: PortableQSpaceGeometry) -> DetectorQSpaceGeometry:
    return DetectorQSpaceGeometry(
        pixel_size_m=float(geometry.pixel_size_m),
        distance_m=float(geometry.distance_m),
        center_row_px=float(geometry.center_row_px),
        center_col_px=float(geometry.center_col_px),
        wavelength_m=float(geometry.wavelength_m),
        gamma_deg=float(geometry.gamma_deg),
        Gamma_deg=float(geometry.Gamma_deg),
        chi_deg=float(geometry.chi_deg),
        psi_deg=float(geometry.psi_deg),
        psi_z_deg=float(geometry.psi_z_deg),
        theta_initial_deg=float(geometry.theta_initial_deg),
        cor_angle_deg=float(geometry.cor_angle_deg),
        zs=float(geometry.zs),
        zb=float(geometry.zb),
    )


def _shared_detector_maps_for_shape(
    detector_shape: tuple[int, int],
    geometry: PortableQSpaceGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = _shared_detector_map_cache_key(geometry, detector_shape)
    with _PROCESS_DETECTOR_MAP_CACHE_LOCK:
        cached = _PROCESS_DETECTOR_MAP_CACHE.get(key)
        if cached is not None:
            _PROCESS_DETECTOR_MAP_CACHE.move_to_end(key)
            return cached

    qx_map, qy_map, qz_map = _detector_corner_sample_q_maps(
        detector_shape,
        _geometry_to_exact(geometry),
    )
    qx_map = np.asarray(qx_map, dtype=np.float64)
    qz_map = np.asarray(qz_map, dtype=np.float64)
    qy_map = np.asarray(qy_map, dtype=np.float64)
    qx_map.setflags(write=False)
    qz_map.setflags(write=False)
    qy_map.setflags(write=False)
    built = (qx_map, qz_map, qy_map)

    with _PROCESS_DETECTOR_MAP_CACHE_LOCK:
        cached = _PROCESS_DETECTOR_MAP_CACHE.get(key)
        if cached is not None:
            _PROCESS_DETECTOR_MAP_CACHE.move_to_end(key)
            return cached
        _PROCESS_DETECTOR_MAP_CACHE[key] = built
        while len(_PROCESS_DETECTOR_MAP_CACHE) > _PROCESS_DETECTOR_MAP_CACHE_LIMIT:
            _PROCESS_DETECTOR_MAP_CACHE.popitem(last=False)
    return built


def _shared_qspace_lut_for_request(
    detector_shape: tuple[int, int],
    qr: np.ndarray,
    qz: np.ndarray,
    geometry: DetectorQSpaceGeometry,
    portable_geometry: PortableQSpaceGeometry,
    *,
    qx_map: np.ndarray,
    qz_map: np.ndarray,
    qy_map: np.ndarray,
    strict: bool = False,
) -> DetectorQSpaceLUT | None:
    key = _shared_qspace_lut_cache_key(portable_geometry, detector_shape, qr, qz)
    state: _SharedQSpaceLutInFlightState | None = None
    is_builder = False
    with _PROCESS_QSPACE_LUT_CACHE_LOCK:
        cached = _PROCESS_QSPACE_LUT_CACHE.get(key)
        if cached is not None:
            _PROCESS_QSPACE_LUT_CACHE.move_to_end(key)
            return cached
        state = _PROCESS_QSPACE_LUT_IN_FLIGHT.get(key)
        if state is None:
            state = _SharedQSpaceLutInFlightState()
            _PROCESS_QSPACE_LUT_IN_FLIGHT[key] = state
            is_builder = True

    assert state is not None
    if not is_builder:
        state.event.wait()
        if state.result is not None:
            return state.result
        error = state.error
        if error is None:
            error = RuntimeError("Exact-q-space LUT build finished without a result or error.")
        if strict:
            raise error
        return None

    try:
        built = build_detector_to_qspace_lut(
            detector_shape,
            qr,
            qz,
            geometry,
            qx_map=qx_map,
            qz_map=qz_map,
            qy_map=qy_map,
        )
        if built is None:
            raise RuntimeError("Exact-q-space LUT build returned no lookup table.")
    except Exception as exc:
        with _PROCESS_QSPACE_LUT_CACHE_LOCK:
            _PROCESS_QSPACE_LUT_IN_FLIGHT.pop(key, None)
            state.error = exc
            state.event.set()
        if strict:
            raise
        return None

    with _PROCESS_QSPACE_LUT_CACHE_LOCK:
        cached = _PROCESS_QSPACE_LUT_CACHE.get(key)
        if cached is not None:
            _PROCESS_QSPACE_LUT_CACHE.move_to_end(key)
            result = cached
        else:
            _PROCESS_QSPACE_LUT_CACHE[key] = built
            while len(_PROCESS_QSPACE_LUT_CACHE) > _PROCESS_QSPACE_LUT_CACHE_LIMIT:
                _PROCESS_QSPACE_LUT_CACHE.popitem(last=False)
            result = built
        _PROCESS_QSPACE_LUT_IN_FLIGHT.pop(key, None)
        state.result = result
        state.event.set()
    return result


def _normalize_method(method: str | None) -> str:
    method_name = "lut" if method is None else str(method).strip().lower()
    if method_name in {"", "auto"}:
        return "lut"
    if method_name not in {"lut", "exact"}:
        raise ValueError("method must be one of: lut, exact.")
    return method_name


def _qspace_axes_for_shape(
    detector_shape: tuple[int, int],
    geometry: PortableQSpaceGeometry,
    *,
    npt_rad: int,
    npt_azim: int,
    qx_map: np.ndarray,
    qz_map: np.ndarray,
    qy_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qr_max, qz_min, qz_max = detector_q_extents(
        detector_shape,
        _geometry_to_exact(geometry),
        qx_map=qx_map,
        qz_map=qz_map,
        qy_map=qy_map,
    )
    qr, qz = build_qspace_axes(
        npt_rad=int(max(2, npt_rad)),
        npt_azim=int(max(2, npt_azim)),
        qr_max=qr_max,
        qz_min=qz_min,
        qz_max=qz_max,
    )
    return _readonly_float64_vector(qr), _readonly_float64_vector(qz)


def convert_image_to_q_space(
    image: np.ndarray,
    *,
    pixel_size_m: float,
    distance_m: float,
    center_row_px: float,
    center_col_px: float,
    wavelength_m: float,
    gamma_deg: float,
    Gamma_deg: float,
    chi_deg: float,
    psi_deg: float,
    psi_z_deg: float,
    theta_initial_deg: float,
    cor_angle_deg: float,
    zs: float,
    zb: float,
    npt_rad: int = 1000,
    npt_azim: int = 720,
    normalization: np.ndarray | None = None,
    correct_solid_angle: bool = False,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    method: str | None = None,
    engine: str = "auto",
    workers: int | str | None = "auto",
) -> DetectorQSpaceResult:
    """Convert one detector image directly into sample-frame ``(Qr, Qz)`` space."""

    del engine, workers
    image_arr = np.asarray(image)
    detector_shape = _normalize_detector_shape(tuple(image_arr.shape))
    if detector_shape is None:
        raise ValueError("image must have two positive detector dimensions.")

    portable_geometry = PortableQSpaceGeometry(
        pixel_size_m=float(pixel_size_m),
        distance_m=float(distance_m),
        center_row_px=float(center_row_px),
        center_col_px=float(center_col_px),
        wavelength_m=float(wavelength_m),
        gamma_deg=float(gamma_deg),
        Gamma_deg=float(Gamma_deg),
        chi_deg=float(chi_deg),
        psi_deg=float(psi_deg),
        psi_z_deg=float(psi_z_deg),
        theta_initial_deg=float(theta_initial_deg),
        cor_angle_deg=float(cor_angle_deg),
        zs=float(zs),
        zb=float(zb),
    )
    qx_map, qz_map, qy_map = _shared_detector_maps_for_shape(detector_shape, portable_geometry)
    qr, qz = _qspace_axes_for_shape(
        detector_shape,
        portable_geometry,
        npt_rad=npt_rad,
        npt_azim=npt_azim,
        qx_map=qx_map,
        qz_map=qz_map,
        qy_map=qy_map,
    )
    norm = normalization
    if norm is None and bool(correct_solid_angle):
        norm = flat_solid_angle_normalization(detector_shape, portable_geometry)

    exact_geometry = _geometry_to_exact(portable_geometry)
    method_name = _normalize_method(method)
    use_exact = rows is not None or cols is not None or method_name == "exact"
    if use_exact:
        return integrate_detector_to_qspace_exact(
            image_arr,
            qr,
            qz,
            exact_geometry,
            normalization=norm,
            mask=mask,
            rows=rows,
            cols=cols,
            qx_map=qx_map,
            qz_map=qz_map,
            qy_map=qy_map,
        )

    lut = _shared_qspace_lut_for_request(
        detector_shape,
        qr,
        qz,
        exact_geometry,
        portable_geometry,
        qx_map=qx_map,
        qz_map=qz_map,
        qy_map=qy_map,
        strict=True,
    )
    if lut is None:
        raise RuntimeError("Exact-q-space LUT unavailable for requested detector/Q transform.")
    return integrate_detector_to_qspace_lut(
        image_arr,
        qr,
        qz,
        lut,
        normalization=norm,
        mask=mask,
    )


__all__ = [
    "PortableQSpaceGeometry",
    "_clear_shared_exact_qspace_caches",
    "convert_image_to_q_space",
]
