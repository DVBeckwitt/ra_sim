"""Direct detector-to-sample-frame Q-space remap with exact pixel splitting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .exact_cake import (
    _HAS_SCIPY,
    _integrate_edge,
    _inverse_calc_upper_bound,
    _prepare_selection,
    _scipy_sparse,
)
from .intersection_analysis import build_nominal_projection_frame


_SEAM_EPSILON = 1.0e-12


@dataclass(frozen=True)
class DetectorQSpaceGeometry:
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


@dataclass(frozen=True)
class DetectorQSpaceResult:
    qr: np.ndarray
    qz: np.ndarray
    intensity: np.ndarray
    sum_signal: np.ndarray
    sum_normalization: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class DetectorQSpaceLUT:
    image_shape: tuple[int, int]
    n_qr: int
    n_qz: int
    qr: np.ndarray
    qz: np.ndarray
    signed_qr: np.ndarray
    matrix: object
    count_flat: np.ndarray


def _validate_public_axes(qr: np.ndarray, qz: np.ndarray) -> None:
    if qr.ndim != 1 or qz.ndim != 1:
        raise ValueError("qr and qz must be 1D arrays.")
    if qr.size < 2 or qz.size < 2:
        raise ValueError("qr and qz need at least 2 bins each.")
    if not np.all(np.isfinite(qr)) or not np.all(np.isfinite(qz)):
        raise ValueError("qr and qz must be finite.")
    if np.any(qr <= 0.0):
        raise ValueError("qr bin centers must be positive.")
    qr_step = np.diff(qr)
    qz_step = np.diff(qz)
    if not np.all(qr_step > 0.0):
        raise ValueError("qr must be strictly increasing.")
    if not np.all(qz_step > 0.0):
        raise ValueError("qz must be strictly increasing.")
    if not np.allclose(qr_step, qr_step[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("qr must be uniformly spaced.")
    if not np.allclose(qz_step, qz_step[0], rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("qz must be uniformly spaced.")


def _normalize_image_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError("image must have two detector dimensions.")
    detector_shape = tuple(int(v) for v in tuple(shape)[:2])
    if detector_shape[0] <= 0 or detector_shape[1] <= 0:
        raise ValueError("image detector dimensions must be positive.")
    return detector_shape


def _prepare_inputs(
    image: np.ndarray,
    qr: np.ndarray,
    qz: np.ndarray,
    normalization: np.ndarray | None,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    signal = np.asarray(image, dtype=np.float32)
    qr_axis = np.asarray(qr, dtype=np.float64)
    qz_axis = np.asarray(qz, dtype=np.float64)
    _validate_public_axes(qr_axis, qz_axis)
    _normalize_image_shape(signal.shape)
    if normalization is None:
        norm = np.ones(signal.shape, dtype=np.float32)
    else:
        norm = np.asarray(normalization, dtype=np.float32)
        if norm.shape != signal.shape:
            raise ValueError("normalization must match image shape.")
    if mask is None:
        return signal, norm, qr_axis, qz_axis, np.zeros((1, 1), dtype=np.int8), False
    mask_array = np.asarray(mask, dtype=np.int8)
    if mask_array.shape != signal.shape:
        raise ValueError("mask must match image shape.")
    return signal, norm, qr_axis, qz_axis, mask_array, True


def _signed_qr_axis(qr: np.ndarray) -> np.ndarray:
    return np.concatenate((-np.asarray(qr[::-1], dtype=np.float64), np.asarray(qr, dtype=np.float64)))


def _fold_signed_accumulator(values: np.ndarray, n_qr: int) -> np.ndarray:
    signed_values = np.asarray(values, dtype=np.float64)
    if signed_values.ndim != 2 or signed_values.shape[1] != 2 * int(n_qr):
        raise ValueError("signed accumulator shape does not match qr bin count.")
    return np.fliplr(signed_values[:, :n_qr]) + signed_values[:, n_qr:]


def _projection_frame(geometry: DetectorQSpaceGeometry):
    return build_nominal_projection_frame(
        distance_cor_to_detector=float(geometry.distance_m),
        gamma_deg=float(geometry.gamma_deg),
        Gamma_deg=float(geometry.Gamma_deg),
        chi_deg=float(geometry.chi_deg),
        psi_deg=float(geometry.psi_deg),
        psi_z_deg=float(geometry.psi_z_deg),
        zs=float(geometry.zs),
        zb=float(geometry.zb),
        theta_initial_deg=float(geometry.theta_initial_deg),
        cor_angle_deg=float(geometry.cor_angle_deg),
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )


def _detector_corner_sample_q_maps(
    image_shape: tuple[int, int],
    geometry: DetectorQSpaceGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detector_shape = _normalize_image_shape(tuple(image_shape))
    frame = _projection_frame(geometry)
    row_edges = (
        np.arange(int(detector_shape[0]) + 1, dtype=np.float64) - float(geometry.center_row_px)
    ) * float(geometry.pixel_size_m)
    col_edges = (
        np.arange(int(detector_shape[1]) + 1, dtype=np.float64) - float(geometry.center_col_px)
    ) * float(geometry.pixel_size_m)
    x_det = col_edges[None, :]
    y_det = row_edges[:, None]

    detector_points = (
        np.asarray(frame.detector_pos, dtype=np.float64)[None, None, :]
        + x_det[:, :, None] * np.asarray(frame.e1_det, dtype=np.float64)[None, None, :]
        + y_det[:, :, None] * np.asarray(frame.e2_det, dtype=np.float64)[None, None, :]
    )
    outgoing_lab = detector_points - np.asarray(frame.i_plane, dtype=np.float64)[None, None, :]
    outgoing_norm = np.linalg.norm(outgoing_lab, axis=-1, keepdims=True)
    if np.any(outgoing_norm <= 0.0):
        raise ValueError("Detector corner projection produced a zero-length outgoing ray.")
    u_f_lab = outgoing_lab / outgoing_norm
    u_i_lab = np.asarray(frame.u_i_lab, dtype=np.float64).reshape(1, 1, 3)
    r_sample = np.asarray(frame.r_sample, dtype=np.float64)
    u_f_sample = np.einsum("...i,ij->...j", u_f_lab, r_sample)
    u_i_sample = np.einsum("...i,ij->...j", u_i_lab, r_sample)
    wavelength_angstrom = float(geometry.wavelength_m) * 1.0e10
    if not np.isfinite(wavelength_angstrom) or wavelength_angstrom <= 0.0:
        raise ValueError("wavelength_m must be positive.")
    k0 = 2.0 * np.pi / wavelength_angstrom
    q_sample = k0 * (u_f_sample - u_i_sample)
    return (
        np.asarray(q_sample[..., 0], dtype=np.float64),
        np.asarray(q_sample[..., 1], dtype=np.float64),
        np.asarray(q_sample[..., 2], dtype=np.float64),
    )


def detector_corner_q_maps(
    image_shape: tuple[int, int],
    geometry: DetectorQSpaceGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qx, qy, qz = _detector_corner_sample_q_maps(image_shape, geometry)
    qr_abs = np.hypot(qx, qy)
    signed_qr = np.where(
        qy > _SEAM_EPSILON,
        qr_abs,
        np.where(qy < -_SEAM_EPSILON, -qr_abs, qr_abs),
    ).astype(np.float64, copy=False)
    return (
        np.asarray(signed_qr, dtype=np.float64),
        np.asarray(qz, dtype=np.float64),
        np.asarray(qy, dtype=np.float64),
    )


def detector_q_extents(
    image_shape: tuple[int, int],
    geometry: DetectorQSpaceGeometry,
    *,
    qx_map: np.ndarray | None = None,
    signed_qr_map: np.ndarray | None = None,
    qz_map: np.ndarray | None = None,
    qy_map: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if qx_map is None or qz_map is None or qy_map is None:
        qx_map, qy_map, qz_map = _detector_corner_sample_q_maps(image_shape, geometry)
    qr_max = float(
        np.max(
            np.hypot(
                np.asarray(qx_map, dtype=np.float64),
                np.asarray(qy_map, dtype=np.float64),
            )
        )
    )
    qz_min = float(np.min(np.asarray(qz_map, dtype=np.float64)))
    qz_max = float(np.max(np.asarray(qz_map, dtype=np.float64)))
    if not (np.isfinite(qr_max) and qr_max > 0.0):
        raise ValueError("Detector geometry did not produce a positive qr extent.")
    if not (np.isfinite(qz_min) and np.isfinite(qz_max) and qz_max > qz_min):
        raise ValueError("Detector geometry did not produce a valid qz extent.")
    return qr_max, qz_min, qz_max


def build_qspace_axes(
    *,
    npt_rad: int,
    npt_azim: int,
    qr_max: float,
    qz_min: float,
    qz_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_qr = int(max(2, npt_rad))
    n_qz = int(max(2, npt_azim))
    qr_edges = np.linspace(0.0, float(qr_max), n_qr + 1, dtype=np.float64)
    qz_edges = np.linspace(float(qz_min), float(qz_max), n_qz + 1, dtype=np.float64)
    qr = 0.5 * (qr_edges[:-1] + qr_edges[1:])
    qz = 0.5 * (qz_edges[:-1] + qz_edges[1:])
    return np.asarray(qr, dtype=np.float64), np.asarray(qz, dtype=np.float64)


def _normalize_vertex_sign(
    qx: float,
    qy: float,
    qz: float,
    sign: int,
) -> tuple[float, float, float]:
    qr_abs = math.hypot(float(qx), float(qy))
    if abs(float(qy)) <= _SEAM_EPSILON:
        return float(sign) * abs(float(qx)), float(qz), 0.0
    return float(sign) * qr_abs, float(qz), float(qy)


def _clip_polygon_to_qy_half_plane(
    vertices: list[tuple[float, float, float, float]],
    *,
    sign: int,
) -> list[tuple[float, float, float]]:
    if not vertices:
        return []
    clipped: list[tuple[float, float, float]] = []
    previous = vertices[-1]
    previous_inside = float(sign) * float(previous[1]) >= -_SEAM_EPSILON
    for current in vertices:
        current_inside = float(sign) * float(current[1]) >= -_SEAM_EPSILON
        if current_inside != previous_inside:
            denom = float(current[1]) - float(previous[1])
            if abs(denom) > _SEAM_EPSILON:
                t = -float(previous[1]) / denom
                seam_qx = float(previous[0]) + t * (float(current[0]) - float(previous[0]))
                seam_qz = float(previous[2]) + t * (float(current[2]) - float(previous[2]))
                clipped.append((float(sign) * abs(seam_qx), seam_qz, 0.0))
        if current_inside:
            clipped.append(
                _normalize_vertex_sign(
                    current[0],
                    current[1],
                    current[2],
                    sign,
                )
            )
        previous = current
        previous_inside = current_inside
    return clipped


def _split_quad_for_signed_qr(
    qx: np.ndarray,
    qz: np.ndarray,
    qy: np.ndarray,
) -> list[list[tuple[float, float, float]]]:
    vertices = [
        (float(qx[0]), float(qy[0]), float(qz[0]), 0.0),
        (float(qx[1]), float(qy[1]), float(qz[1]), 0.0),
        (float(qx[2]), float(qy[2]), float(qz[2]), 0.0),
        (float(qx[3]), float(qy[3]), float(qz[3]), 0.0),
    ]
    qy_values = np.asarray(qy, dtype=np.float64)
    if np.all(np.abs(qy_values) <= _SEAM_EPSILON):
        return [[(abs(float(v[0])), float(v[2]), 0.0) for v in vertices]]
    if np.all(qy_values >= -_SEAM_EPSILON):
        return [[_normalize_vertex_sign(v[0], v[1], v[2], 1) for v in vertices]]
    if np.all(qy_values <= _SEAM_EPSILON):
        return [[_normalize_vertex_sign(v[0], v[1], v[2], -1) for v in vertices]]
    polygons = []
    positive = _clip_polygon_to_qy_half_plane(vertices, sign=1)
    negative = _clip_polygon_to_qy_half_plane(vertices, sign=-1)
    if len(positive) >= 3:
        polygons.append(positive)
    if len(negative) >= 3:
        polygons.append(negative)
    return polygons


def _orient_polygon_for_integrator(
    polygon: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    if len(polygon) < 3:
        return []
    area = 0.0
    for index, current in enumerate(polygon):
        nxt = polygon[(index + 1) % len(polygon)]
        area += float(current[0]) * float(nxt[1]) - float(current[1]) * float(nxt[0])
    if abs(area) <= _SEAM_EPSILON:
        return []
    # `_integrate_edge()` expects detector-style winding, which is opposite the
    # usual positive shoelace orientation in this target plane.
    if area > 0.0:
        return list(reversed(polygon))
    return list(polygon)


def _polygon_raw_area_weights(
    polygon: list[tuple[float, float, float]],
    *,
    signed_qr: np.ndarray,
    qz: np.ndarray,
    box: np.ndarray,
) -> dict[tuple[int, int], float]:
    ordered_polygon = _orient_polygon_for_integrator(polygon)
    if len(ordered_polygon) < 3:
        return {}
    delta0 = float(signed_qr[1] - signed_qr[0])
    delta1 = float(qz[1] - qz[0])
    pos0_min = float(signed_qr[0] - 0.5 * delta0)
    pos1_min = float(qz[0] - 0.5 * delta1)
    pos0_maxin = _inverse_calc_upper_bound(float(signed_qr[-1] + 0.5 * delta0))
    pos1_maxin = _inverse_calc_upper_bound(float(qz[-1] + 0.5 * delta1))

    coords0 = []
    coords1 = []
    for vertex in ordered_polygon:
        coords0.append((min(max(float(vertex[0]), pos0_min), pos0_maxin) - pos0_min) / delta0)
        coords1.append((min(max(float(vertex[1]), pos1_min), pos1_maxin) - pos1_min) / delta1)
    min0 = min(coords0)
    max0 = max(coords0)
    min1 = min(coords1)
    max1 = max(coords1)
    foffset0 = math.floor(min0)
    foffset1 = math.floor(min1)
    ioffset0 = int(foffset0)
    ioffset1 = int(foffset1)
    width0 = int(math.ceil(max0) - foffset0)
    width1 = int(math.ceil(max1) - foffset1)
    if width0 <= 0 or width1 <= 0:
        return {}
    for index0 in range(width0 + 1):
        for index1 in range(width1 + 1):
            box[index0, index1] = 0.0
    shifted0 = [value - foffset0 for value in coords0]
    shifted1 = [value - foffset1 for value in coords1]
    for index in range(len(ordered_polygon)):
        next_index = (index + 1) % len(ordered_polygon)
        _integrate_edge(
            box,
            shifted0[index],
            shifted1[index],
            shifted0[next_index],
            shifted1[next_index],
        )
    raw_weights: dict[tuple[int, int], float] = {}
    for index0 in range(width0):
        bin_qr = ioffset0 + index0
        if bin_qr < 0 or bin_qr >= int(signed_qr.size):
            continue
        for index1 in range(width1):
            bin_qz = ioffset1 + index1
            if bin_qz < 0 or bin_qz >= int(qz.size):
                continue
            area = float(box[index0, index1])
            if area == 0.0 or not math.isfinite(area):
                continue
            key = (int(bin_qz), int(bin_qr))
            raw_weights[key] = float(raw_weights.get(key, 0.0)) + area
    return raw_weights


def _pixel_signed_qr_qz_raw_weights(
    row: int,
    col: int,
    qx_map: np.ndarray,
    qz_map: np.ndarray,
    qy_map: np.ndarray,
    signed_qr_axis: np.ndarray,
    qz_axis: np.ndarray,
    box: np.ndarray,
) -> dict[tuple[int, int], float]:
    qx = np.array(
        [
            qx_map[row, col],
            qx_map[row + 1, col],
            qx_map[row + 1, col + 1],
            qx_map[row, col + 1],
        ],
        dtype=np.float64,
    )
    qz = np.array(
        [
            qz_map[row, col],
            qz_map[row + 1, col],
            qz_map[row + 1, col + 1],
            qz_map[row, col + 1],
        ],
        dtype=np.float64,
    )
    qy = np.array(
        [
            qy_map[row, col],
            qy_map[row + 1, col],
            qy_map[row + 1, col + 1],
            qy_map[row, col + 1],
        ],
        dtype=np.float64,
    )
    polygons = _split_quad_for_signed_qr(qx, qz, qy)
    raw_weights: dict[tuple[int, int], float] = {}
    for polygon in polygons:
        polygon_weights = _polygon_raw_area_weights(
            polygon,
            signed_qr=signed_qr_axis,
            qz=qz_axis,
            box=box,
        )
        for key, area in polygon_weights.items():
            raw_weights[key] = float(raw_weights.get(key, 0.0)) + float(area)
    return raw_weights


def integrate_detector_to_qspace_exact(
    image: np.ndarray,
    qr: np.ndarray,
    qz: np.ndarray,
    geometry: DetectorQSpaceGeometry,
    *,
    normalization: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    qx_map: np.ndarray | None = None,
    signed_qr_map: np.ndarray | None = None,
    qz_map: np.ndarray | None = None,
    qy_map: np.ndarray | None = None,
) -> DetectorQSpaceResult:
    signal, norm, qr_axis, qz_axis, mask_array, has_mask = _prepare_inputs(
        image,
        qr,
        qz,
        normalization,
        mask,
    )
    detector_shape = _normalize_image_shape(signal.shape)
    rows_array, cols_array, use_selection = _prepare_selection(signal.shape, rows, cols)
    if qx_map is None or qz_map is None or qy_map is None:
        qx_map, qy_map, qz_map = _detector_corner_sample_q_maps(detector_shape, geometry)
    if signed_qr_map is None:
        signed_qr_map = np.where(
            qy_map > _SEAM_EPSILON,
            np.hypot(qx_map, qy_map),
            np.where(qy_map < -_SEAM_EPSILON, -np.hypot(qx_map, qy_map), np.hypot(qx_map, qy_map)),
        )
    signed_qr_axis = _signed_qr_axis(qr_axis)
    signed_sum_signal = np.zeros((int(qz_axis.size), int(signed_qr_axis.size)), dtype=np.float64)
    signed_sum_norm = np.zeros_like(signed_sum_signal)
    signed_count = np.zeros_like(signed_sum_signal)
    box = np.zeros((signed_qr_axis.size + 1, qz_axis.size + 1), dtype=np.float32)

    if use_selection:
        iterable = zip(np.asarray(rows_array, dtype=np.int64), np.asarray(cols_array, dtype=np.int64), strict=False)
    else:
        iterable = (
            (row, col)
            for row in range(int(detector_shape[0]))
            for col in range(int(detector_shape[1]))
        )

    for row, col in iterable:
        row_idx = int(row)
        col_idx = int(col)
        if has_mask and mask_array[row_idx, col_idx] != 0:
            continue
        signal_value = float(signal[row_idx, col_idx])
        norm_value = float(norm[row_idx, col_idx])
        if (not math.isfinite(signal_value)) or (not math.isfinite(norm_value)) or norm_value == 0.0:
            continue
        raw_weights = _pixel_signed_qr_qz_raw_weights(
            row_idx,
            col_idx,
            qx_map,
            qz_map,
            qy_map,
            signed_qr_axis,
            qz_axis,
            box,
        )
        total_area = float(sum(raw_weights.values()))
        if total_area <= 0.0 or not math.isfinite(total_area):
            continue
        inv_area = 1.0 / total_area
        for (bin_qz, bin_qr), area in raw_weights.items():
            weight = float(area) * inv_area
            if weight == 0.0:
                continue
            signed_sum_signal[bin_qz, bin_qr] += signal_value * weight
            signed_sum_norm[bin_qz, bin_qr] += norm_value * weight
            signed_count[bin_qz, bin_qr] += weight

    sum_signal = _fold_signed_accumulator(signed_sum_signal, int(qr_axis.size))
    sum_normalization = _fold_signed_accumulator(signed_sum_norm, int(qr_axis.size))
    count = _fold_signed_accumulator(signed_count, int(qr_axis.size))
    intensity = np.zeros_like(sum_signal, dtype=np.float32)
    valid = sum_normalization > 0.0
    intensity[valid] = (sum_signal[valid] / sum_normalization[valid]).astype(np.float32, copy=False)
    return DetectorQSpaceResult(
        qr=np.array(qr_axis, copy=True),
        qz=np.array(qz_axis, copy=True),
        intensity=intensity,
        sum_signal=sum_signal,
        sum_normalization=sum_normalization,
        count=count,
    )


def build_detector_to_qspace_lut(
    image_shape: tuple[int, int],
    qr: np.ndarray,
    qz: np.ndarray,
    geometry: DetectorQSpaceGeometry,
    *,
    qx_map: np.ndarray | None = None,
    signed_qr_map: np.ndarray | None = None,
    qz_map: np.ndarray | None = None,
    qy_map: np.ndarray | None = None,
) -> DetectorQSpaceLUT:
    if not _HAS_SCIPY:
        raise RuntimeError("Exact-q-space LUT building requires scipy.")
    detector_shape = _normalize_image_shape(tuple(image_shape))
    qr_axis = np.asarray(qr, dtype=np.float64)
    qz_axis = np.asarray(qz, dtype=np.float64)
    _validate_public_axes(qr_axis, qz_axis)
    if qx_map is None or qz_map is None or qy_map is None:
        qx_map, qy_map, qz_map = _detector_corner_sample_q_maps(detector_shape, geometry)
    if signed_qr_map is None:
        signed_qr_map = np.where(
            qy_map > _SEAM_EPSILON,
            np.hypot(qx_map, qy_map),
            np.where(qy_map < -_SEAM_EPSILON, -np.hypot(qx_map, qy_map), np.hypot(qx_map, qy_map)),
        )
    signed_qr_axis = _signed_qr_axis(qr_axis)
    box = np.zeros((signed_qr_axis.size + 1, qz_axis.size + 1), dtype=np.float32)
    bin_indices: list[int] = []
    pixel_indices: list[int] = []
    weights: list[float] = []
    internal_count = np.zeros(int(qz_axis.size * signed_qr_axis.size), dtype=np.float64)
    width = int(detector_shape[1])

    for row in range(int(detector_shape[0])):
        for col in range(int(detector_shape[1])):
            raw_weights = _pixel_signed_qr_qz_raw_weights(
                row,
                col,
                qx_map,
                qz_map,
                qy_map,
                signed_qr_axis,
                qz_axis,
                box,
            )
            total_area = float(sum(raw_weights.values()))
            if total_area <= 0.0 or not math.isfinite(total_area):
                continue
            inv_area = 1.0 / total_area
            pixel_index = int(row * width + col)
            for (bin_qz, bin_qr), area in raw_weights.items():
                weight = float(area) * inv_area
                if weight == 0.0:
                    continue
                flat_bin = int(bin_qz * signed_qr_axis.size + bin_qr)
                bin_indices.append(flat_bin)
                pixel_indices.append(pixel_index)
                weights.append(weight)
                internal_count[flat_bin] += weight

    matrix = _scipy_sparse.coo_matrix(
        (
            np.asarray(weights, dtype=np.float32),
            (
                np.asarray(bin_indices, dtype=np.int32),
                np.asarray(pixel_indices, dtype=np.int32),
            ),
        ),
        shape=(int(qz_axis.size * signed_qr_axis.size), int(np.prod(detector_shape))),
        dtype=np.float32,
    ).tocsr()
    return DetectorQSpaceLUT(
        image_shape=detector_shape,
        n_qr=int(qr_axis.size),
        n_qz=int(qz_axis.size),
        qr=np.array(qr_axis, copy=True),
        qz=np.array(qz_axis, copy=True),
        signed_qr=np.array(signed_qr_axis, copy=True),
        matrix=matrix,
        count_flat=internal_count,
    )


def integrate_detector_to_qspace_lut(
    image: np.ndarray,
    qr: np.ndarray,
    qz: np.ndarray,
    lut: DetectorQSpaceLUT,
    *,
    normalization: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> DetectorQSpaceResult:
    signal, norm, qr_axis, qz_axis, mask_array, has_mask = _prepare_inputs(
        image,
        qr,
        qz,
        normalization,
        mask,
    )
    if tuple(int(v) for v in signal.shape[:2]) != tuple(int(v) for v in lut.image_shape):
        raise ValueError("image shape does not match the cached exact-q-space LUT.")
    if int(qr_axis.size) != int(lut.n_qr) or int(qz_axis.size) != int(lut.n_qz):
        raise ValueError("qr/qz bins do not match the cached exact-q-space LUT.")
    if not np.allclose(np.asarray(lut.qr, dtype=np.float64), qr_axis, atol=1.0e-12, rtol=0.0):
        raise ValueError("qr bins do not match the cached exact-q-space LUT.")
    if not np.allclose(np.asarray(lut.qz, dtype=np.float64), qz_axis, atol=1.0e-12, rtol=0.0):
        raise ValueError("qz bins do not match the cached exact-q-space LUT.")

    signal_flat = np.asarray(signal, dtype=np.float32).reshape(-1)
    norm_flat = np.asarray(norm, dtype=np.float32).reshape(-1)
    valid_flat = np.isfinite(signal_flat) & np.isfinite(norm_flat) & (norm_flat != 0.0)
    if has_mask:
        valid_flat &= np.asarray(mask_array, dtype=np.int8).reshape(-1) == 0
    if np.all(valid_flat):
        signal_input = signal_flat
        norm_input = norm_flat
        count_flat = np.asarray(lut.count_flat, dtype=np.float64).reshape(-1)
    else:
        signal_input = np.zeros(signal_flat.shape, dtype=np.float32)
        norm_input = np.zeros(norm_flat.shape, dtype=np.float32)
        signal_input[valid_flat] = signal_flat[valid_flat]
        norm_input[valid_flat] = norm_flat[valid_flat]
        count_flat = np.asarray(
            lut.matrix @ np.asarray(valid_flat, dtype=np.float32),
            dtype=np.float64,
        ).reshape(-1)

    n_signed_qr = int(lut.signed_qr.size)
    signed_sum_signal = np.asarray(
        lut.matrix @ signal_input,
        dtype=np.float64,
    ).reshape(int(lut.n_qz), n_signed_qr)
    signed_sum_norm = np.asarray(
        lut.matrix @ norm_input,
        dtype=np.float64,
    ).reshape(int(lut.n_qz), n_signed_qr)
    signed_count = np.asarray(count_flat, dtype=np.float64).reshape(int(lut.n_qz), n_signed_qr)

    sum_signal = _fold_signed_accumulator(signed_sum_signal, int(lut.n_qr))
    sum_normalization = _fold_signed_accumulator(signed_sum_norm, int(lut.n_qr))
    count = _fold_signed_accumulator(signed_count, int(lut.n_qr))
    intensity = np.zeros_like(sum_signal, dtype=np.float32)
    valid = sum_normalization > 0.0
    intensity[valid] = (sum_signal[valid] / sum_normalization[valid]).astype(np.float32, copy=False)
    return DetectorQSpaceResult(
        qr=np.array(qr_axis, copy=True),
        qz=np.array(qz_axis, copy=True),
        intensity=intensity,
        sum_signal=sum_signal,
        sum_normalization=sum_normalization,
        count=count,
    )


__all__ = [
    "DetectorQSpaceGeometry",
    "DetectorQSpaceLUT",
    "DetectorQSpaceResult",
    "_detector_corner_sample_q_maps",
    "build_detector_to_qspace_lut",
    "build_qspace_axes",
    "detector_corner_q_maps",
    "detector_q_extents",
    "integrate_detector_to_qspace_exact",
    "integrate_detector_to_qspace_lut",
]
