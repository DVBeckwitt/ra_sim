"""Mosaic-profile helpers extracted from the monolithic optimization module."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from ra_sim.utils.calculations import resolve_canonical_branch


def _coerce_sequence_items(values: Optional[Sequence[object]]) -> List[object]:
    if values is None:
        return []
    try:
        return list(values)
    except TypeError:
        return []


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        if isinstance(value, (int, float, np.integer, np.floating, str)):
            return float(value)
        return float(str(value))
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = -1) -> int:
    try:
        if isinstance(value, (int, np.integer, str)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return int(value)
        return int(str(value))
    except Exception:
        return int(default)


def _miller_key_from_row(row: Sequence[float]) -> Optional[Tuple[int, int, int]]:
    """Return an integer HKL tuple from a Miller-array row."""

    try:
        return (
            int(round(float(row[0]))),
            int(round(float(row[1]))),
            int(round(float(row[2]))),
        )
    except Exception:
        return None


def _normalized_q_group_key(
    value: object,
) -> Optional[Tuple[object, ...]]:
    """Return a hashable Qr/Qz group key when one is present."""

    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return None


def _normalized_hkl_key(
    value: object,
) -> Optional[Tuple[int, int, int]]:
    """Return one integer HKL tuple from a stored entry value."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (
            int(value[0]),
            int(value[1]),
            int(value[2]),
        )
    except Exception:
        return None


@dataclass
class LorentzianGaussianProfileFit:
    """Fit result for an additive Lorentzian plus Gaussian peak profile."""

    center_deg: float
    gaussian_amplitude: float
    gaussian_sigma_deg: float
    lorentzian_amplitude: float
    lorentzian_gamma_deg: float
    baseline: float
    area: float
    cost: float
    residual_rms: float
    success: bool
    message: str = ""


@dataclass
class MosaicPointPair:
    """One selected Qr point paired with one measured background point."""

    pair_index: int
    pair_key: Tuple[object, ...]
    selected_qr_point: Dict[str, object]
    background_point: Dict[str, object]
    dataset_index: int | None = None
    dataset_label: str = ""
    q_group_key: Tuple[object, ...] | None = None
    hkl: Tuple[int, int, int] | None = None


@dataclass
class MosaicPhiProfile:
    """Background-subtracted phi integration extracted from one local peak ROI."""

    pair: MosaicPointPair
    phi_deg: np.ndarray
    intensity: np.ndarray
    signal_counts: np.ndarray
    background_level: np.ndarray
    center_col: float
    center_row: float
    phi_anchor_deg: float
    row_bounds: Tuple[int, int]
    col_bounds: Tuple[int, int]
    fit: LorentzianGaussianProfileFit | None = None
    weight: float = 1.0


@dataclass
class CenteredMosaicProfileComparison:
    """Residual bundle comparing profiles after shifting fitted peaks to zero."""

    residuals: np.ndarray
    profile_count: int
    residual_rms: float
    per_profile: List[Dict[str, object]]


def _freeze_key_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_freeze_key_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_key_value(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_key_value(val)) for key, val in value.items()))
    return value


def _mosaic_scaffold_pair_key(entry: Mapping[str, object] | None) -> Tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    for key in ("pair_key", "manual_pair_id", "pair_id"):
        if key in entry:
            return (str(key), _freeze_key_value(entry.get(key)))
    q_group_key = _normalized_q_group_key(entry.get("q_group_key"))
    if q_group_key is not None:
        branch, _branch_source, _branch_reason = resolve_canonical_branch(
            entry,
            allow_legacy_peak_fallback=True,
        )
        if branch in {0, 1}:
            return (
                "q_group_branch",
                tuple(_freeze_key_value(item) for item in q_group_key),
                _freeze_key_value(branch),
            )
        return ("q_group", tuple(_freeze_key_value(item) for item in q_group_key))
    hkl = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
    if hkl is not None:
        return ("hkl", int(hkl[0]), int(hkl[1]), int(hkl[2]))
    if entry.get("label") is not None:
        return ("label", str(entry.get("label")))
    return None


def _mosaic_scaffold_pair_weight(*entries: Mapping[str, object] | None) -> float:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for key in ("fit_weight", "profile_weight", "weight", "confidence"):
            value = _safe_float(entry.get(key, np.nan))
            if np.isfinite(value) and value > 0.0:
                return float(value)
    return 1.0


def _mosaic_scaffold_dataset_index(*entries: Mapping[str, object] | None) -> int | None:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for key in ("dataset_index", "background_index"):
            value = _safe_int(entry.get(key), default=-1)
            if value >= 0:
                return int(value)
    return None


def _mosaic_scaffold_dataset_label(*entries: Mapping[str, object] | None) -> str:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for key in ("dataset_label", "background_name", "label"):
            value = entry.get(key)
            if value is not None and str(value):
                return str(value)
    return ""


def _mosaic_scaffold_hkl(*entries: Mapping[str, object] | None) -> Tuple[int, int, int] | None:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        hkl = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl is not None:
            return hkl
    return None


def _mosaic_scaffold_q_group_key(
    *entries: Mapping[str, object] | None,
) -> Tuple[object, ...] | None:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        q_group_key = _normalized_q_group_key(entry.get("q_group_key"))
        if q_group_key is not None:
            return q_group_key
    return None


def pair_selected_qr_and_background_points(
    selected_qr_points: Sequence[Mapping[str, object]],
    background_points: Sequence[Mapping[str, object]] | None = None,
) -> List[MosaicPointPair]:
    """Pair user-selected Qr metadata with corresponding measured background anchors.

    Manual geometry entries already carry both pieces of information, so callers may
    pass them as ``selected_qr_points`` and leave ``background_points`` unset. When
    separate lists are provided, equal-length lists are paired by order. Otherwise a
    stable Qr/HKL/label key is used with FIFO handling for duplicate keys.
    """

    selected_entries = [dict(point) for point in selected_qr_points if isinstance(point, Mapping)]
    background_entries = (
        [dict(point) for point in background_points if isinstance(point, Mapping)]
        if background_points is not None
        else []
    )
    if not selected_entries:
        return []

    if background_points is None:
        raw_pairs = [(entry, entry) for entry in selected_entries]
    else:
        lookup: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
        for background_entry in background_entries:
            key = _mosaic_scaffold_pair_key(background_entry)
            if key is None:
                continue
            lookup.setdefault(key, []).append(background_entry)
        keyed_pairs: List[Tuple[Dict[str, object], Dict[str, object]]] = []
        for selected_entry in selected_entries:
            key = _mosaic_scaffold_pair_key(selected_entry)
            if key is None:
                continue
            candidates = lookup.get(key)
            if not candidates:
                continue
            keyed_pairs.append((selected_entry, candidates.pop(0)))
        if len(keyed_pairs) == len(selected_entries):
            raw_pairs = keyed_pairs
        elif len(selected_entries) == len(background_entries):
            raw_pairs = list(zip(selected_entries, background_entries))
        else:
            raw_pairs = keyed_pairs

    pairs: List[MosaicPointPair] = []
    for pair_index, (selected_entry, background_entry) in enumerate(raw_pairs):
        pair_key = _mosaic_scaffold_pair_key(background_entry) or _mosaic_scaffold_pair_key(
            selected_entry
        )
        if pair_key is None:
            pair_key = ("index", int(pair_index))
        pairs.append(
            MosaicPointPair(
                pair_index=int(pair_index),
                pair_key=tuple(pair_key),
                selected_qr_point=dict(selected_entry),
                background_point=dict(background_entry),
                dataset_index=_mosaic_scaffold_dataset_index(background_entry, selected_entry),
                dataset_label=_mosaic_scaffold_dataset_label(background_entry, selected_entry),
                q_group_key=_mosaic_scaffold_q_group_key(background_entry, selected_entry),
                hkl=_mosaic_scaffold_hkl(background_entry, selected_entry),
            )
        )
    return pairs


def lorentzian_plus_gaussian_profile(
    phi_deg: Sequence[float] | np.ndarray,
    center_deg: float,
    gaussian_amplitude: float,
    gaussian_sigma_deg: float,
    lorentzian_amplitude: float,
    lorentzian_gamma_deg: float,
    baseline: float,
) -> np.ndarray:
    """Return an additive Lorentzian plus Gaussian peak, not a pseudo-Voigt mix."""

    x = np.asarray(phi_deg, dtype=np.float64)
    sigma = max(float(gaussian_sigma_deg), 1.0e-12)
    gamma = max(float(lorentzian_gamma_deg), 1.0e-12)
    delta = x - float(center_deg)
    gaussian = float(gaussian_amplitude) * np.exp(-0.5 * (delta / sigma) ** 2)
    lorentzian = float(lorentzian_amplitude) / (1.0 + (delta / gamma) ** 2)
    return gaussian + lorentzian + float(baseline)


def _profile_positive_area(x_values: np.ndarray, y_values: np.ndarray) -> float:
    y = np.clip(np.asarray(y_values, dtype=np.float64), 0.0, None)
    x = np.asarray(x_values, dtype=np.float64)
    if x.size < 2:
        return float(np.sum(y))
    order = np.argsort(x)
    try:
        return float(np.trapezoid(y[order], x[order]))
    except Exception:
        return float(np.sum(y))


def _estimate_profile_center(x_values: np.ndarray, y_values: np.ndarray) -> float:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite):
        return 0.0
    x = np.asarray(x_values[finite], dtype=np.float64)
    y = np.asarray(y_values[finite], dtype=np.float64)
    if y.size == 0:
        return 0.0
    shifted = y - float(np.nanmin(y))
    shifted = np.clip(shifted, 0.0, None)
    total = float(np.sum(shifted))
    if total > 0.0 and np.isfinite(total):
        return float(np.sum(x * shifted) / total)
    return float(x[int(np.nanargmax(y))])


def fit_lorentzian_plus_gaussian_profile(
    phi_deg: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    *,
    center_guess_deg: float | None = None,
    min_width_deg: float = 1.0e-3,
    max_width_deg: float | None = None,
    max_nfev: int = 500,
) -> LorentzianGaussianProfileFit:
    """Fit one profile with independent Gaussian and Lorentzian components."""

    x_all = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    y_all = np.asarray(intensity, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    x = x_all[finite]
    y = y_all[finite]
    if x.size < 5 or y.size < 5:
        return LorentzianGaussianProfileFit(
            center_deg=float(center_guess_deg or 0.0),
            gaussian_amplitude=0.0,
            gaussian_sigma_deg=float(min_width_deg),
            lorentzian_amplitude=0.0,
            lorentzian_gamma_deg=float(min_width_deg),
            baseline=0.0,
            area=0.0,
            cost=float("inf"),
            residual_rms=float("inf"),
            success=False,
            message="insufficient_profile_points",
        )

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_span = max(x_max - x_min, float(min_width_deg) * 10.0)
    if center_guess_deg is None or not np.isfinite(float(center_guess_deg)):
        center_guess = _estimate_profile_center(x, y)
    else:
        center_guess = float(center_guess_deg)
    center_guess = float(np.clip(center_guess, x_min, x_max))

    finite_y = y[np.isfinite(y)]
    baseline_guess = float(np.percentile(finite_y, 10.0)) if finite_y.size else 0.0
    peak_guess = max(float(np.max(y) - baseline_guess), 1.0e-9)
    width_guess = max(min(0.15 * x_span, x_span / 2.0), float(min_width_deg))
    width_upper = (
        float(max_width_deg) if max_width_deg is not None else float(max(x_span, width_guess))
    )
    width_upper = max(width_upper, float(min_width_deg) * 1.01)

    p0 = np.asarray(
        [
            center_guess,
            0.55 * peak_guess,
            min(width_guess, width_upper),
            0.45 * peak_guess,
            min(1.5 * width_guess, width_upper),
            baseline_guess,
        ],
        dtype=np.float64,
    )
    lower = np.asarray(
        [x_min, 0.0, float(min_width_deg), 0.0, float(min_width_deg), -np.inf],
        dtype=np.float64,
    )
    upper = np.asarray([x_max, np.inf, width_upper, np.inf, width_upper, np.inf], dtype=np.float64)
    p0 = np.minimum(np.maximum(p0, lower), upper)

    y_scale = max(float(np.percentile(np.abs(y - baseline_guess), 90.0)), 1.0)

    def _residual(vector: np.ndarray) -> np.ndarray:
        model = lorentzian_plus_gaussian_profile(
            x,
            center_deg=float(vector[0]),
            gaussian_amplitude=float(vector[1]),
            gaussian_sigma_deg=float(vector[2]),
            lorentzian_amplitude=float(vector[3]),
            lorentzian_gamma_deg=float(vector[4]),
            baseline=float(vector[5]),
        )
        return (model - y) / y_scale

    try:
        result = least_squares(
            _residual,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=max(1, int(max_nfev)),
        )
        params = np.asarray(result.x, dtype=np.float64)
        residual = _residual(params)
        raw_model = lorentzian_plus_gaussian_profile(
            x,
            center_deg=float(params[0]),
            gaussian_amplitude=float(params[1]),
            gaussian_sigma_deg=float(params[2]),
            lorentzian_amplitude=float(params[3]),
            lorentzian_gamma_deg=float(params[4]),
            baseline=float(params[5]),
        )
        area = _profile_positive_area(x, raw_model - float(params[5]))
        return LorentzianGaussianProfileFit(
            center_deg=float(params[0]),
            gaussian_amplitude=float(params[1]),
            gaussian_sigma_deg=float(params[2]),
            lorentzian_amplitude=float(params[3]),
            lorentzian_gamma_deg=float(params[4]),
            baseline=float(params[5]),
            area=float(area),
            cost=float(result.cost),
            residual_rms=float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
            success=bool(result.success),
            message=str(result.message),
        )
    except Exception as exc:
        centered = np.clip(y - baseline_guess, 0.0, None)
        return LorentzianGaussianProfileFit(
            center_deg=float(center_guess),
            gaussian_amplitude=float(peak_guess),
            gaussian_sigma_deg=float(width_guess),
            lorentzian_amplitude=0.0,
            lorentzian_gamma_deg=float(width_guess),
            baseline=float(baseline_guess),
            area=_profile_positive_area(x, centered),
            cost=float("inf"),
            residual_rms=float("inf"),
            success=False,
            message=f"fit_failed: {exc}",
        )


def _detector_anchor_from_mosaic_point(
    entry: Mapping[str, object],
) -> Tuple[Tuple[float, float] | None, str]:
    for x_key, y_key, reason in (
        ("native_col", "native_row", "resolved_native_anchor"),
        ("background_detector_x", "background_detector_y", "resolved_background_detector_anchor"),
        ("detector_x", "detector_y", "resolved_detector_anchor"),
        ("x", "y", "resolved_display_anchor"),
    ):
        col = _safe_float(entry.get(x_key, np.nan))
        row = _safe_float(entry.get(y_key, np.nan))
        if np.isfinite(col) and np.isfinite(row):
            return (float(col), float(row)), str(reason)
    return None, "missing_detector_anchor"


def _phi_anchor_from_mosaic_point(entry: Mapping[str, object]) -> float:
    for key in ("background_phi_deg", "phi_deg", "caked_y", "raw_caked_y"):
        value = _safe_float(entry.get(key, np.nan))
        if np.isfinite(value):
            return float(value)
    return float("nan")


def _coerce_phi_axis(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"0", "row", "rows", "y", "height"}:
            return 0
        if text in {"1", "col", "cols", "column", "columns", "x", "width"}:
            return 1
    axis = _safe_int(value, default=-1)
    if axis not in {0, 1}:
        raise ValueError("phi_axis must be 0/'row' or 1/'col'")
    return int(axis)


def _local_phi_values(
    phi_deg_map: Sequence[float] | np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    phi_axis: object = None,
) -> Tuple[np.ndarray, int]:
    phi_arr = np.asarray(phi_deg_map, dtype=np.float64)
    requested_axis = _coerce_phi_axis(phi_axis)
    if phi_arr.ndim == 2:
        if phi_arr.shape != tuple(image_shape):
            raise ValueError("2D phi_deg_map must match the image shape")
        local = phi_arr[np.ix_(rows, cols)]
        if requested_axis is not None:
            return np.asarray(local, dtype=np.float64), int(requested_axis)
        row_step = (
            float(np.nanmedian(np.abs(np.diff(local, axis=0)))) if local.shape[0] > 1 else 0.0
        )
        col_step = (
            float(np.nanmedian(np.abs(np.diff(local, axis=1)))) if local.shape[1] > 1 else 0.0
        )
        inferred_axis = 0 if row_step >= col_step else 1
        return np.asarray(local, dtype=np.float64), int(inferred_axis)
    if phi_arr.ndim != 1:
        raise ValueError("phi_deg_map must be a 1D axis or 2D map")
    height, width = int(image_shape[0]), int(image_shape[1])
    if requested_axis is None and height == width and phi_arr.size == height:
        raise ValueError("ambiguous 1D phi_deg_map for square image; pass phi_axis")
    if requested_axis == 0:
        if phi_arr.size != height:
            raise ValueError("row phi_deg_map length must match image height")
        return np.broadcast_to(phi_arr[rows][:, None], (rows.size, cols.size)).copy(), 0
    if requested_axis == 1:
        if phi_arr.size != width:
            raise ValueError("column phi_deg_map length must match image width")
        return np.broadcast_to(phi_arr[cols][None, :], (rows.size, cols.size)).copy(), 1
    if phi_arr.size == height:
        return np.broadcast_to(phi_arr[rows][:, None], (rows.size, cols.size)).copy(), 0
    if phi_arr.size == width:
        return np.broadcast_to(phi_arr[cols][None, :], (rows.size, cols.size)).copy(), 1
    raise ValueError("1D phi_deg_map length must match image height or width")


def integrate_selected_qr_phi_profiles(
    image: np.ndarray,
    selected_qr_points: Sequence[Mapping[str, object]],
    background_points: Sequence[Mapping[str, object]] | None = None,
    *,
    phi_deg_map: Sequence[float] | np.ndarray,
    phi_axis: object = None,
    roi_half_width_px: int = 16,
    orthogonal_half_width_px: int | None = None,
    phi_bin_count: int = 61,
    fit_profiles: bool = True,
) -> Tuple[List[MosaicPhiProfile], List[Dict[str, object]]]:
    """Integrate selected Qr/background peak ROIs into intensity versus phi."""

    image_arr = np.asarray(image, dtype=np.float64)
    if image_arr.ndim != 2:
        raise ValueError("image must be a 2D array")
    height, width = image_arr.shape
    roi_half_width = max(1, int(roi_half_width_px))
    orth_half_width = (
        max(0, int(orthogonal_half_width_px))
        if orthogonal_half_width_px is not None
        else max(1, roi_half_width // 3)
    )
    bin_count = max(5, int(phi_bin_count))
    pairs = pair_selected_qr_and_background_points(selected_qr_points, background_points)
    profiles: List[MosaicPhiProfile] = []
    rejected: List[Dict[str, object]] = []

    for pair in pairs:
        anchor, anchor_reason = _detector_anchor_from_mosaic_point(pair.background_point)
        if anchor is None:
            rejected.append(
                {
                    "pair_index": int(pair.pair_index),
                    "pair_key": list(pair.pair_key),
                    "reason": anchor_reason,
                }
            )
            continue
        center_col, center_row = float(anchor[0]), float(anchor[1])
        row_center = int(round(center_row))
        col_center = int(round(center_col))
        row0 = max(0, row_center - roi_half_width)
        row1 = min(height, row_center + roi_half_width + 1)
        col0 = max(0, col_center - roi_half_width)
        col1 = min(width, col_center + roi_half_width + 1)
        if row1 <= row0 or col1 <= col0:
            rejected.append(
                {
                    "pair_index": int(pair.pair_index),
                    "pair_key": list(pair.pair_key),
                    "reason": "empty_roi",
                }
            )
            continue
        rows = np.arange(row0, row1, dtype=np.int64)
        cols = np.arange(col0, col1, dtype=np.int64)
        try:
            local_phi, resolved_phi_axis = _local_phi_values(
                phi_deg_map,
                rows,
                cols,
                (height, width),
                phi_axis=phi_axis,
            )
        except Exception as exc:
            rejected.append(
                {
                    "pair_index": int(pair.pair_index),
                    "pair_key": list(pair.pair_key),
                    "reason": f"invalid_phi_map: {exc}",
                }
            )
            continue
        phi_anchor = _phi_anchor_from_mosaic_point(pair.background_point)
        if not np.isfinite(phi_anchor):
            local_row = int(np.clip(row_center - row0, 0, local_phi.shape[0] - 1))
            local_col = int(np.clip(col_center - col0, 0, local_phi.shape[1] - 1))
            phi_anchor = _safe_float(local_phi[local_row, local_col], 0.0)
        phi_delta = _angular_difference_array_deg(local_phi, float(phi_anchor))
        rr, cc = np.meshgrid(rows.astype(np.float64), cols.astype(np.float64), indexing="ij")
        if resolved_phi_axis == 0:
            orthogonal_offset = cc - float(center_col)
        else:
            orthogonal_offset = rr - float(center_row)
        signal_mask = np.abs(orthogonal_offset) <= float(orth_half_width)
        side_inner = min(float(roi_half_width), float(orth_half_width + 1))
        side_mask = (np.abs(orthogonal_offset) >= side_inner) & (
            np.abs(orthogonal_offset) <= float(roi_half_width)
        )
        valid_signal = signal_mask & np.isfinite(phi_delta)
        if int(np.count_nonzero(valid_signal)) < 3:
            rejected.append(
                {
                    "pair_index": int(pair.pair_index),
                    "pair_key": list(pair.pair_key),
                    "reason": "insufficient_signal_pixels",
                }
            )
            continue
        finite_abs_delta = np.abs(phi_delta[valid_signal])
        finite_abs_delta = finite_abs_delta[np.isfinite(finite_abs_delta)]
        half_span = float(np.max(finite_abs_delta)) if finite_abs_delta.size else 0.0
        if not np.isfinite(half_span) or half_span <= 0.0:
            half_span = 0.5
        half_span = max(float(half_span), 1.0e-6)
        bin_indices = _profile_bin_indices(phi_delta, half_span_deg=half_span, bin_count=bin_count)
        valid_bins = np.asarray(bin_indices >= 0, dtype=bool)
        signal = valid_signal & valid_bins
        side = side_mask & valid_bins & np.isfinite(phi_delta)
        local_values = np.nan_to_num(image_arr[np.ix_(rows, cols)], nan=0.0, posinf=0.0, neginf=0.0)
        flat_bins = bin_indices.reshape(-1)
        signal_flat = signal.reshape(-1)
        side_flat = side.reshape(-1)
        values_flat = local_values.reshape(-1)
        signal_sums = np.bincount(
            flat_bins[signal_flat],
            weights=values_flat[signal_flat],
            minlength=bin_count,
        ).astype(np.float64, copy=False)
        signal_counts = np.bincount(flat_bins[signal_flat], minlength=bin_count).astype(
            np.float64,
            copy=False,
        )
        if np.any(side_flat):
            side_sums = np.bincount(
                flat_bins[side_flat],
                weights=values_flat[side_flat],
                minlength=bin_count,
            ).astype(np.float64, copy=False)
            side_counts = np.bincount(flat_bins[side_flat], minlength=bin_count).astype(
                np.float64,
                copy=False,
            )
            fallback_background = float(np.median(values_flat[side_flat]))
            background_level = np.full(bin_count, fallback_background, dtype=np.float64)
            valid_side = side_counts > 0.0
            background_level[valid_side] = side_sums[valid_side] / side_counts[valid_side]
        else:
            fallback_background = float(np.median(values_flat[np.isfinite(values_flat)]))
            background_level = np.full(bin_count, fallback_background, dtype=np.float64)
        intensity = signal_sums - background_level * signal_counts
        intensity = np.clip(np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        bin_centers = np.linspace(-half_span, half_span, bin_count, dtype=np.float64)
        profile = MosaicPhiProfile(
            pair=pair,
            phi_deg=np.asarray(bin_centers, dtype=np.float64),
            intensity=np.asarray(intensity, dtype=np.float64),
            signal_counts=np.asarray(signal_counts, dtype=np.float64),
            background_level=np.asarray(background_level, dtype=np.float64),
            center_col=float(center_col),
            center_row=float(center_row),
            phi_anchor_deg=float(phi_anchor),
            row_bounds=(int(row0), int(row1)),
            col_bounds=(int(col0), int(col1)),
            fit=None,
            weight=_mosaic_scaffold_pair_weight(pair.background_point, pair.selected_qr_point),
        )
        if fit_profiles:
            profile.fit = fit_lorentzian_plus_gaussian_profile(
                profile.phi_deg,
                profile.intensity,
                center_guess_deg=0.0,
            )
        profiles.append(profile)

    return profiles, rejected


def _profile_center_deg(profile: MosaicPhiProfile) -> float:
    if profile.fit is not None and np.isfinite(float(profile.fit.center_deg)):
        return float(profile.fit.center_deg)
    return _estimate_profile_center(
        np.asarray(profile.phi_deg, dtype=np.float64),
        np.asarray(profile.intensity, dtype=np.float64),
    )


def _normalized_profile_intensity(profile: MosaicPhiProfile) -> np.ndarray:
    y = np.clip(np.asarray(profile.intensity, dtype=np.float64), 0.0, None)
    total = float(np.sum(y))
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(y, dtype=np.float64)
    return y / total


def _interpolated_centered_profile(
    profile: MosaicPhiProfile,
    grid_deg: np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    x = np.asarray(profile.phi_deg, dtype=np.float64) - _profile_center_deg(profile)
    y = _normalized_profile_intensity(profile) if normalize else np.asarray(profile.intensity)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.zeros_like(grid_deg, dtype=np.float64)
    x = x[finite]
    y = y[finite]
    order = np.argsort(x)
    return np.interp(
        np.asarray(grid_deg, dtype=np.float64),
        x[order],
        y[order],
        left=0.0,
        right=0.0,
    )


def _default_centered_grid(profiles: Sequence[MosaicPhiProfile], bin_count: int) -> np.ndarray:
    left_values: List[float] = []
    right_values: List[float] = []
    for profile in profiles:
        x = np.asarray(profile.phi_deg, dtype=np.float64) - _profile_center_deg(profile)
        finite = x[np.isfinite(x)]
        if finite.size:
            left_values.append(float(np.min(finite)))
            right_values.append(float(np.max(finite)))
    if not left_values or not right_values:
        return np.linspace(-1.0, 1.0, max(5, int(bin_count)), dtype=np.float64)
    left = max(left_values)
    right = min(right_values)
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        left = min(left_values)
        right = max(right_values)
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        left, right = -1.0, 1.0
    return np.linspace(float(left), float(right), max(5, int(bin_count)), dtype=np.float64)


def stack_centered_phi_profiles(
    profiles: Sequence[MosaicPhiProfile],
    *,
    common_phi_grid_deg: Sequence[float] | np.ndarray | None = None,
    normalize: bool = True,
    bin_count: int = 101,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return profiles interpolated onto a common peak-centered phi grid."""

    profile_list = list(profiles)
    if common_phi_grid_deg is None:
        grid = _default_centered_grid(profile_list, bin_count=int(bin_count))
    else:
        grid = np.asarray(common_phi_grid_deg, dtype=np.float64).reshape(-1)
    rows = [
        _interpolated_centered_profile(profile, grid, normalize=bool(normalize))
        for profile in profile_list
    ]
    if not rows:
        return grid, np.empty((0, grid.size), dtype=np.float64)
    return grid, np.vstack(rows).astype(np.float64, copy=False)


def compare_centered_phi_profiles(
    measured_profiles: Sequence[MosaicPhiProfile],
    candidate_profiles: Sequence[MosaicPhiProfile],
    *,
    common_phi_grid_deg: Sequence[float] | np.ndarray | None = None,
    normalize: bool = True,
    missing_profile_penalty: float = 1.0,
) -> CenteredMosaicProfileComparison:
    """Compare measured and candidate phi integrations after centering each peak."""

    measured = list(measured_profiles)
    candidate = list(candidate_profiles)
    if common_phi_grid_deg is None:
        grid = _default_centered_grid(measured + candidate, bin_count=101)
    else:
        grid = np.asarray(common_phi_grid_deg, dtype=np.float64).reshape(-1)
    candidate_lookup: Dict[Tuple[object, ...], List[MosaicPhiProfile]] = {}
    for profile in candidate:
        candidate_lookup.setdefault(tuple(profile.pair.pair_key), []).append(profile)
    residual_blocks: List[np.ndarray] = []
    per_profile: List[Dict[str, object]] = []

    def _comparison_weight(profile: MosaicPhiProfile) -> float:
        value = _safe_float(profile.weight, 1.0)
        if not np.isfinite(value) or value <= 0.0:
            return 0.0
        return math.sqrt(float(value))

    for profile in measured:
        measured_y = _interpolated_centered_profile(profile, grid, normalize=bool(normalize))
        matches = candidate_lookup.get(tuple(profile.pair.pair_key), [])
        match = matches.pop(0) if matches else None
        weight = _comparison_weight(profile)
        if match is None:
            residual = (float(missing_profile_penalty) + measured_y) * weight
            per_profile.append(
                {
                    "pair_key": list(profile.pair.pair_key),
                    "matched": False,
                    "rms": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
                }
            )
        else:
            candidate_y = _interpolated_centered_profile(match, grid, normalize=bool(normalize))
            residual = (candidate_y - measured_y) * weight
            per_profile.append(
                {
                    "pair_key": list(profile.pair.pair_key),
                    "matched": True,
                    "measured_center_deg": float(_profile_center_deg(profile)),
                    "candidate_center_deg": float(_profile_center_deg(match)),
                    "rms": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
                }
            )
        residual_blocks.append(np.asarray(residual, dtype=np.float64))
    extra_residual_sq = np.zeros_like(grid, dtype=np.float64)
    for leftovers in candidate_lookup.values():
        for profile in leftovers:
            candidate_y = _interpolated_centered_profile(profile, grid, normalize=bool(normalize))
            weight = _comparison_weight(profile)
            residual = (float(missing_profile_penalty) + candidate_y) * weight
            extra_residual_sq += residual * residual
            per_profile.append(
                {
                    "pair_key": list(profile.pair.pair_key),
                    "matched": False,
                    "extra_candidate": True,
                    "candidate_center_deg": float(_profile_center_deg(profile)),
                    "rms": float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0,
                }
            )
    residual_blocks.append(np.sqrt(extra_residual_sq).astype(np.float64, copy=False))
    residuals = (
        np.concatenate(residual_blocks).astype(np.float64, copy=False)
        if residual_blocks
        else np.empty(0, dtype=np.float64)
    )
    residual_rms = float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else 0.0
    return CenteredMosaicProfileComparison(
        residuals=residuals,
        profile_count=int(len(per_profile)),
        residual_rms=float(residual_rms),
        per_profile=per_profile,
    )


def _ensure_lorentzian_gaussian_fits(
    profiles: Sequence[MosaicPhiProfile],
) -> List[MosaicPhiProfile]:
    output = list(profiles)
    for profile in output:
        if profile.fit is None:
            profile.fit = fit_lorentzian_plus_gaussian_profile(
                profile.phi_deg,
                profile.intensity,
                center_guess_deg=0.0,
            )
    return output


def fit_mosaic_parameters_from_centered_phi_profiles(
    measured_profiles: Sequence[MosaicPhiProfile],
    initial_parameters: Mapping[str, float],
    active_parameter_names: Sequence[str],
    simulate_profiles: Callable[[Mapping[str, float]], Sequence[MosaicPhiProfile]],
    *,
    bounds: Mapping[str, Tuple[float, float]] | None = None,
    common_phi_grid_deg: Sequence[float] | np.ndarray | None = None,
    normalize_profiles: bool = True,
    max_nfev: int = 80,
    loss: str = "soft_l1",
    f_scale: float = 1.0,
    progress_callback: Callable[[str], None] | None = None,
) -> OptimizeResult:
    """Fit chosen parameters by minimizing centered phi-profile residuals.

    This is the geometry-locked scaffold: callers provide the selected profiles
    and a simulation callback for the current parameter dictionary. Only names in
    ``active_parameter_names`` are changed by the optimizer.
    """

    measured = _ensure_lorentzian_gaussian_fits(measured_profiles)
    if not measured:
        raise RuntimeError("mosaic parameter fit needs at least one measured phi profile")
    active_names = [str(name) for name in active_parameter_names if str(name)]
    if not active_names:
        raise RuntimeError("mosaic parameter fit needs at least one active parameter")
    base_params = {str(key): float(value) for key, value in dict(initial_parameters).items()}
    bound_map = dict(bounds or {})
    x0: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    for name in active_names:
        if name not in base_params:
            raise RuntimeError(f"missing initial mosaic parameter: {name}")
        lo, hi = bound_map.get(name, (-np.inf, np.inf))
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f <= lo_f:
            raise RuntimeError(f"invalid bounds for mosaic parameter: {name}")
        value = float(np.clip(base_params[name], lo_f, hi_f))
        x0.append(value)
        lower.append(lo_f)
        upper.append(hi_f)

    evaluation_count = 0

    def _params_from_vector(vector: np.ndarray) -> Dict[str, float]:
        params = dict(base_params)
        for idx, name in enumerate(active_names):
            params[name] = float(vector[idx])
        return params

    def _residual(vector: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count
        evaluation_count += 1
        params = _params_from_vector(np.asarray(vector, dtype=np.float64))
        predicted = _ensure_lorentzian_gaussian_fits(simulate_profiles(params))
        comparison = compare_centered_phi_profiles(
            measured,
            predicted,
            common_phi_grid_deg=common_phi_grid_deg,
            normalize=bool(normalize_profiles),
        )
        if progress_callback is not None:
            try:
                progress_callback(
                    "mosaic scaffold eval "
                    f"{evaluation_count}: rms={comparison.residual_rms:.6g}, "
                    + ", ".join(f"{name}={params[name]:.6g}" for name in active_names)
                )
            except Exception:
                pass
        return comparison.residuals

    initial_residual = _residual(np.asarray(x0, dtype=np.float64))
    result = least_squares(
        _residual,
        np.asarray(x0, dtype=np.float64),
        bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        loss=str(loss),
        f_scale=float(f_scale),
        max_nfev=max(1, int(max_nfev)),
    )
    best_parameters = _params_from_vector(np.asarray(result.x, dtype=np.float64))
    predicted_best = _ensure_lorentzian_gaussian_fits(simulate_profiles(best_parameters))
    final_comparison = compare_centered_phi_profiles(
        measured,
        predicted_best,
        common_phi_grid_deg=common_phi_grid_deg,
        normalize=bool(normalize_profiles),
    )
    result.best_parameters = best_parameters
    result.active_parameter_names = list(active_names)
    result.initial_parameters = dict(base_params)
    result.initial_cost = float(0.5 * np.dot(initial_residual, initial_residual))
    result.final_cost = float(0.5 * np.dot(final_comparison.residuals, final_comparison.residuals))
    result.final_comparison = final_comparison
    result.measured_profile_count = int(len(measured))
    result.evaluation_count = int(evaluation_count)
    return result


@dataclass
class MosaicProfileROI:
    """Prepared fixed-window angular profile extracted around one measured peak."""

    dataset_index: int
    dataset_label: str
    reflection_index: int
    hkl: Tuple[int, int, int]
    family: str
    axis_name: str
    center_row: float
    center_col: float
    row_bounds: Tuple[int, int]
    col_bounds: Tuple[int, int]
    flat_indices: np.ndarray
    axis_bin_indices: np.ndarray
    signal_mask: np.ndarray
    side_mask: np.ndarray
    signal_counts: np.ndarray
    side_counts: np.ndarray
    axis_bin_centers: np.ndarray
    axis_half_span_deg: float
    orthogonal_half_window_deg: float
    measured_two_theta_deg: float
    measured_phi_deg: float
    measured_profile: np.ndarray
    measured_area: float
    measured_shape_profile: np.ndarray


@dataclass
class MosaicProfileDatasetContext:
    """Prepared per-dataset inputs for profile-based mosaic fitting."""

    dataset_index: int
    label: str
    theta_initial: float
    experimental_image: np.ndarray
    miller: np.ndarray
    intensities: np.ndarray
    rois: List[MosaicProfileROI]
    measured_peak_count: int
    in_plane_roi_count: int
    specular_roi_count: int


def _mosaic_profile_group_key(
    entry: Mapping[str, object] | None,
    *,
    hkl: Optional[Tuple[int, int, int]] = None,
) -> Optional[Tuple[object, ...]]:
    """Return the stable in-plane grouping key used by focused mosaic fitting."""

    if isinstance(entry, Mapping):
        q_group_key = _normalized_q_group_key(entry.get("q_group_key"))
        if q_group_key is not None:
            return ("q_group",) + tuple(q_group_key)
    if hkl is None and isinstance(entry, Mapping):
        hkl = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
    if hkl is None:
        return None
    return ("hkl", int(hkl[0]), int(hkl[1]), int(hkl[2]))


def _mosaic_profile_intensity_lookup(
    miller: np.ndarray,
    intensities: np.ndarray,
) -> Dict[Tuple[int, int, int], float]:
    """Return the strongest absolute intensity recorded for each HKL."""

    lookup: Dict[Tuple[int, int, int], float] = {}
    miller_arr = np.asarray(miller, dtype=np.float64)
    intensities_arr = np.asarray(intensities, dtype=np.float64).reshape(-1)
    row_count = min(
        int(miller_arr.shape[0]) if miller_arr.ndim == 2 else 0,
        int(intensities_arr.shape[0]),
    )
    for idx in range(max(row_count, 0)):
        hkl_key = _miller_key_from_row(miller_arr[idx])
        if hkl_key is None:
            continue
        try:
            intensity_value = float(abs(float(intensities_arr[idx])))
        except Exception:
            continue
        if not np.isfinite(intensity_value):
            continue
        lookup[hkl_key] = max(float(lookup.get(hkl_key, 0.0)), intensity_value)
    return lookup


def _mosaic_profile_peak_score(
    entry: Mapping[str, object] | None,
    *,
    miller: np.ndarray,
    intensities: np.ndarray,
    intensity_lookup: Mapping[Tuple[int, int, int], float],
) -> float:
    """Return the ranking score for one focused mosaic-profile peak candidate."""

    if not isinstance(entry, Mapping):
        return 0.0

    for key in ("weight", "background_intensity", "prominence_sigma", "confidence"):
        value = _safe_float(entry.get(key, np.nan))
        if np.isfinite(value) and value > 0.0:
            return float(value)

    hkl_key = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
    miller_arr = np.asarray(miller, dtype=np.float64)
    intensities_arr = np.asarray(intensities, dtype=np.float64).reshape(-1)
    table_idx = _safe_int(entry.get("source_table_index"))
    if (
        hkl_key is not None
        and miller_arr.ndim == 2
        and 0 <= table_idx < miller_arr.shape[0]
        and table_idx < intensities_arr.shape[0]
    ):
        source_hkl = _miller_key_from_row(miller_arr[table_idx])
        if source_hkl == hkl_key:
            try:
                score = float(abs(float(intensities_arr[table_idx])))
            except Exception:
                score = float("nan")
            if np.isfinite(score) and score > 0.0:
                return float(score)

    if hkl_key is not None:
        try:
            score = float(intensity_lookup.get(hkl_key, 0.0))
        except Exception:
            score = 0.0
        if np.isfinite(score) and score > 0.0:
            return float(score)
    return 0.0


def _mosaic_profile_entry_priority(
    entry: Mapping[str, object] | None,
    *,
    miller: np.ndarray,
    intensities: np.ndarray,
    intensity_lookup: Mapping[Tuple[int, int, int], float],
) -> Tuple[object, ...]:
    """Return a stable ordering tuple for focused mosaic-profile candidates."""

    score = _mosaic_profile_peak_score(
        entry,
        miller=miller,
        intensities=intensities,
        intensity_lookup=intensity_lookup,
    )
    source_branch_index = math.inf
    if isinstance(entry, Mapping):
        branch_idx, _branch_source, _branch_reason = resolve_canonical_branch(
            entry,
            allow_legacy_peak_fallback=False,
        )
        if branch_idx in {0, 1}:
            source_branch_index = int(branch_idx)
    source_row_index = (
        _safe_int(entry.get("source_row_index")) if isinstance(entry, Mapping) else math.inf
    )
    hkl_key = (
        _normalized_hkl_key(entry.get("hkl", entry.get("label")))
        if isinstance(entry, Mapping)
        else None
    )
    group_key = (
        _normalized_q_group_key(entry.get("q_group_key")) if isinstance(entry, Mapping) else None
    )
    return (
        -float(score),
        int(source_branch_index) if np.isfinite(source_branch_index) else math.inf,
        int(source_row_index) if np.isfinite(source_row_index) else math.inf,
        str(group_key) if group_key is not None else "",
        str(hkl_key) if hkl_key is not None else "",
        str(entry.get("label", "")) if isinstance(entry, Mapping) else "",
    )


def focus_mosaic_profile_dataset_specs(
    dataset_specs: Sequence[Dict[str, object]],
    *,
    source_miller: np.ndarray,
    source_intensities: np.ndarray,
    reference_dataset_index: object = None,
    max_in_plane_groups: int = 3,
    coerce_sequence_items: Callable[[Optional[Sequence[object]]], List[object]] | None = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Reduce mosaic-fit datasets to all 00L peaks plus the top in-plane groups."""

    coerce_items = coerce_sequence_items or _coerce_sequence_items
    dataset_entries = [
        dict(spec) for spec in coerce_items(dataset_specs) if isinstance(spec, Mapping)
    ]
    max_groups = max(int(max_in_plane_groups), 0)
    input_peak_count_by_dataset: Dict[str, int] = {}
    selected_peak_count_by_dataset: Dict[str, int] = {}
    summary: Dict[str, object] = {
        "mode": "focused",
        "max_in_plane_groups": int(max_groups),
        "reference_dataset_index": None,
        "reference_dataset_label": None,
        "selected_specular_hkls": [],
        "selected_in_plane_hkls": [],
        "selected_in_plane_groups": [],
        "input_peak_count_by_dataset": input_peak_count_by_dataset,
        "selected_peak_count_by_dataset": selected_peak_count_by_dataset,
    }
    if not dataset_entries:
        return [], summary

    source_miller_arr = np.asarray(source_miller, dtype=np.float64)
    source_intensities_arr = np.asarray(source_intensities, dtype=np.float64)
    intensity_lookup = _mosaic_profile_intensity_lookup(
        source_miller_arr,
        source_intensities_arr,
    )

    reference_pos = 0
    try:
        reference_idx = _safe_int(reference_dataset_index)
    except Exception:
        reference_idx = None
    if reference_idx is not None:
        for idx, spec in enumerate(dataset_entries):
            try:
                if int(spec.get("dataset_index", idx)) == reference_idx:
                    reference_pos = idx
                    break
            except Exception:
                continue

    reference_spec = dict(dataset_entries[reference_pos])
    summary["reference_dataset_index"] = reference_spec.get(
        "dataset_index",
        reference_pos,
    )
    summary["reference_dataset_label"] = reference_spec.get(
        "label",
        f"dataset_{reference_pos}",
    )

    specular_order: List[Tuple[int, int, int]] = []
    seen_specular: Set[Tuple[int, int, int]] = set()
    in_plane_groups: Dict[Tuple[object, ...], Dict[str, object]] = {}

    for raw_entry in coerce_items(reference_spec.get("measured_peaks")):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        hkl_key = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
        family_info = _classify_mosaic_profile_family(hkl_key) if hkl_key is not None else None
        if family_info is None or hkl_key is None:
            continue
        family, _ = family_info
        if family == "specular":
            if hkl_key not in seen_specular:
                seen_specular.add(hkl_key)
                specular_order.append(hkl_key)
            continue

        group_key = _mosaic_profile_group_key(entry, hkl=hkl_key)
        if group_key is None:
            continue
        group_info = in_plane_groups.setdefault(
            group_key,
            {"entries": [], "score": 0.0},
        )
        group_info["entries"].append(dict(entry))
        group_info["score"] = _safe_float(
            group_info.get("score", 0.0), 0.0
        ) + _mosaic_profile_peak_score(
            entry,
            miller=source_miller_arr,
            intensities=source_intensities_arr,
            intensity_lookup=intensity_lookup,
        )

    ranked_group_items = sorted(
        in_plane_groups.items(),
        key=lambda item: (
            -_safe_float(item[1].get("score", 0.0), 0.0),
            min(
                _mosaic_profile_entry_priority(
                    entry,
                    miller=source_miller_arr,
                    intensities=source_intensities_arr,
                    intensity_lookup=intensity_lookup,
                )
                for entry in item[1].get("entries", [])
                if isinstance(entry, Mapping)
            )
            if item[1].get("entries")
            else (math.inf,),
        ),
    )

    selected_in_plane_group_order: List[Tuple[object, ...]] = []
    selected_in_plane_hkls: List[Tuple[int, int, int]] = []
    for group_key, group_info in ranked_group_items[:max_groups]:
        entries = [
            dict(entry) for entry in group_info.get("entries", []) if isinstance(entry, Mapping)
        ]
        if not entries:
            continue
        representative = min(
            entries,
            key=lambda entry: _mosaic_profile_entry_priority(
                entry,
                miller=source_miller_arr,
                intensities=source_intensities_arr,
                intensity_lookup=intensity_lookup,
            ),
        )
        representative_hkl = _normalized_hkl_key(
            representative.get("hkl", representative.get("label"))
        )
        if representative_hkl is None:
            continue
        selected_in_plane_group_order.append(group_key)
        selected_in_plane_hkls.append(representative_hkl)

    allowed_specular = set(specular_order)
    allowed_in_plane_groups = set(selected_in_plane_group_order)
    summary["selected_specular_hkls"] = [[int(v) for v in hkl] for hkl in specular_order]
    summary["selected_in_plane_hkls"] = [[int(v) for v in hkl] for hkl in selected_in_plane_hkls]
    summary["selected_in_plane_groups"] = [
        list(group_key) for group_key in selected_in_plane_group_order
    ]

    focused_specs: List[Dict[str, object]] = []
    for spec_idx, raw_spec in enumerate(dataset_entries):
        spec = dict(raw_spec)
        dataset_key = str(spec.get("dataset_index", spec_idx))
        raw_measured = coerce_items(spec.get("measured_peaks"))
        input_peak_count_by_dataset[dataset_key] = int(len(raw_measured))

        best_specular_entries: Dict[Tuple[int, int, int], Dict[str, object]] = {}
        best_in_plane_entries: Dict[Tuple[object, ...], Dict[str, object]] = {}
        for raw_entry in raw_measured:
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            hkl_key = _normalized_hkl_key(entry.get("hkl", entry.get("label")))
            family_info = _classify_mosaic_profile_family(hkl_key) if hkl_key is not None else None
            if family_info is None or hkl_key is None:
                continue
            family, _ = family_info
            priority = _mosaic_profile_entry_priority(
                entry,
                miller=source_miller_arr,
                intensities=source_intensities_arr,
                intensity_lookup=intensity_lookup,
            )
            if family == "specular":
                if hkl_key not in allowed_specular:
                    continue
                previous = best_specular_entries.get(hkl_key)
                if previous is None or priority < _mosaic_profile_entry_priority(
                    previous,
                    miller=source_miller_arr,
                    intensities=source_intensities_arr,
                    intensity_lookup=intensity_lookup,
                ):
                    best_specular_entries[hkl_key] = dict(entry)
                continue

            group_key = _mosaic_profile_group_key(entry, hkl=hkl_key)
            if group_key is None or group_key not in allowed_in_plane_groups:
                continue
            previous = best_in_plane_entries.get(group_key)
            if previous is None or priority < _mosaic_profile_entry_priority(
                previous,
                miller=source_miller_arr,
                intensities=source_intensities_arr,
                intensity_lookup=intensity_lookup,
            ):
                best_in_plane_entries[group_key] = dict(entry)

        focused_measured: List[Dict[str, object]] = []
        for hkl_key in specular_order:
            selected_entry = best_specular_entries.get(hkl_key)
            if selected_entry is not None:
                focused_measured.append(dict(selected_entry))
        for group_key in selected_in_plane_group_order:
            selected_entry = best_in_plane_entries.get(group_key)
            if selected_entry is not None:
                focused_measured.append(dict(selected_entry))

        spec["measured_peaks"] = focused_measured
        selected_peak_count_by_dataset[dataset_key] = int(len(focused_measured))
        focused_specs.append(spec)

    return focused_specs, summary


def _angular_difference_array_deg(
    values: Sequence[float] | np.ndarray,
    reference: float,
) -> np.ndarray:
    """Return wrapped angular deltas relative to one reference angle."""

    arr = np.asarray(values, dtype=np.float64)
    return np.mod(arr - float(reference) + 180.0, 360.0) - 180.0


def _classify_mosaic_profile_family(
    hkl: Tuple[int, int, int],
) -> Optional[Tuple[str, str]]:
    """Classify one reflection into the supported mosaic-profile families."""

    h, k, _ = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
    if h == 0 and k == 0:
        return "specular", "two_theta"
    if h != 0 or k != 0:
        return "in_plane", "phi"
    return None


def _profile_half_span_deg(
    values: Sequence[float] | np.ndarray,
    *,
    min_half_span: float,
    max_half_span: float,
) -> float:
    """Choose one robust symmetric half-span for a fixed profile window."""

    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(min_half_span)
    try:
        span = float(np.percentile(finite, 98.0))
    except Exception:
        span = float(np.max(finite))
    if not np.isfinite(span) or span <= 0.0:
        span = float(np.max(finite))
    return float(np.clip(span, float(min_half_span), float(max_half_span)))


def _profile_bin_indices(
    values: Sequence[float] | np.ndarray,
    *,
    half_span_deg: float,
    bin_count: int,
) -> np.ndarray:
    """Return fixed symmetric profile-bin indices for one angular axis."""

    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=np.int64)
    if int(bin_count) <= 0 or not np.isfinite(float(half_span_deg)) or float(half_span_deg) <= 0.0:
        return out
    half_span = float(half_span_deg)
    valid = np.isfinite(arr) & (arr >= -half_span) & (arr <= half_span)
    if not np.any(valid):
        return out
    scaled = (arr[valid] + half_span) / max(2.0 * half_span, 1.0e-12) * float(bin_count)
    scaled = np.minimum(scaled, float(bin_count) - 1.0e-12)
    out[valid] = np.floor(scaled).astype(np.int64)
    return out


def _extract_profile_from_flat_image(
    flat_image: np.ndarray,
    roi: MosaicProfileROI,
) -> np.ndarray:
    """Extract one background-subtracted 1D profile from a flat image buffer."""

    values = np.asarray(flat_image[roi.flat_indices], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    profile = np.zeros_like(roi.signal_counts, dtype=np.float64)
    if np.any(roi.signal_mask):
        profile = np.bincount(
            roi.axis_bin_indices[roi.signal_mask],
            weights=values[roi.signal_mask],
            minlength=roi.signal_counts.size,
        ).astype(np.float64, copy=False)

    if np.any(roi.side_mask):
        side_sums = np.bincount(
            roi.axis_bin_indices[roi.side_mask],
            weights=values[roi.side_mask],
            minlength=roi.side_counts.size,
        ).astype(np.float64, copy=False)
        bg_level = np.full(
            roi.side_counts.shape,
            float(np.mean(values[roi.side_mask])),
            dtype=np.float64,
        )
        valid_side = roi.side_counts > 0
        bg_level[valid_side] = side_sums[valid_side] / roi.side_counts[valid_side]
    else:
        fallback = float(np.median(values)) if values.size else 0.0
        bg_level = np.full(roi.side_counts.shape, fallback, dtype=np.float64)

    corrected = profile - bg_level * roi.signal_counts.astype(np.float64)
    corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(corrected, 0.0, None)


def _estimate_pixel_size(params: Mapping[str, object]) -> float:
    provenance = _fit_space_pixel_size_provenance(params)
    try:
        return float(provenance.get("value", np.nan))
    except Exception:
        return float("nan")


def _fit_space_pixel_size_provenance(
    params: Mapping[str, object],
) -> Dict[str, float | str]:
    """Return the chosen fit-space pixel size together with its raw candidates."""

    def _raw_value(key: str) -> float:
        return _safe_float(params.get(key, np.nan))

    raw_pixel_size = _raw_value("pixel_size")
    raw_pixel_size_m = _raw_value("pixel_size_m")
    raw_debye_x = _raw_value("debye_x")
    raw_debye_y = _raw_value("debye_y")

    for key, value in (
        ("pixel_size", raw_pixel_size),
        ("pixel_size_m", raw_pixel_size_m),
        ("debye_x", raw_debye_x),
    ):
        if np.isfinite(value) and value > 0.0:
            return {
                "source": str(key),
                "value": float(value),
                "raw_pixel_size": float(raw_pixel_size),
                "raw_pixel_size_m": float(raw_pixel_size_m),
                "raw_debye_x": float(raw_debye_x),
                "raw_debye_y": float(raw_debye_y),
            }

    fallback = _safe_float(params.get("corto_detector", 1.0), 1.0) / 4096.0
    try:
        pixel_value = float(fallback)
    except Exception:
        pixel_value = float("nan")
    if not np.isfinite(pixel_value) or pixel_value <= 0.0:
        pixel_value = 1.0e-6
    return {
        "source": "fallback",
        "value": float(pixel_value),
        "raw_pixel_size": float(raw_pixel_size),
        "raw_pixel_size_m": float(raw_pixel_size_m),
        "raw_debye_x": float(raw_debye_x),
        "raw_debye_y": float(raw_debye_y),
    }


def _build_mosaic_profile_dataset_contexts(
    miller: np.ndarray,
    intensities: np.ndarray,
    image_size: int,
    params: Dict[str, object],
    dataset_specs: Sequence[Dict[str, object]],
    *,
    roi_half_width: int,
    build_geometry_fit_dataset_contexts: Callable[..., Sequence[object]],
    miller_key_from_row: Callable[[Sequence[float]], Optional[Tuple[int, int, int]]],
    normalized_hkl_key: Callable[[object], Optional[Tuple[int, int, int]]],
    measured_detector_anchor: Callable[[Mapping[str, object]], Tuple[object, object]],
    detector_pixels_to_fit_space: Callable[..., Tuple[np.ndarray, np.ndarray]],
) -> Tuple[List[MosaicProfileDatasetContext], List[Dict[str, object]]]:
    """Prepare fixed angular-profile windows for the sigma/gamma/eta fitter."""

    contexts = build_geometry_fit_dataset_contexts(
        miller,
        intensities,
        params,
        measured_peaks=None,
        experimental_image=None,
        dataset_specs=dataset_specs,
    )

    detector_distance = _safe_float(params.get("corto_detector", 0.0), 0.0)
    pixel_size = _estimate_pixel_size(params)
    centre = params.get("center")
    gamma_deg = _safe_float(params.get("gamma", 0.0), 0.0)
    Gamma_deg = _safe_float(params.get("Gamma", 0.0), 0.0)
    image_size = int(image_size)
    roi_half_width = max(int(roi_half_width), 1)
    prepared: List[MosaicProfileDatasetContext] = []
    rejected_rois: List[Dict[str, object]] = []

    for dataset_ctx in contexts:
        experimental_image = (
            np.asarray(dataset_ctx.experimental_image, dtype=np.float64)
            if dataset_ctx.experimental_image is not None
            else None
        )
        if experimental_image is None or experimental_image.size == 0:
            rejected_rois.append(
                {
                    "dataset_index": int(dataset_ctx.dataset_index),
                    "dataset_label": str(dataset_ctx.label),
                    "stage": "dataset_prep",
                    "reason": "missing_experimental_image",
                }
            )
            continue

        subset_miller = np.asarray(dataset_ctx.subset.miller, dtype=np.float64)
        subset_intensities = np.asarray(dataset_ctx.subset.intensities, dtype=np.float64)
        measured_entries = list(dataset_ctx.subset.measured_entries or [])
        hkl_to_reflection_index = {
            miller_key_from_row(row): int(idx)
            for idx, row in enumerate(subset_miller)
            if miller_key_from_row(row) is not None
        }
        rois: List[MosaicProfileROI] = []
        in_plane_count = 0
        specular_count = 0
        flat_experimental = np.asarray(experimental_image, dtype=np.float64).ravel()

        for measured_entry in measured_entries:
            if not isinstance(measured_entry, Mapping):
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "stage": "profile_prep",
                        "reason": "invalid_measured_entry",
                    }
                )
                continue

            hkl_key = normalized_hkl_key(measured_entry.get("hkl"))
            if hkl_key is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "stage": "profile_prep",
                        "reason": "missing_hkl",
                    }
                )
                continue

            family_info = _classify_mosaic_profile_family(hkl_key)
            if family_info is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "unsupported_peak_family",
                    }
                )
                continue
            family, axis_name = family_info

            reflection_index = hkl_to_reflection_index.get(hkl_key)
            if reflection_index is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "hkl_not_in_subset",
                    }
                )
                continue

            anchor, anchor_reason = measured_detector_anchor(measured_entry)
            if anchor is None:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": anchor_reason,
                    }
                )
                continue

            if not isinstance(anchor, (list, tuple, np.ndarray)) or len(anchor) < 2:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "invalid_anchor_shape",
                    }
                )
                continue

            center_col = _safe_float(anchor[0], 0.0)
            center_row = _safe_float(anchor[1], 0.0)
            row0 = max(int(math.floor(center_row)) - roi_half_width, 0)
            row1 = min(int(math.floor(center_row)) + roi_half_width + 1, image_size)
            col0 = max(int(math.floor(center_col)) - roi_half_width, 0)
            col1 = min(int(math.floor(center_col)) + roi_half_width + 1, image_size)
            if row1 <= row0 or col1 <= col0:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "degenerate_roi",
                    }
                )
                continue

            rows = np.arange(row0, row1, dtype=np.int64)
            cols = np.arange(col0, col1, dtype=np.int64)
            yy, xx = np.meshgrid(rows, cols, indexing="ij")
            flat_indices = (yy * image_size + xx).reshape(-1)
            two_theta_deg, phi_deg = detector_pixels_to_fit_space(
                xx.reshape(-1),
                yy.reshape(-1),
                center=centre,
                detector_distance=detector_distance,
                pixel_size=pixel_size,
                gamma_deg=gamma_deg,
                Gamma_deg=Gamma_deg,
            )
            peak_two_theta, peak_phi = detector_pixels_to_fit_space(
                np.asarray([center_col], dtype=np.float64),
                np.asarray([center_row], dtype=np.float64),
                center=centre,
                detector_distance=detector_distance,
                pixel_size=pixel_size,
                gamma_deg=gamma_deg,
                Gamma_deg=Gamma_deg,
            )
            if (
                peak_two_theta.size == 0
                or peak_phi.size == 0
                or not np.isfinite(float(peak_two_theta[0]))
                or not np.isfinite(float(peak_phi[0]))
            ):
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "invalid_peak_angles",
                    }
                )
                continue

            measured_two_theta_deg = float(peak_two_theta[0])
            measured_phi_deg = float(peak_phi[0])
            if family == "in_plane":
                axis_delta = _angular_difference_array_deg(phi_deg, measured_phi_deg)
                orth_delta = two_theta_deg - measured_two_theta_deg
                axis_half_span_deg = _profile_half_span_deg(
                    axis_delta,
                    min_half_span=0.35,
                    max_half_span=20.0,
                )
                orth_half_span_deg = _profile_half_span_deg(
                    orth_delta,
                    min_half_span=0.03,
                    max_half_span=5.0,
                )
                orthogonal_half_window_deg = float(
                    np.clip(
                        0.35 * orth_half_span_deg,
                        0.04,
                        max(orth_half_span_deg, 0.15),
                    )
                )
            else:
                axis_delta = two_theta_deg - measured_two_theta_deg
                orth_delta = _angular_difference_array_deg(phi_deg, measured_phi_deg)
                axis_half_span_deg = _profile_half_span_deg(
                    axis_delta,
                    min_half_span=0.15,
                    max_half_span=15.0,
                )
                orth_half_span_deg = _profile_half_span_deg(
                    orth_delta,
                    min_half_span=0.15,
                    max_half_span=15.0,
                )
                orthogonal_half_window_deg = float(
                    np.clip(
                        0.35 * orth_half_span_deg,
                        0.15,
                        max(orth_half_span_deg, 0.75),
                    )
                )

            finite_orth = orth_delta[np.isfinite(orth_delta)]
            max_abs_orth = float(np.max(np.abs(finite_orth))) if finite_orth.size else 0.0
            if not np.isfinite(max_abs_orth) or max_abs_orth <= 0.0:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "invalid_orthogonal_span",
                    }
                )
                continue

            orthogonal_half_window_deg = min(
                float(orthogonal_half_window_deg),
                max_abs_orth,
            )
            side_inner = min(max_abs_orth, 1.5 * orthogonal_half_window_deg)
            side_outer = min(
                max_abs_orth,
                max(side_inner + 0.05, 3.0 * orthogonal_half_window_deg),
            )

            axis_bin_count = 41
            axis_bin_indices = _profile_bin_indices(
                axis_delta,
                half_span_deg=axis_half_span_deg,
                bin_count=axis_bin_count,
            )
            valid = np.isfinite(axis_delta) & np.isfinite(orth_delta) & (axis_bin_indices >= 0)
            signal_mask = valid & (np.abs(orth_delta) <= orthogonal_half_window_deg)
            side_mask = (
                valid & (np.abs(orth_delta) >= side_inner) & (np.abs(orth_delta) <= side_outer)
            )
            if int(np.count_nonzero(signal_mask)) < 5:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "insufficient_support_pixels",
                    }
                )
                continue

            signal_counts = np.bincount(
                axis_bin_indices[signal_mask],
                minlength=axis_bin_count,
            ).astype(np.float64, copy=False)
            side_counts = np.bincount(
                axis_bin_indices[side_mask],
                minlength=axis_bin_count,
            ).astype(np.float64, copy=False)
            axis_bin_centers = np.linspace(
                -axis_half_span_deg,
                axis_half_span_deg,
                axis_bin_count,
                dtype=np.float64,
            )

            roi = MosaicProfileROI(
                dataset_index=int(dataset_ctx.dataset_index),
                dataset_label=str(dataset_ctx.label),
                reflection_index=int(reflection_index),
                hkl=hkl_key,
                family=str(family),
                axis_name=str(axis_name),
                center_row=float(center_row),
                center_col=float(center_col),
                row_bounds=(int(row0), int(row1)),
                col_bounds=(int(col0), int(col1)),
                flat_indices=np.asarray(flat_indices, dtype=np.int64),
                axis_bin_indices=np.asarray(axis_bin_indices, dtype=np.int64),
                signal_mask=np.asarray(signal_mask, dtype=bool),
                side_mask=np.asarray(side_mask, dtype=bool),
                signal_counts=np.asarray(signal_counts, dtype=np.float64),
                side_counts=np.asarray(side_counts, dtype=np.float64),
                axis_bin_centers=np.asarray(axis_bin_centers, dtype=np.float64),
                axis_half_span_deg=float(axis_half_span_deg),
                orthogonal_half_window_deg=float(orthogonal_half_window_deg),
                measured_two_theta_deg=float(measured_two_theta_deg),
                measured_phi_deg=float(measured_phi_deg),
                measured_profile=np.zeros(axis_bin_count, dtype=np.float64),
                measured_area=0.0,
                measured_shape_profile=np.zeros(axis_bin_count, dtype=np.float64),
            )
            measured_profile = _extract_profile_from_flat_image(flat_experimental, roi)
            measured_area = float(np.sum(measured_profile))
            if not np.isfinite(measured_area) or measured_area <= 0.0:
                rejected_rois.append(
                    {
                        "dataset_index": int(dataset_ctx.dataset_index),
                        "dataset_label": str(dataset_ctx.label),
                        "hkl": [int(v) for v in hkl_key],
                        "stage": "profile_prep",
                        "reason": "empty_measured_profile",
                    }
                )
                continue

            roi.measured_profile = np.asarray(measured_profile, dtype=np.float64)
            roi.measured_area = float(measured_area)
            roi.measured_shape_profile = (
                np.asarray(measured_profile, dtype=np.float64) / measured_area
            )
            rois.append(roi)
            if family == "in_plane":
                in_plane_count += 1
            else:
                specular_count += 1

        selected_reflection_indices = sorted({int(roi.reflection_index) for roi in rois})
        if selected_reflection_indices:
            selected_miller = np.asarray(
                subset_miller[selected_reflection_indices],
                dtype=np.float64,
            )
            selected_intensities = np.asarray(
                subset_intensities[selected_reflection_indices],
                dtype=np.float64,
            )
        else:
            selected_miller = np.empty((0, 3), dtype=np.float64)
            selected_intensities = np.empty((0,), dtype=np.float64)

        prepared.append(
            MosaicProfileDatasetContext(
                dataset_index=int(dataset_ctx.dataset_index),
                label=str(dataset_ctx.label),
                theta_initial=float(dataset_ctx.theta_initial),
                experimental_image=np.asarray(experimental_image, dtype=np.float64),
                miller=np.asarray(selected_miller, dtype=np.float64),
                intensities=np.asarray(selected_intensities, dtype=np.float64),
                rois=rois,
                measured_peak_count=int(len(measured_entries)),
                in_plane_roi_count=int(in_plane_count),
                specular_roi_count=int(specular_count),
            )
        )

    return prepared, rejected_rois


__all__ = [
    "CenteredMosaicProfileComparison",
    "LorentzianGaussianProfileFit",
    "MosaicPhiProfile",
    "MosaicPointPair",
    "MosaicProfileDatasetContext",
    "MosaicProfileROI",
    "compare_centered_phi_profiles",
    "fit_lorentzian_plus_gaussian_profile",
    "fit_mosaic_parameters_from_centered_phi_profiles",
    "integrate_selected_qr_phi_profiles",
    "lorentzian_plus_gaussian_profile",
    "pair_selected_qr_and_background_points",
    "stack_centered_phi_profiles",
    "focus_mosaic_profile_dataset_specs",
    "_angular_difference_array_deg",
    "_build_mosaic_profile_dataset_contexts",
    "_classify_mosaic_profile_family",
    "_estimate_pixel_size",
    "_extract_profile_from_flat_image",
    "_fit_space_pixel_size_provenance",
    "_miller_key_from_row",
    "_mosaic_profile_entry_priority",
    "_mosaic_profile_group_key",
    "_mosaic_profile_intensity_lookup",
    "_mosaic_profile_peak_score",
    "_normalized_hkl_key",
    "_normalized_q_group_key",
    "_profile_bin_indices",
    "_profile_half_span_deg",
]
