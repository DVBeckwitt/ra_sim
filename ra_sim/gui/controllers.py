"""GUI controller helpers."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ra_sim.config import get_instrument_config
from ra_sim.utils.stacking_fault import (
    normalize_phi_l_divisor,
    normalize_phase_delta_expression,
    validate_phase_delta_expression,
)

from .state import (
    AppState,
    GeometryFitDatasetCacheState,
    BraggQrManagerState,
    GeometryFitHistoryState,
    GeometryPreviewOverlayState,
    GeometryPreviewState,
    GeometryQGroupState,
    ManualGeometryState,
    ManualGeometryUndoSnapshot,
    SimulationRuntimeState,
)


BRAGG_QR_L_KEY_SCALE = 1_000_000
BRAGG_QR_L_INVALID_KEY = int(np.iinfo(np.int64).min)
SF_PRUNE_BIAS_MIN = -2.0
SF_PRUNE_BIAS_MAX = 3.0
DEFAULT_ROD_L_STEP = 0.01
ROD_POINTS_PER_GZ_MIN = 10
ROD_POINTS_PER_GZ_MAX = 2000
_SF_PRUNE_RETAIN_BASE = 0.9997
_SF_PRUNE_RETAIN_MAX = 0.99998
_SF_PRUNE_RETAIN_MIN = 0.90
_SF_PRUNE_REL_FLOOR_BASE = 3.0e-5
_SF_PRUNE_REL_FLOOR_MIN = 1.0e-8
_SF_PRUNE_REL_FLOOR_MAX = 8.0e-2
_SF_PRUNE_MIN_KEEP_BASE = 18


def clear_tk_after_token(root: Any, token: Any) -> None:
    """Cancel one Tk `after` token if it is still schedulable."""

    cancel = getattr(root, "after_cancel", None)
    if token is None or not callable(cancel):
        return
    try:
        cancel(token)
    except Exception:
        pass


def parse_sampling_count(
    raw_value: object,
    fallback: object,
    *,
    minimum: int = 1,
) -> int:
    """Coerce one sampling-count value to a positive integer."""

    try:
        parsed = int(round(float(str(raw_value).strip().replace(",", ""))))
    except (TypeError, ValueError):
        try:
            parsed = int(round(float(str(fallback).strip().replace(",", ""))))
        except (TypeError, ValueError):
            parsed = int(minimum)
    return max(int(minimum), parsed)


def normalize_sample_count(
    raw_value: object,
    fallback: object,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Normalize one sample-count value to an integer slider range."""

    normalized = parse_sampling_count(raw_value, fallback, minimum=max(1, int(minimum)))
    if maximum is not None:
        upper = int(maximum)
        lower = max(1, int(minimum))
        if upper < lower:
            lower, upper = upper, lower
        normalized = int(np.clip(normalized, lower, upper))
    return int(normalized)


def parse_sampling_float(
    raw_value: object,
    fallback: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Coerce one numeric sampling value to a finite float."""

    try:
        parsed = float(str(raw_value).strip().replace(",", ""))
    except (TypeError, ValueError):
        try:
            parsed = float(str(fallback).strip().replace(",", ""))
        except (TypeError, ValueError):
            parsed = 0.0

    if not np.isfinite(parsed):
        try:
            fallback_value = float(fallback)
        except (TypeError, ValueError):
            fallback_value = 0.0
        parsed = fallback_value if np.isfinite(fallback_value) else 0.0
    if minimum is not None:
        parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return float(parsed)


def normalize_sampling_resolution_choice(
    resolution_value: object,
    *,
    allowed_options: Sequence[object],
    fallback: object,
) -> str:
    """Normalize one sampling-resolution choice against the allowed labels."""

    allowed = [str(option) for option in allowed_options]
    value_text = str(resolution_value) if resolution_value is not None else ""
    if value_text in allowed:
        return value_text

    fallback_text = str(fallback)
    if fallback_text in allowed:
        return fallback_text
    if allowed:
        return allowed[0]
    return ""


def resolve_sampling_count(
    resolution_value: object,
    *,
    custom_option: object,
    custom_value: object,
    preset_counts: Mapping[object, object],
    fallback_resolution: object,
    fallback_count: object,
) -> int:
    """Return the effective sample count for one resolution selection."""

    normalized = str(resolution_value) if resolution_value is not None else ""
    custom_label = str(custom_option)
    if normalized == custom_label:
        return parse_sampling_count(custom_value, fallback_count)

    fallback_label = str(fallback_resolution)
    preset_value = preset_counts.get(normalized)
    if preset_value is None:
        preset_value = preset_counts.get(fallback_label, fallback_count)
    return parse_sampling_count(preset_value, fallback_count)


def format_sampling_resolution_summary(
    resolution_value: object,
    *,
    custom_option: object,
    custom_value: object,
    preset_counts: Mapping[object, object],
    fallback_resolution: object,
    fallback_count: object,
) -> str:
    """Format the GUI sampling summary label for the current selection."""

    custom_label = str(custom_option)
    normalized = str(resolution_value) if resolution_value is not None else ""
    count = resolve_sampling_count(
        normalized,
        custom_option=custom_label,
        custom_value=custom_value,
        preset_counts=preset_counts,
        fallback_resolution=fallback_resolution,
        fallback_count=fallback_count,
    )
    suffix = " (custom)" if normalized == custom_label else ""
    return f"{count:,} samples{suffix}" if count >= 1000 else f"{count} samples{suffix}"


def format_sampling_count_summary(sample_count: object) -> str:
    """Format one numeric sample count for GUI display."""

    count = parse_sampling_count(sample_count, 1)
    return f"{count:,} samples" if count >= 1000 else f"{count} samples"


def stratified_total_ray_count(sample_counts: Sequence[object]) -> int:
    """Return the total Cartesian-product ray count for one sample-count list."""

    total = 1
    for count in sample_counts:
        total *= parse_sampling_count(count, 1)
    return int(max(1, total))


def format_stratified_ray_count_summary(sample_counts: Sequence[object]) -> str:
    """Format the live total-ray label for stratified Gaussian sampling."""

    counts = [parse_sampling_count(count, 1) for count in sample_counts]
    total = stratified_total_ray_count(counts)
    factors = " x ".join(str(count) for count in counts)
    return f"Total rays: {total:,} = {factors}"


def format_stratified_ray_count_warning(
    sample_counts: Sequence[object],
    *,
    warning_threshold: int,
) -> str:
    """Return a guardrail warning when stratified ray counts grow large."""

    total = stratified_total_ray_count(sample_counts)
    if total < int(max(1, warning_threshold)):
        return ""
    return (
        f"Warning: {total:,} rays may make updates slow or memory-heavy."
    )


def default_rod_points_per_gz(
    c_lattice: object,
    *,
    l_step: float = DEFAULT_ROD_L_STEP,
    fallback: int = 500,
) -> int:
    """Return the legacy-equivalent rod density in points per unit ``Gz``."""

    try:
        c_value = float(c_lattice)
        l_step_value = float(l_step)
    except (TypeError, ValueError):
        c_value = float("nan")
        l_step_value = float("nan")

    if (
        np.isfinite(c_value)
        and c_value > 0.0
        and np.isfinite(l_step_value)
        and l_step_value > 0.0
    ):
        points = int(round(c_value / (2.0 * np.pi * l_step_value)))
    else:
        points = int(fallback)
    return int(np.clip(points, ROD_POINTS_PER_GZ_MIN, ROD_POINTS_PER_GZ_MAX))


def normalize_rod_points_per_gz(
    raw_value: object,
    fallback: object,
    *,
    minimum: int = ROD_POINTS_PER_GZ_MIN,
    maximum: int = ROD_POINTS_PER_GZ_MAX,
) -> int:
    """Normalize one rod-density value to a bounded positive integer."""

    parsed = parse_sampling_count(raw_value, fallback, minimum=max(1, int(minimum)))
    lower = int(minimum)
    upper = int(maximum)
    if upper < lower:
        lower, upper = upper, lower
    return int(np.clip(parsed, lower, upper))


def rod_l_step_from_points_per_gz(
    points_per_gz: object,
    c_lattice: object,
    *,
    fallback_points: object = 500,
    fallback_l_step: float = DEFAULT_ROD_L_STEP,
) -> float:
    """Return the ``L``-grid spacing that matches one rod-density setting."""

    normalized_points = normalize_rod_points_per_gz(
        points_per_gz,
        fallback_points,
    )
    try:
        c_value = float(c_lattice)
    except (TypeError, ValueError):
        c_value = float("nan")
    if not np.isfinite(c_value) or c_value <= 0.0:
        return float(fallback_l_step)
    return float(c_value / (2.0 * np.pi * float(normalized_points)))


def rod_gz_max_from_two_theta(
    two_theta_max: object,
    lambda_angstrom: object,
) -> float:
    """Return the largest accessible ``Gz`` span for the specular rod."""

    try:
        two_theta_value = float(two_theta_max)
        lambda_value = float(lambda_angstrom)
    except (TypeError, ValueError):
        return 0.0
    if (
        not np.isfinite(two_theta_value)
        or not np.isfinite(lambda_value)
        or lambda_value <= 0.0
    ):
        return 0.0
    half_angle_rad = np.radians(max(two_theta_value, 0.0) / 2.0)
    return float((4.0 * np.pi / lambda_value) * np.sin(half_angle_rad))


def longest_rod_point_count(
    points_per_gz: object,
    *,
    two_theta_max: object,
    lambda_angstrom: object,
) -> int:
    """Return the positive-point count on the longest rod in the active window."""

    normalized_points = normalize_rod_points_per_gz(
        points_per_gz,
        default_rod_points_per_gz(1.0),
    )
    gz_max = rod_gz_max_from_two_theta(two_theta_max, lambda_angstrom)
    if not np.isfinite(gz_max) or gz_max <= 0.0:
        return 0
    return int(max(0.0, np.floor(gz_max * float(normalized_points) + 0.5)))


def format_rod_points_per_gz(points_per_gz: object) -> str:
    """Format one rod-density value for the sampling/optics panel."""

    normalized_points = normalize_rod_points_per_gz(
        points_per_gz,
        default_rod_points_per_gz(1.0),
    )
    return f"{normalized_points:,} / Gz"


def format_longest_rod_point_summary(
    points_per_gz: object,
    *,
    two_theta_max: object,
    lambda_angstrom: object,
) -> str:
    """Format the live summary for the longest rod in the active HT window."""

    point_count = longest_rod_point_count(
        points_per_gz,
        two_theta_max=two_theta_max,
        lambda_angstrom=lambda_angstrom,
    )
    gz_max = rod_gz_max_from_two_theta(two_theta_max, lambda_angstrom)
    if np.isfinite(gz_max) and gz_max > 0.0:
        return f"Longest rod: {point_count:,} points (Gz max {gz_max:.3f})"
    return f"Longest rod: {point_count:,} points"


def normalize_finite_stack_layer_count(
    raw_value: object,
    fallback: object,
) -> int:
    """Normalize one finite-stack layer-count input to a positive integer."""

    return parse_sampling_count(raw_value, fallback)


def format_finite_stack_layer_count(value: object) -> str:
    """Format one finite-stack layer count for entry display."""

    return str(normalize_finite_stack_layer_count(value, 1))


def normalize_finite_stack_phase_delta_expression(
    raw_value: object,
    *,
    fallback: object,
) -> str:
    """Normalize and validate one finite-stack phase-delta expression."""

    normalized = normalize_phase_delta_expression(raw_value, fallback=str(fallback))
    return validate_phase_delta_expression(normalized)


def normalize_finite_stack_phi_l_divisor(
    raw_value: object,
    *,
    fallback: object,
) -> float:
    """Normalize one finite-stack phi-L divisor to a positive finite float."""

    return normalize_phi_l_divisor(raw_value, fallback=float(fallback))


def format_finite_stack_phi_l_divisor(value: object) -> str:
    """Format one finite-stack phi-L divisor for entry display."""

    normalized = normalize_finite_stack_phi_l_divisor(value, fallback=1.0)
    return f"{normalized:.6g}"


def clamp_site_occupancy_values(
    values: Sequence[object],
    *,
    fallback_values: Sequence[object] | None = None,
) -> list[float]:
    """Clamp occupancy values to finite fractions in the inclusive [0, 1] range."""

    fallbacks = list(fallback_values or [])
    clamped: list[float] = []
    for idx, raw_value in enumerate(values):
        try:
            normalized = float(raw_value)
        except (TypeError, ValueError):
            fallback = fallbacks[idx] if idx < len(fallbacks) else 1.0
            try:
                normalized = float(fallback)
            except (TypeError, ValueError):
                normalized = 1.0
        if not math.isfinite(normalized):
            fallback = fallbacks[idx] if idx < len(fallbacks) else 1.0
            try:
                normalized = float(fallback)
            except (TypeError, ValueError):
                normalized = 1.0
            if not math.isfinite(normalized):
                normalized = 1.0
        clamped.append(min(1.0, max(0.0, float(normalized))))
    return clamped


def normalize_stacking_weight_values(values: Sequence[object]) -> list[float]:
    """Normalize stacking-disorder weight percentages to fractional weights."""

    normalized: list[float] = []
    for raw_value in values:
        try:
            weight = float(raw_value)
        except (TypeError, ValueError):
            weight = 0.0
        if not math.isfinite(weight):
            weight = 0.0
        normalized.append(float(weight))

    weight_sum = float(sum(normalized)) or 1.0
    return [float(weight) / weight_sum for weight in normalized]


def combine_cif_weighted_intensities(
    primary_intensities: Sequence[object],
    secondary_intensities: Sequence[object],
    *,
    weight1: object,
    weight2: object,
) -> np.ndarray:
    """Combine two CIF intensity arrays with raw slider weights and renormalize."""

    try:
        w1 = float(weight1)
    except (TypeError, ValueError):
        w1 = 1.0
    try:
        w2 = float(weight2)
    except (TypeError, ValueError):
        w2 = 0.0
    if not math.isfinite(w1):
        w1 = 1.0
    if not math.isfinite(w2):
        w2 = 0.0

    primary = np.asarray(primary_intensities, dtype=np.float64)
    secondary = np.asarray(secondary_intensities, dtype=np.float64)
    combined = w1 * primary + w2 * secondary
    max_intensity = float(np.max(combined)) if combined.size else 0.0
    if max_intensity > 0.0:
        combined = combined * (100.0 / max_intensity)
    return combined


def ensure_display_intensity_range(
    min_value: object,
    max_value: object,
) -> tuple[float, float]:
    """Return a finite ascending display-intensity range."""

    try:
        min_val = float(min_value)
    except (TypeError, ValueError):
        min_val = 0.0
    try:
        max_val = float(max_value)
    except (TypeError, ValueError):
        max_val = max(min_val + 1.0, 1.0)

    if not math.isfinite(min_val):
        min_val = 0.0
    if not math.isfinite(max_val):
        max_val = max(min_val + 1.0, 1.0)
    if max_val <= min_val:
        max_val = min_val + max(abs(min_val) * 1e-3, 1.0)
    return min_val, max_val


def normalize_display_scale_factor(
    value: object,
    *,
    fallback: object,
) -> float:
    """Normalize one display scale-factor input to a finite non-negative float."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        try:
            normalized = float(fallback)
        except (TypeError, ValueError):
            normalized = 1.0
    if not math.isfinite(normalized):
        try:
            normalized = float(fallback)
        except (TypeError, ValueError):
            normalized = 1.0
        if not math.isfinite(normalized):
            normalized = 1.0
    return max(0.0, float(normalized))


def clip_structure_factor_prune_bias(
    value: object,
    *,
    fallback: object,
    minimum: float,
    maximum: float,
) -> float:
    """Clamp one SF-pruning bias input to the supported finite range."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = 0.0
    return float(min(max(float(normalized), float(minimum)), float(maximum)))


def clip_solve_q_steps(
    value: object,
    *,
    fallback: object,
    minimum: int,
    maximum: int,
) -> int:
    """Clamp one solve-q interval-count input to the supported integer range."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = float(minimum)
    rounded = int(round(float(normalized)))
    return int(min(max(rounded, int(minimum)), int(maximum)))


def clip_solve_q_rel_tol(
    value: object,
    *,
    fallback: object,
    minimum: float,
    maximum: float,
) -> float:
    """Clamp one solve-q relative tolerance input to the supported range."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = float(minimum)
    return float(min(max(float(normalized), float(minimum)), float(maximum)))


def normalize_solve_q_mode_label(value: object) -> str:
    """Normalize one solve-q mode input to the supported label set."""

    if not isinstance(value, (str, bytes)) and not isinstance(value, bool):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value is not None and math.isfinite(numeric_value):
            return "uniform" if int(round(float(numeric_value))) == 0 else "adaptive"

    text = str(value).strip().lower()
    if text in {"uniform", "fast", "0"}:
        return "uniform"
    if text in {"adaptive", "robust", "1"}:
        return "adaptive"
    return "uniform"


def solve_q_mode_flag_from_label(
    label: object,
    *,
    uniform_flag: int,
    adaptive_flag: int,
) -> int:
    """Convert one normalized solve-q mode label into a runtime flag."""

    return (
        int(uniform_flag)
        if normalize_solve_q_mode_label(label) == "uniform"
        else int(adaptive_flag)
    )


def normalize_bragg_qr_source_label(source_label: str | None) -> str:
    """Normalize one Bragg-Qr source label to ``primary`` or ``secondary``."""

    return (
        "secondary"
        if str(source_label).strip().lower() == "secondary"
        else "primary"
    )


def bragg_qr_l_value_to_key(l_value: object) -> int:
    """Convert one L float into the stable Bragg-Qr integer key."""

    try:
        val = float(l_value)
    except (TypeError, ValueError):
        return BRAGG_QR_L_INVALID_KEY
    if not np.isfinite(val):
        return BRAGG_QR_L_INVALID_KEY
    return int(np.rint(val * BRAGG_QR_L_KEY_SCALE))


def bragg_qr_l_key_to_value(l_key: object) -> float:
    """Convert one stored Bragg-Qr L key back into its float value."""

    try:
        key = int(l_key)
    except (TypeError, ValueError):
        return float("nan")
    if key == BRAGG_QR_L_INVALID_KEY:
        return float("nan")
    return float(key / BRAGG_QR_L_KEY_SCALE)


def bragg_qr_l_keys_from_l_array(l_vals: np.ndarray) -> np.ndarray:
    """Vectorize one L array into stable Bragg-Qr integer keys."""

    arr = np.asarray(l_vals, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, BRAGG_QR_L_INVALID_KEY, dtype=np.int64)
    finite_mask = np.isfinite(arr)
    if np.any(finite_mask):
        out[finite_mask] = np.rint(arr[finite_mask] * BRAGG_QR_L_KEY_SCALE).astype(
            np.int64,
            copy=False,
        )
    return out


def bragg_qr_m_indices_from_miller_array(
    miller_arr: np.ndarray,
    *,
    unique: bool = False,
) -> np.ndarray:
    """Compute Bragg-Qr ``m = h^2 + hk + k^2`` indices from Miller rows."""

    arr = np.asarray(miller_arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.empty((0,), dtype=np.int64)

    finite_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    if not np.any(finite_mask):
        return np.empty((0,), dtype=np.int64)

    hk_rows = arr[finite_mask, :2]
    h_vals = np.rint(hk_rows[:, 0]).astype(np.int64, copy=False)
    k_vals = np.rint(hk_rows[:, 1]).astype(np.int64, copy=False)
    m_vals = h_vals * h_vals + h_vals * k_vals + k_vals * k_vals
    if unique:
        return np.unique(m_vals)
    return m_vals


def copy_bragg_qr_dict(qr_dict: dict | None) -> dict[int, dict[str, object]]:
    """Clone one Bragg-Qr dictionary into plain ``numpy`` backed arrays."""

    out: dict[int, dict[str, object]] = {}
    if not isinstance(qr_dict, dict):
        return out

    for m_raw, data in qr_dict.items():
        try:
            m_idx = int(m_raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        out[m_idx] = {
            "L": np.asarray(data.get("L", []), dtype=np.float64).copy(),
            "I": np.asarray(data.get("I", []), dtype=np.float64).copy(),
            "hk": tuple(data.get("hk", (0, 0))),
            "deg": int(data.get("deg", 1)),
        }
    return out


def structure_factor_prune_profile_from_bias(
    bias: object,
) -> tuple[float, float, int, int]:
    """Return the adaptive pruning profile for one SF-prune bias value."""

    bias_clipped = clip_structure_factor_prune_bias(
        bias,
        fallback=0.0,
        minimum=SF_PRUNE_BIAS_MIN,
        maximum=SF_PRUNE_BIAS_MAX,
    )

    if bias_clipped >= 0.0:
        extra_aggressive = max(0.0, bias_clipped - 1.0)
        retain_fraction = (
            _SF_PRUNE_RETAIN_BASE
            - 0.0045 * bias_clipped
            - 0.0200 * (extra_aggressive**1.35)
        )
    else:
        retain_fraction = _SF_PRUNE_RETAIN_BASE + 0.00028 * ((-bias_clipped) ** 1.1)
    retain_fraction = float(
        np.clip(retain_fraction, _SF_PRUNE_RETAIN_MIN, _SF_PRUNE_RETAIN_MAX)
    )

    extra_aggressive = max(0.0, bias_clipped - 1.0)
    rel_floor_exp = (1.8 * bias_clipped) + (2.2 * extra_aggressive)
    rel_floor = _SF_PRUNE_REL_FLOOR_BASE * (10.0**rel_floor_exp)
    rel_floor = float(np.clip(rel_floor, _SF_PRUNE_REL_FLOOR_MIN, _SF_PRUNE_REL_FLOOR_MAX))

    min_keep = int(
        round(
            _SF_PRUNE_MIN_KEEP_BASE
            - 8.0 * bias_clipped
            - 10.0 * extra_aggressive
        )
    )
    min_keep = int(max(3, min_keep))

    neighbor_span = 1 if bias_clipped <= 0.7 else 0
    return retain_fraction, rel_floor, min_keep, neighbor_span


def prune_l_intensity_curve(
    l_vals: np.ndarray,
    i_vals: np.ndarray,
    *,
    bias: object,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Prune one diffuse rod L/intensity curve using the configured SF bias."""

    l_arr = np.asarray(l_vals, dtype=np.float64).reshape(-1)
    i_arr = np.asarray(i_vals, dtype=np.float64).reshape(-1)
    row_count = min(l_arr.shape[0], i_arr.shape[0])
    if row_count <= 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            0,
            0,
        )

    l_arr = l_arr[:row_count]
    i_arr = i_arr[:row_count]
    finite_mask = np.isfinite(l_arr) & np.isfinite(i_arr) & (i_arr > 0.0)
    if not np.any(finite_mask):
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            row_count,
            0,
        )

    l_valid = l_arr[finite_mask]
    i_valid = i_arr[finite_mask]
    total_count = int(i_valid.shape[0])
    if total_count <= 8:
        return l_valid.copy(), i_valid.copy(), total_count, total_count

    retain_fraction, rel_floor, min_keep, neighbor_span = (
        structure_factor_prune_profile_from_bias(bias)
    )
    order = np.argsort(i_valid)[::-1]
    keep_mask = np.zeros(total_count, dtype=bool)

    top_n = min(total_count, max(1, min_keep))
    keep_mask[order[:top_n]] = True

    i_max = float(i_valid[order[0]])
    if i_max > 0.0:
        keep_mask |= i_valid >= (i_max * rel_floor)

    total_mass = float(np.sum(i_valid))
    if total_mass > 0.0:
        target_mass = retain_fraction * total_mass
        cum_mass = np.cumsum(i_valid[order])
        mass_n = int(np.searchsorted(cum_mass, target_mass, side="left")) + 1
        keep_mask[order[:mass_n]] = True

    if neighbor_span > 0:
        expanded_mask = keep_mask.copy()
        for delta in range(1, neighbor_span + 1):
            expanded_mask[:-delta] |= keep_mask[delta:]
            expanded_mask[delta:] |= keep_mask[:-delta]
        keep_mask = expanded_mask

    keep_idx = np.nonzero(keep_mask)[0]
    if keep_idx.size == 0:
        keep_idx = order[:1]

    keep_idx = np.sort(keep_idx)
    kept_count = int(keep_idx.size)
    return (
        l_valid[keep_idx].copy(),
        i_valid[keep_idx].copy(),
        total_count,
        kept_count,
    )


def prune_reflection_rows(
    miller_arr: np.ndarray,
    intens_arr: np.ndarray,
    *,
    bias: object,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Prune one HKL/intensity array using the configured SF bias."""

    arr = np.asarray(miller_arr, dtype=np.float64)
    intens = np.asarray(intens_arr, dtype=np.float64).reshape(-1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            0,
            0,
        )

    row_count = min(arr.shape[0], intens.shape[0])
    if row_count <= 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            0,
            0,
        )

    arr = arr[:row_count, :]
    intens = intens[:row_count]
    finite_mask = (
        np.isfinite(arr[:, 0])
        & np.isfinite(arr[:, 1])
        & np.isfinite(arr[:, 2])
        & np.isfinite(intens)
        & (intens > 0.0)
    )
    if not np.any(finite_mask):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            row_count,
            0,
        )

    arr_valid = arr[finite_mask, :]
    intens_valid = intens[finite_mask]
    total_count = int(intens_valid.shape[0])
    if total_count <= 8:
        return arr_valid.copy(), intens_valid.copy(), total_count, total_count

    retain_fraction, rel_floor, min_keep, _ = structure_factor_prune_profile_from_bias(
        bias
    )
    order = np.argsort(intens_valid)[::-1]
    keep_mask = np.zeros(total_count, dtype=bool)

    top_n = min(total_count, max(1, min_keep))
    keep_mask[order[:top_n]] = True

    i_max = float(intens_valid[order[0]])
    if i_max > 0.0:
        keep_mask |= intens_valid >= (i_max * rel_floor)

    total_mass = float(np.sum(intens_valid))
    if total_mass > 0.0:
        target_mass = retain_fraction * total_mass
        cum_mass = np.cumsum(intens_valid[order])
        mass_n = int(np.searchsorted(cum_mass, target_mass, side="left")) + 1
        keep_mask[order[:mass_n]] = True

    keep_idx = np.nonzero(keep_mask)[0]
    if keep_idx.size == 0:
        keep_idx = order[:1]
    keep_idx = np.sort(keep_idx)
    kept_count = int(keep_idx.size)

    return (
        arr_valid[keep_idx, :].copy(),
        intens_valid[keep_idx].copy(),
        total_count,
        kept_count,
    )


def clamp_slider_value_to_bounds(
    value: object,
    *,
    lower_bound: object,
    upper_bound: object,
    fallback: object,
) -> float:
    """Clamp one slider value to finite ordered bounds."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = float(fallback)
    if not math.isfinite(normalized):
        normalized = 0.0

    try:
        lower = float(lower_bound)
    except (TypeError, ValueError):
        lower = normalized
    try:
        upper = float(upper_bound)
    except (TypeError, ValueError):
        upper = normalized
    if not math.isfinite(lower):
        lower = normalized
    if not math.isfinite(upper):
        upper = normalized
    if upper < lower:
        lower, upper = upper, lower
    return float(min(max(float(normalized), lower), upper))


def build_initial_state() -> AppState:
    """Build the initial GUI state snapshot from current configuration."""

    instrument_cfg = get_instrument_config()
    detector_cfg = instrument_cfg.get("instrument", {}).get("detector", {})
    image_size = int(detector_cfg.get("image_size", 3000))
    return AppState(
        instrument_config=instrument_cfg,
        image_size=image_size,
    )


def launch_gui(*, write_excel_flag: bool | None = None) -> Any:
    """Launch the full GUI application."""

    from . import app

    return app.main(write_excel_flag=write_excel_flag)


def replace_manual_geometry_pairs_by_background(
    state: ManualGeometryState,
    pairs_by_background: Mapping[object, Sequence[dict[str, object]]] | None,
) -> dict[int, list[dict[str, object]]]:
    """Replace the stored manual-geometry pair map in place."""

    normalized: dict[int, list[dict[str, object]]] = {}
    for raw_index, raw_entries in (pairs_by_background or {}).items():
        try:
            index = int(raw_index)
        except Exception:
            continue
        entries = [
            copy.deepcopy(dict(entry))
            for entry in raw_entries or ()
            if isinstance(entry, dict)
        ]
        if entries:
            normalized[index] = entries

    state.pairs_by_background.clear()
    state.pairs_by_background.update(normalized)
    return state.pairs_by_background


def replace_manual_geometry_pick_session(
    state: ManualGeometryState,
    pick_session: Mapping[str, object] | None,
) -> dict[str, object]:
    """Replace the active manual-geometry pick session in place."""

    normalized = (
        copy.deepcopy(dict(pick_session))
        if isinstance(pick_session, Mapping)
        else {}
    )
    state.pick_session.clear()
    state.pick_session.update(normalized)
    return state.pick_session


def clear_manual_geometry_undo_stack(state: ManualGeometryState) -> None:
    """Discard all saved manual-geometry undo history."""

    state.undo_stack.clear()


def build_manual_geometry_undo_snapshot(
    state: ManualGeometryState,
) -> ManualGeometryUndoSnapshot:
    """Return a deep copy of manual-geometry state for undo restoration."""

    return ManualGeometryUndoSnapshot(
        pairs_by_background=copy.deepcopy(dict(state.pairs_by_background)),
        pick_session=copy.deepcopy(dict(state.pick_session)),
    )


def push_manual_geometry_undo_state(
    state: ManualGeometryState,
    *,
    limit: int,
) -> None:
    """Push the current manual-geometry state onto the undo stack."""

    state.undo_stack.append(build_manual_geometry_undo_snapshot(state))
    max_items = max(1, int(limit))
    if len(state.undo_stack) > max_items:
        del state.undo_stack[:-max_items]


def restore_last_manual_geometry_undo_state(
    state: ManualGeometryState,
) -> ManualGeometryUndoSnapshot | None:
    """Restore the most recent manual-geometry undo snapshot in place."""

    if not state.undo_stack:
        return None

    snapshot = state.undo_stack.pop()
    replace_manual_geometry_pairs_by_background(
        state,
        snapshot.pairs_by_background,
    )
    replace_manual_geometry_pick_session(
        state,
        snapshot.pick_session,
    )
    return snapshot


def clear_geometry_fit_history(state: GeometryFitHistoryState) -> None:
    """Discard all geometry-fit undo/redo history and overlay state."""

    state.undo_stack.clear()
    state.redo_stack.clear()
    state.last_overlay_state = None


def clear_geometry_fit_dataset_cache(state: GeometryFitDatasetCacheState) -> None:
    """Discard the cached successful geometry-fit dataset bundle."""

    state.payload = None


def replace_geometry_fit_dataset_cache(
    state: GeometryFitDatasetCacheState,
    payload: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
) -> dict[str, object] | None:
    """Replace the cached successful geometry-fit dataset bundle."""

    if payload is None:
        state.payload = None
    else:
        state.payload = copy_state_value(dict(payload))
    return state.payload


def replace_geometry_fit_last_overlay_state(
    state: GeometryFitHistoryState,
    overlay_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
) -> dict[str, object] | None:
    """Replace the remembered geometry-fit overlay state."""

    if overlay_state is None:
        state.last_overlay_state = None
    else:
        state.last_overlay_state = copy_state_value(dict(overlay_state))
    return state.last_overlay_state


def push_geometry_fit_undo_state(
    state: GeometryFitHistoryState,
    fit_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Push one state onto the geometry-fit undo stack and clear redo history."""

    if not isinstance(fit_state, Mapping):
        return
    state.undo_stack.append(copy_state_value(dict(fit_state)))
    max_items = max(1, int(limit))
    if len(state.undo_stack) > max_items:
        del state.undo_stack[:-max_items]
    state.redo_stack.clear()


def push_geometry_fit_redo_state(
    state: GeometryFitHistoryState,
    fit_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Push one state onto the geometry-fit redo stack."""

    if not isinstance(fit_state, Mapping):
        return
    state.redo_stack.append(copy_state_value(dict(fit_state)))
    max_items = max(1, int(limit))
    if len(state.redo_stack) > max_items:
        del state.redo_stack[:-max_items]


def peek_last_geometry_fit_undo_state(
    state: GeometryFitHistoryState,
    *,
    copy_state_value: Any = copy.deepcopy,
) -> dict[str, object] | None:
    """Return a detached copy of the most recent geometry-fit undo state."""

    if not state.undo_stack:
        return None
    return copy_state_value(state.undo_stack[-1])


def peek_last_geometry_fit_redo_state(
    state: GeometryFitHistoryState,
    *,
    copy_state_value: Any = copy.deepcopy,
) -> dict[str, object] | None:
    """Return a detached copy of the most recent geometry-fit redo state."""

    if not state.redo_stack:
        return None
    return copy_state_value(state.redo_stack[-1])


def commit_geometry_fit_undo(
    state: GeometryFitHistoryState,
    current_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Finalize a successful geometry-fit undo transition."""

    if state.undo_stack:
        state.undo_stack.pop()
    push_geometry_fit_redo_state(
        state,
        current_state,
        copy_state_value=copy_state_value,
        limit=limit,
    )


def commit_geometry_fit_redo(
    state: GeometryFitHistoryState,
    current_state: Mapping[str, object] | None,
    *,
    copy_state_value: Any = copy.deepcopy,
    limit: int,
) -> None:
    """Finalize a successful geometry-fit redo transition."""

    if state.redo_stack:
        state.redo_stack.pop()
    if isinstance(current_state, Mapping):
        state.undo_stack.append(copy_state_value(dict(current_state)))
        max_items = max(1, int(limit))
        if len(state.undo_stack) > max_items:
            del state.undo_stack[:-max_items]


def replace_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
    excluded_q_groups: Sequence[object] | None,
) -> set[tuple[object, ...]]:
    """Replace the excluded Qr/Qz group key set in place."""

    normalized: set[tuple[object, ...]] = set()
    for raw_key in excluded_q_groups or ():
        if isinstance(raw_key, tuple):
            normalized.add(raw_key)
        elif isinstance(raw_key, list):
            normalized.add(tuple(raw_key))

    state.excluded_q_groups.clear()
    state.excluded_q_groups.update(normalized)
    return state.excluded_q_groups


def retain_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
    allowed_keys: Sequence[object] | set[tuple[object, ...]] | None,
) -> set[tuple[object, ...]]:
    """Keep only excluded Qr/Qz keys that still exist in the listed snapshot."""

    normalized: set[tuple[object, ...]] = set()
    for raw_key in allowed_keys or ():
        if isinstance(raw_key, tuple):
            normalized.add(raw_key)
        elif isinstance(raw_key, list):
            normalized.add(tuple(raw_key))

    if normalized:
        state.excluded_q_groups.intersection_update(normalized)
    else:
        state.excluded_q_groups.clear()
    return state.excluded_q_groups


def clear_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
) -> None:
    """Clear all excluded Qr/Qz group keys."""

    state.excluded_q_groups.clear()


def clear_geometry_preview_excluded_keys(
    state: GeometryPreviewState,
) -> None:
    """Clear all excluded live-preview pair keys."""

    state.excluded_keys.clear()


def set_geometry_preview_match_included(
    state: GeometryPreviewState,
    match_key: tuple[object, ...] | None,
    *,
    included: bool,
) -> bool:
    """Toggle one live-preview match in the exclusion set."""

    if match_key is None:
        return False
    if included:
        state.excluded_keys.discard(match_key)
    else:
        state.excluded_keys.add(match_key)
    return True


def set_geometry_preview_exclude_mode(
    state: GeometryPreviewState,
    enabled: bool,
) -> bool:
    """Arm or disarm live-preview exclusion editing."""

    state.exclude_armed = bool(enabled)
    return state.exclude_armed


def _preview_int(value: object, default: int = 0, *, minimum: int | None = None) -> int:
    """Coerce one preview-state value to int with an optional lower bound."""

    try:
        numeric = int(value)
    except Exception:
        numeric = int(default)
    if minimum is not None:
        numeric = max(int(minimum), numeric)
    return numeric


def _preview_float(value: object, default: float) -> float:
    """Coerce one preview-state value to float."""

    try:
        return float(value)
    except Exception:
        return float(default)


def replace_geometry_preview_overlay_state(
    state: GeometryPreviewState,
    overlay_state: Mapping[str, object] | None,
) -> GeometryPreviewOverlayState:
    """Replace cached live-preview overlay data and summary metrics in place."""

    overlay = state.overlay
    source = overlay_state if isinstance(overlay_state, Mapping) else {}

    overlay.signature = source.get("signature")
    overlay.pairs.clear()
    overlay.pairs.extend(
        copy.deepcopy(dict(entry))
        for entry in source.get("pairs", [])
        if isinstance(entry, Mapping)
    )
    overlay.simulated_count = _preview_int(source.get("simulated_count"), 0)
    overlay.min_matches = _preview_int(source.get("min_matches"), 0)
    overlay.best_radius = _preview_float(source.get("best_radius"), float("nan"))
    overlay.mean_dist = _preview_float(source.get("mean_dist"), float("nan"))
    overlay.p90_dist = _preview_float(source.get("p90_dist"), float("nan"))
    overlay.quality_fail = bool(source.get("quality_fail", False))
    overlay.max_display_markers = _preview_int(
        source.get("max_display_markers"),
        120,
        minimum=1,
    )
    overlay.auto_match_attempts.clear()
    overlay.auto_match_attempts.extend(
        copy.deepcopy(dict(entry))
        for entry in source.get("auto_match_attempts", [])
        if isinstance(entry, Mapping)
    )
    overlay.q_group_total = _preview_int(source.get("q_group_total"), 0)
    overlay.q_group_excluded = _preview_int(source.get("q_group_excluded"), 0)
    overlay.excluded_q_peaks = _preview_int(source.get("excluded_q_peaks"), 0)
    overlay.collapsed_degenerate_peaks = _preview_int(
        source.get("collapsed_degenerate_peaks"),
        0,
    )
    return overlay


def set_geometry_preview_q_group_included(
    state: GeometryPreviewState,
    group_key: tuple[object, ...] | None,
    *,
    included: bool,
) -> bool:
    """Toggle one Qr/Qz group in the preview exclusion set."""

    if group_key is None:
        return False
    if included:
        state.excluded_q_groups.discard(group_key)
    else:
        state.excluded_q_groups.add(group_key)
    return True


def count_geometry_preview_excluded_q_groups(
    state: GeometryPreviewState,
    keys: Sequence[object] | None = None,
) -> int:
    """Count excluded Qr/Qz keys, optionally restricted to one listing."""

    if keys is None:
        return int(len(state.excluded_q_groups))

    normalized: set[tuple[object, ...]] = set()
    for raw_key in keys:
        if isinstance(raw_key, tuple):
            normalized.add(raw_key)
        elif isinstance(raw_key, list):
            normalized.add(tuple(raw_key))
    return int(sum(1 for key in normalized if key in state.excluded_q_groups))


def request_geometry_preview_skip_once(state: GeometryPreviewState) -> None:
    """Skip one live-preview refresh on the next update cycle."""

    state.skip_once = True


def consume_geometry_preview_skip_once(
    state: GeometryPreviewState,
) -> bool:
    """Return and clear the one-shot live-preview skip flag."""

    requested = bool(state.skip_once)
    state.skip_once = False
    return requested


def clear_geometry_preview_skip_once(state: GeometryPreviewState) -> None:
    """Clear the one-shot live-preview skip flag."""

    state.skip_once = False


def get_geometry_auto_match_background_cache(
    state: GeometryPreviewState,
    cache_key: object,
) -> dict[str, object] | None:
    """Return the cached auto-match background context when the key matches."""

    if state.auto_match_background_cache_key != cache_key:
        return None
    if not isinstance(state.auto_match_background_cache_data, dict):
        return None
    return state.auto_match_background_cache_data


def replace_geometry_auto_match_background_cache(
    state: GeometryPreviewState,
    cache_key: object,
    cache_data: dict[str, object] | None,
) -> dict[str, object] | None:
    """Replace the cached auto-match background context."""

    state.auto_match_background_cache_key = cache_key
    state.auto_match_background_cache_data = cache_data
    return state.auto_match_background_cache_data


def clear_geometry_auto_match_background_cache(
    state: GeometryPreviewState,
) -> None:
    """Discard the cached auto-match background context."""

    state.auto_match_background_cache_key = None
    state.auto_match_background_cache_data = None


def clone_geometry_q_group_entries(
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Return detached copies of Qr/Qz selector entries."""

    cloned: list[dict[str, object]] = []
    for raw_entry in entries or []:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        entry["hkl_preview"] = list(raw_entry.get("hkl_preview", []))
        cloned.append(entry)
    return cloned


def listed_geometry_q_group_entries(
    state: GeometryQGroupState,
) -> list[dict[str, object]]:
    """Return detached copies of the stored Qr/Qz selector entries."""

    return clone_geometry_q_group_entries(state.cached_entries)


def listed_geometry_q_group_keys(
    state: GeometryQGroupState,
    entries: Sequence[dict[str, object]] | None = None,
) -> set[tuple[object, ...]]:
    """Return the stable keys for the stored Qr/Qz selector entries."""

    keys: set[tuple[object, ...]] = set()
    source_entries = entries if entries is not None else state.cached_entries
    for entry in source_entries:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if key is not None:
            keys.add(key)
    return keys


def replace_geometry_q_group_cached_entries(
    state: GeometryQGroupState,
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Replace the stored Qr/Qz selector entry snapshot in place."""

    cloned = clone_geometry_q_group_entries(entries)
    state.cached_entries.clear()
    state.cached_entries.extend(cloned)
    return listed_geometry_q_group_entries(state)


def clear_geometry_q_group_row_vars(state: GeometryQGroupState) -> None:
    """Discard the current Qr/Qz selector row-var map."""

    state.row_vars.clear()


def set_geometry_q_group_row_var(
    state: GeometryQGroupState,
    group_key: tuple[object, ...] | None,
    row_var: Any,
) -> None:
    """Store one Qr/Qz selector row-var binding."""

    if group_key is None:
        return
    state.row_vars[group_key] = row_var


def request_geometry_q_group_refresh(state: GeometryQGroupState) -> None:
    """Mark the Qr/Qz selector listing for refresh on the next update."""

    state.refresh_requested = True


def consume_geometry_q_group_refresh_request(
    state: GeometryQGroupState,
) -> bool:
    """Return and clear the pending Qr/Qz selector refresh flag."""

    requested = bool(state.refresh_requested)
    state.refresh_requested = False
    return requested


def _source_all_miller_for_label(
    simulation_runtime_state: SimulationRuntimeState,
    source_label: str | None,
) -> np.ndarray:
    label = normalize_bragg_qr_source_label(source_label)
    if label == "secondary":
        return np.asarray(simulation_runtime_state.sim_miller2_all, dtype=np.float64)
    return np.asarray(simulation_runtime_state.sim_miller1_all, dtype=np.float64)


def _available_bragg_qr_group_keys(
    simulation_runtime_state: SimulationRuntimeState,
) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()

    if isinstance(simulation_runtime_state.sim_primary_qr_all, dict):
        for m_raw in simulation_runtime_state.sim_primary_qr_all.keys():
            try:
                m_idx = int(m_raw)
            except (TypeError, ValueError):
                continue
            keys.add(("primary", m_idx))

    for source_label in ("primary", "secondary"):
        m_vals = bragg_qr_m_indices_from_miller_array(
            _source_all_miller_for_label(simulation_runtime_state, source_label),
            unique=True,
        )
        for m_val in m_vals:
            keys.add((source_label, int(m_val)))

    return keys


def _available_bragg_qr_l_keys(
    simulation_runtime_state: SimulationRuntimeState,
) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()

    if isinstance(simulation_runtime_state.sim_primary_qr_all, dict):
        for m_idx, entry in copy_bragg_qr_dict(
            simulation_runtime_state.sim_primary_qr_all
        ).items():
            l_keys = bragg_qr_l_keys_from_l_array(entry.get("L", []))
            for l_key in l_keys:
                lk = int(l_key)
                if lk == BRAGG_QR_L_INVALID_KEY:
                    continue
                keys.add(("primary", int(m_idx), lk))

    for source_label in ("primary", "secondary"):
        arr = np.asarray(
            _source_all_miller_for_label(simulation_runtime_state, source_label),
            dtype=np.float64,
        )
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
            continue
        finite_mask = (
            np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
        )
        if not np.any(finite_mask):
            continue
        rows = arr[finite_mask, :3]
        m_vals = bragg_qr_m_indices_from_miller_array(rows, unique=False)
        l_keys = bragg_qr_l_keys_from_l_array(rows[:, 2])
        for m_idx, l_key in zip(m_vals, l_keys):
            lk = int(l_key)
            if lk == BRAGG_QR_L_INVALID_KEY:
                continue
            keys.add(
                (
                    normalize_bragg_qr_source_label(source_label),
                    int(m_idx),
                    lk,
                )
            )

    return keys


def prune_disabled_bragg_qr_filters(
    simulation_runtime_state: SimulationRuntimeState,
    bragg_qr_manager_state: BraggQrManagerState,
) -> None:
    """Drop disabled Bragg-Qr entries that no longer exist in the source data."""

    available_groups = _available_bragg_qr_group_keys(simulation_runtime_state)
    bragg_qr_manager_state.disabled_groups = {
        (normalize_bragg_qr_source_label(src), int(m_idx))
        for src, m_idx in bragg_qr_manager_state.disabled_groups
        if (normalize_bragg_qr_source_label(src), int(m_idx)) in available_groups
    }

    available_l = _available_bragg_qr_l_keys(simulation_runtime_state)
    bragg_qr_manager_state.disabled_l_values = {
        (normalize_bragg_qr_source_label(src), int(m_idx), int(l_key))
        for src, m_idx, l_key in bragg_qr_manager_state.disabled_l_values
        if (normalize_bragg_qr_source_label(src), int(m_idx), int(l_key))
        in available_l
    }


def filtered_primary_qr_dict(
    simulation_runtime_state: SimulationRuntimeState,
    bragg_qr_manager_state: BraggQrManagerState,
    *,
    prune_bias: object,
) -> tuple[dict[int, dict[str, object]], int, int]:
    """Return the filtered primary Bragg-Qr dictionary and prune counts."""

    out: dict[int, dict[str, object]] = {}
    total_before_prune = 0
    total_after_prune = 0
    disabled_primary_m = {
        int(m_idx)
        for src, m_idx in bragg_qr_manager_state.disabled_groups
        if normalize_bragg_qr_source_label(src) == "primary"
    }
    disabled_primary_l: dict[int, set[int]] = defaultdict(set)
    for src, m_idx, l_key in bragg_qr_manager_state.disabled_l_values:
        if normalize_bragg_qr_source_label(src) != "primary":
            continue
        disabled_primary_l[int(m_idx)].add(int(l_key))

    for m_idx, entry in copy_bragg_qr_dict(simulation_runtime_state.sim_primary_qr_all).items():
        m_int = int(m_idx)
        if m_int in disabled_primary_m:
            continue

        l_vals = np.asarray(entry.get("L", []), dtype=np.float64).reshape(-1)
        i_vals = np.asarray(entry.get("I", []), dtype=np.float64).reshape(-1)
        row_count = min(l_vals.shape[0], i_vals.shape[0])
        if row_count <= 0:
            continue

        l_vals = l_vals[:row_count]
        i_vals = i_vals[:row_count]

        disabled_l_for_m = disabled_primary_l.get(m_int)
        if disabled_l_for_m:
            l_keys = bragg_qr_l_keys_from_l_array(l_vals)
            disabled_arr = np.fromiter(
                disabled_l_for_m,
                dtype=np.int64,
                count=len(disabled_l_for_m),
            )
            keep_mask = ~np.isin(l_keys, disabled_arr)
            if not np.any(keep_mask):
                continue
            l_vals = l_vals[keep_mask]
            i_vals = i_vals[keep_mask]

        pruned_l, pruned_i, src_count, kept_count = prune_l_intensity_curve(
            l_vals,
            i_vals,
            bias=prune_bias,
        )
        total_before_prune += int(src_count)
        total_after_prune += int(kept_count)
        if kept_count <= 0:
            continue

        filtered = dict(entry)
        filtered["L"] = pruned_l
        filtered["I"] = pruned_i
        out[m_int] = filtered

    return out, total_before_prune, total_after_prune


def filtered_miller_and_intensities(
    bragg_qr_manager_state: BraggQrManagerState,
    miller_arr: np.ndarray,
    intens_arr: np.ndarray,
    source_label: str,
    *,
    prune_bias: object,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return filtered Miller/intensity arrays for one Bragg-Qr source."""

    arr = np.asarray(miller_arr, dtype=np.float64)
    intens = np.asarray(intens_arr, dtype=np.float64).reshape(-1)

    if arr.ndim != 2 or arr.shape[1] < 3:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            0,
            0,
        )

    row_count = min(arr.shape[0], intens.shape[0])
    if row_count <= 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            0,
            0,
        )

    arr = arr[:row_count, :]
    intens = intens[:row_count]
    source_norm = normalize_bragg_qr_source_label(source_label)
    disabled_m = {
        int(m_idx)
        for src, m_idx in bragg_qr_manager_state.disabled_groups
        if normalize_bragg_qr_source_label(src) == source_norm
    }
    disabled_l_pairs = [
        (int(m_idx), int(l_key))
        for src, m_idx, l_key in bragg_qr_manager_state.disabled_l_values
        if normalize_bragg_qr_source_label(src) == source_norm
    ]
    m_vals = bragg_qr_m_indices_from_miller_array(arr, unique=False)
    if m_vals.shape[0] != arr.shape[0]:
        return arr.copy(), intens.copy(), int(arr.shape[0]), int(arr.shape[0])

    keep_mask = np.ones(arr.shape[0], dtype=bool)
    if disabled_m:
        disabled_arr = np.fromiter(disabled_m, dtype=np.int64, count=len(disabled_m))
        keep_mask &= ~np.isin(m_vals, disabled_arr)

    if disabled_l_pairs:
        l_keys = bragg_qr_l_keys_from_l_array(arr[:, 2])
        for m_idx, l_key in disabled_l_pairs:
            keep_mask &= ~((m_vals == int(m_idx)) & (l_keys == int(l_key)))

    filtered_arr = arr[keep_mask]
    filtered_intens = intens[keep_mask]
    total_after_manual = int(filtered_arr.shape[0])
    if total_after_manual <= 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            int(arr.shape[0]),
            0,
        )

    if source_norm != "primary":
        return (
            filtered_arr.copy(),
            filtered_intens.copy(),
            total_after_manual,
            total_after_manual,
        )

    pruned_arr, pruned_intens, src_count, kept_count = prune_reflection_rows(
        filtered_arr,
        filtered_intens,
        bias=prune_bias,
    )
    return pruned_arr, pruned_intens, int(src_count), int(kept_count)


def apply_bragg_qr_filters(
    simulation_runtime_state: SimulationRuntimeState,
    bragg_qr_manager_state: BraggQrManagerState,
    *,
    prune_bias: object,
) -> dict[str, int]:
    """Apply Bragg-Qr disables and SF pruning to the live simulation state."""

    prune_disabled_bragg_qr_filters(
        simulation_runtime_state,
        bragg_qr_manager_state,
    )

    (
        simulation_runtime_state.sim_primary_qr,
        qr_total,
        qr_kept,
    ) = filtered_primary_qr_dict(
        simulation_runtime_state,
        bragg_qr_manager_state,
        prune_bias=prune_bias,
    )
    (
        simulation_runtime_state.sim_miller1,
        simulation_runtime_state.sim_intens1,
        hkl_primary_total,
        hkl_primary_kept,
    ) = filtered_miller_and_intensities(
        bragg_qr_manager_state,
        simulation_runtime_state.sim_miller1_all,
        simulation_runtime_state.sim_intens1_all,
        "primary",
        prune_bias=prune_bias,
    )
    (
        simulation_runtime_state.sim_miller2,
        simulation_runtime_state.sim_intens2,
        hkl_secondary_total,
        hkl_secondary_kept,
    ) = filtered_miller_and_intensities(
        bragg_qr_manager_state,
        simulation_runtime_state.sim_miller2_all,
        simulation_runtime_state.sim_intens2_all,
        "secondary",
        prune_bias=prune_bias,
    )
    stats = {
        "qr_total": int(qr_total),
        "qr_kept": int(qr_kept),
        "hkl_primary_total": int(hkl_primary_total),
        "hkl_primary_kept": int(hkl_primary_kept),
        "hkl_secondary_total": int(hkl_secondary_total),
        "hkl_secondary_kept": int(hkl_secondary_kept),
    }
    simulation_runtime_state.sf_prune_stats = stats
    return dict(stats)


def format_structure_factor_pruning_status(
    stats: Mapping[str, object] | None,
    *,
    prune_bias: object,
) -> str:
    """Format the SF-pruning status text for the GUI."""

    if not isinstance(stats, Mapping):
        stats = {}
    qr_total = int(stats.get("qr_total", 0) or 0)
    qr_kept = int(stats.get("qr_kept", 0) or 0)
    hk_total = int(stats.get("hkl_primary_total", 0) or 0)
    hk_kept = int(stats.get("hkl_primary_kept", 0) or 0)
    bias = clip_structure_factor_prune_bias(
        prune_bias,
        fallback=0.0,
        minimum=SF_PRUNE_BIAS_MIN,
        maximum=SF_PRUNE_BIAS_MAX,
    )

    if qr_total > 0:
        pct = (100.0 * qr_kept / qr_total) if qr_total else 0.0
        return (
            f"SF pruning keeps {qr_kept:,}/{qr_total:,} rod points "
            f"({pct:.1f}%), bias={bias:+.2f}"
        )

    if hk_total > 0:
        pct = (100.0 * hk_kept / hk_total) if hk_total else 0.0
        return (
            f"SF pruning keeps {hk_kept:,}/{hk_total:,} HKL points "
            f"({pct:.1f}%), bias={bias:+.2f}"
        )

    return f"SF pruning bias={bias:+.2f}"


def qr_value_for_m(m_idx: object, lattice_a: object) -> float:
    """Return the Qr value for one Bragg-Qr ``m`` index and lattice constant."""

    try:
        m_val = float(m_idx)
        a_val = float(lattice_a)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(m_val) or not np.isfinite(a_val) or a_val <= 0.0 or m_val < 0.0:
        return float("nan")
    return float((2.0 * np.pi / a_val) * np.sqrt((4.0 / 3.0) * m_val))


def hk_pairs_grouped_by_m(miller_arr: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Group unique integer ``(h, k)`` pairs by their Bragg-Qr ``m`` index."""

    grouped: dict[int, set[tuple[int, int]]] = defaultdict(set)
    arr = np.asarray(miller_arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return {}

    for row in arr:
        h_val = row[0]
        k_val = row[1]
        if not (np.isfinite(h_val) and np.isfinite(k_val)):
            continue
        h_int = int(np.rint(float(h_val)))
        k_int = int(np.rint(float(k_val)))
        m_idx = int(h_int * h_int + h_int * k_int + k_int * k_int)
        grouped[m_idx].add((h_int, k_int))

    return {
        int(m_idx): sorted(pairs, key=lambda pair: (pair[0], pair[1]))
        for m_idx, pairs in grouped.items()
    }


def build_bragg_qr_entries(
    simulation_runtime_state: SimulationRuntimeState,
    *,
    primary_a: object,
    secondary_a: object,
) -> list[dict[str, object]]:
    """Build the Bragg-Qr group listing entries for the manager window."""

    primary_hk = hk_pairs_grouped_by_m(simulation_runtime_state.sim_miller1_all)
    secondary_hk = hk_pairs_grouped_by_m(simulation_runtime_state.sim_miller2_all)

    primary_m_values = set(primary_hk.keys())
    if isinstance(simulation_runtime_state.sim_primary_qr_all, dict):
        for m_raw in simulation_runtime_state.sim_primary_qr_all.keys():
            try:
                primary_m_values.add(int(m_raw))
            except (TypeError, ValueError):
                continue

    secondary_m_values = set(secondary_hk.keys())
    entries: list[dict[str, object]] = []
    for source_label, m_values, lattice_a, hk_map in (
        ("primary", primary_m_values, primary_a, primary_hk),
        ("secondary", secondary_m_values, secondary_a, secondary_hk),
    ):
        for m_idx in sorted(m_values):
            hk_pairs = hk_map.get(m_idx, [])
            preview_pairs = hk_pairs[:8]
            preview = ", ".join(f"({h} {k})" for h, k in preview_pairs)
            if len(hk_pairs) > len(preview_pairs):
                preview += f", +{len(hk_pairs) - len(preview_pairs)}"
            if not preview:
                preview = "n/a"
            entries.append(
                {
                    "key": (str(source_label), int(m_idx)),
                    "source": str(source_label),
                    "m": int(m_idx),
                    "qr": qr_value_for_m(int(m_idx), lattice_a),
                    "hk_preview": preview,
                }
            )

    return entries


def build_bragg_qr_l_value_map(
    simulation_runtime_state: SimulationRuntimeState,
    source_label: str,
    m_idx: int,
) -> dict[int, float]:
    """Collect the unique L values available for one Bragg-Qr group."""

    source_norm = normalize_bragg_qr_source_label(source_label)
    m_target = int(m_idx)
    out: dict[int, float] = {}

    arr = np.asarray(
        _source_all_miller_for_label(simulation_runtime_state, source_norm),
        dtype=np.float64,
    )
    if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
        finite_mask = (
            np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
        )
        if np.any(finite_mask):
            rows = arr[finite_mask, :3]
            m_vals = bragg_qr_m_indices_from_miller_array(rows, unique=False)
            l_keys = bragg_qr_l_keys_from_l_array(rows[:, 2])
            for mm, l_key in zip(m_vals, l_keys):
                lk = int(l_key)
                if int(mm) != m_target or lk == BRAGG_QR_L_INVALID_KEY:
                    continue
                if lk not in out:
                    out[lk] = bragg_qr_l_key_to_value(lk)

    if source_norm == "primary" and isinstance(simulation_runtime_state.sim_primary_qr_all, dict):
        for m_raw, data in simulation_runtime_state.sim_primary_qr_all.items():
            try:
                mm = int(m_raw)
            except (TypeError, ValueError):
                continue
            if mm != m_target or not isinstance(data, dict):
                continue
            l_vals = np.asarray(data.get("L", []), dtype=np.float64).reshape(-1)
            l_keys = bragg_qr_l_keys_from_l_array(l_vals)
            for l_key in l_keys:
                lk = int(l_key)
                if lk == BRAGG_QR_L_INVALID_KEY:
                    continue
                if lk not in out:
                    out[lk] = bragg_qr_l_key_to_value(lk)
            break

    return out


def build_bragg_qr_qr_list_model(
    state: BraggQrManagerState,
    entries: Sequence[Mapping[str, object]] | None,
    *,
    selected_keys: Sequence[tuple[str, int]] | None = None,
) -> dict[str, object]:
    """Build the Bragg-Qr group listbox lines, selection, and status text."""

    normalized_selected = {
        (normalize_bragg_qr_source_label(src), int(m_idx))
        for src, m_idx in (selected_keys or ())
    }
    enabled_count = 0
    index_keys: list[tuple[str, int]] = []
    lines: list[str] = []
    entries_list = list(entries or ())

    for entry in entries_list:
        raw_key = entry.get("key", ("primary", -1))
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        key = (normalize_bragg_qr_source_label(raw_key[0]), int(raw_key[1]))
        is_enabled = key not in state.disabled_groups
        if is_enabled:
            enabled_count += 1
        state_text = "ON " if is_enabled else "OFF"
        qr_val = float(entry.get("qr", float("nan")))
        qr_text = f"{qr_val:.4f}" if np.isfinite(qr_val) else "nan"
        lines.append(
            f"[{state_text}] {str(entry.get('source', key[0])):<9} "
            f"Qr={qr_text} A^-1  m={int(entry.get('m', key[1])):>3}  "
            f"HK={str(entry.get('hk_preview', 'n/a'))}"
        )
        index_keys.append(key)

    if not entries_list:
        lines = ["No Bragg Qr groups are available for the current simulation."]

    selected_indices = [
        idx for idx, key in enumerate(index_keys) if key in normalized_selected
    ]
    see_index = selected_indices[0] if selected_indices else None
    if index_keys and not selected_indices:
        default_idx = 0
        selected_group_key = state.selected_group_key
        if isinstance(selected_group_key, tuple) and len(selected_group_key) >= 2:
            normalized_group = (
                normalize_bragg_qr_source_label(selected_group_key[0]),
                int(selected_group_key[1]),
            )
            if normalized_group in index_keys:
                default_idx = index_keys.index(normalized_group)
        selected_indices = [default_idx]
        see_index = default_idx

    return {
        "lines": lines,
        "index_keys": index_keys,
        "selected_indices": selected_indices,
        "status_text": f"Enabled: {enabled_count} / {len(entries_list)}",
        "see_index": see_index,
    }


def build_bragg_qr_l_list_model(
    state: BraggQrManagerState,
    *,
    group_key: tuple[str, int] | None,
    l_value_map: Mapping[int, object] | None,
    selected_l_keys: Sequence[object] | None = None,
) -> dict[str, object]:
    """Build the Bragg-Qr L listbox lines, selection, and status text."""

    if group_key is None:
        return {
            "selected_group_key": None,
            "lines": ["Select a Qr group on the left to view L values."],
            "index_keys": [],
            "selected_indices": [],
            "status_text": "No Qr group selected.",
        }

    source_label = normalize_bragg_qr_source_label(group_key[0])
    m_idx = int(group_key[1])
    normalized_map = {
        int(l_key): float(l_val)
        for l_key, l_val in (l_value_map or {}).items()
        if int(l_key) != BRAGG_QR_L_INVALID_KEY
    }
    if not normalized_map:
        return {
            "selected_group_key": (source_label, m_idx),
            "lines": ["No L values available for the selected Qr group."],
            "index_keys": [],
            "selected_indices": [],
            "status_text": f"{source_label} m={m_idx} has no L entries.",
        }

    selected_set = {
        int(raw_l_key)
        for raw_l_key in (selected_l_keys or ())
        if int(raw_l_key) != BRAGG_QR_L_INVALID_KEY
    }
    group_disabled = (source_label, m_idx) in state.disabled_groups

    enabled_count = 0
    lines: list[str] = []
    index_keys: list[int] = []
    for l_key, l_val in sorted(normalized_map.items(), key=lambda item: item[1]):
        l_disabled = (source_label, m_idx, int(l_key)) in state.disabled_l_values
        is_enabled = (not group_disabled) and (not l_disabled)
        if is_enabled:
            enabled_count += 1
        state_text = "ON " if is_enabled else "OFF"
        lines.append(f"[{state_text}] L={float(l_val):.4f}")
        index_keys.append(int(l_key))

    selected_indices = [
        idx for idx, l_key in enumerate(index_keys) if int(l_key) in selected_set
    ]
    suffix = " | Qr group disabled" if group_disabled else ""
    return {
        "selected_group_key": (source_label, m_idx),
        "lines": lines,
        "index_keys": index_keys,
        "selected_indices": selected_indices,
        "status_text": (
            f"Selected: {source_label} m={m_idx} | "
            f"Enabled L: {enabled_count} / {len(index_keys)}{suffix}"
        ),
    }


def clear_bragg_qr_manager_state(state: BraggQrManagerState) -> None:
    """Discard all Bragg-Qr manager list bookkeeping."""

    state.qr_index_keys.clear()
    state.l_index_keys.clear()
    state.selected_group_key = None


def replace_bragg_qr_index_keys(
    state: BraggQrManagerState,
    keys: Sequence[object] | None,
) -> list[tuple[str, int]]:
    """Replace the stored Bragg-Qr group index-key list in place."""

    normalized: list[tuple[str, int]] = []
    for raw_key in keys or ():
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        try:
            normalized.append((str(raw_key[0]), int(raw_key[1])))
        except Exception:
            continue
    state.qr_index_keys.clear()
    state.qr_index_keys.extend(normalized)
    return list(state.qr_index_keys)


def replace_bragg_qr_l_index_keys(
    state: BraggQrManagerState,
    keys: Sequence[object] | None,
) -> list[int]:
    """Replace the stored Bragg-Qr L-index list in place."""

    normalized: list[int] = []
    for raw_key in keys or ():
        try:
            normalized.append(int(raw_key))
        except Exception:
            continue
    state.l_index_keys.clear()
    state.l_index_keys.extend(normalized)
    return list(state.l_index_keys)


def set_bragg_qr_selected_group_key(
    state: BraggQrManagerState,
    group_key: tuple[str, int] | Sequence[object] | None,
) -> tuple[str, int] | None:
    """Store the currently selected Bragg-Qr group key."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        state.selected_group_key = None
    else:
        try:
            state.selected_group_key = (str(group_key[0]), int(group_key[1]))
        except Exception:
            state.selected_group_key = None
    return state.selected_group_key


def selected_bragg_qr_keys(
    state: BraggQrManagerState,
    selected_indices: Sequence[object] | None,
) -> list[tuple[str, int]]:
    """Map selected listbox indices to Bragg-Qr group keys."""

    selected_keys: list[tuple[str, int]] = []
    for raw_idx in selected_indices or ():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if 0 <= idx < len(state.qr_index_keys):
            selected_keys.append(state.qr_index_keys[idx])
    return selected_keys


def selected_bragg_qr_l_keys(
    state: BraggQrManagerState,
    selected_indices: Sequence[object] | None,
) -> list[int]:
    """Map selected listbox indices to Bragg-Qr L keys."""

    selected_keys: list[int] = []
    for raw_idx in selected_indices or ():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if 0 <= idx < len(state.l_index_keys):
            selected_keys.append(int(state.l_index_keys[idx]))
    return selected_keys


def set_bragg_qr_groups_enabled(
    disabled_groups: set[tuple[str, int]],
    group_keys: Sequence[tuple[str, int]] | None,
    *,
    enabled: bool,
) -> int:
    """Enable or disable the provided normalized Bragg-Qr groups in place."""

    changed = 0
    for raw_key in group_keys or ():
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        try:
            key = (str(raw_key[0]), int(raw_key[1]))
        except Exception:
            continue
        if enabled:
            if key in disabled_groups:
                disabled_groups.remove(key)
                changed += 1
        else:
            if key not in disabled_groups:
                disabled_groups.add(key)
                changed += 1
    return changed


def toggle_bragg_qr_groups(
    disabled_groups: set[tuple[str, int]],
    group_keys: Sequence[tuple[str, int]] | None,
) -> int:
    """Toggle the provided normalized Bragg-Qr groups in place."""

    changed = 0
    for raw_key in group_keys or ():
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        try:
            key = (str(raw_key[0]), int(raw_key[1]))
        except Exception:
            continue
        if key in disabled_groups:
            disabled_groups.remove(key)
        else:
            disabled_groups.add(key)
        changed += 1
    return changed


def set_bragg_qr_l_values_enabled(
    disabled_l_values: set[tuple[str, int, int]],
    group_key: tuple[str, int] | Sequence[object] | None,
    l_keys: Sequence[object] | None,
    *,
    enabled: bool,
    invalid_key: int,
) -> int:
    """Enable or disable the provided normalized Bragg-Qr L values in place."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        return 0
    try:
        source_label = str(group_key[0])
        m_idx = int(group_key[1])
    except Exception:
        return 0

    changed = 0
    for raw_l_key in l_keys or ():
        try:
            l_key = int(raw_l_key)
        except Exception:
            continue
        if l_key == int(invalid_key):
            continue
        key = (source_label, m_idx, l_key)
        if enabled:
            if key in disabled_l_values:
                disabled_l_values.remove(key)
                changed += 1
        else:
            if key not in disabled_l_values:
                disabled_l_values.add(key)
                changed += 1
    return changed


def toggle_bragg_qr_l_values(
    disabled_l_values: set[tuple[str, int, int]],
    group_key: tuple[str, int] | Sequence[object] | None,
    l_keys: Sequence[object] | None,
    *,
    invalid_key: int,
) -> int:
    """Toggle the provided normalized Bragg-Qr L values in place."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        return 0
    try:
        source_label = str(group_key[0])
        m_idx = int(group_key[1])
    except Exception:
        return 0

    changed = 0
    for raw_l_key in l_keys or ():
        try:
            l_key = int(raw_l_key)
        except Exception:
            continue
        if l_key == int(invalid_key):
            continue
        key = (source_label, m_idx, l_key)
        if key in disabled_l_values:
            disabled_l_values.remove(key)
        else:
            disabled_l_values.add(key)
        changed += 1
    return changed


def clear_bragg_qr_l_values_for_group(
    disabled_l_values: set[tuple[str, int, int]],
    group_key: tuple[str, int] | Sequence[object] | None,
) -> bool:
    """Enable every L value for one normalized Bragg-Qr group."""

    if not isinstance(group_key, (list, tuple)) or len(group_key) < 2:
        return False
    try:
        source_label = str(group_key[0])
        m_idx = int(group_key[1])
    except Exception:
        return False

    filtered = {
        (src, mm, lk)
        for src, mm, lk in disabled_l_values
        if not (str(src) == source_label and int(mm) == m_idx)
    }
    if len(filtered) == len(disabled_l_values):
        return False
    disabled_l_values.clear()
    disabled_l_values.update(filtered)
    return True
