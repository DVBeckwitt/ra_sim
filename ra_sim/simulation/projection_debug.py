from __future__ import annotations

import contextvars
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ra_sim.config import get_dir
from ra_sim.debug_controls import (
    projection_debug_logging_enabled as _projection_debug_logging_enabled,
)


EXIT_PROJECTION_INTERNAL = 0
EXIT_PROJECTION_REFRACTED = 1

PROJECTION_DEBUG_COUNTER_NAMES = (
    "n_total",
    "n_invalid_det",
    "n_prop_att_reject",
    "n_prop_fac_reject",
    "n_no_valid_sample",
    "n_large_exit_remap",
    "n_near_critical_band",
)

PROJECTION_DEBUG_COUNTER_COLS = len(PROJECTION_DEBUG_COUNTER_NAMES)
PROJECTION_DEBUG_REJECT_RECORD_COLS = 10
PROJECTION_DEBUG_REJECT_RECORD_CAPACITY = 8
PROJECTION_DEBUG_EMIT_REJECT_LIMIT = 20

PROJECTION_DEBUG_REASON_LABELS = {
    0: "unknown",
    1: "no_valid_sample",
    2: "invalid_det",
    3: "prop_att_reject",
    4: "prop_fac_reject",
    5: "projection_norm",
    6: "invalid_det_coord",
    7: "off_detector",
}


@dataclass
class ProjectionDebugSession:
    stamp: str
    source: str
    settings_snapshot: dict[str, Any]
    backgrounds: list[dict[str, Any]] = field(default_factory=list)
    log_path: str | None = None
    _context_token: Any | None = None


_CURRENT_PROJECTION_DEBUG_SESSION: contextvars.ContextVar[ProjectionDebugSession | None] = (
    contextvars.ContextVar("projection_debug_session", default=None)
)


def projection_debug_logging_enabled() -> bool:
    return _projection_debug_logging_enabled()


def resolve_exit_projection_mode_flag(value: str | int | None) -> int:
    if value == EXIT_PROJECTION_INTERNAL or value is None:
        return EXIT_PROJECTION_INTERNAL
    if value == EXIT_PROJECTION_REFRACTED:
        return EXIT_PROJECTION_REFRACTED
    mode = str(value).strip().lower()
    if mode == "internal":
        return EXIT_PROJECTION_INTERNAL
    if mode == "refracted":
        return EXIT_PROJECTION_REFRACTED
    raise ValueError(f"Unsupported exit_projection_mode: {value!r}")


def exit_projection_mode_label(value: str | int | None) -> str:
    return (
        "refracted"
        if resolve_exit_projection_mode_flag(value) == EXIT_PROJECTION_REFRACTED
        else "internal"
    )


def _resolve_projection_debug_log_root() -> Path:
    root: str | None
    try:
        root = get_dir("debug_log_dir")
    except Exception:
        root = None
    path = Path(root) if root else (Path.cwd() / "logs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _round_or_none(value: Any, digits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return round(scalar, digits)


def _deg_or_none(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return round(math.degrees(scalar), digits)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_q_group_key(hkl: np.ndarray, av_row: np.ndarray | None) -> list[Any]:
    h = int(round(float(hkl[0])))
    k = int(round(float(hkl[1])))
    l = int(round(float(hkl[2])))
    qr = None
    if av_row is not None and av_row.size > 0:
        qr = _round_or_none(av_row[0], 8)
    if qr is not None and abs(qr) > 0.0:
        return ["non_specular", qr, l]
    if h == 0 and k == 0:
        return ["specular", l]
    return ["non_specular_hkl", h, k, l]


def _band_means_for_center(
    counts: np.ndarray,
    mean_tthp_deg: np.ndarray,
    mean_tth_deg: np.ndarray,
    center_index: int | None,
) -> tuple[float | None, float | None]:
    if center_index is None or center_index < 0:
        return None, None
    row_count = int(counts.shape[0])
    if row_count <= 0:
        return None, None

    left = center_index
    while left >= 0 and counts[left] <= 0:
        left -= 1
    right = center_index
    while right < row_count and counts[right] <= 0:
        right += 1

    left_tthp = mean_tthp_deg[left] if left >= 0 and counts[left] > 0 else np.nan
    right_tthp = mean_tthp_deg[right] if right < row_count and counts[right] > 0 else np.nan
    left_tth = mean_tth_deg[left] if left >= 0 and counts[left] > 0 else np.nan
    right_tth = mean_tth_deg[right] if right < row_count and counts[right] > 0 else np.nan

    if np.isfinite(left_tthp) and np.isfinite(right_tthp):
        band_tthp = 0.5 * (float(left_tthp) + float(right_tthp))
    elif np.isfinite(left_tthp):
        band_tthp = float(left_tthp)
    elif np.isfinite(right_tthp):
        band_tthp = float(right_tthp)
    else:
        band_tthp = None

    if np.isfinite(left_tth) and np.isfinite(right_tth):
        band_tth = 0.5 * (float(left_tth) + float(right_tth))
    elif np.isfinite(left_tth):
        band_tth = float(left_tth)
    elif np.isfinite(right_tth):
        band_tth = float(right_tth)
    else:
        band_tth = None

    return (
        round(band_tthp, 6) if band_tthp is not None else None,
        round(band_tth, 6) if band_tth is not None else None,
    )


def start_projection_debug_session(
    settings_snapshot: dict[str, Any] | None = None,
    *,
    source: str = "simulation",
) -> ProjectionDebugSession:
    if not projection_debug_logging_enabled():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = ProjectionDebugSession(
        stamp=stamp,
        source=source,
        settings_snapshot=dict(settings_snapshot or {}),
    )
    session._context_token = _CURRENT_PROJECTION_DEBUG_SESSION.set(session)
    return session


def get_current_projection_debug_session() -> ProjectionDebugSession | None:
    return _CURRENT_PROJECTION_DEBUG_SESSION.get()


def append_projection_debug_background(
    session: ProjectionDebugSession | None,
    background_payload: dict[str, Any],
) -> None:
    if session is None:
        return
    session.backgrounds.append(background_payload)


def finalize_projection_debug_session(session: ProjectionDebugSession | None) -> str | None:
    if session is None:
        return None
    payload = {
        "source": session.source,
        "settings_snapshot": session.settings_snapshot,
        "backgrounds": session.backgrounds,
    }
    path = _resolve_projection_debug_log_root() / f"projection_debug_{session.stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    session.log_path = str(path)
    if session._context_token is not None:
        try:
            _CURRENT_PROJECTION_DEBUG_SESSION.reset(session._context_token)
        except ValueError:
            pass
        session._context_token = None
    return session.log_path


def allocate_projection_debug_buffers(
    peak_capacity: int,
    image_size: int,
    thread_count: int,
    *,
    reject_capacity: int = PROJECTION_DEBUG_REJECT_RECORD_CAPACITY,
) -> dict[str, np.ndarray]:
    if not projection_debug_logging_enabled():
        return {}
    peak_count = max(int(peak_capacity), 1)
    row_count = max(int(thread_count), 1)
    row_bins = max(int(image_size), 1)
    reject_count = max(int(reject_capacity), 1)
    return {
        "projection_debug_counters": np.zeros(
            (peak_count, PROJECTION_DEBUG_COUNTER_COLS), dtype=np.int64
        ),
        "projection_debug_reject_counts": np.zeros(peak_count, dtype=np.int64),
        "projection_debug_reject_records": np.full(
            (peak_count, reject_count, PROJECTION_DEBUG_REJECT_RECORD_COLS),
            np.nan,
            dtype=np.float64,
        ),
        "projection_debug_row_hit_counts": np.zeros((row_count, row_bins), dtype=np.int64),
        "projection_debug_row_tthp_sums": np.zeros((row_count, row_bins), dtype=np.float64),
        "projection_debug_row_tth_sums": np.zeros((row_count, row_bins), dtype=np.float64),
    }


def projection_debug_request_settings(
    request: Any,
    *,
    source: str = "simulation",
) -> dict[str, Any]:
    geometry = getattr(request, "geometry", None)
    return {
        "source": source,
        "theta_i_deg": _round_or_none(
            getattr(geometry, "theta_initial_deg", None) if geometry is not None else None,
            4,
        ),
        "optics_mode": getattr(request, "optics_mode", None),
        "exit_projection_mode": exit_projection_mode_label(
            getattr(request, "exit_projection_mode", None)
        ),
        "image_size": int(getattr(geometry, "image_size", 0) or 0),
        "center_x_px": _round_or_none(
            geometry.center[0] if geometry is not None else None, 4
        ),
        "center_y_px": _round_or_none(
            geometry.center[1] if geometry is not None else None, 4
        ),
        "pixel_size_um": _round_or_none(
            (geometry.pixel_size_m if geometry is not None else None) * 1.0e6
            if geometry is not None
            else None,
            4,
        ),
    }


def build_projection_debug_background(
    raw_buffers: dict[str, np.ndarray],
    miller: np.ndarray,
    av: np.ndarray | None,
    settings_snapshot: dict[str, Any] | None,
    *,
    background_index: int | None = None,
    background_label: str | None = None,
) -> dict[str, Any]:
    if not raw_buffers:
        return {}
    counters = np.asarray(raw_buffers["projection_debug_counters"], dtype=np.int64)
    reject_counts = np.asarray(raw_buffers["projection_debug_reject_counts"], dtype=np.int64)
    reject_records = np.asarray(raw_buffers["projection_debug_reject_records"], dtype=np.float64)
    row_hit_counts = np.asarray(raw_buffers["projection_debug_row_hit_counts"], dtype=np.int64)
    row_tthp_sums = np.asarray(raw_buffers["projection_debug_row_tthp_sums"], dtype=np.float64)
    row_tth_sums = np.asarray(raw_buffers["projection_debug_row_tth_sums"], dtype=np.float64)

    merged_row_counts = row_hit_counts.sum(axis=0) if row_hit_counts.ndim > 1 else row_hit_counts
    merged_row_tthp = row_tthp_sums.sum(axis=0) if row_tthp_sums.ndim > 1 else row_tthp_sums
    merged_row_tth = row_tth_sums.sum(axis=0) if row_tth_sums.ndim > 1 else row_tth_sums

    mean_tthp_deg = np.full(merged_row_counts.shape, np.nan, dtype=np.float64)
    mean_tth_deg = np.full(merged_row_counts.shape, np.nan, dtype=np.float64)
    nonzero_rows = merged_row_counts > 0
    if np.any(nonzero_rows):
        mean_tthp_deg[nonzero_rows] = np.degrees(
            merged_row_tthp[nonzero_rows] / merged_row_counts[nonzero_rows]
        )
        mean_tth_deg[nonzero_rows] = np.degrees(
            merged_row_tth[nonzero_rows] / merged_row_counts[nonzero_rows]
        )

    longest_start = -1
    longest_len = 0
    current_start = -1
    current_len = 0
    for idx, count in enumerate(merged_row_counts):
        if int(count) == 0:
            if current_len == 0:
                current_start = idx
            current_len += 1
            if current_len > longest_len:
                longest_len = current_len
                longest_start = current_start
        else:
            current_start = -1
            current_len = 0

    band_center = (longest_start + (longest_len // 2)) if longest_len > 0 else None
    band_tthp_deg, band_tth_deg = _band_means_for_center(
        merged_row_counts,
        mean_tthp_deg,
        mean_tth_deg,
        band_center,
    )

    per_reflection: list[dict[str, Any]] = []
    emit_count = min(int(counters.shape[0]), int(miller.shape[0]))
    background_counters = {
        name: int(counters[:, col].sum()) if counters.ndim == 2 else 0
        for col, name in enumerate(PROJECTION_DEBUG_COUNTER_NAMES)
    }
    for idx in range(emit_count):
        hkl_row = np.asarray(miller[idx], dtype=np.float64)
        av_row = None
        if av is not None and idx < len(av):
            av_row = np.asarray(av[idx], dtype=np.float64)
        counter_values = {
            name: int(counters[idx, col])
            for col, name in enumerate(PROJECTION_DEBUG_COUNTER_NAMES)
        }
        per_reflection.append(
            {
                "reflection_index": idx,
                "hkl": [
                    int(round(float(hkl_row[0]))),
                    int(round(float(hkl_row[1]))),
                    int(round(float(hkl_row[2]))),
                ],
                "q_group_key": _build_q_group_key(hkl_row, av_row),
                "counters": counter_values,
            }
        )

    rejected_records: list[dict[str, Any]] = []
    theta_i_deg = _round_or_none((settings_snapshot or {}).get("theta_i_deg"), 6)
    for idx in range(emit_count):
        if len(rejected_records) >= PROJECTION_DEBUG_EMIT_REJECT_LIMIT:
            break
        hkl_row = np.asarray(miller[idx], dtype=np.float64)
        av_row = None
        if av is not None and idx < len(av):
            av_row = np.asarray(av[idx], dtype=np.float64)
        count = min(int(reject_counts[idx]), int(reject_records.shape[1]))
        for reject_idx in range(count):
            if len(rejected_records) >= PROJECTION_DEBUG_EMIT_REJECT_LIMIT:
                break
            rec = reject_records[idx, reject_idx]
            twotheta_t_prime = rec[4]
            twotheta_t = rec[5]
            rejected_records.append(
                {
                    "theta_i": theta_i_deg,
                    "hkl": [
                        int(round(float(hkl_row[0]))),
                        int(round(float(hkl_row[1]))),
                        int(round(float(hkl_row[2]))),
                    ],
                    "q_group_key": _build_q_group_key(hkl_row, av_row),
                    "twotheta_t_prime": _deg_or_none(twotheta_t_prime, 6),
                    "twotheta_t": _deg_or_none(twotheta_t, 6),
                    "delta_tth": _deg_or_none(twotheta_t - twotheta_t_prime, 6),
                    "kr": _round_or_none(rec[6], 8),
                    "k0": _round_or_none(rec[7], 8),
                    "n2_real": _round_or_none(rec[8], 8),
                    "rejection_reason": PROJECTION_DEBUG_REASON_LABELS.get(
                        int(round(float(rec[9]))),
                        "unknown",
                    ),
                }
            )

    row_histogram = {
        "row_hit_counts": [int(value) for value in merged_row_counts.tolist()],
        "mean_twotheta_t_prime_deg": [
            None if not math.isfinite(float(value)) else round(float(value), 6)
            for value in mean_tthp_deg.tolist()
        ],
        "mean_twotheta_t_deg": [
            None if not math.isfinite(float(value)) else round(float(value), 6)
            for value in mean_tth_deg.tolist()
        ],
    }

    summary = {
        "longest_empty_row_band": int(longest_len),
        "band_center_row_index": band_center,
        "mean_twotheta_t_prime_deg": band_tthp_deg,
        "mean_twotheta_t_deg": band_tth_deg,
    }

    return {
        "background_index": background_index,
        "background_label": background_label,
        "settings_snapshot": dict(settings_snapshot or {}),
        "background_counters": background_counters,
        "per_reflection": per_reflection,
        "row_histogram": row_histogram,
        "longest_empty_row_band_summary": summary,
        "first_rejected_rays": rejected_records,
    }
