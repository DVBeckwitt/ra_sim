"""Run frozen-state geometry-fit baselines and summarize fit-quality evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from ra_sim.user_paths import user_cache_root, user_data_root


DEFAULT_STATE_PATHS = (
    user_data_root() / "Bi2Se3.json",
    user_data_root() / "Bi2Te3.json",
)
DEFAULT_CACHE_LOG_DIR = user_cache_root() / "logs"
HEARTBEAT_INTERVAL_SECONDS = 20.0
SAVED_STATE_RMS_TOLERANCE_PX = 0.25
SAVED_STATE_MAX_ERROR_TOLERANCE_PX = 1.0
NO_MATCHED_FIXED_PAIR_REJECTION = "No matched peak pairs were available for the fitted solution."
PAIR_PHASE_BEFORE = "requested_start_correspondence"
PAIR_PHASE_AFTER = "acceptance_residuals"
CAKED_FIT_SPACE_PROJECTOR_KINDS = {
    "exact_caked_bundle",
    "point_only_dynamic_source_row",
}
PAIR_PHASE_BEFORE_FALLBACKS = (
    (
        "requested_start_correspondence",
        ("matched",),
        ("detector_residual_px", "optimizer_residual_px"),
    ),
    ("seed_correspondence", ("matched",), ("detector_residual_px", "optimizer_residual_px")),
    (
        "preflight_normalized_pairs",
        ("matched", "selected"),
        ("optimizer_residual_px", "detector_residual_px"),
    ),
    ("saved_pairs", ("matched", "selected"), ("detector_residual_px", "optimizer_residual_px")),
)
PARAM_LINE_RE = re.compile(
    r"^param\[(?P<name>[^\]]+)\]\s+"
    r"group=(?P<group>\S+)\s+"
    r"start=(?P<start>\S+)\s+"
    r"final=(?P<final>\S+)\s+"
    r"delta=(?P<delta>\S+)\s+"
    r"bounds=\[(?P<lower>[^,]+),\s*(?P<upper>[^\]]+)\]\s+"
    r"scale=(?P<scale>\S+)"
    r"(?:\s+prior_center=(?P<prior_center>\S+)\s+prior_sigma=(?P<prior_sigma>\S+))?$"
)
DIRECT_SOLVE_EVAL_RE = re.compile(
    r"^Geometry fit: identity fixed manual pairs eval=(?P<eval>\d+)\s+"
    r"cost=(?P<cost>\S+)\s+best_cost=(?P<best_cost>\S+)\s+"
    r"weighted_rms=(?P<weighted_rms>\S+)px$"
)
DIRECT_SOLVE_PROGRESS_RE = re.compile(
    r"^solve_progress label=identity fixed manual pairs evaluations=(?P<evaluations>\d+)\s+"
    r"best_cost=(?P<best_cost>\S+)\s+last_cost=(?P<last_cost>\S+)\s+"
    r"best_weighted_rms_px=(?P<best_weighted_rms_px>\S+)\s+"
    r"last_weighted_rms_px=(?P<last_weighted_rms_px>\S+)"
)


@dataclass(frozen=True)
class RunArtifacts:
    state_path: Path
    run_dir: Path
    out_state_path: Path
    log_path: Path | None
    trace_path: Path | None
    matched_peaks_path: Path | None
    cli_stdout_path: Path
    cli_stderr_path: Path
    cli_returncode: int
    timed_out: bool = False
    timeout_reason: str | None = None
    timeout_last_phase: str | None = None
    timeout_last_pair_id: str | None = None


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return float(number) if math.isfinite(number) else None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _rms(values: Sequence[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.sqrt(np.mean(finite * finite)))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _point_xy_or_none(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) < 2:
        return None
    try:
        col = float(value[0])
        row = float(value[1])
    except Exception:
        return None
    if not (math.isfinite(col) and math.isfinite(row)):
        return None
    return float(col), float(row)


def _latest_path(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _is_current_run_artifact(
    path: Path | None,
    *,
    started_at_wall_time: float | None,
) -> bool:
    if path is None or not path.is_file():
        return False
    if started_at_wall_time is None:
        return True
    return bool(path.stat().st_mtime >= started_at_wall_time)


def _extract_reported_artifact_path(text: str, label: str) -> Path | None:
    prefix = f"{label}:"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith(prefix):
            continue
        candidate = line.partition(":")[2].strip()
        return Path(candidate) if candidate else None
    return None


def _last_nonempty_line(text: str) -> str | None:
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if line:
            return line
    return None


def _parse_trace_path_from_log(path: Path | None) -> Path | None:
    if path is None or not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("trace_path="):
            candidate = line.partition("=")[2].strip()
            return Path(candidate) if candidate else None
    return None


def _tail_lines(path: Path | None, *, max_lines: int = 80) -> list[str]:
    if path is None or not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as handle:
        return list(deque(handle, maxlen=max_lines))


def _default_live_log_path(state_path: Path) -> Path:
    return DEFAULT_CACHE_LOG_DIR / f"geometry_fit_log_{state_path.stem}.txt"


def _default_live_trace_path(state_path: Path) -> Path:
    return DEFAULT_CACHE_LOG_DIR / f"geometry_fit_trace_{state_path.stem}.jsonl"


def _resolve_live_artifact_paths(
    *,
    state_path: Path,
    run_dir: Path,
    cli_stdout_path: Path,
    cli_stderr_path: Path,
    started_at_wall_time: float,
) -> tuple[Path | None, Path | None]:
    stdout_text = cli_stdout_path.read_text(encoding="utf-8") if cli_stdout_path.is_file() else ""
    stderr_text = cli_stderr_path.read_text(encoding="utf-8") if cli_stderr_path.is_file() else ""
    log_path = _extract_reported_artifact_path(stdout_text, "Geometry fit log")
    if log_path is None:
        log_path = _extract_reported_artifact_path(stderr_text, "Geometry fit log")
    if not _is_current_run_artifact(log_path, started_at_wall_time=started_at_wall_time):
        default_log_path = _default_live_log_path(state_path)
        latest_run_log_path = _latest_path(run_dir, "geometry_fit_log_*.txt")
        log_path = (
            default_log_path
            if _is_current_run_artifact(default_log_path, started_at_wall_time=started_at_wall_time)
            else latest_run_log_path
            if _is_current_run_artifact(latest_run_log_path, started_at_wall_time=started_at_wall_time)
            else None
        )

    trace_path = _parse_trace_path_from_log(log_path)
    if not _is_current_run_artifact(trace_path, started_at_wall_time=started_at_wall_time):
        default_trace_path = _default_live_trace_path(state_path)
        latest_run_trace_path = _latest_path(run_dir, "geometry_fit_trace_*.jsonl")
        trace_path = (
            default_trace_path
            if _is_current_run_artifact(default_trace_path, started_at_wall_time=started_at_wall_time)
            else latest_run_trace_path
            if _is_current_run_artifact(latest_run_trace_path, started_at_wall_time=started_at_wall_time)
            else None
        )
    return log_path, trace_path


def _infer_phase_from_trace(path: Path | None) -> str | None:
    for raw_line in reversed(_tail_lines(path)):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, Mapping):
            continue
        phase = str(record.get("phase", "") or "").strip()
        if phase:
            return phase
        record_type = str(record.get("record_type", "") or "").strip()
        if record_type == "run":
            accepted = record.get("accepted")
            if accepted is True:
                return "run_complete_accepted"
            if accepted is False:
                return "run_complete_rejected"
            return "run_complete"
        if record_type:
            return record_type
    return None


def _infer_phase_from_log(path: Path | None) -> str | None:
    phase_markers = (
        ("Fit rejected:", "fit_rejected"),
        ("Fit accepted:", "fit_accepted"),
        ("Phase trace:", "phase_trace"),
        ("Fit-space calibration:", "fit_space_calibration"),
        ("Point-match diagnostics:", "point_match_diagnostics"),
        ("Geometry fit log:", "log_initialized"),
    )
    last_nonempty: str | None = None
    for raw_line in reversed(_tail_lines(path)):
        line = raw_line.strip()
        if not line:
            continue
        if last_nonempty is None:
            last_nonempty = line
        for prefix, phase_name in phase_markers:
            if line.startswith(prefix):
                return phase_name
    return last_nonempty


def _active_phase_label(
    *,
    log_path: Path | None,
    trace_path: Path | None,
) -> str:
    return _infer_phase_from_trace(trace_path) or _infer_phase_from_log(log_path) or "launching"


def _last_trace_phase_and_pair_id(
    trace_path: Path | None,
    *,
    max_lines: int = 400,
) -> tuple[str | None, str | None]:
    last_phase: str | None = None
    last_pair_id: str | None = None
    for raw_line in reversed(_tail_lines(trace_path, max_lines=max_lines)):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if not isinstance(record, Mapping):
            continue
        phase = str(record.get("phase", "") or "").strip() or None
        if last_phase is None and phase is not None:
            last_phase = phase
        if str(record.get("record_type", "") or "").strip() != "pair":
            continue
        pair_id = str(record.get("pair_id", "") or "").strip() or None
        if pair_id is not None:
            last_pair_id = pair_id
            if last_phase is None:
                last_phase = phase
            break
    return last_phase, last_pair_id


def _mirror_artifact_into_run_dir(
    source_path: Path | None,
    *,
    run_dir: Path,
) -> Path | None:
    if source_path is None or not source_path.is_file():
        return None
    target_path = run_dir / source_path.name
    if source_path.resolve() == target_path.resolve():
        return source_path
    shutil.copy2(source_path, target_path)
    return target_path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace_records(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        data = json.loads(line)
        if isinstance(data, Mapping):
            records.append(dict(data))
    return records


def _phase_pair_records(
    records: Sequence[Mapping[str, object]],
    phase: str,
) -> list[dict[str, object]]:
    return [
        dict(record)
        for record in records
        if str(record.get("record_type", "")) == "pair"
        and str(record.get("phase", "")) == phase
    ]


def _normalize_pair_record(record: Mapping[str, object]) -> dict[str, object]:
    residual = _float_or_none(record.get("detector_residual_px"))
    optimizer_residual = _float_or_none(record.get("optimizer_residual_px"))
    measured_point = _point_xy_or_none(record.get("measured_point"))
    simulated_point = _point_xy_or_none(record.get("simulated_point"))
    dx_px = _float_or_none(record.get("dx_px"))
    dy_px = _float_or_none(record.get("dy_px"))
    if dx_px is None and measured_point is not None and simulated_point is not None:
        dx_px = float(simulated_point[0] - measured_point[0])
    if dy_px is None and measured_point is not None and simulated_point is not None:
        dy_px = float(simulated_point[1] - measured_point[1])
    return {
        "pair_id": str(record.get("pair_id", "") or ""),
        "dataset_index": int(record.get("dataset_index", 0) or 0),
        "background_label": str(record.get("background_label", "") or ""),
        "overlay_match_index": record.get("overlay_match_index"),
        "hkl": _json_safe(record.get("hkl")),
        "match_status": str(record.get("match_status", "") or ""),
        "detector_residual_px": residual,
        "optimizer_residual_px": optimizer_residual,
        "measured_point": _json_safe(measured_point),
        "simulated_point": _json_safe(simulated_point),
        "dx_px": dx_px,
        "dy_px": dy_px,
        "canonical_identity": _json_safe(record.get("canonical_identity")),
        "resolution_kind": _json_safe(record.get("resolution_kind")),
        "resolution_reason": _json_safe(record.get("resolution_reason")),
        "source_branch_index": _int_or_none(record.get("source_branch_index")),
        "source_peak_index": _int_or_none(record.get("source_peak_index")),
        "resolved_peak_index": _int_or_none(record.get("resolved_peak_index")),
        "match_radius_exceeded": bool(record.get("match_radius_exceeded", False)),
        "measured_detector_field_name": record.get("measured_detector_field_name"),
        "measured_detector_input_frame": record.get("measured_detector_input_frame"),
        "measured_detector_frame_reason": record.get("measured_detector_frame_reason"),
        "simulated_detector_field_name": record.get("simulated_detector_field_name"),
        "simulated_detector_input_frame": record.get("simulated_detector_input_frame"),
        "simulated_detector_frame_reason": record.get("simulated_detector_frame_reason"),
        "measured_native_col": _float_or_none(record.get("measured_native_col")),
        "measured_native_row": _float_or_none(record.get("measured_native_row")),
        "simulated_native_col": _float_or_none(record.get("simulated_native_col")),
        "simulated_native_row": _float_or_none(record.get("simulated_native_row")),
        "measured_two_theta_deg": _float_or_none(record.get("measured_two_theta_deg")),
        "measured_phi_deg": _float_or_none(record.get("measured_phi_deg")),
        "simulated_two_theta_deg": _float_or_none(record.get("simulated_two_theta_deg")),
        "simulated_phi_deg": _float_or_none(record.get("simulated_phi_deg")),
        "measured_fit_space_source": (
            str(record.get("measured_fit_space_source", "") or "") or None
        ),
        "simulated_fit_space_source": (
            str(record.get("simulated_fit_space_source", "") or "") or None
        ),
        "fit_space_anchor_override": (
            bool(record.get("fit_space_anchor_override"))
            if record.get("fit_space_anchor_override") is not None
            else False
        ),
        "fit_space_projector_kind": (
            str(record.get("fit_space_projector_kind", "") or "") or None
        ),
        "cake_bundle_signature": (
            str(record.get("cake_bundle_signature", "") or "") or None
        ),
        "measured_native_frame_conversion_source": (
            str(record.get("measured_native_frame_conversion_source", "") or "") or None
        ),
        "simulated_native_frame_conversion_source": (
            str(record.get("simulated_native_frame_conversion_source", "") or "") or None
        ),
        "measured_native_frame_conversion_count": _int_or_none(
            record.get("measured_native_frame_conversion_count")
        ),
        "simulated_native_frame_conversion_count": _int_or_none(
            record.get("simulated_native_frame_conversion_count")
        ),
        "valid": (
            bool(record.get("valid"))
            if record.get("valid") is not None
            else None
        ),
        "invalid_projection_reason": (
            str(record.get("invalid_projection_reason", "") or "") or None
        ),
        "measured_invalid_projection_reason": (
            str(record.get("measured_invalid_projection_reason", "") or "") or None
        ),
        "simulated_invalid_projection_reason": (
            str(record.get("simulated_invalid_projection_reason", "") or "") or None
        ),
    }


def _residual_value(
    record: Mapping[str, object],
    residual_priority: Sequence[str],
) -> float | None:
    for field_name in residual_priority:
        value = _float_or_none(record.get(field_name))
        if value is not None:
            return value
    return None


def _build_phase_summary(
    records: Sequence[Mapping[str, object]],
    phase: str,
    *,
    matched_statuses: Sequence[str] = ("matched",),
    residual_priority: Sequence[str] = ("detector_residual_px",),
) -> dict[str, object]:
    pair_records = _phase_pair_records(records, phase)
    normalized = [_normalize_pair_record(record) for record in pair_records]
    matched_status_set = {status.lower() for status in matched_statuses}
    matched = [
        record
        for record in normalized
        if str(record["match_status"]).lower() in matched_status_set
    ]
    finite_residual_records = [
        record
        for record in matched
        if _residual_value(record, residual_priority) is not None
    ]
    finite_residuals = [
        float(_residual_value(record, residual_priority))
        for record in finite_residual_records
        if _residual_value(record, residual_priority) is not None
    ]
    unresolved_pair_ids = [
        str(record["pair_id"])
        for record in normalized
        if str(record["match_status"]).lower() not in matched_status_set
    ]
    unresolved_records = [
        record
        for record in normalized
        if str(record["match_status"]).lower() not in matched_status_set
    ]
    worst_records = sorted(
        finite_residual_records,
        key=lambda record: float(_residual_value(record, residual_priority) or 0.0),
        reverse=True,
    )[:5]
    if not worst_records and unresolved_records:
        worst_records = unresolved_records[:5]
    worst_pair_ids = [str(record["pair_id"]) for record in worst_records]
    if not worst_pair_ids and unresolved_pair_ids:
        worst_pair_ids = unresolved_pair_ids[:5]
    outside_radius_count = sum(
        1 for record in matched if bool(record.get("match_radius_exceeded", False))
    )
    return {
        "phase": phase,
        "matched_statuses": list(matched_statuses),
        "residual_priority": list(residual_priority),
        "pair_record_count": len(normalized),
        "matched_count": len(matched),
        "unresolved_count": len(unresolved_pair_ids),
        "outside_radius_count": int(outside_radius_count),
        "unresolved_pair_ids": unresolved_pair_ids,
        "rms_px": _rms(finite_residuals),
        "median_px": _median(finite_residuals),
        "max_residual_px": max(finite_residuals) if finite_residuals else None,
        "worst_pair_ids": worst_pair_ids,
        "top_outliers": worst_records,
    }


def _pair_sort_key(pair_id: object) -> tuple[int, int, str]:
    pair_text = str(pair_id or "")
    match = re.search(r"bg(?P<bg>\d+):pair(?P<pair>\d+)$", pair_text)
    if match:
        return (
            int(match.group("bg")),
            int(match.group("pair")),
            pair_text,
        )
    return (10**9, 10**9, pair_text)


def _pair_index_from_pair_id(pair_id: object) -> int | None:
    pair_text = str(pair_id or "")
    match = re.search(r"pair(?:\[(?P<bracket>\d+)\]|(?P<suffix>\d+))$", pair_text)
    if match is None:
        return None
    token = match.group("bracket") or match.group("suffix")
    if token is None:
        return None
    try:
        return int(token)
    except Exception:
        return None


def _preflight_slot_lookup(
    slot_results: Sequence[Mapping[str, object]] | None,
) -> dict[int, dict[str, object]]:
    lookup: dict[int, dict[str, object]] = {}
    for raw_slot in slot_results or ():
        if not isinstance(raw_slot, Mapping):
            continue
        try:
            slot_index = int(raw_slot.get("slot_index", -1))
        except Exception:
            continue
        if slot_index < 0:
            continue
        lookup[int(slot_index)] = dict(raw_slot)
    return lookup


def _pair_alignment_rows(
    trace_records: Sequence[Mapping[str, object]],
    *,
    before_phase: str,
    before_residual_priority: Sequence[str],
    after_phase: str,
    after_residual_priority: Sequence[str],
    preflight_slot_results: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    before_records = {
        str(record.get("pair_id", "") or ""): _normalize_pair_record(record)
        for record in _phase_pair_records(trace_records, before_phase)
    }
    after_records = {
        str(record.get("pair_id", "") or ""): _normalize_pair_record(record)
        for record in _phase_pair_records(trace_records, after_phase)
    }
    pair_ids = sorted(set(before_records) | set(after_records), key=_pair_sort_key)
    preflight_lookup = _preflight_slot_lookup(preflight_slot_results)
    rows: list[dict[str, object]] = []
    for pair_id in pair_ids:
        before = before_records.get(pair_id, {})
        after = after_records.get(pair_id, {})
        reference = after if after else before
        slot_result = preflight_lookup.get(
            _pair_index_from_pair_id(pair_id) or -1,
            {},
        )
        candidate_selection = (
            dict(slot_result.get("candidate_selection", {}))
            if isinstance(slot_result.get("candidate_selection"), Mapping)
            else {}
        )
        hkl_value = reference.get("hkl")
        hkl_out = tuple(hkl_value) if isinstance(hkl_value, list) else hkl_value
        before_distance = _residual_value(before, before_residual_priority)
        after_distance = _residual_value(after, after_residual_priority)
        selected_distance = _float_or_none(
            candidate_selection.get("background_distance_px")
        )
        minimum_distance = _float_or_none(candidate_selection.get("best_distance_px"))
        rows.append(
            {
                "pair_id": pair_id,
                "pair_index": len(rows),
                "hkl": _json_safe(hkl_out),
                "source_branch_index": reference.get("source_branch_index"),
                "source_peak_index": reference.get("source_peak_index"),
                "saved_background_current_view_point": _json_safe(
                    candidate_selection.get("saved_background_current_view_point")
                ),
                "selected_live_simulated_current_view_point": _json_safe(
                    candidate_selection.get(
                        "selected_live_simulated_current_view_point"
                    )
                ),
                "selected_to_background_distance_px": selected_distance,
                "minimum_same_hkl_branch_background_distance_px": minimum_distance,
                "selected_is_minimum_background_distance": bool(
                    selected_distance is not None
                    and minimum_distance is not None
                    and math.isfinite(float(selected_distance))
                    and math.isfinite(float(minimum_distance))
                    and abs(float(selected_distance) - float(minimum_distance))
                    <= 1.0e-6
                ),
                "measured_detector_field_name": reference.get(
                    "measured_detector_field_name"
                ),
                "measured_detector_input_frame": reference.get(
                    "measured_detector_input_frame"
                ),
                "measured_detector_frame_reason": reference.get(
                    "measured_detector_frame_reason"
                ),
                "simulated_detector_field_name": reference.get(
                    "simulated_detector_field_name"
                ),
                "simulated_detector_input_frame": reference.get(
                    "simulated_detector_input_frame"
                ),
                "simulated_detector_frame_reason": reference.get(
                    "simulated_detector_frame_reason"
                ),
                "measured_native_col": _float_or_none(reference.get("measured_native_col")),
                "measured_native_row": _float_or_none(reference.get("measured_native_row")),
                "simulated_native_col": _float_or_none(reference.get("simulated_native_col")),
                "simulated_native_row": _float_or_none(reference.get("simulated_native_row")),
                "measured_two_theta_deg": _float_or_none(reference.get("measured_two_theta_deg")),
                "measured_phi_deg": _float_or_none(reference.get("measured_phi_deg")),
                "simulated_two_theta_deg": _float_or_none(reference.get("simulated_two_theta_deg")),
                "simulated_phi_deg": _float_or_none(reference.get("simulated_phi_deg")),
                "measured_fit_space_source": reference.get("measured_fit_space_source"),
                "simulated_fit_space_source": reference.get("simulated_fit_space_source"),
                "fit_space_anchor_override": bool(
                    reference.get("fit_space_anchor_override", False)
                ),
                "fit_space_projector_kind": reference.get("fit_space_projector_kind"),
                "cake_bundle_signature": reference.get("cake_bundle_signature"),
                "measured_native_frame_conversion_source": reference.get(
                    "measured_native_frame_conversion_source"
                ),
                "simulated_native_frame_conversion_source": reference.get(
                    "simulated_native_frame_conversion_source"
                ),
                "measured_native_frame_conversion_count": _int_or_none(
                    reference.get("measured_native_frame_conversion_count")
                ),
                "simulated_native_frame_conversion_count": _int_or_none(
                    reference.get("simulated_native_frame_conversion_count")
                ),
                "valid": reference.get("valid"),
                "invalid_projection_reason": reference.get("invalid_projection_reason"),
                "measured_invalid_projection_reason": reference.get(
                    "measured_invalid_projection_reason"
                ),
                "simulated_invalid_projection_reason": reference.get(
                    "simulated_invalid_projection_reason"
                ),
                "residual_before": before_distance,
                "residual_after": after_distance,
                "before_dx_px": _float_or_none(before.get("dx_px")),
                "before_dy_px": _float_or_none(before.get("dy_px")),
                "before_distance_px": before_distance,
                "after_dx_px": _float_or_none(after.get("dx_px")),
                "after_dy_px": _float_or_none(after.get("dy_px")),
                "after_distance_px": after_distance,
                "improved": bool(
                    before_distance is not None
                    and after_distance is not None
                    and after_distance < before_distance
                ),
            }
        )
    return rows


def validate_manual_caked_fit_space_provenance(
    report_or_rows: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
) -> list[str]:
    has_report_metadata = isinstance(report_or_rows, Mapping)
    if isinstance(report_or_rows, Mapping):
        report = dict(report_or_rows)
        rows = [
            dict(entry)
            for entry in list(report.get("pair_alignment_rows", []) or [])
            if isinstance(entry, Mapping)
        ]
        state_name = str(report.get("state_name", "") or "")
        exact_available = bool(report.get("exact_fit_space_projector_available", False))
        exact_reason = str(report.get("exact_fit_space_projection_reason", "") or "")
        manual_row_count = _int_or_none(report.get("manual_caked_residual_row_count")) or 0
        dataset_projector_row_count = (
            _int_or_none(report.get("dataset_fit_space_projector_row_count")) or 0
        )
        invalid_projector_row_count = (
            _int_or_none(report.get("invalid_dataset_fit_space_projector_row_count")) or 0
        )
        analytic_row_count = (
            _int_or_none(report.get("analytic_detector_fit_space_row_count")) or 0
        )
    else:
        report = {}
        rows = [
            dict(entry)
            for entry in list(report_or_rows or [])
            if isinstance(entry, Mapping)
        ]
        state_name = ""
        exact_available = False
        exact_reason = ""
        manual_row_count = len(rows)
        dataset_projector_row_count = 0
        invalid_projector_row_count = 0
        analytic_row_count = 0

    violations: list[str] = []
    state_requires_fixed_manual_caked = state_name.lower() in {"bi2se3", "bi2te3"}
    has_fixed_manual_caked_evidence = any(
        count > 0
        for count in (
            manual_row_count,
            dataset_projector_row_count,
            invalid_projector_row_count,
            analytic_row_count,
        )
    )
    validate_fixed_manual_caked = bool(
        state_requires_fixed_manual_caked or has_fixed_manual_caked_evidence
    )
    if has_report_metadata and validate_fixed_manual_caked and manual_row_count <= 0:
        violations.append("manual_caked_residual_row_count is empty")
    if has_report_metadata and validate_fixed_manual_caked and not exact_available:
        violations.append("exact_fit_space_projector_available is false")
    if has_report_metadata and validate_fixed_manual_caked and dataset_projector_row_count <= 0:
        violations.append("dataset_fit_space_projector_row_count is empty")
    if has_report_metadata and validate_fixed_manual_caked and invalid_projector_row_count > 0:
        violations.append("invalid_dataset_fit_space_projector_row_count is nonzero")
    if has_report_metadata and validate_fixed_manual_caked and analytic_row_count > 0:
        violations.append("analytic_detector_fit_space_row_count is nonzero")
    if has_report_metadata and not exact_available and not exact_reason:
        violations.append(
            "exact_fit_space_projector_available is false but reason is missing"
        )

    for row in rows:
        pair_id = str(row.get("pair_id", "") or "<unknown>")
        projector_kind = str(row.get("fit_space_projector_kind", "") or "")
        point_only_dynamic_row = projector_kind == "point_only_dynamic_source_row"
        if (
            row.get("selected_is_minimum_background_distance") is False
            and not point_only_dynamic_row
        ):
            violations.append(
                f"{pair_id}: selected_is_minimum_background_distance is false"
            )
        exact_row = projector_kind in CAKED_FIT_SPACE_PROJECTOR_KINDS
        if not exact_row:
            continue
        for field_name in (
            "measured_fit_space_source",
            "simulated_fit_space_source",
            "measured_detector_input_frame",
            "simulated_detector_input_frame",
            "measured_native_frame_conversion_count",
            "simulated_native_frame_conversion_count",
            "fit_space_projector_kind",
        ):
            value = row.get(field_name)
            if value in (None, ""):
                violations.append(f"{pair_id}: missing required field {field_name}")
        if not point_only_dynamic_row and row.get("cake_bundle_signature") in (None, ""):
            violations.append(f"{pair_id}: missing required field cake_bundle_signature")
        measured_source = str(row.get("measured_fit_space_source", "") or "")
        simulated_source = str(row.get("simulated_fit_space_source", "") or "")
        measured_frame = str(row.get("measured_detector_input_frame", "") or "")
        simulated_frame = str(row.get("simulated_detector_input_frame", "") or "")
        measured_count = _int_or_none(row.get("measured_native_frame_conversion_count"))
        simulated_count = _int_or_none(row.get("simulated_native_frame_conversion_count"))
        explicit_override = bool(row.get("fit_space_anchor_override", False))

        expected_simulated_source = (
            "sim_visual_caked_deg"
            if point_only_dynamic_row
            else "dataset_fit_space_projector"
        )
        if simulated_source != expected_simulated_source:
            violations.append(
                f"{pair_id}: simulated_fit_space_source={simulated_source!r}"
            )
        if explicit_override:
            if measured_frame != "explicit_override":
                violations.append(
                    f"{pair_id}: fit_space_anchor_override requires explicit_override frame"
                )
            if measured_source != "cached_fit_space_anchor":
                violations.append(
                    f"{pair_id}: fit_space_anchor_override requires cached_fit_space_anchor"
                )
        elif measured_frame == "explicit_override" and not point_only_dynamic_row:
            violations.append(
                f"{pair_id}: explicit_override frame requires fit_space_anchor_override"
            )
        else:
            expected_measured_source = (
                "cached_fit_space_anchor"
                if point_only_dynamic_row
                else "dataset_fit_space_projector"
            )
            if measured_source != expected_measured_source:
                violations.append(
                    f"{pair_id}: measured_fit_space_source={measured_source!r}"
                )
        if "analytic_detector_fit_space" in {measured_source, simulated_source}:
            violations.append(f"{pair_id}: analytic_detector_fit_space present on exact row")
        if (
            "invalid_dataset_fit_space_projector" in {measured_source, simulated_source}
            and "analytic_detector_fit_space" in {measured_source, simulated_source}
        ):
            violations.append(
                f"{pair_id}: invalid_dataset_fit_space_projector coexists with analytic fallback"
            )
        if measured_source == "invalid_dataset_fit_space_projector" and not str(
            row.get("measured_invalid_projection_reason", "")
            or row.get("invalid_projection_reason", "")
        ).strip():
            violations.append(f"{pair_id}: missing measured invalid projection reason")
        if simulated_source == "invalid_dataset_fit_space_projector" and not str(
            row.get("simulated_invalid_projection_reason", "")
            or row.get("invalid_projection_reason", "")
        ).strip():
            violations.append(f"{pair_id}: missing simulated invalid projection reason")

        if measured_frame not in {"native_detector", "fit_detector", "explicit_override"}:
            violations.append(f"{pair_id}: invalid measured_detector_input_frame={measured_frame!r}")
        allowed_simulated_frames = {"native_detector", "fit_detector"}
        if point_only_dynamic_row:
            allowed_simulated_frames.add("sim_visual_caked_deg")
        if simulated_frame not in allowed_simulated_frames:
            violations.append(f"{pair_id}: invalid simulated_detector_input_frame={simulated_frame!r}")
        if measured_frame == "native_detector" and measured_count != 0:
            violations.append(f"{pair_id}: measured native frame count should be 0")
        if measured_frame == "fit_detector" and measured_count != 1:
            violations.append(f"{pair_id}: measured fit frame count should be 1")
        if measured_frame == "explicit_override" and measured_source != "cached_fit_space_anchor":
            violations.append(f"{pair_id}: explicit override does not use cached_fit_space_anchor")
        if simulated_frame == "native_detector" and simulated_count != 0:
            violations.append(f"{pair_id}: simulated native frame count should be 0")
        if simulated_frame == "fit_detector" and simulated_count != 1:
            violations.append(f"{pair_id}: simulated fit frame count should be 1")

    return violations


def _count_reason_values(
    records: Sequence[Mapping[str, object]],
    field_name: str,
) -> dict[str, int]:
    counts = Counter(
        str(record.get(field_name) or "unknown")
        for record in records
    )
    return dict(sorted(counts.items()))


def _build_identity_retention_summary(
    records: Sequence[Mapping[str, object]],
    phase: str,
    *,
    matched_statuses: Sequence[str] = ("matched",),
) -> dict[str, object]:
    pair_records = _phase_pair_records(records, phase)
    normalized = [_normalize_pair_record(record) for record in pair_records]
    matched_status_set = {status.lower() for status in matched_statuses}
    matched = [
        record
        for record in normalized
        if str(record["match_status"]).lower() in matched_status_set
    ]
    source_branch_records = [
        record for record in normalized if record.get("source_branch_index") is not None
    ]
    source_peak_records = [
        record for record in normalized if record.get("source_peak_index") is not None
    ]
    matched_with_resolved_peak = [
        record for record in matched if record.get("resolved_peak_index") is not None
    ]
    branch_retained = [
        record
        for record in matched_with_resolved_peak
        if record.get("source_branch_index") is not None
        and int(record["source_branch_index"]) == int(record["resolved_peak_index"])
    ]
    branch_mismatched = [
        record
        for record in matched_with_resolved_peak
        if record.get("source_branch_index") is not None
        and int(record["source_branch_index"]) != int(record["resolved_peak_index"])
    ]
    peak_retained = [
        record
        for record in matched_with_resolved_peak
        if record.get("source_peak_index") is not None
        and int(record["source_peak_index"]) == int(record["resolved_peak_index"])
    ]
    peak_mismatched = [
        record
        for record in matched_with_resolved_peak
        if record.get("source_peak_index") is not None
        and int(record["source_peak_index"]) != int(record["resolved_peak_index"])
    ]
    unresolved_identity_records = [
        record
        for record in normalized
        if (
            record.get("source_branch_index") is not None
            or record.get("source_peak_index") is not None
        )
        and (
            str(record["match_status"]).lower() not in matched_status_set
            or record.get("resolved_peak_index") is None
        )
    ]
    issue_preview = (branch_mismatched + peak_mismatched + unresolved_identity_records)[:5]
    return {
        "phase": phase,
        "matched_statuses": list(matched_statuses),
        "pair_record_count": len(normalized),
        "matched_pair_count": len(matched),
        "source_branch_count": len(source_branch_records),
        "source_peak_count": len(source_peak_records),
        "resolved_peak_count": len(
            [record for record in normalized if record.get("resolved_peak_index") is not None]
        ),
        "matched_with_resolved_peak_count": len(matched_with_resolved_peak),
        "branch_retained_count": len(branch_retained),
        "branch_mismatch_count": len(branch_mismatched),
        "branch_unresolved_count": len(source_branch_records) - len(branch_retained) - len(branch_mismatched),
        "peak_retained_count": len(peak_retained),
        "peak_mismatch_count": len(peak_mismatched),
        "peak_unresolved_count": len(source_peak_records) - len(peak_retained) - len(peak_mismatched),
        "matched_resolution_kind_counts": _count_reason_values(matched, "resolution_kind"),
        "matched_resolution_reason_counts": _count_reason_values(matched, "resolution_reason"),
        "unmatched_resolution_reason_counts": _count_reason_values(
            [
                record
                for record in normalized
                if str(record["match_status"]).lower() not in matched_status_set
            ],
            "resolution_reason",
        ),
        "issue_preview": issue_preview,
    }


def _parse_log_artifacts(path: Path | None) -> dict[str, object]:
    parsed: dict[str, object] = {
        "parameter_entries": [],
        "bound_hits": [],
        "boundary_warning": None,
        "direct_solver_evals": [],
        "direct_solver_summary": {},
        "final_summary": {},
        "optimizer_summary": {},
        "stage_summaries": {},
    }
    if path is None or not path.is_file():
        return parsed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Geometry fit: running fixed-manual-pair direct least-squares solve"):
            parsed["direct_solver_evals"] = []
            parsed["direct_solver_summary"] = {}
            continue
        direct_eval_match = DIRECT_SOLVE_EVAL_RE.match(line)
        if direct_eval_match:
            groups = direct_eval_match.groupdict()
            evals = list(parsed.get("direct_solver_evals", []) or [])
            evals.append(
                {
                    "eval": _int_or_none(groups["eval"]),
                    "cost": _float_or_none(groups["cost"]),
                    "best_cost": _float_or_none(groups["best_cost"]),
                    "weighted_rms_px": _float_or_none(groups["weighted_rms"]),
                }
            )
            parsed["direct_solver_evals"] = evals
            continue
        direct_progress_match = DIRECT_SOLVE_PROGRESS_RE.match(line)
        if direct_progress_match:
            groups = direct_progress_match.groupdict()
            parsed["direct_solver_summary"] = {
                "evaluations": _int_or_none(groups["evaluations"]),
                "best_cost": _float_or_none(groups["best_cost"]),
                "last_cost": _float_or_none(groups["last_cost"]),
                "best_weighted_rms_px": _float_or_none(groups["best_weighted_rms_px"]),
                "last_weighted_rms_px": _float_or_none(groups["last_weighted_rms_px"]),
            }
            continue
        param_match = PARAM_LINE_RE.match(line)
        if param_match:
            groups = param_match.groupdict()
            parsed["parameter_entries"].append(
                {
                    "name": groups["name"],
                    "group": groups["group"],
                    "start": _float_or_none(groups["start"]),
                    "final": _float_or_none(groups["final"]),
                    "delta": _float_or_none(groups["delta"]),
                    "lower_bound": _float_or_none(groups["lower"]),
                    "upper_bound": _float_or_none(groups["upper"]),
                    "scale": _float_or_none(groups["scale"]),
                    "prior_center": _float_or_none(groups.get("prior_center")),
                    "prior_sigma": _float_or_none(groups.get("prior_sigma")),
                }
            )
            continue
        if line.startswith("bound_hits=[") and line.endswith("]"):
            inner = line[len("bound_hits=[") : -1].strip()
            parsed["bound_hits"] = [item.strip() for item in inner.split(",") if item.strip()]
            continue
        if line.startswith("boundary_warning="):
            warning = line.partition("=")[2].strip()
            parsed["boundary_warning"] = warning or None
            continue
        if line.startswith("final metric="):
            final_summary = dict(parsed.get("final_summary", {}))
            for token in line.split():
                if "=" not in token:
                    continue
                key, _, raw_value = token.partition("=")
                if key == "final":
                    continue
                if key == "metric":
                    final_summary["metric_name"] = raw_value
                elif key in {
                    "cost",
                    "robust_cost",
                    "weighted_rms_px",
                    "final_full_beam_rms_px",
                }:
                    final_summary[key] = _float_or_none(raw_value)
            parsed["final_summary"] = final_summary
            continue
        if line.startswith("full_beam_polish:"):
            stage_summaries = dict(parsed.get("stage_summaries", {}))
            stage_summaries["full_beam_polish"] = line
            parsed["stage_summaries"] = stage_summaries
            continue
        if "=" in line and not line.startswith("param[") and " " not in line.partition("=")[0]:
            key, _, raw_value = line.partition("=")
            if key in {
                "success",
                "status",
                "message",
                "nfev",
                "cost",
                "robust_cost",
                "solver_loss",
                "solver_f_scale",
                "optimality",
                "optimizer_method",
                "weighted_residual_rms_px",
                "display_rms_px",
                "final_metric_name",
            }:
                optimizer_summary = dict(parsed.get("optimizer_summary", {}))
                value = (
                    _float_or_none(raw_value)
                    if key
                    in {
                        "cost",
                        "robust_cost",
                        "solver_f_scale",
                        "optimality",
                        "weighted_residual_rms_px",
                        "display_rms_px",
                    }
                    else raw_value
                )
                optimizer_summary[key] = value
                parsed["optimizer_summary"] = optimizer_summary
    return parsed


def _resolve_artifact_paths(
    *,
    state_path: Path,
    run_dir: Path,
    cli_stdout_path: Path,
    cli_stderr_path: Path,
    allow_default_live_cache: bool = False,
    started_at_wall_time: float | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    stdout_text = cli_stdout_path.read_text(encoding="utf-8") if cli_stdout_path.is_file() else ""
    stderr_text = cli_stderr_path.read_text(encoding="utf-8") if cli_stderr_path.is_file() else ""
    log_path = _latest_path(run_dir, "geometry_fit_log_*.txt")
    trace_path = _latest_path(run_dir, "geometry_fit_trace_*.jsonl")
    matched_peaks_path = _latest_path(run_dir, "matched_peaks_*.npy")
    if not _is_current_run_artifact(log_path, started_at_wall_time=started_at_wall_time):
        log_path = None
    if not _is_current_run_artifact(trace_path, started_at_wall_time=started_at_wall_time):
        trace_path = None
    if not _is_current_run_artifact(
        matched_peaks_path,
        started_at_wall_time=started_at_wall_time,
    ):
        matched_peaks_path = None
    default_log_path = _default_live_log_path(state_path)
    default_trace_path = _default_live_trace_path(state_path)

    reported_log_path = _extract_reported_artifact_path(stdout_text, "Geometry fit log")
    if reported_log_path is None:
        reported_log_path = _extract_reported_artifact_path(stderr_text, "Geometry fit log")
    if reported_log_path is not None and reported_log_path.is_file():
        mirrored_log_path = _mirror_artifact_into_run_dir(reported_log_path, run_dir=run_dir)
        if _is_current_run_artifact(
            mirrored_log_path,
            started_at_wall_time=started_at_wall_time,
        ):
            log_path = mirrored_log_path
    if log_path is None and allow_default_live_cache:
        mirrored_default_log_path = _mirror_artifact_into_run_dir(default_log_path, run_dir=run_dir)
        if _is_current_run_artifact(
            mirrored_default_log_path,
            started_at_wall_time=started_at_wall_time,
        ):
            log_path = mirrored_default_log_path

    if log_path is not None:
        reported_trace_path = _parse_trace_path_from_log(log_path)
        if reported_trace_path is None:
            reported_trace_path = _latest_path(log_path.parent, f"geometry_fit_trace_{state_path.stem}*.jsonl")
        mirrored_trace_path = _mirror_artifact_into_run_dir(reported_trace_path, run_dir=run_dir)
        if _is_current_run_artifact(
            mirrored_trace_path,
            started_at_wall_time=started_at_wall_time,
        ):
            trace_path = mirrored_trace_path
        if matched_peaks_path is None:
            reported_matched_peaks_path = _extract_reported_artifact_path(stdout_text, "Matched peaks")
            if reported_matched_peaks_path is None:
                reported_matched_peaks_path = _latest_path(
                    log_path.parent,
                    f"matched_peaks_{state_path.stem}*.npy",
                )
            mirrored_matched_peaks_path = _mirror_artifact_into_run_dir(
                reported_matched_peaks_path,
                run_dir=run_dir,
            )
            if _is_current_run_artifact(
                mirrored_matched_peaks_path,
                started_at_wall_time=started_at_wall_time,
            ):
                matched_peaks_path = mirrored_matched_peaks_path
    if trace_path is None and allow_default_live_cache:
        mirrored_default_trace_path = _mirror_artifact_into_run_dir(
            default_trace_path,
            run_dir=run_dir,
        )
        if _is_current_run_artifact(
            mirrored_default_trace_path,
            started_at_wall_time=started_at_wall_time,
        ):
            trace_path = mirrored_default_trace_path
    if matched_peaks_path is None and allow_default_live_cache:
        mirrored_default_matched_peaks_path = _mirror_artifact_into_run_dir(
            _latest_path(DEFAULT_CACHE_LOG_DIR, f"matched_peaks_{state_path.stem}*.npy"),
            run_dir=run_dir,
        )
        if _is_current_run_artifact(
            mirrored_default_matched_peaks_path,
            started_at_wall_time=started_at_wall_time,
        ):
            matched_peaks_path = mirrored_default_matched_peaks_path

    return log_path, trace_path, matched_peaks_path


def _infer_existing_cli_returncode(
    *,
    cli_stdout_path: Path,
    cli_stderr_path: Path,
    out_state_path: Path,
) -> int:
    if out_state_path.is_file():
        return 0
    stderr_text = cli_stderr_path.read_text(encoding="utf-8") if cli_stderr_path.is_file() else ""
    stdout_text = cli_stdout_path.read_text(encoding="utf-8") if cli_stdout_path.is_file() else ""
    if _last_nonempty_line(stderr_text) is not None:
        return 1
    if "Geometry fit unavailable:" in stdout_text:
        return 1
    return 0


def _load_matched_peaks_preview(path: Path | None, *, limit: int = 5) -> dict[str, object]:
    if path is None or not path.is_file():
        return {"record_count": 0, "preview": []}
    records = np.load(path, allow_pickle=True)
    values = records.tolist() if isinstance(records, np.ndarray) else list(records)
    return {
        "record_count": len(values),
        "preview": [_json_safe(item) for item in values[:limit]],
    }


def _candidate_preflight_report_paths(state_path: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    stem = state_path.stem
    for suffix in ("_fresh_all", "_fresh"):
        if stem.endswith(suffix):
            base_stem = stem[: -len(suffix)]
            candidates.append(state_path.with_name(f"{base_stem}_preflight_report.json"))
            break
    candidates.append(state_path.with_name(f"{state_path.stem}_preflight_report.json"))
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)
    return tuple(unique_candidates)


def _load_preflight_summary(state_path: Path) -> dict[str, object]:
    for candidate in _candidate_preflight_report_paths(state_path):
        if not candidate.is_file():
            continue
        try:
            payload = _load_json(candidate)
        except Exception:
            continue
        slot_results = (
            list(payload.get("slot_results", []))
            if isinstance(payload.get("slot_results"), Sequence)
            and not isinstance(payload.get("slot_results"), (str, bytes, bytearray))
            else []
        )
        bound_manual_entry_count = _int_or_none(payload.get("bound_manual_entry_count"))
        if bound_manual_entry_count is None:
            bound_manual_entry_count = sum(
                1
                for item in slot_results
                if isinstance(item, Mapping) and bool(item.get("ok", False))
            )
        processed_manual_entry_count = _int_or_none(
            payload.get("processed_manual_entry_count")
        )
        if processed_manual_entry_count is None:
            processed_manual_entry_count = len(slot_results)
        return {
            "path": str(candidate),
            "ok": bool(payload.get("ok", False)),
            "classification": payload.get("classification"),
            "bound_manual_entry_count": int(max(0, bound_manual_entry_count or 0)),
            "processed_manual_entry_count": int(max(0, processed_manual_entry_count or 0)),
            "dataset_resolved_source_pair_count": _int_or_none(
                payload.get("dataset_resolved_source_pair_count")
            ),
            "background_distance_gate_ok": bool(
                payload.get("background_distance_gate_ok", False)
            ),
            "branch_mismatch_count": _int_or_none(payload.get("branch_mismatch_count")),
            "missing_manual_entry_count": _int_or_none(
                payload.get("missing_manual_entry_count")
            ),
            "slot_results": [
                dict(item) for item in slot_results if isinstance(item, Mapping)
            ],
        }
    return {
        "path": None,
        "ok": None,
        "classification": None,
        "bound_manual_entry_count": None,
        "processed_manual_entry_count": None,
        "dataset_resolved_source_pair_count": None,
        "background_distance_gate_ok": None,
        "branch_mismatch_count": None,
        "missing_manual_entry_count": None,
        "slot_results": [],
    }


def _background_context(state_path: Path, out_state_path: Path | None) -> dict[str, object]:
    source_payload = _load_json(state_path)
    source_state = source_payload.get("state", {})
    files = source_state.get("files", {}) if isinstance(source_state, Mapping) else {}
    geometry = source_state.get("geometry", {}) if isinstance(source_state, Mapping) else {}
    backgrounds = list(files.get("background_files", [])) if isinstance(files, Mapping) else []
    current_index = int(files.get("current_background_index", 0) or 0) if isinstance(files, Mapping) else 0
    manual_pair_groups = (
        list(geometry.get("manual_pairs", []))
        if isinstance(geometry, Mapping)
        else []
    )
    current_manual_pair_group = next(
        (
            dict(raw_group)
            for raw_group in manual_pair_groups
            if isinstance(raw_group, Mapping)
            and int(raw_group.get("background_index", -1)) == current_index
        ),
        {},
    )
    current_manual_pair_entries = (
        list(current_manual_pair_group.get("entries", []))
        if isinstance(current_manual_pair_group.get("entries"), Sequence)
        and not isinstance(current_manual_pair_group.get("entries"), (str, bytes, bytearray))
        else []
    )
    preflight_summary = _load_preflight_summary(state_path)
    out_state_exists = out_state_path is not None and out_state_path.is_file()
    return {
        "source_state_path": str(state_path),
        "fitted_state_path": str(out_state_path) if out_state_exists else None,
        "background_files": [str(item) for item in backgrounds],
        "current_background_index": current_index,
        "current_background_file": (
            str(backgrounds[current_index]) if 0 <= current_index < len(backgrounds) else None
        ),
        "current_background_manual_pair_count": int(len(current_manual_pair_entries)),
        "current_background_manual_pair_group_present": bool(current_manual_pair_group),
        "preflight_report_path": preflight_summary.get("path"),
        "preflight_ok": preflight_summary.get("ok"),
        "preflight_classification": preflight_summary.get("classification"),
        "preflight_valid_count": preflight_summary.get("bound_manual_entry_count"),
        "preflight_processed_manual_entry_count": preflight_summary.get(
            "processed_manual_entry_count"
        ),
        "preflight_dataset_resolved_source_pair_count": preflight_summary.get(
            "dataset_resolved_source_pair_count"
        ),
        "preflight_background_distance_gate_ok": preflight_summary.get(
            "background_distance_gate_ok"
        ),
        "preflight_branch_mismatch_count": preflight_summary.get(
            "branch_mismatch_count"
        ),
        "preflight_missing_manual_entry_count": preflight_summary.get(
            "missing_manual_entry_count"
        ),
        "preflight_slot_results": list(preflight_summary.get("slot_results", []) or []),
    }


def _saved_state_gate_summary(report: Mapping[str, object]) -> dict[str, object]:
    decision = dict(report.get("decision_row", {}))
    run_summary = (
        dict(report.get("run_summary", {}))
        if isinstance(report.get("run_summary"), Mapping)
        else {}
    )
    artifact_summary = (
        dict(report.get("artifacts", {}))
        if isinstance(report.get("artifacts"), Mapping)
        else {}
    )
    before_fit = dict(report.get("before_fit", {})) if isinstance(report.get("before_fit"), Mapping) else {}
    after_fit = dict(report.get("after_fit", {})) if isinstance(report.get("after_fit"), Mapping) else {}
    background_context = (
        dict(report.get("background_context", {}))
        if isinstance(report.get("background_context"), Mapping)
        else {}
    )

    preflight_report_path = str(background_context.get("preflight_report_path") or "").strip()
    preflight_valid_count = _int_or_none(background_context.get("preflight_valid_count"))
    preflight_valid_count_source = "preflight_report"
    if preflight_valid_count is None and preflight_report_path:
        preflight_valid_count = int(
            background_context.get("current_background_manual_pair_count", 0) or 0
        )
        preflight_valid_count_source = "manual_pair_count_fallback"
    elif preflight_valid_count is None:
        preflight_valid_count = 0
        preflight_valid_count_source = "missing_preflight_report"
    preflight_ok = background_context.get("preflight_ok")
    matched_fixed_pair_count_before = int(
        run_summary.get(
            "matched_fixed_pair_count_before",
            decision.get("matched_count_before", before_fit.get("matched_count", 0)),
        )
        or 0
    )
    matched_fixed_pair_count_after = int(
        run_summary.get(
            "matched_fixed_pair_count_after",
            decision.get("matched_count_after", after_fit.get("matched_count", 0)),
        )
        or 0
    )
    missing_fixed_pair_count_before = max(
        0,
        preflight_valid_count - matched_fixed_pair_count_before,
    )
    missing_fixed_pair_count_after = int(
        run_summary.get(
            "missing_fixed_pair_count_after",
            after_fit.get("unresolved_count", 0),
        )
        or 0
    )
    branch_mismatch_count = int(
        run_summary.get("branch_mismatch_count", 0) or 0
    )
    before_rms_px = _float_or_none(decision.get("rms_before_px", before_fit.get("rms_px")))
    after_rms_px = _float_or_none(decision.get("rms_after_px", after_fit.get("rms_px")))
    before_max_px = _float_or_none(
        decision.get("max_residual_before_px", before_fit.get("max_residual_px"))
    )
    after_max_px = _float_or_none(
        decision.get("max_residual_after_px", after_fit.get("max_residual_px"))
    )
    before_median_px = _float_or_none(
        decision.get("median_before_px", before_fit.get("median_px"))
    )
    after_median_px = _float_or_none(
        decision.get("median_after_px", after_fit.get("median_px"))
    )
    outside_radius_count_before = int(
        run_summary.get("outside_radius_count_before", before_fit.get("outside_radius_count", 0))
        or 0
    )
    outside_radius_count_after = int(
        run_summary.get("outside_radius_count_after", after_fit.get("outside_radius_count", 0))
        or 0
    )
    fit_quality_passed = bool(run_summary.get("fit_quality_passed", False))
    selection_status = str(run_summary.get("selection_status", "") or "")
    retained_start_safe = selection_status == "retained_start_safe_fallback"
    selected_candidate_name = run_summary.get("selected_candidate_name")
    selected_candidate_source = run_summary.get("selected_candidate_source")
    best_valid_raw_detector_candidate_name = run_summary.get(
        "best_valid_raw_detector_candidate_name"
    )
    best_valid_raw_detector_candidate_source = run_summary.get(
        "best_valid_raw_detector_candidate_source"
    )

    def _normalized_candidate_id(value: object) -> str:
        try:
            return str(value or "").strip()
        except Exception:
            return ""

    normalized_selected_candidate_name = _normalized_candidate_id(selected_candidate_name)
    normalized_selected_candidate_source = _normalized_candidate_id(selected_candidate_source)
    normalized_best_candidate_name = _normalized_candidate_id(best_valid_raw_detector_candidate_name)
    normalized_best_candidate_source = _normalized_candidate_id(
        best_valid_raw_detector_candidate_source
    )
    selected_candidate_matches_best_valid_raw = bool(
        normalized_selected_candidate_name
        and normalized_selected_candidate_source
        and normalized_best_candidate_name
        and normalized_best_candidate_source
        and normalized_selected_candidate_name == normalized_best_candidate_name
        and normalized_selected_candidate_source == normalized_best_candidate_source
    )
    rejection_reason = str(decision.get("rejection_reason") or "")
    timed_out = bool(artifact_summary.get("timed_out", False))
    timeout_reason = str(artifact_summary.get("timeout_reason") or "").strip()
    raw_alignment_checks_pass = bool(
        not timed_out
        and preflight_report_path
        and preflight_ok is True
        and preflight_valid_count > 0
        and matched_fixed_pair_count_after == preflight_valid_count
        and missing_fixed_pair_count_after <= missing_fixed_pair_count_before
        and branch_mismatch_count == 0
        and rejection_reason != NO_MATCHED_FIXED_PAIR_REJECTION
        and before_rms_px is not None
        and after_rms_px is not None
        and after_rms_px <= before_rms_px + SAVED_STATE_RMS_TOLERANCE_PX
        and before_max_px is not None
        and after_max_px is not None
        and after_max_px <= before_max_px + SAVED_STATE_MAX_ERROR_TOLERANCE_PX
        and outside_radius_count_after <= outside_radius_count_before
        and selected_candidate_matches_best_valid_raw
    )
    direct_initial_rms_px = _float_or_none(run_summary.get("direct_initial_rms_px"))
    direct_final_rms_px = _float_or_none(run_summary.get("direct_final_rms_px"))
    if (
        preflight_valid_count <= 0
        and direct_initial_rms_px is not None
        and direct_final_rms_px is not None
        and matched_fixed_pair_count_before > 0
    ):
        preflight_valid_count = int(matched_fixed_pair_count_before)
        preflight_valid_count_source = "matched_fixed_pair_count_before"
        if preflight_ok is None:
            preflight_ok = True
        missing_fixed_pair_count_before = max(
            0,
            preflight_valid_count - matched_fixed_pair_count_before,
        )
    fixed_manual_caked_checks_pass = bool(
        not timed_out
        and bool(decision.get("accepted", False))
        and preflight_valid_count > 0
        and matched_fixed_pair_count_after == preflight_valid_count
        and missing_fixed_pair_count_after == 0
        and branch_mismatch_count == 0
        and rejection_reason != NO_MATCHED_FIXED_PAIR_REJECTION
        and direct_initial_rms_px is not None
        and direct_final_rms_px is not None
        and direct_final_rms_px < direct_initial_rms_px
    )
    failures: list[str] = []
    if timed_out:
        failures.append(timeout_reason or "fit run timed out")
    if not preflight_report_path and not fixed_manual_caked_checks_pass:
        failures.append("preflight report was unavailable")
    if preflight_ok is not True and not fixed_manual_caked_checks_pass:
        failures.append("preflight report was not ok")
    if preflight_valid_count <= 0 and not fixed_manual_caked_checks_pass:
        failures.append("preflight valid pair count was 0")
    if matched_fixed_pair_count_after != preflight_valid_count:
        failures.append(
            "matched fixed pairs after fit did not equal preflight-valid count "
            f"({matched_fixed_pair_count_after} != {preflight_valid_count})"
        )
    if missing_fixed_pair_count_after > missing_fixed_pair_count_before:
        failures.append(
            "missing fixed pairs increased after fit "
            f"({missing_fixed_pair_count_after} > {missing_fixed_pair_count_before})"
        )
    if branch_mismatch_count != 0:
        failures.append(f"branch mismatch count was {branch_mismatch_count}")
    if (
        not fit_quality_passed
        and not (retained_start_safe and raw_alignment_checks_pass)
        and not fixed_manual_caked_checks_pass
    ):
        failures.append(
            f"fit quality did not pass raw detector gate (selection_status={selection_status or 'unknown'})"
        )
    if not selected_candidate_matches_best_valid_raw and not fixed_manual_caked_checks_pass:
        failures.append("selected candidate was not the best valid raw detector candidate")
    if rejection_reason == NO_MATCHED_FIXED_PAIR_REJECTION:
        failures.append("top-level rejection reported no matched peak pairs")
    if fixed_manual_caked_checks_pass:
        pass
    elif before_rms_px is None or after_rms_px is None:
        failures.append("before/after RMS metrics were unavailable")
    elif after_rms_px > before_rms_px + SAVED_STATE_RMS_TOLERANCE_PX:
        failures.append(
            "after RMS exceeded before-fit RMS tolerance "
            f"({after_rms_px:.3f} > {before_rms_px:.3f} + {SAVED_STATE_RMS_TOLERANCE_PX:.3f})"
        )
    if fixed_manual_caked_checks_pass:
        pass
    elif before_max_px is None or after_max_px is None:
        failures.append("before/after max-error metrics were unavailable")
    elif after_max_px > before_max_px + SAVED_STATE_MAX_ERROR_TOLERANCE_PX:
        failures.append(
            "after max error exceeded before-fit max-error tolerance "
            f"({after_max_px:.3f} > {before_max_px:.3f} + {SAVED_STATE_MAX_ERROR_TOLERANCE_PX:.3f})"
        )
    if outside_radius_count_after > outside_radius_count_before and not fixed_manual_caked_checks_pass:
        failures.append(
            "outside-radius pair count increased "
            f"({outside_radius_count_after} > {outside_radius_count_before})"
        )
    payload = {
        "ok": not failures,
        "failures": failures,
        "timed_out": timed_out,
        "timeout_reason": timeout_reason or None,
        "preflight_valid_count": preflight_valid_count,
        "preflight_valid_count_source": preflight_valid_count_source,
        "matched_fixed_pair_count_before": matched_fixed_pair_count_before,
        "matched_fixed_pair_count_after": matched_fixed_pair_count_after,
        "missing_fixed_pair_count_before": missing_fixed_pair_count_before,
        "missing_fixed_pair_count_after": missing_fixed_pair_count_after,
        "branch_mismatch_count": branch_mismatch_count,
        "before_rms_px": before_rms_px,
        "after_rms_px": after_rms_px,
        "before_median_px": before_median_px,
        "after_median_px": after_median_px,
        "before_max_px": before_max_px,
        "after_max_px": after_max_px,
        "outside_radius_count_before": outside_radius_count_before,
        "outside_radius_count_after": outside_radius_count_after,
        "fit_quality_passed": fit_quality_passed,
        "selection_status": selection_status,
        "selected_candidate_name": selected_candidate_name,
        "selected_candidate_source": selected_candidate_source,
        "best_valid_raw_detector_candidate_name": best_valid_raw_detector_candidate_name,
        "best_valid_raw_detector_candidate_source": best_valid_raw_detector_candidate_source,
        "selected_candidate_matches_best_valid_raw": selected_candidate_matches_best_valid_raw,
        "rms_tolerance_px": SAVED_STATE_RMS_TOLERANCE_PX,
        "max_error_tolerance_px": SAVED_STATE_MAX_ERROR_TOLERANCE_PX,
    }
    if direct_initial_rms_px is not None or direct_final_rms_px is not None:
        payload.update(
            {
                "fixed_manual_caked_checks_pass": fixed_manual_caked_checks_pass,
                "direct_initial_rms_px": direct_initial_rms_px,
                "direct_final_rms_px": direct_final_rms_px,
                "direct_residual_reduced": bool(
                    direct_initial_rms_px is not None
                    and direct_final_rms_px is not None
                    and direct_final_rms_px < direct_initial_rms_px
                ),
            }
        )
    return _json_safe(payload)


def build_quality_report(artifacts: RunArtifacts) -> dict[str, object]:
    trace_records = _load_trace_records(artifacts.trace_path)
    run_record = next(
        (
            dict(record)
            for record in trace_records
            if str(record.get("record_type", "")) == "run"
        ),
        {},
    )
    log_artifacts = _parse_log_artifacts(artifacts.log_path)
    before_summary = _build_phase_summary(trace_records, PAIR_PHASE_BEFORE)
    for phase_name, matched_statuses, residual_priority in PAIR_PHASE_BEFORE_FALLBACKS:
        candidate_summary = _build_phase_summary(
            trace_records,
            phase_name,
            matched_statuses=matched_statuses,
            residual_priority=residual_priority,
        )
        if candidate_summary["pair_record_count"] > 0:
            before_summary = candidate_summary
            break
    after_summary = _build_phase_summary(
        trace_records,
        PAIR_PHASE_AFTER,
        matched_statuses=("matched",),
        residual_priority=("detector_residual_px",),
    )
    background_context = _background_context(
        artifacts.state_path,
        artifacts.out_state_path if artifacts.out_state_path.is_file() else None,
    )
    pair_alignment_rows = _pair_alignment_rows(
        trace_records,
        before_phase=str(before_summary["phase"]),
        before_residual_priority=tuple(
            str(item) for item in before_summary.get("residual_priority", ())
        ),
        after_phase=str(after_summary["phase"]),
        after_residual_priority=tuple(
            str(item) for item in after_summary.get("residual_priority", ())
        ),
        preflight_slot_results=list(
            background_context.get("preflight_slot_results", []) or []
        ),
    )
    worst_start_pair = next(
        (
            dict(entry)
            for entry in sorted(
                pair_alignment_rows,
                key=lambda entry: (
                    -float(_float_or_none(entry.get("before_distance_px")) or -1.0),
                    _pair_sort_key(entry.get("pair_id")),
                ),
            )
            if _float_or_none(entry.get("before_distance_px")) is not None
        ),
        None,
    )
    identity_retention_after_fit = _build_identity_retention_summary(
        trace_records,
        str(after_summary["phase"]),
        matched_statuses=tuple(str(item) for item in after_summary["matched_statuses"]),
    )
    final_summary = dict(log_artifacts.get("final_summary", {}))
    optimizer_summary = dict(log_artifacts.get("optimizer_summary", {}))
    matched_preview = _load_matched_peaks_preview(artifacts.matched_peaks_path)
    cli_stdout_text = (
        artifacts.cli_stdout_path.read_text(encoding="utf-8")
        if artifacts.cli_stdout_path.is_file()
        else ""
    )
    cli_stderr_text = (
        artifacts.cli_stderr_path.read_text(encoding="utf-8")
        if artifacts.cli_stderr_path.is_file()
        else ""
    )
    accepted = bool(run_record.get("accepted", False)) if run_record else False
    rejection_reason = run_record.get("rejection_reason")
    if rejection_reason in (None, "") and artifacts.timeout_reason:
        rejection_reason = artifacts.timeout_reason
    if rejection_reason in (None, ""):
        rejection_reason = _last_nonempty_line(cli_stderr_text)
    if rejection_reason in (None, "") and "Geometry fit unavailable:" in cli_stdout_text:
        rejection_reason = _last_nonempty_line(cli_stdout_text)
    rejection_text = str(rejection_reason) if rejection_reason not in (None, "") else None
    final_metric_name = str(
        run_record.get(
            "final_metric_name",
            optimizer_summary.get("final_metric_name", final_summary.get("metric_name", "")),
        )
        or ""
    )
    direct_solver_evals = [
        dict(entry)
        for entry in list(log_artifacts.get("direct_solver_evals", []) or [])
        if isinstance(entry, Mapping)
    ]
    direct_solver_summary = (
        dict(log_artifacts.get("direct_solver_summary", {}))
        if isinstance(log_artifacts.get("direct_solver_summary"), Mapping)
        else {}
    )
    direct_initial_rms_px = (
        _float_or_none(direct_solver_evals[0].get("weighted_rms_px"))
        if direct_solver_evals
        else None
    )
    direct_final_rms_px = _float_or_none(
        direct_solver_summary.get("best_weighted_rms_px")
    )
    if direct_final_rms_px is None and direct_solver_evals:
        direct_final_rms_px = _float_or_none(direct_solver_evals[-1].get("weighted_rms_px"))
    full_beam_summary = (
        dict(run_record.get("full_beam_polish_summary", {}))
        if isinstance(run_record.get("full_beam_polish_summary"), Mapping)
        else {}
    )
    run_stage_timing_s = (
        {
            str(key): _float_or_none(value)
            for key, value in dict(run_record.get("stage_timing_s", {})).items()
            if _float_or_none(value) is not None
        }
        if isinstance(run_record.get("stage_timing_s"), Mapping)
        else {}
    )
    run_summary = {
        "dynamic_point_geometry_fit": bool(run_record.get("dynamic_point_geometry_fit", False)),
        "fit_quality_passed": bool(run_record.get("fit_quality_passed", False)),
        "selection_status": str(run_record.get("selection_status", "") or ""),
        "selected_candidate_name": run_record.get("selected_candidate_name"),
        "selected_candidate_source": run_record.get("selected_candidate_source"),
        "best_valid_raw_detector_candidate_name": run_record.get(
            "best_valid_raw_detector_candidate_name"
        ),
        "best_valid_raw_detector_candidate_source": run_record.get(
            "best_valid_raw_detector_candidate_source"
        ),
        "constraint_count": int(run_record.get("constraint_count", 0) or 0),
        "active_fit_variable_count": int(run_record.get("active_fit_variable_count", 0) or 0),
        "active_fit_variables": [
            str(item)
            for item in list(run_record.get("active_fit_variables", []) or [])
            if str(item).strip()
        ],
        "candidate_ledger": [
            dict(entry)
            for entry in list(run_record.get("candidate_ledger", []) or [])
            if isinstance(entry, Mapping)
        ],
        "full_beam_polish_enabled": bool(
            run_record.get("full_beam_polish_enabled", full_beam_summary.get("enabled", False))
        ),
        "full_beam_polish_accepted": bool(
            run_record.get("full_beam_polish_accepted", full_beam_summary.get("accepted", False))
        ),
        "full_beam_start_vector_source": str(
            run_record.get(
                "full_beam_start_vector_source",
                full_beam_summary.get("start_vector_source", ""),
            )
            or ""
        ),
        "seed_correspondence_count": int(
            run_record.get(
                "seed_correspondence_count",
                full_beam_summary.get("seed_correspondence_count", 0),
            )
            or 0
        ),
        "nfev": int(run_record.get("nfev", optimizer_summary.get("nfev", 0)) or 0),
        "matched_fixed_pair_count_before": int(
            full_beam_summary.get("matched_pair_count_before", before_summary["matched_count"]) or 0
        ),
        "matched_fixed_pair_count_after": int(
            full_beam_summary.get("matched_pair_count_after", after_summary["matched_count"]) or 0
        ),
        "missing_fixed_pair_count_after": int(
            full_beam_summary.get(
                "missing_pair_count_after",
                after_summary["unresolved_count"],
            )
            or 0
        ),
        "outside_radius_count_before": int(
            full_beam_summary.get(
                "start_outside_radius_count",
                before_summary["outside_radius_count"],
            )
            or 0
        ),
        "outside_radius_count_after": int(
            full_beam_summary.get(
                "outside_radius_count_after",
                after_summary["outside_radius_count"],
            )
            or 0
        ),
        "branch_mismatch_count": int(
            full_beam_summary.get(
                "branch_mismatch_count_after",
                identity_retention_after_fit["branch_mismatch_count"],
            )
            or 0
        ),
        "stage_timing_s": run_stage_timing_s,
    }
    if direct_initial_rms_px is not None or direct_final_rms_px is not None:
        run_summary.update(
            {
                "direct_initial_rms_px": direct_initial_rms_px,
                "direct_final_rms_px": direct_final_rms_px,
                "direct_residual_reduced": bool(
                    direct_initial_rms_px is not None
                    and direct_final_rms_px is not None
                    and direct_final_rms_px < direct_initial_rms_px
                ),
                "direct_solver_evaluations": int(
                    direct_solver_summary.get("evaluations", len(direct_solver_evals)) or 0
                ),
            }
        )
    decision_row = {
        "accepted": accepted,
        "rejection_reason": rejection_text,
        "final_metric_name": final_metric_name or None,
        "bound_hits": list(log_artifacts.get("bound_hits", [])),
        "boundary_warning": log_artifacts.get("boundary_warning"),
        "matched_count_before": before_summary["matched_count"],
        "matched_count_after": after_summary["matched_count"],
        "rms_before_px": before_summary["rms_px"],
        "rms_after_px": after_summary["rms_px"],
        "median_before_px": before_summary["median_px"],
        "median_after_px": after_summary["median_px"],
        "max_residual_before_px": before_summary["max_residual_px"],
        "max_residual_after_px": after_summary["max_residual_px"],
        "worst_5_pair_ids_before": before_summary["worst_pair_ids"],
        "worst_5_pair_ids_after": after_summary["worst_pair_ids"],
    }
    report = {
        "state_name": artifacts.state_path.stem,
        "artifacts": {
            "run_dir": str(artifacts.run_dir),
            "source_state_path": str(artifacts.state_path),
            "fitted_state_path": str(artifacts.out_state_path) if artifacts.out_state_path.is_file() else None,
            "log_path": str(artifacts.log_path) if artifacts.log_path is not None else None,
            "trace_path": str(artifacts.trace_path) if artifacts.trace_path is not None else None,
            "matched_peaks_path": (
                str(artifacts.matched_peaks_path)
                if artifacts.matched_peaks_path is not None
                else None
            ),
            "cli_stdout_path": str(artifacts.cli_stdout_path),
            "cli_stderr_path": str(artifacts.cli_stderr_path),
            "cli_returncode": int(artifacts.cli_returncode),
            "timed_out": bool(artifacts.timed_out),
            "timeout_reason": artifacts.timeout_reason,
            "timeout_last_phase": artifacts.timeout_last_phase,
            "timeout_last_pair_id": artifacts.timeout_last_pair_id,
        },
        "background_context": background_context,
        "decision_row": decision_row,
        "run_summary": run_summary,
        "start_parameters": [
            {
                "name": entry["name"],
                "group": entry["group"],
                "value": entry["start"],
                "lower_bound": entry["lower_bound"],
                "upper_bound": entry["upper_bound"],
            }
            for entry in log_artifacts.get("parameter_entries", [])
        ],
        "fitted_parameters": [
            {
                "name": entry["name"],
                "group": entry["group"],
                "value": entry["final"],
                "delta": entry["delta"],
                "lower_bound": entry["lower_bound"],
                "upper_bound": entry["upper_bound"],
            }
            for entry in log_artifacts.get("parameter_entries", [])
        ],
        "final_metric": {
            "name": final_metric_name or None,
            "weighted_residual_rms_px": run_record.get(
                "weighted_residual_rms_px",
                optimizer_summary.get("weighted_residual_rms_px"),
            ),
            "detector_rms_px": run_record.get(
                "detector_rms_px",
                optimizer_summary.get("display_rms_px"),
            ),
            "cost": final_summary.get("cost", optimizer_summary.get("cost")),
            "robust_cost": final_summary.get(
                "robust_cost",
                optimizer_summary.get("robust_cost"),
            ),
        },
        "before_fit": before_summary,
        "after_fit": after_summary,
        "pair_alignment_rows": pair_alignment_rows,
        "worst_start_pair": worst_start_pair,
        "identity_retention_after_fit": identity_retention_after_fit,
        "overlay_evidence": {
            "matched_peaks": matched_preview,
            "trace_phases": {
                "before": before_summary["phase"],
                "after": after_summary["phase"],
            },
            "artifacts_are_record_based": True,
            "note": (
                "Exported match records and structured trace are the primary overlay evidence. "
                "Screenshots are optional and were not generated by default."
            ),
        },
        "fit_log_summary": {
            "optimizer_summary": optimizer_summary,
            "full_beam_polish_summary_line": dict(log_artifacts.get("stage_summaries", {})).get(
                "full_beam_polish"
            ),
        },
    }
    point_match_summary = (
        dict(run_record.get("point_match_summary", {}))
        if isinstance(run_record.get("point_match_summary"), Mapping)
        else {}
    )
    manual_caked_rows = [
        dict(entry)
        for entry in pair_alignment_rows
        if str(entry.get("fit_space_projector_kind", "") or "")
        in CAKED_FIT_SPACE_PROJECTOR_KINDS
    ]
    point_only_dynamic_row_count = int(
        sum(
            1
            for entry in manual_caked_rows
            if str(entry.get("fit_space_projector_kind", "") or "")
            == "point_only_dynamic_source_row"
        )
    )
    exact_projector_available = bool(
        point_match_summary.get("exact_fit_space_projector_available", False)
        or point_only_dynamic_row_count > 0
    )
    report.update(
        {
            "processed_manual_entry_count": _int_or_none(
                background_context.get("preflight_processed_manual_entry_count")
            ),
            "bound_manual_entry_count": _int_or_none(
                background_context.get("preflight_valid_count")
            ),
            "missing_manual_entry_count": _int_or_none(
                background_context.get("preflight_missing_manual_entry_count")
            ),
            "branch_mismatch_count": _int_or_none(
                background_context.get("preflight_branch_mismatch_count")
            ),
            "background_distance_gate_ok": bool(
                background_context.get("preflight_background_distance_gate_ok", False)
            ),
            "dataset_resolved_source_pair_count": _int_or_none(
                background_context.get("preflight_dataset_resolved_source_pair_count")
            ),
            "manual_caked_residual_row_count": int(len(manual_caked_rows)),
            "dataset_fit_space_projector_row_count": int(
                sum(
                    1
                    for entry in manual_caked_rows
                    if str(entry.get("fit_space_projector_kind", "") or "")
                    == "point_only_dynamic_source_row"
                    or "dataset_fit_space_projector"
                    in {
                        str(entry.get("measured_fit_space_source", "") or ""),
                        str(entry.get("simulated_fit_space_source", "") or ""),
                    }
                )
            ),
            "invalid_dataset_fit_space_projector_row_count": int(
                sum(
                    1
                    for entry in manual_caked_rows
                    if "invalid_dataset_fit_space_projector"
                    in {
                        str(entry.get("measured_fit_space_source", "") or ""),
                        str(entry.get("simulated_fit_space_source", "") or ""),
                    }
                )
            ),
            "analytic_detector_fit_space_row_count": int(
                sum(
                    1
                    for entry in manual_caked_rows
                    if "analytic_detector_fit_space"
                    in {
                        str(entry.get("measured_fit_space_source", "") or ""),
                        str(entry.get("simulated_fit_space_source", "") or ""),
                    }
                )
            ),
            "cached_fit_space_anchor_row_count": int(
                sum(
                    1
                    for entry in manual_caked_rows
                    if "cached_fit_space_anchor"
                    in {
                        str(entry.get("measured_fit_space_source", "") or ""),
                        str(entry.get("simulated_fit_space_source", "") or ""),
                    }
                )
            ),
            "point_only_dynamic_source_row_count": point_only_dynamic_row_count,
            "exact_fit_space_projector_available": exact_projector_available,
            "exact_fit_space_projection_reason": (
                str(point_match_summary.get("exact_fit_space_projection_reason", "") or "")
                or (
                    None
                    if exact_projector_available
                    else "exact_projector_provenance_missing"
                )
            ),
            "fit_space_local_params_signature": point_match_summary.get(
                "fit_space_local_params_signature"
            ),
        }
    )
    report["fit_space_provenance_validation"] = validate_manual_caked_fit_space_provenance(
        report
    )
    report["saved_state_gate"] = _saved_state_gate_summary(report)
    return _json_safe(report)  # type: ignore[return-value]


def _compact_summary_from_report(
    report: Mapping[str, object],
    *,
    elapsed_s: float,
    baseline_report_writing_s: float,
) -> dict[str, object]:
    decision = dict(report.get("decision_row", {}))
    run_summary = (
        dict(report.get("run_summary", {}))
        if isinstance(report.get("run_summary"), Mapping)
        else {}
    )
    before_fit = dict(report.get("before_fit", {})) if isinstance(report.get("before_fit"), Mapping) else {}
    after_fit = dict(report.get("after_fit", {})) if isinstance(report.get("after_fit"), Mapping) else {}
    identity = (
        dict(report.get("identity_retention_after_fit", {}))
        if isinstance(report.get("identity_retention_after_fit"), Mapping)
        else {}
    )
    saved_state_gate = (
        dict(report.get("saved_state_gate", {}))
        if isinstance(report.get("saved_state_gate"), Mapping)
        else {}
    )
    stage_timing_s = {
        str(key): float(value)
        for key, value in dict(run_summary.get("stage_timing_s", {})).items()
        if _float_or_none(value) is not None
        for value in [float(_float_or_none(value))]
    }
    stage_timing_s["baseline_report_writing"] = float(max(0.0, baseline_report_writing_s))
    return _json_safe(
        {
            "state": str(report.get("state_name", "") or ""),
            "accepted": bool(decision.get("accepted", False)),
            "rejection_reason": decision.get("rejection_reason"),
            "before_rms_px": _float_or_none(decision.get("rms_before_px", before_fit.get("rms_px"))),
            "after_rms_px": _float_or_none(decision.get("rms_after_px", after_fit.get("rms_px"))),
            "initial_rms_px": _float_or_none(run_summary.get("direct_initial_rms_px")),
            "final_rms_px": _float_or_none(run_summary.get("direct_final_rms_px")),
            "direct_residual_reduced": bool(run_summary.get("direct_residual_reduced", False)),
            "before_median_px": _float_or_none(
                decision.get("median_before_px", before_fit.get("median_px"))
            ),
            "after_median_px": _float_or_none(
                decision.get("median_after_px", after_fit.get("median_px"))
            ),
            "before_max_px": _float_or_none(
                decision.get("max_residual_before_px", before_fit.get("max_residual_px"))
            ),
            "after_max_px": _float_or_none(
                decision.get("max_residual_after_px", after_fit.get("max_residual_px"))
            ),
            "preflight_valid_count": int(saved_state_gate.get("preflight_valid_count", 0) or 0),
            "matched_fixed_pair_count_before": int(
                run_summary.get(
                    "matched_fixed_pair_count_before",
                    decision.get("matched_count_before", before_fit.get("matched_count", 0)),
                )
                or 0
            ),
            "missing_fixed_pair_count_before": int(
                saved_state_gate.get("missing_fixed_pair_count_before", 0) or 0
            ),
            "matched_fixed_pair_count_after": int(
                run_summary.get(
                    "matched_fixed_pair_count_after",
                    decision.get("matched_count_after", after_fit.get("matched_count", 0)),
                )
                or 0
            ),
            "missing_fixed_pair_count_after": int(
                run_summary.get(
                    "missing_fixed_pair_count_after",
                    after_fit.get("unresolved_count", 0),
                )
                or 0
            ),
            "outside_radius_count_before": int(
                run_summary.get(
                    "outside_radius_count_before",
                    before_fit.get("outside_radius_count", 0),
                )
                or 0
            ),
            "outside_radius_count_after": int(
                run_summary.get(
                    "outside_radius_count_after",
                    after_fit.get("outside_radius_count", 0),
                )
                or 0
            ),
            "branch_mismatch_count": int(
                run_summary.get(
                    "branch_mismatch_count",
                    identity.get("branch_mismatch_count", 0),
                )
                or 0
            ),
            "fit_quality_passed": bool(run_summary.get("fit_quality_passed", False)),
            "fixed_manual_caked_checks_pass": bool(
                saved_state_gate.get("fixed_manual_caked_checks_pass", False)
            ),
            "selection_status": str(run_summary.get("selection_status", "") or ""),
            "selected_candidate_name": run_summary.get("selected_candidate_name"),
            "selected_candidate_source": run_summary.get("selected_candidate_source"),
            "best_valid_raw_detector_candidate_name": run_summary.get(
                "best_valid_raw_detector_candidate_name"
            ),
            "best_valid_raw_detector_candidate_source": run_summary.get(
                "best_valid_raw_detector_candidate_source"
            ),
            "dynamic_point_geometry_fit": bool(
                run_summary.get("dynamic_point_geometry_fit", False)
            ),
            "full_beam_polish_enabled": bool(
                run_summary.get("full_beam_polish_enabled", False)
            ),
            "full_beam_polish_accepted": bool(
                run_summary.get("full_beam_polish_accepted", False)
            ),
            "full_beam_start_vector_source": str(
                run_summary.get("full_beam_start_vector_source", "") or ""
            ),
            "seed_correspondence_count": int(
                run_summary.get("seed_correspondence_count", 0) or 0
            ),
            "constraint_count": int(run_summary.get("constraint_count", 0) or 0),
            "active_fit_variable_count": int(
                run_summary.get("active_fit_variable_count", 0) or 0
            ),
            "active_fit_variables": [
                str(item)
                for item in list(run_summary.get("active_fit_variables", []) or [])
                if str(item).strip()
            ],
            "nfev": int(run_summary.get("nfev", 0) or 0),
            "saved_state_gate_ok": bool(saved_state_gate.get("ok", False)),
            "saved_state_gate_failures": [
                str(item)
                for item in list(saved_state_gate.get("failures", []))
                if str(item).strip()
            ],
            "manual_caked_residual_row_count": int(
                report.get("manual_caked_residual_row_count", 0) or 0
            ),
            "dataset_fit_space_projector_row_count": int(
                report.get("dataset_fit_space_projector_row_count", 0) or 0
            ),
            "invalid_dataset_fit_space_projector_row_count": int(
                report.get("invalid_dataset_fit_space_projector_row_count", 0) or 0
            ),
            "analytic_detector_fit_space_row_count": int(
                report.get("analytic_detector_fit_space_row_count", 0) or 0
            ),
            "cached_fit_space_anchor_row_count": int(
                report.get("cached_fit_space_anchor_row_count", 0) or 0
            ),
            "exact_fit_space_projector_available": bool(
                report.get("exact_fit_space_projector_available", False)
            ),
            "exact_fit_space_projection_reason": report.get(
                "exact_fit_space_projection_reason"
            ),
            "fit_space_provenance_validation": [
                str(item)
                for item in list(report.get("fit_space_provenance_validation", []) or [])
                if str(item).strip()
            ],
            "elapsed_s": float(max(0.0, elapsed_s)),
            "stage_timing_s": stage_timing_s,
        }
    )


def _markdown_pair_rows(entries: Sequence[Mapping[str, object]]) -> list[str]:
    lines = ["| Pair | Dataset | HKL | Residual px | Status |", "| --- | --- | --- | ---: | --- |"]
    for entry in entries:
        hkl = entry.get("hkl")
        hkl_text = str(tuple(hkl)) if isinstance(hkl, list) else str(hkl)
        residual = entry.get("detector_residual_px")
        residual_text = f"{float(residual):.3f}" if isinstance(residual, (float, int)) else "n/a"
        lines.append(
            "| {pair} | {dataset} | {hkl} | {residual} | {status} |".format(
                pair=str(entry.get("pair_id", "")),
                dataset=int(entry.get("dataset_index", 0) or 0),
                hkl=hkl_text,
                residual=residual_text,
                status=str(entry.get("match_status", "")),
            )
        )
    return lines


def render_quality_report_markdown(report: Mapping[str, object]) -> str:
    decision = dict(report.get("decision_row", {}))
    final_metric = dict(report.get("final_metric", {}))
    artifacts = dict(report.get("artifacts", {}))
    backgrounds = dict(report.get("background_context", {}))
    run_summary = dict(report.get("run_summary", {}))
    before_fit = dict(report.get("before_fit", {}))
    after_fit = dict(report.get("after_fit", {}))
    worst_start_pair = (
        dict(report.get("worst_start_pair", {}))
        if isinstance(report.get("worst_start_pair"), Mapping)
        else {}
    )
    identity_retention = dict(report.get("identity_retention_after_fit", {}))
    overlay = dict(report.get("overlay_evidence", {}))
    start_parameters = list(report.get("start_parameters", []))
    fitted_parameters = list(report.get("fitted_parameters", []))
    pair_alignment_rows = [
        dict(entry)
        for entry in list(report.get("pair_alignment_rows", []) or [])
        if isinstance(entry, Mapping)
    ]
    candidate_ledger = [
        dict(entry)
        for entry in list(run_summary.get("candidate_ledger", []) or [])
        if isinstance(entry, Mapping)
    ]
    lines = [
        f"# {report.get('state_name', 'geometry_fit')} quality report",
        "",
        "## Decision row",
        "",
        "| accepted | rejection_reason | final_metric_name | bound_hits | boundary_warning | matched_before | matched_after | rms_before_px | rms_after_px | max_before_px | max_after_px | worst_5_before | worst_5_after |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        "| {accepted} | {rejection} | {metric} | {bound_hits} | {warning} | {mb} | {ma} | {rb} | {ra} | {xb} | {xa} | {wb} | {wa} |".format(
            accepted=str(decision.get("accepted")),
            rejection=str(decision.get("rejection_reason") or ""),
            metric=str(decision.get("final_metric_name") or ""),
            bound_hits=", ".join(str(item) for item in decision.get("bound_hits", [])) or "",
            warning=str(decision.get("boundary_warning") or ""),
            mb=decision.get("matched_count_before", "n/a"),
            ma=decision.get("matched_count_after", "n/a"),
            rb="n/a"
            if decision.get("rms_before_px") is None
            else f"{float(decision['rms_before_px']):.3f}",
            ra="n/a"
            if decision.get("rms_after_px") is None
            else f"{float(decision['rms_after_px']):.3f}",
            xb="n/a"
            if decision.get("max_residual_before_px") is None
            else f"{float(decision['max_residual_before_px']):.3f}",
            xa="n/a"
            if decision.get("max_residual_after_px") is None
            else f"{float(decision['max_residual_after_px']):.3f}",
            wb=", ".join(str(item) for item in decision.get("worst_5_pair_ids_before", [])),
            wa=", ".join(str(item) for item in decision.get("worst_5_pair_ids_after", [])),
        ),
        "",
        "## Final metric",
        "",
        f"- name: `{final_metric.get('name')}`",
        f"- weighted residual RMS px: `{final_metric.get('weighted_residual_rms_px')}`",
        f"- detector RMS px: `{final_metric.get('detector_rms_px')}`",
        f"- cost: `{final_metric.get('cost')}`",
        f"- robust cost: `{final_metric.get('robust_cost')}`",
        "",
        "## Fit-quality gate",
        "",
        f"- fit quality passed: `{run_summary.get('fit_quality_passed')}`",
        f"- selection status: `{run_summary.get('selection_status')}`",
        f"- selected candidate: `{run_summary.get('selected_candidate_name')}` from `{run_summary.get('selected_candidate_source')}`",
        f"- best valid raw candidate: `{run_summary.get('best_valid_raw_detector_candidate_name')}` from `{run_summary.get('best_valid_raw_detector_candidate_source')}`",
        f"- constraint count: `{run_summary.get('constraint_count')}`",
        f"- active fit variable count: `{run_summary.get('active_fit_variable_count')}`",
        f"- active fit variables: `{run_summary.get('active_fit_variables')}`",
        "",
        "## Parameters",
        "",
        "| parameter | start | final | delta | bounds |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    final_by_name = {
        str(entry.get("name", "")): dict(entry)
        for entry in fitted_parameters
        if isinstance(entry, Mapping)
    }
    for raw_entry in start_parameters:
        if not isinstance(raw_entry, Mapping):
            continue
        name = str(raw_entry.get("name", ""))
        final_entry = final_by_name.get(name, {})
        start_value = raw_entry.get("value")
        final_value = final_entry.get("value")
        delta_value = final_entry.get("delta")
        lower = raw_entry.get("lower_bound")
        upper = raw_entry.get("upper_bound")
        lines.append(
            "| {name} | {start} | {final} | {delta} | [{lower}, {upper}] |".format(
                name=name,
                start="n/a" if start_value is None else f"{float(start_value):.6f}",
                final="n/a" if final_value is None else f"{float(final_value):.6f}",
                delta="n/a" if delta_value is None else f"{float(delta_value):.6f}",
                lower="n/a" if lower is None else f"{float(lower):.6f}",
                upper="n/a" if upper is None else f"{float(upper):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Pair alignment",
            "",
            "| Pair | HKL | Branch | Peak | Before dx | Before dy | Before dist | After dx | After dy | After dist | Improved |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *[
                "| {pair_id} | {hkl} | {branch} | {peak} | {before_dx} | {before_dy} | {before_dist} | {after_dx} | {after_dy} | {after_dist} | {improved} |".format(
                    pair_id=str(entry.get("pair_id", "")),
                    hkl=str(tuple(entry.get("hkl"))) if isinstance(entry.get("hkl"), list) else str(entry.get("hkl")),
                    branch=("n/a" if entry.get("source_branch_index") is None else int(entry.get("source_branch_index"))),
                    peak=("n/a" if entry.get("source_peak_index") is None else int(entry.get("source_peak_index"))),
                    before_dx=("n/a" if _float_or_none(entry.get("before_dx_px")) is None else f"{float(entry['before_dx_px']):.3f}"),
                    before_dy=("n/a" if _float_or_none(entry.get("before_dy_px")) is None else f"{float(entry['before_dy_px']):.3f}"),
                    before_dist=("n/a" if _float_or_none(entry.get("before_distance_px")) is None else f"{float(entry['before_distance_px']):.3f}"),
                    after_dx=("n/a" if _float_or_none(entry.get("after_dx_px")) is None else f"{float(entry['after_dx_px']):.3f}"),
                    after_dy=("n/a" if _float_or_none(entry.get("after_dy_px")) is None else f"{float(entry['after_dy_px']):.3f}"),
                    after_dist=("n/a" if _float_or_none(entry.get("after_distance_px")) is None else f"{float(entry['after_distance_px']):.3f}"),
                    improved=str(bool(entry.get("improved", False))).lower(),
                )
                for entry in pair_alignment_rows
            ],
            "",
            "## Worst start pair",
            "",
            f"- pair id: `{worst_start_pair.get('pair_id')}`",
            f"- before distance px: `{worst_start_pair.get('before_distance_px')}`",
            f"- after distance px: `{worst_start_pair.get('after_distance_px')}`",
            f"- improved: `{worst_start_pair.get('improved')}`",
            "",
            "## Residual outliers before fit",
            "",
            *_markdown_pair_rows(
                [
                    dict(entry)
                    for entry in before_fit.get("top_outliers", [])
                    if isinstance(entry, Mapping)
                ]
            ),
            "",
            "## Residual outliers after fit",
            "",
            *_markdown_pair_rows(
                [
                    dict(entry)
                    for entry in after_fit.get("top_outliers", [])
                    if isinstance(entry, Mapping)
                ]
            ),
            "",
            "## Branch / peak retention after fit",
            "",
            f"- phase: `{identity_retention.get('phase')}`",
            f"- matched pairs: `{identity_retention.get('matched_pair_count')}`",
            f"- branch retained: `{identity_retention.get('branch_retained_count')}` / `{identity_retention.get('source_branch_count')}`",
            f"- branch mismatches: `{identity_retention.get('branch_mismatch_count')}`",
            f"- branch unresolved: `{identity_retention.get('branch_unresolved_count')}`",
            f"- peak retained: `{identity_retention.get('peak_retained_count')}` / `{identity_retention.get('source_peak_count')}`",
            f"- peak mismatches: `{identity_retention.get('peak_mismatch_count')}`",
            f"- peak unresolved: `{identity_retention.get('peak_unresolved_count')}`",
            f"- unmatched resolution reasons: `{dict(identity_retention.get('unmatched_resolution_reason_counts', {}))}`",
            "",
            "## Candidate ledger",
            "",
            "| Candidate | Source | Matched | Missing | Branch mismatch | RMS px | Median px | Max px | Outside radius | Weighted objective | Valid | Selected | Rejection reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            *[
                "| {candidate_name} | {x_vector_source} | {matched_pair_count} | {missing_pair_count} | {branch_mismatch_count} | {rms_px} | {median_px} | {max_px} | {outside_radius_count} | {weighted_objective} | {valid} | {selected} | {rejection_reason} |".format(
                    candidate_name=str(entry.get("candidate_name", "")),
                    x_vector_source=str(entry.get("x_vector_source", "")),
                    matched_pair_count=int(entry.get("matched_pair_count", 0) or 0),
                    missing_pair_count=int(entry.get("missing_pair_count", 0) or 0),
                    branch_mismatch_count=int(entry.get("branch_mismatch_count", 0) or 0),
                    rms_px=("n/a" if _float_or_none(entry.get("rms_px")) is None else f"{float(entry['rms_px']):.3f}"),
                    median_px=("n/a" if _float_or_none(entry.get("median_px")) is None else f"{float(entry['median_px']):.3f}"),
                    max_px=("n/a" if _float_or_none(entry.get("max_px")) is None else f"{float(entry['max_px']):.3f}"),
                    outside_radius_count=int(entry.get("outside_radius_count", 0) or 0),
                    weighted_objective=("n/a" if _float_or_none(entry.get("weighted_objective")) is None else f"{float(entry['weighted_objective']):.6f}"),
                    valid=str(bool(entry.get("valid_raw_detector_candidate", False))).lower(),
                    selected=str(bool(entry.get("selected", False))).lower(),
                    rejection_reason=str(entry.get("rejection_reason") or ""),
                )
                for entry in candidate_ledger
            ],
            "",
            "## Overlay evidence",
            "",
            f"- current background index: `{backgrounds.get('current_background_index')}`",
            f"- current background file: `{backgrounds.get('current_background_file')}`",
            f"- matched peaks path: `{artifacts.get('matched_peaks_path')}`",
            f"- matched peak record count: `{dict(overlay.get('matched_peaks', {})).get('record_count')}`",
            f"- fitted state path: `{artifacts.get('fitted_state_path')}`",
            f"- trace path: `{artifacts.get('trace_path')}`",
            f"- log path: `{artifacts.get('log_path')}`",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _reuse_existing_run(
    state_path: Path,
    run_dir: Path,
) -> RunArtifacts | None:
    out_state_path = run_dir / f"{state_path.stem}_fit.json"
    cli_stdout_path = run_dir / "cli_stdout.txt"
    cli_stderr_path = run_dir / "cli_stderr.txt"
    if not (
        out_state_path.is_file()
        or cli_stdout_path.is_file()
        or cli_stderr_path.is_file()
    ):
        return None
    log_path, trace_path, matched_peaks_path = _resolve_artifact_paths(
        state_path=state_path,
        run_dir=run_dir,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        # Reused runs must not backfill from the global live cache because those
        # artifacts can belong to a different invocation entirely.
        allow_default_live_cache=False,
    )
    return RunArtifacts(
        state_path=state_path,
        run_dir=run_dir,
        out_state_path=out_state_path,
        log_path=log_path,
        trace_path=trace_path,
        matched_peaks_path=matched_peaks_path,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        cli_returncode=_infer_existing_cli_returncode(
            cli_stdout_path=cli_stdout_path,
            cli_stderr_path=cli_stderr_path,
            out_state_path=out_state_path,
        ),
    )


def _run_fit_geometry(
    state_path: Path,
    run_dir: Path,
    *,
    timeout_seconds: float | None = None,
) -> RunArtifacts:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_state_path = run_dir / f"{state_path.stem}_fit.json"
    cli_stdout_path = run_dir / "cli_stdout.txt"
    cli_stderr_path = run_dir / "cli_stderr.txt"
    cmd = [
        sys.executable,
        "-m",
        "ra_sim.cli",
        "fit-geometry",
        str(state_path),
        "--seed-policy",
        "direct",
        "--out-state",
        str(out_state_path),
    ]
    started_at = time.monotonic()
    started_at_wall_time = time.time()
    next_heartbeat_at = started_at + HEARTBEAT_INTERVAL_SECONDS
    last_phase = "launching"
    with (
        cli_stdout_path.open("w", encoding="utf-8") as stdout_handle,
        cli_stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        while True:
            stdout_handle.flush()
            stderr_handle.flush()
            live_log_path, live_trace_path = _resolve_live_artifact_paths(
                state_path=state_path,
                run_dir=run_dir,
                cli_stdout_path=cli_stdout_path,
                cli_stderr_path=cli_stderr_path,
                started_at_wall_time=started_at_wall_time,
            )
            last_phase = _active_phase_label(
                log_path=live_log_path,
                trace_path=live_trace_path,
            )
            returncode = process.poll()
            if returncode is not None:
                break
            now = time.monotonic()
            elapsed_seconds = now - started_at
            if now >= next_heartbeat_at:
                print(
                    "HEARTBEAT {name}: phase={phase} pid={pid} out_dir={out_dir}".format(
                        name=state_path.stem,
                        phase=last_phase,
                        pid=process.pid,
                        out_dir=run_dir,
                    ),
                    flush=True,
                )
                next_heartbeat_at = now + HEARTBEAT_INTERVAL_SECONDS
            if timeout_seconds is not None and elapsed_seconds >= timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
                log_path, trace_path, matched_peaks_path = _resolve_artifact_paths(
                    state_path=state_path,
                    run_dir=run_dir,
                    cli_stdout_path=cli_stdout_path,
                    cli_stderr_path=cli_stderr_path,
                    allow_default_live_cache=True,
                    started_at_wall_time=started_at_wall_time,
                )
                trace_last_phase, trace_last_pair_id = _last_trace_phase_and_pair_id(
                    trace_path,
                )
                timeout_last_phase = trace_last_phase or last_phase
                timeout_reason = (
                    "{name} exceeded {timeout:.1f}s last_phase={phase}{pair_suffix}".format(
                        name=state_path.stem,
                        timeout=timeout_seconds,
                        phase=timeout_last_phase,
                        pair_suffix=(
                            f" last_pair_id={trace_last_pair_id}"
                            if trace_last_pair_id is not None
                            else ""
                        ),
                    )
                )
                return RunArtifacts(
                    state_path=state_path,
                    run_dir=run_dir,
                    out_state_path=out_state_path,
                    log_path=log_path,
                    trace_path=trace_path,
                    matched_peaks_path=matched_peaks_path,
                    cli_stdout_path=cli_stdout_path,
                    cli_stderr_path=cli_stderr_path,
                    cli_returncode=int(process.returncode if process.returncode is not None else -1),
                    timed_out=True,
                    timeout_reason=timeout_reason,
                    timeout_last_phase=timeout_last_phase,
                    timeout_last_pair_id=trace_last_pair_id,
                )
            time.sleep(1.0)
    log_path, trace_path, matched_peaks_path = _resolve_artifact_paths(
        state_path=state_path,
        run_dir=run_dir,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        allow_default_live_cache=True,
        started_at_wall_time=started_at_wall_time,
    )
    return RunArtifacts(
        state_path=state_path,
        run_dir=run_dir,
        out_state_path=out_state_path,
        log_path=log_path,
        trace_path=trace_path,
        matched_peaks_path=matched_peaks_path,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        cli_returncode=returncode,
    )


def _write_report_files(
    report: Mapping[str, object],
    run_dir: Path,
    stem: str,
    *,
    timed_out: bool = False,
) -> tuple[Path, Path]:
    json_path = run_dir / f"{stem}_quality_report.json"
    md_path = run_dir / f"{stem}_quality_report.md"
    timeout_json_path = run_dir / f"{stem}_timeout_report.json"
    timeout_md_path = run_dir / f"{stem}_timeout_report.md"
    if bool(timed_out):
        timeout_json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        timeout_md_path.write_text(
            render_quality_report_markdown(report),
            encoding="utf-8",
        )
        return timeout_json_path, timeout_md_path
    else:
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        md_path.write_text(
            render_quality_report_markdown(report),
            encoding="utf-8",
        )
        for stale_path in (timeout_json_path, timeout_md_path):
            if stale_path.exists():
                stale_path.unlink()
        return json_path, md_path


def _default_output_root() -> Path:
    return user_data_root() / "fit_quality_baseline" / date.today().isoformat()


def run_baseline(
    state_paths: Sequence[Path],
    output_root: Path,
    *,
    reuse_existing: bool = False,
    state_timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for state_path in state_paths:
        run_dir = output_root / state_path.stem
        print(f"START {state_path.stem}: {state_path}", flush=True)
        started_at = time.monotonic()
        artifacts = (
            _reuse_existing_run(state_path, run_dir)
            if reuse_existing
            else None
        )
        if artifacts is None:
            artifacts = _run_fit_geometry(
                state_path,
                run_dir,
                timeout_seconds=state_timeout_seconds,
            )
        report = build_quality_report(artifacts)
        report_write_started_at = time.monotonic()
        _write_report_files(
            report,
            run_dir,
            state_path.stem,
            timed_out=bool(artifacts.timed_out),
        )
        baseline_report_writing_s = time.monotonic() - report_write_started_at
        elapsed_seconds = time.monotonic() - started_at
        compact_summary = _compact_summary_from_report(
            report,
            elapsed_s=elapsed_seconds,
            baseline_report_writing_s=baseline_report_writing_s,
        )
        report_with_summary = dict(report)
        report_with_summary["compact_summary"] = dict(compact_summary)
        reports.append(report_with_summary)
        print(json.dumps(compact_summary, sort_keys=True), flush=True)
        saved_state_gate = (
            dict(report.get("saved_state_gate", {}))
            if isinstance(report.get("saved_state_gate"), Mapping)
            else {}
        )
        if not bool(saved_state_gate.get("ok", False)):
            failures = [
                str(item)
                for item in list(saved_state_gate.get("failures", []))
                if str(item).strip()
            ]
            raise RuntimeError(
                f"{state_path.stem} failed saved-state gate: {'; '.join(failures) or 'unknown failure'}"
            )
    return reports


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run normal headless geometry fits on frozen saved states and write compact "
            "fit-quality baseline reports."
        )
    )
    parser.add_argument(
        "states",
        nargs="*",
        help=(
            "Optional saved-state JSON paths. Defaults to "
            "Bi2Se3.json and Bi2Te3.json under the RA-SIM user data root."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root directory. Defaults to ~/.local/share/ra_sim/fit_quality_baseline/<today>/",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-state CLI stdout/stderr and fitted state artifacts instead of rerunning the fit.",
    )
    parser.add_argument(
        "--state-timeout-seconds",
        type=float,
        default=None,
        help="Optional per-state timeout in seconds. Stops on the first slow state.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    state_paths = [Path(path).expanduser() for path in args.states] if args.states else list(DEFAULT_STATE_PATHS)
    output_root = Path(args.output_root).expanduser() if args.output_root else _default_output_root()
    try:
        reports = run_baseline(
            state_paths,
            output_root,
            reuse_existing=bool(args.reuse_existing),
            state_timeout_seconds=args.state_timeout_seconds,
        )
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 1
    print(f"Output root: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
