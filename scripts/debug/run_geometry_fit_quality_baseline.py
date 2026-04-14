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

import numpy as np


DEFAULT_STATE_PATHS = (
    Path(r"C:\Users\Kenpo\.local\share\ra_sim\new2_fresh_all.json"),
    Path(r"C:\Users\Kenpo\.local\share\ra_sim\new3_fresh_all.json"),
    Path(r"C:\Users\Kenpo\.local\share\ra_sim\new3.json"),
)
DEFAULT_CACHE_LOG_DIR = Path.home() / ".cache" / "ra_sim" / "logs"
HEARTBEAT_INTERVAL_SECONDS = 20.0
PAIR_PHASE_BEFORE = "seed_correspondence"
PAIR_PHASE_AFTER = "acceptance_residuals"
PAIR_PHASE_BEFORE_FALLBACKS = (
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


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
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


def _latest_path(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


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
    def _is_current_run_artifact(path: Path | None) -> bool:
        return bool(
            path is not None
            and path.is_file()
            and path.stat().st_mtime >= (started_at_wall_time - 1.0)
        )

    stdout_text = cli_stdout_path.read_text(encoding="utf-8") if cli_stdout_path.is_file() else ""
    stderr_text = cli_stderr_path.read_text(encoding="utf-8") if cli_stderr_path.is_file() else ""
    log_path = _extract_reported_artifact_path(stdout_text, "Geometry fit log")
    if log_path is None:
        log_path = _extract_reported_artifact_path(stderr_text, "Geometry fit log")
    if not _is_current_run_artifact(log_path):
        default_log_path = _default_live_log_path(state_path)
        latest_run_log_path = _latest_path(run_dir, "geometry_fit_log_*.txt")
        log_path = (
            default_log_path
            if _is_current_run_artifact(default_log_path)
            else latest_run_log_path
            if _is_current_run_artifact(latest_run_log_path)
            else None
        )

    trace_path = _parse_trace_path_from_log(log_path)
    if not _is_current_run_artifact(trace_path):
        default_trace_path = _default_live_trace_path(state_path)
        latest_run_trace_path = _latest_path(run_dir, "geometry_fit_trace_*.jsonl")
        trace_path = (
            default_trace_path
            if _is_current_run_artifact(default_trace_path)
            else latest_run_trace_path
            if _is_current_run_artifact(latest_run_trace_path)
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
    return {
        "pair_id": str(record.get("pair_id", "") or ""),
        "dataset_index": int(record.get("dataset_index", 0) or 0),
        "background_label": str(record.get("background_label", "") or ""),
        "overlay_match_index": record.get("overlay_match_index"),
        "hkl": _json_safe(record.get("hkl")),
        "match_status": str(record.get("match_status", "") or ""),
        "detector_residual_px": residual,
        "optimizer_residual_px": optimizer_residual,
        "measured_point": _json_safe(record.get("measured_point")),
        "simulated_point": _json_safe(record.get("simulated_point")),
        "canonical_identity": _json_safe(record.get("canonical_identity")),
        "resolution_kind": _json_safe(record.get("resolution_kind")),
        "resolution_reason": _json_safe(record.get("resolution_reason")),
        "source_branch_index": _int_or_none(record.get("source_branch_index")),
        "source_peak_index": _int_or_none(record.get("source_peak_index")),
        "resolved_peak_index": _int_or_none(record.get("resolved_peak_index")),
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
    return {
        "phase": phase,
        "matched_statuses": list(matched_statuses),
        "residual_priority": list(residual_priority),
        "pair_record_count": len(normalized),
        "matched_count": len(matched),
        "unresolved_count": len(unresolved_pair_ids),
        "unresolved_pair_ids": unresolved_pair_ids,
        "rms_px": _rms(finite_residuals),
        "max_residual_px": max(finite_residuals) if finite_residuals else None,
        "worst_pair_ids": worst_pair_ids,
        "top_outliers": worst_records,
    }


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
) -> tuple[Path | None, Path | None, Path | None]:
    stdout_text = cli_stdout_path.read_text(encoding="utf-8") if cli_stdout_path.is_file() else ""
    stderr_text = cli_stderr_path.read_text(encoding="utf-8") if cli_stderr_path.is_file() else ""
    log_path = _latest_path(run_dir, "geometry_fit_log_*.txt")
    trace_path = _latest_path(run_dir, "geometry_fit_trace_*.jsonl")
    matched_peaks_path = _latest_path(run_dir, "matched_peaks_*.npy")

    reported_log_path = _extract_reported_artifact_path(stdout_text, "Geometry fit log")
    if reported_log_path is None:
        reported_log_path = _extract_reported_artifact_path(stderr_text, "Geometry fit log")
    if reported_log_path is not None and reported_log_path.is_file():
        mirrored_log_path = _mirror_artifact_into_run_dir(reported_log_path, run_dir=run_dir)
        if mirrored_log_path is not None:
            log_path = mirrored_log_path

    if log_path is not None:
        reported_trace_path = _parse_trace_path_from_log(log_path)
        if reported_trace_path is None:
            reported_trace_path = _latest_path(log_path.parent, f"geometry_fit_trace_{state_path.stem}*.jsonl")
        mirrored_trace_path = _mirror_artifact_into_run_dir(reported_trace_path, run_dir=run_dir)
        if mirrored_trace_path is not None:
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
            if mirrored_matched_peaks_path is not None:
                matched_peaks_path = mirrored_matched_peaks_path

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


def _background_context(state_path: Path, out_state_path: Path | None) -> dict[str, object]:
    source_payload = _load_json(state_path)
    source_state = source_payload.get("state", {})
    files = source_state.get("files", {}) if isinstance(source_state, Mapping) else {}
    backgrounds = list(files.get("background_files", [])) if isinstance(files, Mapping) else []
    current_index = int(files.get("current_background_index", 0) or 0) if isinstance(files, Mapping) else 0
    out_state_exists = out_state_path is not None and out_state_path.is_file()
    return {
        "source_state_path": str(state_path),
        "fitted_state_path": str(out_state_path) if out_state_exists else None,
        "background_files": [str(item) for item in backgrounds],
        "current_background_index": current_index,
        "current_background_file": (
            str(backgrounds[current_index]) if 0 <= current_index < len(backgrounds) else None
        ),
    }


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
    identity_retention_after_fit = _build_identity_retention_summary(
        trace_records,
        str(after_summary["phase"]),
        matched_statuses=tuple(str(item) for item in after_summary["matched_statuses"]),
    )
    final_summary = dict(log_artifacts.get("final_summary", {}))
    optimizer_summary = dict(log_artifacts.get("optimizer_summary", {}))
    background_context = _background_context(
        artifacts.state_path,
        artifacts.out_state_path if artifacts.out_state_path.is_file() else None,
    )
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
        },
        "background_context": background_context,
        "decision_row": decision_row,
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
    return _json_safe(report)  # type: ignore[return-value]


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
    before_fit = dict(report.get("before_fit", {}))
    after_fit = dict(report.get("after_fit", {}))
    identity_retention = dict(report.get("identity_retention_after_fit", {}))
    overlay = dict(report.get("overlay_evidence", {}))
    start_parameters = list(report.get("start_parameters", []))
    fitted_parameters = list(report.get("fitted_parameters", []))
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
                raise RuntimeError(
                    "{name} exceeded {timeout:.1f}s last_phase={phase} pid={pid} out_dir={out_dir}".format(
                        name=state_path.stem,
                        timeout=timeout_seconds,
                        phase=last_phase,
                        pid=process.pid,
                        out_dir=run_dir,
                    )
                )
            time.sleep(1.0)
    log_path, trace_path, matched_peaks_path = _resolve_artifact_paths(
        state_path=state_path,
        run_dir=run_dir,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
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


def _write_report_files(report: Mapping[str, object], run_dir: Path, stem: str) -> tuple[Path, Path]:
    json_path = run_dir / f"{stem}_quality_report.json"
    md_path = run_dir / f"{stem}_quality_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_quality_report_markdown(report), encoding="utf-8")
    return json_path, md_path


def _default_output_root() -> Path:
    return Path.home() / ".local" / "share" / "ra_sim" / "fit_quality_baseline" / date.today().isoformat()


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
        _write_report_files(report, run_dir, state_path.stem)
        reports.append(dict(report))
        decision = dict(report.get("decision_row", {}))
        elapsed_seconds = time.monotonic() - started_at
        print(
            "{name}: finished in {elapsed:.1f}s accepted={accepted} metric={metric} rms_after_px={rms}".format(
                name=state_path.stem,
                elapsed=elapsed_seconds,
                accepted=decision.get("accepted"),
                metric=decision.get("final_metric_name"),
                rms=decision.get("rms_after_px"),
            ),
            flush=True,
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
        help="Optional saved-state JSON paths. Defaults to new2/new3 canonical baseline states.",
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
    for report in reports:
        decision = dict(report.get("decision_row", {}))
        print(
            "{name}: accepted={accepted} metric={metric} rms_after_px={rms}".format(
                name=str(report.get("state_name", "")),
                accepted=decision.get("accepted"),
                metric=decision.get("final_metric_name"),
                rms=decision.get("rms_after_px"),
            ),
            flush=True,
        )
    print(f"Output root: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
