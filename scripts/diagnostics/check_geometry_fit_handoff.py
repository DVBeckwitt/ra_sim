from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


def _summary_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    out: dict[str, object] = {}
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            continue
        key = str(item[0])
        out[key] = item[1]
    return out


def _flatten_record(record: Mapping[str, object]) -> dict[str, object]:
    flat = dict(record)
    for key in (
        "point_match_summary",
        "live_runtime_cache_validation",
        "geometry_fit_debug_summary",
    ):
        flat.update(_summary_mapping(record.get(key)))
    return flat


def _is_false(value: object) -> bool:
    return value is False or str(value).strip().lower() == "false"


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _record_from_text_line(text: str, line_number: int, parse_error: str) -> dict[str, object]:
    lowered = text.strip().lower()
    record: dict[str, object] = {
        "record_type": "text_log_line",
        "_line_number": int(line_number),
        "_text_line": text,
        "parse_error": parse_error,
    }
    if "objective_space=caked_deg" in lowered:
        record["objective_space"] = "caked_deg"
    if (
        "fit_observed_caked_deg=<unavailable" in lowered
        or "observed_caked_deg=<unavailable" in lowered
    ):
        record["fit_observed_caked_unavailable"] = True
    if "detector_to_caked_unavailable=true" in lowered:
        record["detector_to_caked_unavailable"] = True
    if "preflight: ready to solve geometry fit" in lowered:
        record["preflight_ready"] = True
    if "geometry fit: setup" in lowered or "optimizer_started=true" in lowered:
        record["optimizer_started"] = True
    if "weighted_rms=" in lowered and " px" in lowered:
        record["weighted_rms_unit"] = "px"
    if (
        "metric=central_point_match" in lowered
        or "final_metric_name=central_point_match" in lowered
    ):
        record["final_metric_name"] = "central_point_match"
    if "metric_unit=px" in lowered:
        record["metric_unit"] = "px"
    if "manual_caked_fit_space_required=false" in lowered:
        record["manual_caked_fit_space_required"] = False
    if "manual_caked_fit_space_required=true" in lowered:
        record["manual_caked_fit_space_required"] = True
    if "validator_finite_caked_rows=0" in lowered:
        record["validator_finite_caked_rows"] = 0
    if "matched=0" in lowered:
        record["matched_pair_count"] = 0
    return record


def _read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                records.append(_record_from_text_line(text, line_number, str(exc)))
                continue
            if isinstance(payload, Mapping):
                row = dict(payload)
                row["_line_number"] = int(line_number)
                records.append(row)
    return records


def violations_for_trace(path: Path) -> list[str]:
    records = [_flatten_record(record) for record in _read_records(path)]
    violations: list[str] = []

    saw_caked_required_missing_rows = False
    saw_caked_not_required_missing_rows = False
    saw_caked_objective = False
    saw_missing_observed_caked = False
    saw_preflight_ready = False
    saw_optimizer_run = False
    saw_central_point_match = False
    saw_matched_zero = False
    saw_metric_px = False
    saw_px_rms = False
    saw_central_px = False

    for record in records:
        line = int(record.get("_line_number", 0) or 0)
        objective_space = str(record.get("objective_space") or "").strip().lower()
        final_metric_name = str(
            record.get("final_metric_name") or record.get("metric_name") or ""
        ).strip()
        metric_unit = str(record.get("metric_unit") or record.get("final_metric_units") or "")
        caked_required = record.get("manual_caked_fit_space_required")
        finite_caked_rows = _as_int(record.get("validator_finite_caked_rows"), default=-1)
        matched_pair_count = _as_int(record.get("matched_pair_count"), default=-1)
        optimizer_started = (
            _as_int(record.get("nfev"), default=0) > 0
            or bool(record.get("stage_timing_s"))
            or _is_true(record.get("optimizer_started"))
        )

        if objective_space == "caked_deg":
            saw_caked_objective = True
        if (
            _is_true(record.get("fit_observed_caked_unavailable"))
            or _is_true(record.get("detector_to_caked_unavailable"))
            or str(record.get("fit_observed_caked_deg") or "")
            .strip()
            .lower()
            .startswith("<unavailable")
        ):
            saw_missing_observed_caked = True
        if _is_true(record.get("preflight_ready")):
            saw_preflight_ready = True
        if final_metric_name == "central_point_match":
            saw_central_point_match = True
        if matched_pair_count == 0:
            saw_matched_zero = True
        if metric_unit == "px":
            saw_metric_px = True
        if str(record.get("weighted_rms_unit") or "").strip().lower() == "px":
            saw_px_rms = True
        if objective_space == "caked_deg" and metric_unit == "px":
            violations.append(f"{path}:{line}: objective_space=caked_deg used metric_unit=px")
        if objective_space == "caked_deg" and final_metric_name == "central_point_match":
            violations.append(f"{path}:{line}: objective_space=caked_deg used central_point_match")
        if objective_space == "caked_deg" and str(
            record.get("weighted_rms_unit") or ""
        ).strip().lower() == "px":
            violations.append(f"{path}:{line}: objective_space=caked_deg used weighted_rms in px")
        if objective_space == "caked_deg" and _is_false(caked_required):
            violations.append(
                f"{path}:{line}: objective_space=caked_deg has manual_caked_fit_space_required=false"
            )
        if _is_true(caked_required) and finite_caked_rows == 0 and optimizer_started:
            violations.append(
                f"{path}:{line}: caked-required fit with zero finite caked rows reached optimizer"
            )

        if final_metric_name == "central_point_match" and metric_unit == "px":
            saw_central_px = True
        if optimizer_started:
            saw_optimizer_run = True
        if _is_true(caked_required) and finite_caked_rows == 0:
            saw_caked_required_missing_rows = True
        if _is_false(caked_required) and finite_caked_rows == 0:
            saw_caked_not_required_missing_rows = True

    if saw_caked_required_missing_rows and saw_optimizer_run:
        violations.append(
            f"{path}: manual_caked_fit_space_required=true with zero caked rows reached optimizer"
        )
    if saw_caked_objective and saw_central_point_match:
        violations.append(f"{path}: objective_space=caked_deg used central_point_match")
    if saw_caked_objective and saw_metric_px:
        violations.append(f"{path}: objective_space=caked_deg used metric_unit=px")
    if saw_caked_objective and saw_px_rms:
        violations.append(f"{path}: objective_space=caked_deg used weighted_rms in px")
    if saw_caked_objective and saw_optimizer_run and saw_matched_zero:
        violations.append(f"{path}: objective_space=caked_deg reached optimizer with matched=0")
    if saw_caked_objective and saw_missing_observed_caked and saw_preflight_ready:
        violations.append(
            f"{path}: missing observed caked coordinates reached preflight ready"
        )
    if saw_caked_not_required_missing_rows and saw_central_px:
        violations.append(
            f"{path}: broken signature present: manual_caked_fit_space_required=false, "
            "validator_finite_caked_rows=0, final_metric_name=central_point_match, metric_unit=px"
        )
    return violations


def _expand_patterns(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check geometry-fit traces for impossible caked handoff states."
    )
    parser.add_argument("traces", nargs="+", help="Trace JSONL path or glob pattern")
    args = parser.parse_args(argv)

    violations: list[str] = []
    for path in _expand_patterns(args.traces):
        if not path.exists():
            violations.append(f"{path}: trace file not found")
            continue
        violations.extend(violations_for_trace(path))

    if violations:
        for violation in violations:
            print(violation, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
