"""Headless Qr/Qz peak sensitivity export helpers.

This module is intentionally an analysis adapter.  It centralizes private GUI
helper calls and keeps finite-difference assembly, diagnostics, and artifact
writing independent from the simulation/cache schema.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ra_sim.io.data_loading import load_gui_state_file


DEFAULT_PARAMETER_NAMES: tuple[str, ...] = (
    "theta_initial",
    "psi_z",
    "chi",
    "cor_angle",
    "gamma",
    "Gamma",
    "corto_detector",
    "center_x",
    "center_y",
    "a",
    "c",
)
OPTIONAL_PARAMETER_NAMES: tuple[str, ...] = (
    "theta_offset",
    "sigma_mosaic",
    "gamma_mosaic",
    "eta",
    "beam_x",
    "beam_y",
    "wavelength",
)
COORDINATE_NAMES: tuple[str, str] = ("two_theta_deg", "phi_deg")
STABLE_DIAGNOSTIC_STATUSES: frozenset[str] = frozenset(
    {
        "ok",
        "missing_plus",
        "missing_minus",
        "missing_both",
        "identity_changed",
        "peak_jump",
        "eval_error",
        "nonfinite_peak",
        "missing_caked_payload",
    }
)
LONG_CSV_FIELDS: tuple[str, ...] = (
    "group_key",
    "branch_id",
    "coordinate",
    "parameter",
    "baseline_parameter_value",
    "step",
    "baseline_coordinate",
    "plus_coordinate",
    "minus_coordinate",
    "derivative",
    "normalized_derivative",
    "plus_identity",
    "minus_identity",
    "identity_changed",
    "status",
    "one_sided_plus",
    "one_sided_minus",
    "asymmetry",
    "branch_missing",
    "peak_jump_flag",
    "baseline_gui_mismatch",
    "eval_error",
)
BASELINE_CSV_FIELDS: tuple[str, ...] = (
    "group_key",
    "branch_id",
    "hkl",
    "source_reflection_index",
    "source_table_index",
    "source_row_index",
    "best_sample_index",
    "two_theta_deg",
    "phi_deg",
    "intensity",
    "refined_by",
    "selection_reason",
    "mosaic_top_rank_key",
)
PROVENANCE_FIELDS: tuple[str, ...] = (
    "hkl",
    "source_reflection_index",
    "source_table_index",
    "source_row_index",
    "source_branch_index",
    "source_peak_index",
    "source_reflection_namespace",
    "source_reflection_is_full",
)
TRUSTED_REFLECTION_FIELDS: tuple[str, ...] = (
    "source_reflection_index",
    "source_reflection_namespace",
    "source_reflection_is_full",
)
STABLE_REFLECTION_MATCH_FIELDS: tuple[str, ...] = (
    "source_table_index",
    "source_row_index",
    "source_branch_index",
    "source_peak_index",
)


@dataclass(frozen=True)
class PeakSensitivityParameter:
    name: str
    baseline_value: float
    step: float
    units: str
    scale_for_normalized_output: float


@dataclass(frozen=True)
class PeakObservation:
    group_key: tuple[object, ...] | None
    branch_id: str
    hkl: tuple[int, int, int] | None = None
    source_reflection_index: object = None
    source_table_index: object = None
    source_row_index: object = None
    best_sample_index: object = None
    two_theta_deg: float = math.nan
    phi_deg: float = math.nan
    intensity: float = math.nan
    refined_by: str | None = None
    selection_reason: str | None = None
    mosaic_top_rank_key: object = None
    source_branch_index: object = None
    source_peak_index: object = None
    source_reflection_namespace: object = None
    source_reflection_is_full: object = None
    source_record: dict[str, object] = field(default_factory=dict, compare=False)
    status: str = "ok"
    eval_error: str | None = None

    def provenance_key(self) -> tuple[object, ...]:
        return (
            self.hkl,
            self.source_reflection_index,
            self.source_table_index,
            self.source_row_index,
            self.source_branch_index,
            self.source_peak_index,
            self.source_reflection_namespace,
            self.source_reflection_is_full,
        )

    def match_key(self) -> tuple[tuple[object, ...] | None, str]:
        return self.group_key, self.branch_id


@dataclass(frozen=True)
class PeakSensitivityResult:
    baseline_observations: list[PeakObservation]
    plus_observations: dict[str, list[PeakObservation]]
    minus_observations: dict[str, list[PeakObservation]]
    jacobian: dict[tuple[str, str], dict[str, float]]
    normalized_jacobian: dict[tuple[str, str], dict[str, float]]
    diagnostics: dict[tuple[str, str, str], dict[str, object]]
    long_rows: list[dict[str, object]]
    parameters: list[PeakSensitivityParameter]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _ParameterRegistryEntry:
    name: str
    units: str
    var_name: str | None
    category: str
    getter: Callable[[Mapping[str, object]], float]
    param_setter: Callable[[dict[str, object], float], None]


class _CapturedPreflight(RuntimeError):
    """Stop the headless fit after preflight context capture."""


class PeakSensitivityEvaluationError(RuntimeError):
    """Baseline sensitivity evaluation failed before a trustworthy matrix existed."""


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    return float(numeric) if math.isfinite(numeric) else float(default)


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _stable_json(value: object) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


def _normalize_hkl(value: object) -> tuple[int, int, int] | None:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    if len(value) < 3:
        return None
    try:
        return tuple(int(round(float(item))) for item in value[:3])  # type: ignore[return-value]
    except Exception:
        return None


def _normalize_group_key(value: object) -> tuple[object, ...] | None:
    if isinstance(value, tuple):
        return tuple(_json_safe(item) for item in value)
    if isinstance(value, list):
        return tuple(_json_safe(item) for item in value)
    return None


def parse_group_key(value: str | Sequence[object] | tuple[object, ...]) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    text = str(value).strip()
    if not text:
        raise ValueError("group key is required")
    try:
        decoded = json.loads(text)
    except Exception:
        decoded = None
    if isinstance(decoded, list):
        return tuple(decoded)
    return tuple(_parse_group_key_token(part) for part in text.split(","))


def _parse_group_key_token(text: str) -> object:
    token = str(text).strip()
    if not token:
        return token
    try:
        as_int = int(token)
    except Exception:
        return token
    return int(as_int)


def wrapped_phi_delta(to_value: float, from_value: float) -> float:
    """Return the shortest signed angular delta from ``from_value`` to ``to_value``."""

    if not (math.isfinite(float(to_value)) and math.isfinite(float(from_value))):
        return math.nan
    return float((float(to_value) - float(from_value) + 180.0) % 360.0 - 180.0)


def coordinate_delta(
    to_value: float,
    from_value: float,
    *,
    coordinate: str,
) -> float:
    if str(coordinate) == "phi_deg":
        return wrapped_phi_delta(float(to_value), float(from_value))
    if not (math.isfinite(float(to_value)) and math.isfinite(float(from_value))):
        return math.nan
    return float(to_value) - float(from_value)


def _coordinate_value(observation: PeakObservation | None, coordinate: str) -> float:
    if observation is None:
        return math.nan
    if coordinate == "two_theta_deg":
        return float(observation.two_theta_deg)
    if coordinate == "phi_deg":
        return float(observation.phi_deg)
    raise ValueError(f"Unknown coordinate {coordinate!r}")


def _identity_text(observation: PeakObservation | None) -> str:
    if observation is None:
        return ""
    return _stable_json(observation.provenance_key())


def _index_identity_value(value: object) -> object:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return value


def _reflection_anchor_key(
    record: Mapping[str, object],
    *,
    group_key: tuple[object, ...] | None,
) -> tuple[object, ...] | None:
    normalized_group_key = _normalize_group_key(record.get("q_group_key")) or group_key
    normalized_hkl = _normalize_hkl(record.get("hkl", record.get("hkl_raw")))
    if normalized_group_key is None or normalized_hkl is None:
        return None
    values: list[object] = [normalized_group_key, normalized_hkl]
    for field in STABLE_REFLECTION_MATCH_FIELDS:
        value = _index_identity_value(record.get(field))
        if value is None:
            return None
        values.append(value)
    return tuple(values)


def _has_complete_trusted_reflection(record: Mapping[str, object]) -> bool:
    try:
        reflection_index = int(record.get("source_reflection_index"))
    except Exception:
        return False
    namespace = str(record.get("source_reflection_namespace", "") or "").strip().lower()
    return (
        reflection_index >= 0
        and namespace == "full_reflection"
        and record.get("source_reflection_is_full") is True
    )


def _restore_trusted_reflection_provenance(
    record: Mapping[str, object],
    *,
    group_key: tuple[object, ...] | None,
    required_pairs: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    restored = dict(record)
    if not required_pairs or _has_complete_trusted_reflection(restored):
        return restored
    anchor_key = _reflection_anchor_key(restored, group_key=group_key)
    if anchor_key is None:
        return restored
    matches = [
        pair
        for pair in required_pairs
        if isinstance(pair, Mapping)
        and _has_complete_trusted_reflection(pair)
        and _reflection_anchor_key(pair, group_key=group_key) == anchor_key
    ]
    if len(matches) != 1:
        return restored
    match = matches[0]
    for field in TRUSTED_REFLECTION_FIELDS:
        if restored.get(field) is None:
            restored[field] = match.get(field)
    return restored


def _observations_by_match_key(
    observations: Sequence[PeakObservation],
) -> dict[tuple[tuple[object, ...] | None, str], PeakObservation]:
    by_key: dict[tuple[tuple[object, ...] | None, str], PeakObservation] = {}
    for observation in observations:
        by_key.setdefault(observation.match_key(), observation)
    return by_key


def _diagnostic_status(
    *,
    missing_plus: bool,
    missing_minus: bool,
    missing_caked_payload: bool,
    coordinate_missing: bool,
    identity_changed: bool,
    peak_jump_flag: bool,
    eval_error: str | None,
    observation_status: str | None = None,
) -> str:
    observed_status = str(observation_status or "")
    if eval_error:
        return "eval_error"
    if observed_status == "eval_error":
        return "eval_error"
    if missing_caked_payload:
        return "missing_caked_payload"
    if observed_status == "missing_caked_payload":
        return "missing_caked_payload"
    if missing_plus and missing_minus:
        return "missing_both"
    if missing_plus:
        return "missing_plus"
    if missing_minus:
        return "missing_minus"
    if coordinate_missing or observed_status not in {"", "ok"}:
        return "nonfinite_peak"
    if identity_changed:
        return "identity_changed"
    if peak_jump_flag:
        return "peak_jump"
    return "ok"


def assemble_finite_difference_result(
    *,
    baseline_observations: Sequence[PeakObservation],
    plus_observations: Mapping[str, Sequence[PeakObservation]],
    minus_observations: Mapping[str, Sequence[PeakObservation]],
    parameters: Sequence[PeakSensitivityParameter],
    plus_eval_errors: Mapping[str, str] | None = None,
    minus_eval_errors: Mapping[str, str] | None = None,
    peak_jump_threshold: float | None = None,
) -> PeakSensitivityResult:
    """Assemble a deterministic central-difference sensitivity result."""

    baseline = list(baseline_observations)
    params = list(parameters)
    plus_errors = dict(plus_eval_errors or {})
    minus_errors = dict(minus_eval_errors or {})
    plus_map = {name: list(rows) for name, rows in plus_observations.items()}
    minus_map = {name: list(rows) for name, rows in minus_observations.items()}
    jacobian: dict[tuple[str, str], dict[str, float]] = {}
    normalized: dict[tuple[str, str], dict[str, float]] = {}
    diagnostics: dict[tuple[str, str, str], dict[str, object]] = {}
    long_rows: list[dict[str, object]] = []

    for observation in baseline:
        for coordinate in COORDINATE_NAMES:
            row_key = (str(observation.branch_id), coordinate)
            jacobian[row_key] = {}
            normalized[row_key] = {}

    for parameter in params:
        plus_by_key = _observations_by_match_key(plus_map.get(parameter.name, ()))
        minus_by_key = _observations_by_match_key(minus_map.get(parameter.name, ()))
        eval_error = plus_errors.get(parameter.name) or minus_errors.get(parameter.name)
        for observation in baseline:
            match_key = observation.match_key()
            plus = plus_by_key.get(match_key)
            minus = minus_by_key.get(match_key)
            missing_plus = plus is None
            missing_minus = minus is None
            status_candidates = [
                str(item.status)
                for item in (observation, plus, minus)
                if item is not None and item.status not in {"", "ok"}
            ]
            missing_caked_payload = "missing_caked_payload" in status_candidates
            observation_status = next(
                (status for status in status_candidates if status != "missing_caked_payload"),
                "",
            )
            observation_status_unusable = bool(observation_status and observation_status != "ok")
            identity_changed = bool(
                plus is not None
                and plus.provenance_key() != observation.provenance_key()
                or minus is not None
                and minus.provenance_key() != observation.provenance_key()
            )
            for coordinate in COORDINATE_NAMES:
                baseline_value = _coordinate_value(observation, coordinate)
                plus_value = _coordinate_value(plus, coordinate)
                minus_value = _coordinate_value(minus, coordinate)
                baseline_finite = math.isfinite(float(baseline_value))
                plus_finite = plus is not None and math.isfinite(float(plus_value))
                minus_finite = minus is not None and math.isfinite(float(minus_value))
                plus_delta = (
                    coordinate_delta(plus_value, baseline_value, coordinate=coordinate)
                    if baseline_finite and plus_finite
                    else math.nan
                )
                minus_delta = (
                    coordinate_delta(baseline_value, minus_value, coordinate=coordinate)
                    if baseline_finite and minus_finite
                    else math.nan
                )
                central_delta = (
                    coordinate_delta(plus_value, minus_value, coordinate=coordinate)
                    if plus_finite and minus_finite
                    else math.nan
                )
                one_sided_plus = (
                    float(plus_delta) / float(parameter.step)
                    if not eval_error and math.isfinite(float(plus_delta))
                    else math.nan
                )
                one_sided_minus = (
                    float(minus_delta) / float(parameter.step)
                    if not eval_error and math.isfinite(float(minus_delta))
                    else math.nan
                )
                coordinate_missing = not (baseline_finite and plus_finite and minus_finite)
                if (
                    eval_error
                    or missing_plus
                    or missing_minus
                    or missing_caked_payload
                    or coordinate_missing
                    or observation_status_unusable
                ):
                    derivative = math.nan
                    asymmetry = math.nan
                else:
                    derivative = float(central_delta) / (2.0 * float(parameter.step))
                    asymmetry = abs(float(one_sided_plus) - float(one_sided_minus))
                peak_jump_flag = False
                if peak_jump_threshold is not None and math.isfinite(float(peak_jump_threshold)):
                    peak_jump_flag = bool(
                        math.isfinite(float(plus_delta))
                        and abs(float(plus_delta)) > float(peak_jump_threshold)
                        or math.isfinite(float(minus_delta))
                        and abs(float(minus_delta)) > float(peak_jump_threshold)
                    )
                normalized_value = (
                    float(derivative) * float(parameter.scale_for_normalized_output)
                    if math.isfinite(float(derivative))
                    else math.nan
                )
                status = _diagnostic_status(
                    missing_plus=missing_plus,
                    missing_minus=missing_minus,
                    missing_caked_payload=missing_caked_payload,
                    coordinate_missing=coordinate_missing,
                    identity_changed=identity_changed,
                    peak_jump_flag=peak_jump_flag,
                    eval_error=eval_error,
                    observation_status=observation_status,
                )
                row_key = (str(observation.branch_id), coordinate)
                jacobian[row_key][parameter.name] = derivative
                normalized[row_key][parameter.name] = normalized_value
                diag = {
                    "one_sided_plus": one_sided_plus,
                    "one_sided_minus": one_sided_minus,
                    "asymmetry": asymmetry,
                    "identity_changed": bool(identity_changed),
                    "branch_missing": bool(missing_plus or missing_minus),
                    "missing_plus": bool(missing_plus),
                    "missing_minus": bool(missing_minus),
                    "coordinate_missing": bool(coordinate_missing),
                    "peak_jump_flag": bool(peak_jump_flag),
                    "status": status,
                    "eval_error": eval_error,
                    "observation_status": observation_status,
                    "plus_identity": _identity_text(plus),
                    "minus_identity": _identity_text(minus),
                }
                diagnostics[(str(observation.branch_id), coordinate, parameter.name)] = diag
                long_rows.append(
                    {
                        "group_key": _stable_json(observation.group_key),
                        "branch_id": str(observation.branch_id),
                        "coordinate": coordinate,
                        "parameter": parameter.name,
                        "baseline_parameter_value": parameter.baseline_value,
                        "step": parameter.step,
                        "baseline_coordinate": baseline_value,
                        "plus_coordinate": plus_value,
                        "minus_coordinate": minus_value,
                        "derivative": derivative,
                        "normalized_derivative": normalized_value,
                        "plus_identity": diag["plus_identity"],
                        "minus_identity": diag["minus_identity"],
                        "identity_changed": bool(identity_changed),
                        "status": status,
                        "one_sided_plus": one_sided_plus,
                        "one_sided_minus": one_sided_minus,
                        "asymmetry": asymmetry,
                        "branch_missing": bool(missing_plus or missing_minus),
                        "peak_jump_flag": bool(peak_jump_flag),
                        "baseline_gui_mismatch": False,
                        "eval_error": eval_error,
                    }
                )

    return PeakSensitivityResult(
        baseline_observations=baseline,
        plus_observations=plus_map,
        minus_observations=minus_map,
        jacobian=jacobian,
        normalized_jacobian=normalized,
        diagnostics=diagnostics,
        long_rows=long_rows,
        parameters=params,
    )


def _csv_value(value: object) -> object:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, (list, tuple, dict)):
        return _stable_json(value)
    return value


def write_sensitivity_artifacts(
    result: PeakSensitivityResult,
    outdir: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write matrix, long-form, baseline, metadata, and diagnostics artifacts."""

    output_dir = Path(outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_names = [parameter.name for parameter in result.parameters]
    paths = {
        "sensitivity_matrix": output_dir / "sensitivity_matrix.csv",
        "normalized_sensitivity_matrix": output_dir / "normalized_sensitivity_matrix.csv",
        "sensitivity_long": output_dir / "sensitivity_long.csv",
        "baseline_peaks": output_dir / "baseline_peaks.csv",
        "metadata": output_dir / "metadata.json",
        "diagnostics": output_dir / "diagnostics.json",
    }

    with paths["sensitivity_matrix"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["branch_id", "coordinate", *parameter_names])
        for row_key, values in result.jacobian.items():
            branch_id, coordinate = row_key
            writer.writerow(
                [
                    branch_id,
                    coordinate,
                    *[_csv_value(values.get(name, math.nan)) for name in parameter_names],
                ]
            )

    with paths["normalized_sensitivity_matrix"].open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["branch_id", "coordinate", *parameter_names])
        for row_key, values in result.normalized_jacobian.items():
            branch_id, coordinate = row_key
            writer.writerow(
                [
                    branch_id,
                    coordinate,
                    *[_csv_value(values.get(name, math.nan)) for name in parameter_names],
                ]
            )

    with paths["sensitivity_long"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LONG_CSV_FIELDS))
        writer.writeheader()
        for row in result.long_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in LONG_CSV_FIELDS})

    with paths["baseline_peaks"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BASELINE_CSV_FIELDS))
        writer.writeheader()
        for observation in result.baseline_observations:
            payload = {
                "group_key": _stable_json(observation.group_key),
                "branch_id": observation.branch_id,
                "hkl": _stable_json(observation.hkl),
                "source_reflection_index": observation.source_reflection_index,
                "source_table_index": observation.source_table_index,
                "source_row_index": observation.source_row_index,
                "best_sample_index": observation.best_sample_index,
                "two_theta_deg": observation.two_theta_deg,
                "phi_deg": observation.phi_deg,
                "intensity": observation.intensity,
                "refined_by": observation.refined_by,
                "selection_reason": observation.selection_reason,
                "mosaic_top_rank_key": _stable_json(observation.mosaic_top_rank_key),
            }
            writer.writerow(
                {field: _csv_value(payload.get(field)) for field in BASELINE_CSV_FIELDS}
            )

    metadata_payload = dict(metadata or {})
    metadata_payload.setdefault(
        "parameters", [_parameter_metadata(item) for item in result.parameters]
    )
    metadata_payload.setdefault("row_order", list(result.jacobian.keys()))
    paths["metadata"].write_text(
        json.dumps(_json_safe(metadata_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    diagnostics_payload = {
        "derivatives": [
            {
                "branch_id": key[0],
                "coordinate": key[1],
                "parameter": key[2],
                **dict(value),
            }
            for key, value in result.diagnostics.items()
        ]
    }
    paths["diagnostics"].write_text(
        json.dumps(_json_safe(diagnostics_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def _parameter_metadata(parameter: PeakSensitivityParameter) -> dict[str, object]:
    return {
        "name": parameter.name,
        "baseline_value": parameter.baseline_value,
        "step": parameter.step,
        "units": parameter.units,
        "scale_for_normalized_output": parameter.scale_for_normalized_output,
    }


def _plain_registry_getter(name: str) -> Callable[[Mapping[str, object]], float]:
    return lambda params: finite_float(params.get(name))


def _center_registry_getter(name: str) -> Callable[[Mapping[str, object]], float]:
    def _getter(params: Mapping[str, object]) -> float:
        if name in params:
            return finite_float(params.get(name))
        center = params.get("center")
        if (
            isinstance(center, Sequence)
            and not isinstance(center, (str, bytes))
            and len(center) >= 2
        ):
            return finite_float(center[0 if name == "center_x" else 1])
        return math.nan

    return _getter


def _set_plain_param(name: str) -> Callable[[dict[str, object], float], None]:
    def _setter(params: dict[str, object], value: float) -> None:
        params[name] = float(value)

    return _setter


def _set_center_param(name: str) -> Callable[[dict[str, object], float], None]:
    def _setter(params: dict[str, object], value: float) -> None:
        center = list(params.get("center", [math.nan, math.nan]))
        if len(center) < 2:
            center = [math.nan, math.nan]
        index = 0 if name == "center_x" else 1
        center[index] = float(value)
        params["center"] = center
        params[name] = float(value)

    return _setter


def _mosaic_getter(public_name: str, mosaic_name: str) -> Callable[[Mapping[str, object]], float]:
    def _getter(params: Mapping[str, object]) -> float:
        if public_name in params:
            return finite_float(params.get(public_name))
        mosaic = params.get("mosaic_params")
        if isinstance(mosaic, Mapping):
            return finite_float(mosaic.get(mosaic_name))
        return math.nan

    return _getter


def _set_mosaic_param(
    public_name: str, mosaic_name: str
) -> Callable[[dict[str, object], float], None]:
    def _setter(params: dict[str, object], value: float) -> None:
        mosaic = dict(
            params.get("mosaic_params", {})
            if isinstance(params.get("mosaic_params"), Mapping)
            else {}
        )
        mosaic[mosaic_name] = float(value)
        params["mosaic_params"] = mosaic
        params[public_name] = float(value)

    return _setter


def _array_mosaic_getter(
    public_name: str, array_name: str
) -> Callable[[Mapping[str, object]], float]:
    def _getter(params: Mapping[str, object]) -> float:
        if public_name in params:
            return finite_float(params.get(public_name))
        mosaic = params.get("mosaic_params")
        if isinstance(mosaic, Mapping):
            array = np.asarray(mosaic.get(array_name, []), dtype=np.float64).reshape(-1)
            if array.size:
                return finite_float(array[0])
        return math.nan

    return _getter


def _set_mosaic_array(
    public_name: str, array_name: str
) -> Callable[[dict[str, object], float], None]:
    def _setter(params: dict[str, object], value: float) -> None:
        mosaic = dict(
            params.get("mosaic_params", {})
            if isinstance(params.get("mosaic_params"), Mapping)
            else {}
        )
        source = np.asarray(mosaic.get(array_name, [0.0]), dtype=np.float64).reshape(-1)
        if source.size == 0:
            source = np.zeros(1, dtype=np.float64)
        source = source.copy()
        source[0] = float(value)
        mosaic[array_name] = source
        params["mosaic_params"] = mosaic
        params[public_name] = float(value)

    return _setter


def _wavelength_getter(params: Mapping[str, object]) -> float:
    if "wavelength" in params:
        return finite_float(params.get("wavelength"))
    if "lambda" in params:
        return finite_float(params.get("lambda"))
    mosaic = params.get("mosaic_params")
    if isinstance(mosaic, Mapping):
        for key in ("wavelength_array", "wavelength_i_array"):
            array = np.asarray(mosaic.get(key, []), dtype=np.float64).reshape(-1)
            if array.size:
                return finite_float(array[0])
    return math.nan


def _set_wavelength(params: dict[str, object], value: float) -> None:
    params["lambda"] = float(value)
    params["wavelength"] = float(value)
    mosaic = dict(
        params.get("mosaic_params", {}) if isinstance(params.get("mosaic_params"), Mapping) else {}
    )
    for key in ("wavelength_array", "wavelength_i_array"):
        source = np.asarray(mosaic.get(key, [value]), dtype=np.float64).reshape(-1)
        if source.size == 0:
            source = np.zeros(1, dtype=np.float64)
        source = source.copy()
        source[:] = float(value)
        mosaic[key] = source
    params["mosaic_params"] = mosaic


def build_parameter_registry() -> dict[str, _ParameterRegistryEntry]:
    entries = [
        _ParameterRegistryEntry(
            "theta_initial",
            "deg",
            "theta_initial_var",
            "angle",
            _plain_registry_getter("theta_initial"),
            _set_plain_param("theta_initial"),
        ),
        _ParameterRegistryEntry(
            "theta_offset",
            "deg",
            "geometry_theta_offset_var",
            "angle",
            _plain_registry_getter("theta_offset"),
            _set_plain_param("theta_offset"),
        ),
        _ParameterRegistryEntry(
            "psi_z",
            "deg",
            "psi_z_var",
            "angle",
            _plain_registry_getter("psi_z"),
            _set_plain_param("psi_z"),
        ),
        _ParameterRegistryEntry(
            "chi", "deg", "chi_var", "angle", _plain_registry_getter("chi"), _set_plain_param("chi")
        ),
        _ParameterRegistryEntry(
            "cor_angle",
            "deg",
            "cor_angle_var",
            "angle",
            _plain_registry_getter("cor_angle"),
            _set_plain_param("cor_angle"),
        ),
        _ParameterRegistryEntry(
            "gamma",
            "deg",
            "gamma_var",
            "angle",
            _plain_registry_getter("gamma"),
            _set_plain_param("gamma"),
        ),
        _ParameterRegistryEntry(
            "Gamma",
            "deg",
            "Gamma_var",
            "angle",
            _plain_registry_getter("Gamma"),
            _set_plain_param("Gamma"),
        ),
        _ParameterRegistryEntry(
            "corto_detector",
            "m",
            "corto_detector_var",
            "distance",
            _plain_registry_getter("corto_detector"),
            _set_plain_param("corto_detector"),
        ),
        _ParameterRegistryEntry(
            "center_x",
            "px",
            "center_x_var",
            "pixel",
            _center_registry_getter("center_x"),
            _set_center_param("center_x"),
        ),
        _ParameterRegistryEntry(
            "center_y",
            "px",
            "center_y_var",
            "pixel",
            _center_registry_getter("center_y"),
            _set_center_param("center_y"),
        ),
        _ParameterRegistryEntry(
            "a", "A", "a_var", "lattice", _plain_registry_getter("a"), _set_plain_param("a")
        ),
        _ParameterRegistryEntry(
            "c", "A", "c_var", "lattice", _plain_registry_getter("c"), _set_plain_param("c")
        ),
        _ParameterRegistryEntry(
            "sigma_mosaic",
            "deg",
            "sigma_mosaic_var",
            "dimensionless",
            _mosaic_getter("sigma_mosaic", "sigma_mosaic_deg"),
            _set_mosaic_param("sigma_mosaic", "sigma_mosaic_deg"),
        ),
        _ParameterRegistryEntry(
            "gamma_mosaic",
            "deg",
            "gamma_mosaic_var",
            "dimensionless",
            _mosaic_getter("gamma_mosaic", "gamma_mosaic_deg"),
            _set_mosaic_param("gamma_mosaic", "gamma_mosaic_deg"),
        ),
        _ParameterRegistryEntry(
            "eta",
            "dimensionless",
            "eta_var",
            "dimensionless",
            _mosaic_getter("eta", "eta"),
            _set_mosaic_param("eta", "eta"),
        ),
        _ParameterRegistryEntry(
            "beam_x",
            "m",
            None,
            "dimensionless",
            _array_mosaic_getter("beam_x", "beam_x_array"),
            _set_mosaic_array("beam_x", "beam_x_array"),
        ),
        _ParameterRegistryEntry(
            "beam_y",
            "m",
            None,
            "dimensionless",
            _array_mosaic_getter("beam_y", "beam_y_array"),
            _set_mosaic_array("beam_y", "beam_y_array"),
        ),
        _ParameterRegistryEntry(
            "wavelength", "A", None, "lattice", _wavelength_getter, _set_wavelength
        ),
    ]
    return {entry.name: entry for entry in entries}


def default_step_for_parameter(
    entry: _ParameterRegistryEntry,
    baseline_value: float,
    *,
    relative_step: float = 1.0e-4,
    step_mode: str = "default",
) -> float:
    category = str(entry.category)
    use_relative_all = str(step_mode).strip().lower() == "relative-all"
    if category == "angle" and not use_relative_all:
        return 0.01
    if category == "pixel" and not use_relative_all:
        return 0.25
    if category == "distance":
        return max(1.0e-5, abs(float(baseline_value)) * float(relative_step))
    if category == "lattice":
        return max(1.0e-4, abs(float(baseline_value)) * float(relative_step))
    return max(1.0e-5, abs(float(baseline_value)) * float(relative_step))


def build_sensitivity_parameters(
    names: Sequence[str],
    baseline_params: Mapping[str, object],
    *,
    relative_step: float = 1.0e-4,
    step_mode: str = "default",
) -> list[PeakSensitivityParameter]:
    registry = build_parameter_registry()
    parameters: list[PeakSensitivityParameter] = []
    for name in names:
        param_name = str(name).strip()
        if not param_name:
            continue
        if param_name not in registry:
            raise ValueError(f"Unknown sensitivity parameter: {param_name}")
        entry = registry[param_name]
        baseline_value = float(entry.getter(baseline_params))
        if not math.isfinite(baseline_value):
            raise ValueError(f"Parameter {param_name!r} has no finite baseline value")
        step = default_step_for_parameter(
            entry,
            baseline_value,
            relative_step=float(relative_step),
            step_mode=str(step_mode),
        )
        parameters.append(
            PeakSensitivityParameter(
                name=param_name,
                baseline_value=float(baseline_value),
                step=float(step),
                units=entry.units,
                scale_for_normalized_output=float(step),
            )
        )
    return parameters


def _apply_parameter_overrides(
    params: Mapping[str, object],
    overrides: Mapping[str, float],
) -> dict[str, object]:
    registry = build_parameter_registry()
    updated = copy.deepcopy(dict(params))
    for name, value in overrides.items():
        if name not in registry:
            raise ValueError(f"Unknown sensitivity parameter: {name}")
        registry[name].param_setter(updated, float(value))
    return updated


def _record_to_observation(
    record: Mapping[str, object],
    *,
    group_key: tuple[object, ...] | None,
    branch_id: str,
    status: str = "ok",
    eval_error: str | None = None,
) -> PeakObservation:
    return PeakObservation(
        group_key=group_key,
        branch_id=str(branch_id),
        hkl=_normalize_hkl(record.get("hkl", record.get("hkl_raw"))),
        source_reflection_index=record.get("source_reflection_index"),
        source_table_index=record.get("source_table_index"),
        source_row_index=record.get("source_row_index"),
        best_sample_index=record.get("best_sample_index"),
        two_theta_deg=finite_float(record.get("two_theta_deg", record.get("caked_x"))),
        phi_deg=finite_float(record.get("phi_deg", record.get("caked_y"))),
        intensity=finite_float(record.get("intensity", record.get("weight"))),
        refined_by=(
            str(record.get("refined_by")) if record.get("refined_by") is not None else None
        ),
        selection_reason=(
            str(record.get("selection_reason"))
            if record.get("selection_reason") is not None
            else None
        ),
        mosaic_top_rank_key=record.get("mosaic_top_rank_key"),
        source_branch_index=record.get("source_branch_index"),
        source_peak_index=record.get("source_peak_index"),
        source_reflection_namespace=record.get("source_reflection_namespace"),
        source_reflection_is_full=record.get("source_reflection_is_full"),
        source_record=dict(record),
        status=str(status),
        eval_error=eval_error,
    )


def _load_adapter_modules() -> SimpleNamespace:
    from ra_sim import headless_geometry_fit as hgf
    from ra_sim.gui import geometry_q_group_manager as q_groups
    from ra_sim.gui import manual_geometry
    from ra_sim.gui import peak_selection
    from ra_sim.simulation import exact_cake_portable

    return SimpleNamespace(
        hgf=hgf,
        q_groups=q_groups,
        manual_geometry=manual_geometry,
        peak_selection=peak_selection,
        exact_cake_portable=exact_cake_portable,
    )


@dataclass
class _CapturedRuntimeContext:
    state_path: Path
    saved_state: dict[str, object]
    prepare_kwargs: dict[str, object]
    prepare_result: object
    projection_callbacks: object | None
    bindings: object
    manual_dataset_bindings: object
    params: dict[str, object]
    dataset: dict[str, object]
    background_index: int
    modules: SimpleNamespace


def _capture_saved_state_preflight(
    *,
    saved_state: dict[str, object],
    state_path: Path,
) -> dict[str, object]:
    modules = _load_adapter_modules()
    hgf = modules.hgf
    captured: dict[str, object] = {}
    original_prepare = hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run
    original_projection = hgf.gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks

    def _projection_wrapper(*args, **kwargs):
        callbacks = original_projection(*args, **kwargs)
        captured["projection_callbacks"] = callbacks
        return callbacks

    def _prepare_wrapper(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        captured["prepare_kwargs"] = dict(kwargs)
        captured["prepare_result"] = result
        raise _CapturedPreflight()

    hgf.gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks = _projection_wrapper
    hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = _prepare_wrapper
    try:
        hgf.run_headless_geometry_fit(
            saved_state,
            state_path=state_path,
            downloads_dir=state_path.parent,
            stamp=f"{state_path.stem}_peak_sensitivity_probe",
        )
    except _CapturedPreflight:
        pass
    finally:
        hgf.gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks = (
            original_projection
        )
        hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = original_prepare
    if "prepare_kwargs" not in captured or "prepare_result" not in captured:
        raise RuntimeError("Failed to capture saved-state geometry-fit preflight context.")
    captured["modules"] = modules
    return captured


def _prepare_runtime_context(state_path: str | Path) -> _CapturedRuntimeContext:
    path = Path(state_path).expanduser().resolve()
    payload = load_gui_state_file(path)
    saved_state = dict(payload["state"])
    captured = _capture_saved_state_preflight(saved_state=saved_state, state_path=path)
    prepare_kwargs = dict(captured["prepare_kwargs"])
    prepare_result = captured["prepare_result"]
    prepared_run = getattr(prepare_result, "prepared_run", None)
    bindings = prepare_kwargs["bindings"]
    params = dict(prepare_kwargs.get("params") or {})
    manual_dataset_bindings = bindings.manual_dataset_bindings
    if prepared_run is not None:
        dataset = dict(getattr(prepared_run, "current_dataset", {}) or {})
    else:
        dataset = {}
    try:
        background_index = int(getattr(manual_dataset_bindings, "current_background_index"))
    except Exception:
        background_index = 0
    return _CapturedRuntimeContext(
        state_path=path,
        saved_state=saved_state,
        prepare_kwargs=prepare_kwargs,
        prepare_result=prepare_result,
        projection_callbacks=captured.get("projection_callbacks"),
        bindings=bindings,
        manual_dataset_bindings=manual_dataset_bindings,
        params=params,
        dataset=dataset,
        background_index=int(background_index),
        modules=captured["modules"],
    )


def _call_source_rows_for_background(
    callback: Callable[..., object],
    background_index: int,
    params: Mapping[str, object],
) -> list[dict[str, object]]:
    try:
        rows = callback(
            int(background_index),
            dict(params),
            consumer="peak_sensitivity",
        )
    except TypeError:
        rows = callback(int(background_index), dict(params))
    return [dict(row) for row in (rows or ()) if isinstance(row, Mapping)]


def _group_key_from_entry(
    modules: SimpleNamespace, entry: Mapping[str, object]
) -> tuple[object, ...] | None:
    key = modules.q_groups.geometry_q_group_key_from_entry(entry)
    return _normalize_group_key(key)


def _filter_group_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    modules: SimpleNamespace,
    group_key: tuple[object, ...],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in rows:
        row_group_key = _group_key_from_entry(modules, row)
        if row_group_key == group_key:
            item = dict(row)
            item["q_group_key"] = group_key
            filtered.append(item)
    return filtered


def _collapse_group_rows(
    rows: Sequence[dict[str, object]],
    *,
    modules: SimpleNamespace,
    profile_cache: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in modules.manual_geometry._geometry_manual_collapse_q_group_representatives(
            list(rows),
            profile_cache=profile_cache,
        )
        if isinstance(row, Mapping)
    ]


def _branch_id_for_row(
    row: Mapping[str, object],
    *,
    modules: SimpleNamespace,
    group_key: tuple[object, ...],
    profile_cache: Mapping[str, object] | None = None,
) -> str:
    from ra_sim.gui import mosaic_top_selection as mosaic_top

    branch_id, _source = mosaic_top.normalize_branch_id(
        row,
        target_key=group_key,
        profile_cache=profile_cache,
    )
    return str(branch_id)


def _detector_point_from_record(record: Mapping[str, object]) -> tuple[float, float] | None:
    for col_key, row_key in (
        ("native_col", "native_row"),
        ("detector_x", "detector_y"),
        ("background_detector_x", "background_detector_y"),
        ("sim_col_raw", "sim_row_raw"),
        ("sim_col", "sim_row"),
    ):
        col = finite_float(record.get(col_key))
        row = finite_float(record.get(row_key))
        if math.isfinite(col) and math.isfinite(row):
            return float(col), float(row)
    return None


def _bundle_detector_point_from_record(
    context: _CapturedRuntimeContext,
    record: Mapping[str, object],
) -> tuple[float, float] | None:
    point = _detector_point_from_record(record)
    if point is None:
        return None
    native_to_bundle = getattr(
        context.manual_dataset_bindings,
        "native_detector_coords_to_bundle_detector_coords",
        None,
    )
    if not callable(native_to_bundle):
        return point
    try:
        mapped = native_to_bundle(float(point[0]), float(point[1]))
    except Exception:
        return point
    if not isinstance(mapped, tuple) or len(mapped) < 2:
        return point
    col = finite_float(mapped[0])
    row = finite_float(mapped[1])
    if math.isfinite(col) and math.isfinite(row):
        return float(col), float(row)
    return point


def _build_integrator_for_params(
    params: Mapping[str, object],
    detector_shape: tuple[int, int],
    modules: SimpleNamespace,
):
    center = params.get("center")
    if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) < 2:
        center = [params.get("center_x"), params.get("center_y")]
    center_x = finite_float(center[0])
    center_y = finite_float(center[1])
    pixel_size = finite_float(params.get("pixel_size_m", params.get("pixel_size")), 100e-6)
    distance = finite_float(params.get("corto_detector"))
    if not (
        math.isfinite(center_x)
        and math.isfinite(center_y)
        and math.isfinite(pixel_size)
        and pixel_size > 0.0
        and math.isfinite(distance)
        and distance > 0.0
        and detector_shape[0] > 0
        and detector_shape[1] > 0
    ):
        return None
    return modules.exact_cake_portable.FastAzimuthalIntegrator(
        dist=float(distance),
        poni1=float(center_x) * float(pixel_size),
        poni2=float(center_y) * float(pixel_size),
        pixel1=float(pixel_size),
        pixel2=float(pixel_size),
    )


def _detector_shape_from_context(
    context: _CapturedRuntimeContext,
    *,
    image: np.ndarray | None = None,
) -> tuple[int, int]:
    if image is not None:
        shape = tuple(int(v) for v in np.asarray(image).shape[:2])
        if len(shape) >= 2 and min(shape) > 0:
            return shape[0], shape[1]
    dataset = context.dataset
    for owner in (
        dataset,
        dataset.get("spec") if isinstance(dataset.get("spec"), Mapping) else {},
    ):
        if not isinstance(owner, Mapping):
            continue
        for key in ("native_shape", "detector_shape", "image_shape", "shape"):
            value = owner.get(key)
            if (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes))
                and len(value) >= 2
            ):
                try:
                    return int(value[0]), int(value[1])
                except Exception:
                    pass
    prepared_run = getattr(context.prepare_result, "prepared_run", None)
    solver_request = getattr(prepared_run, "solver_request", None)
    try:
        image_size = int(getattr(solver_request, "image_size", 0) or 0)
    except Exception:
        image_size = 0
    if image_size > 0:
        return int(image_size), int(image_size)
    try:
        image_size = int(getattr(context.manual_dataset_bindings, "image_size", 0) or 0)
    except Exception:
        image_size = 0
    if image_size > 0:
        return int(image_size), int(image_size)
    return int(finite_float(context.params.get("image_size"), 0)), int(
        finite_float(context.params.get("image_size"), 0)
    )


def _detector_shape_from_transform_bundle(transform_bundle: object) -> tuple[int, int] | None:
    shape = getattr(transform_bundle, "detector_shape", None)
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or len(shape) < 2:
        return None
    try:
        detector_shape = int(shape[0]), int(shape[1])
    except Exception:
        return None
    if detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None
    return detector_shape


def _existing_caked_detector_shape(
    context: _CapturedRuntimeContext,
    transform_bundle: object,
) -> tuple[int, int]:
    bundle_shape = _detector_shape_from_transform_bundle(transform_bundle)
    if bundle_shape is not None:
        return bundle_shape
    return _detector_shape_from_context(context)


def _signature_float(value: object) -> float | None:
    numeric = finite_float(value)
    return float(numeric) if math.isfinite(numeric) else None


def _cake_geometry_signature(
    params: Mapping[str, object],
    detector_shape: Sequence[int] | tuple[int, int],
) -> dict[str, object]:
    center = params.get("center")
    if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) < 2:
        center = [params.get("center_x"), params.get("center_y")]
    shape = tuple(int(v) for v in list(detector_shape)[:2]) if len(detector_shape) >= 2 else (0, 0)
    return {
        "center": [_signature_float(center[0]), _signature_float(center[1])],
        "center_x": _signature_float(params.get("center_x", center[0])),
        "center_y": _signature_float(params.get("center_y", center[1])),
        "corto_detector": _signature_float(params.get("corto_detector")),
        "pixel_size": _signature_float(params.get("pixel_size")),
        "pixel_size_m": _signature_float(params.get("pixel_size_m", params.get("pixel_size"))),
        "wavelength": _signature_float(_wavelength_getter(params)),
        "lambda": _signature_float(params.get("lambda")),
        "detector_shape": [int(shape[0]), int(shape[1])],
    }


def _runtime_value(value: object) -> object:
    if callable(value):
        try:
            return value()
        except TypeError:
            return value
        except Exception:
            return None
    return value


def _existing_caked_payload_from_context(
    context: _CapturedRuntimeContext,
) -> tuple[SimpleNamespace | None, dict[str, object]]:
    image_names = ("last_caked_image_unscaled", "last_caked_background_image_unscaled")
    radial_names = ("last_caked_radial_values",)
    azimuth_names = ("last_caked_azimuth_values",)
    diagnostics: dict[str, object] = {"existing_caked_payload_checked": True}
    owners = (
        getattr(context, "prepare_result", None),
        getattr(context, "projection_callbacks", None),
        getattr(context, "manual_dataset_bindings", None),
        getattr(context, "bindings", None),
    )
    for owner in owners:
        if owner is None:
            continue
        image = next(
            (
                _runtime_value(getattr(owner, name, None))
                for name in image_names
                if getattr(owner, name, None) is not None
            ),
            None,
        )
        radial = next(
            (
                _runtime_value(getattr(owner, name, None))
                for name in radial_names
                if getattr(owner, name, None) is not None
            ),
            None,
        )
        azimuth = next(
            (
                _runtime_value(getattr(owner, name, None))
                for name in azimuth_names
                if getattr(owner, name, None) is not None
            ),
            None,
        )
        transform_bundle = _runtime_value(getattr(owner, "last_caked_transform_bundle", None))
        payload = SimpleNamespace(
            image=image,
            radial=radial,
            azimuth=azimuth,
            transform_bundle=transform_bundle,
            detector_shape=_existing_caked_detector_shape(context, transform_bundle),
        )
        if _caked_payload_available(payload):
            diagnostics.update(
                {
                    "caked_payload_status": "ok",
                    "payload_source": "existing_runtime_caked_payload",
                    "transform_bundle_available": payload.transform_bundle is not None,
                }
            )
            return payload, diagnostics
    diagnostics["existing_caked_payload_status"] = "unavailable"
    return None, diagnostics


def _real_detector_image_from_context(
    context: _CapturedRuntimeContext,
) -> tuple[np.ndarray | None, dict[str, object]]:
    diagnostics: dict[str, object] = {"real_runtime_image_checked": True}
    dataset = getattr(context, "dataset", {})
    candidates: list[Mapping[str, object]] = []
    if isinstance(dataset, Mapping):
        candidates.append(dataset)
        spec = dataset.get("spec")
        if isinstance(spec, Mapping):
            candidates.append(spec)
    for owner in candidates:
        for key in (
            "simulation_image",
            "detector_image",
            "background_image",
            "experimental_image",
            "image",
            "native_image",
        ):
            value = _runtime_value(owner.get(key))
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 2 and min(arr.shape[:2]) > 0:
                diagnostics.update(
                    {
                        "real_runtime_image_status": "ok",
                        "real_runtime_image_key": key,
                        "image_shape": [int(arr.shape[0]), int(arr.shape[1])],
                    }
                )
                return np.asarray(arr, dtype=np.float64), diagnostics
    diagnostics["real_runtime_image_status"] = "unavailable"
    return None, diagnostics


def _build_caked_payload(
    context: _CapturedRuntimeContext,
    params: Mapping[str, object],
) -> tuple[SimpleNamespace, dict[str, object]]:
    modules = context.modules
    existing_payload, existing_diag = _existing_caked_payload_from_context(context)
    if existing_payload is not None:
        existing_shape = getattr(existing_payload, "detector_shape", None)
        if not isinstance(existing_shape, Sequence) or len(existing_shape) < 2:
            existing_shape = _detector_shape_from_context(
                context, image=getattr(existing_payload, "image", None)
            )
        requested_signature = _cake_geometry_signature(params, existing_shape)
        baseline_signature = _cake_geometry_signature(context.params, existing_shape)
        if requested_signature == baseline_signature:
            existing_diag.update(
                {
                    "direct_simulation_fallback_used": False,
                    "direct_simulation_fallback_status": "disabled_unconfirmed",
                    "existing_caked_payload_reused": True,
                    "existing_caked_payload_reuse_reason": "matches_baseline_cake_geometry",
                    "cake_geometry_signature": requested_signature,
                }
            )
            return existing_payload, existing_diag
        existing_diag.update(
            {
                "existing_caked_payload_reused": False,
                "existing_caked_payload_reuse_reason": "cake_geometry_changed",
                "existing_caked_payload_baseline_signature": baseline_signature,
            }
        )
    image, image_diag = _real_detector_image_from_context(context)
    detector_shape = _detector_shape_from_context(context, image=image)
    requested_signature = _cake_geometry_signature(params, detector_shape)
    payload_diag: dict[str, object] = {
        **existing_diag,
        **image_diag,
        "direct_simulation_fallback_used": False,
        "direct_simulation_fallback_status": "disabled_unconfirmed",
        "existing_caked_payload_reused": False,
        "existing_caked_payload_reuse_reason": existing_diag.get(
            "existing_caked_payload_reuse_reason",
            "no_existing_payload",
        ),
        "cake_geometry_signature": requested_signature,
        "payload_source": "unavailable",
        "detector_shape": list(detector_shape),
        "npt_rad": 1000,
        "npt_azim": 720,
    }
    ai = _build_integrator_for_params(params, detector_shape, modules)
    if image is None or ai is None:
        payload_diag["caked_payload_status"] = "missing_caked_payload"
        return (
            SimpleNamespace(
                image=None,
                radial=None,
                azimuth=None,
                transform_bundle=None,
                detector_shape=detector_shape,
            ),
            payload_diag,
        )
    try:
        result = ai.integrate2d(
            image,
            npt_rad=int(payload_diag["npt_rad"]),
            npt_azim=int(payload_diag["npt_azim"]),
            correctSolidAngle=True,
            method="lut",
            unit="2th_deg",
        )
        cake, radial, azimuth = modules.exact_cake_portable.prepare_gui_phi_display(result)
        radial_mask = (np.asarray(radial) >= 0.0) & (np.asarray(radial) <= 90.0)
        if np.any(radial_mask):
            radial = np.asarray(radial, dtype=np.float64)[radial_mask]
            cake = np.asarray(cake, dtype=np.float64)[:, radial_mask]
        bundle = modules.exact_cake_portable.resolve_cake_transform_bundle(
            ai,
            detector_shape,
            np.asarray(radial, dtype=np.float64),
            gui_azimuth_deg=np.asarray(azimuth, dtype=np.float64),
            raw_azimuth_deg=np.asarray(result.azimuthal, dtype=np.float64),
            require_gui_display_match=True,
        )
        payload_diag["caked_payload_status"] = "ok"
        payload_diag["payload_source"] = "real_runtime_image"
        payload_diag["radial_count"] = int(np.asarray(radial).size)
        payload_diag["azimuth_count"] = int(np.asarray(azimuth).size)
        payload_diag["transform_bundle_available"] = bool(bundle is not None)
        return (
            SimpleNamespace(
                image=np.asarray(cake, dtype=np.float64),
                radial=np.asarray(radial, dtype=np.float64),
                azimuth=np.asarray(azimuth, dtype=np.float64),
                transform_bundle=bundle,
                detector_shape=detector_shape,
            ),
            payload_diag,
        )
    except Exception as exc:
        payload_diag.update(
            {
                "caked_payload_status": "failed",
                "error_type": type(exc).__name__,
                "error_text": str(exc),
            }
        )
        return (
            SimpleNamespace(
                image=None,
                radial=None,
                azimuth=None,
                transform_bundle=None,
                detector_shape=detector_shape,
            ),
            payload_diag,
        )


def _caked_payload_available(caked_payload: SimpleNamespace) -> bool:
    image = getattr(caked_payload, "image", None)
    radial = getattr(caked_payload, "radial", None)
    azimuth = getattr(caked_payload, "azimuth", None)
    try:
        return bool(
            image is not None
            and radial is not None
            and azimuth is not None
            and np.asarray(image).ndim == 2
            and np.asarray(radial).size > 0
            and np.asarray(azimuth).size > 0
        )
    except Exception:
        return False


def _record_with_missing_caked_peak(record: Mapping[str, object]) -> dict[str, object]:
    payload = dict(record)
    payload["caked_x"] = math.nan
    payload["caked_y"] = math.nan
    payload["two_theta_deg"] = math.nan
    payload["phi_deg"] = math.nan
    return payload


def refine_record_in_caked_payload(
    record: Mapping[str, object],
    *,
    caked_image: np.ndarray | None,
    radial_axis: Sequence[float] | None,
    azimuth_axis: Sequence[float] | None,
) -> tuple[dict[str, object] | None, tuple[float, float] | None]:
    """Testable wrapper around the existing caked peak refinement path."""

    from ra_sim.gui.peak_selection import _refine_selected_hkl_caked_record

    state = SimpleNamespace(
        last_caked_image_unscaled=caked_image,
        last_caked_radial_values=radial_axis,
        last_caked_azimuth_values=azimuth_axis,
        last_simulation_signature=None,
    )
    return _refine_selected_hkl_caked_record(state, record)


class PeakSensitivityEvaluator:
    def __init__(self, state_path: str | Path) -> None:
        self.context = _prepare_runtime_context(state_path)
        self.metadata: dict[str, object] = {
            "state_path": str(self.context.state_path),
            "state_sha256": _file_sha256(self.context.state_path),
            "code_version": _git_head(self.context.state_path.parent),
            "captured_preflight_error_text": getattr(
                self.context.prepare_result,
                "error_text",
                None,
            ),
        }
        self._last_eval_metadata: dict[str, object] = {}

    @property
    def baseline_params(self) -> dict[str, object]:
        return copy.deepcopy(dict(self.context.params))

    def evaluate_peak_observations(
        self,
        param_overrides: Mapping[str, float] | None,
        group_key: str | Sequence[object] | tuple[object, ...],
        branch_ids: Sequence[str] | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[PeakObservation]:
        requested_group_key = parse_group_key(group_key)
        var_restore: list[tuple[object, object]] = []
        registry = build_parameter_registry()
        try:
            params = _apply_parameter_overrides(self.context.params, param_overrides or {})
            for name, value in (param_overrides or {}).items():
                entry = registry.get(str(name))
                if entry is None or not entry.var_name:
                    continue
                variables = getattr(self.context.bindings, "__dict__", {})
                var_obj = variables.get(entry.var_name)
                if var_obj is None:
                    var_obj = getattr(self.context.manual_dataset_bindings, entry.var_name, None)
                if hasattr(var_obj, "get") and hasattr(var_obj, "set"):
                    old_value = var_obj.get()
                    var_restore.append((var_obj, old_value))
                    var_obj.set(float(value))
            return self._evaluate_with_params(
                params,
                requested_group_key,
                branch_ids=branch_ids,
                required_pairs=required_pairs,
            )
        except Exception as exc:
            self._last_eval_metadata = {
                "eval_error": f"{type(exc).__name__}: {exc}",
            }
            if branch_ids:
                return [
                    PeakObservation(
                        group_key=requested_group_key,
                        branch_id=str(branch),
                        status="eval_error",
                        eval_error=str(exc),
                    )
                    for branch in branch_ids
                ]
            return []
        finally:
            for var_obj, old_value in reversed(var_restore):
                try:
                    var_obj.set(old_value)
                except Exception:
                    pass

    def _evaluate_with_params(
        self,
        params: Mapping[str, object],
        group_key: tuple[object, ...],
        *,
        branch_ids: Sequence[str] | None,
        required_pairs: Sequence[Mapping[str, object]] | None,
    ) -> list[PeakObservation]:
        source_rows_callback = getattr(
            self.context.manual_dataset_bindings,
            "geometry_manual_source_rows_for_background",
            None,
        )
        if not callable(source_rows_callback):
            raise RuntimeError("Captured runtime context has no source-row adapter.")
        rows = _call_source_rows_for_background(
            source_rows_callback,
            self.context.background_index,
            params,
        )
        group_rows = _filter_group_rows(rows, modules=self.context.modules, group_key=group_key)
        caked_payload, cake_metadata = _build_caked_payload(
            self.context,
            params,
        )
        caked_available = _caked_payload_available(caked_payload)
        projected_rows = [
            self._ensure_caked_coordinates(row, group_key, caked_payload) for row in group_rows
        ]
        collapsed = _collapse_group_rows(
            projected_rows,
            modules=self.context.modules,
        )
        branch_filter = {str(branch) for branch in branch_ids or ()}
        observations: list[PeakObservation] = []
        for row in collapsed:
            branch_id = _branch_id_for_row(
                row,
                modules=self.context.modules,
                group_key=group_key,
            )
            if branch_filter and branch_id not in branch_filter:
                continue
            if not caked_available:
                observations.append(
                    _record_to_observation(
                        _record_with_missing_caked_peak(row),
                        group_key=group_key,
                        branch_id=branch_id,
                        status="missing_caked_payload",
                    )
                )
                continue
            refined, _point = self.context.modules.peak_selection._refine_selected_hkl_caked_record(
                SimpleNamespace(
                    last_caked_image_unscaled=caked_payload.image,
                    last_caked_radial_values=caked_payload.radial,
                    last_caked_azimuth_values=caked_payload.azimuth,
                    last_simulation_signature=None,
                ),
                row,
            )
            if not isinstance(refined, Mapping) or _point is None:
                observations.append(
                    _record_to_observation(
                        _record_with_missing_caked_peak(row),
                        group_key=group_key,
                        branch_id=branch_id,
                        status="nonfinite_peak",
                    )
                )
                continue
            record = _restore_trusted_reflection_provenance(
                refined,
                group_key=group_key,
                required_pairs=required_pairs,
            )
            observations.append(
                _record_to_observation(
                    record,
                    group_key=group_key,
                    branch_id=branch_id,
                )
            )
        observations.sort(key=lambda item: _branch_sort_key(item.branch_id))
        self._last_eval_metadata = {
            "source_row_count": int(len(rows)),
            "group_row_count": int(len(group_rows)),
            "collapsed_row_count": int(len(collapsed)),
            "observation_count": int(len(observations)),
            "cake": cake_metadata,
        }
        return observations

    def _ensure_caked_coordinates(
        self,
        row: Mapping[str, object],
        group_key: tuple[object, ...],
        caked_payload: SimpleNamespace,
    ) -> dict[str, object]:
        record = dict(row)
        if math.isfinite(finite_float(record.get("caked_x"))) and math.isfinite(
            finite_float(record.get("caked_y"))
        ):
            record.setdefault("two_theta_deg", finite_float(record.get("caked_x")))
            record.setdefault("phi_deg", finite_float(record.get("caked_y")))
            return record
        detector_point = _bundle_detector_point_from_record(self.context, record)
        bundle = getattr(caked_payload, "transform_bundle", None)
        if detector_point is not None and bundle is not None:
            two_theta, phi = self.context.modules.exact_cake_portable.detector_pixel_to_caked_bin(
                bundle,
                float(detector_point[0]),
                float(detector_point[1]),
            )
            if two_theta is not None and phi is not None:
                record["caked_x"] = float(two_theta)
                record["caked_y"] = float(phi)
                record["two_theta_deg"] = float(two_theta)
                record["phi_deg"] = float(phi)
                record["caked_projection_source"] = "detector_pixel_to_caked_bin"
        record["q_group_key"] = group_key
        return record


def _branch_sort_key(branch_id: str) -> tuple[int, str]:
    order = {"+x": 0, "-x": 1, "00l": 2}
    return order.get(str(branch_id), 100), str(branch_id)


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return None


def _git_head(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    return text or None


def _baseline_saved_peak_smoke_check(
    saved_state: Mapping[str, object],
    baseline: Sequence[PeakObservation],
    *,
    group_key: tuple[object, ...],
) -> dict[str, object]:
    geometry = (
        saved_state.get("geometry") if isinstance(saved_state.get("geometry"), Mapping) else {}
    )
    manual_pairs = geometry.get("manual_pairs") if isinstance(geometry, Mapping) else None
    saved_entries: list[dict[str, object]] = []
    for raw_group in manual_pairs or ():
        if not isinstance(raw_group, Mapping):
            continue
        for entry in raw_group.get("entries", []) or ():
            if (
                isinstance(entry, Mapping)
                and _normalize_group_key(entry.get("q_group_key")) == group_key
            ):
                saved_entries.append(dict(entry))
    if not saved_entries:
        return {"status": "skipped_no_saved_gui_peak", "checked_count": 0, "unreliable": False}
    by_branch = {item.branch_id: item for item in baseline}
    checked = 0
    failures: list[dict[str, object]] = []
    for entry in saved_entries:
        branch = str(
            entry.get("branch_id")
            or (
                "00l"
                if _normalize_hkl(entry.get("hkl"))
                and _normalize_hkl(entry.get("hkl"))[:2] == (0, 0)
                else ""
            )
        )
        if not branch:
            source_branch = entry.get("source_branch_index")
            if source_branch is not None:
                try:
                    branch = "+x" if int(source_branch) >= 0 else "-x"
                except Exception:
                    branch = ""
        observation = by_branch.get(branch)
        if observation is None:
            continue
        saved_tth = finite_float(entry.get("refined_sim_caked_x", entry.get("caked_x")))
        saved_phi = finite_float(entry.get("refined_sim_caked_y", entry.get("caked_y")))
        if math.isfinite(saved_tth) and math.isfinite(saved_phi):
            checked += 1
            runtime_tth_finite = math.isfinite(float(observation.two_theta_deg))
            runtime_phi_finite = math.isfinite(float(observation.phi_deg))
            tth_delta = (
                float(observation.two_theta_deg) - float(saved_tth)
                if runtime_tth_finite
                else math.nan
            )
            phi_delta = wrapped_phi_delta(observation.phi_deg, saved_phi)
            if (
                not runtime_tth_finite
                or not runtime_phi_finite
                or abs(tth_delta) > 1.0e-6
                or abs(phi_delta) > 1.0e-6
            ):
                failures.append(
                    {
                        "branch_id": branch,
                        "baseline": [observation.two_theta_deg, observation.phi_deg],
                        "saved": [saved_tth, saved_phi],
                        "delta": [tth_delta, phi_delta],
                    }
                )
    if checked <= 0:
        return {"status": "skipped_no_saved_gui_peak", "checked_count": 0, "unreliable": False}
    return {
        "status": "pass" if not failures else "mismatch",
        "unreliable": bool(failures),
        "checked_count": int(checked),
        "failures": failures,
    }


def _mark_baseline_gui_mismatches(
    result: PeakSensitivityResult,
    smoke_check: Mapping[str, object],
) -> None:
    failures = smoke_check.get("failures") if isinstance(smoke_check, Mapping) else None
    branches = {
        str(item.get("branch_id"))
        for item in failures or ()
        if isinstance(item, Mapping) and item.get("branch_id") is not None
    }
    if not branches:
        return
    for key, diag in result.diagnostics.items():
        branch_id = str(key[0])
        if branch_id not in branches:
            continue
        diag["baseline_gui_mismatch"] = True
    for row in result.long_rows:
        if str(row.get("branch_id")) not in branches:
            continue
        row["baseline_gui_mismatch"] = True


def _baseline_required_pairs(
    baseline: Sequence[PeakObservation],
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for observation in baseline:
        record = (
            dict(observation.source_record)
            if isinstance(observation.source_record, Mapping)
            else {}
        )
        record.setdefault("q_group_key", observation.group_key)
        if observation.hkl is not None:
            record.setdefault("hkl", observation.hkl)
        for field in (
            *TRUSTED_REFLECTION_FIELDS,
            *STABLE_REFLECTION_MATCH_FIELDS,
        ):
            value = getattr(observation, field, None)
            if value is not None:
                record.setdefault(field, value)
        pairs.append(record)
    return pairs


def _baseline_unusable_reason(baseline: Sequence[PeakObservation]) -> str | None:
    if not baseline:
        return "Baseline evaluation returned no observations."
    if all(item.status == "missing_caked_payload" for item in baseline):
        return "Baseline evaluation missing caked payload."
    has_finite_caked_peak = any(
        math.isfinite(float(item.two_theta_deg)) and math.isfinite(float(item.phi_deg))
        for item in baseline
    )
    if not has_finite_caked_peak:
        return "Baseline evaluation returned no finite caked peaks."
    return None


def run_peak_sensitivity(
    *,
    state_path: str | Path,
    group_key: str | Sequence[object] | tuple[object, ...],
    parameter_names: Sequence[str] = DEFAULT_PARAMETER_NAMES,
    relative_step: float = 1.0e-4,
    step_mode: str = "default",
    outdir: str | Path | None = None,
) -> PeakSensitivityResult:
    evaluator = PeakSensitivityEvaluator(state_path)
    parsed_group_key = parse_group_key(group_key)
    baseline = evaluator.evaluate_peak_observations({}, parsed_group_key, branch_ids=None)
    baseline_eval_metadata = dict(evaluator._last_eval_metadata)
    top_level_eval_error = baseline_eval_metadata.get("eval_error")
    if top_level_eval_error:
        raise PeakSensitivityEvaluationError(str(top_level_eval_error))
    unusable_reason = _baseline_unusable_reason(baseline)
    if unusable_reason is not None:
        raise PeakSensitivityEvaluationError(unusable_reason)
    branch_ids = [item.branch_id for item in baseline]
    required_pairs = _baseline_required_pairs(baseline)
    parameters = build_sensitivity_parameters(
        parameter_names,
        evaluator.baseline_params,
        relative_step=float(relative_step),
        step_mode=str(step_mode),
    )
    plus: dict[str, list[PeakObservation]] = {}
    minus: dict[str, list[PeakObservation]] = {}
    plus_errors: dict[str, str] = {}
    minus_errors: dict[str, str] = {}
    eval_metadata: dict[str, object] = {"baseline": baseline_eval_metadata}
    for parameter in parameters:
        plus_value = parameter.baseline_value + parameter.step
        minus_value = parameter.baseline_value - parameter.step
        plus[parameter.name] = evaluator.evaluate_peak_observations(
            {parameter.name: plus_value},
            parsed_group_key,
            branch_ids=branch_ids,
            required_pairs=required_pairs,
        )
        eval_metadata[f"{parameter.name}:plus"] = evaluator._last_eval_metadata
        if evaluator._last_eval_metadata.get("eval_error"):
            plus_errors[parameter.name] = str(evaluator._last_eval_metadata["eval_error"])
        minus[parameter.name] = evaluator.evaluate_peak_observations(
            {parameter.name: minus_value},
            parsed_group_key,
            branch_ids=branch_ids,
            required_pairs=required_pairs,
        )
        eval_metadata[f"{parameter.name}:minus"] = evaluator._last_eval_metadata
        if evaluator._last_eval_metadata.get("eval_error"):
            minus_errors[parameter.name] = str(evaluator._last_eval_metadata["eval_error"])
    result = assemble_finite_difference_result(
        baseline_observations=baseline,
        plus_observations=plus,
        minus_observations=minus,
        parameters=parameters,
        plus_eval_errors=plus_errors,
        minus_eval_errors=minus_errors,
        peak_jump_threshold=0.5,
    )
    smoke_check = _baseline_saved_peak_smoke_check(
        evaluator.context.saved_state,
        baseline,
        group_key=parsed_group_key,
    )
    _mark_baseline_gui_mismatches(result, smoke_check)
    metadata = dict(evaluator.metadata)
    metadata.update(
        {
            "status": "ok",
            "eval_error": None,
            "group_key": parsed_group_key,
            "params": [parameter.name for parameter in parameters],
            "steps": {parameter.name: parameter.step for parameter in parameters},
            "units": {parameter.name: parameter.units for parameter in parameters},
            "parameter_scales": {
                parameter.name: parameter.scale_for_normalized_output for parameter in parameters
            },
            "relative_step": float(relative_step),
            "step_mode": str(step_mode),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "caked_search_window": {
                "peak_jump_threshold": 0.5,
                "units": "deg",
            },
            "evaluations": eval_metadata,
            "baseline_coordinate_source": "runtime_evaluator",
            "baseline_gui_smoke_check": smoke_check,
        }
    )
    final_result = PeakSensitivityResult(
        baseline_observations=result.baseline_observations,
        plus_observations=result.plus_observations,
        minus_observations=result.minus_observations,
        jacobian=result.jacobian,
        normalized_jacobian=result.normalized_jacobian,
        diagnostics=result.diagnostics,
        long_rows=result.long_rows,
        parameters=result.parameters,
        metadata=metadata,
    )
    if outdir is not None:
        write_sensitivity_artifacts(final_result, outdir, metadata=metadata)
    return final_result
