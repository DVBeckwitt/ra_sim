"""Run bounded new4 geometry-fit optimizer probes.

This script is intentionally narrow: it validates that the optimizer can
consume the already-verified new4 manual point-provider pairs. It does not
change saved GUI state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.fitting import optimization as opt
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.io.data_loading import load_gui_state_file
from scripts.debug import validate_geometry_preflight_rebind as preflight_probe


REQUIRED_PROVIDER_COUNTS = {
    "manual_picker_pair_count": 7,
    "point_provider_pair_count": 7,
    "dataset_provider_mismatch_count": 0,
    "fallback_pair_count": 0,
}
EXPECTED_PROVIDER_PAIR_COUNT = 7

STRICT_POINT_SUMMARY_KEYS = (
    "measured_count",
    "fixed_source_resolved_count",
    "fixed_source_reflection_count",
    "matched_fixed_pair_count",
    "missing_fixed_pair_count",
    "fallback_entry_count",
    "fallback_hkl_count",
    "subset_fallback_hkl_count",
)

REQUEST_HANDOFF_FIELDS = (
    "provider_pair_count",
    "dataset_pair_count",
    "optimizer_request_pair_count",
    "fixed_source_pair_count",
    "fallback_row_count",
    "fixed_source_resolution_fallback_count",
    "missing_fixed_source_count",
    "branch_mismatch_count",
    "provider_to_optimizer_identity_match",
    "provider_to_optimizer_point_match",
)

NO_MATCH_REJECTION = "No matched peak pairs were available for the fitted solution."

ONE_PARAM_FIXED_SOURCE_COUNTS = {
    "fixed_source_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "fallback_row_count": 0,
    "fixed_source_resolution_fallback_count": 0,
    "missing_fixed_source_count": 0,
    "fixed_source_resolved_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "fallback_entry_count": 0,
    "matched_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "missing_pair_count": 0,
    "branch_mismatch_count": 0,
}

RUNG2_FIXED_SOURCE_COUNTS = {
    "provider_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "dataset_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "optimizer_request_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    **ONE_PARAM_FIXED_SOURCE_COUNTS,
}

PROVIDER_MATCH_BOOLS = (
    "provider_to_optimizer_identity_match",
    "provider_to_optimizer_point_match",
)

LIVE_HEARTBEAT_COUNTERS = {
    "fixed_source_resolved_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "fallback_entry_count": 0,
    "matched_pair_count": EXPECTED_PROVIDER_PAIR_COUNT,
    "missing_pair_count": 0,
    "branch_mismatch_count": 0,
}

ONE_PARAM_A_VARIANTS = (
    ("a_nfev5_t120", 5, 120.0),
    ("a_nfev10_t120", 10, 120.0),
    ("a_nfev20_t300", 20, 300.0),
)

ONE_PARAM_ORDER = [
    "center_x",
    "center_y",
    "gamma",
    "Gamma",
    "chi",
    "cor_angle",
    "theta_initial",
    "corto_detector",
    "zs",
    "zb",
    "a",
    "c",
    "psi_z",
]

CENTER_PARAMS = ["center_x", "center_y"]
FEATURE_RUNS = [
    "dynamic_reanchor",
    "discrete_modes",
    "seed_multistart",
    "full_beam_polish",
    "identifiability_features",
]


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _state_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _rung_path(run_dir: Path, number: int, name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
    return run_dir / f"rung_{int(number):02d}_{safe_name}.json"


def _provider_guard_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if bool(report.get("ok", False)) is not True:
        failures.append("ok_not_true")
    if str(report.get("classification", "")) != "point_provider_parity_ok":
        failures.append("classification_not_point_provider_parity_ok")
    gate = report.get("point_provider_parity_gate")
    if not isinstance(gate, Mapping) or bool(gate.get("ok", False)) is not True:
        failures.append("point_provider_parity_gate_not_ok")
    for key, expected in REQUIRED_PROVIDER_COUNTS.items():
        try:
            actual = int(report.get(key, -999999))
        except Exception:
            actual = -999999
        if actual != int(expected):
            failures.append(f"{key}_{actual}_expected_{expected}")
    if bool(report.get("optimizer_called", True)):
        failures.append("optimizer_called")
    return failures


def run_provider_guard(
    *,
    state_path: Path,
    background_index: int,
    output_path: Path,
) -> dict[str, object]:
    started = time.monotonic()
    report = preflight_probe._run_point_provider_report_only(
        Path(state_path),
        background_index=int(background_index),
    )
    failures = _provider_guard_failures(report)
    payload = {
        **dict(report),
        "rung": 0,
        "rung_name": "provider_guard",
        "status": "pass" if not failures else "fail",
        "provider_guard_ok": not failures,
        "provider_guard_failures": failures,
        "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
    }
    _write_json(output_path, payload)
    return payload


def _load_saved_state_for_background(state_path: Path, background_index: int) -> dict[str, object]:
    payload = load_gui_state_file(Path(state_path))
    saved_state = copy.deepcopy(dict(payload["state"]))
    files_state = saved_state.setdefault("files", {})
    if isinstance(files_state, dict):
        files_state["current_background_index"] = int(background_index)
    variables = saved_state.setdefault("variables", {})
    if isinstance(variables, dict):
        variables["geometry_fit_background_selection_var"] = "current"
    return saved_state


def _capture_solver_context(state_path: Path, background_index: int) -> dict[str, object]:
    saved_state = _load_saved_state_for_background(state_path, background_index)
    captured = preflight_probe._capture_execution_setup(
        saved_state=saved_state,
        state_path=Path(state_path),
    )
    prepare_result = captured.get("prepare_result")
    prepared_run = getattr(prepare_result, "prepared_run", None)
    if prepared_run is None:
        error_text = str(getattr(prepare_result, "error_text", "") or "")
        raise RuntimeError(error_text or "Geometry fit preflight failed.")
    execute_kwargs = (
        dict(captured.get("execute_kwargs", {}))
        if isinstance(captured.get("execute_kwargs"), Mapping)
        else {}
    )
    setup = execute_kwargs.get("setup")
    postprocess_config = getattr(setup, "postprocess_config", None)
    solver_inputs = getattr(postprocess_config, "solver_inputs", None)
    if solver_inputs is None:
        raise RuntimeError("Captured headless setup did not include solver inputs.")
    prepare_kwargs = (
        dict(captured.get("prepare_kwargs", {}))
        if isinstance(captured.get("prepare_kwargs"), Mapping)
        else {}
    )
    return {
        "state_path": str(state_path),
        "background_index": int(background_index),
        "prepared_run": prepared_run,
        "solver_inputs": solver_inputs,
        "saved_var_names": [
            str(name)
            for name in execute_kwargs.get("var_names", prepare_kwargs.get("var_names", ())) or ()
        ],
    }


def _coerce_active_names(context: Mapping[str, object], names: Sequence[object]) -> list[str]:
    prepared_run = context["prepared_run"]
    fit_params = (
        dict(getattr(prepared_run, "fit_params", {}) or {})
        if prepared_run is not None
        else {}
    )
    active: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        if name == "theta_offset" and "theta_offset" not in fit_params:
            name = "theta_initial"
        if name not in fit_params:
            continue
        if name not in active:
            active.append(name)
    return active


def _lean_runtime_config(
    runtime_cfg: Mapping[str, object] | None,
    *,
    active_names: Sequence[str],
    max_nfev: int,
    feature: str | None = None,
) -> dict[str, object]:
    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    cfg["candidate_param_names"] = [str(name) for name in active_names]

    solver_raw = cfg.get("solver", cfg.get("optimizer", {}))
    solver = dict(solver_raw) if isinstance(solver_raw, Mapping) else {}
    solver["manual_point_fit_mode"] = True
    solver.pop("dynamic_point_geometry_fit", None)
    solver["max_nfev"] = int(max_nfev)
    solver["loss"] = "linear"
    solver["f_scale_px"] = 1.0
    solver["weighted_matching"] = False
    solver["use_measurement_uncertainty"] = False
    solver["anisotropic_measurement_uncertainty"] = False
    solver["workers"] = solver.get("workers", "auto")
    solver["parallel_mode"] = "off"
    solver["worker_numba_threads"] = 0
    cfg["solver"] = solver
    cfg["optimizer"] = solver

    seed = dict(cfg.get("seed_search", {}) if isinstance(cfg.get("seed_search"), Mapping) else {})
    seed["prescore_top_k"] = 1
    seed["n_global"] = 0
    seed["n_jitter"] = 0
    seed["min_seed_separation_u"] = 2.0
    cfg["seed_search"] = seed

    cfg["use_numba"] = bool(cfg.get("use_numba", False))
    cfg["allow_unsafe_runtime"] = False
    cfg.pop("full_beam_polish", None)
    cfg.pop("ridge_refinement", None)
    cfg.pop("image_refinement", None)

    discrete = dict(
        cfg.get("discrete_modes", {}) if isinstance(cfg.get("discrete_modes"), Mapping) else {}
    )
    discrete["enabled"] = False
    cfg["discrete_modes"] = discrete

    ident = dict(
        cfg.get("identifiability", {}) if isinstance(cfg.get("identifiability"), Mapping) else {}
    )
    ident["enabled"] = True
    ident.pop("auto_freeze", None)
    ident.pop("selective_thaw", None)
    ident.pop("adaptive_regularization", None)
    cfg["identifiability"] = ident

    feature_name = str(feature or "").strip().lower()
    if feature_name == "discrete_modes":
        cfg["discrete_modes"] = {"enabled": True}
    elif feature_name == "seed_multistart":
        seed["prescore_top_k"] = 4
        seed["n_global"] = 4
        seed["n_jitter"] = 2
        seed["min_seed_separation_u"] = 0.5
    elif feature_name == "full_beam_polish":
        cfg["full_beam_polish"] = {"enabled": True, "max_nfev": max(5, int(max_nfev))}
    elif feature_name == "identifiability_features":
        ident["auto_freeze"] = True
        ident["selective_thaw"] = {"enabled": True}
        ident["adaptive_regularization"] = {"enabled": True}
    elif feature_name == "dynamic_reanchor":
        cfg["dynamic_reanchor_probe_requested"] = True
    return cfg


def _prepared_run_for_active_names(
    context: Mapping[str, object],
    active_names: Sequence[str],
    *,
    max_nfev: int,
    feature: str | None = None,
):
    prepared_run = context["prepared_run"]
    runtime_cfg = _lean_runtime_config(
        getattr(prepared_run, "geometry_runtime_cfg", {}),
        active_names=active_names,
        max_nfev=int(max_nfev),
        feature=feature,
    )
    return preflight_probe.replace(
        prepared_run,
        geometry_runtime_cfg=runtime_cfg,
    )


def build_solver_request(
    context: Mapping[str, object],
    active_names: Sequence[object],
    *,
    max_nfev: int = 20,
    feature: str | None = None,
) -> gui_geometry_fit.GeometryFitSolverRequest:
    names = _coerce_active_names(context, active_names)
    prepared_run = _prepared_run_for_active_names(
        context,
        names,
        max_nfev=int(max_nfev),
        feature=feature,
    )
    request = gui_geometry_fit.build_geometry_fit_solver_request(
        prepared_run=prepared_run,
        var_names=names,
        solver_inputs=context["solver_inputs"],
    )
    if request.candidate_param_names != names:
        request = gui_geometry_fit.GeometryFitSolverRequest(
            miller=request.miller,
            intensities=request.intensities,
            image_size=request.image_size,
            params=request.params,
            measured_peaks=request.measured_peaks,
            var_names=list(names),
            candidate_param_names=list(names),
            dataset_specs=request.dataset_specs,
            refinement_config={
                **dict(request.refinement_config),
                "candidate_param_names": list(names),
            },
            runtime_safety_note=request.runtime_safety_note,
        )
    return request


def _rows_from_request(request: gui_geometry_fit.GeometryFitSolverRequest) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in request.dataset_specs or ():
        if not isinstance(spec, Mapping):
            continue
        for entry in spec.get("measured_peaks", ()) or ():
            if isinstance(entry, Mapping):
                rows.append(dict(entry))
    if rows:
        return rows
    for entry in request.measured_peaks or ():
        if isinstance(entry, Mapping):
            rows.append(dict(entry))
    return rows


def _request_handoff_summary(
    request: gui_geometry_fit.GeometryFitSolverRequest,
) -> dict[str, object]:
    rows = _rows_from_request(request)
    cfg = request.refinement_config if isinstance(request.refinement_config, Mapping) else {}
    bridge_summary = (
        dict(cfg.get("optimizer_request_handoff_summary", {}))
        if isinstance(cfg.get("optimizer_request_handoff_summary", {}), Mapping)
        else {}
    )
    fallback_rows: list[dict[str, object]] = []
    trusted_rows = 0
    source_identity_rows = 0
    for index, row in enumerate(rows):
        fit_kind = str(row.get("fit_source_resolution_kind", "") or "").strip().lower()
        resolution_kind = str(row.get("resolution_kind", "") or "").strip().lower()
        fallback_used = bool(row.get("rebinding_fallback_used", False))
        request_fallback = bool(row.get("optimizer_request_fallback_row", False))
        is_fallback = bool(
            request_fallback
            or fallback_used
            or "fallback" in fit_kind
            or resolution_kind == "hkl_fallback"
        )
        if is_fallback:
            fallback_rows.append(
                {
                    "row_index": int(index),
                    "pair_id": row.get("pair_id"),
                    "hkl": row.get("hkl"),
                    "fit_source_resolution_kind": row.get("fit_source_resolution_kind"),
                    "resolution_kind": row.get("resolution_kind"),
                    "rebinding_fallback_used": fallback_used,
                    "optimizer_request_fallback_reason": row.get(
                        "optimizer_request_fallback_reason"
                    ),
                }
            )
        if bool(row.get("optimizer_request_has_fixed_source", False)) or (
            row.get("source_reflection_namespace") in {"full", "full_reflection", "miller"}
            or bool(row.get("source_reflection_is_full", False))
        ):
            trusted_rows += 1
        if any(
            row.get(key) is not None
            for key in (
                "source_reflection_index",
                "source_table_index",
                "source_row_index",
                "source_branch_index",
                "source_peak_index",
            )
        ):
            source_identity_rows += 1
    computed = {
        "handoff_row_count": int(len(rows)),
        "handoff_source_identity_row_count": int(source_identity_rows),
        "handoff_trusted_full_source_row_count": int(trusted_rows),
        "provider_pair_count": int(bridge_summary.get("provider_pair_count", 0) or 0),
        "dataset_pair_count": int(bridge_summary.get("dataset_pair_count", 0) or 0),
        "optimizer_request_pair_count": int(len(rows)),
        "fixed_source_pair_count": int(trusted_rows),
        "fallback_row_count": int(len(fallback_rows)),
        "fixed_source_resolution_fallback_count": int(len(fallback_rows)),
        "missing_fixed_source_count": max(0, int(len(rows)) - int(trusted_rows)),
        "provider_to_optimizer_identity_match": bool(
            bridge_summary.get("provider_to_optimizer_identity_match", False)
        ),
        "provider_to_optimizer_point_match": bool(
            bridge_summary.get("provider_to_optimizer_point_match", False)
        ),
        "provider_row_fallbacks": fallback_rows,
    }
    computed.update(bridge_summary)
    computed["provider_row_fallback_count"] = int(computed.get("fallback_row_count", 0) or 0)
    computed["provider_row_fallbacks"] = fallback_rows
    for key in REQUEST_HANDOFF_FIELDS:
        computed.setdefault(key, None if key.startswith("provider_to_") else 0)
    return computed


def _metric_float(value: object) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float("nan")
    return numeric if math.isfinite(numeric) else float("nan")


def _summary_rms(summary: Mapping[str, object] | None) -> float:
    if not isinstance(summary, Mapping):
        return float("nan")
    return _metric_float(summary.get("unweighted_peak_rms_px", summary.get("rms_px", np.nan)))


def _summary_max(summary: Mapping[str, object] | None) -> float:
    if not isinstance(summary, Mapping):
        return float("nan")
    return _metric_float(summary.get("unweighted_peak_max_px", summary.get("max_error_px", np.nan)))


def _point_match_summary(result: object) -> dict[str, object]:
    summary = getattr(result, "point_match_summary", None)
    return dict(summary) if isinstance(summary, Mapping) else {}


def _summary_int(summary: Mapping[str, object], key: str) -> int:
    try:
        return int(summary.get(key, 0) or 0)
    except Exception:
        return 0


def _append_unique(target: list[str], items: Sequence[object]) -> None:
    for item in items:
        text = str(item)
        if text and text not in target:
            target.append(text)


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item)
        if text and text not in result:
            result.append(text)
    return result


def _as_mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _bounds_for_request(request: gui_geometry_fit.GeometryFitSolverRequest) -> dict[str, object]:
    cfg = request.refinement_config if isinstance(request.refinement_config, Mapping) else {}
    bounds = cfg.get("bounds", {}) if isinstance(cfg, Mapping) else {}
    return dict(bounds) if isinstance(bounds, Mapping) else {}


def _param_report(
    request: gui_geometry_fit.GeometryFitSolverRequest,
    result: object,
) -> list[dict[str, object]]:
    final_x = np.asarray(getattr(result, "x", []), dtype=float).reshape(-1)
    bounds = _bounds_for_request(request)
    params: list[dict[str, object]] = []
    for idx, name in enumerate(request.var_names):
        start = _metric_float(request.params.get(str(name), np.nan))
        final = float(final_x[idx]) if idx < final_x.size and np.isfinite(final_x[idx]) else float("nan")
        bound = bounds.get(str(name))
        lower = upper = float("nan")
        in_bounds = True
        if isinstance(bound, Sequence) and not isinstance(bound, (str, bytes)) and len(bound) >= 2:
            lower = _metric_float(bound[0])
            upper = _metric_float(bound[1])
            if math.isfinite(lower) and math.isfinite(upper) and math.isfinite(final):
                in_bounds = lower - 1.0e-9 <= final <= upper + 1.0e-9
        params.append(
            {
                "name": str(name),
                "start": start,
                "final": final,
                "delta": final - start if math.isfinite(start) and math.isfinite(final) else float("nan"),
                "lower": lower,
                "upper": upper,
                "within_bounds": bool(in_bounds),
            }
        )
    return params


def _rejection_reasons(result: object, rms: float) -> list[str]:
    try:
        return gui_geometry_fit.build_geometry_fit_rejection_reason_lines(result, rms=float(rms))
    except Exception as exc:
        return [f"rejection_reason_build_failed: {exc}"]


def _apply_single_param_fields(report: dict[str, object]) -> None:
    names = _as_str_list(report.get("var_names"))
    if not names:
        names = _as_str_list(report.get("active_params"))
    candidates = _as_str_list(report.get("candidate_param_names"))
    if len(names) == 1:
        report["param_name"] = str(report.get("param_name") or names[0])
        report["var_names"] = [names[0]]
        if not candidates:
            report["candidate_param_names"] = [names[0]]
    elif names:
        report["var_names"] = list(names)
    if len(candidates) == 1:
        report["candidate_param_names"] = [candidates[0]]

    param_name = str(report.get("param_name", ""))
    deltas = report.get("parameter_deltas", [])
    entry: Mapping[str, object] | None = None
    if isinstance(deltas, Sequence) and not isinstance(deltas, (str, bytes)):
        for raw_entry in deltas:
            if not isinstance(raw_entry, Mapping):
                continue
            if str(raw_entry.get("name", "")) == param_name or entry is None:
                entry = raw_entry
            if str(raw_entry.get("name", "")) == param_name:
                break
    if entry is None:
        return

    before = _metric_float(entry.get("start", np.nan))
    after = _metric_float(entry.get("final", np.nan))
    delta = _metric_float(entry.get("delta", np.nan))
    lower = _metric_float(entry.get("lower", np.nan))
    upper = _metric_float(entry.get("upper", np.nan))
    report["parameter_before"] = before
    report["parameter_after"] = after
    report["parameter_delta"] = delta
    report["parameter_bounds"] = {"lower": lower, "upper": upper}
    report["parameter_within_bounds"] = bool(entry.get("within_bounds", True))


def _residual_norm_from_value(value: object) -> float:
    numeric = _metric_float(value)
    if math.isfinite(numeric):
        return float(numeric)
    return float("nan")


def _residual_norm_from_cost(value: object) -> float:
    cost = _metric_float(value)
    if math.isfinite(cost) and cost >= 0.0:
        return float(math.sqrt(2.0 * cost))
    return float("nan")


def _current_bounds_for_param(
    request: gui_geometry_fit.GeometryFitSolverRequest,
    param_name: str,
) -> dict[str, object] | None:
    bound = _bounds_for_request(request).get(str(param_name))
    if not isinstance(bound, Sequence) or isinstance(bound, (str, bytes)) or len(bound) < 2:
        return None
    return {
        "lower": _metric_float(bound[0]),
        "upper": _metric_float(bound[1]),
    }


def _live_fixed_source_counter_failures(summary: Mapping[str, object] | None) -> list[str]:
    if not isinstance(summary, Mapping) or not summary:
        return ["missing_point_match_summary"]
    failures: list[str] = []
    fixed_value = None
    for key in (
        "fixed_source_pair_count",
        "fixed_source_resolved_count",
        "matched_fixed_pair_count",
        "fixed_source_reflection_count",
    ):
        if key in summary:
            fixed_value = summary.get(key)
            break
    fixed_actual = _safe_int(fixed_value, default=-999999)
    if fixed_actual != EXPECTED_PROVIDER_PAIR_COUNT:
        failures.append(
            "fixed_source_resolved_count_"
            f"{fixed_actual}_expected_{EXPECTED_PROVIDER_PAIR_COUNT}"
        )
    for key, expected in LIVE_HEARTBEAT_COUNTERS.items():
        if key == "fixed_source_resolved_count":
            continue
        actual = _safe_int(summary.get(key, -999999), default=-999999)
        if actual != expected:
            failures.append(f"{key}_{actual}_expected_{expected}")
    for key in ("fallback_hkl_count", "subset_fallback_hkl_count", "missing_fixed_pair_count"):
        if key not in summary:
            continue
        actual = _safe_int(summary.get(key, -999999), default=-999999)
        if actual != 0:
            failures.append(f"{key}_{actual}_expected_0")
    return failures


def _heartbeat_fixed_source_clean(summary: Mapping[str, object] | None) -> tuple[bool, list[str]]:
    failures = _live_fixed_source_counter_failures(summary)
    return not failures, failures


def _finite_timeout_progress(report: Mapping[str, object]) -> bool:
    last_nfev = _safe_int(report.get("last_nfev", report.get("nfev")), default=0)
    last_residual = _metric_float(report.get("last_residual_norm", np.nan))
    last_rms = _metric_float(report.get("last_rms_px", np.nan))
    last_max = _metric_float(report.get("last_max_error_px", np.nan))
    return bool(
        last_nfev > 0
        and math.isfinite(last_residual)
        and math.isfinite(last_rms)
        and math.isfinite(last_max)
    )


def _diagnosis_classification(report: Mapping[str, object]) -> str | None:
    status = str(report.get("status", ""))
    heartbeat_dirty = bool(report.get("fixed_source_counters_dirty_seen", False))
    heartbeat_failures = (
        _as_str_list(report.get("fixed_source_counter_failures_seen"))
        + _as_str_list(report.get("fixed_source_counter_failures_at_last_heartbeat"))
    )
    if status == "timeout":
        if heartbeat_dirty or any(
            failure != "missing_point_match_summary" for failure in heartbeat_failures
        ):
            return "fixed_source_or_pair_integrity_lost"
        if (
            _finite_timeout_progress(report)
            and bool(report.get("fixed_source_counters_clean_at_last_heartbeat", False))
            and report.get("child_process_killed_cleanly") is True
        ):
            return "slow_needs_separate_strategy"
        return "hang_solver_pathology"
    if heartbeat_dirty or heartbeat_failures:
        return "fixed_source_or_pair_integrity_lost"
    if _one_param_integrity_failures(report):
        return "fixed_source_or_pair_integrity_lost"
    residual_finite = bool(report.get("residuals_finite", False))
    after_rms = _metric_float(report.get("after_rms_px", report.get("last_rms_px", np.nan)))
    after_max = _metric_float(
        report.get("after_max_error_px", report.get("last_max_error_px", np.nan))
    )
    if status in {"ok", "pass"} and residual_finite and math.isfinite(after_rms) and math.isfinite(after_max):
        return "usable"
    return None


def _apply_one_param_diagnostic_aliases(report: dict[str, object]) -> None:
    trace = _as_mapping_list(report.get("residual_eval_trace"))
    last_record = (
        dict(report.get("last_residual_eval"))
        if isinstance(report.get("last_residual_eval"), Mapping)
        else (dict(trace[-1]) if trace else {})
    )
    point_summary = (
        dict(report.get("last_point_match_summary"))
        if isinstance(report.get("last_point_match_summary"), Mapping)
        else (
            dict(last_record.get("point_match_summary"))
            if isinstance(last_record.get("point_match_summary"), Mapping)
            else (
                dict(report.get("point_match_summary"))
                if isinstance(report.get("point_match_summary"), Mapping)
                else {}
            )
        )
    )
    if trace:
        report["residual_eval_trace"] = trace
        report["last_residual_eval"] = last_record
    report.setdefault("heartbeat_count", int(len(trace)))
    report.setdefault("last_heartbeat_elapsed_s", last_record.get("elapsed_s"))
    report.setdefault(
        "last_nfev",
        last_record.get("nfev", report.get("nfev")),
    )
    residual_norm = _residual_norm_from_value(
        last_record.get("residual_norm", report.get("residual_norm", np.nan))
    )
    if not math.isfinite(residual_norm):
        residual_norm = _residual_norm_from_cost(
            last_record.get("cost", report.get("cost", report.get("last_cost", np.nan)))
        )
    report.setdefault("last_residual_norm", residual_norm)
    report.setdefault("last_cost", last_record.get("cost", report.get("cost")))
    report.setdefault(
        "last_rms_px",
        last_record.get("rms_px", report.get("after_rms_px", report.get("current_rms_px"))),
    )
    report.setdefault(
        "last_max_error_px",
        last_record.get("max_error_px", report.get("after_max_error_px")),
    )
    report.setdefault(
        "last_parameter_value",
        last_record.get("parameter_value", report.get("parameter_after")),
    )
    report.setdefault(
        "current_bounds",
        last_record.get("bounds", report.get("parameter_bounds")),
    )
    report.setdefault("last_point_match_summary", point_summary if point_summary else None)
    if "fixed_source_counters_clean_at_last_heartbeat" not in report:
        if isinstance(last_record.get("fixed_source_counters_clean"), bool):
            report["fixed_source_counters_clean_at_last_heartbeat"] = bool(
                last_record.get("fixed_source_counters_clean")
            )
            report["fixed_source_counter_failures_at_last_heartbeat"] = _as_str_list(
                last_record.get("fixed_source_counter_failures")
            )
        elif point_summary:
            clean, failures = _heartbeat_fixed_source_clean(point_summary)
            report["fixed_source_counters_clean_at_last_heartbeat"] = bool(clean)
            report["fixed_source_counter_failures_at_last_heartbeat"] = failures
        elif str(report.get("status", "")) != "timeout":
            failures = _one_param_integrity_failures(report)
            report["fixed_source_counters_clean_at_last_heartbeat"] = not failures
            report["fixed_source_counter_failures_at_last_heartbeat"] = failures
        else:
            report["fixed_source_counters_clean_at_last_heartbeat"] = False
            report["fixed_source_counter_failures_at_last_heartbeat"] = []
    report.setdefault("fixed_source_counters_dirty_seen", False)
    report.setdefault("fixed_source_counter_failures_seen", [])
    report.setdefault("child_process_killed_cleanly", None)
    report.setdefault("dirty_timeout_abort", False)
    report["diagnosis_classification"] = _diagnosis_classification(report)


def _result_report(
    *,
    request: gui_geometry_fit.GeometryFitSolverRequest,
    result: object,
    rung: int,
    rung_name: str,
    started_at: float,
    before_summary: Mapping[str, object] | None = None,
    status: str = "ok",
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    point_summary = _point_match_summary(result)
    residual_arr = np.asarray(getattr(result, "fun", []), dtype=float).reshape(-1)
    finite_residual = residual_arr[np.isfinite(residual_arr)]
    residual_norm = (
        float(np.linalg.norm(finite_residual)) if finite_residual.size else float("nan")
    )
    after_rms = _summary_rms(point_summary)
    if not math.isfinite(after_rms):
        after_rms = _metric_float(getattr(result, "rms_px", np.nan))
    after_max = _summary_max(point_summary)
    rejection_reasons = _rejection_reasons(result, after_rms)
    report = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": str(status),
        "active_params": [str(name) for name in request.var_names],
        "var_names": [str(name) for name in request.var_names],
        "candidate_param_names": (
            [str(name) for name in request.candidate_param_names]
            if request.candidate_param_names is not None
            else None
        ),
        "solver_config": dict(
            request.refinement_config.get("solver", {})
            if isinstance(request.refinement_config, Mapping)
            and isinstance(request.refinement_config.get("solver"), Mapping)
            else {}
        ),
        "before_rms_px": _summary_rms(before_summary),
        "before_max_error_px": _summary_max(before_summary),
        "after_rms_px": float(after_rms),
        "after_max_error_px": float(after_max),
        "residual_norm": residual_norm,
        "matched_pair_count": int(point_summary.get("matched_pair_count", 0) or 0),
        "missing_pair_count": int(point_summary.get("missing_pair_count", 0) or 0),
        "branch_mismatch_count": int(point_summary.get("branch_mismatch_count", 0) or 0),
        "nfev": getattr(result, "nfev", None),
        "elapsed_seconds": float(max(0.0, time.monotonic() - started_at)),
        "solver_success": bool(getattr(result, "success", False)),
        "solver_message": str(getattr(result, "message", "") or ""),
        "rejection_reasons": rejection_reasons,
        "rejection_reason": " ".join(str(reason) for reason in rejection_reasons),
        "point_match_summary": point_summary,
        "stage_timing_s": dict(
            getattr(result, "geometry_fit_stage_timings", {})
            if isinstance(getattr(result, "geometry_fit_stage_timings", {}), Mapping)
            else {}
        ),
        "parameter_deltas": _param_report(request, result),
    }
    report["elapsed_s"] = report["elapsed_seconds"]
    for key in STRICT_POINT_SUMMARY_KEYS:
        report[key] = _summary_int(point_summary, key)
    report.update(_request_handoff_summary(request))
    report["branch_mismatch_count"] = int(
        point_summary.get("branch_mismatch_count", 0) or 0
    )
    if isinstance(extra, Mapping):
        report.update(dict(extra))
    report["residuals_finite"] = bool(
        math.isfinite(_metric_float(report.get("before_rms_px", np.nan)))
        and math.isfinite(_metric_float(report.get("after_rms_px", np.nan)))
    )
    _apply_single_param_fields(report)
    _apply_one_param_diagnostic_aliases(report)
    fallback_failures = _strict_no_fallback_failures(report)
    if fallback_failures:
        report["fallback_guard_failures"] = fallback_failures
    report["pass"] = bool(_rung_passed(report))
    if report["status"] == "ok" and not bool(report["pass"]):
        report["status"] = "fail"
    return report


def _strict_no_fallback_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()

    def _add(item: str) -> None:
        if item not in seen:
            seen.add(item)
            failures.append(item)

    def _report_int(key: str) -> int:
        try:
            return int(report.get(key, 0) or 0)
        except Exception:
            return 0

    for key in (
        "fallback_row_count",
        "fixed_source_resolution_fallback_count",
        "missing_fixed_source_count",
        "provider_row_fallback_count",
        "fallback_entry_count",
        "fallback_hkl_count",
        "subset_fallback_hkl_count",
        "missing_fixed_pair_count",
    ):
        value = _report_int(key)
        if value != 0:
            _add(f"{key}_{value}_expected_0")
    for key in (
        "measured_count",
        "fixed_source_resolved_count",
        "fixed_source_reflection_count",
        "matched_fixed_pair_count",
    ):
        value = _report_int(key)
        if value and value != EXPECTED_PROVIDER_PAIR_COUNT:
            _add(f"{key}_{value}_expected_{EXPECTED_PROVIDER_PAIR_COUNT}")
    if _report_int("measured_count") == EXPECTED_PROVIDER_PAIR_COUNT:
        for key in (
            "fixed_source_resolved_count",
            "fixed_source_reflection_count",
            "matched_fixed_pair_count",
        ):
            value = _report_int(key)
            if value != EXPECTED_PROVIDER_PAIR_COUNT:
                _add(f"{key}_{value}_expected_{EXPECTED_PROVIDER_PAIR_COUNT}")
    optimizer_pair_count = _report_int("optimizer_request_pair_count")
    fixed_source_pair_count = _report_int("fixed_source_pair_count")
    if optimizer_pair_count:
        if fixed_source_pair_count != optimizer_pair_count:
            _add(
                "fixed_source_pair_count_"
                f"{fixed_source_pair_count}_expected_{optimizer_pair_count}"
            )
        for key in (
            "provider_to_optimizer_identity_match",
            "provider_to_optimizer_point_match",
        ):
            if report.get(key) is False:
                _add(f"{key}_false")
    return failures


def _one_param_integrity_failures(report: Mapping[str, object]) -> list[str]:
    return _fixed_source_contract_failures(
        report,
        expected_counts=ONE_PARAM_FIXED_SOURCE_COUNTS,
        required_bool_keys=PROVIDER_MATCH_BOOLS,
    )


def _fixed_source_contract_failures(
    report: Mapping[str, object],
    *,
    expected_counts: Mapping[str, int],
    required_bool_keys: Sequence[str],
    prefix: str = "",
    require_present: bool = True,
) -> list[str]:
    failures: list[str] = []
    for key, expected in expected_counts.items():
        if not require_present and key not in report:
            continue
        raw_value = report.get(key, -999999)
        actual = -999999 if isinstance(raw_value, bool) else _safe_int(raw_value, default=-999999)
        if actual != expected:
            failures.append(f"{prefix}{key}_{actual}_expected_{expected}")
    for key in required_bool_keys:
        if not require_present and key not in report:
            continue
        actual = report.get(key)
        if actual is not True:
            failures.append(f"{prefix}{key}_{actual}_expected_True")
    return failures


def _one_param_metric_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    before_rms = _metric_float(report.get("before_rms_px", np.nan))
    after_rms = _metric_float(report.get("after_rms_px", np.nan))
    before_max = _metric_float(report.get("before_max_error_px", np.nan))
    after_max = _metric_float(report.get("after_max_error_px", np.nan))
    if not math.isfinite(before_rms) or not math.isfinite(after_rms):
        failures.append("non_finite_residual")
    if not math.isfinite(before_max) or not math.isfinite(after_max):
        failures.append("non_finite_max_error")
    if math.isfinite(before_rms) and math.isfinite(after_rms) and after_rms > before_rms + 0.25:
        failures.append("rms_guard_failed")
    if math.isfinite(before_max) and math.isfinite(after_max) and after_max > before_max + 1.0:
        failures.append("max_error_guard_failed")
    if NO_MATCH_REJECTION in str(report.get("rejection_reason", "")):
        failures.append("no_matched_peak_rejection")
    if bool(report.get("parameter_within_bounds", True)) is not True:
        failures.append("parameter_out_of_bounds")
    return failures


def _rung_passed(report: Mapping[str, object]) -> bool:
    if str(report.get("status", "")) not in {"ok", "pass"}:
        return False
    if not math.isfinite(_metric_float(report.get("after_rms_px", np.nan))):
        return False
    if int(report.get("rung", 0) or 0) == 3:
        if _one_param_integrity_failures(report):
            return False
        if _one_param_metric_failures(report):
            return False
        if bool(report.get("least_squares_called", False)) is not True:
            return False
        if not (
            bool(report.get("optimizer_solve_called", False))
            or bool(report.get("real_solve_called", False))
        ):
            return False
        if bool(report.get("state_hash_unchanged", True)) is not True:
            return False
        names = _as_str_list(report.get("var_names"))
        candidates = _as_str_list(report.get("candidate_param_names"))
        if len(names) != 1 or candidates != names:
            return False
        return True
    if _strict_no_fallback_failures(report):
        return False
    if int(report.get("matched_pair_count", 0) or 0) != 7:
        return False
    if int(report.get("missing_pair_count", 0) or 0) != 0:
        return False
    if int(report.get("branch_mismatch_count", 0) or 0) != 0:
        return False
    rejection_reason = str(report.get("rejection_reason", ""))
    if "No matched peak pairs were available" in rejection_reason:
        return False
    before_rms = _metric_float(report.get("before_rms_px", np.nan))
    after_rms = _metric_float(report.get("after_rms_px", np.nan))
    if math.isfinite(before_rms) and math.isfinite(after_rms) and after_rms > before_rms + 0.25:
        return False
    before_max = _metric_float(report.get("before_max_error_px", np.nan))
    after_max = _metric_float(report.get("after_max_error_px", np.nan))
    if math.isfinite(before_max) and math.isfinite(after_max) and after_max > before_max + 1.0:
        return False
    for entry in report.get("parameter_deltas", []) or []:
        if isinstance(entry, Mapping) and not bool(entry.get("within_bounds", True)):
            return False
    if bool(report.get("least_squares_called", False)):
        return False
    if bool(report.get("optimizer_solve_called", False)):
        return False
    if "objective_dry_run_residual_finite" in report and not bool(
        report.get("objective_dry_run_residual_finite", False)
    ):
        return False
    return True


class _ProbeLeastSquares:
    def __init__(self, *, mode: str) -> None:
        self.mode = str(mode)
        self.records: list[dict[str, object]] = []

    @staticmethod
    def _residual_norm(residual: np.ndarray) -> float:
        finite = residual[np.isfinite(residual)]
        if not finite.size:
            return float("nan")
        return float(np.linalg.norm(finite))

    @staticmethod
    def _step_for_value(value: float) -> float:
        if not math.isfinite(value):
            return 1.0e-6
        return float(max(abs(float(value)) * 1.0e-4, 1.0e-6))

    @staticmethod
    def _bounds(kwargs: Mapping[str, object], size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        lower, upper = kwargs.get("bounds", (None, None))
        lower_arr = np.asarray(lower, dtype=float).reshape(-1) if lower is not None else None
        upper_arr = np.asarray(upper, dtype=float).reshape(-1) if upper is not None else None
        if lower_arr is not None and lower_arr.size != size:
            lower_arr = None
        if upper_arr is not None and upper_arr.size != size:
            upper_arr = None
        return lower_arr, upper_arr

    def _safe_eval(
        self,
        fun: Callable[[np.ndarray], np.ndarray],
        vector: np.ndarray,
        *,
        label: str,
        moved: bool = True,
    ) -> tuple[dict[str, object], np.ndarray | None]:
        try:
            residual = np.asarray(fun(np.array(vector, dtype=float)), dtype=float)
        except Exception as exc:
            return (
                {
                    "label": str(label),
                    "x": np.asarray(vector, dtype=float).tolist(),
                    "moved": bool(moved),
                    "raised": True,
                    "error_text": str(exc),
                    "residual_size": 0,
                    "residual_norm": float("nan"),
                    "finite": False,
                },
                None,
            )
        return (
            {
                "label": str(label),
                "x": np.asarray(vector, dtype=float).tolist(),
                "moved": bool(moved),
                "raised": False,
                "error_text": "",
                "residual_size": int(residual.size),
                "residual_norm": self._residual_norm(residual),
                "finite": bool(np.all(np.isfinite(residual))),
            },
            residual,
        )

    def __call__(self, fun: Callable[[np.ndarray], np.ndarray], x0, *args, **kwargs):
        del args
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if self.mode == "sensitivity":
            base_eval, residual0 = self._safe_eval(fun, x0_arr, label="base")
            if residual0 is None:
                residual0 = np.array([float("nan")], dtype=float)
        else:
            residual0 = np.asarray(fun(np.array(x0_arr, dtype=float)), dtype=float)
            base_eval = {
                "label": "base",
                "x": x0_arr.tolist(),
                "moved": True,
                "raised": False,
                "error_text": "",
                "residual_size": int(residual0.size),
                "residual_norm": self._residual_norm(residual0),
                "finite": bool(np.all(np.isfinite(residual0))),
            }
        record: dict[str, object] = {
            "x0": x0_arr.tolist(),
            "residual_size": int(base_eval.get("residual_size", 0) or 0),
            "residual_norm": _metric_float(base_eval.get("residual_norm", np.nan)),
            "finite": bool(base_eval.get("finite", False)),
        }
        if self.mode == "sensitivity" and x0_arr.size:
            lower_arr, upper_arr = self._bounds(kwargs, x0_arr.size)
            base_value = float(x0_arr[0])
            requested_step = self._step_for_value(base_value)

            def _trial(direction: float) -> tuple[np.ndarray, float, bool]:
                target = np.array(x0_arr, dtype=float)
                requested = float(direction) * requested_step
                target[0] = target[0] + requested
                clipped = False
                if lower_arr is not None:
                    clipped = clipped or bool(np.any(target < lower_arr))
                    target = np.maximum(target, lower_arr)
                if upper_arr is not None:
                    clipped = clipped or bool(np.any(target > upper_arr))
                    target = np.minimum(target, upper_arr)
                applied = float(target[0] - x0_arr[0])
                if not math.isclose(applied, requested, rel_tol=0.0, abs_tol=1.0e-15):
                    clipped = True
                return target, applied, clipped

            plus_x, plus_applied, plus_clipped = _trial(1.0)
            minus_x, minus_applied, minus_clipped = _trial(-1.0)

            evals = [dict(base_eval)]
            nfev = 1
            if abs(plus_applied) > 0.0:
                plus_eval, residual_plus = self._safe_eval(fun, plus_x, label="plus")
                if residual_plus is not None and residual_plus.shape == residual0.shape:
                    delta = residual_plus - residual0
                    plus_eval["delta_norm"] = self._residual_norm(delta)
                else:
                    plus_eval["delta_norm"] = float("nan")
                nfev += 1
            else:
                plus_eval = {
                    "label": "plus",
                    "x": plus_x.tolist(),
                    "moved": False,
                    "raised": False,
                    "error_text": "",
                    "residual_size": 0,
                    "residual_norm": float("nan"),
                    "finite": False,
                    "delta_norm": float("nan"),
                }
            plus_eval["step_applied"] = float(plus_applied)
            plus_eval["clipped"] = bool(plus_clipped)
            evals.append(plus_eval)

            if abs(minus_applied) > 0.0:
                minus_eval, residual_minus = self._safe_eval(fun, minus_x, label="minus")
                if residual_minus is not None and residual_minus.shape == residual0.shape:
                    delta = residual_minus - residual0
                    minus_eval["delta_norm"] = self._residual_norm(delta)
                else:
                    minus_eval["delta_norm"] = float("nan")
                nfev += 1
            else:
                minus_eval = {
                    "label": "minus",
                    "x": minus_x.tolist(),
                    "moved": False,
                    "raised": False,
                    "error_text": "",
                    "residual_size": 0,
                    "residual_norm": float("nan"),
                    "finite": False,
                    "delta_norm": float("nan"),
                }
            minus_eval["step_applied"] = float(minus_applied)
            minus_eval["clipped"] = bool(minus_clipped)
            evals.append(minus_eval)

            record.update(
                {
                    "requested_step": float(requested_step),
                    "base_value": float(base_value),
                    "evals": evals,
                    "plus_step_applied": float(plus_applied),
                    "minus_step_applied": float(minus_applied),
                    "plus_clipped": bool(plus_clipped),
                    "minus_clipped": bool(minus_clipped),
                    "delta_step": float(plus_applied),
                    "delta_norm": _metric_float(plus_eval.get("delta_norm", np.nan)),
                    "delta_finite": bool(plus_eval.get("finite", False)),
                }
            )
        else:
            nfev = 1
        self.records.append(record)
        return opt.OptimizeResult(
            x=x0_arr,
            fun=residual0,
            success=True,
            status=1,
            message=f"{self.mode} probe",
            nfev=nfev,
            active_mask=np.zeros(x0_arr.shape, dtype=int),
            optimality=float("nan"),
        )


def _run_with_probe_least_squares(
    request: gui_geometry_fit.GeometryFitSolverRequest,
    *,
    mode: str,
) -> tuple[object, list[dict[str, object]]]:
    probe = _ProbeLeastSquares(mode=mode)
    live_payloads: list[dict[str, object]] = []

    def _live_update(payload: Mapping[str, object]) -> None:
        live_payloads.append(dict(payload))

    original = opt.least_squares
    opt.least_squares = probe
    try:
        result = gui_geometry_fit.solve_geometry_fit_request(
            request,
            solve_fit=opt.fit_geometry_parameters,
            live_update_callback=_live_update,
        )
    finally:
        opt.least_squares = original
    records = list(probe.records)
    payload_index = 0
    for record in records:
        evals = record.get("evals")
        if isinstance(evals, list):
            for entry in evals:
                if not isinstance(entry, dict) or not bool(entry.get("moved", True)):
                    continue
                if payload_index >= len(live_payloads):
                    continue
                entry["live_update_payload"] = dict(live_payloads[payload_index])
                point_summary = live_payloads[payload_index].get("point_match_summary")
                if isinstance(point_summary, Mapping):
                    entry["point_match_summary"] = dict(point_summary)
                payload_index += 1
            continue
        if payload_index < len(live_payloads):
            record["live_update_payload"] = dict(live_payloads[payload_index])
            point_summary = live_payloads[payload_index].get("point_match_summary")
            if isinstance(point_summary, Mapping):
                record["point_match_summary"] = dict(point_summary)
            payload_index += 1
    return result, records


def _request_only_report(
    *,
    request: gui_geometry_fit.GeometryFitSolverRequest,
    rung: int,
    rung_name: str,
    started_at: float,
    status: str,
    failure_reason: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": str(status),
        "pass": False,
        "failure_reason": str(failure_reason),
        "active_params": [str(name) for name in request.var_names],
        "var_names": [str(name) for name in request.var_names],
        "candidate_param_names": (
            [str(name) for name in request.candidate_param_names]
            if request.candidate_param_names is not None
            else None
        ),
        "solver_config": dict(
            request.refinement_config.get("solver", {})
            if isinstance(request.refinement_config, Mapping)
            and isinstance(request.refinement_config.get("solver"), Mapping)
            else {}
        ),
        "before_rms_px": float("nan"),
        "before_max_error_px": float("nan"),
        "after_rms_px": float("nan"),
        "after_max_error_px": float("nan"),
        "matched_pair_count": 0,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
        "nfev": None,
        "elapsed_seconds": float(max(0.0, time.monotonic() - started_at)),
        "elapsed_s": float(max(0.0, time.monotonic() - started_at)),
        "solver_success": False,
        "solver_message": "",
        "rejection_reasons": [],
        "rejection_reason": "",
        "point_match_summary": {},
        "stage_timing_s": {},
        "parameter_deltas": [],
        "optimizer_called": False,
        "objective_eval_called": False,
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "objective_dry_run_residual_finite": False,
    }
    for key in STRICT_POINT_SUMMARY_KEYS:
        report[key] = 0
    report.update(_request_handoff_summary(request))
    if isinstance(extra, Mapping):
        report.update(dict(extra))
    _apply_single_param_fields(report)
    _apply_one_param_diagnostic_aliases(report)
    return report


def run_objective_dry_run(
    context: Mapping[str, object],
    *,
    output_path: Path,
    max_nfev: int,
) -> dict[str, object]:
    started = time.monotonic()
    request = build_solver_request(
        context,
        ["center_x"],
        max_nfev=max_nfev,
    )
    request_summary = _request_handoff_summary(request)
    request_fallback_failures = _strict_no_fallback_failures(request_summary)
    if request_fallback_failures:
        report = _request_only_report(
            request=request,
            rung=1,
            rung_name="objective_dry_run",
            started_at=started,
            status="failed",
            failure_reason="optimizer_request_used_fallback_rows",
            extra={
                "fallback_guard_failures": request_fallback_failures,
                "least_squares_probe_records": [],
                "objective_eval_called": False,
                "least_squares_called": False,
                "optimizer_solve_called": False,
                "objective_dry_run_residual_finite": False,
            },
        )
        _write_json(output_path, report)
        return report
    result, records = _run_with_probe_least_squares(request, mode="dry_run")
    residual_finite = any(
        bool(record.get("finite", False))
        and math.isfinite(_metric_float(record.get("residual_norm", np.nan)))
        for record in records
        if isinstance(record, Mapping)
    )
    report = _result_report(
        request=request,
        result=result,
        rung=1,
        rung_name="objective_dry_run",
        started_at=started,
        status="ok",
        extra={
            "least_squares_probe_records": records,
            "objective_eval_called": bool(records),
            "least_squares_called": False,
            "optimizer_solve_called": False,
            "objective_dry_run_residual_finite": bool(residual_finite),
        },
    )
    fallback_failures = _strict_no_fallback_failures(report)
    if fallback_failures:
        report["status"] = "failed"
        report["failure_reason"] = "optimizer_request_used_fallback_rows"
        report["fallback_guard_failures"] = fallback_failures
    if int(report.get("matched_pair_count", 0) or 0) != 7:
        report["status"] = "failed"
        report.setdefault("failure_reason", "pair_count_not_7")
    if int(report.get("missing_pair_count", 0) or 0) != 0:
        report["status"] = "failed"
        report.setdefault("failure_reason", "missing_pairs_present")
    if int(report.get("branch_mismatch_count", 0) or 0) != 0:
        report["status"] = "failed"
        report.setdefault("failure_reason", "branch_mismatch_present")
    if not math.isfinite(_metric_float(report.get("after_rms_px", np.nan))):
        report["status"] = "failed"
        report.setdefault("failure_reason", "non_finite_initial_residual")
    report["pass"] = str(report.get("status")) == "ok"
    _write_json(output_path, report)
    return report


def _active_theta_name(context: Mapping[str, object]) -> str | None:
    saved_var_names = [str(name) for name in context.get("saved_var_names", []) or []]
    prepared_run = context.get("prepared_run")
    fit_params = (
        dict(getattr(prepared_run, "fit_params", {}) or {})
        if prepared_run is not None
        else {}
    )
    if "theta_offset" in saved_var_names and "theta_offset" in fit_params:
        return "theta_offset"
    if "theta_initial" in saved_var_names and "theta_initial" in fit_params:
        return "theta_initial"
    if "theta_offset" in fit_params and "theta_initial" not in fit_params:
        return "theta_offset"
    if "theta_initial" in fit_params:
        return "theta_initial"
    return None


def _candidate_order(context: Mapping[str, object]) -> list[str]:
    theta_name = _active_theta_name(context)
    ordered: list[str] = []
    for name in ONE_PARAM_ORDER:
        if name == "theta_initial" and theta_name is None:
            continue
        actual = theta_name if name == "theta_initial" else name
        if actual not in ordered:
            ordered.append(actual)
    return _coerce_active_names(context, ordered)


def _safe_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        numeric = float(value)
    except Exception:
        return int(default)
    if not math.isfinite(numeric) or not numeric.is_integer():
        return int(default)
    return int(numeric)


def _rung1_green_failures(report: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if str(report.get("status", "")) != "ok" or bool(report.get("pass", False)) is not True:
        failures.append("rung_1_status_not_ok")
    expected_counts = {
        "provider_pair_count": 7,
        "dataset_pair_count": 7,
        "optimizer_request_pair_count": 7,
        "fixed_source_pair_count": 7,
        "fallback_row_count": 0,
        "fixed_source_resolution_fallback_count": 0,
        "missing_fixed_source_count": 0,
        "matched_pair_count": 7,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
    }
    for key, expected in expected_counts.items():
        actual = _safe_int(report.get(key, -999999), default=-999999)
        if actual != expected:
            failures.append(f"{key}_{actual}_expected_{expected}")
    for key in (
        "provider_to_optimizer_identity_match",
        "provider_to_optimizer_point_match",
    ):
        actual = report.get(key)
        if actual is not True:
            failures.append(f"{key}_{actual}_expected_True")
    if bool(report.get("objective_dry_run_residual_finite", False)) is not True:
        failures.append("objective_dry_run_residual_not_finite")
    if bool(report.get("least_squares_called", True)):
        failures.append("least_squares_called")
    if bool(report.get("optimizer_solve_called", True)):
        failures.append("optimizer_solve_called")
    return failures


def _eval_int(summary: Mapping[str, object], keys: Sequence[str], default: int = -1) -> int:
    for key in keys:
        if key not in summary:
            continue
        return _safe_int(summary.get(key, default), default=default)
    return int(default)


def _sensitivity_eval_summary(
    eval_record: Mapping[str, object] | None,
    *,
    request_summary: Mapping[str, object],
) -> dict[str, object]:
    del request_summary
    record = dict(eval_record or {})
    point_summary = (
        dict(record.get("point_match_summary", {}))
        if isinstance(record.get("point_match_summary", {}), Mapping)
        else {}
    )
    moved = bool(record.get("moved", True))
    if not moved:
        point_summary = {}
        counter_source = "not_evaluated"
    elif point_summary:
        counter_source = "point_match_summary"
    else:
        counter_source = "missing"
    summary = {
        "label": str(record.get("label", "")),
        "moved": moved,
        "raised": bool(record.get("raised", False)),
        "error_text": str(record.get("error_text", "") or ""),
        "residual_norm": _metric_float(record.get("residual_norm", np.nan)),
        "finite": bool(record.get("finite", False)),
        "delta_norm": _metric_float(record.get("delta_norm", np.nan)),
        "step_applied": _metric_float(record.get("step_applied", 0.0)),
        "clipped": bool(record.get("clipped", False)),
        "fixed_source_pair_count": _eval_int(
            point_summary,
            (
                "fixed_source_resolved_count",
                "matched_fixed_pair_count",
                "fixed_source_reflection_count",
            ),
            default=-1,
        ),
        "fallback_entry_count": _eval_int(
            point_summary,
            ("fallback_entry_count",),
            default=-1,
        ),
        "matched_pair_count": _eval_int(
            point_summary,
            ("matched_pair_count",),
            default=-1,
        ),
        "missing_pair_count": _eval_int(
            point_summary,
            ("missing_pair_count",),
            default=-1,
        ),
        "branch_mismatch_count": _eval_int(
            point_summary,
            ("branch_mismatch_count",),
            default=-1,
        ),
        "counter_source": counter_source,
    }
    failures: list[str] = []
    if moved and not point_summary:
        failures.append("missing_point_match_summary")
    dirty_counter = False
    if point_summary:
        for key, expected in (
            ("fixed_source_pair_count", 7),
            ("fallback_entry_count", 0),
            ("matched_pair_count", 7),
            ("missing_pair_count", 0),
            ("branch_mismatch_count", 0),
        ):
            actual = _safe_int(summary.get(key, -999999), default=-999999)
            if actual != expected:
                failures.append(f"{key}_{actual}_expected_{expected}")
            if actual < 0:
                dirty_counter = True
    if dirty_counter and counter_source == "point_match_summary":
        counter_source = "missing"
        summary["counter_source"] = counter_source
    summary["fixed_source_clean"] = not failures
    summary["fixed_source_failures"] = failures
    return summary


def _eval_by_label(record: Mapping[str, object], label: str) -> Mapping[str, object] | None:
    evals = record.get("evals")
    if not isinstance(evals, Sequence) or isinstance(evals, (str, bytes)):
        if label == "base":
            return record
        return None
    for entry in evals:
        if isinstance(entry, Mapping) and str(entry.get("label", "")) == label:
            return entry
    return None


def _sensitivity_status(
    *,
    base_eval: Mapping[str, object],
    plus_eval: Mapping[str, object],
    minus_eval: Mapping[str, object],
    threshold: float,
) -> tuple[str, list[str], float]:
    reasons: list[str] = []
    if bool(base_eval.get("raised", False)):
        return "unsafe", ["base_eval_raised"], float("nan")
    if not bool(base_eval.get("fixed_source_clean", False)):
        return "unsafe", list(base_eval.get("fixed_source_failures", []) or []), float("nan")
    moved = [
        eval_report
        for eval_report in (plus_eval, minus_eval)
        if bool(eval_report.get("moved", False))
    ]
    if not moved:
        return "unsafe", ["no_valid_movement"], float("nan")
    for eval_report in moved:
        label = str(eval_report.get("label", "direction"))
        if bool(eval_report.get("raised", False)):
            reasons.append(f"{label}_eval_raised")
        if not bool(eval_report.get("fixed_source_clean", False)):
            reasons.extend(
                f"{label}_{reason}"
                for reason in eval_report.get("fixed_source_failures", []) or []
            )
    if reasons:
        return "unsafe", reasons, float("nan")
    if any(not bool(eval_report.get("finite", False)) for eval_report in moved):
        return "non_finite", ["non_finite_residual"], float("nan")
    delta_values = [
        _metric_float(eval_report.get("delta_norm", np.nan))
        for eval_report in moved
        if math.isfinite(_metric_float(eval_report.get("delta_norm", np.nan)))
    ]
    if not delta_values:
        return "non_finite", ["non_finite_delta"], float("nan")
    sensitivity_values: list[float] = []
    for eval_report in moved:
        delta_norm = _metric_float(eval_report.get("delta_norm", np.nan))
        step = abs(_metric_float(eval_report.get("step_applied", np.nan)))
        if math.isfinite(delta_norm) and math.isfinite(step) and step > 0.0:
            sensitivity_values.append(float(delta_norm / step))
    sensitivity_norm = max(sensitivity_values) if sensitivity_values else float("nan")
    if any(delta > threshold for delta in delta_values):
        return "active", [], sensitivity_norm
    return "near_zero", [], sensitivity_norm


def run_sensitivity_scan(
    context: Mapping[str, object],
    *,
    output_path: Path,
    max_nfev: int,
    rung_1_report: Mapping[str, object] | None = None,
    state_path: Path | None = None,
    state_hash_before: str | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    initial_state_hash = (
        str(state_hash_before)
        if state_hash_before is not None
        else (_state_sha256(Path(state_path)) if state_path is not None else None)
    )
    entries: list[dict[str, object]] = []
    rung_1 = dict(rung_1_report or {})
    rung1_failures = _rung1_green_failures(rung_1)
    if rung1_failures:
        state_hash_after = (
            _state_sha256(Path(state_path)) if state_path is not None else initial_state_hash
        )
        state_hash_unchanged = (
            bool(initial_state_hash == state_hash_after)
            if initial_state_hash is not None and state_hash_after is not None
            else True
        )
        payload = {
            "rung": 2,
            "rung_name": "sensitivity_scan",
            "status": "aborted",
            "pass": False,
            "reason": "rung_1_not_green",
            "rung_1_status": str(rung_1.get("status", "")),
            "rung_1_failures": list(rung1_failures),
            "provider_pair_count": _safe_int(rung_1.get("provider_pair_count"), default=0),
            "dataset_pair_count": _safe_int(rung_1.get("dataset_pair_count"), default=0),
            "optimizer_request_pair_count": _safe_int(
                rung_1.get("optimizer_request_pair_count"),
                default=0,
            ),
            "fixed_source_pair_count": _safe_int(
                rung_1.get("fixed_source_pair_count"),
                default=0,
            ),
            "fallback_row_count": _safe_int(rung_1.get("fallback_row_count"), default=0),
            "fixed_source_resolution_fallback_count": _safe_int(
                rung_1.get("fixed_source_resolution_fallback_count"),
                default=0,
            ),
            "missing_fixed_source_count": _safe_int(
                rung_1.get("missing_fixed_source_count"),
                default=0,
            ),
            "provider_to_optimizer_identity_match": (
                rung_1.get("provider_to_optimizer_identity_match") is True
            ),
            "provider_to_optimizer_point_match": (
                rung_1.get("provider_to_optimizer_point_match") is True
            ),
            "fallback_entry_count": _safe_int(
                rung_1.get("fallback_entry_count"),
                default=0,
            ),
            "residual_probe_called": False,
            "least_squares_called": False,
            "optimizer_solve_called": False,
            "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
            "params": [],
            "parameters": [],
            "active_param_count": 0,
            "near_zero_param_count": 0,
            "non_finite_param_count": 0,
            "unsafe_param_count": 0,
            "active_params": [],
            "near_zero_params": [],
            "non_finite_params": [],
            "unsafe_params": [],
            "active_parameters": [],
            "near_zero_parameters": [],
            "unsafe_parameters": [],
            "state_sha256_before": initial_state_hash,
            "state_sha256_after": state_hash_after,
            "state_hash_unchanged": bool(state_hash_unchanged),
        }
        _write_json(output_path, payload)
        return payload
    residual_probe_called = False
    for name in _candidate_order(context):
        param_started = time.monotonic()
        try:
            request = build_solver_request(context, [name], max_nfev=max_nfev)
            request_summary = _request_handoff_summary(request)
            request_failures = _strict_no_fallback_failures(request_summary)
            if request_failures:
                entries.append(
                    {
                        "param_name": str(name),
                        "name": str(name),
                        "status": "unsafe",
                        "unsafe_reasons": list(request_failures),
                        "elapsed_seconds": float(max(0.0, time.monotonic() - param_started)),
                    }
                )
                continue
            residual_probe_called = True
            result, records = _run_with_probe_least_squares(request, mode="sensitivity")
            probe_record = records[-1] if records else {}
            base_eval = _sensitivity_eval_summary(
                _eval_by_label(probe_record, "base"),
                request_summary=request_summary,
            )
            plus_eval = _sensitivity_eval_summary(
                _eval_by_label(probe_record, "plus"),
                request_summary=request_summary,
            )
            minus_eval = _sensitivity_eval_summary(
                _eval_by_label(probe_record, "minus"),
                request_summary=request_summary,
            )
            residual_norm_base = _metric_float(base_eval.get("residual_norm", np.nan))
            threshold = max(
                1.0e-7,
                1.0e-7 * abs(residual_norm_base)
                if math.isfinite(residual_norm_base)
                else 0.0,
            )
            status, unsafe_reasons, sensitivity_norm = _sensitivity_status(
                base_eval=base_eval,
                plus_eval=plus_eval,
                minus_eval=minus_eval,
                threshold=float(threshold),
            )
            plus_step = _metric_float(plus_eval.get("step_applied", 0.0))
            minus_step = _metric_float(minus_eval.get("step_applied", 0.0))
            entries.append(
                {
                    "param_name": str(name),
                    "name": str(name),
                    "status": str(status),
                    "classification": str(status),
                    "base_value": _metric_float(probe_record.get("base_value", np.nan)),
                    "step_size": _metric_float(probe_record.get("requested_step", np.nan)),
                    "plus_step_applied": plus_step,
                    "minus_step_applied": minus_step,
                    "plus_clipped": bool(plus_eval.get("clipped", False)),
                    "minus_clipped": bool(minus_eval.get("clipped", False)),
                    "residual_norm_base": residual_norm_base,
                    "residual_norm_plus": _metric_float(plus_eval.get("residual_norm", np.nan)),
                    "residual_norm_minus": _metric_float(minus_eval.get("residual_norm", np.nan)),
                    "delta_norm_plus": _metric_float(plus_eval.get("delta_norm", np.nan)),
                    "delta_norm_minus": _metric_float(minus_eval.get("delta_norm", np.nan)),
                    "sensitivity_norm": sensitivity_norm,
                    "finite_plus": bool(plus_eval.get("finite", False)),
                    "finite_minus": bool(minus_eval.get("finite", False)),
                    "unsafe_reasons": list(unsafe_reasons),
                    "base_eval": base_eval,
                    "plus_eval": plus_eval,
                    "minus_eval": minus_eval,
                    "provider_pair_count": _safe_int(
                        request_summary.get("provider_pair_count"),
                        default=-1,
                    ),
                    "dataset_pair_count": _safe_int(
                        request_summary.get("dataset_pair_count"),
                        default=-1,
                    ),
                    "optimizer_request_pair_count": _safe_int(
                        request_summary.get("optimizer_request_pair_count"),
                        default=-1,
                    ),
                    "fixed_source_pair_count": int(base_eval.get("fixed_source_pair_count", -1)),
                    "fallback_row_count": _safe_int(
                        request_summary.get("fallback_row_count"),
                        default=-1,
                    ),
                    "fixed_source_resolution_fallback_count": _safe_int(
                        request_summary.get("fixed_source_resolution_fallback_count"),
                        default=-1,
                    ),
                    "missing_fixed_source_count": _safe_int(
                        request_summary.get("missing_fixed_source_count"),
                        default=-1,
                    ),
                    "fixed_source_resolved_count": int(
                        base_eval.get("fixed_source_pair_count", -1)
                    ),
                    "fallback_entry_count": int(base_eval.get("fallback_entry_count", -1)),
                    "matched_pair_count": int(base_eval.get("matched_pair_count", -1)),
                    "missing_pair_count": int(base_eval.get("missing_pair_count", -1)),
                    "branch_mismatch_count": int(base_eval.get("branch_mismatch_count", -1)),
                    "provider_to_optimizer_identity_match": (
                        request_summary.get("provider_to_optimizer_identity_match") is True
                    ),
                    "provider_to_optimizer_point_match": (
                        request_summary.get("provider_to_optimizer_point_match") is True
                    ),
                    "probe_records": records,
                    "elapsed_seconds": float(max(0.0, time.monotonic() - param_started)),
                }
            )
        except Exception as exc:
            entries.append(
                {
                    "param_name": str(name),
                    "name": str(name),
                    "status": "unsafe",
                    "classification": "unsafe",
                    "error_text": str(exc),
                    "unsafe_reasons": ["residual_eval_raised"],
                    "elapsed_seconds": float(max(0.0, time.monotonic() - param_started)),
                }
            )
    state_hash_after = _state_sha256(Path(state_path)) if state_path is not None else initial_state_hash
    state_hash_unchanged = (
        bool(initial_state_hash == state_hash_after)
        if initial_state_hash is not None and state_hash_after is not None
        else True
    )
    active_params = [str(entry["param_name"]) for entry in entries if entry.get("status") == "active"]
    near_zero_params = [
        str(entry["param_name"]) for entry in entries if entry.get("status") == "near_zero"
    ]
    non_finite_params = [
        str(entry["param_name"]) for entry in entries if entry.get("status") == "non_finite"
    ]
    unsafe_params = [
        str(entry["param_name"]) for entry in entries if entry.get("status") == "unsafe"
    ]
    status = "ok" if active_params and state_hash_unchanged else "fail"
    payload = {
        "rung": 2,
        "rung_name": "sensitivity_scan",
        "status": status,
        "rung_1_status": str(rung_1.get("status", "")),
        "provider_pair_count": _safe_int(rung_1.get("provider_pair_count"), default=0),
        "dataset_pair_count": _safe_int(rung_1.get("dataset_pair_count"), default=0),
        "optimizer_request_pair_count": _safe_int(
            rung_1.get("optimizer_request_pair_count"),
            default=0,
        ),
        "fixed_source_pair_count": _safe_int(
            rung_1.get("fixed_source_pair_count"),
            default=0,
        ),
        "fallback_row_count": _safe_int(rung_1.get("fallback_row_count"), default=0),
        "fixed_source_resolution_fallback_count": _safe_int(
            rung_1.get("fixed_source_resolution_fallback_count"),
            default=0,
        ),
        "missing_fixed_source_count": _safe_int(
            rung_1.get("missing_fixed_source_count"),
            default=0,
        ),
        "fixed_source_resolved_count": _safe_int(
            rung_1.get("fixed_source_resolved_count", rung_1.get("fixed_source_pair_count")),
            default=0,
        ),
        "fallback_entry_count": _safe_int(rung_1.get("fallback_entry_count"), default=0),
        "matched_pair_count": _safe_int(rung_1.get("matched_pair_count"), default=0),
        "missing_pair_count": _safe_int(rung_1.get("missing_pair_count"), default=0),
        "branch_mismatch_count": _safe_int(rung_1.get("branch_mismatch_count"), default=0),
        "provider_to_optimizer_identity_match": (
            rung_1.get("provider_to_optimizer_identity_match") is True
        ),
        "provider_to_optimizer_point_match": (
            rung_1.get("provider_to_optimizer_point_match") is True
        ),
        "residual_probe_called": bool(residual_probe_called),
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
        "params": entries,
        "parameters": entries,
        "active_param_count": int(len(active_params)),
        "near_zero_param_count": int(len(near_zero_params)),
        "non_finite_param_count": int(len(non_finite_params)),
        "unsafe_param_count": int(len(unsafe_params)),
        "active_params": active_params,
        "near_zero_params": near_zero_params,
        "non_finite_params": non_finite_params,
        "unsafe_params": unsafe_params,
        "active_parameters": active_params,
        "near_zero_parameters": near_zero_params,
        "unsafe_parameters": unsafe_params,
        "state_sha256_before": initial_state_hash,
        "state_sha256_after": state_hash_after,
        "state_hash_unchanged": bool(state_hash_unchanged),
    }
    payload["pass"] = str(payload["status"]) == "ok"
    _write_json(output_path, payload)
    return payload


def _heartbeat_write(path: Path, payload: Mapping[str, object]) -> None:
    if not path:
        return
    merged = _read_json(path)
    merged.update(dict(payload))
    merged["heartbeat_updated_at"] = time.time()
    _write_json(path, merged)


def _worker_solve_once(
    *,
    state_path: Path,
    background_index: int,
    active_names: Sequence[str],
    output_path: Path,
    heartbeat_path: Path,
    max_nfev: int,
    rung: int,
    rung_name: str,
    feature: str | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    state_hash_before = _state_sha256(Path(state_path))
    _heartbeat_write(
        heartbeat_path,
        {
            "status": "starting",
            "rung": int(rung),
            "rung_name": str(rung_name),
            "active_params": [str(name) for name in active_names],
            "state_sha256_before": state_hash_before,
        },
    )
    context = _capture_solver_context(Path(state_path), int(background_index))
    request = build_solver_request(
        context,
        active_names,
        max_nfev=int(max_nfev),
        feature=feature,
    )
    request_summary = _request_handoff_summary(request)
    _heartbeat_write(heartbeat_path, request_summary)
    request_fallback_failures = _strict_no_fallback_failures(request_summary)
    if request_fallback_failures:
        report = _request_only_report(
            request=request,
            rung=int(rung),
            rung_name=str(rung_name),
            started_at=started,
            status="failed",
            failure_reason="optimizer_request_used_fallback_rows",
            extra={
                "feature": feature,
                "fallback_guard_failures": request_fallback_failures,
            },
        )
        _write_json(output_path, report)
        _heartbeat_write(heartbeat_path, {"status": str(report.get("status", "")), "done": True})
        return report
    before_result, _records = _run_with_probe_least_squares(request, mode="dry_run")
    before_summary = _point_match_summary(before_result)
    param_name = str(active_names[0]) if len(active_names) == 1 else ""
    current_bounds = _current_bounds_for_param(request, param_name) if param_name else None
    heartbeat_count = 0
    residual_eval_trace: list[dict[str, object]] = []
    fixed_source_counters_dirty_seen = False
    fixed_source_counter_failures_seen: list[str] = []

    def _status_callback(text: str) -> None:
        payload: dict[str, object] = {
            "status": "running",
            "last_optimizer_status": str(text),
        }
        if "eval=" in str(text):
            try:
                payload["nfev"] = int(str(text).split("eval=", 1)[1].split()[0])
            except Exception:
                pass
        _heartbeat_write(heartbeat_path, payload)

    def _live_update(payload: Mapping[str, object]) -> None:
        nonlocal heartbeat_count, fixed_source_counters_dirty_seen
        point_summary = (
            dict(payload.get("point_match_summary", {}))
            if isinstance(payload.get("point_match_summary", {}), Mapping)
            else {}
        )
        heartbeat_count += 1
        eval_count = _safe_int(
            payload.get("evaluation_count", payload.get("nfev", heartbeat_count)),
            default=heartbeat_count,
        )
        cost = _metric_float(payload.get("current_cost", payload.get("cost", np.nan)))
        residual_norm = _residual_norm_from_value(payload.get("residual_norm", np.nan))
        if not math.isfinite(residual_norm):
            residual_norm = _residual_norm_from_cost(cost)
        rms_px = _summary_rms(point_summary)
        if not math.isfinite(rms_px):
            rms_px = _metric_float(payload.get("weighted_rms_px", np.nan))
        max_error_px = _summary_max(point_summary)
        params = payload.get("params", {})
        parameter_value = None
        if param_name and isinstance(params, Mapping) and param_name in params:
            parameter_value = params.get(param_name)
        elif isinstance(payload.get("x_trial"), Sequence) and not isinstance(
            payload.get("x_trial"),
            (str, bytes),
        ):
            try:
                parameter_value = list(payload.get("x_trial", []))[0]
            except Exception:
                parameter_value = None
        clean, failures = _heartbeat_fixed_source_clean(point_summary)
        if not clean:
            fixed_source_counters_dirty_seen = True
            _append_unique(fixed_source_counter_failures_seen, failures)
        trace_entry = {
            "eval_count": int(eval_count),
            "nfev": int(eval_count),
            "elapsed_s": float(max(0.0, time.monotonic() - started)),
            "residual_norm": residual_norm,
            "cost": cost,
            "rms_px": rms_px,
            "max_error_px": max_error_px,
            "parameter_name": param_name or None,
            "parameter_value": parameter_value,
            "bounds": current_bounds,
            "point_match_summary": point_summary,
            "fixed_source_counters_clean": bool(clean),
            "fixed_source_counter_failures": failures,
        }
        residual_eval_trace.append(trace_entry)
        _heartbeat_write(
            heartbeat_path,
            {
                "status": "running",
                "last_point_match_summary": point_summary,
                "current_rms_px": rms_px,
                "current_max_error_px": max_error_px,
                "last_residual_eval": trace_entry,
                "residual_eval_trace": residual_eval_trace,
                "heartbeat_count": int(heartbeat_count),
                "last_heartbeat_elapsed_s": trace_entry["elapsed_s"],
                "last_nfev": int(eval_count),
                "nfev": int(eval_count),
                "last_residual_norm": residual_norm,
                "last_cost": cost,
                "last_rms_px": rms_px,
                "last_max_error_px": max_error_px,
                "last_parameter_value": parameter_value,
                "current_bounds": current_bounds,
                "fixed_source_counters_clean_at_last_heartbeat": bool(clean),
                "fixed_source_counter_failures_at_last_heartbeat": failures,
                "fixed_source_counters_dirty_seen": bool(fixed_source_counters_dirty_seen),
                "fixed_source_counter_failures_seen": list(
                    fixed_source_counter_failures_seen
                ),
            },
        )

    least_squares_called = False
    optimizer_solve_called = False
    try:
        original_least_squares = opt.least_squares

        def _counted_least_squares(*args, **kwargs):
            nonlocal least_squares_called
            least_squares_called = True
            _heartbeat_write(heartbeat_path, {"least_squares_called": True})
            return original_least_squares(*args, **kwargs)

        opt.least_squares = _counted_least_squares
        try:
            optimizer_solve_called = True
            _heartbeat_write(heartbeat_path, {"optimizer_solve_called": True})
            result = gui_geometry_fit.solve_geometry_fit_request(
                request,
                solve_fit=opt.fit_geometry_parameters,
                status_callback=_status_callback,
                live_update_callback=_live_update,
            )
        finally:
            opt.least_squares = original_least_squares
        state_hash_after = _state_sha256(Path(state_path))
        report = _result_report(
            request=request,
            result=result,
            rung=int(rung),
            rung_name=str(rung_name),
            started_at=started,
            before_summary=before_summary,
            status="ok",
            extra={
                "feature": feature,
                "least_squares_called": bool(least_squares_called),
                "optimizer_solve_called": bool(optimizer_solve_called),
                "real_solve_called": bool(optimizer_solve_called),
                "state_sha256_before": state_hash_before,
                "state_sha256_after": state_hash_after,
                "state_hash_unchanged": state_hash_before == state_hash_after,
                "heartbeat_count": int(heartbeat_count),
                "residual_eval_trace": list(residual_eval_trace),
                "fixed_source_counters_dirty_seen": bool(
                    fixed_source_counters_dirty_seen
                ),
                "fixed_source_counter_failures_seen": list(
                    fixed_source_counter_failures_seen
                ),
                "dirty_timeout_abort": False,
                "child_process_killed_cleanly": None,
            },
        )
    except Exception as exc:
        state_hash_after = _state_sha256(Path(state_path))
        report = {
            "rung": int(rung),
            "rung_name": str(rung_name),
            "status": "error",
            "pass": False,
            "active_params": [str(name) for name in active_names],
            "var_names": [str(name) for name in active_names],
            "candidate_param_names": [str(name) for name in active_names],
            "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
            "elapsed_s": float(max(0.0, time.monotonic() - started)),
            "error_text": str(exc),
            "feature": feature,
            "least_squares_called": bool(least_squares_called),
            "optimizer_solve_called": bool(optimizer_solve_called),
            "real_solve_called": bool(optimizer_solve_called),
            "state_sha256_before": state_hash_before,
            "state_sha256_after": state_hash_after,
            "state_hash_unchanged": state_hash_before == state_hash_after,
            "heartbeat_count": int(heartbeat_count),
            "residual_eval_trace": list(residual_eval_trace),
            "fixed_source_counters_dirty_seen": bool(fixed_source_counters_dirty_seen),
            "fixed_source_counter_failures_seen": list(
                fixed_source_counter_failures_seen
            ),
            "dirty_timeout_abort": False,
            "child_process_killed_cleanly": None,
        }
        report.update(request_summary)
        _apply_single_param_fields(report)
        _apply_one_param_diagnostic_aliases(report)
    _write_json(output_path, report)
    _heartbeat_write(heartbeat_path, {"status": str(report.get("status", "")), "done": True})
    return report


def _solver_worker_command(
    *,
    state_path: Path,
    background_index: int,
    active_names: Sequence[str],
    output_path: Path,
    heartbeat_path: Path,
    max_nfev: int,
    rung: int,
    rung_name: str,
    feature: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-solve",
        "--state",
        str(state_path),
        "--background-index",
        str(int(background_index)),
        "--output-json",
        str(output_path),
        "--heartbeat-json",
        str(heartbeat_path),
        "--active-params-json",
        json.dumps([str(name) for name in active_names]),
        "--max-nfev",
        str(int(max_nfev)),
        "--rung-number",
        str(int(rung)),
        "--rung-name",
        str(rung_name),
    ]
    if feature:
        cmd.extend(["--feature", str(feature)])
    return cmd


def _timeout_report(
    *,
    rung: int,
    rung_name: str,
    active_names: Sequence[str],
    started_at: float,
    heartbeat_path: Path,
    timeout_seconds: float,
    state_path: Path | None = None,
    state_hash_before: str | None = None,
    dirty_timeout_abort: bool = False,
    child_process_killed_cleanly: bool | None = None,
    feature: str | None = None,
) -> dict[str, object]:
    heartbeat = _read_json(heartbeat_path)
    hash_before = str(state_hash_before or heartbeat.get("state_sha256_before") or "")
    hash_after = _state_sha256(Path(state_path)) if state_path is not None else ""
    point_summary = (
        dict(heartbeat.get("last_point_match_summary", {}))
        if isinstance(heartbeat.get("last_point_match_summary", {}), Mapping)
        else {}
    )
    trace = _as_mapping_list(heartbeat.get("residual_eval_trace"))
    last_record = (
        dict(heartbeat.get("last_residual_eval"))
        if isinstance(heartbeat.get("last_residual_eval"), Mapping)
        else (dict(trace[-1]) if trace else {})
    )

    def _partial_value(key: str, *aliases: str) -> object:
        for candidate in (key, *aliases):
            if candidate in last_record and last_record.get(candidate) is not None:
                return last_record.get(candidate)
            if candidate in heartbeat and heartbeat.get(candidate) is not None:
                return heartbeat.get(candidate)
            if candidate in point_summary and point_summary.get(candidate) is not None:
                return point_summary.get(candidate)
        return None

    after_rms = _partial_value("after_rms_px", "rms_px", "current_rms_px")
    before_rms = _partial_value("before_rms_px")
    before_max = _partial_value("before_max_error_px")
    after_max = _partial_value("after_max_error_px", "max_error_px", "current_max_error_px")
    last_residual_norm = _residual_norm_from_value(
        _partial_value("last_residual_norm", "residual_norm")
    )
    if not math.isfinite(last_residual_norm):
        last_residual_norm = _residual_norm_from_cost(_partial_value("last_cost", "cost"))
    report = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": "timeout",
        "pass": False,
        "failure_reason": "timeout",
        "active_params": [str(name) for name in active_names],
        "var_names": [str(name) for name in active_names],
        "candidate_param_names": [str(name) for name in active_names],
        "before_rms_px": before_rms,
        "after_rms_px": after_rms,
        "before_max_error_px": before_max,
        "after_max_error_px": after_max,
        "parameter_before": _partial_value("parameter_before"),
        "parameter_after": _partial_value("parameter_after"),
        "parameter_delta": _partial_value("parameter_delta"),
        "parameter_bounds": _partial_value("parameter_bounds"),
        "residuals_finite": bool(
            math.isfinite(_metric_float(before_rms))
            and math.isfinite(_metric_float(after_rms))
        ),
        "elapsed_seconds": float(max(0.0, time.monotonic() - started_at)),
        "elapsed_s": float(max(0.0, time.monotonic() - started_at)),
        "timeout_seconds": float(timeout_seconds),
        "timeout_s": float(timeout_seconds),
        "last_heartbeat_elapsed_s": _partial_value("last_heartbeat_elapsed_s", "elapsed_s"),
        "heartbeat_count": _safe_int(heartbeat.get("heartbeat_count", len(trace)), default=0),
        "last_emitted_optimizer_status": heartbeat.get("last_optimizer_status"),
        "nfev": _partial_value("nfev", "last_nfev", "eval_count"),
        "last_nfev": _partial_value("last_nfev", "nfev", "eval_count"),
        "last_residual_norm": last_residual_norm,
        "last_cost": _partial_value("last_cost", "cost"),
        "last_rms_px": after_rms,
        "last_max_error_px": after_max,
        "last_parameter_value": _partial_value("last_parameter_value", "parameter_value"),
        "current_bounds": _partial_value("current_bounds", "bounds"),
        "current_rms_px": heartbeat.get("current_rms_px"),
        "last_point_match_summary": point_summary if point_summary else None,
        "point_match_summary": point_summary,
        "last_residual_eval": last_record if last_record else None,
        "residual_eval_trace": trace,
        "least_squares_called": bool(heartbeat.get("least_squares_called", False)),
        "optimizer_solve_called": bool(heartbeat.get("optimizer_solve_called", False)),
        "real_solve_called": bool(
            heartbeat.get("real_solve_called", heartbeat.get("optimizer_solve_called", False))
        ),
        "dirty_timeout_abort": bool(dirty_timeout_abort),
        "child_process_killed_cleanly": child_process_killed_cleanly,
        "fixed_source_counters_clean_at_last_heartbeat": bool(
            heartbeat.get("fixed_source_counters_clean_at_last_heartbeat", False)
        ),
        "fixed_source_counter_failures_at_last_heartbeat": _as_str_list(
            heartbeat.get("fixed_source_counter_failures_at_last_heartbeat")
        ),
        "fixed_source_counters_dirty_seen": bool(
            heartbeat.get("fixed_source_counters_dirty_seen", False)
        ),
        "fixed_source_counter_failures_seen": _as_str_list(
            heartbeat.get("fixed_source_counter_failures_seen")
        ),
        "state_sha256_before": hash_before or None,
        "state_sha256_after": hash_after or None,
        "state_hash_unchanged": bool(hash_before and hash_after and hash_before == hash_after),
        "feature": feature,
    }
    counter_aliases = {
        "fixed_source_pair_count": (
            "fixed_source_pair_count",
            "matched_fixed_pair_count",
            "fixed_source_resolved_count",
        ),
    }
    for key in RUNG2_FIXED_SOURCE_COUNTS:
        aliases = counter_aliases.get(key, (key,))
        report[key] = _partial_value(key, *tuple(alias for alias in aliases if alias != key))
    for key in PROVIDER_MATCH_BOOLS:
        report[key] = _partial_value(key)
    _apply_single_param_fields(report)
    _apply_one_param_diagnostic_aliases(report)
    report.setdefault("parameter_bounds", None)
    return report


def _run_threaded_worker_for_tests(
    *,
    worker: Callable[[], dict[str, object] | None],
    output_path: Path,
    timeout_report_factory: Callable[[], dict[str, object]],
    timeout_seconds: float,
) -> dict[str, object]:
    result_box: dict[str, object] = {}

    def _target() -> None:
        try:
            result = worker()
            if isinstance(result, Mapping):
                result_box.update(dict(result))
                if not output_path.exists():
                    _write_json(output_path, dict(result))
        except Exception as exc:
            result_box.update({"status": "error", "pass": False, "error_text": str(exc)})

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=float(timeout_seconds))
    if thread.is_alive():
        report = timeout_report_factory()
        _write_json(output_path, report)
        return report
    if output_path.exists():
        return _read_json(output_path)
    report = dict(result_box) if result_box else {"status": "error", "pass": False}
    _write_json(output_path, report)
    return report


def _run_solver_rung_with_timeout(
    *,
    state_path: Path,
    background_index: int,
    active_names: Sequence[str],
    output_path: Path,
    max_nfev: int,
    timeout_seconds: float,
    rung: int,
    rung_name: str,
    feature: str | None = None,
    state_hash_before: str | None = None,
    use_subprocess: bool = True,
) -> dict[str, object]:
    started = time.monotonic()
    heartbeat_path = output_path.with_suffix(".heartbeat.json")
    initial_state_hash = str(state_hash_before or _state_sha256(Path(state_path)))

    def _make_timeout_report(
        *,
        dirty_timeout_abort: bool = False,
        child_process_killed_cleanly: bool | None = None,
    ) -> dict[str, object]:
        return _timeout_report(
            rung=int(rung),
            rung_name=str(rung_name),
            active_names=active_names,
            started_at=started,
            heartbeat_path=heartbeat_path,
            timeout_seconds=float(timeout_seconds),
            state_path=Path(state_path),
            state_hash_before=initial_state_hash,
            dirty_timeout_abort=bool(dirty_timeout_abort),
            child_process_killed_cleanly=child_process_killed_cleanly,
            feature=feature,
        )

    if not use_subprocess:
        return _run_threaded_worker_for_tests(
            worker=lambda: _worker_solve_once(
                state_path=state_path,
                background_index=int(background_index),
                active_names=active_names,
                output_path=output_path,
                heartbeat_path=heartbeat_path,
                max_nfev=int(max_nfev),
                rung=int(rung),
                rung_name=str(rung_name),
                feature=feature,
            ),
            output_path=output_path,
            timeout_report_factory=_make_timeout_report,
            timeout_seconds=float(timeout_seconds),
        )

    cmd = _solver_worker_command(
        state_path=state_path,
        background_index=int(background_index),
        active_names=active_names,
        output_path=output_path,
        heartbeat_path=heartbeat_path,
        max_nfev=int(max_nfev),
        rung=int(rung),
        rung_name=str(rung_name),
        feature=feature,
    )
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout_text, stderr_text = process.communicate(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        dirty_timeout_abort = False
        child_process_killed_cleanly = True
        try:
            process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            dirty_timeout_abort = True
            child_process_killed_cleanly = False
            pass
        if process.poll() is None:
            dirty_timeout_abort = True
            child_process_killed_cleanly = False
        report = _make_timeout_report(
            dirty_timeout_abort=dirty_timeout_abort,
            child_process_killed_cleanly=child_process_killed_cleanly,
        )
        _write_json(output_path, report)
        return report

    if output_path.exists():
        report = _read_json(output_path)
        if stdout_text.strip():
            report["worker_stdout_tail"] = stdout_text.strip()[-2000:]
        if stderr_text.strip():
            report["worker_stderr_tail"] = stderr_text.strip()[-2000:]
        _write_json(output_path, report)
        return report
    report = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": "error",
        "pass": False,
        "active_params": [str(name) for name in active_names],
        "candidate_param_names": [str(name) for name in active_names],
        "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
        "returncode": process.returncode,
        "worker_stdout_tail": stdout_text.strip()[-2000:],
        "worker_stderr_tail": stderr_text.strip()[-2000:],
        "feature": feature,
    }
    _write_json(output_path, report)
    return report


def _finalize_one_param_report(
    report: Mapping[str, object],
    *,
    param_name: str,
    state_path: Path,
    state_hash_before: str,
    timeout_seconds: float,
) -> dict[str, object]:
    finalized = dict(report)
    finalized.setdefault("rung", 3)
    finalized.setdefault("rung_name", f"one_param_{param_name}")
    finalized["param_name"] = str(param_name)
    finalized["active_params"] = [str(param_name)]
    finalized["var_names"] = [str(param_name)]
    finalized.setdefault("candidate_param_names", [str(param_name)])
    finalized["elapsed_s"] = _metric_float(
        finalized.get("elapsed_s", finalized.get("elapsed_seconds", 0.0))
    )
    finalized.setdefault("elapsed_seconds", finalized["elapsed_s"])
    finalized["timeout_s"] = float(timeout_seconds)
    finalized.setdefault("timeout_seconds", float(timeout_seconds))
    finalized["state_sha256_before"] = str(
        finalized.get("state_sha256_before") or state_hash_before
    )
    finalized["state_sha256_after"] = str(
        finalized.get("state_sha256_after") or _state_sha256(Path(state_path))
    )
    finalized["state_hash_unchanged"] = (
        finalized["state_sha256_before"] == finalized["state_sha256_after"]
    )
    _apply_single_param_fields(finalized)
    _apply_one_param_diagnostic_aliases(finalized)
    for key in (
        "before_rms_px",
        "after_rms_px",
        "before_max_error_px",
        "after_max_error_px",
        "parameter_before",
        "parameter_after",
        "parameter_delta",
    ):
        finalized.setdefault(key, None)
    finalized.setdefault("parameter_bounds", None)
    finalized.setdefault("point_match_summary", {})
    finalized.setdefault("last_nfev", finalized.get("nfev"))
    finalized.setdefault("last_residual_norm", finalized.get("residual_norm"))
    finalized.setdefault("last_rms_px", finalized.get("after_rms_px"))
    finalized.setdefault("last_max_error_px", finalized.get("after_max_error_px"))
    finalized.setdefault("last_parameter_value", finalized.get("parameter_after"))
    finalized.setdefault("current_bounds", finalized.get("parameter_bounds"))
    finalized.setdefault(
        "last_point_match_summary",
        finalized.get("point_match_summary") if finalized.get("point_match_summary") else None,
    )
    finalized.setdefault("last_heartbeat_elapsed_s", None)
    finalized.setdefault("heartbeat_count", 0)
    finalized.setdefault("child_process_killed_cleanly", None)
    finalized.setdefault("dirty_timeout_abort", False)
    for key in RUNG2_FIXED_SOURCE_COUNTS:
        finalized.setdefault(key, None)
    for key in PROVIDER_MATCH_BOOLS:
        finalized.setdefault(key, None)
    finalized["residuals_finite"] = bool(
        math.isfinite(_metric_float(finalized.get("before_rms_px", np.nan)))
        and math.isfinite(_metric_float(finalized.get("after_rms_px", np.nan)))
    )
    _apply_one_param_diagnostic_aliases(finalized)

    if str(finalized.get("status", "")) == "timeout":
        finalized["pass"] = False
        classification = str(finalized.get("diagnosis_classification") or "")
        if classification == "fixed_source_or_pair_integrity_lost":
            finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
        elif bool(finalized.get("dirty_timeout_abort", False)):
            finalized["failure_reason"] = "dirty_timeout_abort"
        elif not finalized.get("failure_reason"):
            finalized["failure_reason"] = "timeout"
        return finalized

    integrity_failures = _one_param_integrity_failures(finalized)
    metric_failures = _one_param_metric_failures(finalized)
    solve_flag_failures: list[str] = []
    if bool(finalized.get("least_squares_called", False)) is not True:
        solve_flag_failures.append("least_squares_not_called")
    if not (
        bool(finalized.get("optimizer_solve_called", False))
        or bool(finalized.get("real_solve_called", False))
    ):
        solve_flag_failures.append("optimizer_solve_not_called")
    if finalized.get("candidate_param_names") != [str(param_name)]:
        solve_flag_failures.append("candidate_param_names_not_singleton")
    if finalized.get("var_names") != [str(param_name)]:
        solve_flag_failures.append("var_names_not_singleton")
    if bool(finalized.get("state_hash_unchanged", False)) is not True:
        solve_flag_failures.append("state_hash_changed")

    guard_failures = integrity_failures + metric_failures + solve_flag_failures
    finalized["one_param_guard_failures"] = guard_failures
    finalized["diagnosis_classification"] = _diagnosis_classification(finalized)
    if not guard_failures and str(finalized.get("status", "")) in {"ok", "pass"}:
        finalized["status"] = "ok"
        finalized["pass"] = True
        finalized["failure_reason"] = None
        finalized["diagnosis_classification"] = "usable"
        return finalized

    finalized["status"] = "failed"
    finalized["pass"] = False
    if integrity_failures:
        finalized["failure_reason"] = "fixed_source_or_pair_integrity_lost"
    elif any(reason.startswith("non_finite") for reason in metric_failures):
        finalized["failure_reason"] = "non_finite_residual"
    elif "no_matched_peak_rejection" in metric_failures:
        finalized["failure_reason"] = "no_matched_peak_rejection"
    elif metric_failures:
        finalized["failure_reason"] = "metric_guard_failed"
    elif solve_flag_failures:
        finalized["failure_reason"] = "solve_flag_guard_failed"
    else:
        finalized.setdefault("failure_reason", "one_param_solve_failed")
    return finalized


def _active_params_from_sensitivity(
    sensitivity: Mapping[str, object],
    context: Mapping[str, object],
) -> tuple[list[str], list[str]]:
    active = _as_str_list(sensitivity.get("active_params"))
    excluded = (
        set(_as_str_list(sensitivity.get("near_zero_params")))
        | set(_as_str_list(sensitivity.get("non_finite_params")))
        | set(_as_str_list(sensitivity.get("unsafe_params")))
    )
    skipped = [name for name in active if name in excluded]
    filtered = [name for name in active if name not in excluded]
    return _coerce_active_names(context, filtered), skipped


def _rung2_green_failures(sensitivity: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if str(sensitivity.get("status", "")) != "ok" or bool(sensitivity.get("pass", False)) is not True:
        failures.append("sensitivity_status_not_ok")
    if bool(sensitivity.get("least_squares_called", False)):
        failures.append("least_squares_called")
    if bool(sensitivity.get("optimizer_solve_called", False)):
        failures.append("optimizer_solve_called")
    if "state_hash_unchanged" in sensitivity and bool(
        sensitivity.get("state_hash_unchanged", False)
    ) is not True:
        failures.append("state_hash_changed")
    failures.extend(
        _fixed_source_contract_failures(
            sensitivity,
            expected_counts=RUNG2_FIXED_SOURCE_COUNTS,
            required_bool_keys=PROVIDER_MATCH_BOOLS,
        )
    )

    params_by_name = {
        str(entry.get("param_name", entry.get("name", ""))): entry
        for entry in sensitivity.get("params", []) or []
        if isinstance(entry, Mapping)
    }
    for name in _as_str_list(sensitivity.get("active_params")):
        entry = params_by_name.get(name)
        if entry is None:
            failures.append(f"{name}_missing_sensitivity_entry")
            continue
        failures.extend(
            _fixed_source_contract_failures(
                entry,
                expected_counts=RUNG2_FIXED_SOURCE_COUNTS,
                required_bool_keys=PROVIDER_MATCH_BOOLS,
                prefix=f"{name}_",
            )
        )
    return failures


def _best_single_param(
    reports: Sequence[Mapping[str, object]],
    metric_key: str,
) -> dict[str, object] | None:
    best_report: Mapping[str, object] | None = None
    best_value = float("inf")
    for report in reports:
        if bool(report.get("pass", False)) is not True:
            continue
        value = _metric_float(report.get(metric_key, np.nan))
        if math.isfinite(value) and value < best_value:
            best_value = value
            best_report = report
    if best_report is None:
        return None
    return {
        "param_name": str(best_report.get("param_name", "")),
        metric_key: best_value,
    }


def _one_param_summary(
    *,
    run_dir: Path,
    sensitivity_report_path: Path,
    sensitivity: Mapping[str, object],
    active_params: Sequence[str],
    attempted_reports: Sequence[Mapping[str, object]],
    skipped_params: Sequence[str],
    state_hash_before: str,
    state_hash_after: str,
    provider_after: Mapping[str, object] | None,
    dirty_timeout_abort: bool = False,
    failure_reason: str | None = None,
    reports: Sequence[Mapping[str, object]] | None = None,
    filtered_params: Sequence[str] | None = None,
) -> dict[str, object]:
    attempted_params = [str(report.get("param_name", "")) for report in attempted_reports]
    passed_params = [
        str(report.get("param_name", ""))
        for report in attempted_reports
        if bool(report.get("pass", False))
    ]
    passing_reports = [
        report for report in attempted_reports if bool(report.get("pass", False))
    ]
    timed_out_params = [
        str(report.get("param_name", ""))
        for report in attempted_reports
        if str(report.get("status", "")) == "timeout"
    ]
    failed_params = [
        str(report.get("param_name", ""))
        for report in attempted_reports
        if str(report.get("status", "")) == "failed"
    ]
    diagnosis_by_param = {
        str(report.get("param_name", "")): str(report.get("diagnosis_classification", ""))
        for report in attempted_reports
        if report.get("diagnosis_classification")
    }
    any_pair_loss = any(
        str(report.get("failure_reason", "")) == "fixed_source_or_pair_integrity_lost"
        or str(report.get("diagnosis_classification", ""))
        == "fixed_source_or_pair_integrity_lost"
        or (
            str(report.get("status", "")) != "timeout"
            and bool(_one_param_integrity_failures(report))
        )
        for report in attempted_reports
    )
    passing_pair_loss = any(
        str(report.get("failure_reason", "")) == "fixed_source_or_pair_integrity_lost"
        or str(report.get("diagnosis_classification", ""))
        == "fixed_source_or_pair_integrity_lost"
        or bool(_one_param_integrity_failures(report))
        for report in passing_reports
    )
    any_timeout = bool(timed_out_params)
    provider_guard_after_ok = (
        bool(provider_after.get("provider_guard_ok", False))
        and str(provider_after.get("classification", "")) == "point_provider_parity_ok"
        if isinstance(provider_after, Mapping)
        else False
    )
    state_hash_unchanged = state_hash_before == state_hash_after
    all_fixed_source_counters_clean = (
        bool(attempted_reports)
        and not any_pair_loss
        and all(
            bool(report.get("fixed_source_counters_clean_at_last_heartbeat", True))
            for report in attempted_reports
        )
    )
    all_passing_fixed_source_counters_clean = bool(passing_reports) and not any(
        _one_param_integrity_failures(report) for report in passing_reports
    )

    if failure_reason:
        status = "failed"
    elif dirty_timeout_abort:
        status = "failed"
        failure_reason = "dirty_timeout_abort"
    elif not passed_params:
        status = "failed"
        failure_reason = "no_one_param_solve_passed"
    elif passing_pair_loss:
        status = "failed"
        failure_reason = "fixed_source_or_pair_integrity_lost"
    elif (
        failed_params
        or timed_out_params
        or list(skipped_params)
        or provider_guard_after_ok is not True
        or state_hash_unchanged is not True
    ):
        status = "ok_with_failures" if provider_guard_after_ok and state_hash_unchanged else "failed"
        if status == "failed" and not failure_reason:
            failure_reason = "provider_guard_after_failed" if not provider_guard_after_ok else "state_hash_changed"
    else:
        status = "ok"

    summary = {
        "rung": 3,
        "rung_name": "one_param_summary",
        "status": status,
        "failure_reason": failure_reason,
        "run_dir": str(run_dir),
        "sensitivity_report_path": str(sensitivity_report_path),
        "active_params_from_sensitivity": list(active_params),
        "attempted_params": attempted_params,
        "passed_params": passed_params,
        "failed_params": failed_params,
        "timed_out_params": timed_out_params,
        "skipped_params": [str(name) for name in skipped_params],
        "filtered_params": [str(name) for name in (filtered_params or [])],
        "diagnosis_by_param": diagnosis_by_param,
        "diagnosis_classification": (
            next(iter(diagnosis_by_param.values()))
            if len(diagnosis_by_param) == 1
            else None
        ),
        "best_single_param_by_rms": _best_single_param(attempted_reports, "after_rms_px"),
        "best_single_param_by_max_error": _best_single_param(
            attempted_reports,
            "after_max_error_px",
        ),
        "all_fixed_source_counters_clean": bool(all_fixed_source_counters_clean),
        "all_passing_fixed_source_counters_clean": bool(
            all_passing_fixed_source_counters_clean
        ),
        "any_timeout": bool(any_timeout),
        "any_pair_loss": bool(any_pair_loss),
        "any_branch_mismatch": any(
            _safe_int(report.get("branch_mismatch_count", 0), default=0) != 0
            for report in attempted_reports
        ),
        "any_no_matched_peak_rejection": any(
            NO_MATCH_REJECTION in str(report.get("rejection_reason", ""))
            for report in attempted_reports
        ),
        "dirty_timeout_abort": bool(dirty_timeout_abort),
        "state_sha256_before": state_hash_before,
        "state_sha256_after": state_hash_after,
        "state_hash_unchanged": bool(state_hash_unchanged),
        "provider_guard_after_ok": bool(provider_guard_after_ok),
        "provider_guard_after": dict(provider_after or {}),
        "rung_0_green": True,
        "rung_1_green": True,
        "rung_2_green": not _rung2_green_failures(sensitivity),
        "reports": list(reports or []),
    }
    return summary


def _passed_params_from_one_param_reports(reports: Sequence[Mapping[str, object]]) -> list[str]:
    passed: list[str] = []
    for report in reports:
        if bool(report.get("pass", False)):
            names = [str(name) for name in report.get("active_params", []) or []]
            if len(names) == 1 and names[0] not in passed:
                passed.append(names[0])
    return passed


def _run_one_param_stage(
    *,
    state_path: Path,
    background_index: int,
    run_dir: Path,
    context: Mapping[str, object],
    sensitivity: Mapping[str, object],
    sensitivity_report_path: Path,
    reports: list[dict[str, object]],
    state_hash_before: str,
    max_nfev: int,
    timeout_seconds: float,
    one_param_filter: str | None = None,
) -> dict[str, object]:
    raw_active_params = _as_str_list(sensitivity.get("active_params"))
    filter_name = str(one_param_filter or "").strip()
    rung2_failures = _rung2_green_failures(sensitivity)
    if rung2_failures:
        state_hash_after = _state_sha256(state_path)
        summary = _one_param_summary(
            run_dir=run_dir,
            sensitivity_report_path=sensitivity_report_path,
            sensitivity=sensitivity,
            active_params=raw_active_params,
            attempted_reports=[],
            skipped_params=[],
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            provider_after=None,
            failure_reason="sensitivity_not_green",
            reports=reports,
        )
        summary["rung_2_failures"] = rung2_failures
        _write_json(run_dir / "rung_03_one_param_summary.json", summary)
        return summary

    active_params, skipped_params = _active_params_from_sensitivity(sensitivity, context)
    filtered_params: list[str] = []
    if filter_name:
        filtered_params = [name for name in raw_active_params if name != filter_name]
        if filter_name not in raw_active_params or filter_name not in active_params:
            state_hash_after = _state_sha256(state_path)
            summary = _one_param_summary(
                run_dir=run_dir,
                sensitivity_report_path=sensitivity_report_path,
                sensitivity=sensitivity,
                active_params=raw_active_params,
                attempted_reports=[],
                skipped_params=skipped_params,
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                provider_after=None,
                failure_reason="filtered_param_not_active",
                reports=reports,
                filtered_params=filtered_params,
            )
            summary["one_param_filter"] = filter_name
            _write_json(run_dir / "rung_03_one_param_summary.json", summary)
            return summary
        active_params = [filter_name]
        skipped_params = [name for name in skipped_params if name != filter_name]
    if not active_params:
        state_hash_after = _state_sha256(state_path)
        summary = _one_param_summary(
            run_dir=run_dir,
            sensitivity_report_path=sensitivity_report_path,
            sensitivity=sensitivity,
            active_params=raw_active_params,
            attempted_reports=[],
            skipped_params=skipped_params,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            provider_after=None,
            failure_reason="no_active_params",
            reports=reports,
            filtered_params=filtered_params,
        )
        _write_json(run_dir / "rung_03_one_param_summary.json", summary)
        return summary

    one_param_reports: list[dict[str, object]] = []
    dirty_timeout_abort = False
    for index, name in enumerate(active_params):
        output_path = _rung_path(run_dir, 3, f"one_param_{name}")
        report = _run_solver_rung_with_timeout(
            state_path=state_path,
            background_index=int(background_index),
            active_names=[name],
            output_path=output_path,
            max_nfev=int(max_nfev),
            timeout_seconds=float(timeout_seconds),
            rung=3,
            rung_name=f"one_param_{name}",
            state_hash_before=state_hash_before,
        )
        report = _finalize_one_param_report(
            report,
            param_name=name,
            state_path=state_path,
            state_hash_before=state_hash_before,
            timeout_seconds=float(timeout_seconds),
        )
        _write_json(output_path, report)
        reports.append(report)
        one_param_reports.append(report)
        if bool(report.get("dirty_timeout_abort", False)):
            dirty_timeout_abort = True
            skipped_params.extend(active_params[index + 1 :])
            break

    provider_after: dict[str, object] | None = None
    if not dirty_timeout_abort:
        provider_after = run_provider_guard(
            state_path=state_path,
            background_index=int(background_index),
            output_path=_rung_path(run_dir, 3, "provider_guard_after"),
        )
        reports.append(provider_after)

    state_hash_after = _state_sha256(state_path)
    summary = _one_param_summary(
        run_dir=run_dir,
        sensitivity_report_path=sensitivity_report_path,
        sensitivity=sensitivity,
        active_params=raw_active_params,
        attempted_reports=one_param_reports,
        skipped_params=skipped_params,
        state_hash_before=state_hash_before,
        state_hash_after=state_hash_after,
        provider_after=provider_after,
        dirty_timeout_abort=dirty_timeout_abort,
        reports=reports,
        filtered_params=filtered_params,
    )
    if filter_name:
        summary["one_param_filter"] = filter_name
    _write_json(run_dir / "rung_03_one_param_summary.json", summary)
    return summary


def _groups_for_pair_rungs(passed_params: Sequence[str], theta_name: str) -> list[tuple[str, list[str]]]:
    passed = set(str(name) for name in passed_params)
    candidates = [
        ("center_xy", ["center_x", "center_y"]),
        ("gamma_Gamma", ["gamma", "Gamma"]),
        ("chi_cor_angle", ["chi", "cor_angle"]),
        ("theta_cor_angle", [theta_name, "cor_angle"]),
        ("corto_detector", ["corto_detector"]),
        ("zs_zb", ["zs", "zb"]),
        ("a_c", ["a", "c"]),
        ("a_c_psi_z", ["a", "c", "psi_z"]),
    ]
    return [(name, params) for name, params in candidates if all(param in passed for param in params)]


def _groups_for_cumulative_rungs(passed_params: Sequence[str], theta_name: str) -> list[tuple[str, list[str]]]:
    passed = set(str(name) for name in passed_params)
    groups: list[tuple[str, list[str]]] = []
    current: list[str] = []
    for name, additions in [
        ("center", ["center_x", "center_y"]),
        ("center_primary_tilt", ["gamma", "Gamma"]),
        ("center_tilt", ["chi", "cor_angle"]),
        ("center_tilt_theta", [theta_name]),
        ("center_tilt_theta_distance", ["corto_detector"]),
        ("center_tilt_theta_distance_z", ["zs", "zb"]),
        ("center_tilt_theta_distance_lattice", ["a", "c", "psi_z"]),
    ]:
        if all(param in passed for param in additions):
            for param in additions:
                if param not in current:
                    current.append(param)
            groups.append((name, list(current)))
    return groups


def run_ladder(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    max_rung: str,
    timeout_seconds: float = 120.0,
    max_nfev: int = 20,
    timestamp: str | None = None,
    sensitivity_report: Path | None = None,
    one_param_filter: str | None = None,
) -> dict[str, object]:
    state_path = Path(state_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    run_dir = output_root / str(timestamp or _run_stamp())
    run_dir.mkdir(parents=True, exist_ok=True)
    state_hash_before = _state_sha256(state_path)
    reports: list[dict[str, object]] = []

    provider_report = run_provider_guard(
        state_path=state_path,
        background_index=int(background_index),
        output_path=_rung_path(run_dir, 0, "provider_guard"),
    )
    reports.append(provider_report)
    if not bool(provider_report.get("provider_guard_ok", False)):
        result = {
            "status": "aborted",
            "reason": "provider_guard_failed",
            "run_dir": str(run_dir),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
        }
        _write_json(run_dir / "ladder_summary.json", result)
        return result

    context = _capture_solver_context(state_path, int(background_index))
    dry_report = run_objective_dry_run(
        context,
        output_path=_rung_path(run_dir, 1, "objective_dry_run"),
        max_nfev=int(max_nfev),
    )
    reports.append(dry_report)
    rung1_failures = _rung1_green_failures(dry_report)
    if rung1_failures:
        result = {
            "status": "aborted",
            "reason": "objective_dry_run_failed",
            "failure_reason": (
                "sensitivity_not_green"
                if str(max_rung).strip().lower() == "one-param"
                else "objective_dry_run_failed"
            ),
            "rung_1_failures": rung1_failures,
            "run_dir": str(run_dir),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
            "residual_probe_called": False,
            "least_squares_called": False,
            "optimizer_solve_called": False,
        }
        _write_json(run_dir / "ladder_summary.json", result)
        return result

    max_rung_name = str(max_rung).strip().lower()
    sensitivity_output_path = _rung_path(run_dir, 2, "sensitivity_scan")
    if sensitivity_report is not None:
        sensitivity = _read_json(Path(sensitivity_report).expanduser().resolve())
        sensitivity["debug_sensitivity_report_override"] = str(
            Path(sensitivity_report).expanduser().resolve()
        )
        _write_json(sensitivity_output_path, sensitivity)
    else:
        sensitivity = run_sensitivity_scan(
            context,
            output_path=sensitivity_output_path,
            max_nfev=int(max_nfev),
            rung_1_report=dry_report,
            state_path=state_path,
            state_hash_before=state_hash_before,
        )
    reports.append(sensitivity)
    if not bool(sensitivity.get("pass", False)):
        if max_rung_name == "one-param":
            result = _run_one_param_stage(
                state_path=state_path,
                background_index=int(background_index),
                run_dir=run_dir,
                context=context,
                sensitivity=sensitivity,
                sensitivity_report_path=sensitivity_output_path,
                reports=reports,
                state_hash_before=state_hash_before,
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                one_param_filter=one_param_filter,
            )
            _write_json(run_dir / "ladder_summary.json", result)
            return result
        result = {
            "status": "aborted",
            "reason": "sensitivity_scan_failed",
            "run_dir": str(run_dir),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
        }
        _write_json(run_dir / "ladder_summary.json", result)
        return result

    if max_rung_name == "sensitivity":
        result = {
            "status": "pass",
            "run_dir": str(run_dir),
            "final_selected_params": list(sensitivity.get("active_params", []) or []),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
            "state_unchanged": state_hash_before == _state_sha256(state_path),
            "residual_probe_called": bool(sensitivity.get("residual_probe_called", False)),
            "least_squares_called": False,
            "optimizer_solve_called": False,
        }
        _write_json(run_dir / "ladder_summary.json", result)
        return result

    if max_rung_name == "one-param":
        result = _run_one_param_stage(
            state_path=state_path,
            background_index=int(background_index),
            run_dir=run_dir,
            context=context,
            sensitivity=sensitivity,
            sensitivity_report_path=sensitivity_output_path,
            reports=reports,
            state_hash_before=state_hash_before,
            max_nfev=int(max_nfev),
            timeout_seconds=float(timeout_seconds),
            one_param_filter=one_param_filter,
        )
        _write_json(run_dir / "ladder_summary.json", result)
        return result

    sensitive = set(str(name) for name in sensitivity.get("active_parameters", []) or [])
    one_param_reports: list[dict[str, object]] = []
    center_only = max_rung_name == "center"
    params_to_run = CENTER_PARAMS if center_only else _candidate_order(context)
    for name in params_to_run:
        if name not in sensitive and not (center_only and name in CENTER_PARAMS):
            continue
        report = _run_solver_rung_with_timeout(
            state_path=state_path,
            background_index=int(background_index),
            active_names=[name],
            output_path=_rung_path(run_dir, 3, f"one_param_{name}"),
            max_nfev=int(max_nfev),
            timeout_seconds=float(timeout_seconds),
            rung=3,
            rung_name=f"one_param_{name}",
        )
        reports.append(report)
        one_param_reports.append(report)
        if not bool(report.get("pass", False)):
            result = {
                "status": "stopped",
                "reason": "one_param_failed",
                "failed_parameter": str(name),
                "run_dir": str(run_dir),
                "reports": reports,
                "state_sha256_before": state_hash_before,
                "state_sha256_after": _state_sha256(state_path),
            }
            _write_json(run_dir / "ladder_summary.json", result)
            return result

    passed_params = _passed_params_from_one_param_reports(one_param_reports)
    theta_name = _active_theta_name(context) or "theta_initial"
    pair_groups = (
        [("center_xy", ["center_x", "center_y"])]
        if center_only
        else _groups_for_pair_rungs(passed_params, theta_name)
    )
    for group_name, names in pair_groups:
        report = _run_solver_rung_with_timeout(
            state_path=state_path,
            background_index=int(background_index),
            active_names=names,
            output_path=_rung_path(run_dir, 4, f"pair_{group_name}"),
            max_nfev=int(max_nfev),
            timeout_seconds=float(timeout_seconds),
            rung=4,
            rung_name=f"pair_{group_name}",
        )
        reports.append(report)
        if not bool(report.get("pass", False)):
            result = {
                "status": "stopped",
                "reason": "pair_failed",
                "failed_group": str(group_name),
                "run_dir": str(run_dir),
                "reports": reports,
                "state_sha256_before": state_hash_before,
                "state_sha256_after": _state_sha256(state_path),
            }
            _write_json(run_dir / "ladder_summary.json", result)
            return result

    final_selected_params = list(pair_groups[-1][1]) if pair_groups else list(passed_params)
    if not center_only:
        for group_name, names in _groups_for_cumulative_rungs(passed_params, theta_name):
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=names,
                output_path=_rung_path(run_dir, 5, f"block_{group_name}"),
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=5,
                rung_name=f"block_{group_name}",
            )
            reports.append(report)
            if not bool(report.get("pass", False)):
                result = {
                    "status": "stopped",
                    "reason": "cumulative_block_failed",
                    "failed_group": str(group_name),
                    "run_dir": str(run_dir),
                    "reports": reports,
                    "state_sha256_before": state_hash_before,
                    "state_sha256_after": _state_sha256(state_path),
                }
                _write_json(run_dir / "ladder_summary.json", result)
                return result
            final_selected_params = list(names)

    if max_rung_name == "features" and final_selected_params:
        for feature in FEATURE_RUNS:
            report = _run_solver_rung_with_timeout(
                state_path=state_path,
                background_index=int(background_index),
                active_names=final_selected_params,
                output_path=_rung_path(run_dir, 6, f"feature_{feature}"),
                max_nfev=int(max_nfev),
                timeout_seconds=float(timeout_seconds),
                rung=6,
                rung_name=f"feature_{feature}",
                feature=feature,
            )
            reports.append(report)
            if not bool(report.get("pass", False)):
                result = {
                    "status": "stopped",
                    "reason": "feature_failed",
                    "failed_feature": str(feature),
                    "run_dir": str(run_dir),
                    "reports": reports,
                    "state_sha256_before": state_hash_before,
                    "state_sha256_after": _state_sha256(state_path),
                }
                _write_json(run_dir / "ladder_summary.json", result)
                return result

    result = {
        "status": "pass",
        "run_dir": str(run_dir),
        "final_selected_params": final_selected_params,
        "reports": reports,
        "state_sha256_before": state_hash_before,
        "state_sha256_after": _state_sha256(state_path),
        "state_unchanged": state_hash_before == _state_sha256(state_path),
    }
    _write_json(run_dir / "ladder_summary.json", result)
    return result


def _variant_can_continue(report: Mapping[str, object]) -> bool:
    if str(report.get("failure_reason", "")) == "dirty_timeout_abort":
        return False
    if str(report.get("diagnosis_classification", "")) == "fixed_source_or_pair_integrity_lost":
        return False
    if _safe_int(report.get("last_nfev", report.get("nfev")), default=0) <= 0:
        return False
    if not (
        math.isfinite(_metric_float(report.get("last_residual_norm", np.nan)))
        and math.isfinite(_metric_float(report.get("last_rms_px", np.nan)))
        and math.isfinite(_metric_float(report.get("last_max_error_px", np.nan)))
    ):
        return False
    if bool(report.get("fixed_source_counters_clean_at_last_heartbeat", True)) is not True:
        return False
    if bool(report.get("state_hash_unchanged", False)) is not True:
        return False
    if str(report.get("status", "")) == "timeout" and report.get("child_process_killed_cleanly") is not True:
        return False
    return True


def _single_attempt_report(result: Mapping[str, object]) -> Mapping[str, object]:
    reports = result.get("reports", [])
    if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        for report in reversed(reports):
            if not isinstance(report, Mapping):
                continue
            if int(report.get("rung", -1) or -1) == 3 and report.get("param_name"):
                return report
    return result


def run_one_param_diagnosis_variants(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    one_param_filter: str = "a",
) -> dict[str, object]:
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    attempted_variants: list[str] = []
    stopped_reason: str | None = None
    for index, (label, max_nfev, timeout_seconds) in enumerate(ONE_PARAM_A_VARIANTS):
        if index == 2 and not all(
            _variant_can_continue(_single_attempt_report(report)) for report in reports[:2]
        ):
            stopped_reason = "variant_c_conditions_not_met"
            break
        result = run_ladder(
            state_path=state_path,
            background_index=int(background_index),
            output_root=output_root,
            max_rung="one-param",
            timeout_seconds=float(timeout_seconds),
            max_nfev=int(max_nfev),
            timestamp=label,
            one_param_filter=str(one_param_filter),
        )
        reports.append(result)
        attempted_variants.append(label)
        single_report = _single_attempt_report(result)
        if bool(single_report.get("dirty_timeout_abort", False)):
            stopped_reason = "dirty_timeout_abort"
            break
        if str(single_report.get("diagnosis_classification", "")) == (
            "fixed_source_or_pair_integrity_lost"
        ):
            stopped_reason = "fixed_source_or_pair_integrity_lost"
            break
    final_report = _single_attempt_report(reports[-1]) if reports else {}
    summary = {
        "status": "failed" if stopped_reason in {"dirty_timeout_abort", "fixed_source_or_pair_integrity_lost"} else "ok",
        "failure_reason": stopped_reason,
        "one_param_filter": str(one_param_filter),
        "attempted_variants": attempted_variants,
        "diagnosis_classification": final_report.get("diagnosis_classification"),
        "reports": reports,
    }
    _write_json(output_root / "variant_summary.json", summary)
    return summary


def _worker_main(args: argparse.Namespace) -> int:
    active_names = [str(name) for name in json.loads(str(args.active_params_json))]
    _worker_solve_once(
        state_path=Path(args.state).expanduser().resolve(),
        background_index=int(args.background_index),
        active_names=active_names,
        output_path=Path(args.output_json).expanduser().resolve(),
        heartbeat_path=Path(args.heartbeat_json).expanduser().resolve(),
        max_nfev=int(args.max_nfev),
        rung=int(args.rung_number),
        rung_name=str(args.rung_name),
        feature=args.feature,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded new4 geometry-fit optimizer ladder.",
    )
    parser.add_argument("--state", required=True, help="Path to saved GUI state JSON.")
    parser.add_argument("--background-index", required=True, type=int)
    parser.add_argument("--output-root", help="Root directory for ladder artifacts.")
    parser.add_argument(
        "--max-rung",
        choices=("sensitivity", "one-param", "center", "full", "features"),
        default="center",
    )
    parser.add_argument(
        "--sensitivity-report",
        help="Explicit debug override for Rung 2 sensitivity JSON.",
    )
    parser.add_argument(
        "--one-param-filter",
        help="Run only this current-run active parameter when --max-rung=one-param.",
    )
    parser.add_argument(
        "--run-one-param-variants",
        action="store_true",
        help="Run the bounded A/B/C one-param diagnostic sequence.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument("--_worker-solve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", help=argparse.SUPPRESS)
    parser.add_argument("--heartbeat-json", help=argparse.SUPPRESS)
    parser.add_argument("--active-params-json", help=argparse.SUPPRESS)
    parser.add_argument("--rung-number", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rung-name", help=argparse.SUPPRESS)
    parser.add_argument("--feature", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args._worker_solve:
        return _worker_main(args)
    if not args.output_root:
        parser.error("--output-root is required.")
    if bool(getattr(args, "run_one_param_variants", False)):
        result = run_one_param_diagnosis_variants(
            state_path=Path(args.state),
            background_index=int(args.background_index),
            output_root=Path(args.output_root),
            one_param_filter=str(args.one_param_filter or "a"),
        )
    else:
        result = run_ladder(
            state_path=Path(args.state),
            background_index=int(args.background_index),
            output_root=Path(args.output_root),
            max_rung=str(args.max_rung),
            timeout_seconds=float(args.timeout_seconds),
            max_nfev=int(args.max_nfev),
            sensitivity_report=Path(args.sensitivity_report) if args.sensitivity_report else None,
            one_param_filter=str(args.one_param_filter or "") or None,
        )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "reason": result.get("reason"),
                "failure_reason": result.get("failure_reason"),
                "run_dir": result.get("run_dir"),
                "state_unchanged": result.get("state_unchanged"),
            },
            sort_keys=True,
        )
    )
    return 0 if str(result.get("status")) in {"pass", "ok", "ok_with_failures", "stopped", "aborted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
