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
        if name not in fit_params and name != "theta_initial":
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
    for key in STRICT_POINT_SUMMARY_KEYS:
        report[key] = _summary_int(point_summary, key)
    report.update(_request_handoff_summary(request))
    report["branch_mismatch_count"] = int(
        point_summary.get("branch_mismatch_count", 0) or 0
    )
    if isinstance(extra, Mapping):
        report.update(dict(extra))
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


def _rung_passed(report: Mapping[str, object]) -> bool:
    if str(report.get("status", "")) not in {"ok", "pass"}:
        return False
    if not math.isfinite(_metric_float(report.get("after_rms_px", np.nan))):
        return False
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
    return True


class _ProbeLeastSquares:
    def __init__(self, *, mode: str, step: float = 1.0e-3) -> None:
        self.mode = str(mode)
        self.step = float(step)
        self.records: list[dict[str, object]] = []

    def __call__(self, fun: Callable[[np.ndarray], np.ndarray], x0, *args, **kwargs):
        del args
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        residual0 = np.asarray(fun(np.array(x0_arr, dtype=float)), dtype=float)
        record: dict[str, object] = {
            "x0": x0_arr.tolist(),
            "residual_size": int(residual0.size),
            "residual_norm": float(np.linalg.norm(residual0[np.isfinite(residual0)]))
            if np.any(np.isfinite(residual0))
            else float("nan"),
            "finite": bool(np.all(np.isfinite(residual0))),
        }
        if self.mode == "sensitivity" and x0_arr.size:
            lower, upper = kwargs.get("bounds", (None, None))
            lower_arr = np.asarray(lower, dtype=float).reshape(-1) if lower is not None else None
            upper_arr = np.asarray(upper, dtype=float).reshape(-1) if upper is not None else None
            x1 = np.array(x0_arr, dtype=float)
            x1[0] = x1[0] + float(self.step)
            if lower_arr is not None and lower_arr.size == x1.size:
                x1 = np.maximum(x1, lower_arr)
            if upper_arr is not None and upper_arr.size == x1.size:
                x1 = np.minimum(x1, upper_arr)
            residual1 = np.asarray(fun(x1), dtype=float)
            delta = (
                residual1 - residual0
                if residual1.shape == residual0.shape
                else np.array([float("nan")], dtype=float)
            )
            record.update(
                {
                    "x1": x1.tolist(),
                    "delta_step": float(x1[0] - x0_arr[0]),
                    "delta_norm": float(np.linalg.norm(delta[np.isfinite(delta)]))
                    if np.any(np.isfinite(delta))
                    else float("nan"),
                    "delta_finite": bool(np.all(np.isfinite(residual1))),
                }
            )
            nfev = 2
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
    original = opt.least_squares
    opt.least_squares = probe
    try:
        result = gui_geometry_fit.solve_geometry_fit_request(
            request,
            solve_fit=opt.fit_geometry_parameters,
        )
    finally:
        opt.least_squares = original
    return result, list(probe.records)


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
        "solver_success": False,
        "solver_message": "",
        "rejection_reasons": [],
        "rejection_reason": "",
        "point_match_summary": {},
        "stage_timing_s": {},
        "parameter_deltas": [],
        "optimizer_called": False,
    }
    for key in STRICT_POINT_SUMMARY_KEYS:
        report[key] = 0
    report.update(_request_handoff_summary(request))
    if isinstance(extra, Mapping):
        report.update(dict(extra))
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
            },
        )
        _write_json(output_path, report)
        return report
    result, records = _run_with_probe_least_squares(request, mode="dry_run")
    report = _result_report(
        request=request,
        result=result,
        rung=1,
        rung_name="objective_dry_run",
        started_at=started,
        status="ok",
        extra={"least_squares_probe_records": records},
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


def _active_theta_name(context: Mapping[str, object]) -> str:
    saved_var_names = [str(name) for name in context.get("saved_var_names", []) or []]
    if "theta_offset" in saved_var_names:
        return "theta_offset"
    return "theta_initial"


def _candidate_order(context: Mapping[str, object]) -> list[str]:
    theta_name = _active_theta_name(context)
    ordered: list[str] = []
    for name in ONE_PARAM_ORDER:
        actual = theta_name if name == "theta_initial" else name
        if actual not in ordered:
            ordered.append(actual)
    return _coerce_active_names(context, ordered)


def run_sensitivity_scan(
    context: Mapping[str, object],
    *,
    output_path: Path,
    max_nfev: int,
) -> dict[str, object]:
    started = time.monotonic()
    entries: list[dict[str, object]] = []
    for name in _candidate_order(context):
        param_started = time.monotonic()
        try:
            request = build_solver_request(context, [name], max_nfev=max_nfev)
            result, records = _run_with_probe_least_squares(request, mode="sensitivity")
            point_summary = _point_match_summary(result)
            probe_record = records[-1] if records else {}
            delta_norm = _metric_float(probe_record.get("delta_norm", np.nan))
            residual_norm = _metric_float(probe_record.get("residual_norm", np.nan))
            finite = bool(probe_record.get("finite", False)) and bool(
                probe_record.get("delta_finite", False)
            )
            threshold = max(1.0e-7, 1.0e-7 * abs(residual_norm) if math.isfinite(residual_norm) else 0.0)
            if not finite:
                classification = "non-finite/unsafe"
            elif not math.isfinite(delta_norm) or delta_norm <= threshold:
                classification = "near-zero sensitivity"
            else:
                classification = "active/sensitive"
            entries.append(
                {
                    "name": str(name),
                    "classification": classification,
                    "delta_norm": delta_norm,
                    "residual_norm": residual_norm,
                    "matched_pair_count": int(point_summary.get("matched_pair_count", 0) or 0),
                    "missing_pair_count": int(point_summary.get("missing_pair_count", 0) or 0),
                    "branch_mismatch_count": int(point_summary.get("branch_mismatch_count", 0) or 0),
                    "probe_records": records,
                    "elapsed_seconds": float(max(0.0, time.monotonic() - param_started)),
                }
            )
        except Exception as exc:
            entries.append(
                {
                    "name": str(name),
                    "classification": "non-finite/unsafe",
                    "error_text": str(exc),
                    "elapsed_seconds": float(max(0.0, time.monotonic() - param_started)),
                }
            )
    payload = {
        "rung": 2,
        "rung_name": "sensitivity_scan",
        "status": "ok"
        if all(str(entry.get("classification")) != "non-finite/unsafe" for entry in entries)
        else "fail",
        "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
        "parameters": entries,
        "active_parameters": [
            str(entry["name"])
            for entry in entries
            if str(entry.get("classification")) == "active/sensitive"
        ],
        "near_zero_parameters": [
            str(entry["name"])
            for entry in entries
            if str(entry.get("classification")) == "near-zero sensitivity"
        ],
        "unsafe_parameters": [
            str(entry["name"])
            for entry in entries
            if str(entry.get("classification")) == "non-finite/unsafe"
        ],
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
    _heartbeat_write(
        heartbeat_path,
        {
            "status": "starting",
            "rung": int(rung),
            "rung_name": str(rung_name),
            "active_params": [str(name) for name in active_names],
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
        point_summary = (
            dict(payload.get("point_match_summary", {}))
            if isinstance(payload.get("point_match_summary", {}), Mapping)
            else {}
        )
        _heartbeat_write(
            heartbeat_path,
            {
                "status": "running",
                "last_point_match_summary": point_summary,
                "current_rms_px": _summary_rms(point_summary),
            },
        )

    try:
        result = gui_geometry_fit.solve_geometry_fit_request(
            request,
            solve_fit=opt.fit_geometry_parameters,
            status_callback=_status_callback,
            live_update_callback=_live_update,
        )
        report = _result_report(
            request=request,
            result=result,
            rung=int(rung),
            rung_name=str(rung_name),
            started_at=started,
            before_summary=before_summary,
            status="ok",
            extra={"feature": feature},
        )
    except Exception as exc:
        report = {
            "rung": int(rung),
            "rung_name": str(rung_name),
            "status": "error",
            "pass": False,
            "active_params": [str(name) for name in active_names],
            "candidate_param_names": [str(name) for name in active_names],
            "elapsed_seconds": float(max(0.0, time.monotonic() - started)),
            "error_text": str(exc),
            "feature": feature,
        }
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
    feature: str | None = None,
) -> dict[str, object]:
    heartbeat = _read_json(heartbeat_path)
    report = {
        "rung": int(rung),
        "rung_name": str(rung_name),
        "status": "timeout",
        "pass": False,
        "active_params": [str(name) for name in active_names],
        "candidate_param_names": [str(name) for name in active_names],
        "elapsed_seconds": float(max(0.0, time.monotonic() - started_at)),
        "timeout_seconds": float(timeout_seconds),
        "last_emitted_optimizer_status": heartbeat.get("last_optimizer_status"),
        "nfev": heartbeat.get("nfev"),
        "current_rms_px": heartbeat.get("current_rms_px"),
        "last_point_match_summary": heartbeat.get("last_point_match_summary"),
        "feature": feature,
    }
    for key in REQUEST_HANDOFF_FIELDS:
        report[key] = heartbeat.get(key)
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
    use_subprocess: bool = True,
) -> dict[str, object]:
    started = time.monotonic()
    heartbeat_path = output_path.with_suffix(".heartbeat.json")

    def _make_timeout_report() -> dict[str, object]:
        return _timeout_report(
            rung=int(rung),
            rung_name=str(rung_name),
            active_names=active_names,
            started_at=started,
            heartbeat_path=heartbeat_path,
            timeout_seconds=float(timeout_seconds),
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
        try:
            process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        report = _make_timeout_report()
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


def _passed_params_from_one_param_reports(reports: Sequence[Mapping[str, object]]) -> list[str]:
    passed: list[str] = []
    for report in reports:
        if bool(report.get("pass", False)):
            names = [str(name) for name in report.get("active_params", []) or []]
            if len(names) == 1 and names[0] not in passed:
                passed.append(names[0])
    return passed


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
    if not bool(dry_report.get("pass", False)):
        result = {
            "status": "aborted",
            "reason": "objective_dry_run_failed",
            "run_dir": str(run_dir),
            "reports": reports,
            "state_sha256_before": state_hash_before,
            "state_sha256_after": _state_sha256(state_path),
        }
        _write_json(run_dir / "ladder_summary.json", result)
        return result

    sensitivity = run_sensitivity_scan(
        context,
        output_path=_rung_path(run_dir, 2, "sensitivity_scan"),
        max_nfev=int(max_nfev),
    )
    reports.append(sensitivity)
    if not bool(sensitivity.get("pass", False)):
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

    sensitive = set(str(name) for name in sensitivity.get("active_parameters", []) or [])
    one_param_reports: list[dict[str, object]] = []
    center_only = str(max_rung).strip().lower() == "center"
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
    theta_name = _active_theta_name(context)
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

    if str(max_rung).strip().lower() == "features" and final_selected_params:
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
        choices=("center", "full", "features"),
        default="center",
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
    result = run_ladder(
        state_path=Path(args.state),
        background_index=int(args.background_index),
        output_root=Path(args.output_root),
        max_rung=str(args.max_rung),
        timeout_seconds=float(args.timeout_seconds),
        max_nfev=int(args.max_nfev),
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "reason": result.get("reason"),
                "run_dir": result.get("run_dir"),
                "state_unchanged": result.get("state_unchanged"),
            },
            sort_keys=True,
        )
    )
    return 0 if str(result.get("status")) in {"pass", "stopped", "aborted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
