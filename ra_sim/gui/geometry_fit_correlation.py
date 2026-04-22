"""Headless geometry-fit parameter correlation export helpers.

The tool computes the same data-only identifiability matrix used by the
geometry fitter, but it does so from a saved GUI state without opening Tk or
changing the saved state.  Correlations are derived from finite-difference
Jacobians of the geometry-fit residual with respect to selectable geometry-fit
parameters.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np

from ra_sim import headless_geometry_fit as hgf
from ra_sim.fitting import optimization as opt
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.io.data_loading import load_gui_state_file


DEFAULT_GEOMETRY_FIT_PARAMETER_ORDER: tuple[str, ...] = (
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
)

MATRIX_CSV_NAME = "geometry_fit_correlation_matrix.csv"
PAIR_CSV_NAME = "geometry_fit_correlation_pairs.csv"
PARAMETER_CSV_NAME = "geometry_fit_parameter_sensitivity.csv"
SUMMARY_JSON_NAME = "geometry_fit_correlation_summary.json"


@dataclass(frozen=True)
class GeometryFitCorrelationResult:
    """Structured result for one headless geometry-fit correlation run."""

    state_path: Path
    background_index: int | None
    parameters: list[str]
    summary: dict[str, object]
    request_summary: dict[str, object]
    solver_probe_records: list[dict[str, object]]
    metadata: dict[str, object]


class GeometryFitCorrelationError(RuntimeError):
    """Raised when the headless correlation map cannot be computed."""


class _CapturedExecutionSetup(RuntimeError):
    """Internal stop used to abort the headless run after setup capture."""


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    return float(numeric) if math.isfinite(numeric) else float(default)


def _load_saved_state_for_background(
    state_path: Path,
    background_index: int | None,
) -> dict[str, object]:
    payload = load_gui_state_file(Path(state_path))
    saved_state = copy.deepcopy(dict(payload["state"]))
    if background_index is None:
        return saved_state
    files_state = saved_state.setdefault("files", {})
    if isinstance(files_state, dict):
        files_state["current_background_index"] = int(background_index)
    variables = saved_state.setdefault("variables", {})
    if isinstance(variables, dict):
        variables["geometry_fit_background_selection_var"] = "current"
    return saved_state


def capture_solver_context(
    state_path: str | Path,
    *,
    background_index: int | None = None,
) -> dict[str, object]:
    """Capture a prepared geometry-fit solver context without running the fit."""

    resolved_state_path = Path(state_path).expanduser().resolve()
    saved_state = _load_saved_state_for_background(resolved_state_path, background_index)
    captured: dict[str, object] = {}
    original_prepare = hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run
    original_execute = hgf.gui_geometry_fit.execute_runtime_geometry_fit

    def _prepare_wrapper(*args: object, **kwargs: object) -> object:
        result = original_prepare(*args, **kwargs)
        captured["prepare_kwargs"] = dict(kwargs)
        captured["prepare_result"] = result
        return result

    def _execute_wrapper(*args: object, **kwargs: object) -> object:
        del args
        captured["execute_kwargs"] = dict(kwargs)
        raise _CapturedExecutionSetup()

    hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = _prepare_wrapper
    hgf.gui_geometry_fit.execute_runtime_geometry_fit = _execute_wrapper
    try:
        hgf.run_headless_geometry_fit(
            saved_state,
            state_path=resolved_state_path,
            downloads_dir=resolved_state_path.parent,
            stamp=f"{resolved_state_path.stem}_correlation_probe",
        )
    except _CapturedExecutionSetup:
        pass
    except Exception as exc:
        captured["execution_error_text"] = str(exc)
    finally:
        hgf.gui_geometry_fit.prepare_runtime_geometry_fit_run = original_prepare
        hgf.gui_geometry_fit.execute_runtime_geometry_fit = original_execute

    prepare_result = captured.get("prepare_result")
    execute_kwargs = (
        dict(captured.get("execute_kwargs", {}))
        if isinstance(captured.get("execute_kwargs"), Mapping)
        else {}
    )
    prepared_run = execute_kwargs.get("prepared_run")
    if prepared_run is None:
        prepared_run = getattr(prepare_result, "prepared_run", None)
    if prepared_run is None:
        error_text = str(getattr(prepare_result, "error_text", "") or "")
        if not error_text:
            error_text = str(captured.get("execution_error_text", ""))
        raise GeometryFitCorrelationError(error_text or "Geometry fit preflight failed.")
    setup = execute_kwargs.get("setup")
    postprocess_config = getattr(setup, "postprocess_config", None)
    solver_inputs = getattr(postprocess_config, "solver_inputs", None)
    if solver_inputs is None:
        raise GeometryFitCorrelationError("Captured setup did not include solver inputs.")
    prepare_kwargs = (
        dict(captured.get("prepare_kwargs", {}))
        if isinstance(captured.get("prepare_kwargs"), Mapping)
        else {}
    )
    return {
        "state_path": str(resolved_state_path),
        "background_index": background_index,
        "prepared_run": prepared_run,
        "solver_inputs": solver_inputs,
        "saved_var_names": [
            str(name)
            for name in execute_kwargs.get("var_names", prepare_kwargs.get("var_names", ()))
            or ()
        ],
    }


def _active_theta_name(context: Mapping[str, object]) -> str | None:
    saved_var_names = [str(name) for name in context.get("saved_var_names", []) or []]
    prepared_run = context.get("prepared_run")
    fit_params = (
        dict(getattr(prepared_run, "fit_params", {}) or {}) if prepared_run is not None else {}
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


def available_geometry_fit_parameters(context: Mapping[str, object]) -> list[str]:
    """Return selectable continuous geometry-fit parameters for a captured context."""

    prepared_run = context.get("prepared_run")
    fit_params = (
        dict(getattr(prepared_run, "fit_params", {}) or {}) if prepared_run is not None else {}
    )
    theta_name = _active_theta_name(context)
    ordered: list[str] = []
    for raw_name in DEFAULT_GEOMETRY_FIT_PARAMETER_ORDER:
        name = theta_name if raw_name == "theta_initial" and theta_name else raw_name
        if raw_name == "theta_initial" and theta_name is None:
            continue
        if name in fit_params and name not in ordered:
            ordered.append(str(name))
    return ordered


def _parse_requested_parameters(
    context: Mapping[str, object],
    parameter_names: Sequence[str] | str | None,
) -> list[str]:
    available = available_geometry_fit_parameters(context)
    if parameter_names is None:
        return available
    if isinstance(parameter_names, str):
        text = parameter_names.strip()
        if not text or text.lower() == "all":
            return available
        if text.lower() == "active":
            active = [str(name) for name in context.get("saved_var_names", []) or []]
            return [name for name in active if name in available]
        requested = [part.strip() for part in text.split(",") if part.strip()]
    else:
        requested = [str(name).strip() for name in parameter_names if str(name).strip()]
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise GeometryFitCorrelationError(
            "Unknown or unavailable geometry-fit parameter(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(available)
        )
    deduped: list[str] = []
    for name in requested:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _correlation_runtime_config(
    runtime_cfg: Mapping[str, object] | None,
    *,
    candidate_names: Sequence[str],
    max_nfev: int,
    correlation_threshold: float,
) -> dict[str, object]:
    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    cfg["candidate_param_names"] = [str(name) for name in candidate_names]

    solver_raw = cfg.get("solver", cfg.get("optimizer", {}))
    solver = dict(solver_raw) if isinstance(solver_raw, Mapping) else {}
    solver["max_nfev"] = max(1, int(max_nfev))
    solver.setdefault("loss", "linear")
    solver.setdefault("f_scale_px", 1.0)
    solver["workers"] = solver.get("workers", "auto")
    solver["parallel_mode"] = "off"
    solver["worker_numba_threads"] = 0
    solver["restarts"] = 0
    solver.pop("reparameterize_pairs", None)
    solver.pop("staged_release", None)
    cfg["solver"] = solver
    cfg["optimizer"] = dict(solver)

    seed = dict(cfg.get("seed_search", {}) if isinstance(cfg.get("seed_search"), Mapping) else {})
    seed["prescore_top_k"] = 1
    seed["n_global"] = 0
    seed["n_jitter"] = 0
    seed["min_seed_separation_u"] = 2.0
    cfg["seed_search"] = seed

    discrete = dict(
        cfg.get("discrete_modes", {}) if isinstance(cfg.get("discrete_modes"), Mapping) else {}
    )
    discrete["enabled"] = False
    cfg["discrete_modes"] = discrete

    for key in ("full_beam_polish", "ridge_refinement", "image_refinement"):
        stage = dict(cfg.get(key, {}) if isinstance(cfg.get(key), Mapping) else {})
        stage["enabled"] = False
        cfg[key] = stage
    for key in ("reparameterize_pairs", "staged_release"):
        cfg.pop(key, None)

    ident = dict(
        cfg.get("identifiability", {}) if isinstance(cfg.get("identifiability"), Mapping) else {}
    )
    ident["enabled"] = True
    ident["block_corr_abs"] = float(correlation_threshold)
    ident["correlation_warn"] = float(correlation_threshold)
    ident.pop("auto_freeze", None)
    ident.pop("selective_thaw", None)
    ident.pop("adaptive_regularization", None)
    cfg["identifiability"] = ident
    cfg["use_numba"] = bool(cfg.get("use_numba", False))
    return cfg


def _build_solver_request(
    context: Mapping[str, object],
    *,
    active_names: Sequence[str],
    candidate_names: Sequence[str],
    max_nfev: int,
    correlation_threshold: float,
) -> gui_geometry_fit.GeometryFitSolverRequest:
    prepared_run = context["prepared_run"]
    runtime_cfg = _correlation_runtime_config(
        getattr(prepared_run, "geometry_runtime_cfg", {}),
        candidate_names=candidate_names,
        max_nfev=max_nfev,
        correlation_threshold=correlation_threshold,
    )
    prepared_for_probe = replace(prepared_run, geometry_runtime_cfg=runtime_cfg)
    request = gui_geometry_fit.build_geometry_fit_solver_request(
        prepared_run=prepared_for_probe,
        var_names=[str(name) for name in active_names],
        solver_inputs=context["solver_inputs"],
    )
    candidate_list = [str(name) for name in candidate_names]
    if request.candidate_param_names != candidate_list:
        request = gui_geometry_fit.GeometryFitSolverRequest(
            miller=request.miller,
            intensities=request.intensities,
            image_size=request.image_size,
            params=request.params,
            measured_peaks=request.measured_peaks,
            var_names=list(request.var_names),
            candidate_param_names=candidate_list,
            dataset_specs=request.dataset_specs,
            refinement_config={
                **dict(request.refinement_config),
                "candidate_param_names": candidate_list,
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


def request_handoff_summary(request: gui_geometry_fit.GeometryFitSolverRequest) -> dict[str, object]:
    """Return compact source-row and pair-count diagnostics for a solver request."""

    rows = _rows_from_request(request)
    cfg = request.refinement_config if isinstance(request.refinement_config, Mapping) else {}
    bridge_summary = (
        dict(cfg.get("optimizer_request_handoff_summary", {}))
        if isinstance(cfg.get("optimizer_request_handoff_summary", {}), Mapping)
        else {}
    )
    fallback_count = 0
    trusted_count = 0
    identity_count = 0
    for row in rows:
        fit_kind = str(row.get("fit_source_resolution_kind", "") or "").strip().lower()
        resolution_kind = str(row.get("resolution_kind", "") or "").strip().lower()
        if (
            bool(row.get("optimizer_request_fallback_row", False))
            or bool(row.get("rebinding_fallback_used", False))
            or "fallback" in fit_kind
            or resolution_kind == "hkl_fallback"
        ):
            fallback_count += 1
        if bool(row.get("optimizer_request_has_fixed_source", False)) or (
            row.get("source_reflection_namespace") in {"full", "full_reflection", "miller"}
            or bool(row.get("source_reflection_is_full", False))
        ):
            trusted_count += 1
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
            identity_count += 1
    computed: dict[str, object] = {
        "handoff_row_count": int(len(rows)),
        "handoff_source_identity_row_count": int(identity_count),
        "handoff_trusted_full_source_row_count": int(trusted_count),
        "optimizer_request_pair_count": int(len(rows)),
        "fallback_row_count": int(fallback_count),
        "fixed_source_pair_count": int(trusted_count),
        "missing_fixed_source_count": max(0, int(len(rows)) - int(trusted_count)),
        "var_names": [str(name) for name in request.var_names],
        "candidate_param_names": [
            str(name) for name in (request.candidate_param_names or request.var_names)
        ],
        "runtime_safety_note": request.runtime_safety_note,
    }
    computed.update(bridge_summary)
    return computed


class _NoSolveLeastSquaresProbe:
    """Patch target for scipy least_squares that evaluates only the baseline residual."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    @staticmethod
    def _residual_norm(residual: np.ndarray) -> float:
        finite = np.asarray(residual, dtype=float).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            return float("nan")
        return float(np.linalg.norm(finite))

    def __call__(self, fun: object, x0: object, *args: object, **kwargs: object) -> object:
        del args
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        try:
            residual = np.asarray(fun(np.array(x0_arr, dtype=float)), dtype=float).reshape(-1)
            error_text = ""
            raised = False
        except Exception as exc:
            residual = np.array([float("nan")], dtype=float)
            error_text = str(exc)
            raised = True
        record = {
            "x0": x0_arr.tolist(),
            "dimension": int(x0_arr.size),
            "residual_size": int(residual.size),
            "residual_norm": self._residual_norm(residual),
            "finite": bool(np.all(np.isfinite(residual))),
            "raised": bool(raised),
            "error_text": error_text,
            "bounds_present": "bounds" in kwargs,
        }
        self.records.append(record)
        return opt.OptimizeResult(
            x=x0_arr,
            fun=residual,
            success=not raised,
            status=1 if not raised else 0,
            message=(
                "headless geometry-fit correlation baseline probe"
                if not raised
                else f"correlation probe failed: {error_text}"
            ),
            nfev=1,
            active_mask=np.zeros(x0_arr.shape, dtype=int),
            optimality=float("nan"),
        )


def _run_solver_probe(request: gui_geometry_fit.GeometryFitSolverRequest) -> tuple[object, list[dict[str, object]]]:
    probe = _NoSolveLeastSquaresProbe()
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


def _summary_parameters(summary: Mapping[str, object]) -> list[str]:
    entries = summary.get("parameter_entries", [])
    names: list[str] = []
    if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name") is not None:
                names.append(str(entry.get("name")))
    return names


def _matrix_from_summary(summary: Mapping[str, object]) -> np.ndarray:
    matrix_raw = summary.get("correlation_u", summary.get("correlation_matrix", []))
    try:
        matrix = np.asarray(matrix_raw, dtype=float)
    except Exception:
        return np.empty((0, 0), dtype=float)
    if matrix.ndim != 2:
        return np.empty((0, 0), dtype=float)
    return matrix


def _pair_rows_from_summary(
    summary: Mapping[str, object],
    parameters: Sequence[str],
    *,
    correlation_threshold: float,
) -> list[dict[str, object]]:
    matrix = _matrix_from_summary(summary)
    n = min(len(parameters), matrix.shape[0], matrix.shape[1]) if matrix.size else 0
    rows: list[dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = float(matrix[i, j])
            rows.append(
                {
                    "parameter_i": str(parameters[i]),
                    "index_i": int(i),
                    "parameter_j": str(parameters[j]),
                    "index_j": int(j),
                    "correlation": corr if math.isfinite(corr) else float("nan"),
                    "abs_correlation": abs(corr) if math.isfinite(corr) else float("nan"),
                    "high_correlation": bool(
                        math.isfinite(corr) and abs(corr) >= float(correlation_threshold)
                    ),
                }
            )
    rows.sort(
        key=lambda item: (
            -_finite_float(item.get("abs_correlation"), default=-1.0),
            str(item.get("parameter_i", "")),
            str(item.get("parameter_j", "")),
        )
    )
    return rows


def run_geometry_fit_correlation(
    *,
    state_path: str | Path,
    background_index: int | None = None,
    parameter_names: Sequence[str] | str | None = None,
    max_nfev: int = 1,
    correlation_threshold: float = 0.90,
    outdir: str | Path | None = None,
) -> GeometryFitCorrelationResult:
    """Compute a headless pairwise correlation map for geometry-fit parameters."""

    context = capture_solver_context(state_path, background_index=background_index)
    candidates = _parse_requested_parameters(context, parameter_names)
    if not candidates:
        raise GeometryFitCorrelationError("No available geometry-fit parameters were selected.")
    active_names = list(candidates)
    request = _build_solver_request(
        context,
        active_names=active_names,
        candidate_names=candidates,
        max_nfev=max_nfev,
        correlation_threshold=correlation_threshold,
    )
    result, probe_records = _run_solver_probe(request)
    summary = getattr(result, "data_only_identifiability_summary", None)
    if not isinstance(summary, Mapping):
        raise GeometryFitCorrelationError("Solver did not return a data-only identifiability summary.")
    summary_dict = copy.deepcopy(dict(summary))
    status = str(summary_dict.get("status", ""))
    if status != "ok":
        reason = str(summary_dict.get("reason", ""))
        raise GeometryFitCorrelationError(
            f"Geometry-fit correlation summary failed: {status or 'unknown'}"
            + (f" ({reason})" if reason else "")
        )
    parameters = _summary_parameters(summary_dict) or list(candidates)
    pair_rows = _pair_rows_from_summary(
        summary_dict,
        parameters,
        correlation_threshold=correlation_threshold,
    )
    request_summary = request_handoff_summary(request)
    metadata: dict[str, object] = {
        "status": "ok",
        "state_path": str(Path(state_path).expanduser().resolve()),
        "background_index": background_index,
        "parameters": list(parameters),
        "requested_parameters": list(candidates),
        "active_parameters": [str(name) for name in active_names],
        "correlation_threshold": float(correlation_threshold),
        "max_nfev": int(max_nfev),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "diagnostic_scope": str(summary_dict.get("diagnostic_scope", "data_only_all_selectable")),
        "includes_priors": bool(summary_dict.get("includes_priors", False)),
        "pair_count": int(len(pair_rows)),
        "high_correlation_pair_count": int(
            sum(1 for row in pair_rows if bool(row.get("high_correlation", False)))
        ),
        "baseline_residual_probe": probe_records[-1] if probe_records else {},
    }
    summary_dict["all_correlation_pairs"] = pair_rows
    final = GeometryFitCorrelationResult(
        state_path=Path(state_path).expanduser().resolve(),
        background_index=background_index,
        parameters=list(parameters),
        summary=summary_dict,
        request_summary=request_summary,
        solver_probe_records=probe_records,
        metadata=metadata,
    )
    if outdir is not None:
        write_correlation_artifacts(final, outdir)
    return final


def write_correlation_artifacts(
    result: GeometryFitCorrelationResult,
    outdir: str | Path,
) -> dict[str, Path]:
    """Write JSON and CSV artifacts for a correlation result."""

    output_dir = Path(outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_json": output_dir / SUMMARY_JSON_NAME,
        "correlation_matrix": output_dir / MATRIX_CSV_NAME,
        "correlation_pairs": output_dir / PAIR_CSV_NAME,
        "parameter_sensitivity": output_dir / PARAMETER_CSV_NAME,
    }
    payload = {
        "metadata": result.metadata,
        "summary": result.summary,
        "request_summary": result.request_summary,
        "solver_probe_records": result.solver_probe_records,
    }
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)

    matrix = _matrix_from_summary(result.summary)
    parameters = list(result.parameters)
    with paths["correlation_matrix"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", *parameters])
        n = min(len(parameters), matrix.shape[0], matrix.shape[1]) if matrix.size else 0
        for i, name in enumerate(parameters):
            row_values: list[object] = [name]
            for j in range(len(parameters)):
                if i < n and j < n:
                    value = float(matrix[i, j])
                    row_values.append(value if math.isfinite(value) else "")
                else:
                    row_values.append("")
            writer.writerow(row_values)

    pair_rows = result.summary.get("all_correlation_pairs", [])
    with paths["correlation_pairs"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "parameter_i",
            "index_i",
            "parameter_j",
            "index_j",
            "correlation",
            "abs_correlation",
            "high_correlation",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(pair_rows, Sequence) and not isinstance(pair_rows, (str, bytes)):
            for row in pair_rows:
                if isinstance(row, Mapping):
                    writer.writerow({key: row.get(key, "") for key in fieldnames})

    entries = result.summary.get("parameter_entries", [])
    with paths["parameter_sensitivity"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "name",
            "index",
            "valid",
            "column_norm",
            "relative_sensitivity",
            "std_u",
            "std_theta",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
            for idx, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    continue
                writer.writerow(
                    {
                        "name": entry.get("name", ""),
                        "index": entry.get("index", idx),
                        "valid": entry.get("valid", ""),
                        "column_norm": entry.get("column_norm", ""),
                        "relative_sensitivity": entry.get("relative_sensitivity", ""),
                        "std_u": entry.get("std_u", ""),
                        "std_theta": entry.get("std_theta", ""),
                    }
                )
    return paths
