#!/usr/bin/env python
"""Audit New4 manual caked Qr fit coordinates as JSON plus scatter PNG."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.fitting import optimization as opt  # noqa: E402
from scripts.debug import run_new4_caked_point_reprojection_check as reprojection  # noqa: E402
from scripts.debug import run_new4_geometry_fit_ladder as ladder  # noqa: E402
from scripts.debug import validate_geometry_preflight_rebind as preflight  # noqa: E402


DEFAULT_STATE_PATH = REPO_ROOT / "artifacts" / "geometry_fit_gui_states" / "new4.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "geometry_fit_ladder" / "new4_coordinate_audit"
REPORT_NAME = "new4_qr_fit_coordinates.json"
PLOT_NAME = "new4_qr_fit_coordinates.png"
EXACT_TOL_DEG = 1.0e-6
AUDIT_SCHEMA_VERSION = 1
OBJECTIVE_AUDIT_REQUIRED_PAIR_FIELDS: tuple[str, ...] = (
    "pair_id",
    "q_group_key",
    "normalized_hkl",
    "physical_branch_slot",
    "fit_qr_branch_key",
    "manual_visual_x",
    "manual_visual_y",
    "manual_visual_frame",
    "cached_target_two_theta_deg",
    "cached_target_phi_deg",
    "cached_target_source",
    "optimizer_measured_two_theta_deg",
    "optimizer_measured_phi_deg",
    "optimizer_measured_source",
    "dynamic_source_two_theta_deg",
    "dynamic_source_phi_deg",
    "dynamic_source_source",
    "optimizer_source_two_theta_deg",
    "optimizer_source_phi_deg",
    "optimizer_source_source",
    "residual_two_theta_deg",
    "residual_phi_deg_wrapped",
    "objective_residual_two_theta_deg",
    "objective_residual_phi_deg",
    "target_delta_vs_cached_two_theta_deg",
    "target_delta_vs_cached_phi_deg_wrapped",
    "source_delta_vs_dynamic_two_theta_deg",
    "source_delta_vs_dynamic_phi_deg_wrapped",
    "frame_match",
    "unit_match",
    "branch_slot_match",
    "q_group_match",
    "hkl_match",
    "residual_contract_match",
)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _finite_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except Exception:
        return None
    if math.isfinite(x) and math.isfinite(y):
        return float(x), float(y)
    return None


def _cached_target(entry: Mapping[str, object]) -> tuple[float, float] | None:
    point, _source = _cached_target_with_source(entry)
    return point


def _cached_target_with_source(
    entry: Mapping[str, object],
) -> tuple[tuple[float, float] | None, str | None]:
    for x_key, y_key in (
        ("background_two_theta_deg", "background_phi_deg"),
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
    ):
        try:
            point = (float(entry.get(x_key, np.nan)), float(entry.get(y_key, np.nan)))
        except Exception:
            continue
        if math.isfinite(point[0]) and math.isfinite(point[1]):
            if bool(entry.get("fit_space_anchor_override", False)):
                source = "cached_fit_space_anchor"
            elif x_key == "background_two_theta_deg":
                source = "cached_background_caked_deg"
            elif x_key == "caked_x":
                source = "cached_caked_xy_deg"
            else:
                source = "cached_raw_caked_xy_deg"
            return point, source
    return None, None


def _delta(
    source: tuple[float, float] | None, target: tuple[float, float] | None
) -> dict[str, object]:
    if source is None or target is None:
        return {
            "delta_two_theta": None,
            "delta_phi_wrapped": None,
            "norm": None,
            "within_exact_tolerance": False,
        }
    dt = float(source[0] - target[0])
    dp = float(opt._angular_difference_deg(float(source[1]), float(target[1])))
    return {
        "delta_two_theta": dt,
        "delta_phi_wrapped": dp,
        "norm": float(math.hypot(dt, dp)),
        "within_exact_tolerance": bool(abs(dt) <= EXACT_TOL_DEG and abs(dp) <= EXACT_TOL_DEG),
    }


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return float(out) if math.isfinite(out) else None


def _pair_list(value: tuple[float, float] | None) -> list[float] | None:
    if value is None:
        return None
    return [float(value[0]), float(value[1])]


def _pair_from_fields(
    row: Mapping[str, object],
    theta_key: str,
    phi_key: str,
) -> tuple[float, float] | None:
    theta = _finite_float(row.get(theta_key))
    phi = _finite_float(row.get(phi_key))
    if theta is None or phi is None:
        return None
    return float(theta), float(phi)


def _normal_identity(value: object) -> tuple[object, ...] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return tuple((str(key), _normal_identity(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(
            _normal_identity(item) if isinstance(item, (list, tuple, Mapping)) else item
            for item in value
        )
    return (value,)


def _identity_match(first: object, second: object) -> bool:
    first_norm = _normal_identity(first)
    second_norm = _normal_identity(second)
    return bool(first_norm is not None and second_norm is not None and first_norm == second_norm)


def _normal_hkl(value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.strip("()[]").replace(";", ",").split(",")]
        parts = [part for part in parts if part]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts = list(value)
    else:
        return None
    if len(parts) != 3:
        return None
    try:
        return tuple(int(round(float(part))) for part in parts)  # type: ignore[return-value]
    except Exception:
        return None


def _nonnegative_index(value: object) -> int | None:
    try:
        out = int(value)
    except Exception:
        return None
    return out if out >= 0 else None


def _branch_slot_match(
    *,
    entry: Mapping[str, object],
    prediction: Mapping[str, object],
    fit_qr_branch_key: object,
    expected_slot: object,
    actual_slot: object,
) -> bool:
    expected_index = _nonnegative_index(expected_slot)
    if expected_index is None:
        expected_index = _nonnegative_index(entry.get("source_branch_index"))
    if expected_index is None:
        expected_index = _nonnegative_index(
            _fit_qr_branch_value(fit_qr_branch_key, "source_branch_index")
        )
    actual_index = _nonnegative_index(actual_slot)
    if actual_index is None:
        actual_index = _nonnegative_index(prediction.get("resolved_source_branch_index"))
    if actual_index is None:
        actual_index = _nonnegative_index(entry.get("source_branch_index"))
    if actual_index is None:
        actual_index = _nonnegative_index(
            _fit_qr_branch_value(fit_qr_branch_key, "source_branch_index")
        )
    if expected_index is not None and actual_index is not None:
        return bool(expected_index == actual_index)
    return _identity_match(expected_slot, actual_slot)


def _fit_qr_branch_value(
    branch_key: object,
    key: str,
) -> object | None:
    if isinstance(branch_key, Mapping):
        return branch_key.get(key)
    if isinstance(branch_key, Sequence) and not isinstance(branch_key, (str, bytes, bytearray)):
        index_by_key = {
            "q_group_key": 2,
            "hkl": 3,
            "physical_branch_slot": 4,
            "source_branch_index": 4,
            "source_peak_index": 5,
            "source_table_index": 6,
            "source_row_index": 7,
            "source_reflection_index": 8,
            "source_reflection_namespace": 9,
            "source_reflection_is_full": 10,
            "branch_id": 11,
            "best_sample_index": 12,
        }
        index = index_by_key.get(key)
        if index is not None and len(branch_key) > index:
            return branch_key[index]
    return None


def _manual_visual_frame(entry: Mapping[str, object]) -> str:
    frame = (
        entry.get("manual_visual_frame")
        or entry.get("fit_space_anchor_frame")
        or entry.get("projection_frame")
    )
    text = str(frame or "").strip()
    return text or "caked_display"


def _dynamic_source_from_prediction(
    prediction: Mapping[str, object],
) -> tuple[float, float] | None:
    point = _finite_pair(prediction.get("dynamic_baseline_anchor_caked_deg"))
    if point is not None:
        return point
    return _finite_pair(prediction.get("sim_visual_caked_deg"))


def _dynamic_source_label(prediction: Mapping[str, object]) -> str | None:
    for key in (
        "dynamic_baseline_anchor_actual_source",
        "dynamic_baseline_anchor_source_kind",
        "dynamic_baseline_anchor_source",
    ):
        value = str(prediction.get(key, "") or "").strip()
        if value:
            return value
    return None


def _optimizer_source_label(
    prediction: Mapping[str, object], source_delta: Mapping[str, object]
) -> str | None:
    if bool(source_delta.get("within_exact_tolerance")):
        dynamic_label = _dynamic_source_label(prediction)
        if dynamic_label:
            return dynamic_label
    for key in ("sim_refinement_caked_image_source", "sim_refinement_status"):
        value = str(prediction.get(key, "") or "").strip()
        if value:
            return value
    return None


def _objective_diag_residual(
    objective_diag: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if not isinstance(objective_diag, Mapping):
        return None
    point = _pair_from_fields(
        objective_diag,
        "delta_two_theta_deg",
        "wrapped_delta_phi_deg",
    )
    if point is not None:
        return point
    solver_vector = objective_diag.get("solver_residual_vector")
    return _finite_pair(solver_vector)


def _point_match(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
    *,
    phi_wrapped: bool = True,
) -> bool:
    delta = _delta(first, second) if phi_wrapped else None
    if delta is not None:
        return bool(delta.get("within_exact_tolerance", False))
    if first is None or second is None:
        return False
    return bool(
        abs(float(first[0]) - float(second[0])) <= EXACT_TOL_DEG
        and abs(float(first[1]) - float(second[1])) <= EXACT_TOL_DEG
    )


def _schema_missing_fields(row: Mapping[str, object]) -> list[str]:
    missing: list[str] = []
    for key in OBJECTIVE_AUDIT_REQUIRED_PAIR_FIELDS:
        if key not in row or row.get(key) is None:
            missing.append(key)
    return missing


def _build_objective_audit_pair_row(
    *,
    index: int,
    entry: Mapping[str, object],
    measured_anchor: tuple[float, float] | None,
    measured_reason: str | None,
    measured_meta: Mapping[str, object],
    prediction: Mapping[str, object],
    objective_diag: Mapping[str, object] | None,
) -> dict[str, object]:
    cached, cached_source = _cached_target_with_source(entry)
    dynamic_source = _dynamic_source_from_prediction(prediction)
    optimizer_source = _finite_pair(prediction.get("sim_refined_caked_deg"))
    target_delta = _delta(measured_anchor, cached)
    source_delta = _delta(optimizer_source, dynamic_source)
    residual_delta = _delta(optimizer_source, cached)
    objective_residual = _objective_diag_residual(objective_diag)
    q_group_key = entry.get("q_group_key")
    normalized_hkl = entry.get("normalized_hkl", entry.get("hkl"))
    fit_qr_branch_key = entry.get("fit_qr_branch_key", prediction.get("fit_qr_branch_key"))
    physical_branch_slot = entry.get(
        "physical_branch_slot",
        entry.get(
            "source_branch_index",
            _fit_qr_branch_value(fit_qr_branch_key, "physical_branch_slot"),
        ),
    )
    dynamic_frame = str(
        prediction.get(
            "dynamic_baseline_anchor_projection_frame",
            prediction.get("dynamic_baseline_anchor_frame", ""),
        )
        or ""
    )
    manual_frame = _manual_visual_frame(entry)
    dynamic_units = str(prediction.get("dynamic_baseline_anchor_units", "deg") or "deg").lower()
    expected_q_group = (
        q_group_key
        if q_group_key is not None
        else _fit_qr_branch_value(fit_qr_branch_key, "q_group_key")
    )
    actual_q_group = prediction.get(
        "dynamic_baseline_anchor_q_group_key",
        _fit_qr_branch_value(fit_qr_branch_key, "q_group_key"),
    )
    expected_hkl = _normal_hkl(normalized_hkl)
    actual_hkl = _normal_hkl(
        prediction.get(
            "dynamic_baseline_anchor_hkl",
            _fit_qr_branch_value(fit_qr_branch_key, "hkl"),
        )
    )
    expected_slot = physical_branch_slot
    actual_slot = prediction.get(
        "dynamic_baseline_anchor_physical_branch_slot",
        _fit_qr_branch_value(fit_qr_branch_key, "physical_branch_slot"),
    )
    objective_delta = _delta(objective_residual, (0.0, 0.0))
    residual_contract_match = bool(
        residual_delta.get("delta_two_theta") is not None
        and residual_delta.get("delta_phi_wrapped") is not None
        and objective_residual is not None
        and abs(float(residual_delta["delta_two_theta"]) - float(objective_residual[0]))
        <= EXACT_TOL_DEG
        and abs(
            opt._angular_difference_deg(
                float(residual_delta["delta_phi_wrapped"]),
                float(objective_residual[1]),
            )
        )
        <= EXACT_TOL_DEG
        and bool(target_delta.get("within_exact_tolerance"))
        and bool(source_delta.get("within_exact_tolerance"))
    )
    row = {
        "pair_index": int(index),
        "pair_id": reprojection._pair_id(entry, index),
        "q_group_key": q_group_key,
        "normalized_hkl": normalized_hkl,
        "physical_branch_slot": physical_branch_slot,
        "fit_qr_branch_key": fit_qr_branch_key,
        "manual_visual_x": float(cached[0]) if cached else None,
        "manual_visual_y": float(cached[1]) if cached else None,
        "manual_visual_frame": manual_frame,
        "cached_target_two_theta_deg": float(cached[0]) if cached else None,
        "cached_target_phi_deg": float(cached[1]) if cached else None,
        "cached_target_source": cached_source,
        "optimizer_measured_two_theta_deg": (
            float(measured_anchor[0]) if measured_anchor else None
        ),
        "optimizer_measured_phi_deg": float(measured_anchor[1]) if measured_anchor else None,
        "optimizer_measured_source": str(
            measured_meta.get("fit_space_source", measured_reason) or measured_reason or ""
        ),
        "dynamic_source_two_theta_deg": (float(dynamic_source[0]) if dynamic_source else None),
        "dynamic_source_phi_deg": float(dynamic_source[1]) if dynamic_source else None,
        "dynamic_source_source": _dynamic_source_label(prediction),
        "optimizer_source_two_theta_deg": (
            float(optimizer_source[0]) if optimizer_source else None
        ),
        "optimizer_source_phi_deg": float(optimizer_source[1]) if optimizer_source else None,
        "optimizer_source_source": _optimizer_source_label(prediction, source_delta),
        "residual_two_theta_deg": residual_delta["delta_two_theta"],
        "residual_phi_deg_wrapped": residual_delta["delta_phi_wrapped"],
        "residual_delta_two_theta": residual_delta["delta_two_theta"],
        "residual_delta_phi_wrapped": residual_delta["delta_phi_wrapped"],
        "objective_residual_two_theta_deg": (
            float(objective_residual[0]) if objective_residual else None
        ),
        "objective_residual_phi_deg": (
            float(objective_residual[1]) if objective_residual else None
        ),
        "target_delta_vs_cached_two_theta_deg": target_delta["delta_two_theta"],
        "target_delta_vs_cached_phi_deg_wrapped": target_delta["delta_phi_wrapped"],
        "source_delta_vs_dynamic_two_theta_deg": source_delta["delta_two_theta"],
        "source_delta_vs_dynamic_phi_deg_wrapped": source_delta["delta_phi_wrapped"],
        "frame_match": bool(manual_frame == "caked_display" and dynamic_frame == "caked_display"),
        "unit_match": bool(dynamic_units in {"deg", "degree", "degrees"}),
        "branch_slot_match": _branch_slot_match(
            entry=entry,
            prediction=prediction,
            fit_qr_branch_key=fit_qr_branch_key,
            expected_slot=expected_slot,
            actual_slot=actual_slot,
        ),
        "q_group_match": _identity_match(expected_q_group, actual_q_group),
        "hkl_match": bool(expected_hkl is not None and expected_hkl == actual_hkl),
        "residual_contract_match": residual_contract_match,
        "objective_residual_norm_deg": objective_delta["norm"],
        "schema_missing_fields": [],
        "cached_click_candidate_two_theta_phi": _pair_list(cached),
        "optimizer_measured_anchor_two_theta_phi": _pair_list(measured_anchor),
        "dynamic_sim_visual_caked_deg_two_theta_phi": _pair_list(dynamic_source),
        "optimizer_simulated_source_two_theta_phi": _pair_list(optimizer_source),
        "target_minus_cached_delta": target_delta,
        "source_minus_dynamic_delta": source_delta,
        "measured_anchor_source": measured_reason,
        "measured_fit_space_source": measured_meta.get("fit_space_source"),
        "prediction_available": bool(prediction.get("available", False)),
        "prediction_unavailable_reason": prediction.get("unavailable_reason"),
        "sim_refinement_status": prediction.get("sim_refinement_status"),
    }
    row["schema_missing_fields"] = _schema_missing_fields(row)
    return row


def _center_from_params(params: Mapping[str, object]) -> list[float]:
    center = params.get("center")
    if isinstance(center, Sequence) and len(center) >= 2:
        try:
            return [float(center[0]), float(center[1])]
        except Exception:
            pass
    return [
        float(params.get("center_x", np.nan)),
        float(params.get("center_y", np.nan)),
    ]


def _apply_perturb(
    params: Mapping[str, object], perturb: str | None
) -> tuple[dict[str, object], dict[str, object]]:
    out = dict(params)
    if not perturb:
        return out, {"applied": False}
    if "=" not in str(perturb):
        raise ValueError("--perturb must be NAME=DELTA")
    name, raw_delta = str(perturb).split("=", 1)
    name = name.strip()
    delta = float(raw_delta)
    base = float(out.get(name, 0.0) or 0.0)
    out[name] = float(base + delta)
    if name in {"center_x", "center_y"}:
        center = _center_from_params(out)
        if name == "center_x":
            center[0] = float(out[name])
        else:
            center[1] = float(out[name])
        out["center"] = center
    return out, {
        "applied": True,
        "name": name,
        "delta": float(delta),
        "base": base,
        "value": float(out[name]),
    }


def _step_perturbation(
    *,
    perturb: str | None,
    step_param: str | None,
    step_size: float | None,
) -> str | None:
    if perturb and (step_param is not None or step_size is not None):
        raise ValueError("--perturb cannot be combined with --step-param/--step-size")
    if perturb:
        return str(perturb)
    if step_param is None and step_size is None:
        return None
    if step_param is None or step_size is None:
        raise ValueError("--step-param and --step-size must be provided together")
    return f"{step_param}={float(step_size)}"


def _solver_cfg(request) -> Mapping[str, object]:
    cfg = getattr(request, "refinement_config", {}) or {}
    if not isinstance(cfg, Mapping):
        return {}
    solver = cfg.get("solver", cfg.get("optimizer", {}))
    return solver if isinstance(solver, Mapping) else {}


def _objective_local_params(request, params: Mapping[str, object]) -> dict[str, object]:
    local = dict(params)
    solver = _solver_cfg(request)
    local["_qr_fit_point_only_projection"] = bool(solver.get("_qr_fit_point_only_projection", True))
    if "q_group_line_constraints_enabled" in solver or "q_group_line_constraints" in solver:
        local["_q_group_line_constraints_enabled"] = bool(
            solver.get(
                "q_group_line_constraints_enabled",
                solver.get("q_group_line_constraints", False),
            )
        )
    return local


def _missing_pair_penalty_deg(request) -> float:
    solver = _solver_cfg(request)
    penalty = _finite_float(solver.get("missing_pair_penalty_deg", 5.0))
    if penalty is None or penalty < 0.0:
        return 5.0
    return float(penalty)


def _objective_diagnostics_by_pair(
    diagnostics: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Mapping[str, object]], dict[int, Mapping[str, object]]]:
    by_pair: dict[str, Mapping[str, object]] = {}
    by_index: dict[int, Mapping[str, object]] = {}
    for index, diag in enumerate(diagnostics):
        if not isinstance(diag, Mapping):
            continue
        pair_id = str(diag.get("manual_pair_id", diag.get("pair_id", "")) or "")
        if pair_id:
            by_pair[pair_id] = diag
        by_index[index] = diag
    return by_pair, by_index


def _build_dataset_context(request, params: Mapping[str, object]):
    contexts = opt._build_geometry_fit_dataset_contexts(
        np.asarray(request.miller, dtype=np.float64),
        np.asarray(request.intensities, dtype=np.float64),
        dict(params),
        request.measured_peaks,
        None,
        request.dataset_specs,
    )
    if not contexts:
        raise RuntimeError("geometry fit dataset context unavailable")
    return contexts[0]


def _evaluate_pairs(request, params: Mapping[str, object]) -> list[dict[str, object]]:
    local_params = _objective_local_params(request, params)
    dataset_ctx = _build_dataset_context(request, local_params)
    objective_residual, objective_diagnostics, objective_summary = (
        opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
            local_params,
            dataset_ctx,
            image_size=int(request.image_size),
            missing_pair_penalty_deg=_missing_pair_penalty_deg(request),
            theta_value=float(local_params.get("theta_initial", dataset_ctx.theta_initial)),
            collect_diagnostics=True,
        )
    )
    objective_by_pair, objective_by_index = _objective_diagnostics_by_pair(objective_diagnostics)
    pixel_size = float(opt._fit_space_pixel_size_provenance(local_params).get("value", np.nan))
    detector_distance = float(local_params.get("corto_detector", np.nan))
    gamma_deg = float(local_params.get("gamma", 0.0) or 0.0)
    Gamma_deg = float(local_params.get("Gamma", 0.0) or 0.0)
    image_size = int(request.image_size)
    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(dataset_ctx.subset.measured_entries):
        if not isinstance(entry, Mapping):
            continue
        measured_anchor, measured_reason, measured_meta = opt._measured_fit_space_anchor(
            entry,
            center=_center_from_params(local_params),
            detector_distance=detector_distance,
            pixel_size=pixel_size,
            gamma_deg=gamma_deg,
            Gamma_deg=Gamma_deg,
            a_lattice=float(local_params.get("a", np.nan)),
            c_lattice=float(local_params.get("c", np.nan)),
            wavelength=float(local_params.get("lambda", np.nan)),
            dataset_ctx=dataset_ctx,
            local_params=local_params,
        )
        prediction = opt._resolve_qr_fit_prediction_from_trial_params(
            entry,
            local_params,
            {
                "dataset_ctx": dataset_ctx,
                "hit_tables": (),
                "sim_buffer": sim_buffer,
                "image_size": image_size,
                "fit_center": _center_from_params(local_params),
                "detector_distance": detector_distance,
                "pixel_size": pixel_size,
                "gamma_deg": gamma_deg,
                "Gamma_deg": Gamma_deg,
                "prediction_source_rows_cache": {},
                "_qr_fit_point_only_projection": True,
            },
            entry,
        )
        pair_id = reprojection._pair_id(entry, index)
        objective_diag = objective_by_pair.get(str(pair_id), objective_by_index.get(index))
        row = _build_objective_audit_pair_row(
            index=index,
            entry=entry,
            measured_anchor=measured_anchor,
            measured_reason=str(measured_reason),
            measured_meta=measured_meta,
            prediction=prediction,
            objective_diag=objective_diag,
        )
        row["objective_summary_metric_name"] = objective_summary.get("metric_name")
        row["objective_summary_metric_unit"] = objective_summary.get("metric_unit")
        row["objective_residual_vector_component_count"] = int(
            np.asarray(objective_residual, dtype=float).reshape(-1).size
        )
        rows.append(row)
    return rows


def _pair_checks(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    measured_ok = all(
        isinstance(row.get("target_minus_cached_delta"), Mapping)
        and bool(row["target_minus_cached_delta"].get("within_exact_tolerance"))
        and str(row.get("measured_fit_space_source")) == "cached_fit_space_anchor"
        for row in rows
    )
    source_ok = all(
        isinstance(row.get("source_minus_dynamic_delta"), Mapping)
        and bool(row["source_minus_dynamic_delta"].get("within_exact_tolerance"))
        for row in rows
    )
    residual_ok = all(
        row.get("residual_delta_two_theta") is not None
        and row.get("residual_delta_phi_wrapped") is not None
        for row in rows
    )
    schema_ok = all(not _schema_missing_fields(row) for row in rows)
    required_row_count = int(len(rows))
    return {
        "objective_audit_schema_version": int(AUDIT_SCHEMA_VERSION),
        "objective_audit_required_fields": list(OBJECTIVE_AUDIT_REQUIRED_PAIR_FIELDS),
        "objective_audit_schema_machine_checkable": bool(schema_ok),
        "objective_audit_pair_count_is_7": bool(required_row_count == 7),
        "optimizer_measured_target_equals_cached_target": bool(measured_ok),
        "optimizer_source_equals_dynamic_source": bool(source_ok),
        "residual_vector_machine_checkable": bool(residual_ok),
        "all_frame_match": bool(rows and all(bool(row.get("frame_match")) for row in rows)),
        "all_unit_match": bool(rows and all(bool(row.get("unit_match")) for row in rows)),
        "all_branch_slot_match": bool(
            rows and all(bool(row.get("branch_slot_match")) for row in rows)
        ),
        "all_q_group_match": bool(rows and all(bool(row.get("q_group_match")) for row in rows)),
        "all_hkl_match": bool(rows and all(bool(row.get("hkl_match")) for row in rows)),
        "all_residual_contract_match": bool(
            rows and all(bool(row.get("residual_contract_match")) for row in rows)
        ),
    }


def _cross_checks(
    base_rows: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    perturb_applied: bool,
) -> dict[str, object]:
    target_fixed: list[bool] = []
    source_shift_norms: list[float] = []
    for base, row in zip(base_rows, rows):
        base_target = _finite_pair(base.get("optimizer_measured_anchor_two_theta_phi"))
        target = _finite_pair(row.get("optimizer_measured_anchor_two_theta_phi"))
        base_source = _finite_pair(base.get("optimizer_simulated_source_two_theta_phi"))
        source = _finite_pair(row.get("optimizer_simulated_source_two_theta_phi"))
        target_fixed.append(bool(_delta(target, base_target)["within_exact_tolerance"]))
        source_shift = _delta(source, base_source)
        norm = source_shift.get("norm")
        if isinstance(norm, (int, float)) and math.isfinite(float(norm)):
            source_shift_norms.append(float(norm))
    return {
        "target_unchanged_under_perturbation": bool(target_fixed and all(target_fixed)),
        "source_shift_norms": source_shift_norms,
        "source_moves_under_perturbation": (
            bool(any(value > EXACT_TOL_DEG for value in source_shift_norms))
            if perturb_applied
            else None
        ),
    }


def _plot_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    series = (
        ("cached target", "cached_click_candidate_two_theta_phi", "o"),
        ("optimizer measured", "optimizer_measured_anchor_two_theta_phi", "x"),
        ("dynamic source", "dynamic_sim_visual_caked_deg_two_theta_phi", "^"),
        ("optimizer source", "optimizer_simulated_source_two_theta_phi", "+"),
    )
    for label, key, marker in series:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            point = _finite_pair(row.get(key))
            if point is not None:
                xs.append(point[0])
                ys.append(point[1])
        if xs:
            ax.scatter(xs, ys, label=label, marker=marker, s=55)
    for row in rows:
        target = _finite_pair(row.get("cached_click_candidate_two_theta_phi"))
        source = _finite_pair(row.get("optimizer_simulated_source_two_theta_phi"))
        if target is not None and source is not None:
            ax.annotate(
                "",
                xy=target,
                xytext=source,
                arrowprops={"arrowstyle": "->", "linewidth": 0.8, "alpha": 0.45},
            )
            ax.text(source[0], source[1], str(row.get("pair_index")), fontsize=8)
    ax.set_xlabel("two_theta_deg")
    ax.set_ylabel("phi_deg")
    ax.set_title("New4 Qr Fit Coordinate Audit")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_coordinate_audit(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    params_mode: str = "base",
    perturb: str | None = None,
    objective_audit: bool = False,
    after_objective_step: bool = False,
    step_param: str | None = None,
    step_size: float | None = None,
) -> dict[str, object]:
    if str(params_mode).strip().lower() != "base":
        raise ValueError("only --params base is supported")
    state_path = state_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    effective_perturb = _step_perturbation(
        perturb=perturb,
        step_param=step_param,
        step_size=step_size,
    )
    provider_report = preflight._run_point_provider_report_only(state_path, int(background_index))
    context = ladder._capture_solver_context(state_path, int(background_index))
    request = ladder.build_solver_request(context, ["center_x"], max_nfev=1)
    prepared_run = context.get("prepared_run")
    prepared_params = (
        getattr(prepared_run, "fit_params", None) if prepared_run is not None else None
    )
    base_params = dict(getattr(request, "params", {}) or {})
    base_params.update(dict(prepared_params or {}))
    current_params, perturb_info = _apply_perturb(base_params, effective_perturb)
    base_rows = _evaluate_pairs(request, base_params)
    rows = _evaluate_pairs(request, current_params)
    checks = _pair_checks(rows)
    checks.update(_cross_checks(base_rows, rows, bool(perturb_info.get("applied", False))))
    status = (
        "pass"
        if all(
            bool(value)
            for key, value in checks.items()
            if key != "source_moves_under_perturbation" or bool(perturb_info.get("applied", False))
        )
        else "fail"
    )
    if (
        not bool(perturb_info.get("applied", False))
        and checks.get("source_moves_under_perturbation") is None
    ):
        status = (
            "pass"
            if all(
                bool(value)
                for key, value in checks.items()
                if key != "source_moves_under_perturbation"
            )
            else "fail"
        )
    report_path = output_root / REPORT_NAME
    plot_path = output_root / PLOT_NAME
    report = {
        "status": status,
        "checks": checks,
        "state_path": str(state_path),
        "background_index": int(background_index),
        "params_mode": "base",
        "audit_schema_version": int(AUDIT_SCHEMA_VERSION),
        "audit_mode": "objective_coordinate_contract",
        "objective_audit": bool(objective_audit),
        "after_objective_step": bool(after_objective_step),
        "step_param": step_param,
        "step_size": step_size,
        "perturb": perturb_info,
        "provider_pair_count": reprojection._provider_pair_count(provider_report),
        "pair_count": int(len(rows)),
        "exact_tolerance_deg": EXACT_TOL_DEG,
        "base_pairs": base_rows,
        "pairs": rows,
        "json_path": str(report_path),
        "png_path": str(plot_path),
        "created_at_unix": time.time(),
    }
    _write_json(report_path, report)
    _plot_rows(plot_path, rows)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize New4 Qr fit coordinate contract.")
    parser.add_argument("--state", "--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--background-index", type=int, default=0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--params", default="base")
    parser.add_argument("--perturb", default=None)
    parser.add_argument("--objective-audit", action="store_true")
    parser.add_argument("--after-objective-step", action="store_true")
    parser.add_argument("--step-param", default=None)
    parser.add_argument("--step-size", type=float, default=None)
    args = parser.parse_args(argv)
    report = run_coordinate_audit(
        state_path=Path(args.state),
        background_index=int(args.background_index),
        output_root=Path(args.output_root),
        params_mode=str(args.params),
        perturb=args.perturb,
        objective_audit=bool(args.objective_audit),
        after_objective_step=bool(args.after_objective_step),
        step_param=args.step_param,
        step_size=args.step_size,
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "status": report["status"],
                    "json_path": report["json_path"],
                    "png_path": report["png_path"],
                }
            ),
            sort_keys=True,
        )
    )
    return 0 if str(report.get("status")) == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
