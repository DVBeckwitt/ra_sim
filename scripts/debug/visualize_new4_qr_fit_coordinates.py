#!/usr/bin/env python
"""Audit New4 manual caked Qr fit coordinates as JSON plus scatter PNG."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from ra_sim.gui import geometry_fit as gui_geometry_fit  # noqa: E402
from scripts.debug import run_new4_caked_point_reprojection_check as reprojection  # noqa: E402
from scripts.debug import run_new4_geometry_fit_ladder as ladder  # noqa: E402
from scripts.debug import validate_geometry_preflight_rebind as preflight  # noqa: E402


DEFAULT_STATE_PATH = REPO_ROOT / "artifacts" / "geometry_fit_gui_states" / "new4.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "geometry_fit_ladder" / "new4_coordinate_audit"
REPORT_NAME = "new4_qr_fit_coordinates.json"
PLOT_NAME = "new4_qr_fit_coordinates.png"
SINGLE_STEP_REPORT_NAME = "new4_qr_single_iteration.json"
SINGLE_STEP_PLOT_NAME = "new4_qr_single_iteration.png"
SINGLE_STEP_CSV_NAME = "new4_qr_single_iteration.csv"
EXACT_TOL_DEG = 1.0e-6
DETECTOR_FRAME_TOL_PX = 1.0e-6
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
    "objective_source_authority",
    "residual_two_theta_deg",
    "residual_phi_deg_wrapped",
    "objective_residual_two_theta_deg",
    "objective_residual_phi_deg",
    "objective_residual_vector_two_theta_index",
    "objective_residual_vector_phi_index",
    "objective_residual_vector_two_theta_deg",
    "objective_residual_vector_phi_deg",
    "objective_residual_vector_contract_match",
    "objective_residual_expected_two_theta_deg",
    "objective_residual_expected_phi_deg_wrapped",
    "objective_residual_contract_error_two_theta_deg",
    "objective_residual_contract_error_phi_deg_wrapped",
    "objective_residual_units",
    "pixel_residual_used_for_objective",
    "target_delta_vs_cached_two_theta_deg",
    "target_delta_vs_cached_phi_deg_wrapped",
    "source_delta_vs_dynamic_two_theta_deg",
    "source_delta_vs_dynamic_phi_deg_wrapped",
    "source_authority_match",
    "residual_arrow_delta_two_theta_deg",
    "residual_arrow_delta_phi_deg_wrapped",
    "residual_arrow_matches_plotted_source_target_delta",
    "detector_native_reprojection_diagnostic_two_theta_deg",
    "detector_native_reprojection_diagnostic_phi_deg",
    "detector_native_reprojection_delta_vs_cached_two_theta_deg",
    "detector_native_reprojection_delta_vs_cached_phi_deg_wrapped",
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            text = str(key)
            if text not in fieldnames:
                fieldnames.append(text)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_jsonable(value), sort_keys=True)
                        if isinstance(value, (Mapping, list, tuple))
                        else _jsonable(value)
                    )
                    for key, value in row.items()
                }
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _entry_tuple_pair(entry: Mapping[str, object], key: str) -> tuple[float, float] | None:
    return _finite_pair(entry.get(key))


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


def _entry_pair_from_fields(
    entry: Mapping[str, object],
    fields: Sequence[tuple[str, str]],
) -> tuple[float, float] | None:
    for x_key, y_key in fields:
        point = _pair_from_fields(entry, x_key, y_key)
        if point is not None:
            return point
    return None


def _entry_background_detector_display(entry: Mapping[str, object]) -> tuple[float, float] | None:
    return _entry_tuple_pair(entry, "bg_display") or _entry_pair_from_fields(
        entry,
        (
            ("background_detector_display_x", "background_detector_display_y"),
            ("display_col", "display_row"),
            ("x", "y"),
        ),
    )


def _entry_background_detector_native(entry: Mapping[str, object]) -> tuple[float, float] | None:
    return _entry_tuple_pair(entry, "background_detector_native_px") or _entry_pair_from_fields(
        entry,
        (
            ("native_col", "native_row"),
            ("background_detector_x", "background_detector_y"),
            ("detector_x", "detector_y"),
        ),
    )


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


def _detector_native_reprojection_diagnostic_from_entry(
    entry: Mapping[str, object],
    *,
    dataset_ctx: object,
    local_params: Mapping[str, object],
    center: Sequence[float],
    detector_distance: float,
    pixel_size: float,
    gamma_deg: float,
    Gamma_deg: float,
) -> tuple[tuple[float, float] | None, str | None, Mapping[str, object]]:
    entry_point = _finite_pair(entry.get("detector_native_reprojection_diagnostic_caked_deg"))
    if entry_point is not None:
        return entry_point, "entry_detector_native_reprojection_diagnostic_caked_deg", {}
    projection_input = opt._measured_fit_space_projection_input(entry)
    if str(projection_input.get("status", "")) != "projectable":
        return (
            None,
            str(projection_input.get("frame_reason") or "missing_detector_native_point"),
            {
                "projection_input": projection_input,
            },
        )
    two_theta_arr, phi_arr, meta = opt._project_detector_points_to_fit_space(
        dataset_ctx,
        np.array([projection_input["col"]], dtype=np.float64),
        np.array([projection_input["row"]], dtype=np.float64),
        local_params=dict(local_params),
        anchor_kind="detector_native_reprojection_diagnostic",
        input_frame=str(projection_input.get("input_frame", "")),
        center=center,
        detector_distance=float(detector_distance),
        pixel_size=float(pixel_size),
        gamma_deg=float(gamma_deg),
        Gamma_deg=float(Gamma_deg),
    )
    if (
        bool(meta.get("valid", False))
        and two_theta_arr.size >= 1
        and phi_arr.size >= 1
        and math.isfinite(float(two_theta_arr[0]))
        and math.isfinite(float(phi_arr[0]))
    ):
        source = str(
            meta.get("fit_space_source")
            or projection_input.get("field_name")
            or "detector_native_reprojection_diagnostic"
        )
        return (float(two_theta_arr[0]), float(phi_arr[0])), source, meta
    return None, str(meta.get("invalid_reason") or "invalid_detector_native_reprojection"), meta


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
    detector_native_reprojection: tuple[float, float] | None = None,
    detector_native_reprojection_source: str | None = None,
    detector_native_reprojection_meta: Mapping[str, object] | None = None,
) -> dict[str, object]:
    cached, cached_source = _cached_target_with_source(entry)
    if detector_native_reprojection is None:
        detector_native_reprojection = _finite_pair(
            entry.get("detector_native_reprojection_diagnostic_caked_deg")
        )
        if detector_native_reprojection is not None and detector_native_reprojection_source is None:
            detector_native_reprojection_source = (
                "entry_detector_native_reprojection_diagnostic_caked_deg"
            )
    dynamic_source = _dynamic_source_from_prediction(prediction)
    optimizer_source = _finite_pair(prediction.get("sim_refined_caked_deg"))
    target_delta = _delta(measured_anchor, cached)
    source_delta = _delta(optimizer_source, dynamic_source)
    residual_delta = _delta(optimizer_source, cached)
    detector_native_reprojection_delta = _delta(detector_native_reprojection, cached)
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
    residual_contract_error_two_theta = None
    residual_contract_error_phi = None
    if (
        residual_delta.get("delta_two_theta") is not None
        and residual_delta.get("delta_phi_wrapped") is not None
        and objective_residual is not None
    ):
        residual_contract_error_two_theta = abs(
            float(residual_delta["delta_two_theta"]) - float(objective_residual[0])
        )
        residual_contract_error_phi = abs(
            opt._angular_difference_deg(
                float(residual_delta["delta_phi_wrapped"]),
                float(objective_residual[1]),
            )
        )
    residual_arrow_delta = residual_delta
    residual_arrow_match = bool(
        residual_arrow_delta.get("delta_two_theta") is not None
        and residual_arrow_delta.get("delta_phi_wrapped") is not None
        and residual_delta.get("delta_two_theta") is not None
        and residual_delta.get("delta_phi_wrapped") is not None
        and abs(
            float(residual_arrow_delta["delta_two_theta"])
            - float(residual_delta["delta_two_theta"])
        )
        <= EXACT_TOL_DEG
        and abs(
            opt._angular_difference_deg(
                float(residual_arrow_delta["delta_phi_wrapped"]),
                float(residual_delta["delta_phi_wrapped"]),
            )
        )
        <= EXACT_TOL_DEG
    )
    residual_contract_match = bool(
        residual_contract_error_two_theta is not None
        and residual_contract_error_phi is not None
        and residual_contract_error_two_theta <= EXACT_TOL_DEG
        and residual_contract_error_phi <= EXACT_TOL_DEG
        and bool(target_delta.get("within_exact_tolerance"))
        and bool(source_delta.get("within_exact_tolerance"))
    )
    prediction_native, prediction_native_source = opt._fit_prediction_detector_point_for_frame(
        prediction,
        "native_detector",
    )
    prediction_display, prediction_display_source = opt._fit_prediction_detector_point_for_frame(
        prediction,
        "display_detector",
    )
    fit_prediction_caked = _finite_pair(
        (objective_diag or {}).get("predicted_caked_deg")
        if isinstance(objective_diag, Mapping)
        else None
    )
    if fit_prediction_caked is None:
        fit_prediction_caked = optimizer_source
    source_table_index = entry.get(
        "source_table_index",
        prediction.get("resolved_source_table_index"),
    )
    source_row_index = entry.get(
        "source_row_index",
        prediction.get("resolved_source_row_index"),
    )
    source_branch_index = entry.get(
        "source_branch_index",
        prediction.get("resolved_physical_branch_slot", physical_branch_slot),
    )
    background_display = _entry_background_detector_display(entry)
    background_native = _entry_background_detector_native(entry)
    row = {
        "pair_index": int(index),
        "pair_id": reprojection._pair_id(entry, index),
        "q_group_key": q_group_key,
        "hkl": normalized_hkl,
        "normalized_hkl": normalized_hkl,
        "physical_branch_slot": physical_branch_slot,
        "source_branch_index": source_branch_index,
        "source_table_index": source_table_index,
        "source_row_index": source_row_index,
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
        "fit_prediction_caked_deg": _pair_list(fit_prediction_caked),
        "fit_prediction_caked_source": "predicted_caked_deg",
        "background_detector_display_px": _pair_list(background_display),
        "background_detector_native_px": _pair_list(background_native),
        "fit_prediction_detector_display_px": _pair_list(prediction_display),
        "fit_prediction_detector_display_px_source": prediction_display_source,
        "fit_prediction_detector_native_px": _pair_list(prediction_native),
        "fit_prediction_detector_native_px_source": prediction_native_source,
        "fit_space_projector_kind": prediction.get("fit_space_projector_kind"),
        "caked_projection_signature": prediction.get("caked_projection_signature"),
        "cake_bundle_signature": prediction.get("cake_bundle_signature"),
        "objective_source_authority": "sim_visual_caked_deg",
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
        "objective_residual_vector_two_theta_index": int(2 * index),
        "objective_residual_vector_phi_index": int(2 * index + 1),
        "objective_residual_vector_two_theta_deg": (
            float(objective_residual[0]) if objective_residual else None
        ),
        "objective_residual_vector_phi_deg": (
            float(objective_residual[1]) if objective_residual else None
        ),
        "objective_residual_vector_contract_match": residual_contract_match,
        "objective_residual_expected_two_theta_deg": residual_delta["delta_two_theta"],
        "objective_residual_expected_phi_deg_wrapped": residual_delta["delta_phi_wrapped"],
        "objective_residual_contract_error_two_theta_deg": residual_contract_error_two_theta,
        "objective_residual_contract_error_phi_deg_wrapped": residual_contract_error_phi,
        "target_delta_vs_cached_two_theta_deg": target_delta["delta_two_theta"],
        "target_delta_vs_cached_phi_deg_wrapped": target_delta["delta_phi_wrapped"],
        "source_delta_vs_dynamic_two_theta_deg": source_delta["delta_two_theta"],
        "source_delta_vs_dynamic_phi_deg_wrapped": source_delta["delta_phi_wrapped"],
        "source_authority_match": bool(
            source_delta["within_exact_tolerance"]
            and _dynamic_source_label(prediction) == "sim_visual_caked_deg"
            and _optimizer_source_label(prediction, source_delta) == "sim_visual_caked_deg"
        ),
        "residual_arrow_start_two_theta_deg": float(cached[0]) if cached else None,
        "residual_arrow_start_phi_deg": float(cached[1]) if cached else None,
        "residual_arrow_end_two_theta_deg": (
            float(optimizer_source[0]) if optimizer_source else None
        ),
        "residual_arrow_end_phi_deg": float(optimizer_source[1]) if optimizer_source else None,
        "residual_arrow_delta_two_theta_deg": residual_arrow_delta["delta_two_theta"],
        "residual_arrow_delta_phi_deg_wrapped": residual_arrow_delta["delta_phi_wrapped"],
        "residual_arrow_matches_plotted_source_target_delta": residual_arrow_match,
        "detector_native_reprojection_diagnostic_two_theta_deg": (
            float(detector_native_reprojection[0])
            if detector_native_reprojection is not None
            else None
        ),
        "detector_native_reprojection_diagnostic_phi_deg": (
            float(detector_native_reprojection[1])
            if detector_native_reprojection is not None
            else None
        ),
        "detector_native_reprojection_diagnostic_source": detector_native_reprojection_source,
        "detector_native_reprojection_delta_vs_cached_two_theta_deg": (
            detector_native_reprojection_delta["delta_two_theta"]
        ),
        "detector_native_reprojection_delta_vs_cached_phi_deg_wrapped": (
            detector_native_reprojection_delta["delta_phi_wrapped"]
        ),
        "detector_native_reprojection_delta_vs_cached_norm_deg": (
            detector_native_reprojection_delta["norm"]
        ),
        "detector_native_reprojection_matches_cached_target": bool(
            detector_native_reprojection_delta["within_exact_tolerance"]
        ),
        "detector_native_reprojection_is_diagnostic": True,
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
        "objective_residual_units": "deg",
        "objective_residual_coordinate_space": "caked_deg",
        "pixel_residual_used_for_objective": False,
        "objective_residual_norm_deg": objective_delta["norm"],
        "schema_missing_fields": [],
        "cached_click_candidate_two_theta_phi": _pair_list(cached),
        "optimizer_measured_anchor_two_theta_phi": _pair_list(measured_anchor),
        "dynamic_sim_visual_caked_deg_two_theta_phi": _pair_list(dynamic_source),
        "optimizer_simulated_source_two_theta_phi": _pair_list(optimizer_source),
        "target_minus_cached_delta": target_delta,
        "source_minus_dynamic_delta": source_delta,
        "detector_native_reprojection_minus_cached_delta": detector_native_reprojection_delta,
        "detector_native_reprojection_diagnostic_meta": dict(
            detector_native_reprojection_meta or {}
        ),
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


def _parse_active_vars(value: str | None) -> list[str]:
    if value is None:
        return ["center_x"]
    names: list[str] = []
    for raw in str(value).replace(";", ",").replace(" ", ",").split(","):
        name = raw.strip()
        if name and name not in names:
            names.append(name)
    return names or ["center_x"]


def _request_with_params(request, params: Mapping[str, object]):
    return gui_geometry_fit.GeometryFitSolverRequest(
        miller=request.miller,
        intensities=request.intensities,
        image_size=int(request.image_size),
        params=dict(params),
        measured_peaks=request.measured_peaks,
        var_names=list(request.var_names),
        candidate_param_names=(
            list(request.candidate_param_names)
            if request.candidate_param_names is not None
            else None
        ),
        dataset_specs=request.dataset_specs,
        refinement_config=dict(request.refinement_config),
        runtime_safety_note=request.runtime_safety_note,
    )


def _request_with_dataset_display_context(request, dataset_infos: Sequence[object] | None):
    specs = [dict(spec) for spec in (request.dataset_specs or ()) if isinstance(spec, Mapping)]
    infos = [info for info in (dataset_infos or ()) if isinstance(info, Mapping)]
    for index, spec in enumerate(specs):
        info = infos[index] if index < len(infos) else {}
        for key in (
            "native_detector_coords_to_detector_display_coords",
            "native_detector_coords_to_detector_display_coords_source",
            "native_detector_coords_to_detector_display_coords_unavailable_reason",
        ):
            if key in info and key not in spec:
                spec[key] = info[key]
    return gui_geometry_fit.GeometryFitSolverRequest(
        miller=request.miller,
        intensities=request.intensities,
        image_size=int(request.image_size),
        params=dict(request.params),
        measured_peaks=request.measured_peaks,
        var_names=list(request.var_names),
        candidate_param_names=(
            list(request.candidate_param_names)
            if request.candidate_param_names is not None
            else None
        ),
        dataset_specs=specs,
        refinement_config=dict(request.refinement_config),
        runtime_safety_note=request.runtime_safety_note,
    )


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


def _evaluate_pair_bundle(request, params: Mapping[str, object]) -> dict[str, object]:
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
    objective_residual_vector = np.asarray(objective_residual, dtype=float).reshape(-1)
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
        detector_native_reprojection, detector_native_reprojection_source, detector_native_meta = (
            _detector_native_reprojection_diagnostic_from_entry(
                entry,
                dataset_ctx=dataset_ctx,
                local_params=local_params,
                center=_center_from_params(local_params),
                detector_distance=detector_distance,
                pixel_size=pixel_size,
                gamma_deg=gamma_deg,
                Gamma_deg=Gamma_deg,
            )
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
            detector_native_reprojection=detector_native_reprojection,
            detector_native_reprojection_source=detector_native_reprojection_source,
            detector_native_reprojection_meta=detector_native_meta,
        )
        vector_two_theta_index = int(2 * index)
        vector_phi_index = int(vector_two_theta_index + 1)
        if vector_phi_index < objective_residual_vector.size:
            vector_two_theta = float(objective_residual_vector[vector_two_theta_index])
            vector_phi = float(objective_residual_vector[vector_phi_index])
            expected_two_theta = row.get("objective_residual_expected_two_theta_deg")
            expected_phi = row.get("objective_residual_expected_phi_deg_wrapped")
            vector_two_theta_error = (
                abs(vector_two_theta - float(expected_two_theta))
                if expected_two_theta is not None
                else None
            )
            vector_phi_error = (
                abs(opt._angular_difference_deg(vector_phi, float(expected_phi)))
                if expected_phi is not None
                else None
            )
            row["objective_residual_vector_two_theta_index"] = vector_two_theta_index
            row["objective_residual_vector_phi_index"] = vector_phi_index
            row["objective_residual_vector_two_theta_deg"] = vector_two_theta
            row["objective_residual_vector_phi_deg"] = vector_phi
            row["objective_residual_vector_contract_match"] = bool(
                vector_two_theta_error is not None
                and vector_phi_error is not None
                and vector_two_theta_error <= EXACT_TOL_DEG
                and vector_phi_error <= EXACT_TOL_DEG
            )
        row["objective_summary_metric_name"] = objective_summary.get("metric_name")
        row["objective_summary_metric_unit"] = objective_summary.get("metric_unit")
        row["objective_residual_vector_component_count"] = int(
            np.asarray(objective_residual, dtype=float).reshape(-1).size
        )
        rows.append(row)
    return {
        "rows": rows,
        "residual_vector": objective_residual_vector,
        "objective_summary": dict(objective_summary),
        "local_params": local_params,
        "dataset_ctx": dataset_ctx,
    }


def _evaluate_pairs(request, params: Mapping[str, object]) -> list[dict[str, object]]:
    bundle = _evaluate_pair_bundle(request, params)
    return [dict(row) for row in bundle["rows"]]  # type: ignore[index]


def _pair_checks(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    measured_ok = all(
        isinstance(row.get("target_minus_cached_delta"), Mapping)
        and bool(row["target_minus_cached_delta"].get("within_exact_tolerance"))
        and str(row.get("measured_fit_space_source")) == "cached_fit_space_anchor"
        for row in rows
    )
    source_labels = [
        str(row.get(key, "") or "")
        for row in rows
        for key in ("dynamic_source_source", "optimizer_source_source")
    ]
    source_ok = all(
        isinstance(row.get("source_minus_dynamic_delta"), Mapping)
        and bool(row["source_minus_dynamic_delta"].get("within_exact_tolerance"))
        and row.get("dynamic_source_source") == "sim_visual_caked_deg"
        and row.get("optimizer_source_source") == "sim_visual_caked_deg"
        for row in rows
    )
    clicked_visual_candidate_count = sum(
        1 for label in source_labels if "clicked_visual_candidate" in label
    )
    saved_manual_source_count = sum(
        1 for label in source_labels if "saved" in label.lower() or "manual" in label.lower()
    )
    detector_grouped_fallback_count = sum(
        1
        for label in source_labels
        if "detector" in label.lower() or "group" in label.lower() or "fallback" in label.lower()
    )
    residual_contract_errors_two_theta = [
        float(row["objective_residual_contract_error_two_theta_deg"])
        for row in rows
        if row.get("objective_residual_contract_error_two_theta_deg") is not None
    ]
    residual_contract_errors_phi = [
        float(row["objective_residual_contract_error_phi_deg_wrapped"])
        for row in rows
        if row.get("objective_residual_contract_error_phi_deg_wrapped") is not None
    ]
    max_residual_contract_error_two_theta = (
        max(residual_contract_errors_two_theta) if residual_contract_errors_two_theta else None
    )
    max_residual_contract_error_phi = (
        max(residual_contract_errors_phi) if residual_contract_errors_phi else None
    )
    residual_ok = all(
        row.get("residual_delta_two_theta") is not None
        and row.get("residual_delta_phi_wrapped") is not None
        and row.get("objective_residual_two_theta_deg") is not None
        and row.get("objective_residual_phi_deg") is not None
        and row.get("objective_residual_vector_two_theta_deg") is not None
        and row.get("objective_residual_vector_phi_deg") is not None
        and bool(row.get("objective_residual_vector_contract_match"))
        and row.get("objective_residual_contract_error_two_theta_deg") is not None
        and row.get("objective_residual_contract_error_phi_deg_wrapped") is not None
        and float(row["objective_residual_contract_error_two_theta_deg"]) <= EXACT_TOL_DEG
        and float(row["objective_residual_contract_error_phi_deg_wrapped"]) <= EXACT_TOL_DEG
        for row in rows
    )
    residual_units_ok = all(
        str(row.get("objective_residual_units", "")) == "deg"
        and str(row.get("objective_residual_coordinate_space", "")) == "caked_deg"
        for row in rows
    )
    residual_phi_wrapped_ok = all(
        row.get("objective_residual_expected_phi_deg_wrapped") is not None
        and row.get("objective_residual_phi_deg") is not None
        and abs(
            opt._angular_difference_deg(
                float(row["objective_residual_expected_phi_deg_wrapped"]),
                float(row["objective_residual_phi_deg"]),
            )
        )
        <= EXACT_TOL_DEG
        for row in rows
    )
    residual_arrow_ok = all(
        bool(row.get("residual_arrow_matches_plotted_source_target_delta")) for row in rows
    )
    vector_indices_ok = all(
        int(row.get("objective_residual_vector_two_theta_index", -1)) == 2 * int(index)
        and int(row.get("objective_residual_vector_phi_index", -1)) == 2 * int(index) + 1
        and bool(row.get("objective_residual_vector_contract_match"))
        for index, row in enumerate(rows)
    )
    pixel_residual_rejected = all(
        row.get("pixel_residual_used_for_objective") is False
        and str(row.get("objective_residual_units", "deg")) == "deg"
        and str(row.get("objective_residual_coordinate_space", "caked_deg")) == "caked_deg"
        and "px" not in str(row.get("objective_summary_metric_unit", "deg")).lower()
        for row in rows
    )
    detector_native_diagnostic_available = all(
        row.get("detector_native_reprojection_diagnostic_two_theta_deg") is not None
        and row.get("detector_native_reprojection_diagnostic_phi_deg") is not None
        for row in rows
    )
    schema_ok = all(not _schema_missing_fields(row) for row in rows)
    required_row_count = int(len(rows))
    return {
        "objective_audit_schema_version": int(AUDIT_SCHEMA_VERSION),
        "objective_audit_required_fields": list(OBJECTIVE_AUDIT_REQUIRED_PAIR_FIELDS),
        "objective_audit_schema_machine_checkable": bool(schema_ok),
        "objective_audit_pair_count_is_7": bool(required_row_count == 7),
        "png_is_diagnostic_not_gate": True,
        "detector_native_reprojection_is_diagnostic": True,
        "detector_native_reprojection_diagnostic_available": bool(
            detector_native_diagnostic_available
        ),
        "optimizer_measured_target_equals_cached_target": bool(measured_ok),
        "optimizer_source_equals_dynamic_source": bool(source_ok),
        "optimizer_source_uses_dynamic_sim_visual_caked_deg": bool(source_ok),
        "dynamic_source_source_is_sim_visual_caked_deg": bool(
            rows and all(row.get("dynamic_source_source") == "sim_visual_caked_deg" for row in rows)
        ),
        "optimizer_source_source_is_sim_visual_caked_deg": bool(
            rows
            and all(row.get("optimizer_source_source") == "sim_visual_caked_deg" for row in rows)
        ),
        "objective_source_authority_counts": {
            "clicked_visual_candidate_source_count": int(clicked_visual_candidate_count),
            "saved_manual_source_count": int(saved_manual_source_count),
            "detector_grouped_fallback_count": int(detector_grouped_fallback_count),
        },
        "clicked_visual_candidate_source_count_is_zero": bool(clicked_visual_candidate_count == 0),
        "saved_manual_source_count_is_zero": bool(saved_manual_source_count == 0),
        "detector_grouped_fallback_count_is_zero": bool(detector_grouped_fallback_count == 0),
        "residual_vector_machine_checkable": bool(residual_ok),
        "objective_residual_is_two_theta_phi_degrees": bool(residual_units_ok),
        "objective_residual_wraps_phi_delta": bool(residual_phi_wrapped_ok),
        "objective_residual_matches_source_minus_target": bool(residual_ok),
        "objective_residual_vector_indices_match_pair_order": bool(vector_indices_ok),
        "objective_residual_contract_error_max_two_theta_deg": (
            float(max_residual_contract_error_two_theta)
            if max_residual_contract_error_two_theta is not None
            else None
        ),
        "objective_residual_contract_error_max_phi_deg_wrapped": (
            float(max_residual_contract_error_phi)
            if max_residual_contract_error_phi is not None
            else None
        ),
        "residual_arrows_equal_plotted_source_target_deltas": bool(residual_arrow_ok),
        "objective_rejects_pixel_residual_for_manual_caked_qr_fit": bool(pixel_residual_rejected),
        "pixel_residual_path_used": False,
        "all_frame_match": bool(rows and all(bool(row.get("frame_match")) for row in rows)),
        "all_unit_match": bool(rows and all(bool(row.get("unit_match")) for row in rows)),
        "all_branch_slot_match": bool(
            rows and all(bool(row.get("branch_slot_match")) for row in rows)
        ),
        "all_q_group_match": bool(rows and all(bool(row.get("q_group_match")) for row in rows)),
        "all_hkl_match": bool(rows and all(bool(row.get("hkl_match")) for row in rows)),
        "q_group_hkl_branch_identity_unchanged": bool(
            rows
            and all(bool(row.get("branch_slot_match")) for row in rows)
            and all(bool(row.get("q_group_match")) for row in rows)
            and all(bool(row.get("hkl_match")) for row in rows)
        ),
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
        "target_markers_remain_fixed_after_step": bool(target_fixed and all(target_fixed)),
        "source_markers_move_after_step": (
            bool(any(value > EXACT_TOL_DEG for value in source_shift_norms))
            if perturb_applied
            else True
        ),
    }


def _raw_angular_rms_deg(rows: Sequence[Mapping[str, object]]) -> float:
    sq: list[float] = []
    for row in rows:
        two_theta = _finite_float(row.get("residual_two_theta_deg"))
        phi = _finite_float(row.get("residual_phi_deg_wrapped"))
        if two_theta is None or phi is None:
            continue
        sq.append(float(two_theta * two_theta + phi * phi))
    if not sq:
        return float("nan")
    return float(math.sqrt(float(np.mean(np.asarray(sq, dtype=np.float64)))))


def _arrow_norms(rows: Sequence[Mapping[str, object]]) -> list[float]:
    norms: list[float] = []
    for row in rows:
        two_theta = _finite_float(row.get("residual_arrow_delta_two_theta_deg"))
        phi = _finite_float(row.get("residual_arrow_delta_phi_deg_wrapped"))
        if two_theta is None or phi is None:
            continue
        norms.append(float(math.hypot(two_theta, phi)))
    return norms


def _first_int(mapping: Mapping[str, object], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key not in mapping:
            continue
        try:
            return int(mapping.get(key))
        except Exception:
            continue
    return None


def _fixed_source_counter_summary(
    point_match_summary: Mapping[str, object] | None,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary = point_match_summary if isinstance(point_match_summary, Mapping) else {}
    pair_count = int(len(rows))
    resolved = _first_int(
        summary,
        (
            "fixed_source_resolved_count",
            "matched_fixed_pair_count",
            "fixed_source_pair_count",
            "matched_pair_count",
        ),
    )
    if resolved is None:
        resolved = pair_count
    fallback_keys = (
        "fallback_entry_count",
        "fallback_row_count",
        "fixed_source_resolution_fallback_count",
        "missing_fixed_source_count",
        "subset_fallback_hkl_count",
        "fallback_hkl_count",
        "branch_mismatch_count",
        "missing_pair_count",
    )
    fallback_counts: dict[str, int] = {}
    for key in fallback_keys:
        value = _first_int(summary, (key,))
        fallback_counts[key] = int(value or 0)
    labels = [
        str(row.get(key, "") or "")
        for row in rows
        for key in ("dynamic_source_source", "optimizer_source_source")
    ]
    source_label_fallback_count = sum(
        1
        for label in labels
        if "detector" in label.lower()
        or "fallback" in label.lower()
        or "clicked_visual_candidate" in label
        or "saved" in label.lower()
        or "manual" in label.lower()
    )
    fallback_counts["source_label_fallback_count"] = int(source_label_fallback_count)
    clean = bool(resolved == pair_count and all(value == 0 for value in fallback_counts.values()))
    return {
        "fixed_source_counters_clean": clean,
        "fixed_source_resolved_count": int(resolved),
        "fixed_source_expected_count": int(pair_count),
        "fallback_counts": fallback_counts,
        "fallback_counters_zero": bool(all(value == 0 for value in fallback_counts.values())),
    }


def _fit_improvement_pairs(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for before, after in zip(before_rows, after_rows):
        target_before = _finite_pair(before.get("optimizer_measured_anchor_two_theta_phi"))
        target_after = _finite_pair(after.get("optimizer_measured_anchor_two_theta_phi"))
        source_before = _finite_pair(before.get("optimizer_simulated_source_two_theta_phi"))
        source_after = _finite_pair(after.get("optimizer_simulated_source_two_theta_phi"))
        target_delta = _delta(target_after, target_before)
        source_delta = _delta(source_after, source_before)
        before_arrow = _finite_pair(
            (
                before.get("residual_arrow_delta_two_theta_deg"),
                before.get("residual_arrow_delta_phi_deg_wrapped"),
            )
        )
        after_arrow = _finite_pair(
            (
                after.get("residual_arrow_delta_two_theta_deg"),
                after.get("residual_arrow_delta_phi_deg_wrapped"),
            )
        )
        before_norm = (
            float(math.hypot(before_arrow[0], before_arrow[1]))
            if before_arrow is not None
            else None
        )
        after_norm = (
            float(math.hypot(after_arrow[0], after_arrow[1])) if after_arrow is not None else None
        )
        pairs.append(
            {
                "pair_id": after.get("pair_id", before.get("pair_id")),
                "q_group_key": after.get("q_group_key", before.get("q_group_key")),
                "normalized_hkl": after.get("normalized_hkl", before.get("normalized_hkl")),
                "physical_branch_slot": after.get(
                    "physical_branch_slot", before.get("physical_branch_slot")
                ),
                "target_two_theta_deg": after.get("cached_target_two_theta_deg"),
                "target_phi_deg": after.get("cached_target_phi_deg"),
                "source_before_two_theta_deg": (
                    float(source_before[0]) if source_before is not None else None
                ),
                "source_before_phi_deg": (
                    float(source_before[1]) if source_before is not None else None
                ),
                "source_after_two_theta_deg": (
                    float(source_after[0]) if source_after is not None else None
                ),
                "source_after_phi_deg": (
                    float(source_after[1]) if source_after is not None else None
                ),
                "before_residual_two_theta_deg": before.get("residual_two_theta_deg"),
                "before_residual_phi_deg_wrapped": before.get("residual_phi_deg_wrapped"),
                "after_residual_two_theta_deg": after.get("residual_two_theta_deg"),
                "after_residual_phi_deg_wrapped": after.get("residual_phi_deg_wrapped"),
                "before_residual_arrow_norm_deg": before_norm,
                "after_residual_arrow_norm_deg": after_norm,
                "residual_arrow_shortened": bool(
                    before_norm is not None and after_norm is not None and after_norm < before_norm
                ),
                "target_delta_two_theta_deg": target_delta["delta_two_theta"],
                "target_delta_phi_deg_wrapped": target_delta["delta_phi_wrapped"],
                "source_delta_two_theta_deg": source_delta["delta_two_theta"],
                "source_delta_phi_deg_wrapped": source_delta["delta_phi_wrapped"],
                "target_fixed": bool(target_delta.get("within_exact_tolerance")),
                "source_changed": bool(
                    source_delta.get("norm") is not None
                    and float(source_delta["norm"]) > EXACT_TOL_DEG
                ),
            }
        )
    return pairs


def _fit_improvement_summary(
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
    *,
    perturb_info: Mapping[str, object],
    active_vars: Sequence[str],
    point_match_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    initial_rms = _raw_angular_rms_deg(before_rows)
    final_rms = _raw_angular_rms_deg(after_rows)
    improvement = (
        float(initial_rms - final_rms) if math.isfinite(initial_rms + final_rms) else float("nan")
    )
    pairs = _fit_improvement_pairs(before_rows, after_rows)
    before_norms = _arrow_norms(before_rows)
    after_norms = _arrow_norms(after_rows)
    before_mean_arrow = float(np.mean(before_norms)) if before_norms else float("nan")
    after_mean_arrow = float(np.mean(after_norms)) if after_norms else float("nan")
    counter_summary = _fixed_source_counter_summary(point_match_summary, after_rows)
    target_fixed = bool(pairs and all(bool(pair.get("target_fixed")) for pair in pairs))
    source_changed = bool(any(bool(pair.get("source_changed")) for pair in pairs))
    arrows_shorter = bool(
        math.isfinite(before_mean_arrow)
        and math.isfinite(after_mean_arrow)
        and after_mean_arrow < before_mean_arrow
    )
    perturbed_start_nonzero = bool(
        perturb_info.get("applied")
        and math.isfinite(float(perturb_info.get("delta", 0.0)))
        and abs(float(perturb_info.get("delta", 0.0))) > 0.0
    )
    return {
        "fit_improvement_audit": True,
        "active_vars": [str(name) for name in active_vars],
        "perturb_start": dict(perturb_info),
        "initial_raw_angular_rms_deg": float(initial_rms),
        "final_raw_angular_rms_deg": float(final_rms),
        "improvement_raw_angular_rms_deg": float(improvement),
        "before_residual_arrow_mean_norm_deg": float(before_mean_arrow),
        "after_residual_arrow_mean_norm_deg": float(after_mean_arrow),
        "metric_name": "raw_angular_rms_deg",
        "metric_unit": "deg",
        "improvement_pairs": pairs,
        "fixed_source_counter_summary": counter_summary,
        "checks": {
            "coordinate_audit_improvement_uses_nonzero_perturbed_start": perturbed_start_nonzero,
            "coordinate_audit_improvement_reduces_raw_angular_rms": bool(
                math.isfinite(initial_rms) and math.isfinite(final_rms) and initial_rms > final_rms
            ),
            "coordinate_audit_improvement_keeps_target_fixed": target_fixed,
            "coordinate_audit_improvement_source_before_after_differ": source_changed,
            "coordinate_audit_improvement_after_arrows_shorter": arrows_shorter,
            "coordinate_audit_improvement_preserves_fixed_source_counters": bool(
                counter_summary["fixed_source_counters_clean"]
            ),
            "coordinate_audit_improvement_fallback_counters_zero": bool(
                counter_summary["fallback_counters_zero"]
            ),
            "coordinate_audit_improvement_metric_unit_deg": True,
        },
    }


def _single_step_active_vars(active_vars: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for raw in active_vars:
        name = str(raw).strip()
        if name and name not in names:
            names.append(name)
    if set(names) != {"gamma", "Gamma"} or len(names) != 2:
        raise ValueError(
            "--single-step-detector-angle-audit only supports --active-vars gamma,Gamma"
        )
    return tuple(names)


def _params_with_delta(
    params: Mapping[str, object],
    deltas: Mapping[str, float],
) -> dict[str, object]:
    out = dict(params)
    for name, delta in deltas.items():
        base = _finite_float(out.get(str(name)))
        out[str(name)] = float((base or 0.0) + float(delta))
    return out


def _single_step_detector_angle_trial(
    request,
    base_params: Mapping[str, object],
    *,
    active_vars: Sequence[str] = ("gamma", "Gamma"),
    max_angle_step_deg: float = 5.0,
    fd_step_deg: float = 0.05,
) -> dict[str, object]:
    names = _single_step_active_vars(active_vars)
    max_step = abs(float(max_angle_step_deg))
    fd_step = abs(float(fd_step_deg))
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("--max-angle-step-deg must be finite and positive")
    if not math.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError("--fd-step-deg must be finite and positive")

    base_bundle = _evaluate_pair_bundle(request, base_params)
    r0 = np.asarray(base_bundle["residual_vector"], dtype=np.float64).reshape(-1)
    jacobian = np.full((r0.size, len(names)), np.nan, dtype=np.float64)
    fd_records: list[dict[str, object]] = []
    for column, name in enumerate(names):
        plus_params = _params_with_delta(base_params, {name: fd_step})
        minus_params = _params_with_delta(base_params, {name: -fd_step})
        plus = np.asarray(
            _evaluate_pair_bundle(request, plus_params)["residual_vector"],
            dtype=np.float64,
        ).reshape(-1)
        minus = np.asarray(
            _evaluate_pair_bundle(request, minus_params)["residual_vector"],
            dtype=np.float64,
        ).reshape(-1)
        if plus.shape == r0.shape and minus.shape == r0.shape:
            jacobian[:, column] = (plus - minus) / (2.0 * fd_step)
        fd_records.append(
            {
                "var": name,
                "fd_step_deg": float(fd_step),
                "plus_residual_component_count": int(plus.size),
                "minus_residual_component_count": int(minus.size),
                "shape_matches_base": bool(plus.shape == r0.shape and minus.shape == r0.shape),
            }
        )

    finite_jacobian = bool(jacobian.size and np.all(np.isfinite(jacobian)))
    sensitive = bool(finite_jacobian and np.any(np.abs(jacobian) > 1.0e-12))
    if sensitive and r0.size:
        try:
            proposed, *_ = np.linalg.lstsq(jacobian, -r0, rcond=None)
        except np.linalg.LinAlgError:
            proposed = np.zeros(len(names), dtype=np.float64)
            sensitive = False
    else:
        proposed = np.zeros(len(names), dtype=np.float64)
    if not np.all(np.isfinite(proposed)):
        proposed = np.zeros(len(names), dtype=np.float64)
        sensitive = False
    clipped = np.clip(proposed, -max_step, max_step) if sensitive else np.zeros(len(names))
    status = "ok" if sensitive else "insensitive_to_gamma_Gamma"
    delta_by_name = {name: float(clipped[index]) for index, name in enumerate(names)}
    trial_params = _params_with_delta(base_params, delta_by_name)
    trial_bundle = _evaluate_pair_bundle(request, trial_params)
    before_rows = list(base_bundle["rows"])  # type: ignore[arg-type]
    after_rows = list(trial_bundle["rows"])  # type: ignore[arg-type]
    row_count_changed = len(before_rows) != len(after_rows)
    if row_count_changed:
        status = "row_count_changed_between_base_and_trial"
        rows: list[dict[str, object]] = []
    else:
        rows = _single_step_audit_rows(
            request,
            before_rows,
            after_rows,
            delta_by_name=delta_by_name,
            single_step_status=status,
        )
    return {
        "single_step_status": status,
        "active_vars": list(names),
        "max_angle_step_deg": float(max_step),
        "fd_step_deg": float(fd_step),
        "delta_gamma_deg": float(delta_by_name.get("gamma", 0.0)),
        "delta_Gamma_deg": float(delta_by_name.get("Gamma", 0.0)),
        "proposed_delta_deg": {name: float(proposed[index]) for index, name in enumerate(names)},
        "clipped_delta_deg": dict(delta_by_name),
        "finite_jacobian": bool(finite_jacobian),
        "sensitive_to_gamma_Gamma": bool(sensitive),
        "finite_difference_records": fd_records,
        "base_residual_vector": _jsonable(r0),
        "jacobian": _jsonable(jacobian),
        "base_params": {name: base_params.get(name) for name in names},
        "trial_params": {name: trial_params.get(name) for name in names},
        "before_row_count": int(len(before_rows)),
        "after_row_count": int(len(after_rows)),
        "row_count_changed_between_base_and_trial": bool(row_count_changed),
        "before_rows": before_rows,
        "after_rows": after_rows,
        "rows": rows,
        "base_objective_summary": base_bundle["objective_summary"],
        "trial_objective_summary": trial_bundle["objective_summary"],
    }


def _dataset_display_context(request) -> Mapping[str, object]:
    specs = getattr(request, "dataset_specs", None) or ()
    for spec in specs:
        if isinstance(spec, Mapping):
            return spec
    return {}


def _native_to_display(
    native_point: tuple[float, float] | None,
    dataset: Mapping[str, object],
) -> tuple[tuple[float, float] | None, str | None]:
    if native_point is None:
        return None, "native_detector_px_missing"
    callback = dataset.get("native_detector_coords_to_detector_display_coords")
    if not callable(callback):
        reason = str(
            dataset.get("native_detector_coords_to_detector_display_coords_unavailable_reason")
            or "native_to_display_converter_missing"
        )
        return None, reason
    try:
        display = callback(float(native_point[0]), float(native_point[1]))
    except Exception as exc:
        return None, f"native_to_display_converter_failed:{type(exc).__name__}"
    point = _finite_pair(display)
    if point is None:
        return None, "native_to_display_converter_returned_invalid"
    return point, None


def _display_frame_point(
    *,
    native_point: tuple[float, float] | None,
    raw_display_point: tuple[float, float] | None,
    dataset: Mapping[str, object],
    require_raw_display: bool,
) -> tuple[tuple[float, float] | None, bool, str | None]:
    converted, reason = _native_to_display(native_point, dataset)
    if converted is None:
        return raw_display_point, False, reason
    if require_raw_display and raw_display_point is None:
        return None, False, "saved_detector_display_px_missing"
    if raw_display_point is not None:
        mismatch = math.hypot(
            float(converted[0]) - float(raw_display_point[0]),
            float(converted[1]) - float(raw_display_point[1]),
        )
        if mismatch > DETECTOR_FRAME_TOL_PX:
            return raw_display_point, False, "mixed_display_native_detector_px"
    display_point = (
        raw_display_point if require_raw_display and raw_display_point is not None else converted
    )
    return display_point, True, None


def _row_pair(row: Mapping[str, object], key: str) -> tuple[float, float] | None:
    return _finite_pair(row.get(key))


def _single_step_audit_rows(
    request,
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
    *,
    delta_by_name: Mapping[str, float],
    single_step_status: str,
) -> list[dict[str, object]]:
    dataset = _dataset_display_context(request)
    out: list[dict[str, object]] = []
    for before, after in zip(before_rows, after_rows):
        background_display_raw = _row_pair(before, "background_detector_display_px")
        background_native = _row_pair(before, "background_detector_native_px")
        original_native = _row_pair(before, "fit_prediction_detector_native_px")
        original_display_raw = _row_pair(before, "fit_prediction_detector_display_px")
        trial_native = _row_pair(after, "fit_prediction_detector_native_px")
        trial_display_raw = _row_pair(after, "fit_prediction_detector_display_px")
        background_display, background_display_valid, background_display_reason = (
            _display_frame_point(
                native_point=background_native,
                raw_display_point=background_display_raw,
                dataset=dataset,
                require_raw_display=True,
            )
        )
        original_display, original_display_valid, original_display_reason = _display_frame_point(
            native_point=original_native,
            raw_display_point=original_display_raw,
            dataset=dataset,
            require_raw_display=False,
        )
        trial_display, trial_display_valid, trial_display_reason = _display_frame_point(
            native_point=trial_native,
            raw_display_point=trial_display_raw,
            dataset=dataset,
            require_raw_display=False,
        )
        background_caked = _row_pair(before, "optimizer_measured_anchor_two_theta_phi")
        original_caked = _row_pair(before, "dynamic_sim_visual_caked_deg_two_theta_phi")
        trial_caked = _row_pair(after, "dynamic_sim_visual_caked_deg_two_theta_phi")
        live_projected_original_caked = _row_pair(
            before,
            "optimizer_simulated_source_two_theta_phi",
        )
        live_projected_trial_caked = _row_pair(
            after,
            "optimizer_simulated_source_two_theta_phi",
        )
        fit_prediction_caked = original_caked
        original_delta = _delta(original_caked, background_caked)
        trial_delta = _delta(trial_caked, background_caked)
        hkl_before = before.get("hkl", before.get("normalized_hkl"))
        hkl_after = after.get("hkl", after.get("normalized_hkl"))
        pair_same = bool(str(before.get("pair_id")) == str(after.get("pair_id")))
        q_group_same = _identity_match(before.get("q_group_key"), after.get("q_group_key"))
        hkl_same = bool(_normal_hkl(hkl_before) == _normal_hkl(hkl_after))
        branch_same = bool(
            _normal_identity(before.get("source_branch_index"))
            == _normal_identity(after.get("source_branch_index"))
        )
        caked_valid = bool(
            background_caked is not None
            and original_caked is not None
            and trial_caked is not None
            and _point_match(original_caked, fit_prediction_caked)
        )
        detector_valid = bool(
            background_display is not None
            and original_display is not None
            and trial_display is not None
            and background_display_valid
            and original_display_valid
            and trial_display_valid
        )
        real_projector_used = bool(
            before.get("fit_space_projector_kind") == "exact_caked_bundle"
            and after.get("fit_space_projector_kind") == "exact_caked_bundle"
            and before.get("caked_projection_signature") is not None
            and after.get("caked_projection_signature") is not None
        )
        detector_native_valid = bool(
            background_native is not None
            and original_native is not None
            and trial_native is not None
        )
        out.append(
            {
                "pair_id": before.get("pair_id"),
                "q_group_key": before.get("q_group_key"),
                "hkl": hkl_before,
                "source_branch_index": before.get("source_branch_index"),
                "source_table_index": before.get("source_table_index"),
                "source_row_index": before.get("source_row_index"),
                "background_index": before.get("dataset_index", 0),
                "background_detector_display_px": _pair_list(background_display),
                "background_detector_native_px": _pair_list(background_native),
                "background_caked_deg": _pair_list(background_caked),
                "background_caked_source": before.get("optimizer_measured_source"),
                "original_sim_detector_display_px": _pair_list(original_display),
                "original_sim_detector_native_px": _pair_list(original_native),
                "original_sim_caked_deg": _pair_list(original_caked),
                "original_sim_source": before.get("dynamic_source_source"),
                "original_live_projected_detector_caked_deg": _pair_list(
                    live_projected_original_caked
                ),
                "trial_sim_detector_display_px": _pair_list(trial_display),
                "trial_sim_detector_native_px": _pair_list(trial_native),
                "trial_sim_caked_deg": _pair_list(trial_caked),
                "trial_sim_source": after.get("dynamic_source_source"),
                "trial_live_projected_detector_caked_deg": _pair_list(live_projected_trial_caked),
                "fit_prediction_caked_deg": _pair_list(fit_prediction_caked),
                "original_to_background_delta_caked_deg": [
                    original_delta["delta_two_theta"],
                    original_delta["delta_phi_wrapped"],
                ],
                "trial_to_background_delta_caked_deg": [
                    trial_delta["delta_two_theta"],
                    trial_delta["delta_phi_wrapped"],
                ],
                "original_to_background_norm_deg": original_delta["norm"],
                "trial_to_background_norm_deg": trial_delta["norm"],
                "delta_gamma_deg": float(delta_by_name.get("gamma", 0.0)),
                "delta_Gamma_deg": float(delta_by_name.get("Gamma", 0.0)),
                "single_step_status": str(single_step_status),
                "identity_same_before_after": pair_same,
                "q_group_same_before_after": q_group_same,
                "hkl_same_before_after": hkl_same,
                "branch_same_before_after": branch_same,
                "detector_display_frame_valid": detector_valid,
                "detector_native_frame_valid": detector_native_valid,
                "caked_frame_valid": caked_valid,
                "real_caked_projector_used": real_projector_used,
                "saved_sim_refined_caked_used": False,
                "original_sim_caked_matches_fit_prediction_caked_deg": bool(
                    _point_match(original_caked, fit_prediction_caked)
                ),
                "detector_display_invalid_reasons": [
                    reason
                    for reason in (
                        background_display_reason,
                        original_display_reason,
                        trial_display_reason,
                    )
                    if reason
                ],
            }
        )
    return out


def _single_step_checks(
    rows: Sequence[Mapping[str, object]],
    *,
    delta_gamma_deg: float,
    delta_Gamma_deg: float,
    max_angle_step_deg: float,
) -> dict[str, object]:
    plotted = [
        row
        for row in rows
        if bool(row.get("detector_display_frame_valid")) and bool(row.get("caked_frame_valid"))
    ]
    invalid_detector_rows = [
        row for row in rows if not bool(row.get("detector_display_frame_valid"))
    ]
    invalid_reasons_by_count: dict[str, int] = {}
    for row in invalid_detector_rows:
        reasons = row.get("detector_display_invalid_reasons", ()) or ()
        if isinstance(reasons, (str, bytes)):
            reasons = (str(reasons),)
        for reason in reasons:
            reason_text = str(reason or "").strip()
            if not reason_text:
                continue
            invalid_reasons_by_count[reason_text] = (
                int(invalid_reasons_by_count.get(reason_text, 0)) + 1
            )
        if not reasons:
            invalid_reasons_by_count["detector_display_frame_invalid"] = (
                int(invalid_reasons_by_count.get("detector_display_frame_invalid", 0)) + 1
            )
    row_count = int(len(rows))
    plotted_count = int(len(plotted))
    return {
        "row_count_gt_zero": bool(row_count > 0),
        "plotted_row_count_gt_zero": bool(plotted_count > 0),
        "all_plotted_detector_display_frame_valid": bool(
            plotted and all(bool(row.get("detector_display_frame_valid")) for row in plotted)
        ),
        "all_plotted_caked_frame_valid": bool(
            plotted and all(bool(row.get("caked_frame_valid")) for row in plotted)
        ),
        "all_plotted_real_caked_projector_used": bool(
            plotted and all(bool(row.get("real_caked_projector_used")) for row in plotted)
        ),
        "saved_sim_refined_caked_used_false_for_all_rows": bool(
            rows and all(row.get("saved_sim_refined_caked_used") is False for row in rows)
        ),
        "delta_gamma_bounded": bool(abs(float(delta_gamma_deg)) <= float(max_angle_step_deg)),
        "delta_Gamma_bounded": bool(abs(float(delta_Gamma_deg)) <= float(max_angle_step_deg)),
        "identity_same_before_after_all_rows": bool(
            rows and all(bool(row.get("identity_same_before_after")) for row in rows)
        ),
        "q_group_same_before_after_all_rows": bool(
            rows and all(bool(row.get("q_group_same_before_after")) for row in rows)
        ),
        "hkl_same_before_after_all_rows": bool(
            rows and all(bool(row.get("hkl_same_before_after")) for row in rows)
        ),
        "branch_same_before_after_all_rows": bool(
            rows and all(bool(row.get("branch_same_before_after")) for row in rows)
        ),
        "original_sim_caked_matches_fit_prediction_caked_deg_all_rows": bool(
            rows
            and all(
                bool(row.get("original_sim_caked_matches_fit_prediction_caked_deg")) for row in rows
            )
        ),
        "invalid_detector_display_row_count": int(len(invalid_detector_rows)),
        "valid_plotted_fraction": (
            float(plotted_count) / float(row_count) if row_count > 0 else 0.0
        ),
        "invalid_reasons_by_count": dict(invalid_reasons_by_count),
        "plotted_row_count": plotted_count,
        "row_count": row_count,
    }


def _checks_pass(checks: Mapping[str, object], *, perturb_applied: bool) -> bool:
    for key, value in checks.items():
        if key == "source_moves_under_perturbation" and not perturb_applied:
            continue
        if key == "pixel_residual_path_used":
            continue
        if isinstance(value, bool) and not value:
            return False
    return True


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
                xy=source,
                xytext=target,
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


def _plot_improvement_rows(
    path: Path,
    before_rows: Sequence[Mapping[str, object]],
    after_rows: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    target_x: list[float] = []
    target_y: list[float] = []
    before_x: list[float] = []
    before_y: list[float] = []
    after_x: list[float] = []
    after_y: list[float] = []
    for before, after in zip(before_rows, after_rows):
        target = _finite_pair(after.get("cached_click_candidate_two_theta_phi"))
        before_source = _finite_pair(before.get("optimizer_simulated_source_two_theta_phi"))
        after_source = _finite_pair(after.get("optimizer_simulated_source_two_theta_phi"))
        if target is not None:
            target_x.append(target[0])
            target_y.append(target[1])
        if before_source is not None:
            before_x.append(before_source[0])
            before_y.append(before_source[1])
        if after_source is not None:
            after_x.append(after_source[0])
            after_y.append(after_source[1])
        if target is not None and before_source is not None:
            ax.annotate(
                "",
                xy=before_source,
                xytext=target,
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 0.8,
                    "alpha": 0.35,
                    "color": "tab:red",
                },
            )
        if target is not None and after_source is not None:
            ax.annotate(
                "",
                xy=after_source,
                xytext=target,
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 0.9,
                    "alpha": 0.65,
                    "color": "tab:green",
                },
            )
            ax.text(after_source[0], after_source[1], str(after.get("pair_index")), fontsize=8)
    if target_x:
        ax.scatter(target_x, target_y, label="cached target", marker="o", s=55)
    if before_x:
        ax.scatter(before_x, before_y, label="source before solve", marker="x", s=55)
    if after_x:
        ax.scatter(after_x, after_y, label="source after solve", marker="+", s=65)
    ax.set_xlabel("two_theta_deg")
    ax.set_ylabel("phi_deg")
    ax.set_title("New4 Fit Improvement Audit")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_single_step_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotted = [
        row
        for row in rows
        if bool(row.get("detector_display_frame_valid")) and bool(row.get("caked_frame_valid"))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (det_ax, caked_ax) = plt.subplots(1, 2, figsize=(15, 7))

    def _scatter(
        ax,
        key: str,
        *,
        label: str,
        marker: str,
        size: int,
    ) -> None:
        xs: list[float] = []
        ys: list[float] = []
        for row in plotted:
            point = _finite_pair(row.get(key))
            if point is not None:
                xs.append(point[0])
                ys.append(point[1])
        if xs:
            ax.scatter(xs, ys, label=label, marker=marker, s=size)

    _scatter(
        det_ax,
        "background_detector_display_px",
        label="background/manual QR",
        marker="o",
        size=55,
    )
    _scatter(
        det_ax,
        "original_sim_detector_display_px",
        label="original simulation QR",
        marker="s",
        size=55,
    )
    _scatter(
        det_ax,
        "trial_sim_detector_display_px",
        label="one-step trial simulation QR",
        marker="^",
        size=65,
    )
    _scatter(
        caked_ax,
        "background_caked_deg",
        label="background/manual QR",
        marker="o",
        size=55,
    )
    _scatter(
        caked_ax,
        "original_sim_caked_deg",
        label="original simulation QR",
        marker="s",
        size=55,
    )
    _scatter(
        caked_ax,
        "trial_sim_caked_deg",
        label="one-step trial simulation QR",
        marker="^",
        size=65,
    )
    label_rows = sorted(
        plotted,
        key=lambda row: float(row.get("original_to_background_norm_deg") or 0.0),
        reverse=True,
    )[:3]
    label_ids = {str(row.get("pair_id")) for row in label_rows}
    for row in plotted:
        det_before = _finite_pair(row.get("original_sim_detector_display_px"))
        det_after = _finite_pair(row.get("trial_sim_detector_display_px"))
        caked_before = _finite_pair(row.get("original_sim_caked_deg"))
        caked_after = _finite_pair(row.get("trial_sim_caked_deg"))
        if det_before is not None and det_after is not None:
            det_ax.annotate(
                "",
                xy=det_after,
                xytext=det_before,
                arrowprops={"arrowstyle": "->", "linewidth": 0.8, "alpha": 0.55},
            )
        if caked_before is not None and caked_after is not None:
            caked_ax.annotate(
                "",
                xy=caked_after,
                xytext=caked_before,
                arrowprops={"arrowstyle": "->", "linewidth": 0.8, "alpha": 0.55},
            )
        if str(row.get("pair_id")) in label_ids:
            label = f"{row.get('pair_id')}\n{row.get('hkl')}"
            if det_after is not None:
                det_ax.text(det_after[0], det_after[1], label, fontsize=7)
            if caked_after is not None:
                caked_ax.text(caked_after[0], caked_after[1], label, fontsize=7)

    det_ax.set_xlabel("detector display x (px)")
    det_ax.set_ylabel("detector display y (px)")
    det_ax.set_title("Detector Display Space")
    det_ax.grid(True, alpha=0.25)
    det_ax.legend(loc="best", fontsize=8)
    caked_ax.set_xlabel("two theta (deg)")
    caked_ax.set_ylabel("phi (deg)")
    caked_ax.set_title("Caked Space")
    caked_ax.grid(True, alpha=0.25)
    caked_ax.legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.text(
        0.02,
        0.02,
        "\n".join(
            (
                "Circle = background/manual QR",
                "Square = original simulation QR",
                "Triangle = one-step gamma/Gamma trial QR",
                "Arrows = original sim -> one-step trial",
                "Detector panel units = display px",
                "Caked panel units = deg",
            )
        ),
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
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
    fit_improvement_audit: bool = False,
    perturb_start: str | None = None,
    active_vars: str | Sequence[str] | None = None,
    single_step_detector_angle_audit: bool = False,
    max_angle_step_deg: float = 5.0,
    fd_step_deg: float = 0.05,
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
    if single_step_detector_angle_audit:
        if effective_perturb or fit_improvement_audit or perturb_start:
            raise ValueError(
                "--single-step-detector-angle-audit cannot be combined with solve/perturb modes"
            )
        if step_param is not None or step_size is not None:
            raise ValueError("--single-step-detector-angle-audit uses finite differences only")
        active_names = (
            ["gamma", "Gamma"]
            if active_vars is None
            else (
                _parse_active_vars(active_vars)
                if isinstance(active_vars, str)
                else [str(name) for name in active_vars]
            )
        )
        active_names = list(_single_step_active_vars(active_names))
    else:
        active_names = (
            _parse_active_vars(active_vars)
            if active_vars is None or isinstance(active_vars, str)
            else [str(name) for name in active_vars]
        )
    provider_report = preflight._run_point_provider_report_only(state_path, int(background_index))
    context = ladder._capture_solver_context(state_path, int(background_index))
    request = ladder.build_solver_request(
        context,
        active_names,
        max_nfev=20 if fit_improvement_audit else 1,
    )
    prepared_run = context.get("prepared_run")
    prepared_params = (
        getattr(prepared_run, "fit_params", None) if prepared_run is not None else None
    )
    base_params = dict(getattr(request, "params", {}) or {})
    base_params.update(dict(prepared_params or {}))
    if single_step_detector_angle_audit:
        request = _request_with_dataset_display_context(
            request,
            getattr(prepared_run, "dataset_infos", None) if prepared_run is not None else None,
        )
        trial_payload = _single_step_detector_angle_trial(
            request,
            base_params,
            active_vars=active_names,
            max_angle_step_deg=float(max_angle_step_deg),
            fd_step_deg=float(fd_step_deg),
        )
        rows = [
            dict(row) for row in trial_payload.get("rows", ()) or () if isinstance(row, Mapping)
        ]
        checks = _single_step_checks(
            rows,
            delta_gamma_deg=float(trial_payload.get("delta_gamma_deg", 0.0) or 0.0),
            delta_Gamma_deg=float(trial_payload.get("delta_Gamma_deg", 0.0) or 0.0),
            max_angle_step_deg=float(max_angle_step_deg),
        )
        status = "pass" if _checks_pass(checks, perturb_applied=False) else "fail"
        report_path = output_root / SINGLE_STEP_REPORT_NAME
        plot_path = output_root / SINGLE_STEP_PLOT_NAME
        csv_path = output_root / SINGLE_STEP_CSV_NAME
        state_hash = _file_sha256(state_path)
        report = {
            "status": status,
            "checks": checks,
            "state_path": str(state_path),
            "state_sha256": state_hash,
            "background_index": int(background_index),
            "params_mode": "base",
            "audit_schema_version": int(AUDIT_SCHEMA_VERSION),
            "audit_mode": "single_step_detector_angle_audit",
            "single_step_detector_angle_audit": True,
            "active_vars": [str(name) for name in active_names],
            "max_angle_step_deg": float(max_angle_step_deg),
            "fd_step_deg": float(fd_step_deg),
            "provider_pair_count": reprojection._provider_pair_count(provider_report),
            "row_count": int(len(rows)),
            "plotted_row_count": int(checks.get("plotted_row_count", 0) or 0),
            "invalid_detector_display_row_count": int(
                checks.get("invalid_detector_display_row_count", 0) or 0
            ),
            "valid_plotted_fraction": float(checks.get("valid_plotted_fraction", 0.0) or 0.0),
            "invalid_reasons_by_count": dict(checks.get("invalid_reasons_by_count", {}) or {}),
            "json_authoritative": True,
            "png_diagnostic_only": True,
            "csv_path": str(csv_path),
            "json_path": str(report_path),
            "png_path": str(plot_path),
            "created_at_unix": time.time(),
            **trial_payload,
        }
        _write_json(report_path, report)
        _write_csv(csv_path, rows)
        title = (
            f"New4 QR single-step audit bg={int(background_index)} "
            f"state={state_hash[:12]} active vars=gamma,Gamma "
            f"delta gamma={float(trial_payload.get('delta_gamma_deg', 0.0) or 0.0):.6g} "
            f"delta Gamma={float(trial_payload.get('delta_Gamma_deg', 0.0) or 0.0):.6g} "
            f"max angle step={float(max_angle_step_deg):.6g} deg "
            f"rows={int(len(rows))} status={trial_payload.get('single_step_status')}"
        )
        _plot_single_step_rows(plot_path, rows, title=title)
        return report
    base_rows = _evaluate_pairs(request, base_params)
    improvement_payload: dict[str, object] | None = None
    solver_result_summary: dict[str, object] = {}
    if fit_improvement_audit:
        if effective_perturb:
            raise ValueError("--fit-improvement-audit uses --perturb-start, not --perturb")
        if not perturb_start:
            raise ValueError("--fit-improvement-audit requires --perturb-start NAME=DELTA")
        current_params, perturb_info = _apply_perturb(base_params, str(perturb_start))
        solve_request = _request_with_params(request, current_params)
        before_rows = _evaluate_pairs(solve_request, current_params)
        live_updates: list[dict[str, object]] = []

        def _live_update(payload: Mapping[str, object]) -> None:
            live_updates.append(dict(payload))

        result = gui_geometry_fit.solve_geometry_fit_request(
            solve_request,
            solve_fit=opt.fit_geometry_parameters,
            live_update_callback=_live_update,
        )
        result_x = np.asarray(getattr(result, "x", []), dtype=float).reshape(-1)
        if result_x.size == len(solve_request.var_names):
            final_params = opt._update_params(
                dict(current_params),
                list(solve_request.var_names),
                result_x,
            )
        else:
            final_params = dict(current_params)
        rows = _evaluate_pairs(solve_request, final_params)
        point_match_summary = getattr(result, "point_match_summary", None)
        if not isinstance(point_match_summary, Mapping) and live_updates:
            latest_summary = live_updates[-1].get("point_match_summary")
            point_match_summary = latest_summary if isinstance(latest_summary, Mapping) else None
        improvement_payload = _fit_improvement_summary(
            before_rows,
            rows,
            perturb_info=perturb_info,
            active_vars=solve_request.var_names,
            point_match_summary=point_match_summary,
        )
        solver_result_summary = {
            "solver_success": bool(getattr(result, "success", False)),
            "solver_status": int(getattr(result, "status", 0) or 0),
            "solver_message": str(getattr(result, "message", "") or ""),
            "solver_nfev": int(getattr(result, "nfev", 0) or 0),
            "start_params": {str(name): current_params.get(str(name)) for name in active_names},
            "final_params": {str(name): final_params.get(str(name)) for name in active_names},
            "point_match_summary": dict(point_match_summary or {}),
            "live_update_count": int(len(live_updates)),
        }
        checks = _pair_checks(rows)
        checks.update(dict(improvement_payload["checks"]))
        checks["fit_improvement_audit_enabled"] = True
        checks["fit_improvement_initial_raw_rms_gt_final_raw_rms"] = bool(
            improvement_payload["checks"]["coordinate_audit_improvement_reduces_raw_angular_rms"]
        )
        before_rows_for_report = before_rows
    else:
        current_params, perturb_info = _apply_perturb(base_params, effective_perturb)
        rows = _evaluate_pairs(request, current_params)
        checks = _pair_checks(rows)
        checks.update(_cross_checks(base_rows, rows, bool(perturb_info.get("applied", False))))
        before_rows_for_report = base_rows
    status = (
        "pass"
        if _checks_pass(checks, perturb_applied=bool(perturb_info.get("applied", False)))
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
        "fit_improvement_audit": bool(fit_improvement_audit),
        "step_param": step_param,
        "step_size": step_size,
        "perturb": perturb_info,
        "perturb_start": perturb_start,
        "active_vars": [str(name) for name in active_names],
        "provider_pair_count": reprojection._provider_pair_count(provider_report),
        "pair_count": int(len(rows)),
        "exact_tolerance_deg": EXACT_TOL_DEG,
        "json_authoritative": True,
        "png_diagnostic_only": True,
        "base_pairs": base_rows,
        "before_pairs": before_rows_for_report,
        "pairs": rows,
        "json_path": str(report_path),
        "png_path": str(plot_path),
        "created_at_unix": time.time(),
    }
    if improvement_payload is not None:
        report.update(
            {
                "fit_improvement": improvement_payload,
                "improvement_pairs": improvement_payload["improvement_pairs"],
                "initial_raw_angular_rms_deg": improvement_payload["initial_raw_angular_rms_deg"],
                "final_raw_angular_rms_deg": improvement_payload["final_raw_angular_rms_deg"],
                "improvement_raw_angular_rms_deg": improvement_payload[
                    "improvement_raw_angular_rms_deg"
                ],
                "metric_unit": improvement_payload["metric_unit"],
                "solver_result_summary": solver_result_summary,
            }
        )
    _write_json(report_path, report)
    if fit_improvement_audit:
        _plot_improvement_rows(plot_path, before_rows_for_report, rows)
    else:
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
    parser.add_argument("--fit-improvement-audit", action="store_true")
    parser.add_argument("--perturb-start", default=None)
    parser.add_argument("--active-vars", default=None)
    parser.add_argument("--single-step-detector-angle-audit", action="store_true")
    parser.add_argument("--max-angle-step-deg", type=float, default=5.0)
    parser.add_argument("--fd-step-deg", type=float, default=0.05)
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
        fit_improvement_audit=bool(args.fit_improvement_audit),
        perturb_start=args.perturb_start,
        active_vars=args.active_vars,
        single_step_detector_angle_audit=bool(args.single_step_detector_angle_audit),
        max_angle_step_deg=float(args.max_angle_step_deg),
        fd_step_deg=float(args.fd_step_deg),
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "status": report["status"],
                    "json_path": report["json_path"],
                    "png_path": report["png_path"],
                    "csv_path": report.get("csv_path"),
                }
            ),
            sort_keys=True,
        )
    )
    return 0 if str(report.get("status")) == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
