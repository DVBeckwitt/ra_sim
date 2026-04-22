"""Headless transport-vs-full-recompute validation for Q-group peak metrics."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ra_sim.gui import peak_sensitivity as ps


METRIC_RAY_CLOUD_COM = ps.METRIC_RAY_CLOUD_COM
METRIC_IMAGE_ROI_COM = ps.METRIC_IMAGE_ROI_COM
METRIC_ALL = ps.METRIC_ALL
METRIC_CHOICES: tuple[str, ...] = (METRIC_IMAGE_ROI_COM, METRIC_RAY_CLOUD_COM, METRIC_ALL)

COM_COORDINATES: tuple[str, ...] = ("com_two_theta_deg", "com_phi_deg")
SHAPE_DECISION_COORDINATES: tuple[str, ...] = (
    "sigma_two_theta_deg",
    "sigma_phi_deg",
    "cov_two_theta_phi",
    "major_sigma_deg",
    "minor_sigma_deg",
    "axis_angle_deg",
)
DECISION_COORDINATES: tuple[str, ...] = (*COM_COORDINATES, *SHAPE_DECISION_COORDINATES)
CAKE_GEOMETRY_PARAMETERS: frozenset[str] = frozenset(
    {"corto_detector", "center_x", "center_y", "wavelength"}
)

COMPARISON_LONG_FIELDS: tuple[str, ...] = (
    "comparison",
    "metric",
    "group_key",
    "branch_id",
    "parameter",
    "side",
    "coordinate",
    "baseline_value",
    "full_recompute_value",
    "transported_value",
    "transport_error",
    "abs_transport_error",
    "tolerance",
    "pass",
    "status",
    "point_count",
    "total_weight",
    "baseline_transport_mismatch",
    "full_recompute_status",
    "transport_status",
    "identity_changed",
    "provenance_missing",
    "direct_simulation_fallback_used",
    "sparse_source_row_fallback_used",
    "full_recompute_roi_definition",
    "transport_roi_definition",
)

DECISION_FIELDS: tuple[str, ...] = (
    "comparison",
    "decision_scope",
    "metric",
    "branch_id",
    "parameter",
    "point_count",
    "max_abs_com_two_theta_error",
    "max_abs_com_phi_error",
    "max_abs_shape_error",
    "plus_pass",
    "minus_pass",
    "max_abs_error",
    "can_transport",
    "recommendation",
    "overall_recommendation",
)

DECISION_SCOPE_COORDINATES: dict[str, tuple[str, ...]] = {
    "com": COM_COORDINATES,
    "shape": SHAPE_DECISION_COORDINATES,
}

BASELINE_POINT_FIELDS: tuple[str, ...] = (
    "metric",
    "branch_id",
    "detector_row",
    "detector_col",
    "weight",
    "baseline_two_theta",
    "baseline_phi",
    "raw_intensity",
    "background_value",
)


@dataclass(frozen=True)
class TransportValidationOptions:
    roi_two_theta_half_width: float = 0.5
    roi_phi_half_width: float = 0.5
    background_percentile: float = 50.0
    min_total_weight: float = 0.0
    min_cloud_points: int = 3
    tol_two_theta_deg: float = 0.002
    tol_phi_deg: float = 0.002
    tol_shape: float = 0.005
    relative_step: float = 1.0e-4
    step_mode: str = "default"


@dataclass(frozen=True)
class TransportValidationResult:
    comparison_rows: list[dict[str, object]]
    decision_rows: list[dict[str, object]]
    baseline_point_rows: list[dict[str, object]]
    metadata: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _TransportBundle:
    bundle: object | None
    ai: object | None
    detector_shape: tuple[int, int]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _TransportComputation:
    values: dict[str, float]
    point_count: int
    total_weight: float
    status: str
    low_point_warning: bool
    provenance_digest: str = ""
    provenance_missing: bool = False


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


def _csv_value(value: object) -> object:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    if isinstance(value, (list, tuple, dict)):
        return _stable_json(value)
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_names(metric: str) -> tuple[str, ...]:
    metric_text = str(metric or METRIC_ALL).strip()
    if metric_text == METRIC_ALL:
        return (METRIC_IMAGE_ROI_COM, METRIC_RAY_CLOUD_COM)
    if metric_text not in METRIC_CHOICES:
        raise ValueError(f"Unknown transport metric: {metric!r}")
    return (metric_text,)


def _tol_for_coordinate(coordinate: str, options: TransportValidationOptions) -> float:
    if coordinate == "com_two_theta_deg":
        return float(options.tol_two_theta_deg)
    if coordinate in {"com_phi_deg", "axis_angle_deg", "delta_com_vs_max_phi"}:
        if coordinate == "com_phi_deg":
            return float(options.tol_phi_deg)
        return float(options.tol_shape)
    if coordinate == "delta_com_vs_max_two_theta":
        return float(options.tol_two_theta_deg)
    return float(options.tol_shape)


def _shape_delta(to_value: float, from_value: float, *, coordinate: str) -> float:
    if coordinate in {"com_phi_deg", "delta_com_vs_max_phi"}:
        return ps.wrapped_phi_delta(float(to_value), float(from_value))
    if coordinate == "axis_angle_deg":
        return ((float(to_value) - float(from_value) + 90.0) % 180.0) - 90.0
    if not (math.isfinite(float(to_value)) and math.isfinite(float(from_value))):
        return math.nan
    return float(to_value) - float(from_value)


def _empty_values() -> dict[str, float]:
    return {coordinate: math.nan for coordinate in ps.SHAPE_COORDINATE_NAMES}


def _fallback_flags(metadata: Mapping[str, object] | None) -> tuple[bool, bool]:
    direct = False
    sparse = False

    def visit(value: object) -> None:
        nonlocal direct, sparse
        if isinstance(value, Mapping):
            if bool(value.get("direct_simulation_fallback_used", False)):
                direct = True
            if bool(value.get("source_row_sparse_image_fallback_used", False)):
                sparse = True
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    visit(metadata or {})
    return direct, sparse


def _transport_bundle_for_params(
    evaluator: ps.PeakSensitivityEvaluator,
    params: Mapping[str, object],
    *,
    detector_shape: tuple[int, int] | None = None,
) -> _TransportBundle:
    image, image_diag = ps._real_detector_image_from_context(evaluator.context)
    shape = detector_shape or ps._detector_shape_from_context(evaluator.context, image=image)
    modules = evaluator.context.modules
    ai = ps._build_integrator_for_params(params, shape, modules)
    diagnostics: dict[str, object] = {
        **dict(image_diag),
        "detector_shape": [int(shape[0]), int(shape[1])],
        "transport_used_integrate2d": False,
        "transport_used_refinement": False,
        "transport_used_full_recompute": False,
        "transform_bundle_status": "missing_integrator",
    }
    if ai is None:
        return _TransportBundle(None, None, shape, diagnostics)
    try:
        exact = modules.exact_cake_portable
        radial, raw_azimuth = exact.build_angle_axes(
            npt_rad=1000,
            npt_azim=720,
            tth_min_deg=0.0,
            tth_max_deg=exact.detector_two_theta_max_deg(shape, ai.geometry),
            azimuth_min_deg=-180.0,
            azimuth_max_deg=180.0,
        )
        gui_azimuth = np.asarray(exact.raw_phi_to_gui_phi(raw_azimuth), dtype=np.float64)
        gui_display = np.asarray(gui_azimuth[np.argsort(gui_azimuth, kind="stable")])
        bundle = exact.resolve_cake_transform_bundle(
            ai,
            shape,
            np.asarray(radial, dtype=np.float64),
            gui_azimuth_deg=gui_display,
            raw_azimuth_deg=np.asarray(raw_azimuth, dtype=np.float64),
            require_gui_display_match=True,
        )
    except Exception as exc:
        diagnostics.update(
            {
                "transform_bundle_status": "failed",
                "error_type": type(exc).__name__,
                "error_text": str(exc),
            }
        )
        return _TransportBundle(None, ai, shape, diagnostics)
    diagnostics.update(
        {
            "transform_bundle_status": "ok" if bundle is not None else "missing_bundle",
            "radial_count": int(np.asarray(radial).size),
            "azimuth_count": int(np.asarray(raw_azimuth).size),
        }
    )
    return _TransportBundle(bundle, ai, shape, diagnostics)


def _project_one(
    evaluator: ps.PeakSensitivityEvaluator,
    bundle: object | None,
    col: float,
    row: float,
) -> tuple[float, float]:
    if bundle is None:
        return math.nan, math.nan
    two_theta, phi = evaluator.context.modules.exact_cake_portable.detector_pixel_to_caked_bin(
        bundle,
        float(col),
        float(row),
    )
    two_theta_value = ps.finite_float(two_theta)
    phi_value = ps.finite_float(phi)
    return float(two_theta_value), float(phi_value)


def _values_from_points(
    points: Sequence[tuple[float, float, float]],
    *,
    reference_phi_deg: float,
    refined_max_two_theta_deg: float,
    refined_max_phi_deg: float,
    metric: str,
    options: TransportValidationOptions,
    provenance_digest: str = "",
    provenance_missing: bool = False,
) -> _TransportComputation:
    values, point_count, total_weight, status, low_warning = ps._weighted_caked_shape_values(
        points,
        reference_phi_deg=float(reference_phi_deg),
        refined_max_two_theta_deg=float(refined_max_two_theta_deg),
        refined_max_phi_deg=float(refined_max_phi_deg),
        min_total_weight=float(options.min_total_weight),
        min_cloud_points=int(options.min_cloud_points),
        metric=str(metric),
    )
    if provenance_missing and status == "ok":
        status = "provenance_missing"
    return _TransportComputation(
        values=dict(values),
        point_count=int(point_count),
        total_weight=float(total_weight),
        status=str(status),
        low_point_warning=bool(low_warning),
        provenance_digest=str(provenance_digest),
        provenance_missing=bool(provenance_missing),
    )


def _shape_obs_by_branch(
    observations: Sequence[ps.ShapeMetricObservation],
) -> dict[str, ps.ShapeMetricObservation]:
    return {str(item.branch_id): item for item in observations}


def _peak_obs_by_branch(
    observations: Sequence[ps.PeakObservation],
) -> dict[str, ps.PeakObservation]:
    return {str(item.branch_id): item for item in observations}


def _transport_from_samples(
    evaluator: ps.PeakSensitivityEvaluator,
    samples: Sequence[Mapping[str, object]],
    bundle: object | None,
    *,
    reference_phi_deg: float,
    refined_max_two_theta_deg: float,
    refined_max_phi_deg: float,
    metric: str,
    options: TransportValidationOptions,
    provenance_digest: str = "",
    provenance_missing: bool = False,
) -> _TransportComputation:
    points: list[tuple[float, float, float]] = []
    status = "ok"
    for sample in samples:
        col = ps.finite_float(sample.get("_bundle_detector_col", sample.get("detector_col")))
        row = ps.finite_float(sample.get("_bundle_detector_row", sample.get("detector_row")))
        weight = ps.finite_float(sample.get("weight"))
        two_theta, phi = _project_one(evaluator, bundle, col, row)
        if not (math.isfinite(two_theta) and math.isfinite(phi)):
            status = "projection_failed"
        points.append((two_theta, phi, weight))
    result = _values_from_points(
        points,
        reference_phi_deg=reference_phi_deg,
        refined_max_two_theta_deg=refined_max_two_theta_deg,
        refined_max_phi_deg=refined_max_phi_deg,
        metric=metric,
        options=options,
        provenance_digest=provenance_digest,
        provenance_missing=provenance_missing,
    )
    if status != "ok" and result.status == "ok":
        return _TransportComputation(
            values=result.values,
            point_count=result.point_count,
            total_weight=result.total_weight,
            status=status,
            low_point_warning=result.low_point_warning,
            provenance_digest=result.provenance_digest,
            provenance_missing=result.provenance_missing,
        )
    return result


def build_image_roi_transport_samples(
    image: np.ndarray,
    *,
    refined_two_theta_deg: float,
    refined_phi_deg: float,
    ai: object,
    bundle: object,
    evaluator: ps.PeakSensitivityEvaluator | None,
    branch_id: str,
    options: TransportValidationOptions,
) -> list[dict[str, object]]:
    image_arr = np.asarray(image, dtype=np.float64)
    if image_arr.ndim != 2 or min(image_arr.shape) <= 0:
        return []
    exact = (
        evaluator.context.modules.exact_cake_portable
        if evaluator is not None
        else ps._load_adapter_modules().exact_cake_portable
    )
    rows_grid, cols_grid = np.indices(image_arr.shape, dtype=np.float64)
    two_theta_map, phi_map = exact.detector_points_to_angles(
        cols_grid.reshape(-1),
        rows_grid.reshape(-1),
        ai.geometry,
    )
    two_theta_map = np.asarray(two_theta_map, dtype=np.float64).reshape(image_arr.shape)
    phi_map = np.asarray(phi_map, dtype=np.float64).reshape(image_arr.shape)
    phi_delta = (phi_map - float(refined_phi_deg) + 180.0) % 360.0 - 180.0
    coarse_mask = (
        np.isfinite(two_theta_map)
        & np.isfinite(phi_map)
        & (
            np.abs(two_theta_map - float(refined_two_theta_deg))
            <= float(options.roi_two_theta_half_width) * 1.5
        )
        & (np.abs(phi_delta) <= float(options.roi_phi_half_width) * 1.5)
    )
    candidate_rows, candidate_cols = np.where(coarse_mask)
    selected: list[dict[str, object]] = []
    for row_index, col_index in zip(candidate_rows, candidate_cols):
        if evaluator is None:
            two_theta = float(two_theta_map[row_index, col_index])
            phi = float(phi_map[row_index, col_index])
            bundle_col = float(col_index)
            bundle_row = float(row_index)
        else:
            bundle_col = float(col_index)
            bundle_row = float(row_index)
            native_to_bundle = getattr(
                evaluator.context.manual_dataset_bindings,
                "native_detector_coords_to_bundle_detector_coords",
                None,
            )
            if callable(native_to_bundle):
                try:
                    mapped = native_to_bundle(float(col_index), float(row_index))
                    if isinstance(mapped, tuple) and len(mapped) >= 2:
                        bundle_col = ps.finite_float(mapped[0], bundle_col)
                        bundle_row = ps.finite_float(mapped[1], bundle_row)
                except Exception:
                    pass
            two_theta, phi = _project_one(evaluator, bundle, bundle_col, bundle_row)
        if not (
            math.isfinite(two_theta)
            and math.isfinite(phi)
            and abs(float(two_theta) - float(refined_two_theta_deg))
            <= float(options.roi_two_theta_half_width)
            and abs(ps.wrapped_phi_delta(float(phi), float(refined_phi_deg)))
            <= float(options.roi_phi_half_width)
        ):
            continue
        selected.append(
            {
                "metric": METRIC_IMAGE_ROI_COM,
                "branch_id": str(branch_id),
                "detector_row": float(row_index),
                "detector_col": float(col_index),
                "_bundle_detector_row": float(bundle_row),
                "_bundle_detector_col": float(bundle_col),
                "raw_intensity": float(image_arr[row_index, col_index]),
                "baseline_two_theta": float(two_theta),
                "baseline_phi": float(phi),
            }
        )
    finite = np.asarray(
        [
            sample["raw_intensity"]
            for sample in selected
            if math.isfinite(float(sample["raw_intensity"]))
        ],
        dtype=np.float64,
    )
    if finite.size <= 0:
        return []
    percentile = min(max(float(options.background_percentile), 0.0), 100.0)
    background = float(np.nanpercentile(finite, percentile))
    weighted: list[dict[str, object]] = []
    for sample in selected:
        raw = ps.finite_float(sample.get("raw_intensity"))
        weight = max(float(raw) - background, 0.0)
        if math.isfinite(weight) and weight > 0.0:
            weighted.append({**sample, "weight": float(weight), "background_value": background})
    return weighted


def _baseline_points_csv_rows(samples: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        rows.append(
            {
                "metric": sample.get("metric"),
                "branch_id": sample.get("branch_id"),
                "detector_row": sample.get("detector_row"),
                "detector_col": sample.get("detector_col"),
                "weight": sample.get("weight"),
                "baseline_two_theta": sample.get("baseline_two_theta"),
                "baseline_phi": sample.get("baseline_phi"),
                "raw_intensity": sample.get("raw_intensity"),
                "background_value": sample.get("background_value"),
            }
        )
    return rows


def _source_row_transport_samples(
    evaluator: ps.PeakSensitivityEvaluator,
    group_key: tuple[object, ...],
    *,
    params: Mapping[str, object],
    branch_ids: Sequence[str],
    required_pairs: Sequence[Mapping[str, object]] | None,
    bundle: object | None,
) -> dict[str, list[dict[str, object]]]:
    source_rows_callback = getattr(
        evaluator.context.manual_dataset_bindings,
        "geometry_manual_source_rows_for_background",
        None,
    )
    if not callable(source_rows_callback):
        raise RuntimeError("Captured runtime context has no source-row adapter.")
    rows = ps._call_source_rows_for_background(
        source_rows_callback,
        evaluator.context.background_index,
        params,
        required_pairs=required_pairs,
    )
    group_rows = ps._filter_group_rows(rows, modules=evaluator.context.modules, group_key=group_key)
    branch_filter = {str(branch) for branch in branch_ids}
    buckets: dict[str, list[dict[str, object]]] = {}
    caked_payload = SimpleNamespace(transform_bundle=bundle)
    for row in group_rows:
        item = evaluator._ensure_caked_coordinates(row, group_key, caked_payload)
        branch_id = ps._branch_id_for_row(
            item,
            modules=evaluator.context.modules,
            group_key=group_key,
        )
        if branch_filter and branch_id not in branch_filter:
            continue
        detector_point = ps._bundle_detector_point_from_record(evaluator.context, item)
        weight = ps._source_row_weight(item)
        sample = {
            "metric": METRIC_RAY_CLOUD_COM,
            "branch_id": str(branch_id),
            "detector_row": math.nan,
            "detector_col": math.nan,
            "_bundle_detector_row": math.nan,
            "_bundle_detector_col": math.nan,
            "weight": weight,
            "baseline_two_theta": ps.finite_float(item.get("two_theta_deg", item.get("caked_x"))),
            "baseline_phi": ps.finite_float(item.get("phi_deg", item.get("caked_y"))),
            "raw_intensity": ps.finite_float(item.get("intensity", item.get("weight"))),
            "background_value": math.nan,
            "source_identity": ps._source_row_identity(item),
        }
        if detector_point is not None:
            col, row_value = detector_point
            sample.update(
                {
                    "detector_col": float(col),
                    "detector_row": float(row_value),
                    "_bundle_detector_col": float(col),
                    "_bundle_detector_row": float(row_value),
                }
            )
        buckets.setdefault(str(branch_id), []).append(sample)
    return buckets


def _source_transport_from_samples(
    evaluator: ps.PeakSensitivityEvaluator,
    samples: Sequence[Mapping[str, object]],
    bundle: object | None,
    *,
    parameter_name: str,
    reference_phi_deg: float,
    refined_max_two_theta_deg: float,
    refined_max_phi_deg: float,
    options: TransportValidationOptions,
) -> _TransportComputation:
    provenance_missing = any(not sample.get("source_identity") for sample in samples)
    detector_missing = any(
        not (
            math.isfinite(ps.finite_float(sample.get("_bundle_detector_col")))
            and math.isfinite(ps.finite_float(sample.get("_bundle_detector_row")))
        )
        for sample in samples
    )
    provenance_digest = ps._branch_provenance_digest(
        [
            dict(sample.get("source_identity", {}))
            for sample in samples
            if isinstance(sample.get("source_identity"), Mapping)
        ]
    )
    if parameter_name in CAKE_GEOMETRY_PARAMETERS:
        result = _transport_from_samples(
            evaluator,
            samples,
            bundle,
            reference_phi_deg=reference_phi_deg,
            refined_max_two_theta_deg=refined_max_two_theta_deg,
            refined_max_phi_deg=refined_max_phi_deg,
            metric=METRIC_RAY_CLOUD_COM,
            options=options,
            provenance_digest=provenance_digest,
            provenance_missing=provenance_missing or detector_missing,
        )
        if detector_missing and result.status == "ok":
            return _TransportComputation(
                values=result.values,
                point_count=result.point_count,
                total_weight=result.total_weight,
                status="missing_detector_coordinates",
                low_point_warning=result.low_point_warning,
                provenance_digest=result.provenance_digest,
                provenance_missing=True,
            )
        return result
    points = [
        (
            ps.finite_float(sample.get("baseline_two_theta")),
            ps.finite_float(sample.get("baseline_phi")),
            ps.finite_float(sample.get("weight")),
        )
        for sample in samples
    ]
    return _values_from_points(
        points,
        reference_phi_deg=reference_phi_deg,
        refined_max_two_theta_deg=refined_max_two_theta_deg,
        refined_max_phi_deg=refined_max_phi_deg,
        metric=METRIC_RAY_CLOUD_COM,
        options=options,
        provenance_digest=provenance_digest,
        provenance_missing=provenance_missing,
    )


def _same_roi_full_from_samples(
    evaluator: ps.PeakSensitivityEvaluator,
    samples: Sequence[Mapping[str, object]],
    transport_bundle: _TransportBundle,
    *,
    reference_phi_deg: float,
    refined_max_two_theta_deg: float,
    refined_max_phi_deg: float,
    options: TransportValidationOptions,
) -> _TransportComputation:
    return _transport_from_samples(
        evaluator,
        samples,
        transport_bundle.bundle,
        reference_phi_deg=reference_phi_deg,
        refined_max_two_theta_deg=refined_max_two_theta_deg,
        refined_max_phi_deg=refined_max_phi_deg,
        metric=METRIC_IMAGE_ROI_COM,
        options=options,
    )


def _baseline_transport_mismatch_coordinates(
    baseline_full: ps.ShapeMetricObservation | None,
    baseline_transport: _TransportComputation,
    options: TransportValidationOptions,
) -> set[str]:
    mismatches: set[str] = set()
    if baseline_full is None or baseline_full.status != "ok" or baseline_transport.status != "ok":
        return set(DECISION_COORDINATES)
    for coordinate in DECISION_COORDINATES:
        full_value = ps.finite_float(baseline_full.values.get(coordinate))
        transport_value = ps.finite_float(baseline_transport.values.get(coordinate))
        delta = abs(_shape_delta(transport_value, full_value, coordinate=coordinate))
        if math.isfinite(delta) and delta > _tol_for_coordinate(coordinate, options):
            mismatches.add(coordinate)
        if not math.isfinite(delta):
            mismatches.add(coordinate)
    return mismatches


def _comparison_rows(
    *,
    comparison: str,
    metric: str,
    group_key: tuple[object, ...],
    branch_id: str,
    parameter_name: str,
    side: str,
    baseline: ps.ShapeMetricObservation | None,
    full: ps.ShapeMetricObservation | _TransportComputation | None,
    transported: _TransportComputation,
    options: TransportValidationOptions,
    baseline_transport_mismatches: set[str],
    full_metadata: Mapping[str, object] | None,
    transport_metadata: Mapping[str, object] | None,
    full_recompute_roi_definition: str,
    transport_roi_definition: str,
) -> list[dict[str, object]]:
    full_values = dict(full.values) if full is not None else _empty_values()
    baseline_values = dict(baseline.values) if baseline is not None else _empty_values()
    full_status = str(getattr(full, "status", "missing_full_recompute"))
    transport_status = str(transported.status)
    direct, sparse = _fallback_flags(full_metadata)
    transport_direct, transport_sparse = _fallback_flags(transport_metadata)
    direct = bool(direct or transport_direct)
    sparse = bool(sparse or transport_sparse)
    identity_changed = False
    if baseline is not None and isinstance(full, ps.ShapeMetricObservation):
        identity_changed = bool(full.provenance_key() != baseline.provenance_key())
    provenance_missing = bool(getattr(transported, "provenance_missing", False))
    rows: list[dict[str, object]] = []
    for coordinate in ps.SHAPE_COORDINATE_NAMES:
        full_value = ps.finite_float(full_values.get(coordinate))
        transported_value = ps.finite_float(transported.values.get(coordinate))
        error = _shape_delta(transported_value, full_value, coordinate=coordinate)
        abs_error = abs(float(error)) if math.isfinite(float(error)) else math.nan
        tolerance = _tol_for_coordinate(coordinate, options)
        baseline_transport_mismatch = bool(
            coordinate in baseline_transport_mismatches
            if coordinate in DECISION_COORDINATES
            else False
        )
        coord_pass = bool(
            full_status == "ok"
            and transport_status == "ok"
            and not baseline_transport_mismatch
            and not identity_changed
            and not provenance_missing
            and not direct
            and not sparse
            and math.isfinite(abs_error)
            and abs_error <= tolerance
        )
        status = "ok" if coord_pass else "transport_error"
        if baseline_transport_mismatch:
            status = "baseline_transport_mismatch"
        elif direct or sparse:
            status = "fallback_used"
        elif provenance_missing:
            status = "provenance_missing"
        elif identity_changed:
            status = "identity_changed"
        elif full_status != "ok":
            status = f"full_{full_status}"
        elif transport_status != "ok":
            status = f"transport_{transport_status}"
        rows.append(
            {
                "comparison": str(comparison),
                "metric": str(metric),
                "group_key": _stable_json(group_key),
                "branch_id": str(branch_id),
                "parameter": str(parameter_name),
                "side": str(side),
                "coordinate": str(coordinate),
                "baseline_value": ps.finite_float(baseline_values.get(coordinate)),
                "full_recompute_value": full_value,
                "transported_value": transported_value,
                "transport_error": error,
                "abs_transport_error": abs_error,
                "tolerance": tolerance,
                "pass": coord_pass,
                "status": status,
                "point_count": int(transported.point_count),
                "total_weight": float(transported.total_weight),
                "baseline_transport_mismatch": bool(baseline_transport_mismatch),
                "full_recompute_status": full_status,
                "transport_status": transport_status,
                "identity_changed": bool(identity_changed),
                "provenance_missing": bool(provenance_missing),
                "direct_simulation_fallback_used": bool(direct),
                "sparse_source_row_fallback_used": bool(sparse),
                "full_recompute_roi_definition": str(full_recompute_roi_definition),
                "transport_roi_definition": str(transport_roi_definition),
            }
        )
    return rows


def _max_error(rows: Sequence[Mapping[str, object]], coordinates: set[str]) -> float:
    values = [
        ps.finite_float(row.get("abs_transport_error"))
        for row in rows
        if str(row.get("coordinate")) in coordinates
    ]
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else math.nan


def _side_pass(
    rows: Sequence[Mapping[str, object]],
    side: str,
    coordinates: set[str],
) -> bool:
    side_rows = [
        row
        for row in rows
        if str(row.get("side")) == side and str(row.get("coordinate")) in coordinates
    ]
    return bool(side_rows and all(bool(row.get("pass")) for row in side_rows))


def _fixed_pass(
    rows: Sequence[Mapping[str, object]],
    options: TransportValidationOptions,
    coordinates: set[str],
) -> bool:
    for row in rows:
        coordinate = str(row.get("coordinate"))
        if coordinate not in coordinates:
            continue
        baseline_value = ps.finite_float(row.get("baseline_value"))
        transported_value = ps.finite_float(row.get("transported_value"))
        delta = abs(_shape_delta(transported_value, baseline_value, coordinate=coordinate))
        if not math.isfinite(delta) or delta > _tol_for_coordinate(coordinate, options):
            return False
    return True


def _recommendation_rank(recommendation: object) -> int:
    order = {
        "keep_fixed": 0,
        "transport_points": 1,
        "full_recompute_required": 2,
        "invalid_baseline_mismatch": 3,
        "invalid_insufficient_points": 4,
        "invalid_fallback_used": 5,
    }
    return int(order.get(str(recommendation), 6))


def _set_overall_recommendations(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("comparison")),
            str(row.get("metric")),
            str(row.get("branch_id")),
            str(row.get("parameter")),
        )
        grouped.setdefault(key, []).append(row)
    for group_rows in grouped.values():
        overall = max(group_rows, key=lambda row: _recommendation_rank(row.get("recommendation")))
        recommendation = str(overall.get("recommendation"))
        for row in group_rows:
            row["overall_recommendation"] = recommendation


def build_transport_decision_rows(
    comparison_rows: Sequence[Mapping[str, object]],
    *,
    options: TransportValidationOptions,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = {}
    for row in comparison_rows:
        key = (
            str(row.get("comparison")),
            str(row.get("metric")),
            str(row.get("branch_id")),
            str(row.get("parameter")),
        )
        grouped.setdefault(key, []).append(row)
    decisions: list[dict[str, object]] = []
    for (comparison, metric, branch_id, parameter_name), rows in sorted(grouped.items()):
        for decision_scope, coordinate_names in DECISION_SCOPE_COORDINATES.items():
            coordinates = set(coordinate_names)
            scope_rows = [row for row in rows if str(row.get("coordinate")) in coordinates]
            plus_pass = _side_pass(scope_rows, "plus", coordinates)
            minus_pass = _side_pass(scope_rows, "minus", coordinates)
            max_tth = _max_error(scope_rows, {"com_two_theta_deg"})
            max_phi = _max_error(scope_rows, {"com_phi_deg"})
            max_shape = _max_error(scope_rows, set(SHAPE_DECISION_COORDINATES))
            max_all = _max_error(scope_rows, coordinates)
            point_counts = [
                int(ps.finite_float(row.get("point_count"), 0))
                for row in scope_rows
                if str(row.get("coordinate")) in coordinates
            ]
            point_count = min(point_counts) if point_counts else 0
            any_baseline_mismatch = any(
                bool(row.get("baseline_transport_mismatch")) for row in scope_rows
            )
            any_fallback = any(
                bool(row.get("direct_simulation_fallback_used"))
                or bool(row.get("sparse_source_row_fallback_used"))
                for row in scope_rows
            )
            any_insufficient = any(
                str(row.get("transport_status"))
                in {"insufficient_cloud_points", "insufficient_weight"}
                or str(row.get("full_recompute_status"))
                in {"insufficient_cloud_points", "insufficient_weight"}
                for row in scope_rows
            )
            can_transport = bool(plus_pass and minus_pass)
            if any_insufficient or point_count < int(options.min_cloud_points):
                recommendation = "invalid_insufficient_points"
                can_transport = False
            elif any_baseline_mismatch:
                recommendation = "invalid_baseline_mismatch"
                can_transport = False
            elif any_fallback:
                recommendation = "invalid_fallback_used"
                can_transport = False
            elif not can_transport:
                recommendation = "full_recompute_required"
            elif _fixed_pass(scope_rows, options, coordinates):
                recommendation = "keep_fixed"
            else:
                recommendation = "transport_points"
            decisions.append(
                {
                    "comparison": comparison,
                    "decision_scope": decision_scope,
                    "metric": metric,
                    "branch_id": branch_id,
                    "parameter": parameter_name,
                    "point_count": int(point_count),
                    "max_abs_com_two_theta_error": max_tth,
                    "max_abs_com_phi_error": max_phi,
                    "max_abs_shape_error": max_shape,
                    "plus_pass": bool(plus_pass),
                    "minus_pass": bool(minus_pass),
                    "max_abs_error": max_all,
                    "can_transport": bool(can_transport),
                    "recommendation": recommendation,
                    "overall_recommendation": recommendation,
                }
            )
    _set_overall_recommendations(decisions)
    return decisions


def _evaluate_full_shapes(
    evaluator: ps.PeakSensitivityEvaluator,
    *,
    params_override: Mapping[str, float],
    group_key: tuple[object, ...],
    metric: str,
    refined: Sequence[ps.PeakObservation],
    branch_ids: Sequence[str],
    reference_phi_by_branch: Mapping[str, float],
    options: TransportValidationOptions,
    required_pairs: Sequence[Mapping[str, object]] | None,
) -> tuple[list[ps.ShapeMetricObservation], dict[str, object]]:
    shape_options = ps.ShapeMetricOptions(
        roi_two_theta_half_width=float(options.roi_two_theta_half_width),
        roi_phi_half_width=float(options.roi_phi_half_width),
        background_percentile=float(options.background_percentile),
        min_total_weight=float(options.min_total_weight),
        min_cloud_points=int(options.min_cloud_points),
    )
    observations = evaluator.evaluate_shape_observations(
        params_override,
        group_key,
        metric=metric,
        refined_max_observations=refined,
        branch_ids=branch_ids,
        reference_phi_by_branch=reference_phi_by_branch,
        options=shape_options,
        required_pairs=required_pairs,
    )
    return observations, dict(evaluator._last_eval_metadata)


def _apply_parameter(
    evaluator: ps.PeakSensitivityEvaluator, parameter: ps.PeakSensitivityParameter, side: str
) -> tuple[dict[str, float], dict[str, object]]:
    value = (
        float(parameter.baseline_value) + float(parameter.step)
        if side == "plus"
        else float(parameter.baseline_value) - float(parameter.step)
    )
    override = {parameter.name: value}
    params = ps._apply_parameter_overrides(evaluator.context.params, override)
    return override, params


def run_peak_transport_validation(
    *,
    state_path: str | Path,
    group_key: str | Sequence[object] | tuple[object, ...],
    parameter_names: Sequence[str],
    metric: str = METRIC_ALL,
    options: TransportValidationOptions | None = None,
) -> TransportValidationResult:
    opts = options or TransportValidationOptions()
    evaluator = ps.PeakSensitivityEvaluator(state_path)
    parsed_group_key = ps.parse_group_key(group_key)
    metric_names = _metric_names(metric)
    baseline_refined = evaluator.evaluate_peak_observations({}, parsed_group_key, branch_ids=None)
    baseline_refined_metadata = dict(evaluator._last_eval_metadata)
    baseline_error = ps._baseline_unusable_reason(baseline_refined)
    if baseline_error is not None:
        raise ps.PeakSensitivityEvaluationError(baseline_error)
    branch_ids = [str(item.branch_id) for item in baseline_refined]
    required_pairs = ps._baseline_required_pairs(baseline_refined)
    reference_phi_by_branch = {
        str(item.branch_id): float(item.phi_deg)
        for item in baseline_refined
        if math.isfinite(float(item.phi_deg))
    }
    parameters = ps.build_sensitivity_parameters(
        parameter_names,
        evaluator.baseline_params,
        relative_step=float(opts.relative_step),
        step_mode=str(opts.step_mode),
    )
    baseline_bundle = _transport_bundle_for_params(evaluator, evaluator.context.params)
    image, image_diag = ps._real_detector_image_from_context(evaluator.context)

    baseline_shapes: dict[str, dict[str, ps.ShapeMetricObservation]] = {}
    baseline_shape_metadata: dict[str, object] = {}
    for metric_name in metric_names:
        observations, metadata = _evaluate_full_shapes(
            evaluator,
            params_override={},
            group_key=parsed_group_key,
            metric=metric_name,
            refined=baseline_refined,
            branch_ids=branch_ids,
            reference_phi_by_branch=reference_phi_by_branch,
            options=opts,
            required_pairs=required_pairs,
        )
        baseline_shapes[metric_name] = _shape_obs_by_branch(observations)
        baseline_shape_metadata[metric_name] = metadata

    baseline_peak_by_branch = _peak_obs_by_branch(baseline_refined)
    image_samples_by_branch: dict[str, list[dict[str, object]]] = {}
    source_samples_by_branch: dict[str, list[dict[str, object]]] = {}
    baseline_transport: dict[tuple[str, str], _TransportComputation] = {}
    baseline_point_rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {
        "baseline_refined": baseline_refined_metadata,
        "baseline_shape": baseline_shape_metadata,
        "baseline_transport_bundle": baseline_bundle.diagnostics,
        "real_detector_image": image_diag,
    }

    if (
        METRIC_IMAGE_ROI_COM in metric_names
        and image is not None
        and baseline_bundle.bundle is not None
    ):
        for branch_id in branch_ids:
            refined = baseline_peak_by_branch.get(branch_id)
            if refined is None:
                image_samples_by_branch[branch_id] = []
                continue
            samples = build_image_roi_transport_samples(
                image,
                refined_two_theta_deg=float(refined.two_theta_deg),
                refined_phi_deg=float(refined.phi_deg),
                ai=baseline_bundle.ai,
                bundle=baseline_bundle.bundle,
                evaluator=evaluator,
                branch_id=branch_id,
                options=opts,
            )
            image_samples_by_branch[branch_id] = samples
            baseline_point_rows.extend(_baseline_points_csv_rows(samples))
            baseline_transport[(METRIC_IMAGE_ROI_COM, branch_id)] = _values_from_points(
                [
                    (
                        ps.finite_float(sample.get("baseline_two_theta")),
                        ps.finite_float(sample.get("baseline_phi")),
                        ps.finite_float(sample.get("weight")),
                    )
                    for sample in samples
                ],
                reference_phi_deg=reference_phi_by_branch.get(branch_id, math.nan),
                refined_max_two_theta_deg=float(refined.two_theta_deg),
                refined_max_phi_deg=float(refined.phi_deg),
                metric=METRIC_IMAGE_ROI_COM,
                options=opts,
            )

    if METRIC_RAY_CLOUD_COM in metric_names:
        source_samples_by_branch = _source_row_transport_samples(
            evaluator,
            parsed_group_key,
            params=evaluator.context.params,
            branch_ids=branch_ids,
            required_pairs=required_pairs,
            bundle=baseline_bundle.bundle,
        )
        for branch_id, samples in source_samples_by_branch.items():
            baseline_point_rows.extend(_baseline_points_csv_rows(samples))
            refined = baseline_peak_by_branch.get(branch_id)
            if refined is None:
                continue
            baseline_transport[(METRIC_RAY_CLOUD_COM, branch_id)] = _source_transport_from_samples(
                evaluator,
                samples,
                baseline_bundle.bundle,
                parameter_name="baseline",
                reference_phi_deg=reference_phi_by_branch.get(branch_id, math.nan),
                refined_max_two_theta_deg=float(refined.two_theta_deg),
                refined_max_phi_deg=float(refined.phi_deg),
                options=opts,
            )

    comparison_rows: list[dict[str, object]] = []
    for parameter in parameters:
        side_refined: dict[str, list[ps.PeakObservation]] = {}
        side_refined_metadata: dict[str, object] = {}
        for side in ("plus", "minus"):
            override, _params = _apply_parameter(evaluator, parameter, side)
            side_refined[side] = evaluator.evaluate_peak_observations(
                override,
                parsed_group_key,
                branch_ids=branch_ids,
                required_pairs=required_pairs,
            )
            side_refined_metadata[side] = dict(evaluator._last_eval_metadata)
        for metric_name in metric_names:
            for side in ("plus", "minus"):
                override, params = _apply_parameter(evaluator, parameter, side)
                full_new, full_new_metadata = _evaluate_full_shapes(
                    evaluator,
                    params_override=override,
                    group_key=parsed_group_key,
                    metric=metric_name,
                    refined=side_refined[side],
                    branch_ids=branch_ids,
                    reference_phi_by_branch=reference_phi_by_branch,
                    options=opts,
                    required_pairs=required_pairs,
                )
                full_new_by_branch = _shape_obs_by_branch(full_new)
                side_refined_by_branch = _peak_obs_by_branch(side_refined[side])
                transport_bundle = _transport_bundle_for_params(
                    evaluator,
                    params,
                    detector_shape=baseline_bundle.detector_shape,
                )
                transport_metadata = {
                    "transport": transport_bundle.diagnostics,
                    "refined": side_refined_metadata.get(side, {}),
                }
                full_same_by_branch: dict[str, _TransportComputation] = {}
                full_same_metadata: dict[str, object] = {}
                if metric_name == METRIC_IMAGE_ROI_COM:
                    full_same_metadata = {"transport": transport_bundle.diagnostics}
                    for branch_id in branch_ids:
                        refined = side_refined_by_branch.get(
                            branch_id
                        ) or baseline_peak_by_branch.get(branch_id)
                        if refined is None:
                            continue
                        full_same_by_branch[branch_id] = _same_roi_full_from_samples(
                            evaluator,
                            image_samples_by_branch.get(branch_id, []),
                            transport_bundle,
                            reference_phi_deg=reference_phi_by_branch.get(branch_id, math.nan),
                            refined_max_two_theta_deg=float(refined.two_theta_deg),
                            refined_max_phi_deg=float(refined.phi_deg),
                            options=opts,
                        )

                for branch_id in branch_ids:
                    baseline_full = baseline_shapes.get(metric_name, {}).get(branch_id)
                    baseline_transported = baseline_transport.get((metric_name, branch_id))
                    if baseline_transported is None:
                        baseline_transported = _TransportComputation(
                            values=_empty_values(),
                            point_count=0,
                            total_weight=math.nan,
                            status="missing_baseline_transport",
                        )
                    baseline_mismatches = _baseline_transport_mismatch_coordinates(
                        baseline_full,
                        baseline_transported,
                        opts,
                    )
                    refined = side_refined_by_branch.get(branch_id) or baseline_peak_by_branch.get(
                        branch_id
                    )
                    if refined is None:
                        continue
                    if metric_name == METRIC_IMAGE_ROI_COM:
                        transported = _transport_from_samples(
                            evaluator,
                            image_samples_by_branch.get(branch_id, []),
                            transport_bundle.bundle,
                            reference_phi_deg=reference_phi_by_branch.get(branch_id, math.nan),
                            refined_max_two_theta_deg=float(refined.two_theta_deg),
                            refined_max_phi_deg=float(refined.phi_deg),
                            metric=METRIC_IMAGE_ROI_COM,
                            options=opts,
                        )
                        comparison_rows.extend(
                            _comparison_rows(
                                comparison="transport_vs_same_roi_full",
                                metric=metric_name,
                                group_key=parsed_group_key,
                                branch_id=branch_id,
                                parameter_name=parameter.name,
                                side=side,
                                baseline=baseline_full,
                                full=full_same_by_branch.get(branch_id),
                                transported=transported,
                                options=opts,
                                baseline_transport_mismatches=baseline_mismatches,
                                full_metadata=full_same_metadata,
                                transport_metadata=transport_metadata,
                                full_recompute_roi_definition=(
                                    "same_frozen_detector_pixel_roi_recaked_under_perturbed_geometry"
                                ),
                                transport_roi_definition=(
                                    "same_frozen_detector_pixels_and_weights_reprojected"
                                ),
                            )
                        )
                        comparison_rows.extend(
                            _comparison_rows(
                                comparison="transport_vs_new_roi_full",
                                metric=metric_name,
                                group_key=parsed_group_key,
                                branch_id=branch_id,
                                parameter_name=parameter.name,
                                side=side,
                                baseline=baseline_full,
                                full=full_new_by_branch.get(branch_id),
                                transported=transported,
                                options=opts,
                                baseline_transport_mismatches=baseline_mismatches,
                                full_metadata=full_new_metadata,
                                transport_metadata=transport_metadata,
                                full_recompute_roi_definition=(
                                    "new_perturbed_caked_roi_around_perturbed_refined_peak"
                                ),
                                transport_roi_definition=(
                                    "same_frozen_detector_pixels_and_weights_reprojected"
                                ),
                            )
                        )
                    elif metric_name == METRIC_RAY_CLOUD_COM:
                        transported = _source_transport_from_samples(
                            evaluator,
                            source_samples_by_branch.get(branch_id, []),
                            transport_bundle.bundle,
                            parameter_name=parameter.name,
                            reference_phi_deg=reference_phi_by_branch.get(branch_id, math.nan),
                            refined_max_two_theta_deg=float(refined.two_theta_deg),
                            refined_max_phi_deg=float(refined.phi_deg),
                            options=opts,
                        )
                        comparison_rows.extend(
                            _comparison_rows(
                                comparison="transport_vs_full_recompute",
                                metric=metric_name,
                                group_key=parsed_group_key,
                                branch_id=branch_id,
                                parameter_name=parameter.name,
                                side=side,
                                baseline=baseline_full,
                                full=full_new_by_branch.get(branch_id),
                                transported=transported,
                                options=opts,
                                baseline_transport_mismatches=baseline_mismatches,
                                full_metadata=full_new_metadata,
                                transport_metadata=transport_metadata,
                                full_recompute_roi_definition="perturbed_source_rows",
                                transport_roi_definition=(
                                    "baseline_source_rows_fixed"
                                    if parameter.name not in CAKE_GEOMETRY_PARAMETERS
                                    else "baseline_source_rows_reprojected"
                                ),
                            )
                        )

    decision_rows = build_transport_decision_rows(comparison_rows, options=opts)
    frozen_counts: dict[str, dict[str, int]] = {}
    for row in baseline_point_rows:
        metric_name = str(row.get("metric"))
        branch_id = str(row.get("branch_id"))
        frozen_counts.setdefault(metric_name, {})
        frozen_counts[metric_name][branch_id] = frozen_counts[metric_name].get(branch_id, 0) + 1
    metadata = {
        **dict(evaluator.metadata),
        "status": "ok",
        "group_key": parsed_group_key,
        "requested_metric": metric,
        "metrics": list(metric_names),
        "params": [parameter.name for parameter in parameters],
        "steps": {parameter.name: parameter.step for parameter in parameters},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "transport_used_integrate2d": False,
        "transport_used_refinement": False,
        "transport_used_full_recompute": False,
        "frozen_detector_pixel_count": frozen_counts,
        "baseline_roi_definition": (
            "baseline detector pixels whose baseline caked coords are inside the "
            "baseline refined branch ROI; weights fixed after percentile background subtraction"
        ),
        "full_recompute_roi_definition": {
            "transport_vs_same_roi_full": (
                "same frozen detector-pixel ROI recaked under perturbed geometry"
            ),
            "transport_vs_new_roi_full": ("new perturbed caked ROI around perturbed refined peak"),
            "transport_vs_full_recompute": "perturbed source rows",
        },
        "tolerances": {
            "tol_two_theta_deg": opts.tol_two_theta_deg,
            "tol_phi_deg": opts.tol_phi_deg,
            "tol_shape": opts.tol_shape,
        },
        "options": opts.__dict__,
    }
    return TransportValidationResult(
        comparison_rows=comparison_rows,
        decision_rows=decision_rows,
        baseline_point_rows=baseline_point_rows,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def write_transport_validation_artifacts(
    result: TransportValidationResult,
    outdir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(outdir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "transport_comparison_long": output_dir / "transport_comparison_long.csv",
        "transport_decision": output_dir / "transport_decision.csv",
        "baseline_transport_points": output_dir / "baseline_transport_points.csv",
        "metadata": output_dir / "metadata.json",
        "diagnostics": output_dir / "diagnostics.json",
    }
    with paths["transport_comparison_long"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_LONG_FIELDS))
        writer.writeheader()
        for row in result.comparison_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in COMPARISON_LONG_FIELDS})
    with paths["transport_decision"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DECISION_FIELDS))
        writer.writeheader()
        for row in result.decision_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in DECISION_FIELDS})
    with paths["baseline_transport_points"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BASELINE_POINT_FIELDS))
        writer.writeheader()
        for row in result.baseline_point_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in BASELINE_POINT_FIELDS})
    _write_json(paths["metadata"], result.metadata)
    _write_json(paths["diagnostics"], result.diagnostics)
    return paths
