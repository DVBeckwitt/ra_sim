"""Pure helpers for GUI geometry-fit state and configuration."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import inspect
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from time import perf_counter
from typing import Any

import numpy as np

from ra_sim.debug_controls import (
    geometry_fit_extra_sections_enabled,
    geometry_fit_log_files_enabled,
    register_run_output_path,
    resolve_startup_debug_log_path,
)
from ra_sim.fitting.background_peak_matching import (
    build_background_peak_context,
    match_simulated_peaks_to_peak_context,
)
from ra_sim.fitting.optimization import (
    _detector_pixels_to_fit_space,
    _fit_space_pixel_size_provenance,
    _resolve_fixed_manual_qr_fit_prediction,
)
from ra_sim.gui import geometry_overlay as gui_geometry_overlay
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.simulation.exact_cake_portable import (
    CakeTransformBundle,
    FastAzimuthalIntegrator,
    build_angle_axes,
    build_cake_transform_bundle,
    detector_pixel_to_caked_bin,
    detector_two_theta_max_deg,
    gui_phi_to_raw_phi,
    integrate_detector_to_cake_lut,
    prepare_gui_phi_display,
    raw_phi_to_gui_phi,
    resolve_cake_transform_bundle,
)
from ra_sim.utils.calculations import (
    SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG,
    d_spacing,
    resolve_canonical_branch,
    source_branch_index_from_phi_deg,
    two_theta,
)
from ra_sim.utils.notifications import play_completion_chime

GeometryFitStageCallback = Callable[[str, Mapping[str, object]], None]


GEOMETRY_FIT_PARAM_ORDER = [
    "zb",
    "zs",
    "theta_initial",
    "psi_z",
    "chi",
    "cor_angle",
    "gamma",
    "Gamma",
    "corto_detector",
    "a",
    "c",
    "center_x",
    "center_y",
]

GEOMETRY_FIT_ACCEPT_MAX_RMS_PX = 100.0
GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX = 150.0
GEOMETRY_FIT_LEGACY_REBIND_PIXEL_TIE_TOLERANCE_PX = 1.0e-6
GEOMETRY_FIT_LEGACY_REBIND_CAKED_TIE_TOLERANCE = 1.0e-6
GEOMETRY_FIT_EXACT_CAKE_AXIS_TOLERANCE = 1.0e-9
GEOMETRY_FIT_STORED_POINT_ABS_TOLERANCE_PX = 1.0e-6
GEOMETRY_FIT_RECOMPUTED_REFINEMENT_TOLERANCE_PX = 1.0e-3


def _geometry_fit_float64_vector(
    values: object,
) -> np.ndarray | None:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if vector.size <= 0 or not np.all(np.isfinite(vector)):
        return None
    return vector


def _geometry_fit_detector_shape_2d(
    shape: object,
) -> tuple[int, int] | None:
    try:
        detector_shape = tuple(int(v) for v in tuple(shape)[:2])
    except Exception:
        return None
    if len(detector_shape) < 2 or detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None
    return detector_shape


def normalize_geometry_fit_caked_view_payload(
    payload: Mapping[str, object] | None,
    *,
    detector_shape: Sequence[object] | None = None,
    ai: FastAzimuthalIntegrator | None = None,
) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None

    transform_bundle = payload.get("transform_bundle")
    normalized_shape = _geometry_fit_detector_shape_2d(detector_shape)
    if normalized_shape is None:
        normalized_shape = _geometry_fit_detector_shape_2d(payload.get("detector_shape"))
    if normalized_shape is None and isinstance(transform_bundle, CakeTransformBundle):
        normalized_shape = _geometry_fit_detector_shape_2d(transform_bundle.detector_shape)
    if normalized_shape is None:
        return None

    background_value = payload.get(
        "background",
        payload.get("background_image", payload.get("image")),
    )
    background = None
    if background_value is not None:
        try:
            background = np.asarray(background_value, dtype=np.float64).copy()
        except Exception:
            return None
        if background.ndim != 2 or background.size <= 0:
            return None

    radial_source = payload.get("radial_axis", payload.get("radial"))
    if radial_source is None and isinstance(transform_bundle, CakeTransformBundle):
        radial_source = transform_bundle.radial_deg
    azimuth_source = payload.get("azimuth_axis", payload.get("azimuth"))
    if azimuth_source is None and isinstance(transform_bundle, CakeTransformBundle):
        bundle_gui_axis = np.asarray(
            raw_phi_to_gui_phi(transform_bundle.raw_azimuth_deg),
            dtype=np.float64,
        ).reshape(-1)
        azimuth_source = bundle_gui_axis[np.argsort(bundle_gui_axis, kind="stable")]
    try:
        radial_axis = np.asarray(radial_source, dtype=np.float64).reshape(-1)
        azimuth_axis = np.asarray(azimuth_source, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if (
        radial_axis.size <= 0
        or azimuth_axis.size <= 0
        or not np.all(np.isfinite(radial_axis))
        or not np.all(np.isfinite(azimuth_axis))
    ):
        return None

    raw_azimuth_value = payload.get(
        "raw_azimuth_axis",
        payload.get("raw_azimuth"),
    )
    if raw_azimuth_value is None:
        raw_azimuth_axis = _geometry_fit_raw_azimuth_axis_from_display_axis(azimuth_axis)
    else:
        try:
            raw_azimuth_axis = np.asarray(
                raw_azimuth_value,
                dtype=np.float64,
            ).reshape(-1)
        except Exception:
            return None
    if (
        raw_azimuth_axis is None
        or raw_azimuth_axis.size != azimuth_axis.size
        or not np.all(np.isfinite(raw_azimuth_axis))
    ):
        return None
    raw_azimuth_axis = np.asarray(raw_azimuth_axis, dtype=np.float64).copy()

    canonical_row_permutation = np.asarray(
        np.argsort(raw_phi_to_gui_phi(raw_azimuth_axis), kind="stable"),
        dtype=np.int32,
    )
    raw_to_gui_value = payload.get("raw_to_gui_row_permutation")
    try:
        raw_to_gui_row_permutation = np.asarray(
            raw_to_gui_value if raw_to_gui_value is not None else canonical_row_permutation,
            dtype=np.int32,
        ).reshape(-1)
    except Exception:
        raw_to_gui_row_permutation = canonical_row_permutation
    if raw_to_gui_row_permutation.shape != canonical_row_permutation.shape or not np.array_equal(
        raw_to_gui_row_permutation,
        canonical_row_permutation,
    ):
        raw_to_gui_row_permutation = canonical_row_permutation

    resolved_bundle = resolve_cake_transform_bundle(
        ai,
        normalized_shape,
        radial_axis,
        gui_azimuth_deg=azimuth_axis,
        raw_azimuth_deg=raw_azimuth_axis,
        transform_bundle=(
            transform_bundle if isinstance(transform_bundle, CakeTransformBundle) else None
        ),
        require_gui_display_match=True,
    )

    normalized_payload = {
        "detector_shape": tuple(int(v) for v in normalized_shape),
        "radial_axis": np.asarray(radial_axis, dtype=np.float64).copy(),
        "azimuth_axis": np.asarray(azimuth_axis, dtype=np.float64).copy(),
        "raw_azimuth_axis": raw_azimuth_axis,
        "raw_to_gui_row_permutation": np.asarray(
            raw_to_gui_row_permutation,
            dtype=np.int32,
        ).copy(),
        "transform_bundle": (
            resolved_bundle if isinstance(resolved_bundle, CakeTransformBundle) else None
        ),
    }
    if background is not None:
        normalized_payload["background"] = background
    return normalized_payload


def build_geometry_fit_exact_caked_projection_view(
    *,
    detector_shape: Sequence[object] | None,
    ai: FastAzimuthalIntegrator | None,
    npt_rad: int | None = None,
    npt_azim: int | None = None,
) -> dict[str, object] | None:
    if not isinstance(ai, FastAzimuthalIntegrator):
        return None
    normalized_shape = _geometry_fit_detector_shape_2d(detector_shape)
    if normalized_shape is None:
        return None

    radial_bins = int(max(1, npt_rad)) if npt_rad is not None else 1000
    azimuth_bins = int(max(1, npt_azim)) if npt_azim is not None else 720
    try:
        radial_axis, raw_azimuth_axis = build_angle_axes(
            npt_rad=radial_bins,
            npt_azim=azimuth_bins,
            tth_min_deg=0.0,
            tth_max_deg=detector_two_theta_max_deg(
                normalized_shape,
                ai.geometry,
            ),
            azimuth_min_deg=-180.0,
            azimuth_max_deg=180.0,
        )
        transform_bundle = build_cake_transform_bundle(
            ai,
            normalized_shape,
            radial_axis,
            raw_azimuth_axis,
        )
    except Exception:
        return None
    if not isinstance(transform_bundle, CakeTransformBundle):
        return None

    gui_azimuth_axis = np.asarray(
        raw_phi_to_gui_phi(raw_azimuth_axis),
        dtype=np.float64,
    )
    raw_to_gui_row_permutation = np.asarray(
        np.argsort(gui_azimuth_axis, kind="stable"),
        dtype=np.int32,
    )
    gui_azimuth_axis = gui_azimuth_axis[raw_to_gui_row_permutation]
    return normalize_geometry_fit_caked_view_payload(
        {
            "detector_shape": normalized_shape,
            "radial_axis": radial_axis,
            "azimuth_axis": gui_azimuth_axis,
            "raw_azimuth_axis": raw_azimuth_axis,
            "raw_to_gui_row_permutation": raw_to_gui_row_permutation,
            "transform_bundle": transform_bundle,
        },
        detector_shape=normalized_shape,
        ai=ai,
    )


def build_geometry_fit_caked_view_payload_from_result(
    res2: object,
    *,
    ai: FastAzimuthalIntegrator | None = None,
    detector_shape: Sequence[object] | None = None,
) -> dict[str, object] | None:
    if res2 is None:
        return None

    try:
        raw_azimuth_axis = np.asarray(res2.azimuthal, dtype=float)
        raw_to_gui_row_permutation = np.asarray(
            np.argsort(raw_phi_to_gui_phi(raw_azimuth_axis), kind="stable"),
            dtype=np.int32,
        )
        caked_img, radial_vals, azimuth_vals = prepare_gui_phi_display(res2)
    except Exception:
        return None

    radial_mask = (radial_vals >= 0.0) & (radial_vals <= 90.0)
    sum_signal = None
    sum_normalization = None
    count = None
    for attr_name, local_name in (
        ("sum_signal", "sum_signal"),
        ("sum_normalization", "sum_normalization"),
        ("count", "count"),
    ):
        try:
            value = np.asarray(getattr(res2, attr_name), dtype=float)
        except Exception:
            value = None
        if value is not None and value.shape == np.asarray(res2.intensity).shape:
            value = value[raw_to_gui_row_permutation, :]
            if np.any(radial_mask):
                value = value[:, radial_mask]
        else:
            value = None
        if local_name == "sum_signal":
            sum_signal = value
        elif local_name == "sum_normalization":
            sum_normalization = value
        else:
            count = value
    if np.any(radial_mask):
        radial_vals = radial_vals[radial_mask]
        caked_img = caked_img[:, radial_mask]

    if radial_vals.size:
        radial_min = float(np.min(radial_vals))
        radial_max = float(np.max(radial_vals))
    else:
        radial_min, radial_max = 0.0, 90.0

    if azimuth_vals.size:
        azimuth_min = float(np.min(azimuth_vals))
        azimuth_max = float(np.max(azimuth_vals))
    else:
        azimuth_min, azimuth_max = -180.0, 180.0

    normalized_payload = normalize_geometry_fit_caked_view_payload(
        {
            "background": caked_img,
            "detector_shape": detector_shape,
            "radial_axis": radial_vals,
            "azimuth_axis": azimuth_vals,
            "raw_azimuth_axis": raw_azimuth_axis,
            "raw_to_gui_row_permutation": raw_to_gui_row_permutation,
            "transform_bundle": None,
        },
        detector_shape=detector_shape,
        ai=ai if isinstance(ai, FastAzimuthalIntegrator) else None,
    )
    if not isinstance(normalized_payload, Mapping):
        return None
    return {
        "image": np.asarray(normalized_payload.get("background"), dtype=float),
        "background": np.asarray(normalized_payload.get("background"), dtype=float),
        "sum_signal": sum_signal,
        "sum_normalization": sum_normalization,
        "count": count,
        "radial": np.asarray(normalized_payload.get("radial_axis"), dtype=float),
        "radial_axis": np.asarray(
            normalized_payload.get("radial_axis"),
            dtype=float,
        ),
        "azimuth": np.asarray(normalized_payload.get("azimuth_axis"), dtype=float),
        "azimuth_axis": np.asarray(
            normalized_payload.get("azimuth_axis"),
            dtype=float,
        ),
        "raw_azimuth": np.asarray(
            normalized_payload.get("raw_azimuth_axis"),
            dtype=float,
        ),
        "raw_azimuth_axis": np.asarray(
            normalized_payload.get("raw_azimuth_axis"),
            dtype=float,
        ),
        "raw_to_gui_row_permutation": np.asarray(
            normalized_payload.get("raw_to_gui_row_permutation"),
            dtype=np.int32,
        ),
        "transform_bundle": normalized_payload.get("transform_bundle"),
        "detector_shape": tuple(normalized_payload.get("detector_shape", ())),
        "extent": [
            radial_min,
            radial_max,
            azimuth_min,
            azimuth_max,
        ],
    }


def build_geometry_fit_exact_caked_view_payload(
    detector_image: object,
    *,
    ai: FastAzimuthalIntegrator | None,
    detector_shape: Sequence[object] | None = None,
    npt_rad: int | None = None,
    npt_azim: int | None = None,
) -> dict[str, object] | None:
    if not isinstance(ai, FastAzimuthalIntegrator):
        return None
    try:
        image = np.asarray(detector_image, dtype=np.float64)
    except Exception:
        return None
    if image.ndim != 2:
        return None
    normalized_shape = _geometry_fit_detector_shape_2d(detector_shape)
    if normalized_shape is None:
        normalized_shape = _geometry_fit_detector_shape_2d(image.shape[:2])
    if normalized_shape is None:
        return None

    radial_bins = int(max(1, npt_rad)) if npt_rad is not None else 1000
    azimuth_bins = int(max(1, npt_azim)) if npt_azim is not None else 720
    try:
        res2 = ai.integrate2d(
            image,
            npt_rad=radial_bins,
            npt_azim=azimuth_bins,
            correctSolidAngle=True,
            method="lut",
            unit="2th_deg",
        )
    except Exception:
        return None
    return build_geometry_fit_caked_view_payload_from_result(
        res2,
        ai=ai,
        detector_shape=normalized_shape,
    )


def _fit_detector_coords_to_native_detector_coords(
    detector_col: float,
    detector_row: float,
    *,
    backend_shape: Sequence[object] | None = None,
    orientation_choice: Mapping[str, object] | None = None,
    native_mapper: Callable[..., tuple[float | None, float | None]] | None = None,
    native_shape: Sequence[object] | None = None,
) -> tuple[float | None, float | None]:
    """Convert fit/oriented detector coords back into native detector coords."""

    try:
        backend_col = float(detector_col)
        backend_row = float(detector_row)
    except Exception:
        return None, None
    if not (np.isfinite(backend_col) and np.isfinite(backend_row)):
        return None, None

    backend_shape_2d = _geometry_fit_detector_shape_2d(backend_shape)
    if backend_shape_2d is not None:
        try:
            backend_points = gui_geometry_overlay.inverse_transform_points_orientation(
                [(float(detector_col), float(detector_row))],
                tuple(backend_shape_2d),
                orientation_choice or {},
            )
        except Exception:
            return None, None
        if not backend_points:
            return None, None
        try:
            backend_col = float(backend_points[0][0])
            backend_row = float(backend_points[0][1])
        except Exception:
            return None, None
        if not (np.isfinite(backend_col) and np.isfinite(backend_row)):
            return None, None

    if callable(native_mapper):
        try:
            native_point = native_mapper(
                float(backend_col),
                float(backend_row),
                native_shape,
            )
        except TypeError:
            try:
                native_point = native_mapper(
                    float(backend_col),
                    float(backend_row),
                )
            except Exception:
                native_point = None
        except Exception:
            native_point = None
        if not (
            isinstance(native_point, tuple)
            and len(native_point) >= 2
            and native_point[0] is not None
            and native_point[1] is not None
        ):
            return None, None
        try:
            native_col = float(native_point[0])
            native_row = float(native_point[1])
        except Exception:
            return None, None
        if not (np.isfinite(native_col) and np.isfinite(native_row)):
            return None, None
        return float(native_col), float(native_row)

    return float(backend_col), float(backend_row)


def _geometry_fit_axes_match(
    lhs: object,
    rhs: object,
) -> bool:
    lhs_vec = _geometry_fit_float64_vector(lhs)
    rhs_vec = _geometry_fit_float64_vector(rhs)
    if lhs_vec is None or rhs_vec is None or lhs_vec.shape != rhs_vec.shape:
        return False
    return bool(np.array_equal(lhs_vec, rhs_vec))


def _geometry_fit_raw_azimuth_axis_from_display_axis(
    azimuth_axis: object,
) -> np.ndarray | None:
    gui_azimuth_vec = _geometry_fit_float64_vector(azimuth_axis)
    if gui_azimuth_vec is None:
        return None
    try:
        raw_azimuth_vec = np.sort(
            np.asarray(gui_phi_to_raw_phi(gui_azimuth_vec), dtype=np.float64),
            kind="stable",
        )
    except Exception:
        return None
    return _geometry_fit_float64_vector(raw_azimuth_vec)


def _geometry_fit_projection_signature(
    payload: Mapping[str, object] | None,
) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    try:
        canonical = repr(_geometry_fit_cache_canonical_tuple(payload))
    except Exception:
        return None
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _geometry_fit_transform_driven_param_payload(
    params: Mapping[str, object] | None,
) -> dict[str, object]:
    center_row = None
    center_col = None
    if isinstance(params, Mapping):
        center = list(params.get("center", []))
        if len(center) >= 2:
            try:
                center_row = float(center[0])
                center_col = float(center[1])
            except Exception:
                center_row = None
                center_col = None
    try:
        detector_distance = (
            float(params.get("corto_detector", np.nan))
            if isinstance(params, Mapping)
            else float("nan")
        )
    except Exception:
        detector_distance = float("nan")
    pixel_size_info = _fit_space_pixel_size_provenance(params or {})
    try:
        gamma_value = (
            float(params.get("gamma", np.nan)) if isinstance(params, Mapping) else float("nan")
        )
    except Exception:
        gamma_value = float("nan")
    try:
        Gamma_value = (
            float(params.get("Gamma", np.nan)) if isinstance(params, Mapping) else float("nan")
        )
    except Exception:
        Gamma_value = float("nan")
    try:
        theta_initial_value = (
            float(params.get("theta_initial", np.nan))
            if isinstance(params, Mapping)
            else float("nan")
        )
    except Exception:
        theta_initial_value = float("nan")
    try:
        theta_offset_value = (
            float(params.get("theta_offset", np.nan))
            if isinstance(params, Mapping)
            else float("nan")
        )
    except Exception:
        theta_offset_value = float("nan")
    pixel_size_value = float(pixel_size_info.get("value", np.nan))
    return {
        "center_row": center_row,
        "center_col": center_col,
        "detector_distance": (float(detector_distance) if np.isfinite(detector_distance) else None),
        "pixel_size": (float(pixel_size_value) if np.isfinite(pixel_size_value) else None),
        "pixel_size_source": str(pixel_size_info.get("source", "") or ""),
        "gamma": float(gamma_value) if np.isfinite(gamma_value) else None,
        "Gamma": float(Gamma_value) if np.isfinite(Gamma_value) else None,
        "theta_initial": (float(theta_initial_value) if np.isfinite(theta_initial_value) else None),
        "theta_offset": (float(theta_offset_value) if np.isfinite(theta_offset_value) else None),
    }


def _geometry_fit_exact_caked_bundle_param_payload(
    params: Mapping[str, object] | None,
) -> dict[str, object]:
    transform_payload = _geometry_fit_transform_driven_param_payload(params)
    return {
        "center_row": transform_payload.get("center_row"),
        "center_col": transform_payload.get("center_col"),
        "detector_distance": transform_payload.get("detector_distance"),
        "pixel_size": transform_payload.get("pixel_size"),
    }


def _geometry_fit_cake_bundle_signature(
    bundle: CakeTransformBundle | None,
    *,
    local_params: Mapping[str, object] | None,
) -> str | None:
    if not isinstance(bundle, CakeTransformBundle):
        return None
    return _geometry_fit_projection_signature(
        {
            "detector_shape": _geometry_fit_cache_jsonable(getattr(bundle, "detector_shape", None)),
            "radial_deg": _geometry_fit_cache_jsonable(getattr(bundle, "radial_deg", None)),
            "gui_azimuth_deg": _geometry_fit_cache_jsonable(
                getattr(bundle, "gui_azimuth_deg", None)
            ),
            "raw_azimuth_deg": _geometry_fit_cache_jsonable(
                getattr(bundle, "raw_azimuth_deg", None)
            ),
            "local_params": _geometry_fit_transform_driven_param_payload(local_params),
        }
    )


def project_geometry_fit_native_detector_points_to_caked_space(
    fit_space_projector: Callable[..., object],
    detector_points: Sequence[Sequence[float]],
    *,
    local_params: Mapping[str, object] | None,
    anchor_kind: str = "measured",
) -> dict[str, object]:
    """Project detector/native points through the exact fit-space projector."""

    try:
        point_arr = np.asarray(detector_points, dtype=np.float64).reshape(-1, 2)
    except Exception:
        point_arr = np.empty((0, 2), dtype=np.float64)
    invalid = {
        "two_theta_deg": np.full(point_arr.shape[:1], np.nan, dtype=np.float64),
        "phi_deg": np.full(point_arr.shape[:1], np.nan, dtype=np.float64),
        "caked_points": np.full((point_arr.shape[0], 2), np.nan, dtype=np.float64),
        "projection_input_frame": "native_detector",
        "fit_space_source": "invalid_dataset_fit_space_projector",
        "fit_space_projector_kind": None,
        "cake_bundle_signature": None,
        "fit_space_local_params_signature": None,
        "valid": False,
        "invalid_reason": "invalid_detector_points",
        "native_frame_conversion_source": "",
        "native_frame_conversion_count": 0,
        "native_cols": np.full(point_arr.shape[:1], np.nan, dtype=np.float64),
        "native_rows": np.full(point_arr.shape[:1], np.nan, dtype=np.float64),
        "exact_projector_used": False,
        "caked_projection_source": "fit_space_projector_native_detector",
    }
    if point_arr.size <= 0 or point_arr.shape[1] != 2:
        return invalid
    if not np.all(np.isfinite(point_arr)):
        invalid["invalid_reason"] = "nonfinite_detector_points"
        return invalid
    if not callable(fit_space_projector):
        invalid["invalid_reason"] = "fit_space_projector_unavailable"
        return invalid

    projected = fit_space_projector(
        point_arr[:, 0],
        point_arr[:, 1],
        local_params=dict(local_params) if isinstance(local_params, Mapping) else {},
        anchor_kind=str(anchor_kind),
        input_frame="native_detector",
    )
    if not isinstance(projected, Mapping):
        invalid["invalid_reason"] = "unsupported_projector_result"
        return invalid
    projected_map = dict(projected)
    try:
        two_theta_arr = np.asarray(
            projected_map.get("two_theta_deg"),
            dtype=np.float64,
        ).reshape(-1)
        phi_arr = np.asarray(
            projected_map.get("phi_deg"),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        invalid["invalid_reason"] = "projector_array_conversion_failed"
        return invalid
    try:
        native_cols = np.asarray(
            projected_map.get("native_cols", point_arr[:, 0]),
            dtype=np.float64,
        ).reshape(-1)
        native_rows = np.asarray(
            projected_map.get("native_rows", point_arr[:, 1]),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        invalid["invalid_reason"] = "projector_array_conversion_failed"
        return invalid
    expected_shape = point_arr.shape[:1]
    if (
        two_theta_arr.shape != expected_shape
        or phi_arr.shape != expected_shape
        or native_cols.shape != expected_shape
        or native_rows.shape != expected_shape
    ):
        invalid["invalid_reason"] = "projector_shape_mismatch"
        return invalid
    valid = bool(projected_map.get("valid", False))
    valid = bool(valid and np.all(np.isfinite(two_theta_arr)) and np.all(np.isfinite(phi_arr)))
    return {
        "two_theta_deg": two_theta_arr,
        "phi_deg": phi_arr,
        "caked_points": np.column_stack((two_theta_arr, phi_arr)),
        "projection_input_frame": str(projected_map.get("input_frame", "native_detector") or ""),
        "fit_space_source": str(
            projected_map.get("fit_space_source", "dataset_fit_space_projector") or ""
        ),
        "fit_space_projector_kind": projected_map.get("fit_space_projector_kind"),
        "cake_bundle_signature": projected_map.get("cake_bundle_signature"),
        "fit_space_local_params_signature": projected_map.get("fit_space_local_params_signature"),
        "valid": valid,
        "invalid_reason": (
            None if valid else projected_map.get("invalid_reason", "invalid_projector_output")
        ),
        "native_frame_conversion_source": str(
            projected_map.get("native_frame_conversion_source", "") or ""
        ),
        "native_frame_conversion_count": int(
            projected_map.get("native_frame_conversion_count", 0) or 0
        ),
        "native_cols": native_cols,
        "native_rows": native_rows,
        "exact_projector_used": bool(
            str(projected_map.get("fit_space_projector_kind", "") or "") == "exact_caked_bundle"
        ),
        "caked_projection_source": str(
            projected_map.get(
                "caked_projection_source",
                "fit_space_projector_native_detector",
            )
            or ""
        ),
    }


def _geometry_fit_display_azimuth_axis_from_raw_axis(
    raw_azimuth_axis: object,
) -> np.ndarray | None:
    raw_azimuth_vec = _geometry_fit_float64_vector(raw_azimuth_axis)
    if raw_azimuth_vec is None:
        return None
    try:
        gui_azimuth_vec = np.asarray(
            raw_phi_to_gui_phi(raw_azimuth_vec),
            dtype=np.float64,
        )
    except Exception:
        return None
    if gui_azimuth_vec.shape != raw_azimuth_vec.shape or not np.all(np.isfinite(gui_azimuth_vec)):
        return None
    order = np.argsort(gui_azimuth_vec, kind="stable")
    return _geometry_fit_float64_vector(gui_azimuth_vec[order])


def _geometry_fit_center_from_params(
    params: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if not isinstance(params, Mapping):
        return None
    center_value = params.get("center")
    if isinstance(center_value, Sequence) and len(center_value) >= 2:
        try:
            center_row = float(center_value[0])
            center_col = float(center_value[1])
        except Exception:
            center_row = center_col = float("nan")
        if np.isfinite(center_row) and np.isfinite(center_col):
            return float(center_row), float(center_col)
    try:
        center_row = float(params.get("center_x", np.nan))
        center_col = float(params.get("center_y", np.nan))
    except Exception:
        return None
    if not (np.isfinite(center_row) and np.isfinite(center_col)):
        return None
    return float(center_row), float(center_col)


def _geometry_fit_caked_bundle_matches_display(
    bundle: object,
    *,
    detector_shape: object,
    radial_axis: object,
    azimuth_axis: object,
    raw_azimuth_axis: object,
) -> bool:
    normalized_shape = _geometry_fit_detector_shape_2d(detector_shape)
    radial_vec = _geometry_fit_float64_vector(radial_axis)
    gui_azimuth_vec = _geometry_fit_float64_vector(azimuth_axis)
    raw_azimuth_vec = _geometry_fit_float64_vector(raw_azimuth_axis)
    if (
        normalized_shape is None
        or radial_vec is None
        or gui_azimuth_vec is None
        or raw_azimuth_vec is None
    ):
        return False
    return isinstance(
        resolve_cake_transform_bundle(
            None,
            normalized_shape,
            radial_vec,
            gui_azimuth_deg=gui_azimuth_vec,
            raw_azimuth_deg=raw_azimuth_vec,
            transform_bundle=(bundle if isinstance(bundle, CakeTransformBundle) else None),
            require_gui_display_match=True,
        ),
        CakeTransformBundle,
    )


def _geometry_fit_rebuild_dynamic_reanchor_caked_bundle(
    *,
    detector_shape: object,
    radial_axis: object,
    azimuth_axis: object,
    raw_azimuth_axis: object | None,
    params: Mapping[str, object] | None,
) -> CakeTransformBundle | None:
    normalized_shape = _geometry_fit_detector_shape_2d(detector_shape)
    radial_vec = _geometry_fit_float64_vector(radial_axis)
    gui_azimuth_vec = _geometry_fit_float64_vector(azimuth_axis)
    if normalized_shape is None or radial_vec is None or gui_azimuth_vec is None:
        return None
    raw_azimuth_vec = _geometry_fit_float64_vector(raw_azimuth_axis)
    if raw_azimuth_vec is None:
        raw_azimuth_vec = _geometry_fit_raw_azimuth_axis_from_display_axis(gui_azimuth_vec)
    if raw_azimuth_vec is None or raw_azimuth_vec.shape != gui_azimuth_vec.shape:
        return None
    center = _geometry_fit_center_from_params(params)
    if center is None:
        return None
    try:
        detector_distance = float(
            params.get("corto_detector", np.nan) if isinstance(params, Mapping) else np.nan
        )
    except Exception:
        detector_distance = float("nan")
    pixel_size = float(_fit_space_pixel_size_provenance(params or {}).get("value", np.nan))
    if (
        not np.isfinite(detector_distance)
        or detector_distance <= 0.0
        or not np.isfinite(pixel_size)
        or pixel_size <= 0.0
    ):
        return None
    try:
        ai = FastAzimuthalIntegrator(
            dist=float(detector_distance),
            poni1=float(center[0]) * float(pixel_size),
            poni2=float(center[1]) * float(pixel_size),
            pixel1=float(pixel_size),
            pixel2=float(pixel_size),
        )
        rebuilt = resolve_cake_transform_bundle(
            ai,
            normalized_shape,
            radial_vec,
            gui_azimuth_deg=gui_azimuth_vec,
            raw_azimuth_deg=raw_azimuth_vec,
            require_gui_display_match=True,
        )
    except Exception:
        rebuilt = None
    return rebuilt if isinstance(rebuilt, CakeTransformBundle) else None


def _geometry_fit_resolve_dynamic_reanchor_caked_bundle(
    *,
    detector_shape: object,
    radial_axis: object,
    azimuth_axis: object,
    raw_azimuth_axis: object | None,
    transform_bundle: object,
    params: Mapping[str, object] | None,
) -> CakeTransformBundle | None:
    normalized_shape = _geometry_fit_detector_shape_2d(detector_shape)
    radial_vec = _geometry_fit_float64_vector(radial_axis)
    gui_azimuth_vec = _geometry_fit_float64_vector(azimuth_axis)
    raw_azimuth_vec = _geometry_fit_float64_vector(raw_azimuth_axis)
    if raw_azimuth_vec is None:
        raw_azimuth_vec = _geometry_fit_raw_azimuth_axis_from_display_axis(azimuth_axis)
    if (
        normalized_shape is None
        or radial_vec is None
        or gui_azimuth_vec is None
        or raw_azimuth_vec is None
    ):
        return None
    resolved_bundle = resolve_cake_transform_bundle(
        None,
        normalized_shape,
        radial_vec,
        gui_azimuth_deg=gui_azimuth_vec,
        raw_azimuth_deg=raw_azimuth_vec,
        transform_bundle=(
            transform_bundle if isinstance(transform_bundle, CakeTransformBundle) else None
        ),
        require_gui_display_match=True,
    )
    if isinstance(resolved_bundle, CakeTransformBundle):
        return resolved_bundle
    return _geometry_fit_rebuild_dynamic_reanchor_caked_bundle(
        detector_shape=normalized_shape,
        radial_axis=radial_vec,
        azimuth_axis=gui_azimuth_vec,
        raw_azimuth_axis=raw_azimuth_vec,
        params=params,
    )


def _geometry_fit_caked_payload_exact_bundle(
    payload: Mapping[str, object] | None,
    *,
    detector_shape: object = None,
    params: Mapping[str, object] | None = None,
    require_background: bool = True,
) -> CakeTransformBundle | None:
    """Return a resolved exact cake bundle only when the payload is usable."""

    if not isinstance(payload, Mapping):
        return None
    transform_bundle = payload.get("transform_bundle")
    normalized_shape = _geometry_fit_detector_shape_2d(detector_shape)
    if normalized_shape is None:
        normalized_shape = _geometry_fit_detector_shape_2d(payload.get("detector_shape"))
    if normalized_shape is None and isinstance(transform_bundle, CakeTransformBundle):
        normalized_shape = _geometry_fit_detector_shape_2d(transform_bundle.detector_shape)
    if normalized_shape is None:
        return None

    background = None
    background_value = payload.get(
        "background",
        payload.get("background_image", payload.get("image")),
    )
    if background_value is not None:
        try:
            background = np.asarray(background_value, dtype=np.float64)
        except Exception:
            return None
        if background.ndim != 2 or background.size <= 0 or not np.all(np.isfinite(background)):
            return None
    elif require_background:
        return None

    radial_axis = _geometry_fit_float64_vector(payload.get("radial_axis", payload.get("radial")))
    azimuth_axis = _geometry_fit_float64_vector(payload.get("azimuth_axis", payload.get("azimuth")))
    raw_azimuth_axis = _geometry_fit_float64_vector(
        payload.get("raw_azimuth_axis", payload.get("raw_azimuth"))
    )
    if raw_azimuth_axis is None:
        raw_azimuth_axis = _geometry_fit_raw_azimuth_axis_from_display_axis(azimuth_axis)
    if radial_axis is None or azimuth_axis is None or raw_azimuth_axis is None:
        return None
    if raw_azimuth_axis.shape != azimuth_axis.shape:
        return None
    if background is not None and background.shape != (
        int(azimuth_axis.size),
        int(radial_axis.size),
    ):
        return None
    return _geometry_fit_resolve_dynamic_reanchor_caked_bundle(
        detector_shape=normalized_shape,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        raw_azimuth_axis=raw_azimuth_axis,
        transform_bundle=transform_bundle,
        params=params,
    )


def _geometry_fit_hydrate_exact_caked_payload(
    payload: Mapping[str, object] | None,
    *,
    detector_shape: object = None,
    params: Mapping[str, object] | None = None,
    require_background: bool = True,
) -> dict[str, object] | None:
    exact_bundle = _geometry_fit_caked_payload_exact_bundle(
        payload,
        detector_shape=detector_shape,
        params=params,
        require_background=bool(require_background),
    )
    if not isinstance(exact_bundle, CakeTransformBundle) or not isinstance(payload, Mapping):
        return None
    hydrated = dict(payload)
    hydrated["transform_bundle"] = exact_bundle
    return hydrated


def geometry_fit_all_logging_disabled(
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether all geometry-fit file/debug logging is disabled."""

    return not geometry_fit_log_files_enabled(os.environ if env is None else env)


def build_geometry_fit_log_path(
    *,
    stamp: str,
    log_dir: Path | str | None = None,
    downloads_dir: Path | str | None = None,
) -> Path:
    """Return the shared startup-scoped log path for geometry-fit diagnostics."""

    return resolve_startup_debug_log_path(
        stamp=stamp,
        log_dir=log_dir,
        downloads_dir=downloads_dir,
    )


def build_geometry_fit_trace_path(
    *,
    stamp: str,
    log_dir: Path | str | None = None,
    downloads_dir: Path | str | None = None,
) -> Path:
    """Return the structured JSONL trace path for one geometry-fit run."""

    resolved_log_path = build_geometry_fit_log_path(
        stamp=stamp,
        log_dir=log_dir,
        downloads_dir=downloads_dir,
    )
    return resolved_log_path.with_name(f"geometry_fit_trace_{stamp}.jsonl")


_GEOMETRY_FIT_LOG_SEPARATOR = "=" * 80


def _build_geometry_fit_log_writers(
    log_path: Path | str,
) -> tuple[Path, Callable[[str], None], Callable[[str, Sequence[str]], None]]:
    """Return append-only writers for one geometry-fit log entry."""

    resolved_log_path = Path(log_path)
    logging_disabled = geometry_fit_all_logging_disabled()
    entry_started = False

    def _ensure_entry_started() -> None:
        nonlocal entry_started
        if entry_started or logging_disabled:
            return
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_separator = resolved_log_path.exists() and resolved_log_path.stat().st_size > 0
        with resolved_log_path.open("a", encoding="utf-8") as log_file:
            if needs_separator:
                log_file.write("\n")
            log_file.write(_GEOMETRY_FIT_LOG_SEPARATOR + "\n")
        entry_started = True

    def _log_line(text: str = "") -> None:
        if logging_disabled:
            return
        try:
            _ensure_entry_started()
            with resolved_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(text + "\n")
        except Exception:
            pass

    def _log_section(title: str, lines: Sequence[str]) -> None:
        _log_line(title)
        for line in lines:
            _log_line(f"  {line}")
        _log_line()

    return resolved_log_path, _log_line, _log_section


@dataclass(frozen=True)
class GeometryFitPreparedRun:
    """Prepared inputs and metadata for one manual-pair geometry-fit run."""

    fit_params: dict[str, object]
    selected_background_indices: list[int]
    background_theta_values: list[float]
    joint_background_mode: bool
    current_dataset: dict[str, object]
    dataset_infos: list[dict[str, object]]
    dataset_specs: list[dict[str, object]]
    start_cmd_line: str
    start_log_sections: list[tuple[str, list[str]]]
    max_display_markers: int
    geometry_runtime_cfg: dict[str, object]
    stage_timing_s: dict[str, float] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeManualDatasetBindings:
    """Live manual-pair dataset callbacks and values reused by geometry-fit prep."""

    osc_files: Sequence[object]
    current_background_index: int
    image_size: int
    display_rotate_k: int
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]]
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]]
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None]
    geometry_manual_simulated_peaks_for_params: Callable[..., object]
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]]
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ]
    unrotate_display_peaks: Callable[..., list[dict[str, object]]]
    display_to_native_sim_coords: Callable[..., tuple[float, float]]
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]]
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]]
    orient_image_for_fit: Callable[..., object]
    geometry_manual_project_peaks_to_current_view: (
        Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]] | None
    ) = None
    geometry_manual_project_peaks_for_background_view: (
        Callable[[int, Sequence[dict[str, object]] | None], list[dict[str, object]]] | None
    ) = None
    backend_detector_coords_to_native_detector_coords: (
        Callable[
            [float, float, Sequence[object] | None],
            tuple[float | None, float | None],
        ]
        | None
    ) = None
    native_detector_coords_to_bundle_detector_coords: (
        Callable[[float, float], tuple[float | None, float | None]] | None
    ) = None
    native_detector_coords_to_detector_display_coords: (
        Callable[[float, float], tuple[float | None, float | None] | None] | None
    ) = None
    native_detector_coords_to_detector_display_coords_for_background: (
        Callable[
            [int],
            Callable[[float, float], tuple[float | None, float | None] | None] | None,
        ]
        | None
    ) = None
    geometry_manual_source_rows_for_background: Callable[..., object] | None = None
    geometry_manual_rebuild_source_rows_for_background: Callable[..., object] | None = None
    geometry_manual_last_source_snapshot_diagnostics: Callable[[], Mapping[str, object]] | None = (
        None
    )
    geometry_manual_last_simulation_diagnostics: Callable[[], Mapping[str, object]] | None = None
    geometry_manual_match_config: Callable[[], Mapping[str, object]] | None = None
    pick_uses_caked_space: Callable[[], bool] | None = None
    geometry_manual_caked_view_for_index: Callable[[int], object] | None = None
    geometry_manual_refresh_pair_entry: (
        Callable[[Mapping[str, object] | None], dict[str, object] | None] | None
    ) = None


@dataclass(frozen=True)
class GeometryFitRuntimePreparationBindings:
    """Runtime values and callbacks used to prepare one geometry-fit run."""

    fit_config: Mapping[str, object] | None
    theta_initial: object
    apply_geometry_fit_background_selection: Callable[..., bool]
    current_geometry_fit_background_indices: Callable[..., list[int]]
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    apply_background_theta_metadata: Callable[..., bool]
    current_background_theta_values: Callable[..., list[float]]
    current_geometry_theta_offset: Callable[..., float]
    ensure_geometry_fit_caked_view: Callable[[], None]
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings
    build_runtime_config: Callable[[Mapping[str, object]], dict[str, object]]


@dataclass(frozen=True)
class GeometryFitRuntimeValueBindings:
    """Live Tk/runtime value sources used by geometry-fit runtime helpers."""

    fit_zb_var: object
    fit_zs_var: object
    fit_theta_var: object
    fit_psi_z_var: object
    fit_chi_var: object
    fit_cor_var: object
    fit_gamma_var: object
    fit_Gamma_var: object
    fit_dist_var: object
    fit_a_var: object
    fit_c_var: object
    fit_center_x_var: object
    fit_center_y_var: object
    zb_var: object
    zs_var: object
    theta_initial_var: object
    psi_z_var: object
    chi_var: object
    cor_angle_var: object
    sample_width_var: object
    sample_length_var: object
    sample_depth_var: object
    gamma_var: object
    Gamma_var: object
    corto_detector_var: object
    a_var: object
    c_var: object
    center_x_var: object
    center_y_var: object
    debye_x_var: object
    debye_y_var: object
    geometry_theta_offset_var: object | None
    current_background_index: object
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    current_geometry_theta_offset: Callable[..., float]
    background_theta_for_index: Callable[..., object]
    build_mosaic_params: Callable[..., Mapping[str, object] | None]
    current_optics_mode_flag: Callable[[], object]
    lambda_value: object
    psi: object
    n2: object
    pixel_size_value: object


@dataclass(frozen=True)
class GeometryFitRuntimeSolverInputs:
    """Simulation inputs needed to invoke the live geometry-fit solver."""

    miller: object
    intensities: object
    image_size: int


@dataclass(frozen=True)
class GeometryFitSolverRequest:
    """One concrete geometry-fit solver request."""

    miller: object
    intensities: object
    image_size: int
    params: dict[str, object]
    measured_peaks: object
    var_names: list[str]
    candidate_param_names: list[str] | None
    dataset_specs: list[dict[str, object]] | None
    refinement_config: dict[str, object]
    runtime_safety_note: str | None = None


@dataclass(frozen=True)
class GeometryFitPreparationResult:
    """One geometry-fit preflight result."""

    prepared_run: GeometryFitPreparedRun | None = None
    error_text: str | None = None
    failure_log_sections: list[tuple[str, list[str]]] | None = None
    log_path: Path | None = None


@dataclass(frozen=True)
class GeometryFitSourceRowRebuildResult:
    """Pure source-row rebuild payload returned before any runtime-state commit."""

    background_index: int
    requested_signature: object
    requested_signature_summary: object
    projected_rows: list[dict[str, object]]
    stored_rows: list[dict[str, object]]
    rebuild_source: str | None
    rebuild_attempts: list[str]
    diagnostics: dict[str, object]
    peak_table_lattice: list[object] | None = None
    hit_tables: list[object] | None = None
    source_reflection_indices: list[int] | None = None
    intersection_cache: list[object] | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GeometryFitBackgroundCacheBundle:
    """Job-scoped geometry-fit cache bundle for one prepared background."""

    background_index: int
    requested_signature: object
    requested_signature_summary: object
    background_label: str
    theta_base: float
    theta_initial: float
    projected_rows: list[dict[str, object]]
    stored_rows: list[dict[str, object]]
    cache_source: str | None
    diagnostics: dict[str, object]
    peak_table_lattice: list[object] | None = None
    hit_tables: list[object] | None = None
    intersection_cache: list[object] | None = None
    cache_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GeometryFitPostprocessResult:
    """Pure post-solver geometry-fit analysis results."""

    fitted_params: dict[str, object]
    point_match_summary_lines: list[str]
    pixel_offsets: list[dict[str, object]]
    overlay_records: list[dict[str, object]]
    overlay_state: dict[str, object]
    overlay_diagnostic_lines: list[str]
    frame_warning: str | None
    export_records: list[dict[str, object]]
    save_path: Path
    fit_summary_lines: list[str]
    progress_text: str


@dataclass(frozen=True)
class GeometryFitRuntimeResultBindings:
    """Runtime callback bundle for applying one successful geometry fit."""

    log_section: Callable[[str, Sequence[str]], None]
    capture_undo_state: Callable[[], dict[str, object]]
    apply_result_values: Callable[[Sequence[object], Sequence[object]], None]
    sync_joint_background_theta: Callable[[], None] | None
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    build_profile_cache: Callable[[], dict[str, object]]
    replace_profile_cache: Callable[[dict[str, object]], None]
    push_undo_state: Callable[[dict[str, object] | None], None]
    request_preview_skip_once: Callable[[], None]
    mark_last_simulation_dirty: Callable[[], None]
    schedule_update: Callable[[], None]
    build_fitted_params: Callable[[], dict[str, object]]
    postprocess_result: Callable[[dict[str, object], float], GeometryFitPostprocessResult]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]
    geometry_runtime_cfg: Mapping[str, object] | None = None
    preview_fitted_params: (
        Callable[[Sequence[object], Sequence[object]], dict[str, object]] | None
    ) = None


@dataclass(frozen=True)
class GeometryFitRuntimeUiBindings:
    """Runtime callbacks and UI state sources used during one geometry fit."""

    fit_params: Mapping[str, object] | None
    base_profile_cache: Mapping[str, object] | None
    mosaic_params: Mapping[str, object] | None
    current_ui_params: Callable[[], Mapping[str, object]]
    var_map: Mapping[str, object]
    geometry_theta_offset_var: object | None
    capture_undo_state: Callable[[], dict[str, object]]
    sync_joint_background_theta: Callable[[], None] | None
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    replace_profile_cache: Callable[[dict[str, object]], None]
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None
    push_undo_state: Callable[[dict[str, object] | None], None]
    request_preview_skip_once: Callable[[], None]
    mark_last_simulation_dirty: Callable[[], None]
    schedule_update: Callable[[], None]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimePostprocessConfig:
    """Post-solver inputs needed to analyze and persist one geometry fit."""

    current_background_index: int
    downloads_dir: Path | str
    stamp: str
    log_path: Path | str
    solver_inputs: GeometryFitRuntimeSolverInputs
    sim_display_rotate_k: int
    background_display_rotate_k: int
    simulate_and_compare_hkl: Callable[..., Any]
    aggregate_match_centers: Callable[..., tuple[object, object, object]]
    build_overlay_records: Callable[..., list[dict[str, object]]]
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]]
    log_dir: Path | str | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionSetup:
    """Prepared runtime execution inputs for one geometry-fit run."""

    ui_bindings: GeometryFitRuntimeUiBindings
    postprocess_config: GeometryFitRuntimePostprocessConfig


@dataclass(frozen=True)
class GeometryFitRuntimeValueCallbacks:
    """Bound callbacks that expose live geometry-fit values from runtime."""

    current_var_names: Callable[[], list[str]]
    current_params: Callable[[], dict[str, object]]
    current_ui_params: Callable[[], dict[str, object]]
    var_map: Mapping[str, object]
    build_mosaic_params: Callable[..., Mapping[str, object] | None] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeActionExecutionBindings:
    """Live runtime sources needed to build and run one geometry-fit action."""

    downloads_dir: Path | str
    simulation_runtime_state: Any
    background_runtime_state: Any
    theta_initial_var: Any
    geometry_theta_offset_var: Any | None
    current_ui_params: Callable[[], Mapping[str, object]]
    var_map: Mapping[str, object]
    background_theta_for_index: Callable[..., object]
    refresh_status: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    capture_undo_state: Callable[[], dict[str, object]]
    push_undo_state: Callable[[dict[str, object] | None], None]
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None
    request_preview_skip_once: Callable[[], None]
    schedule_update: Callable[[], None]
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None]
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None]
    set_last_overlay_state: Callable[[dict[str, object]], None]
    set_progress_text: Callable[[str], None]
    cmd_line: Callable[[str], None]
    solver_inputs: GeometryFitRuntimeSolverInputs
    sim_display_rotate_k: int
    background_display_rotate_k: int
    simulate_and_compare_hkl: Callable[..., Any]
    aggregate_match_centers: Callable[..., tuple[object, object, object]]
    build_overlay_records: Callable[..., list[dict[str, object]]]
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]]
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None
    log_dir: Path | str | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeActionBindings:
    """Bound runtime callbacks that drive one top-level geometry-fit action."""

    value_callbacks: GeometryFitRuntimeValueCallbacks
    prepare_bindings_factory: Callable[[Sequence[str]], GeometryFitRuntimePreparationBindings]
    execution_bindings: GeometryFitRuntimeActionExecutionBindings
    solve_fit: Callable[..., object]
    stamp_factory: Callable[[], str]
    flush_ui: Callable[[], None] | None = None


@dataclass(frozen=True)
class GeometryFitRuntimeActionResult:
    """Result metadata for one top-level geometry-fit action invocation."""

    params: Mapping[str, object]
    var_names: list[str]
    preserve_live_theta: bool
    prepare_result: GeometryFitPreparationResult | None = None
    execution_result: GeometryFitRuntimeExecutionResult | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class GeometryFitActionNotice:
    """User-facing notice derived from one top-level geometry-fit action."""

    level: str
    title: str
    message: str


@dataclass(frozen=True)
class GeometryFitRuntimeApplyResult:
    """Result metadata returned after applying one successful geometry fit."""

    accepted: bool
    rejection_reason: str | None
    rms: float
    fitted_params: dict[str, object] | None
    postprocess: GeometryFitPostprocessResult | None


@dataclass(frozen=True)
class GeometryFitRuntimeExecutionResult:
    """Result metadata for one full runtime geometry-fit execution."""

    log_path: Path
    trace_path: Path | None = None
    solver_request: GeometryFitSolverRequest | None = None
    solver_result: object | None = None
    apply_result: GeometryFitRuntimeApplyResult | None = None
    error_text: str | None = None


@dataclass(frozen=True)
class GeometryToolActionRuntimeCallbacks:
    """Bound runtime callbacks for the geometry tool action control cluster."""

    update_fit_history_button_state: Callable[[], None]
    update_manual_pick_button_label: Callable[[], None]
    set_manual_pick_mode: Callable[[bool, str | None], None]
    toggle_manual_pick_mode: Callable[[], None]
    clear_current_manual_pairs: Callable[[], None]


@dataclass(frozen=True)
class GeometryFitRuntimeHistoryCallbacks:
    """Bound runtime callbacks for geometry-fit undo/redo history transitions."""

    undo: Callable[[], bool]
    redo: Callable[[], bool]


def _emit_geometry_fit_stage_event(
    stage_callback: GeometryFitStageCallback | None,
    stage: str,
    **payload: object,
) -> None:
    """Best-effort helper for optional geometry-fit preflight progress callbacks."""

    if not callable(stage_callback):
        return
    try:
        stage_callback(str(stage), dict(payload))
    except Exception:
        return


def _geometry_fit_coerce_nonnegative_index(value: object) -> int | None:
    try:
        idx = int(value)
    except Exception:
        return None
    return int(idx) if idx >= 0 else None


def _geometry_fit_normalized_hkl(
    value: object,
) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        return (
            int(value[0]),
            int(value[1]),
            int(value[2]),
        )
    except Exception:
        return None


def _geometry_fit_trusted_full_reflection_identity(
    entry: Mapping[str, object] | None,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    namespace = str(entry.get("source_reflection_namespace", "") or "").strip().lower()
    reflection_idx = _geometry_fit_coerce_nonnegative_index(entry.get("source_reflection_index"))
    if reflection_idx is None:
        return False
    if namespace in {"full", "full_reflection", "miller"}:
        return True
    return bool(entry.get("source_reflection_is_full", False))


def _geometry_fit_source_branch_resolution(
    entry: Mapping[str, object] | None,
) -> tuple[int | None, str | None]:
    branch_idx, branch_source, _branch_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=False,
    )
    return branch_idx, branch_source


def _geometry_fit_source_branch_reason(
    entry: Mapping[str, object] | None,
) -> str | None:
    _branch_idx, _branch_source, branch_reason = resolve_canonical_branch(
        entry,
        allow_legacy_peak_fallback=False,
    )
    return str(branch_reason) if branch_reason is not None else None


def _geometry_fit_source_branch_index(
    entry: Mapping[str, object] | None,
) -> int | None:
    branch_idx, _branch_source = _geometry_fit_source_branch_resolution(entry)
    return branch_idx


def _geometry_fit_source_row_key(
    entry: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(entry, Mapping):
        return None
    try:
        return (
            int(entry.get("source_table_index")),
            int(entry.get("source_row_index")),
        )
    except Exception:
        return None


def _geometry_fit_source_reflection_row_key(
    entry: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(entry, Mapping):
        return None
    if not _geometry_fit_trusted_full_reflection_identity(entry):
        return None
    try:
        return (
            int(entry.get("source_reflection_index")),
            int(entry.get("source_row_index")),
        )
    except Exception:
        return None


def _geometry_fit_source_peak_key(
    entry: Mapping[str, object] | None,
) -> tuple[int, int] | None:
    if not isinstance(entry, Mapping):
        return None
    branch_idx = _geometry_fit_source_branch_index(entry)
    reflection_idx = _geometry_fit_coerce_nonnegative_index(entry.get("source_reflection_index"))
    if reflection_idx is None:
        reflection_idx = _geometry_fit_coerce_nonnegative_index(entry.get("source_table_index"))
    if branch_idx not in {0, 1} or reflection_idx is None:
        return None
    return int(reflection_idx), int(branch_idx)


def _geometry_fit_source_entry_hkl_matches(
    entry: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
) -> bool:
    entry_hkl = _geometry_fit_normalized_hkl(
        entry.get("hkl") if isinstance(entry, Mapping) else None
    )
    candidate_hkl = _geometry_fit_normalized_hkl(
        candidate.get("hkl") if isinstance(candidate, Mapping) else None
    )
    if candidate_hkl is None:
        return False
    if entry_hkl is None:
        return True
    if tuple(int(v) for v in candidate_hkl) == tuple(int(v) for v in entry_hkl):
        return True
    entry_group = _geometry_fit_group_identity(entry)
    candidate_group = _geometry_fit_group_identity(candidate)
    return _geometry_fit_group_identity_is_q_group(entry_group) and entry_group == candidate_group


def _geometry_fit_group_identity_is_q_group(value: object) -> bool:
    stable = _geometry_fit_stable_group_identity(value)
    return isinstance(stable, tuple) and len(stable) >= 4 and str(stable[0]) == "q_group"


def _geometry_fit_q_group_key_is_zero_qr(value: object) -> bool:
    stable = _geometry_fit_stable_group_identity(value)
    if not isinstance(stable, tuple) or len(stable) < 4:
        return False
    try:
        return str(stable[0]) == "q_group" and int(stable[2]) == 0
    except Exception:
        return False


def _geometry_fit_is_zero_qr_00l(
    entry: Mapping[str, object] | None,
    group_key: object = None,
) -> bool:
    if isinstance(entry, Mapping):
        hkl = _geometry_fit_normalized_hkl(entry.get("hkl"))
        if hkl is not None and int(hkl[0]) == 0 and int(hkl[1]) == 0:
            return True
        for key in ("q_group_key", "source_q_group_key", "branch_group_key"):
            if _geometry_fit_q_group_key_is_zero_qr(entry.get(key)):
                return True
    return _geometry_fit_q_group_key_is_zero_qr(group_key)


def _geometry_fit_source_entry_branch_matches(
    entry: Mapping[str, object] | None,
    candidate: Mapping[str, object] | None,
) -> bool:
    if _geometry_fit_is_zero_qr_00l(entry) and _geometry_fit_is_zero_qr_00l(candidate):
        return True
    entry_branch = _geometry_fit_source_branch_index(entry)
    candidate_branch = _geometry_fit_source_branch_index(candidate)
    if entry_branch is None:
        return not _geometry_fit_trusted_full_reflection_identity(entry)
    return candidate_branch is None or int(entry_branch) == int(candidate_branch)


def _geometry_fit_filter_branch_candidates(
    entry: Mapping[str, object] | None,
    candidates: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    candidate_pool = [
        dict(candidate) for candidate in (candidates or ()) if isinstance(candidate, Mapping)
    ]
    if not candidate_pool:
        return []
    if _geometry_fit_is_zero_qr_00l(entry):
        return candidate_pool
    entry_branch = _geometry_fit_source_branch_index(entry)
    if entry_branch in {0, 1}:
        matched = [
            dict(candidate)
            for candidate in candidate_pool
            if _geometry_fit_source_branch_index(candidate) == int(entry_branch)
        ]
        if matched:
            return matched
        if _geometry_fit_trusted_full_reflection_identity(entry):
            return []
        return candidate_pool
    if _geometry_fit_trusted_full_reflection_identity(entry):
        return []
    return candidate_pool


def _geometry_fit_stable_group_identity(value: object) -> object | None:
    if isinstance(value, np.ndarray):
        try:
            value = value.tolist()
        except Exception:
            return None
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _geometry_fit_stable_group_identity(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_geometry_fit_stable_group_identity(item) for item in value)
    if isinstance(value, set):
        return tuple(
            sorted(
                (_geometry_fit_stable_group_identity(item) for item in value),
                key=repr,
            )
        )
    return value if value is not None else None


def _geometry_fit_group_identity(
    entry: Mapping[str, object] | None,
) -> object | None:
    if not isinstance(entry, Mapping):
        return None
    for key in (
        "q_group_key",
        "source_q_group_key",
        "branch_group_key",
    ):
        value = _geometry_fit_stable_group_identity(entry.get(key))
        if value is not None:
            return value
    return None


def _geometry_fit_entry_source_label(entry: Mapping[str, object] | None) -> str:
    if not isinstance(entry, Mapping):
        return "primary"
    for key in ("q_group_key", "source_q_group_key"):
        group_value = entry.get(key)
        if isinstance(group_value, (list, tuple)) and len(group_value) >= 4:
            try:
                if str(group_value[0]) == "q_group":
                    return gui_manual_geometry.normalize_bragg_qr_source_label(
                        str(group_value[1])
                    )
            except Exception:
                pass
    raw_source = entry.get("source_label")
    if raw_source is not None:
        return gui_manual_geometry.normalize_bragg_qr_source_label(str(raw_source))
    return "primary"


def _geometry_fit_branch_constraint_status(
    entry: Mapping[str, object] | None,
) -> str:
    if not isinstance(entry, Mapping):
        return "missing_required_branch_identity"
    if _geometry_fit_is_zero_qr_00l(entry):
        return "zero_qr_00l_branch_unconstrained"
    branch_idx = _geometry_fit_source_branch_index(entry)
    group_identity = _geometry_fit_group_identity(entry)
    if branch_idx in {0, 1}:
        return "constrained"
    if group_identity is not None:
        return "recovered_from_q_group"
    for key in (
        "source_reflection_index",
        "source_peak_index",
        "source_row_index",
        "source_table_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
    ):
        if entry.get(key) is not None:
            return "missing_required_branch_identity"
    return "unconstrained_missing_branch"


def _geometry_fit_required_branch_group_key(
    entry: Mapping[str, object] | None,
) -> tuple[tuple[int, int, int], int | None, object | None] | None:
    if not isinstance(entry, Mapping):
        return None
    normalized_hkl = _geometry_fit_normalized_hkl(entry.get("hkl"))
    if normalized_hkl is None:
        return None
    branch_idx = _geometry_fit_source_branch_index(entry)
    normalized_branch = (
        None
        if _geometry_fit_is_zero_qr_00l(entry)
        else int(branch_idx)
        if branch_idx in {0, 1}
        else None
    )
    return (
        normalized_hkl,
        normalized_branch,
        _geometry_fit_group_identity(entry),
    )


_GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT = "00l_collapsed"
_GEOMETRY_FIT_SOURCE_COVERAGE_ALIAS_FIELDS = (
    "source_coverage_aliases",
    "source_coverage_keys",
    "collapsed_source_coverage_keys",
)


def _geometry_fit_source_coverage_key_from_parts(
    hkl: object,
    branch_index: object,
    q_group_key: object,
    *,
    entry: Mapping[str, object] | None = None,
) -> tuple[tuple[int, int, int], object | None, object | None] | None:
    normalized_hkl = _geometry_fit_normalized_hkl(hkl)
    if normalized_hkl is None:
        return None
    stable_group = _geometry_fit_stable_group_identity(q_group_key)
    if _geometry_fit_is_zero_qr_00l(entry, stable_group):
        branch_slot: object | None = _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT
    else:
        try:
            branch_value = int(branch_index)
        except Exception:
            branch_value = -1
        branch_slot = int(branch_value) if branch_value in {0, 1} else None
    return (tuple(int(v) for v in normalized_hkl), branch_slot, stable_group)


def normalize_new4_source_coverage_key(
    row_or_target: Mapping[str, object] | Sequence[object] | None,
) -> tuple[tuple[int, int, int], object | None, object | None] | None:
    """Return the shared source-coverage identity used by New4 preflight gates."""

    if isinstance(row_or_target, Mapping):
        raw_required_key = row_or_target.get("required_branch_group_key")
        if isinstance(raw_required_key, Sequence) and not isinstance(
            raw_required_key,
            (str, bytes),
        ):
            normalized = normalize_new4_source_coverage_key(raw_required_key)
            if normalized is not None:
                return normalized
        hkl = (
            row_or_target.get("normalized_hkl")
            if row_or_target.get("normalized_hkl") is not None
            else row_or_target.get(
                "hkl",
                row_or_target.get("source_hkl", row_or_target.get("label")),
            )
        )
        branch_idx = _geometry_fit_source_branch_index(row_or_target)
        if branch_idx not in {0, 1}:
            branch_idx = _geometry_fit_coerce_nonnegative_index(row_or_target.get("branch_index"))
        if branch_idx not in {0, 1}:
            branch_idx = None
        return _geometry_fit_source_coverage_key_from_parts(
            hkl,
            branch_idx,
            _geometry_fit_group_identity(row_or_target),
            entry=row_or_target,
        )
    if isinstance(row_or_target, Sequence) and not isinstance(row_or_target, (str, bytes)):
        raw_key = list(row_or_target)
        if len(raw_key) >= 3:
            return _geometry_fit_source_coverage_key_from_parts(
                raw_key[0],
                raw_key[1],
                raw_key[2],
            )
    return None


def _geometry_fit_source_coverage_key_payload(
    key: tuple[tuple[int, int, int], object | None, object | None] | None,
) -> dict[str, object] | None:
    if key is None:
        return None
    branch_slot = key[1]
    return {
        "hkl": tuple(int(v) for v in key[0]),
        "branch_slot": branch_slot,
        "branch_index": int(branch_slot) if branch_slot in {0, 1} else None,
        "q_group_key": _geometry_fit_cache_jsonable(key[2]),
    }


def _geometry_fit_apply_source_coverage_identity(row: dict[str, object]) -> None:
    key = normalize_new4_source_coverage_key(row)
    payload = _geometry_fit_source_coverage_key_payload(key)
    if not isinstance(payload, Mapping):
        return

    aliases = list(row.get("source_coverage_aliases") or [])
    payload_dict = dict(payload)
    if payload_dict not in aliases:
        aliases.append(payload_dict)
    row["source_coverage_aliases"] = aliases

    if row.get("normalized_hkl") is None and payload.get("hkl") is not None:
        row["normalized_hkl"] = tuple(int(value) for value in payload["hkl"])
    if row.get("q_group_key") is None and payload.get("q_group_key") is not None:
        row["q_group_key"] = payload.get("q_group_key")

    branch_slot = payload.get("branch_slot")
    if branch_slot is not None:
        row["physical_branch_slot"] = branch_slot
    if branch_slot == _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT:
        row["source_branch_index_namespace"] = "00l_collapsed"
        row["is_00l_collapsed"] = True
        row.setdefault("source_peak_index", 0)
    elif branch_slot in {0, 1}:
        row.setdefault("source_branch_index_namespace", "physical_branch_slot")

    row["fit_qr_branch_key"] = {
        "q_group_key": _geometry_fit_cache_jsonable(row.get("q_group_key")),
        "hkl": _geometry_fit_cache_jsonable(row.get("normalized_hkl", row.get("hkl"))),
        "physical_branch_slot": row.get("physical_branch_slot"),
        "source_branch_index": row.get("source_branch_index"),
        "source_peak_index": row.get("source_peak_index"),
    }


def _geometry_fit_source_coverage_alias_keys(
    entry: Mapping[str, object] | None,
) -> set[tuple[tuple[int, int, int], object | None, object | None]]:
    keys: set[tuple[tuple[int, int, int], object | None, object | None]] = set()
    direct_key = normalize_new4_source_coverage_key(entry)
    if direct_key is not None:
        keys.add(direct_key)
    if not isinstance(entry, Mapping):
        return keys
    for field_name in _GEOMETRY_FIT_SOURCE_COVERAGE_ALIAS_FIELDS:
        raw_aliases = entry.get(field_name)
        if raw_aliases is None:
            continue
        alias_items: Sequence[object]
        if isinstance(raw_aliases, Mapping):
            alias_items = [raw_aliases]
        elif isinstance(raw_aliases, Sequence) and not isinstance(raw_aliases, (str, bytes)):
            alias_items = list(raw_aliases)
        else:
            continue
        for raw_alias in alias_items:
            alias_key = normalize_new4_source_coverage_key(raw_alias)  # type: ignore[arg-type]
            if alias_key is not None:
                keys.add(alias_key)
    return keys


def _geometry_fit_required_branch_group_keys(
    required_manual_fit_targets: Sequence[Mapping[str, object]] | None,
) -> list[tuple[tuple[int, int, int], int | None, object | None]]:
    seen: set[tuple[tuple[int, int, int], int | None, object | None]] = set()
    ordered: list[tuple[tuple[int, int, int], int | None, object | None]] = []
    for raw_target in required_manual_fit_targets or ():
        if not isinstance(raw_target, Mapping):
            continue
        key = raw_target.get("required_branch_group_key")
        if not (
            isinstance(key, (list, tuple))
            and len(key) >= 3
            and isinstance(key[0], (list, tuple, np.ndarray))
        ):
            key = _geometry_fit_required_branch_group_key(raw_target)
        if key is None:
            continue
        normalized_key = (
            tuple(int(v) for v in tuple(key[0])[:3]),
            (int(key[1]) if key[1] is not None and int(key[1]) in {0, 1} else None),
            _geometry_fit_stable_group_identity(key[2]),
        )
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        ordered.append(normalized_key)
    return ordered


def _geometry_fit_digest_payload(value: object) -> str:
    try:
        canonical = json.dumps(
            _geometry_fit_cache_jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        canonical = repr(value)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(canonical.encode("utf-8", "backslashreplace"))
    return str(hasher.hexdigest())


def _geometry_fit_required_branch_group_keys_digest(
    required_branch_group_keys: Sequence[object] | None,
    *,
    background_index: int,
    requested_signature: object = None,
    requested_signature_summary: object = None,
    preflight_mode: str = "full",
    consumer: str | None = None,
    projection_view_mode: str | None = None,
    projection_view_signature: object = None,
) -> str:
    return _geometry_fit_digest_payload(
        {
            "background_index": int(background_index),
            "preflight_mode": str(preflight_mode or "full"),
            "consumer": str(consumer or "unspecified"),
            "projection_view_mode": (
                str(projection_view_mode).strip().lower()
                if projection_view_mode is not None
                else None
            ),
            "projection_view_signature": _geometry_fit_stable_projection_view_signature(
                projection_view_signature
            ),
            "requested_signature": _geometry_fit_cache_jsonable(requested_signature),
            "requested_signature_summary": _geometry_fit_cache_jsonable(
                requested_signature_summary
            ),
            "required_branch_group_keys": list(required_branch_group_keys or ()),
        }
    )


def _geometry_fit_stable_projection_view_signature(
    projection_view_signature: object,
) -> object:
    normalized_signature = (
        dict(projection_view_signature)
        if isinstance(projection_view_signature, Mapping)
        else _geometry_fit_cache_jsonable(projection_view_signature)
    )
    if not isinstance(normalized_signature, Mapping):
        return normalized_signature
    normalized_mode = str(normalized_signature.get("mode") or "").strip().lower()
    if normalized_mode != "detector":
        return normalized_signature
    stable_signature = dict(normalized_signature)
    stable_signature.pop("analysis_bins", None)
    stable_signature.pop("current_background_index", None)
    return stable_signature


def _geometry_fit_projection_signature_available(
    projection_view_signature: object,
) -> bool:
    normalized_signature = (
        dict(projection_view_signature)
        if isinstance(projection_view_signature, Mapping)
        else _geometry_fit_cache_jsonable(projection_view_signature)
    )
    if not isinstance(normalized_signature, Mapping):
        return False
    return bool(normalized_signature.get("available", True))


def _geometry_fit_projected_rows_cacheable(
    *,
    background_index: int,
    projected_rows: Sequence[object] | None,
    projection_view_signature: object,
    requested_projection_view_signature: object,
    consumer: str | None,
) -> bool:
    normalized_signature = (
        dict(projection_view_signature)
        if isinstance(projection_view_signature, Mapping)
        else _geometry_fit_cache_jsonable(projection_view_signature)
    )
    requested_signature = (
        dict(requested_projection_view_signature)
        if isinstance(requested_projection_view_signature, Mapping)
        else _geometry_fit_cache_jsonable(requested_projection_view_signature)
    )
    stable_signature = _geometry_fit_stable_projection_view_signature(normalized_signature)
    stable_requested_signature = _geometry_fit_stable_projection_view_signature(requested_signature)
    if not projected_rows:
        return False
    if not isinstance(normalized_signature, Mapping):
        return False
    if not bool(str(consumer or "").strip()):
        return False
    if not bool(normalized_signature.get("available", True)):
        return False
    if int(normalized_signature.get("background_index", background_index)) != int(background_index):
        return False
    if stable_signature != stable_requested_signature:
        return False
    return True


def _geometry_fit_manual_target_scoring_digest(
    required_manual_fit_targets: Sequence[Mapping[str, object]] | None,
) -> str:
    scoring_targets: list[dict[str, object]] = []
    for raw_target in required_manual_fit_targets or ():
        if not isinstance(raw_target, Mapping):
            continue
        scoring_targets.append(
            {
                "pair_id": raw_target.get("pair_id"),
                "overlay_match_index": raw_target.get("overlay_match_index"),
                "saved_background_current_view_point": raw_target.get(
                    "saved_background_current_view_point"
                ),
                "saved_background_current_view_frame": raw_target.get(
                    "saved_background_current_view_frame"
                ),
            }
        )
    return _geometry_fit_digest_payload(scoring_targets)


def _entry_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    saved_point = entry.get("saved_background_current_view_point")
    if isinstance(saved_point, (list, tuple, np.ndarray)) and len(saved_point) >= 2:
        try:
            saved_col = float(saved_point[0])
            saved_row = float(saved_point[1])
        except Exception:
            saved_col = float("nan")
            saved_row = float("nan")
        if np.isfinite(saved_col) and np.isfinite(saved_row):
            return float(saved_col), float(saved_row)
    point_keys = [
        ("display_col", "display_row"),
        ("x", "y"),
        ("raw_x", "raw_y"),
    ]
    if not _entry_has_stale_caked_fields(entry):
        point_keys.extend(
            [
                ("caked_x", "caked_y"),
                ("raw_caked_x", "raw_caked_y"),
            ]
        )
    for key_x, key_y in point_keys:
        try:
            col = float(entry.get(key_x, np.nan))
            row = float(entry.get(key_y, np.nan))
        except Exception:
            continue
        if np.isfinite(col) and np.isfinite(row):
            return float(col), float(row)
    return None


def _background_current_view_frame(
    entry: Mapping[str, object] | None,
) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    explicit_frame = entry.get("saved_background_current_view_frame")
    if isinstance(explicit_frame, str) and explicit_frame.strip():
        return str(explicit_frame)
    has_caked_point = False
    if not _entry_has_stale_caked_fields(entry):
        has_caked_point = any(
            _finite is not None
            for _finite in (
                _entry_display_point(
                    {
                        "saved_background_current_view_point": (
                            entry.get("caked_x"),
                            entry.get("caked_y"),
                        )
                    }
                ),
                _entry_display_point(
                    {
                        "saved_background_current_view_point": (
                            entry.get("raw_caked_x"),
                            entry.get("raw_caked_y"),
                        )
                    }
                ),
            )
        )
    if has_caked_point:
        return "caked_display"
    return "current_view_display" if _entry_display_point(entry) is not None else None


def _entry_has_stale_caked_fields(entry: object) -> bool:
    return bool(isinstance(entry, Mapping) and entry.get("stale_caked_fields", False))


def _source_rebinding_allows_saved_current_view_point(
    entry: Mapping[str, object],
) -> bool:
    frame = entry.get("saved_background_current_view_frame")
    frame_text = str(frame or "").strip().lower()
    if not frame_text:
        return True
    if any(token in frame_text for token in ("caked", "two_theta", "theta_phi", "qr", "qz")):
        return False
    normalized = gui_manual_geometry.normalize_geometry_point_frame(frame_text)
    return normalized in {"display", "detector_native"} or frame_text in {
        "current_view",
        "current_view_display",
        "display",
        "detector_display",
        "fit_detector",
        "detector_native",
        "native_detector",
        "native_detector_coords",
        "background_detector",
    }


def _entry_source_rebinding_display_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    if _source_rebinding_allows_saved_current_view_point(entry):
        try:
            point = entry.get("saved_background_current_view_point")
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                x_val = float(point[0])
                y_val = float(point[1])
                if np.isfinite(x_val) and np.isfinite(y_val):
                    return float(x_val), float(y_val)
        except Exception:
            pass
    for key_x, key_y in (
        ("display_col", "display_row"),
        ("x", "y"),
        ("raw_x", "raw_y"),
    ):
        try:
            x_val = float(entry.get(key_x, np.nan))
            y_val = float(entry.get(key_y, np.nan))
        except Exception:
            continue
        if np.isfinite(x_val) and np.isfinite(y_val):
            return float(x_val), float(y_val)
    return None


def _source_rebinding_background_point_and_frame(
    entry: Mapping[str, object] | None,
) -> tuple[tuple[float, float] | None, str | None]:
    point = _entry_source_rebinding_display_point(entry)
    return point, "current_view_display" if point is not None else None


def _geometry_fit_normalize_point_frame(frame: object) -> str:
    return gui_manual_geometry.normalize_geometry_point_frame(frame)


def _geometry_fit_point_list(point: object) -> list[float] | None:
    if isinstance(point, np.ndarray):
        point = point.tolist()
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x_val = float(point[0])
        y_val = float(point[1])
    except Exception:
        return None
    if not (np.isfinite(x_val) and np.isfinite(y_val)):
        return None
    return [float(x_val), float(y_val)]


def _geometry_fit_sim_native_source_is_display_to_native(source: object) -> bool:
    text = str(source or "").strip().lower()
    return bool(
        text
        and (
            "display_to_native" in text
            or "display->native" in text
            or "display native roundtrip" in text
        )
    )


def _geometry_fit_points_match(
    left: object,
    right: object,
    *,
    tol: float = GEOMETRY_FIT_STORED_POINT_ABS_TOLERANCE_PX,
) -> bool:
    left_point = _geometry_fit_point_list(left)
    right_point = _geometry_fit_point_list(right)
    if left_point is None or right_point is None:
        return False
    return bool(
        abs(float(left_point[0]) - float(right_point[0])) <= float(tol)
        and abs(float(left_point[1]) - float(right_point[1])) <= float(tol)
    )


def _geometry_fit_jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_geometry_fit_jsonable(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_geometry_fit_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_geometry_fit_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _geometry_fit_jsonable(raw_value)
            for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
            if raw_value is not None
        }
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except Exception:
        return str(value)
    if np.isfinite(numeric) and float(numeric).is_integer():
        return int(numeric)
    return float(numeric)


def _geometry_fit_report_jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return _geometry_fit_report_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_geometry_fit_report_jsonable(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_geometry_fit_report_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_geometry_fit_report_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _geometry_fit_report_jsonable(raw_value)
            for key, raw_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return None
    if float(numeric).is_integer():
        return int(numeric)
    return float(numeric)


def _geometry_fit_pair_fingerprint(
    pair: Mapping[str, object],
    *,
    index_key: str,
) -> str:
    payload = {
        "pair_index": pair.get(index_key),
        "background_point": pair.get("background_point"),
        "background_frame": pair.get("background_frame"),
        "simulated_point": pair.get("simulated_point"),
        "simulated_frame": pair.get("simulated_frame"),
        "normalized_hkl": pair.get("normalized_hkl"),
        "q_group_key": pair.get("q_group_key"),
        "branch_group_key": pair.get("branch_group_key"),
        "source_branch_index": pair.get("source_branch_index"),
        "selected_source_identity": pair.get("selected_source_identity_canonical"),
    }
    rendered = json.dumps(
        _geometry_fit_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _geometry_fit_truth_by_order_key(
    truth_pairs: Sequence[object] | None,
) -> dict[tuple[int, int], dict[str, object]]:
    out: dict[tuple[int, int], dict[str, object]] = {}
    for raw_pair in truth_pairs or ():
        if not isinstance(raw_pair, Mapping):
            continue
        raw_key = raw_pair.get("manual_pair_order_key")
        if not isinstance(raw_key, (list, tuple)) or len(raw_key) < 2:
            continue
        try:
            key = (int(raw_key[0]), int(raw_key[1]))
        except Exception:
            continue
        out[key] = dict(raw_pair)
    return out


def _geometry_fit_handoff_identity(
    measured_entry: Mapping[str, object] | None,
    initial_entry: Mapping[str, object] | None,
) -> dict[str, object]:
    for entry in (initial_entry, measured_entry):
        if not isinstance(entry, Mapping):
            continue
        identity = entry.get("selected_source_identity_canonical")
        if isinstance(identity, Mapping) and identity:
            return dict(identity)
    merged: dict[str, object] = {}
    if isinstance(measured_entry, Mapping):
        merged.update(measured_entry)
    if isinstance(initial_entry, Mapping):
        merged.update(initial_entry)
    return gui_manual_geometry.canonical_geometry_source_identity(merged)


def _geometry_fit_handoff_pair(
    *,
    dataset_index: int,
    pair_index: int,
    provider_pair: Mapping[str, object] | None,
    measured_entry: Mapping[str, object] | None,
    initial_entry: Mapping[str, object] | None,
) -> dict[str, object]:
    provider = provider_pair if isinstance(provider_pair, Mapping) else {}
    measured = measured_entry if isinstance(measured_entry, Mapping) else {}
    initial = initial_entry if isinstance(initial_entry, Mapping) else {}

    provider_background_frame = _geometry_fit_normalize_point_frame(
        measured.get(
            "provider_background_frame",
            initial.get(
                "provider_background_frame",
                provider.get("background_frame"),
            ),
        )
    )
    background_point = _geometry_fit_point_list((measured.get("x"), measured.get("y")))
    if background_point is None:
        background_point = _geometry_fit_point_list(
            (measured.get("display_col"), measured.get("display_row"))
        )
    background_frame = _geometry_fit_normalize_point_frame(
        measured.get("provider_background_frame")
        or measured.get("detector_input_frame")
        or provider_background_frame
        or "display"
    )
    if background_point is None and provider_background_frame == "display":
        background_point = _geometry_fit_point_list(initial.get("bg_display"))
        background_frame = "display" if background_point is not None else background_frame
    if background_point is None and provider_background_frame == "detector_native":
        background_point = _geometry_fit_point_list(
            (
                initial.get("background_detector_x"),
                initial.get("background_detector_y"),
            )
        )
        background_frame = "detector_native" if background_point is not None else background_frame
    if background_point is None and provider_background_frame == "caked_2theta_phi":
        background_point = _geometry_fit_point_list(
            (
                initial.get("background_two_theta_deg"),
                initial.get("background_phi_deg"),
            )
        )
        background_frame = "caked_2theta_phi" if background_point is not None else background_frame
    background_frame = _geometry_fit_normalize_point_frame(background_frame)

    provider_simulated_frame = _geometry_fit_normalize_point_frame(
        initial.get(
            "provider_simulated_frame",
            measured.get("provider_simulated_frame", provider.get("simulated_frame")),
        )
    )
    simulated_point = None
    simulated_frame = "unknown"
    for simulated_entry in (initial, measured):
        if not simulated_entry:
            continue
        if provider_simulated_frame == "caked_2theta_phi":
            simulated_point = _geometry_fit_point_list(
                (
                    simulated_entry.get("simulated_two_theta_deg"),
                    simulated_entry.get("simulated_phi_deg"),
                )
            )
            simulated_frame = "caked_2theta_phi" if simulated_point is not None else "unknown"
        if simulated_point is None and provider_simulated_frame == "detector_native":
            simulated_point = _geometry_fit_point_list(simulated_entry.get("sim_native"))
            simulated_frame = "detector_native" if simulated_point is not None else "unknown"
        if simulated_point is None:
            simulated_point = _geometry_fit_point_list(simulated_entry.get("sim_display"))
            simulated_frame = "display" if simulated_point is not None else "unknown"
        if simulated_point is None:
            simulated_point = _geometry_fit_point_list(
                (
                    simulated_entry.get("simulated_two_theta_deg"),
                    simulated_entry.get("simulated_phi_deg"),
                )
            )
            simulated_frame = "caked_2theta_phi" if simulated_point is not None else "unknown"
        if simulated_point is not None:
            break

    identity = _geometry_fit_handoff_identity(measured, initial)
    normalized_hkl = _geometry_fit_jsonable(
        _geometry_fit_normalized_hkl(
            initial.get("hkl", measured.get("hkl", provider.get("normalized_hkl")))
        )
    )
    source_branch_index = _geometry_fit_coerce_nonnegative_index(
        initial.get(
            "source_branch_index",
            measured.get("source_branch_index", provider.get("source_branch_index")),
        )
    )
    selected_to_background_distance_px = (
        float(
            math.hypot(
                float(simulated_point[0]) - float(background_point[0]),
                float(simulated_point[1]) - float(background_point[1]),
            )
        )
        if simulated_point is not None
        and background_point is not None
        and simulated_frame == background_frame
        and simulated_frame != "unknown"
        else None
    )

    handoff_pair = {
        "pair_index": int(pair_index),
        "provider_pair_index": int(provider.get("provider_pair_index", pair_index)),
        "dataset_pair_index": int(pair_index),
        "background_index": int(dataset_index),
        "manual_pair_order_key": provider.get(
            "manual_pair_order_key",
            [int(dataset_index), int(pair_index)],
        ),
        "semantic_pair_key": provider.get("semantic_pair_key"),
        "q_group_key": _geometry_fit_jsonable(
            initial.get("q_group_key", measured.get("q_group_key", provider.get("q_group_key")))
        ),
        "source_q_group_key": _geometry_fit_jsonable(
            initial.get(
                "source_q_group_key",
                measured.get("source_q_group_key", provider.get("source_q_group_key")),
            )
        ),
        "branch_group_key": _geometry_fit_jsonable(
            initial.get(
                "branch_group_key",
                measured.get("branch_group_key", provider.get("branch_group_key")),
            )
        ),
        "normalized_hkl": normalized_hkl,
        "source_branch_index": (
            int(source_branch_index) if source_branch_index is not None else None
        ),
        "selected_source_identity_canonical": identity,
        "background_point": background_point,
        "background_frame": background_frame,
        "background_point_source": measured.get(
            "provider_background_point_source",
            initial.get(
                "provider_background_point_source",
                provider.get("background_point_source"),
            ),
        ),
        "simulated_point": simulated_point,
        "simulated_frame": simulated_frame,
        "simulated_point_source": initial.get(
            "provider_simulated_point_source",
            measured.get(
                "provider_simulated_point_source",
                provider.get("simulated_point_source"),
            ),
        ),
        "selected_to_background_distance_px": selected_to_background_distance_px,
        "parity_mode": provider.get("parity_mode"),
        "rebinding_fallback_used": provider.get("rebinding_fallback_used"),
        "fallback_reason": provider.get("fallback_reason"),
        "solver_measured_point": background_point,
        "solver_measured_frame": background_frame,
    }
    _geometry_fit_apply_source_coverage_identity(handoff_pair)
    return handoff_pair


def _geometry_fit_dataset_pairs_from_handoff(
    dataset: Mapping[str, object],
    provider_pairs: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    measured_rows = [
        dict(entry)
        for entry in dataset.get("measured_for_fit", ()) or ()
        if isinstance(entry, Mapping)
    ]
    initial_rows = [
        dict(entry)
        for entry in dataset.get("initial_pairs_display", ()) or ()
        if isinstance(entry, Mapping)
    ]
    providers = [dict(pair) for pair in provider_pairs or () if isinstance(pair, Mapping)]
    pair_count = max(len(providers), len(measured_rows), len(initial_rows))
    dataset_index = int(dataset.get("dataset_index", 0) or 0)
    return [
        _geometry_fit_handoff_pair(
            dataset_index=dataset_index,
            pair_index=idx,
            provider_pair=providers[idx] if idx < len(providers) else None,
            measured_entry=measured_rows[idx] if idx < len(measured_rows) else None,
            initial_entry=initial_rows[idx] if idx < len(initial_rows) else None,
        )
        for idx in range(pair_count)
    ]


_GEOMETRY_FIT_QR_HANDOFF_AUDIT_GROUP_KEY = ("q_group", "primary", 1, 10)
_GEOMETRY_FIT_QR_HANDOFF_AUDIT_HKL = (-1, 0, 10)
_GEOMETRY_FIT_QR_HANDOFF_AUDIT_BRANCHES = {0, 1}


def _geometry_fit_audit_tuple(value: object) -> tuple[object, ...] | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, list):
        return tuple(value)
    return None


def _geometry_fit_audit_q_group_key(value: object) -> tuple[object, ...] | None:
    raw_tuple = _geometry_fit_audit_tuple(value)
    if raw_tuple is None:
        return None
    normalized: list[object] = []
    for item in raw_tuple:
        try:
            numeric = float(item)
        except Exception:
            normalized.append(item)
            continue
        if np.isfinite(numeric) and abs(numeric - round(numeric)) <= 1.0e-9:
            normalized.append(int(round(numeric)))
        else:
            normalized.append(float(numeric))
    return tuple(normalized)


def _geometry_fit_audit_hkl(value: object) -> tuple[int, int, int] | None:
    raw_tuple = _geometry_fit_audit_tuple(value)
    if raw_tuple is None or len(raw_tuple) < 3:
        return None
    try:
        return tuple(int(np.rint(float(item))) for item in raw_tuple[:3])  # type: ignore[return-value]
    except Exception:
        return None


def _geometry_fit_audit_first(
    entries: Sequence[Mapping[str, object]],
    key: str,
) -> object:
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get(key) is not None:
            return entry.get(key)
    return None


def _geometry_fit_audit_point_from_tuple_key(
    entries: Sequence[Mapping[str, object]],
    key: str,
) -> tuple[float, float] | None:
    for entry in entries:
        raw_point = entry.get(key) if isinstance(entry, Mapping) else None
        point = _geometry_fit_point_list(raw_point)
        if point is not None:
            return float(point[0]), float(point[1])
    return None


def _geometry_fit_audit_point_from_key_pairs(
    entries: Sequence[Mapping[str, object]],
    key_pairs: Sequence[tuple[str, str]],
) -> tuple[float, float] | None:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for x_key, y_key in key_pairs:
            try:
                x_value = float(entry.get(x_key, np.nan))
                y_value = float(entry.get(y_key, np.nan))
            except Exception:
                continue
            if np.isfinite(x_value) and np.isfinite(y_value):
                return float(x_value), float(y_value)
    return None


def _geometry_fit_audit_point_delta(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
) -> float | None:
    if left is None or right is None:
        return None
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _wrapped_phi_delta_deg(pred_phi: object, obs_phi: object) -> float:
    """Return signed phi residual in degrees using predicted - observed."""

    return float(((float(pred_phi) - float(obs_phi) + 180.0) % 360.0) - 180.0)


def _qr_residual_detector_native_px(
    observed_native: object,
    predicted_native: object,
) -> tuple[float, float] | None:
    """Detector-native residual in px: predicted - observed."""

    observed = _geometry_fit_point_list(observed_native)
    predicted = _geometry_fit_point_list(predicted_native)
    if observed is None or predicted is None:
        return None
    return (
        float(predicted[0] - observed[0]),
        float(predicted[1] - observed[1]),
    )


def _qr_residual_caked_deg(
    observed_caked: object,
    predicted_caked: object,
) -> tuple[float, float] | None:
    """Caked residual in deg: predicted - observed, phi wrapped."""

    observed = _geometry_fit_point_list(observed_caked)
    predicted = _geometry_fit_point_list(predicted_caked)
    if observed is None or predicted is None:
        return None
    return (
        float(predicted[0] - observed[0]),
        _wrapped_phi_delta_deg(predicted[1], observed[1]),
    )


def _qr_residual_norm(values: object, weights: object | None = None) -> float | None:
    """Euclidean residual norm with optional explicit component weights."""

    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if weights is not None:
        try:
            weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if weight_arr.size == 1:
            arr = arr * float(weight_arr[0])
        elif weight_arr.size == arr.size:
            arr = arr * weight_arr
        else:
            return None
        if not np.all(np.isfinite(arr)):
            return None
    return float(np.linalg.norm(arr))


def _geometry_fit_audit_phi_delta(left_phi: float, right_phi: float) -> float:
    return _wrapped_phi_delta_deg(left_phi, right_phi)


def _geometry_fit_audit_caked_delta_pair(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if left is None or right is None:
        return None
    return (
        float(float(left[0]) - float(right[0])),
        _geometry_fit_audit_phi_delta(float(left[1]), float(right[1])),
    )


def _geometry_fit_audit_caked_delta_norm(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
) -> float | None:
    delta = _geometry_fit_audit_caked_delta_pair(left, right)
    if delta is None:
        return None
    return float(math.hypot(float(delta[0]), float(delta[1])))


def _geometry_fit_audit_project_native_to_caked(
    native_point: tuple[float, float] | None,
    *,
    fit_space_projector: object,
    base_fit_params: Mapping[str, object] | None,
) -> tuple[tuple[float, float] | None, str]:
    if native_point is None:
        return None, "sim_refined_detector_native_px_unavailable"
    if not callable(fit_space_projector):
        return None, "real projection callback unavailable"
    try:
        projected = fit_space_projector(
            np.asarray([float(native_point[0])], dtype=np.float64),
            np.asarray([float(native_point[1])], dtype=np.float64),
            local_params=dict(base_fit_params or {}),
            anchor_kind="simulated",
            input_frame="native_detector",
        )
    except Exception as exc:
        return None, f"real projection callback failed:{exc}"
    if not isinstance(projected, Mapping):
        return None, "real projection callback returned non-mapping"
    if projected.get("valid") is False:
        return None, str(projected.get("invalid_reason") or "real projection invalid")
    try:
        two_theta_values = np.asarray(projected.get("two_theta_deg", []), dtype=np.float64).reshape(
            -1
        )
        phi_values = np.asarray(projected.get("phi_deg", []), dtype=np.float64).reshape(-1)
        two_theta = float(two_theta_values[0])
        phi = float(phi_values[0])
    except Exception:
        return None, "real projection callback returned no caked point"
    if not (np.isfinite(two_theta) and np.isfinite(phi)):
        return None, "real projection callback returned non-finite caked point"
    return (float(two_theta), float(phi)), "real_projection_callback"


def _geometry_fit_audit_native_to_display(
    native_point: tuple[float, float] | None,
    dataset: Mapping[str, object],
) -> tuple[float, float] | None:
    point, _reason = _geometry_fit_audit_native_to_display_result(
        native_point,
        dataset,
    )
    return point


def _geometry_fit_audit_native_to_display_result(
    native_point: tuple[float, float] | None,
    dataset: Mapping[str, object],
) -> tuple[tuple[float, float] | None, str | None]:
    if native_point is None:
        return None, "native detector point unavailable"
    native_to_display = dataset.get("native_detector_coords_to_detector_display_coords")
    if callable(native_to_display):
        try:
            projected = native_to_display(float(native_point[0]), float(native_point[1]))
        except Exception as exc:
            return None, f"live native->display callback failed:{exc}"
        point = _geometry_fit_point_list(projected)
        if point is not None:
            return (float(point[0]), float(point[1])), None
        return None, "live native->display callback returned unavailable"
    required_reason = dataset.get(
        "native_detector_coords_to_detector_display_coords_unavailable_reason"
    )
    if required_reason is not None and str(required_reason).strip():
        return None, str(required_reason)
    native_background = dataset.get("native_background")
    try:
        native_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
    except Exception:
        native_shape = ()
    if len(native_shape) < 2 or min(native_shape) <= 0:
        try:
            image_size = int(dataset.get("image_size", 0) or 0)
        except Exception:
            image_size = 0
        if image_size > 0:
            native_shape = (int(image_size), int(image_size))
    if len(native_shape) < 2 or min(native_shape) <= 0:
        return None, "native detector image shape unavailable"
    try:
        rotate_k = int(dataset.get("display_rotate_k", 0) or 0)
    except Exception:
        rotate_k = 0
    try:
        display = gui_manual_geometry._default_rotate_point(
            float(native_point[0]),
            float(native_point[1]),
            native_shape,
            rotate_k,
        )
    except Exception as exc:
        return None, f"rotate fallback failed:{exc}"
    if (
        isinstance(display, tuple)
        and len(display) >= 2
        and np.isfinite(float(display[0]))
        and np.isfinite(float(display[1]))
    ):
        return (float(display[0]), float(display[1])), None
    return None, "rotate fallback returned unavailable"


def _geometry_fit_dataset_native_to_display_callback(
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    background_index: int,
) -> tuple[
    Callable[[float, float], tuple[float | None, float | None] | None] | None,
    str | None,
    str,
]:
    factory = getattr(
        manual_dataset_bindings,
        "native_detector_coords_to_detector_display_coords_for_background",
        None,
    )
    if callable(factory):
        try:
            callback = factory(int(background_index))
        except Exception as exc:
            return (
                None,
                f"background-bound native->display callback failed:{exc}",
                "background_bound_callback",
            )
        if callable(callback):
            return callback, None, "background_bound_callback"
        return (
            None,
            "background-bound native->display callback unavailable",
            "background_bound_callback",
        )
    callback = getattr(
        manual_dataset_bindings,
        "native_detector_coords_to_detector_display_coords",
        None,
    )
    if callable(callback):
        return callback, None, "live_callback"
    return None, None, "rotate_fallback"


def _geometry_fit_audit_sim_refined_caked(
    entries: Sequence[Mapping[str, object]],
    native_point: tuple[float, float] | None,
    *,
    fit_space_projector: object,
    base_fit_params: Mapping[str, object] | None,
) -> tuple[tuple[float, float] | None, dict[str, object]]:
    raw_caked = _geometry_fit_audit_point_from_tuple_key(entries, "sim_refined_caked_deg")
    if raw_caked is None:
        raw_caked = _geometry_fit_audit_point_from_key_pairs(
            entries,
            (("refined_sim_caked_x", "refined_sim_caked_y"),),
        )

    projection_status = str(
        _geometry_fit_audit_first(entries, "sim_refined_caked_projection_status") or ""
    )
    projection_real_raw = _geometry_fit_audit_first(
        entries,
        "sim_refined_caked_projection_real_callback",
    )
    projection_real = bool(projection_real_raw) if projection_real_raw is not None else False
    fake_projection = bool(projection_status == "fake_or_test_callback")

    projected_caked, projected_reason = _geometry_fit_audit_project_native_to_caked(
        native_point,
        fit_space_projector=fit_space_projector,
        base_fit_params=base_fit_params,
    )
    if projected_caked is not None:
        delta = _geometry_fit_audit_caked_delta_norm(projected_caked, raw_caked)
        return projected_caked, {
            "sim_refined_caked_projection_status": "real_projection_callback",
            "sim_refined_caked_projection_real_callback": True,
            "sim_refined_caked_projection_validation_delta_deg": delta,
            "sim_refined_caked_raw_saved_deg": raw_caked,
            "sim_refined_caked_unavailable_reason": None,
            "fit_prediction_uses_fake_or_test_transform": bool(fake_projection),
        }

    if fake_projection:
        return None, {
            "sim_refined_caked_projection_status": projection_status or "fake_or_test_callback",
            "sim_refined_caked_projection_real_callback": False,
            "sim_refined_caked_unavailable_reason": "fake transform not valid for live fit handoff",
            "fit_prediction_uses_fake_or_test_transform": True,
        }

    if raw_caked is not None and projection_real:
        return raw_caked, {
            "sim_refined_caked_projection_status": projection_status or "real_callback",
            "sim_refined_caked_projection_real_callback": True,
            "sim_refined_caked_unavailable_reason": None,
            "fit_prediction_uses_fake_or_test_transform": False,
        }

    if raw_caked is not None and projection_status == "caked_simulation_image_axes":
        return raw_caked, {
            "sim_refined_caked_projection_status": "caked_simulation_image_axes",
            "sim_refined_caked_projection_real_callback": False,
            "sim_refined_caked_raw_saved_deg": raw_caked,
            "sim_refined_caked_unavailable_reason": None,
            "fit_prediction_uses_fake_or_test_transform": False,
            "real_projection_unavailable_reason": projected_reason,
        }

    if raw_caked is not None:
        return None, {
            "sim_refined_caked_projection_status": projection_status
            or "missing_live_projection_provenance",
            "sim_refined_caked_projection_real_callback": False,
            "sim_refined_caked_raw_saved_deg": raw_caked,
            "sim_refined_caked_unavailable_reason": "real projection provenance missing",
            "fit_prediction_uses_fake_or_test_transform": False,
            "real_projection_unavailable_reason": projected_reason,
        }

    return None, {
        "sim_refined_caked_projection_status": projection_status or "missing",
        "sim_refined_caked_projection_real_callback": False,
        "sim_refined_caked_unavailable_reason": projected_reason,
        "fit_prediction_uses_fake_or_test_transform": False,
    }


def _geometry_fit_qr_fit_prediction_source(
    provider_pair: Mapping[str, object],
    manual_pair: Mapping[str, object],
    measured_entry: Mapping[str, object],
    initial_entry: Mapping[str, object],
) -> str:
    identity = _geometry_fit_source_identity_from_pair(provider_pair, manual_pair)
    if identity:
        return "dynamic_current_simulation"
    source_text = str(
        provider_pair.get(
            "simulated_point_source",
            initial_entry.get(
                "provider_simulated_point_source",
                measured_entry.get("provider_simulated_point_source", ""),
            ),
        )
        or ""
    )
    if source_text in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES:
        return "saved_visual_sim_refined"
    if source_text:
        return f"unavailable_reason:{source_text}"
    return "unavailable_reason:no_prediction_source"


def build_geometry_fit_qr_handoff_audit_rows(
    dataset: Mapping[str, object] | None,
    *,
    base_fit_params: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    if not isinstance(dataset, Mapping):
        return []
    spec = dataset.get("spec") if isinstance(dataset.get("spec"), Mapping) else {}
    fit_space_projector = spec.get("fit_space_projector") if isinstance(spec, Mapping) else None
    provider_pairs = [
        dict(row) for row in dataset.get("provider_pairs", ()) or () if isinstance(row, Mapping)
    ]
    manual_pairs = [
        dict(row) for row in dataset.get("manual_point_pairs", ()) or () if isinstance(row, Mapping)
    ]
    measured_display = [
        dict(row) for row in dataset.get("measured_display", ()) or () if isinstance(row, Mapping)
    ]
    measured_for_fit = [
        dict(row) for row in dataset.get("measured_for_fit", ()) or () if isinstance(row, Mapping)
    ]
    initial_rows = [
        dict(row)
        for row in dataset.get("initial_pairs_display", ()) or ()
        if isinstance(row, Mapping)
    ]
    trace_rows = [
        dict(row)
        for row in dataset.get("source_rows_for_trace", ()) or ()
        if isinstance(row, Mapping)
    ]
    pair_count = max(
        len(provider_pairs),
        len(manual_pairs),
        len(measured_display),
        len(measured_for_fit),
        len(initial_rows),
    )
    rows: list[dict[str, object]] = []
    for pair_index in range(pair_count):
        provider_pair = provider_pairs[pair_index] if pair_index < len(provider_pairs) else {}
        manual_pair = manual_pairs[pair_index] if pair_index < len(manual_pairs) else {}
        display_entry = measured_display[pair_index] if pair_index < len(measured_display) else {}
        fit_entry = measured_for_fit[pair_index] if pair_index < len(measured_for_fit) else {}
        initial_entry = initial_rows[pair_index] if pair_index < len(initial_rows) else {}
        entries = [display_entry, initial_entry, fit_entry, provider_pair, manual_pair]

        q_group_key = _geometry_fit_audit_q_group_key(
            _geometry_fit_audit_first(entries, "q_group_key")
            or _geometry_fit_audit_first(entries, "source_q_group_key")
        )
        hkl = _geometry_fit_audit_hkl(
            _geometry_fit_audit_first(entries, "hkl")
            or _geometry_fit_audit_first(entries, "normalized_hkl")
        )
        branch_idx = _geometry_fit_coerce_nonnegative_index(
            _geometry_fit_audit_first(entries, "source_branch_index")
        )
        source_table_index = _geometry_fit_audit_first(entries, "source_table_index")
        source_row_index = _geometry_fit_audit_first(entries, "source_row_index")
        source_peak_index = _geometry_fit_audit_first(entries, "source_peak_index")
        if (
            q_group_key != _GEOMETRY_FIT_QR_HANDOFF_AUDIT_GROUP_KEY
            or hkl != _GEOMETRY_FIT_QR_HANDOFF_AUDIT_HKL
            or branch_idx not in _GEOMETRY_FIT_QR_HANDOFF_AUDIT_BRANCHES
        ):
            continue
        source_trace_entry = {}
        for trace_row in trace_rows:
            if (
                _geometry_fit_audit_q_group_key(trace_row.get("q_group_key")) == q_group_key
                and _geometry_fit_audit_hkl(trace_row.get("hkl")) == hkl
                and _geometry_fit_coerce_nonnegative_index(trace_row.get("source_branch_index"))
                == branch_idx
                and trace_row.get("source_table_index") == source_table_index
                and trace_row.get("source_row_index") == source_row_index
            ):
                source_trace_entry = dict(trace_row)
                break

        observed_refined_display = _geometry_fit_audit_point_from_key_pairs(
            (display_entry, initial_entry),
            (("x", "y"), ("display_col", "display_row")),
        )
        observed_refined_native = _geometry_fit_audit_point_from_key_pairs(
            (display_entry, fit_entry, initial_entry),
            (
                ("detector_x", "detector_y"),
                ("background_detector_x", "background_detector_y"),
                ("native_col", "native_row"),
            ),
        )
        observed_refined_caked = _geometry_fit_audit_point_from_key_pairs(
            (display_entry, fit_entry, initial_entry),
            (("background_two_theta_deg", "background_phi_deg"), ("caked_x", "caked_y")),
        )
        observed_raw_display = _geometry_fit_audit_point_from_key_pairs(
            (display_entry,),
            (("raw_x", "raw_y"),),
        )
        observed_raw_caked = _geometry_fit_audit_point_from_key_pairs(
            (display_entry,),
            (("raw_caked_x", "raw_caked_y"),),
        )

        sim_nominal_display = _geometry_fit_audit_point_from_tuple_key(
            entries,
            "sim_nominal_detector_display_px",
        )
        sim_nominal_native = _geometry_fit_audit_point_from_tuple_key(
            entries,
            "sim_nominal_detector_native_px",
        ) or _geometry_fit_audit_point_from_tuple_key((initial_entry,), "sim_native")
        sim_nominal_display_unavailable_reason = None
        if sim_nominal_display is None:
            (
                sim_nominal_display,
                sim_nominal_display_unavailable_reason,
            ) = _geometry_fit_audit_native_to_display_result(
                sim_nominal_native,
                dataset,
            )
        if sim_nominal_display is None and not callable(
            dataset.get("native_detector_coords_to_detector_display_coords")
        ):
            sim_nominal_display = _geometry_fit_audit_point_from_tuple_key(
                (initial_entry,),
                "sim_display",
            )
            if sim_nominal_display is not None:
                sim_nominal_display_unavailable_reason = None
        sim_nominal_caked = _geometry_fit_audit_point_from_tuple_key(
            entries,
            "sim_nominal_caked_deg",
        ) or _geometry_fit_audit_point_from_key_pairs(
            (initial_entry, display_entry),
            (("simulated_two_theta_deg", "simulated_phi_deg"),),
        )
        sim_refined_display = _geometry_fit_audit_point_from_tuple_key(
            entries,
            "sim_refined_detector_display_px",
        )
        sim_refined_native = _geometry_fit_audit_point_from_tuple_key(
            entries,
            "sim_refined_detector_native_px",
        ) or _geometry_fit_audit_point_from_key_pairs(
            entries,
            (("refined_sim_native_x", "refined_sim_native_y"),),
        )
        sim_refined_display_unavailable_reason = None
        if sim_refined_display is None:
            (
                sim_refined_display,
                sim_refined_display_unavailable_reason,
            ) = _geometry_fit_audit_native_to_display_result(
                sim_refined_native,
                dataset,
            )
        sim_refined_caked, sim_caked_meta = _geometry_fit_audit_sim_refined_caked(
            entries,
            sim_refined_native,
            fit_space_projector=fit_space_projector,
            base_fit_params=base_fit_params,
        )

        fit_observed_display = (
            _geometry_fit_audit_point_from_tuple_key(
                (initial_entry,),
                "bg_display",
            )
            or observed_refined_display
        )
        fit_observed_native = _geometry_fit_audit_point_from_key_pairs(
            (fit_entry, initial_entry),
            (
                ("background_detector_x", "background_detector_y"),
                ("native_col", "native_row"),
                ("detector_x", "detector_y"),
            ),
        )
        fit_observed_caked = _geometry_fit_audit_point_from_key_pairs(
            (fit_entry, initial_entry),
            (("background_two_theta_deg", "background_phi_deg"),),
        )
        fit_prediction_source = _geometry_fit_qr_fit_prediction_source(
            provider_pair,
            manual_pair,
            fit_entry,
            initial_entry,
        )
        resolver_identity = _geometry_fit_source_identity_from_pair(provider_pair, manual_pair)
        resolver_entry: dict[str, object] = {}
        for resolver_source in (
            display_entry,
            fit_entry,
            initial_entry,
            provider_pair,
            manual_pair,
        ):
            if isinstance(resolver_source, Mapping):
                resolver_entry.update(dict(resolver_source))
        if resolver_identity:
            resolver_entry["provider_selected_source_identity_canonical"] = copy.deepcopy(
                resolver_identity
            )
        resolver_entry["fit_source_resolution_kind"] = "provider_fixed_source_local"
        resolver_entry["optimizer_request_has_fixed_source"] = True
        resolver_entry["optimizer_request_source"] = "provider_pair"
        resolver_entry["optimizer_request_fallback_row"] = False
        if branch_idx in {0, 1}:
            resolver_entry["source_branch_index"] = int(branch_idx)
            resolver_entry["source_peak_index"] = int(branch_idx)
            resolver_entry["resolved_peak_index"] = int(branch_idx)
        if hkl is not None:
            resolver_entry["hkl"] = tuple(int(value) for value in hkl)
        if q_group_key is not None:
            resolver_entry["q_group_key"] = q_group_key
        if source_table_index is not None:
            resolver_entry["source_table_index"] = source_table_index
        if source_row_index is not None:
            resolver_entry["source_row_index"] = source_row_index
        (
            resolved_prediction_native,
            resolved_prediction_payload,
            resolved_prediction_reason,
        ) = _resolve_fixed_manual_qr_fit_prediction(resolver_entry)
        resolved_prediction_source = str(
            resolved_prediction_payload.get("prediction_source", "") or ""
        ).strip()
        if resolved_prediction_source:
            fit_prediction_source = resolved_prediction_source
        fit_prediction_display = _geometry_fit_audit_point_from_tuple_key(
            (initial_entry,),
            "sim_display",
        )
        fit_prediction_native = _geometry_fit_audit_point_from_tuple_key(
            (initial_entry,),
            "sim_native",
        )
        if resolved_prediction_native is not None:
            fit_prediction_native = (
                float(resolved_prediction_native[0]),
                float(resolved_prediction_native[1]),
            )
            resolved_display = resolved_prediction_payload.get("saved_sim_detector_display_px")
            if isinstance(resolved_display, (list, tuple)) and len(resolved_display) >= 2:
                try:
                    fit_prediction_display = (
                        float(resolved_display[0]),
                        float(resolved_display[1]),
                    )
                except Exception:
                    pass
        fit_prediction_display_from_native = None
        fit_prediction_display_unavailable_reason = None
        if fit_prediction_display is None and fit_prediction_native is not None:
            (
                fit_prediction_display_from_native,
                fit_prediction_display_unavailable_reason,
            ) = _geometry_fit_audit_native_to_display_result(
                fit_prediction_native,
                dataset,
            )
        if fit_prediction_display_from_native is not None:
            fit_prediction_display = fit_prediction_display_from_native
            fit_prediction_display_unavailable_reason = None
        elif callable(dataset.get("native_detector_coords_to_detector_display_coords")):
            fit_prediction_display = None
        fit_prediction_caked = _geometry_fit_audit_point_from_key_pairs(
            (initial_entry,),
            (("simulated_two_theta_deg", "simulated_phi_deg"),),
        )
        fit_prediction_caked_from_handoff = fit_prediction_caked is not None
        fit_prediction_caked_projection_reason = ""
        fit_prediction_projection_theta_initial = None
        if fit_prediction_native is not None:
            projection_params = dict(base_fit_params or {})
            fit_prediction_projection_theta_initial = _geometry_fit_finite_float(
                projection_params.get("theta_initial")
            )
            projected_prediction_caked, projected_prediction_reason = (
                _geometry_fit_audit_project_native_to_caked(
                    fit_prediction_native,
                    fit_space_projector=fit_space_projector,
                    base_fit_params=projection_params,
                )
            )
            fit_prediction_caked_projection_reason = str(projected_prediction_reason)
            if projected_prediction_caked is not None and not fit_prediction_caked_from_handoff:
                fit_prediction_caked = projected_prediction_caked

        observed_native_delta = _geometry_fit_audit_point_delta(
            fit_observed_native,
            observed_refined_native,
        )
        observed_caked_delta = _geometry_fit_audit_caked_delta_pair(
            fit_observed_caked,
            observed_refined_caked,
        )
        observed_match = bool(
            observed_native_delta is not None
            and observed_native_delta <= 1.0
            and observed_caked_delta is not None
            and abs(float(observed_caked_delta[0])) <= 0.25
            and abs(float(observed_caked_delta[1])) <= 0.5
        )
        sim_dynamic = bool(fit_prediction_source.startswith("dynamic_current_simulation"))
        sim_detector_delta = _geometry_fit_audit_point_delta(
            fit_prediction_native,
            sim_refined_native,
        )
        sim_caked_delta = _geometry_fit_audit_caked_delta_pair(
            fit_prediction_caked,
            sim_refined_caked,
        )
        if sim_dynamic:
            sim_match_text = "not-applicable"
        else:
            sim_match = bool(
                sim_detector_delta is not None
                and sim_detector_delta <= 1.0
                and sim_caked_delta is not None
                and abs(float(sim_caked_delta[0])) <= 0.25
                and abs(float(sim_caked_delta[1])) <= 0.5
            )
            sim_match_text = "yes" if sim_match else "no"

        observed_minus_sim_caked = _geometry_fit_audit_caked_delta_pair(
            observed_refined_caked,
            sim_refined_caked,
        )
        fit_observed_minus_prediction_caked = _geometry_fit_audit_caked_delta_pair(
            fit_observed_caked,
            fit_prediction_caked,
        )
        fit_residual_detector_native = _qr_residual_detector_native_px(
            fit_observed_native,
            fit_prediction_native,
        )
        fit_residual_caked = _qr_residual_caked_deg(
            fit_observed_caked,
            fit_prediction_caked,
        )
        geometry_minus_sim_detector_native = _qr_residual_detector_native_px(
            fit_prediction_native,
            fit_observed_native,
        )
        geometry_minus_sim_caked = _qr_residual_caked_deg(
            fit_prediction_caked,
            fit_observed_caked,
        )
        fit_residual_detector_norm = _qr_residual_norm(fit_residual_detector_native)
        fit_residual_caked_norm = _qr_residual_norm(fit_residual_caked)
        objective_space = (
            "caked_deg"
            if fit_observed_caked is not None
            or bool(spec.get("fit_space_projector_kind") if isinstance(spec, Mapping) else None)
            else "detector_native_px"
        )
        objective_units = "deg" if objective_space == "caked_deg" else "px"
        first_divergence = ""
        if not observed_match:
            first_divergence = "fit_observed_minus_observed_refined"
        elif sim_match_text == "no":
            first_divergence = "fit_prediction_minus_sim_refined"
        elif sim_caked_meta.get("sim_refined_caked_unavailable_reason"):
            first_divergence = "sim_refined_caked_deg"

        row = {
            "pair_index": int(pair_index),
            "q_group_key": q_group_key,
            "hkl": hkl,
            "source_table_index": source_table_index,
            "source_row_index": source_row_index,
            "source_branch_index": branch_idx,
            "source_peak_index": source_peak_index,
            "branch_id": _geometry_fit_audit_first(entries, "branch_id")
            or source_trace_entry.get("branch_id"),
            "observed_raw_detector_display_px": observed_raw_display,
            "observed_raw_detector_native_px": None,
            "observed_raw_detector_native_px_unavailable_reason": "raw native detector point not saved",
            "observed_raw_caked_deg": observed_raw_caked,
            "observed_refined_detector_display_px": observed_refined_display,
            "observed_refined_detector_native_px": observed_refined_native,
            "observed_refined_caked_deg": observed_refined_caked,
            "sim_nominal_detector_display_px": sim_nominal_display,
            "sim_nominal_detector_display_px_unavailable_reason": (
                sim_nominal_display_unavailable_reason
            ),
            "sim_nominal_detector_native_px": sim_nominal_native,
            "sim_nominal_caked_deg": sim_nominal_caked,
            "sim_refined_detector_display_px": sim_refined_display,
            "sim_refined_detector_display_px_unavailable_reason": (
                sim_refined_display_unavailable_reason
                or "no saved detector-display refined sim point"
            ),
            "sim_refined_detector_native_px": sim_refined_native,
            "sim_refined_caked_deg": sim_refined_caked,
            "sim_refinement_status": _geometry_fit_audit_first(entries, "sim_refinement_status"),
            "sim_refinement_source": _geometry_fit_audit_first(entries, "sim_refinement_source"),
            "sim_refinement_delta_detector_px": _geometry_fit_audit_first(
                entries,
                "sim_refinement_delta_detector_px",
            ),
            "sim_refinement_delta_caked_deg": _geometry_fit_audit_first(
                entries,
                "sim_refinement_delta_caked_deg",
            ),
            "fit_observed_detector_display_px": fit_observed_display,
            "fit_observed_detector_native_px": fit_observed_native,
            "fit_observed_caked_deg": fit_observed_caked,
            "fit_prediction_source": fit_prediction_source,
            "fit_prediction_resolver_function": str(
                resolved_prediction_payload.get(
                    "fit_prediction_resolver_function",
                    "_resolve_fixed_manual_qr_fit_prediction",
                )
            ),
            "fit_prediction_source_resolution_reason": str(resolved_prediction_reason),
            "fit_prediction_source_resolution_payload": dict(resolved_prediction_payload),
            "fit_prediction_detector_display_px": fit_prediction_display,
            "fit_prediction_detector_display_px_unavailable_reason": (
                fit_prediction_display_unavailable_reason
            ),
            "fit_prediction_detector_native_px": fit_prediction_native,
            "fit_prediction_caked_deg": fit_prediction_caked,
            "fit_prediction_caked_projection_reason": fit_prediction_caked_projection_reason,
            "fit_prediction_projection_theta_initial_deg": (
                float(fit_prediction_projection_theta_initial)
                if fit_prediction_projection_theta_initial is not None
                else None
            ),
            "observed_source": "background/manual",
            "predicted_source": "simulation",
            "observed_detector_native_px": fit_observed_native,
            "predicted_detector_native_px": fit_prediction_native,
            "residual_detector_native_px": fit_residual_detector_native,
            "observed_caked_deg": fit_observed_caked,
            "predicted_caked_deg": fit_prediction_caked,
            "residual_caked_deg": fit_residual_caked,
            "fit_residual_detector_native_px": fit_residual_detector_native,
            "fit_residual_detector_native_norm_px": fit_residual_detector_norm,
            "fit_residual_caked_deg": fit_residual_caked,
            "fit_residual_caked_norm_deg": fit_residual_caked_norm,
            "geometry_minus_sim_detector_native_px": geometry_minus_sim_detector_native,
            "geometry_minus_sim_caked_deg": geometry_minus_sim_caked,
            "residual_sign_convention": "predicted - observed",
            "residual_detector_native_units": "px",
            "residual_caked_units": "deg",
            "objective_space": objective_space,
            "objective_residual_units": objective_units,
            "objective_mixes_detector_px_and_caked_deg": "no",
            "fit_observed_minus_observed_refined_detector_delta_px": observed_native_delta,
            "fit_observed_minus_observed_refined_caked_delta_deg": observed_caked_delta,
            "fit_prediction_minus_sim_refined_detector_delta_px": sim_detector_delta,
            "fit_prediction_minus_sim_refined_caked_delta_deg": sim_caked_delta,
            "observed_refined_minus_sim_refined_caked_delta_deg": observed_minus_sim_caked,
            "fit_observed_minus_fit_prediction_caked_delta_deg": (
                fit_observed_minus_prediction_caked
            ),
            "observed_visual_to_fit_observed_match": "yes" if observed_match else "no",
            "sim_visual_to_fit_prediction_match": sim_match_text,
            "fit_prediction_is_dynamic": "yes" if sim_dynamic else "no",
            "fit_prediction_uses_fake_or_test_transform": (
                "yes" if sim_caked_meta.get("fit_prediction_uses_fake_or_test_transform") else "no"
            ),
            "caked_values_from_real_projection_callback": (
                "yes" if sim_caked_meta.get("sim_refined_caked_projection_real_callback") else "no"
            ),
            "first_divergence_field": first_divergence or "none",
        }
        row.update(sim_caked_meta)
        row["fit_prediction_uses_fake_or_test_transform"] = (
            "yes" if sim_caked_meta.get("fit_prediction_uses_fake_or_test_transform") else "no"
        )
        row["caked_values_from_real_projection_callback"] = (
            "yes" if sim_caked_meta.get("sim_refined_caked_projection_real_callback") else "no"
        )
        rows.append(row)
    rows.sort(
        key=lambda item: (
            int(item.get("source_branch_index", -1) or -1),
            int(item.get("source_table_index", -1) or -1),
            int(item.get("source_row_index", -1) or -1),
        )
    )
    return rows


def _geometry_fit_audit_value_text(
    value: object,
    *,
    unavailable_reason: object = None,
) -> str:
    point = _geometry_fit_point_list(value)
    if point is not None:
        return f"({float(point[0]):.3f}, {float(point[1]):.3f})"
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(value, bool):
        try:
            numeric = float(value)
        except Exception:
            numeric = float("nan")
        if np.isfinite(numeric):
            return f"{numeric:.6f}"
    if value is None:
        reason = str(unavailable_reason or "missing")
        return f"<unavailable reason={reason}>"
    return str(value)


def build_geometry_fit_qr_handoff_audit_lines(
    audit_rows: Sequence[Mapping[str, object]] | None,
) -> list[str]:
    rows = [dict(row) for row in audit_rows or () if isinstance(row, Mapping)]
    lines = ["[ra-sim] Qr/Qz fit handoff audit"]
    if not rows:
        lines.append("  <unavailable reason=target_q_group_not_in_fit_dataset>")
        return lines
    fields = (
        "observed_raw_detector_display_px",
        "observed_raw_detector_native_px",
        "observed_raw_caked_deg",
        "observed_refined_detector_display_px",
        "observed_refined_detector_native_px",
        "observed_refined_caked_deg",
        "sim_nominal_detector_display_px",
        "sim_nominal_detector_native_px",
        "sim_nominal_caked_deg",
        "sim_refined_detector_display_px",
        "sim_refined_detector_native_px",
        "sim_refined_caked_deg",
        "sim_refinement_status",
        "sim_refinement_source",
        "sim_refinement_delta_detector_px",
        "sim_refinement_delta_caked_deg",
        "fit_observed_detector_display_px",
        "fit_observed_detector_native_px",
        "fit_observed_caked_deg",
        "fit_prediction_source",
        "fit_prediction_detector_display_px",
        "fit_prediction_detector_native_px",
        "fit_prediction_caked_deg",
        "observed_source",
        "predicted_source",
        "observed_detector_native_px",
        "predicted_detector_native_px",
        "residual_detector_native_px",
        "observed_caked_deg",
        "predicted_caked_deg",
        "residual_caked_deg",
        "fit_residual_detector_native_px",
        "fit_residual_detector_native_norm_px",
        "fit_residual_caked_deg",
        "fit_residual_caked_norm_deg",
        "geometry_minus_sim_detector_native_px",
        "geometry_minus_sim_caked_deg",
        "residual_sign_convention",
        "residual_detector_native_units",
        "residual_caked_units",
        "objective_space",
        "objective_residual_units",
        "objective_mixes_detector_px_and_caked_deg",
        "fit_observed_minus_observed_refined_detector_delta_px",
        "fit_observed_minus_observed_refined_caked_delta_deg",
        "fit_prediction_minus_sim_refined_detector_delta_px",
        "fit_prediction_minus_sim_refined_caked_delta_deg",
        "observed_refined_minus_sim_refined_caked_delta_deg",
        "fit_observed_minus_fit_prediction_caked_delta_deg",
        "observed_visual_to_fit_observed_match",
        "sim_visual_to_fit_prediction_match",
        "fit_prediction_is_dynamic",
        "fit_prediction_uses_fake_or_test_transform",
        "caked_values_from_real_projection_callback",
        "first_divergence_field",
    )
    for row in rows:
        lines.append(
            "  branch={branch} q_group_key={q_group} hkl={hkl} "
            "table={table} row={row_idx} peak={peak} branch_id={branch_id}".format(
                branch=row.get("source_branch_index"),
                q_group=repr(row.get("q_group_key")),
                hkl=repr(row.get("hkl")),
                table=row.get("source_table_index"),
                row_idx=row.get("source_row_index"),
                peak=row.get("source_peak_index"),
                branch_id=row.get("branch_id"),
            )
        )
        for field in fields:
            lines.append(
                "    {field}={value}".format(
                    field=field,
                    value=_geometry_fit_audit_value_text(
                        row.get(field),
                        unavailable_reason=row.get(f"{field}_unavailable_reason"),
                    ),
                )
            )
    return lines


_GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES = {
    "manual_picker_saved",
    "manual_picker_cache",
}

_GEOMETRY_FIT_SOURCE_IDENTITY_COPY_FIELDS = (
    "source_table_index",
    "source_reflection_index",
    "source_reflection_namespace",
    "source_reflection_is_full",
    "source_row_index",
    "source_branch_index",
    "source_peak_index",
    "source_label",
    "label",
)


def _geometry_fit_source_identity_fields(
    entry: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(entry, Mapping):
        return {}
    return {
        key: entry.get(key)
        for key in _GEOMETRY_FIT_SOURCE_IDENTITY_COPY_FIELDS
        if entry.get(key) is not None
    }


def _geometry_fit_put_background_point_fields(
    row: dict[str, object],
    point: Sequence[object] | None,
    frame: object,
) -> None:
    point_list = _geometry_fit_point_list(point)
    point_frame = _geometry_fit_normalize_point_frame(frame)
    if point_list is None:
        return
    if point_frame == "detector_native":
        row["x"] = float(point_list[0])
        row["y"] = float(point_list[1])
        row["background_detector_x"] = float(point_list[0])
        row["background_detector_y"] = float(point_list[1])
        row["background_detector_input_frame"] = "native_detector"
    elif point_frame == "caked_2theta_phi":
        row["x"] = float(point_list[0])
        row["y"] = float(point_list[1])
        row["background_two_theta_deg"] = float(point_list[0])
        row["background_phi_deg"] = float(point_list[1])
        row["bg_caked_display"] = (float(point_list[0]), float(point_list[1]))
    else:
        row["x"] = float(point_list[0])
        row["y"] = float(point_list[1])
        row["display_col"] = float(point_list[0])
        row["display_row"] = float(point_list[1])
        row["bg_display"] = (float(point_list[0]), float(point_list[1]))


def _geometry_fit_put_simulated_point_fields(
    row: dict[str, object],
    point: Sequence[object] | None,
    frame: object,
) -> None:
    point_list = _geometry_fit_point_list(point)
    point_frame = _geometry_fit_normalize_point_frame(frame)
    if point_list is None:
        return
    if point_frame == "detector_native":
        row["sim_native"] = (float(point_list[0]), float(point_list[1]))
    elif point_frame == "caked_2theta_phi":
        row["simulated_two_theta_deg"] = float(point_list[0])
        row["simulated_phi_deg"] = float(point_list[1])
        row["sim_caked_display"] = (float(point_list[0]), float(point_list[1]))
    else:
        row["sim_display"] = (float(point_list[0]), float(point_list[1]))


def _geometry_fit_saved_state_provider_pair(
    *,
    background_index: int,
    pair_index: int,
    saved_entry: Mapping[str, object],
    truth_pair: Mapping[str, object],
) -> dict[str, object]:
    background_point = _geometry_fit_point_list(truth_pair.get("manual_background_point"))
    simulated_point = _geometry_fit_point_list(truth_pair.get("manual_selected_simulated_point"))
    background_frame = _geometry_fit_normalize_point_frame(
        truth_pair.get("manual_background_frame")
    )
    simulated_frame = _geometry_fit_normalize_point_frame(
        truth_pair.get("manual_selected_simulated_frame")
    )
    background_source = str(truth_pair.get("manual_background_point_source") or "unknown")
    simulated_source = str(truth_pair.get("manual_simulated_point_source") or "unknown")
    identity = dict(
        truth_pair.get("manual_picker_selected_source_identity_canonical", {})
        if isinstance(
            truth_pair.get("manual_picker_selected_source_identity_canonical"),
            Mapping,
        )
        else gui_manual_geometry.canonical_geometry_source_identity(saved_entry)
    )
    parity_mode = (
        "picker_saved_value_preserved"
        if background_source in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
        and simulated_source in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
        else "picker_saved_value_unavailable"
    )
    provider_pair = {
        "pair_index": int(pair_index),
        "provider_pair_index": int(pair_index),
        "dataset_pair_index": int(pair_index),
        "background_index": int(background_index),
        "manual_pair_order_key": [int(background_index), int(pair_index)],
        "semantic_pair_key": truth_pair.get("semantic_pair_key"),
        "q_group_key": truth_pair.get("q_group_key"),
        "source_q_group_key": _geometry_fit_jsonable(
            saved_entry.get("source_q_group_key", saved_entry.get("q_group_key"))
        ),
        "branch_group_key": truth_pair.get("branch_group_key"),
        "normalized_hkl": truth_pair.get("normalized_hkl"),
        "source_branch_index": truth_pair.get("source_branch_index"),
        "selected_source_identity_canonical": identity,
        "background_point": background_point,
        "background_frame": background_frame,
        "background_point_source": background_source,
        "simulated_point": simulated_point,
        "simulated_frame": simulated_frame,
        "simulated_point_source": simulated_source,
        "selected_to_background_distance_px": truth_pair.get(
            "manual_selected_to_background_distance_px"
        ),
        "parity_mode": parity_mode,
        "rebinding_fallback_used": False,
        "fallback_reason": None,
        "stale_saved_source_identity": None,
    }
    _geometry_fit_apply_source_coverage_identity(provider_pair)
    return provider_pair


def _geometry_fit_saved_state_handoff_rows(
    *,
    background_index: int,
    saved_entry: Mapping[str, object],
    provider_pair: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    pair_index = int(provider_pair.get("pair_index", 0) or 0)
    pair_id = str(saved_entry.get("pair_id") or f"bg{int(background_index)}:pair{pair_index}")
    identity = dict(provider_pair.get("selected_source_identity_canonical", {}) or {})
    shared = {
        "pair_id": pair_id,
        "hkl": saved_entry.get("hkl", saved_entry.get("label")),
        "q_group_key": provider_pair.get("q_group_key"),
        "source_q_group_key": provider_pair.get("source_q_group_key"),
        "branch_group_key": provider_pair.get("branch_group_key"),
        "source_branch_index": provider_pair.get("source_branch_index"),
        "selected_source_identity_canonical": identity,
        **_geometry_fit_source_identity_fields(saved_entry),
    }

    measured_entry: dict[str, object] = {
        **shared,
        "overlay_match_index": pair_index,
        "fit_source_identity_only": True,
        "provider_background_frame": provider_pair.get("background_frame"),
        "provider_background_point_source": provider_pair.get("background_point_source"),
        "provider_simulated_frame": provider_pair.get("simulated_frame"),
        "provider_simulated_point_source": provider_pair.get("simulated_point_source"),
    }
    _geometry_fit_put_background_point_fields(
        measured_entry,
        _geometry_fit_point_list(provider_pair.get("background_point")),
        provider_pair.get("background_frame"),
    )
    _geometry_fit_put_simulated_point_fields(
        measured_entry,
        _geometry_fit_point_list(provider_pair.get("simulated_point")),
        provider_pair.get("simulated_frame"),
    )
    _geometry_fit_apply_source_coverage_identity(measured_entry)

    initial_entry: dict[str, object] = {
        **shared,
        "overlay_match_index": pair_index,
        "provider_background_frame": provider_pair.get("background_frame"),
        "provider_background_point_source": provider_pair.get("background_point_source"),
        "provider_simulated_frame": provider_pair.get("simulated_frame"),
        "provider_simulated_point_source": provider_pair.get("simulated_point_source"),
    }
    _geometry_fit_put_background_point_fields(
        initial_entry,
        _geometry_fit_point_list(provider_pair.get("background_point")),
        provider_pair.get("background_frame"),
    )
    _geometry_fit_put_simulated_point_fields(
        initial_entry,
        _geometry_fit_point_list(provider_pair.get("simulated_point")),
        provider_pair.get("simulated_frame"),
    )
    _geometry_fit_apply_source_coverage_identity(initial_entry)
    return measured_entry, initial_entry


def build_geometry_fit_saved_state_point_provider_dataset(
    background_index: int,
    saved_entries: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    """Build a provider-only dataset from saved picker entries, without live preflight."""

    saved_rows = [dict(entry) for entry in saved_entries or () if isinstance(entry, Mapping)]
    truth_pairs = gui_manual_geometry.build_geometry_manual_picker_truth_pairs(
        int(background_index),
        saved_rows,
    )
    truth_by_key = _geometry_fit_truth_by_order_key(truth_pairs)
    provider_pairs: list[dict[str, object]] = []
    measured_for_fit: list[dict[str, object]] = []
    initial_pairs_display: list[dict[str, object]] = []

    for pair_index, saved_entry in enumerate(saved_rows):
        truth_pair = truth_by_key.get((int(background_index), int(pair_index)), {})
        provider_pair = _geometry_fit_saved_state_provider_pair(
            background_index=int(background_index),
            pair_index=int(pair_index),
            saved_entry=saved_entry,
            truth_pair=truth_pair,
        )
        provider_pairs.append(provider_pair)
        measured_entry, initial_entry = _geometry_fit_saved_state_handoff_rows(
            background_index=int(background_index),
            saved_entry=saved_entry,
            provider_pair=provider_pair,
        )
        measured_for_fit.append(measured_entry)
        initial_pairs_display.append(initial_entry)

    dataset_payload: dict[str, object] = {
        "dataset_index": int(background_index),
        "label": f"saved_state_background_{int(background_index)}",
        "manual_picker_truth_pairs": truth_pairs,
        "provider_pairs": provider_pairs,
        "measured_for_fit": measured_for_fit,
        "initial_pairs_display": initial_pairs_display,
        "pair_count": int(len(provider_pairs)),
        "resolved_source_pair_count": int(len(provider_pairs)),
        "spec": {
            "dataset_index": int(background_index),
            "label": f"saved_state_background_{int(background_index)}",
            "measured_peaks": measured_for_fit,
        },
        "provider_only_saved_state_dataset": True,
    }
    dataset_payload["manual_point_pairs"] = _geometry_fit_dataset_pairs_from_handoff(
        dataset_payload,
        provider_pairs,
    )
    dataset_payload["point_provider_report"] = build_geometry_fit_point_provider_report(
        dataset_payload
    )
    return _geometry_fit_jsonable(dataset_payload)  # type: ignore[return-value]


def _geometry_fit_surface_rows(
    dataset: Mapping[str, object],
    key: str,
) -> list[dict[str, object]]:
    raw_rows = dataset.get(key, ()) or ()
    return [dict(row) for row in raw_rows if isinstance(row, Mapping)]


def _geometry_fit_spec_measured_peak_rows(
    dataset: Mapping[str, object],
) -> list[dict[str, object]]:
    spec = dataset.get("spec")
    if not isinstance(spec, Mapping):
        return []
    raw_rows = spec.get("measured_peaks", ()) or ()
    return [dict(row) for row in raw_rows if isinstance(row, Mapping)]


def _geometry_fit_pair_point_sources_picker_owned(
    pair: Mapping[str, object] | None,
) -> bool:
    if not isinstance(pair, Mapping) or not pair:
        return False
    return bool(
        pair.get("background_point_source") in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
        and pair.get("simulated_point_source") in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
    )


def _geometry_fit_pair_point_sources_match_provider(
    provider_pair: Mapping[str, object],
    surface_pair: Mapping[str, object] | None,
) -> bool:
    if not isinstance(surface_pair, Mapping) or not surface_pair:
        return False
    return bool(
        provider_pair.get("background_point_source") == surface_pair.get("background_point_source")
        and provider_pair.get("simulated_point_source")
        == surface_pair.get("simulated_point_source")
    )


def _geometry_fit_surface_match(
    provider_pair: Mapping[str, object],
    surface_pair: Mapping[str, object] | None,
) -> bool:
    if not isinstance(surface_pair, Mapping) or not surface_pair:
        return False
    provider_bg_frame = _geometry_fit_normalize_point_frame(provider_pair.get("background_frame"))
    provider_sim_frame = _geometry_fit_normalize_point_frame(provider_pair.get("simulated_frame"))
    surface_bg_frame = _geometry_fit_normalize_point_frame(surface_pair.get("background_frame"))
    surface_sim_frame = _geometry_fit_normalize_point_frame(surface_pair.get("simulated_frame"))
    provider_identity = provider_pair.get("selected_source_identity_canonical", {})
    surface_identity = surface_pair.get("selected_source_identity_canonical", {})
    background_ok = bool(
        provider_bg_frame == surface_bg_frame
        and _geometry_fit_points_match(
            provider_pair.get("background_point"),
            surface_pair.get("background_point"),
        )
    )
    surface_simulated_point = _geometry_fit_point_list(surface_pair.get("simulated_point"))
    simulated_ok = bool(
        provider_sim_frame == surface_sim_frame
        and _geometry_fit_points_match(
            provider_pair.get("simulated_point"),
            surface_simulated_point,
        )
    )
    identity_ok = bool(provider_identity == surface_identity)
    source_ok = bool(
        _geometry_fit_pair_point_sources_picker_owned(provider_pair)
        and _geometry_fit_pair_point_sources_picker_owned(surface_pair)
        and _geometry_fit_pair_point_sources_match_provider(
            provider_pair,
            surface_pair,
        )
    )
    return bool(background_ok and simulated_ok and identity_ok and source_ok)


def _geometry_fit_targeted_gate_from_dataset(
    dataset: Mapping[str, object],
) -> dict[str, object]:
    diagnostics = dataset.get("simulation_diagnostics")
    gate = (
        diagnostics.get("targeted_performance_gate") if isinstance(diagnostics, Mapping) else None
    )
    if isinstance(gate, Mapping):
        return {
            "ok": bool(gate.get("ok", False)),
            "unrelated_projected_row_count_for_rebinding": int(
                gate.get("unrelated_projected_row_count_for_rebinding", 0) or 0
            ),
            "unrelated_scored_row_count_for_rebinding": int(
                gate.get("unrelated_scored_row_count_for_rebinding", 0) or 0
            ),
        }
    return {
        "ok": False,
        "unrelated_projected_row_count_for_rebinding": 0,
        "unrelated_scored_row_count_for_rebinding": 0,
    }


def build_geometry_fit_point_provider_report(
    dataset: Mapping[str, object],
    *,
    optimizer_guard_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare picker truth, provider pairs, and fitter handoff rows."""

    truth_pairs = [
        dict(pair)
        for pair in dataset.get("manual_picker_truth_pairs", ()) or ()
        if isinstance(pair, Mapping)
    ]
    provider_pairs = [
        dict(pair) for pair in dataset.get("provider_pairs", ()) or () if isinstance(pair, Mapping)
    ]
    dataset_pairs = _geometry_fit_dataset_pairs_from_handoff(dataset, provider_pairs)
    dataset_index = int(dataset.get("dataset_index", 0) or 0)
    manual_point_rows = _geometry_fit_surface_rows(dataset, "manual_point_pairs")
    initial_rows = _geometry_fit_surface_rows(dataset, "initial_pairs_display")
    measured_rows = _geometry_fit_surface_rows(dataset, "measured_for_fit")
    spec_measured_peak_rows = _geometry_fit_spec_measured_peak_rows(dataset)

    def _surface_pair_by_index(
        rows: Sequence[Mapping[str, object]],
        *,
        surface: str,
    ) -> dict[int, dict[str, object]]:
        out: dict[int, dict[str, object]] = {}
        for row_index, raw_row in enumerate(rows):
            row = dict(raw_row)
            provider_pair = provider_pairs[row_index] if row_index < len(provider_pairs) else {}
            if surface == "manual_point_pairs":
                pair = dict(row)
                pair.setdefault("dataset_pair_index", int(row_index))
                pair.setdefault("pair_index", int(row_index))
                pair.setdefault("background_index", int(dataset_index))
            elif surface == "initial_pairs_display":
                pair = _geometry_fit_handoff_pair(
                    dataset_index=dataset_index,
                    pair_index=row_index,
                    provider_pair=provider_pair,
                    measured_entry=None,
                    initial_entry=row,
                )
            else:
                pair = _geometry_fit_handoff_pair(
                    dataset_index=dataset_index,
                    pair_index=row_index,
                    provider_pair=provider_pair,
                    measured_entry=row,
                    initial_entry=None,
                )
            try:
                key = int(pair.get("dataset_pair_index", row_index))
            except Exception:
                key = int(row_index)
            out[key] = pair
        return out

    manual_point_by_index = _surface_pair_by_index(
        manual_point_rows,
        surface="manual_point_pairs",
    )
    initial_by_index = _surface_pair_by_index(
        initial_rows,
        surface="initial_pairs_display",
    )
    measured_by_index = _surface_pair_by_index(
        measured_rows,
        surface="measured_for_fit",
    )
    spec_measured_by_index = _surface_pair_by_index(
        spec_measured_peak_rows,
        surface="spec_measured_peaks",
    )
    truth_by_key = _geometry_fit_truth_by_order_key(truth_pairs)
    dataset_by_index = {
        int(pair.get("dataset_pair_index", idx)): pair for idx, pair in enumerate(dataset_pairs)
    }
    pairs: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    counts = Counter()

    for provider_index, provider_pair in enumerate(provider_pairs):
        pair_index = int(provider_pair.get("pair_index", provider_index))
        background_index = int(
            provider_pair.get("background_index", dataset.get("dataset_index", 0)) or 0
        )
        truth_pair = truth_by_key.get((background_index, pair_index), {})
        dataset_pair = dataset_by_index.get(
            int(provider_pair.get("dataset_pair_index", pair_index)),
            {},
        )
        manual_point_pair = manual_point_by_index.get(
            int(provider_pair.get("dataset_pair_index", pair_index)),
            {},
        )
        initial_pair = initial_by_index.get(
            int(provider_pair.get("dataset_pair_index", pair_index)),
            {},
        )
        measured_pair = measured_by_index.get(
            int(provider_pair.get("dataset_pair_index", pair_index)),
            {},
        )
        spec_measured_pair = spec_measured_by_index.get(
            int(provider_pair.get("dataset_pair_index", pair_index)),
            {},
        )
        manual_identity = truth_pair.get(
            "manual_picker_selected_source_identity_canonical",
            {},
        )
        provider_identity = provider_pair.get("selected_source_identity_canonical", {})
        dataset_identity = dataset_pair.get("selected_source_identity_canonical", {})
        source_identity_match = bool(manual_identity == provider_identity)

        manual_bg_frame = _geometry_fit_normalize_point_frame(
            truth_pair.get("manual_background_frame")
        )
        provider_bg_frame = _geometry_fit_normalize_point_frame(
            provider_pair.get("background_frame")
        )
        dataset_bg_frame = _geometry_fit_normalize_point_frame(dataset_pair.get("background_frame"))
        manual_sim_frame = _geometry_fit_normalize_point_frame(
            truth_pair.get("manual_selected_simulated_frame")
        )
        provider_sim_frame = _geometry_fit_normalize_point_frame(
            provider_pair.get("simulated_frame")
        )
        dataset_sim_frame = _geometry_fit_normalize_point_frame(dataset_pair.get("simulated_frame"))

        background_frame_match = bool(
            manual_bg_frame != "unknown"
            and manual_bg_frame == provider_bg_frame == dataset_bg_frame
        )
        simulated_frame_match = bool(
            manual_sim_frame != "unknown"
            and manual_sim_frame == provider_sim_frame == dataset_sim_frame
        )
        background_point_match = bool(
            background_frame_match
            and _geometry_fit_points_match(
                truth_pair.get("manual_background_point"),
                provider_pair.get("background_point"),
            )
            and _geometry_fit_points_match(
                truth_pair.get("manual_background_point"),
                dataset_pair.get("background_point"),
            )
        )
        simulated_point_match = bool(
            simulated_frame_match
            and _geometry_fit_points_match(
                truth_pair.get("manual_selected_simulated_point"),
                provider_pair.get("simulated_point"),
            )
            and _geometry_fit_points_match(
                truth_pair.get("manual_selected_simulated_point"),
                dataset_pair.get("simulated_point"),
            )
        )
        dataset_points_match_provider_points = bool(
            provider_bg_frame == dataset_bg_frame
            and provider_sim_frame == dataset_sim_frame
            and _geometry_fit_points_match(
                provider_pair.get("background_point"),
                dataset_pair.get("background_point"),
            )
            and _geometry_fit_points_match(
                provider_pair.get("simulated_point"),
                dataset_pair.get("simulated_point"),
            )
            and provider_identity == dataset_identity
        )
        manual_point_pairs_match_provider_points = _geometry_fit_surface_match(
            provider_pair,
            manual_point_pair,
        )
        initial_pairs_match_provider_points = _geometry_fit_surface_match(
            provider_pair,
            initial_pair,
        )
        measured_for_fit_match_provider_points = _geometry_fit_surface_match(
            provider_pair,
            measured_pair,
        )
        spec_measured_peaks_match_provider_points = _geometry_fit_surface_match(
            provider_pair,
            spec_measured_pair,
        )
        provider_point_sources_picker_owned = _geometry_fit_pair_point_sources_picker_owned(
            provider_pair
        )
        dataset_point_sources_picker_owned = _geometry_fit_pair_point_sources_picker_owned(
            dataset_pair
        )
        dataset_point_sources_match_provider = _geometry_fit_pair_point_sources_match_provider(
            provider_pair,
            dataset_pair,
        )
        manual_point_pairs_point_sources_picker_owned = (
            _geometry_fit_pair_point_sources_picker_owned(manual_point_pair)
        )
        initial_pairs_point_sources_picker_owned = _geometry_fit_pair_point_sources_picker_owned(
            initial_pair
        )
        measured_for_fit_point_sources_picker_owned = _geometry_fit_pair_point_sources_picker_owned(
            measured_pair
        )
        spec_measured_peaks_point_sources_picker_owned = (
            _geometry_fit_pair_point_sources_picker_owned(spec_measured_pair)
        )
        manual_point_pairs_point_sources_match_provider = (
            _geometry_fit_pair_point_sources_match_provider(
                provider_pair,
                manual_point_pair,
            )
        )
        initial_pairs_point_sources_match_provider = (
            _geometry_fit_pair_point_sources_match_provider(
                provider_pair,
                initial_pair,
            )
        )
        measured_for_fit_point_sources_match_provider = (
            _geometry_fit_pair_point_sources_match_provider(
                provider_pair,
                measured_pair,
            )
        )
        spec_measured_peaks_point_sources_match_provider = (
            _geometry_fit_pair_point_sources_match_provider(
                provider_pair,
                spec_measured_pair,
            )
        )
        dataset_points_match_provider_points = bool(
            dataset_points_match_provider_points
            and manual_point_pairs_match_provider_points
            and initial_pairs_match_provider_points
            and measured_for_fit_match_provider_points
            and spec_measured_peaks_match_provider_points
            and provider_point_sources_picker_owned
            and dataset_point_sources_picker_owned
            and dataset_point_sources_match_provider
        )
        try:
            distance_match = bool(
                abs(
                    float(truth_pair.get("manual_selected_to_background_distance_px"))
                    - float(provider_pair.get("selected_to_background_distance_px"))
                )
                <= GEOMETRY_FIT_STORED_POINT_ABS_TOLERANCE_PX
            )
        except Exception:
            distance_match = False

        provider_fingerprint = _geometry_fit_pair_fingerprint(
            provider_pair,
            index_key="provider_pair_index",
        )
        dataset_fingerprint = _geometry_fit_pair_fingerprint(
            dataset_pair,
            index_key="dataset_pair_index",
        )
        provider_dataset_fingerprint_match = bool(provider_fingerprint == dataset_fingerprint)
        dataset_provider_mismatch = bool(
            not dataset_points_match_provider_points or not provider_dataset_fingerprint_match
        )

        row_reasons: list[str] = []
        if not truth_pair.get("manual_picker_truth_available", False):
            row_reasons.append("missing_truth")
            counts["missing_truth_pair_count"] += 1
        if not source_identity_match:
            row_reasons.append("source_identity_mismatch")
            counts["source_identity_mismatch_count"] += 1
        if not background_frame_match or not simulated_frame_match:
            row_reasons.append("frame_mismatch")
            counts["frame_mismatch_count"] += 1
        if not background_point_match:
            row_reasons.append("background_point_mismatch")
            counts["background_point_mismatch_count"] += 1
        if not simulated_point_match:
            row_reasons.append("simulated_point_mismatch")
            counts["simulated_point_mismatch_count"] += 1
        if not distance_match:
            row_reasons.append("distance_mismatch")
            counts["distance_mismatch_count"] += 1
        if bool(provider_pair.get("rebinding_fallback_used", False)):
            row_reasons.append("fallback_used")
            counts["fallback_pair_count"] += 1
        if dataset_provider_mismatch:
            row_reasons.append("dataset_provider_mismatch")
            counts["dataset_provider_mismatch_count"] += 1
        if not provider_dataset_fingerprint_match:
            row_reasons.append("provider_dataset_fingerprint_mismatch")
        if not provider_point_sources_picker_owned:
            row_reasons.append("provider_point_source_not_picker_owned")
        if not dataset_point_sources_picker_owned:
            row_reasons.append("dataset_point_source_not_picker_owned")
        if not dataset_point_sources_match_provider:
            row_reasons.append("dataset_point_source_provider_mismatch")
        if not manual_point_pairs_point_sources_picker_owned:
            row_reasons.append("manual_point_pairs_point_source_not_picker_owned")
        if not initial_pairs_point_sources_picker_owned:
            row_reasons.append("initial_pairs_point_source_not_picker_owned")
        if not measured_for_fit_point_sources_picker_owned:
            row_reasons.append("measured_for_fit_point_source_not_picker_owned")
        if not spec_measured_peaks_point_sources_picker_owned:
            row_reasons.append("spec_measured_peaks_point_source_not_picker_owned")
        if not manual_point_pairs_point_sources_match_provider:
            row_reasons.append("manual_point_pairs_point_source_provider_mismatch")
        if not initial_pairs_point_sources_match_provider:
            row_reasons.append("initial_pairs_point_source_provider_mismatch")
        if not measured_for_fit_point_sources_match_provider:
            row_reasons.append("measured_for_fit_point_source_provider_mismatch")
        if not spec_measured_peaks_point_sources_match_provider:
            row_reasons.append("spec_measured_peaks_point_source_provider_mismatch")
        if not manual_point_pairs_match_provider_points:
            row_reasons.append("manual_point_pairs_provider_mismatch")
        if not initial_pairs_match_provider_points:
            row_reasons.append("initial_pairs_provider_mismatch")
        if not measured_for_fit_match_provider_points:
            row_reasons.append("measured_for_fit_provider_mismatch")
        if not spec_measured_peaks_match_provider_points:
            row_reasons.append("spec_measured_peaks_provider_mismatch")

        row = {
            "pair_index": pair_index,
            "provider_pair_index": int(provider_pair.get("provider_pair_index", provider_index)),
            "dataset_pair_index": int(dataset_pair.get("dataset_pair_index", pair_index)),
            "manual_pair_order_key": [background_index, pair_index],
            "semantic_pair_key": provider_pair.get(
                "semantic_pair_key",
                truth_pair.get("semantic_pair_key"),
            ),
            "parity_mode": provider_pair.get("parity_mode"),
            "manual_picker_truth_available": bool(
                truth_pair.get("manual_picker_truth_available", False)
            ),
            "missing_truth_fields": list(truth_pair.get("missing_truth_fields", []) or []),
            "normalized_hkl": provider_pair.get("normalized_hkl"),
            "q_group_key": provider_pair.get("q_group_key"),
            "source_q_group_key": provider_pair.get("source_q_group_key"),
            "branch_group_key": provider_pair.get("branch_group_key"),
            "source_branch_index": provider_pair.get("source_branch_index"),
            "provider_source_coverage_key": _geometry_fit_source_coverage_key_payload(
                normalize_new4_source_coverage_key(provider_pair)
            ),
            "dataset_source_coverage_key": _geometry_fit_source_coverage_key_payload(
                normalize_new4_source_coverage_key(dataset_pair)
            ),
            "manual_background_point": truth_pair.get("manual_background_point"),
            "manual_background_frame": manual_bg_frame,
            "manual_background_point_source": truth_pair.get("manual_background_point_source"),
            "manual_selected_simulated_point": truth_pair.get("manual_selected_simulated_point"),
            "manual_selected_simulated_frame": manual_sim_frame,
            "manual_simulated_point_source": truth_pair.get("manual_simulated_point_source"),
            "manual_truth_mutated_by_refresh": bool(
                truth_pair.get("manual_truth_mutated_by_refresh", False)
            ),
            "pre_refresh_manual_background_point": truth_pair.get(
                "pre_refresh_manual_background_point"
            ),
            "post_refresh_manual_background_point": truth_pair.get(
                "post_refresh_manual_background_point"
            ),
            "pre_refresh_manual_simulated_point": truth_pair.get(
                "pre_refresh_manual_simulated_point"
            ),
            "post_refresh_manual_simulated_point": truth_pair.get(
                "post_refresh_manual_simulated_point"
            ),
            "manual_picker_selected_source_identity_canonical": manual_identity,
            "provider_selected_source_identity_canonical": provider_identity,
            "source_identity_match": source_identity_match,
            "provider_background_point": provider_pair.get("background_point"),
            "provider_background_frame": provider_bg_frame,
            "provider_background_point_source": provider_pair.get("background_point_source"),
            "provider_selected_simulated_point": provider_pair.get("simulated_point"),
            "provider_selected_simulated_frame": provider_sim_frame,
            "provider_simulated_point_source": provider_pair.get("simulated_point_source"),
            "dataset_background_point": dataset_pair.get("background_point"),
            "dataset_background_frame": dataset_bg_frame,
            "dataset_background_point_source": dataset_pair.get("background_point_source"),
            "dataset_simulated_point": dataset_pair.get("simulated_point"),
            "dataset_simulated_frame": dataset_sim_frame,
            "dataset_simulated_point_source": dataset_pair.get("simulated_point_source"),
            "background_frame_match": background_frame_match,
            "simulated_frame_match": simulated_frame_match,
            "background_point_match": background_point_match,
            "simulated_point_match": simulated_point_match,
            "manual_selected_to_background_distance_px": truth_pair.get(
                "manual_selected_to_background_distance_px"
            ),
            "provider_selected_to_background_distance_px": provider_pair.get(
                "selected_to_background_distance_px"
            ),
            "selected_to_background_distance_match": distance_match,
            "rebinding_fallback_used": bool(provider_pair.get("rebinding_fallback_used", False)),
            "fallback_reason": provider_pair.get("fallback_reason"),
            "source_locator_identity_match": bool(
                provider_pair.get("source_locator_identity_match", False)
            ),
            "source_semantic_identity_match": bool(
                provider_pair.get("source_semantic_identity_match", False)
            ),
            "stale_source_identity_diagnostic": bool(
                provider_pair.get("stale_source_identity_diagnostic", False)
            ),
            "stale_saved_source_identity": provider_pair.get("stale_saved_source_identity"),
            "provider_pair_fingerprint": provider_fingerprint,
            "dataset_pair_fingerprint": dataset_fingerprint,
            "provider_dataset_fingerprint_match": provider_dataset_fingerprint_match,
            "dataset_points_match_provider_points": dataset_points_match_provider_points,
            "provider_point_sources_picker_owned": provider_point_sources_picker_owned,
            "dataset_point_sources_picker_owned": dataset_point_sources_picker_owned,
            "dataset_point_sources_match_provider": (dataset_point_sources_match_provider),
            "manual_point_pairs_point_sources_picker_owned": (
                manual_point_pairs_point_sources_picker_owned
            ),
            "initial_pairs_point_sources_picker_owned": (initial_pairs_point_sources_picker_owned),
            "measured_for_fit_point_sources_picker_owned": (
                measured_for_fit_point_sources_picker_owned
            ),
            "spec_measured_peaks_point_sources_picker_owned": (
                spec_measured_peaks_point_sources_picker_owned
            ),
            "manual_point_pairs_point_sources_match_provider": (
                manual_point_pairs_point_sources_match_provider
            ),
            "initial_pairs_point_sources_match_provider": (
                initial_pairs_point_sources_match_provider
            ),
            "measured_for_fit_point_sources_match_provider": (
                measured_for_fit_point_sources_match_provider
            ),
            "spec_measured_peaks_point_sources_match_provider": (
                spec_measured_peaks_point_sources_match_provider
            ),
            "manual_point_pairs_match_provider_points": (manual_point_pairs_match_provider_points),
            "initial_pairs_match_provider_points": (initial_pairs_match_provider_points),
            "measured_for_fit_match_provider_points": (measured_for_fit_match_provider_points),
            "spec_measured_peaks_match_provider_points": (
                spec_measured_peaks_match_provider_points
            ),
            "mismatch_reasons": row_reasons,
        }
        pairs.append(row)
        for reason in row_reasons:
            mismatches.append(
                {
                    "pair_index": pair_index,
                    "semantic_pair_key": row.get("semantic_pair_key"),
                    "mismatch_reason": reason,
                    "manual_value": {
                        "background_point": truth_pair.get("manual_background_point"),
                        "simulated_point": truth_pair.get("manual_selected_simulated_point"),
                        "source_identity": manual_identity,
                    },
                    "provider_value": {
                        "background_point": provider_pair.get("background_point"),
                        "simulated_point": provider_pair.get("simulated_point"),
                        "source_identity": provider_identity,
                    },
                    "dataset_value": {
                        "background_point": dataset_pair.get("background_point"),
                        "simulated_point": dataset_pair.get("simulated_point"),
                        "source_identity": dataset_identity,
                    },
                    "tolerance_used": GEOMETRY_FIT_STORED_POINT_ABS_TOLERANCE_PX,
                }
            )

    surface_row_count_mismatch_reasons: list[str] = []
    extra_surface_row_count = 0
    missing_surface_row_count = 0
    surface_rows_by_name = {
        "manual_point_pairs": manual_point_rows,
        "initial_pairs_display": initial_rows,
        "measured_for_fit": measured_rows,
        "spec_measured_peaks": spec_measured_peak_rows,
    }
    for surface_name, rows in surface_rows_by_name.items():
        row_delta = int(len(rows) - len(provider_pairs))
        if row_delta == 0:
            continue
        reason = f"{surface_name}_row_count_mismatch"
        surface_row_count_mismatch_reasons.append(reason)
        counts["surface_pair_count_mismatch_count"] += abs(row_delta)
        counts[f"{surface_name}_row_count_mismatch_count"] += abs(row_delta)
        counts["dataset_provider_mismatch_count"] += abs(row_delta)
        if row_delta > 0:
            extra_surface_row_count += row_delta
        else:
            missing_surface_row_count += abs(row_delta)
        mismatches.append(
            {
                "pair_index": None,
                "semantic_pair_key": None,
                "mismatch_reason": reason,
                "surface": surface_name,
                "provider_pair_count": int(len(provider_pairs)),
                "surface_pair_count": int(len(rows)),
                "extra_surface_row_count": int(max(0, row_delta)),
                "missing_surface_row_count": int(max(0, -row_delta)),
                "tolerance_used": GEOMETRY_FIT_STORED_POINT_ABS_TOLERANCE_PX,
            }
        )
    surface_pair_count_match = not surface_row_count_mismatch_reasons
    pair_count_match = bool(
        len(truth_pairs) == len(provider_pairs) == len(dataset_pairs) and surface_pair_count_match
    )
    if not pair_count_match:
        counts["pair_count_mismatch_count"] = 1
    ordered_pairs_match = bool(pair_count_match and not mismatches)
    manual_semantic_keys = [
        _geometry_fit_jsonable(pair.get("semantic_pair_key")) for pair in truth_pairs
    ]
    provider_semantic_keys = [
        _geometry_fit_jsonable(pair.get("semantic_pair_key")) for pair in provider_pairs
    ]
    unordered_pairs_match = bool(
        sorted(json.dumps(item, sort_keys=True) for item in manual_semantic_keys)
        == sorted(json.dumps(item, sort_keys=True) for item in provider_semantic_keys)
    )
    missing_pair_count = max(0, len(truth_pairs) - len(provider_pairs))
    branch_mismatch_count = sum(
        1
        for row in pairs
        if isinstance(row.get("semantic_pair_key"), Mapping)
        and row.get("source_branch_index")
        != row.get("semantic_pair_key", {}).get("source_branch_index")
    )
    guard = dict(optimizer_guard_state or {})
    optimizer_called = bool(guard.get("optimizer_called", False))
    optimizer_call_count = int(guard.get("optimizer_call_count", 0) or 0)
    manual_point_pairs_match_provider_points = bool(
        pairs and all(row["manual_point_pairs_match_provider_points"] for row in pairs)
    )
    initial_pairs_match_provider_points = bool(
        pairs and all(row["initial_pairs_match_provider_points"] for row in pairs)
    )
    measured_for_fit_match_provider_points = bool(
        pairs and all(row["measured_for_fit_match_provider_points"] for row in pairs)
    )
    spec_measured_peaks_match_provider_points = bool(
        pairs and all(row["spec_measured_peaks_match_provider_points"] for row in pairs)
    )
    provider_dataset_fingerprint_match = bool(
        pairs and all(row["provider_dataset_fingerprint_match"] for row in pairs)
    )
    all_dataset_surfaces_match_provider_points = bool(
        manual_point_pairs_match_provider_points
        and initial_pairs_match_provider_points
        and measured_for_fit_match_provider_points
        and spec_measured_peaks_match_provider_points
        and provider_dataset_fingerprint_match
        and counts.get("dataset_provider_mismatch_count", 0) == 0
    )
    all_point_sources_picker_owned = bool(
        pairs
        and all(
            row["provider_point_sources_picker_owned"]
            and row["dataset_point_sources_picker_owned"]
            and row["dataset_point_sources_match_provider"]
            and row["manual_point_pairs_point_sources_picker_owned"]
            and row["initial_pairs_point_sources_picker_owned"]
            and row["measured_for_fit_point_sources_picker_owned"]
            and row["spec_measured_peaks_point_sources_picker_owned"]
            and row["manual_point_pairs_point_sources_match_provider"]
            and row["initial_pairs_point_sources_match_provider"]
            and row["measured_for_fit_point_sources_match_provider"]
            and row["spec_measured_peaks_point_sources_match_provider"]
            for row in pairs
        )
    )
    parity_ok = bool(
        pair_count_match
        and ordered_pairs_match
        and unordered_pairs_match
        and missing_pair_count == 0
        and branch_mismatch_count == 0
        and all_dataset_surfaces_match_provider_points
        and all_point_sources_picker_owned
        and not optimizer_called
        and optimizer_call_count == 0
    )
    report = {
        "ok": parity_ok,
        "point_provider_parity_gate": {
            "ok": parity_ok,
            "reason_codes": sorted(
                {reason for row in pairs for reason in row["mismatch_reasons"]}
                | set(surface_row_count_mismatch_reasons)
            ),
        },
        "targeted_preflight_gate": _geometry_fit_targeted_gate_from_dataset(dataset),
        "background_index": int(dataset.get("dataset_index", 0) or 0),
        "manual_picker_pair_count": int(len(truth_pairs)),
        "point_provider_pair_count": int(len(provider_pairs)),
        "manual_point_pair_count": int(len(manual_point_rows)),
        "initial_pairs_display_count": int(len(initial_rows)),
        "measured_for_fit_count": int(len(measured_rows)),
        "spec_measured_peaks_count": int(len(spec_measured_peak_rows)),
        "pair_count_match": pair_count_match,
        "surface_pair_count_match": surface_pair_count_match,
        "ordered_pairs_match": ordered_pairs_match,
        "unordered_pairs_match": unordered_pairs_match,
        "manual_point_pairs_match_provider_points": (manual_point_pairs_match_provider_points),
        "initial_pairs_match_provider_points": initial_pairs_match_provider_points,
        "measured_for_fit_match_provider_points": (measured_for_fit_match_provider_points),
        "spec_measured_peaks_match_provider_points": (spec_measured_peaks_match_provider_points),
        "provider_dataset_fingerprint_match": provider_dataset_fingerprint_match,
        "all_dataset_surfaces_match_provider_points": (all_dataset_surfaces_match_provider_points),
        "all_point_sources_picker_owned": all_point_sources_picker_owned,
        "missing_pair_count": int(missing_pair_count),
        "branch_mismatch_count": int(branch_mismatch_count),
        "pair_count_mismatch_count": int(counts.get("pair_count_mismatch_count", 0)),
        "surface_pair_count_mismatch_count": int(
            counts.get("surface_pair_count_mismatch_count", 0)
        ),
        "extra_surface_row_count": int(extra_surface_row_count),
        "missing_surface_row_count": int(missing_surface_row_count),
        "source_identity_mismatch_count": int(counts.get("source_identity_mismatch_count", 0)),
        "background_point_mismatch_count": int(counts.get("background_point_mismatch_count", 0)),
        "simulated_point_mismatch_count": int(counts.get("simulated_point_mismatch_count", 0)),
        "frame_mismatch_count": int(counts.get("frame_mismatch_count", 0)),
        "distance_mismatch_count": int(counts.get("distance_mismatch_count", 0)),
        "fallback_pair_count": int(counts.get("fallback_pair_count", 0)),
        "missing_truth_pair_count": int(counts.get("missing_truth_pair_count", 0)),
        "dataset_provider_mismatch_count": int(counts.get("dataset_provider_mismatch_count", 0)),
        "optimizer_guard_installed": bool(guard.get("optimizer_guard_installed", False)),
        "optimizer_called": optimizer_called,
        "optimizer_call_count": optimizer_call_count,
        "optimizer_path_entered": bool(guard.get("optimizer_path_entered", False)),
        "optimizer_entrypoints_called": list(guard.get("optimizer_entrypoints_called", []) or []),
        "optimizer_entrypoints_guarded": list(guard.get("optimizer_entrypoints_guarded", []) or []),
        "stored_point_abs_tolerance_px": GEOMETRY_FIT_STORED_POINT_ABS_TOLERANCE_PX,
        "recomputed_refinement_tolerance_px": GEOMETRY_FIT_RECOMPUTED_REFINEMENT_TOLERANCE_PX,
        "mismatches": mismatches,
        "pairs": pairs,
    }
    return _geometry_fit_report_jsonable(report)  # type: ignore[return-value]


def write_geometry_fit_point_provider_report(
    dataset: Mapping[str, object],
    report_path: Path | str,
    *,
    optimizer_guard_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write the machine-readable point-provider parity report."""

    report = build_geometry_fit_point_provider_report(
        dataset,
        optimizer_guard_state=optimizer_guard_state,
    )
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    register_run_output_path(path)
    return report


def collect_geometry_fit_required_manual_fit_targets(
    required_pairs: Sequence[Mapping[str, object]] | None,
    *,
    background_index: int,
) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    for fallback_index, raw_entry in enumerate(required_pairs or ()):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        required_branch_group_key = _geometry_fit_required_branch_group_key(entry)
        background_point, background_frame = _source_rebinding_background_point_and_frame(entry)
        branch_constraint_status = _geometry_fit_branch_constraint_status(entry)
        target = {
            "background_index": int(background_index),
            "pair_id": str(
                entry.get("pair_id") or f"bg{int(background_index)}:pair{int(fallback_index)}"
            ),
            "overlay_match_index": _geometry_fit_coerce_nonnegative_index(
                entry.get("overlay_match_index")
            ),
            "normalized_hkl": _geometry_fit_normalized_hkl(entry.get("hkl")),
            "source_branch_index": _geometry_fit_source_branch_index(entry),
            "q_group_key": _geometry_fit_group_identity(entry),
            "saved_background_current_view_point": background_point,
            "saved_background_current_view_frame": background_frame,
            "required_branch_group_key": required_branch_group_key,
            "branch_constraint_status": branch_constraint_status,
            "branch_unconstrained": branch_constraint_status
            in {
                "unconstrained_missing_branch",
                "zero_qr_00l_branch_unconstrained",
            },
            "source_reflection_index": _geometry_fit_coerce_nonnegative_index(
                entry.get("source_reflection_index")
            ),
            "source_reflection_namespace": entry.get("source_reflection_namespace"),
            "source_reflection_is_full": bool(entry.get("source_reflection_is_full", False)),
            "source_table_index": _geometry_fit_coerce_nonnegative_index(
                entry.get("source_table_index")
            ),
            "source_row_index": _geometry_fit_coerce_nonnegative_index(
                entry.get("source_row_index")
            ),
            "source_peak_index": _geometry_fit_coerce_nonnegative_index(
                entry.get("source_peak_index")
            ),
        }
        targets.append(target)
    return targets


def _geometry_fit_entry_matches_required_branch_group_key(
    entry: Mapping[str, object] | None,
    required_key: tuple[tuple[int, int, int], int | None, object | None],
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    required_coverage_key = normalize_new4_source_coverage_key(required_key)
    if (
        required_coverage_key is not None
        and required_coverage_key in _geometry_fit_source_coverage_alias_keys(entry)
    ):
        return True
    entry_hkl = _geometry_fit_normalized_hkl(entry.get("hkl"))
    if entry_hkl is None:
        return False
    required_group_identity = _geometry_fit_stable_group_identity(required_key[2])
    hkl_matches = tuple(entry_hkl) == tuple(required_key[0])
    group_matches = (
        _geometry_fit_group_identity_is_q_group(required_group_identity)
        and required_group_identity is not None
        and _geometry_fit_group_identity(entry) == required_group_identity
    )
    if not hkl_matches and not group_matches:
        return False
    if required_group_identity is not None:
        if _geometry_fit_group_identity(entry) != required_group_identity:
            return False
    required_branch = required_key[1]
    if required_branch is None:
        return True
    if _geometry_fit_is_zero_qr_00l(entry, required_key[2]):
        return True
    entry_branch = _geometry_fit_source_branch_index(entry)
    return entry_branch is not None and int(entry_branch) == int(required_branch)


def _geometry_fit_filter_entries_for_required_branch_groups(
    entries: Sequence[object] | None,
    required_branch_group_keys: Sequence[tuple[tuple[int, int, int], int | None, object | None]]
    | None,
) -> tuple[
    list[dict[str, object]],
    dict[str, int],
    list[tuple[tuple[int, int, int], int | None, object | None]],
]:
    copied_entries = [dict(entry) for entry in (entries or ()) if isinstance(entry, Mapping)]
    required_keys = list(required_branch_group_keys or ())
    if not copied_entries or not required_keys:
        matched_keys = [
            _geometry_fit_required_branch_group_key(entry)
            for entry in copied_entries
            if _geometry_fit_required_branch_group_key(entry) is not None
        ]
        return (
            copied_entries,
            {
                "total_count": int(len(copied_entries)),
                "after_hkl_filter_count": int(len(copied_entries)),
                "after_branch_filter_count": int(len(copied_entries)),
                "unrelated_count": 0,
            },
            [key for key in matched_keys if key is not None],
        )

    required_hkls = {tuple(key[0]) for key in required_keys}
    required_group_identities = {
        _geometry_fit_stable_group_identity(key[2])
        for key in required_keys
        if _geometry_fit_group_identity_is_q_group(key[2])
    }
    after_hkl_filter = [
        dict(entry)
        for entry in copied_entries
        if _geometry_fit_normalized_hkl(entry.get("hkl")) in required_hkls
        or _geometry_fit_group_identity(entry) in required_group_identities
    ]
    after_branch_filter = [
        dict(entry)
        for entry in after_hkl_filter
        if any(
            _geometry_fit_entry_matches_required_branch_group_key(entry, required_key)
            for required_key in required_keys
        )
    ]
    matched_keys: list[tuple[tuple[int, int, int], int | None, object | None]] = []
    for entry in after_branch_filter:
        for required_key in required_keys:
            if _geometry_fit_entry_matches_required_branch_group_key(entry, required_key):
                matched_keys.append(required_key)
    return (
        after_branch_filter,
        {
            "total_count": int(len(copied_entries)),
            "after_hkl_filter_count": int(len(after_hkl_filter)),
            "after_branch_filter_count": int(len(after_branch_filter)),
            "unrelated_count": max(0, int(len(copied_entries) - len(after_branch_filter))),
        },
        matched_keys,
    )


def _geometry_fit_hkl_inventory_from_entries(
    entries: Sequence[object] | None,
) -> list[dict[str, object]]:
    counts: dict[tuple[int, int, int], int] = {}
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        hkl = _geometry_fit_normalized_hkl(entry.get("hkl"))
        if hkl is None:
            continue
        counts[tuple(hkl)] = counts.get(tuple(hkl), 0) + 1
    return [
        {"hkl": tuple(hkl), "count": int(count)}
        for hkl, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _geometry_fit_hkl_branch_inventory_from_entries(
    entries: Sequence[object] | None,
) -> list[dict[str, object]]:
    counts: dict[tuple[tuple[int, int, int], int | None, object | None], int] = {}
    for entry in entries or ():
        if not isinstance(entry, Mapping):
            continue
        hkl = _geometry_fit_normalized_hkl(entry.get("hkl"))
        if hkl is None:
            continue
        branch_idx = _geometry_fit_source_branch_index(entry)
        branch = int(branch_idx) if branch_idx in {0, 1} else None
        q_group_key = _geometry_fit_stable_group_identity(entry.get("q_group_key"))
        counts[(tuple(hkl), branch, q_group_key)] = (
            counts.get((tuple(hkl), branch, q_group_key), 0) + 1
        )
    return [
        {
            "hkl": tuple(hkl),
            "branch_index": branch,
            "q_group_key": q_group_key,
            "count": int(count),
        }
        for (hkl, branch, q_group_key), count in sorted(
            counts.items(),
            key=lambda item: (
                item[0][0],
                -1 if item[0][1] is None else int(item[0][1]),
                repr(item[0][2]),
            ),
        )
    ]


def _geometry_fit_required_hkl_inventory(
    required_branch_group_keys: Sequence[tuple[tuple[int, int, int], int | None, object | None]]
    | None,
) -> list[dict[str, object]]:
    counts: dict[tuple[int, int, int], int] = {}
    for key in required_branch_group_keys or ():
        if not isinstance(key, (tuple, list)) or len(key) < 1:
            continue
        hkl = _geometry_fit_normalized_hkl(key[0])
        if hkl is None:
            continue
        counts[tuple(hkl)] = counts.get(tuple(hkl), 0) + 1
    return [
        {"hkl": tuple(hkl), "count": int(count)}
        for hkl, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _geometry_fit_required_branch_group_key_payloads(
    required_branch_group_keys: Sequence[tuple[tuple[int, int, int], int | None, object | None]]
    | None,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for key in required_branch_group_keys or ():
        if not isinstance(key, (tuple, list)) or len(key) < 3:
            continue
        hkl = _geometry_fit_normalized_hkl(key[0])
        if hkl is None:
            continue
        branch = None
        try:
            if key[1] is not None and int(key[1]) in {0, 1}:
                branch = int(key[1])
        except Exception:
            branch = None
        payloads.append(
            {
                "hkl": tuple(hkl),
                "branch_index": branch,
                "q_group_key": _geometry_fit_stable_group_identity(key[2]),
            }
        )
    return payloads


def _geometry_fit_targeted_performance_gate_payload(
    payload: Mapping[str, object] | None,
) -> dict[str, object]:
    gate_payload = dict(payload or {})
    ok = bool(
        gate_payload.get("targeted_preflight_enabled", False)
        and str(gate_payload.get("preflight_mode") or "full") == "manual_geometry_targeted"
        and int(gate_payload.get("unrelated_projected_row_count_for_rebinding", 0) or 0) == 0
        and int(gate_payload.get("unrelated_scored_row_count_for_rebinding", 0) or 0) == 0
        and not bool(gate_payload.get("full_source_rows_built_for_rebinding", False))
        and not bool(gate_payload.get("full_source_rows_projected_for_rebinding", False))
        and (
            bool(gate_payload.get("targeted_cache_hit", False))
            or bool(gate_payload.get("targeted_simulation_used", False))
        )
    )
    if bool(gate_payload.get("full_fresh_simulation_fallback_used", False)):
        ok = False
    if (
        str(gate_payload.get("targeted_simulation_fallback_reason") or "")
        == "simulator_filter_not_supported"
    ):
        ok = False
    gate_payload["ok"] = bool(ok)
    return gate_payload


def _geometry_fit_is_canonical_live_source_entry(
    entry: Mapping[str, object] | None,
) -> tuple[bool, str | None]:
    if not isinstance(entry, Mapping):
        return False, "not_mapping"
    branch_idx = _geometry_fit_source_branch_index(entry)
    if branch_idx not in {0, 1}:
        return False, _geometry_fit_source_branch_reason(entry) or "missing_branch"
    if not _geometry_fit_trusted_full_reflection_identity(entry):
        return False, "untrusted_full_identity"
    return True, None


def _geometry_fit_trace_candidate_inventory(
    candidates: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Build compact per-candidate diagnostics for live-row validation traces."""

    inventory: list[dict[str, object]] = []
    for raw_candidate in candidates or ():
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate = dict(raw_candidate)
        inventory.append(
            {
                "source_reflection_index": candidate.get("source_reflection_index"),
                "source_branch_index": _geometry_fit_source_branch_index(candidate),
                "source_peak_index": candidate.get("source_peak_index"),
                "source_table_index": candidate.get("source_table_index"),
                "source_row_index": candidate.get("source_row_index"),
                "hkl": _geometry_fit_normalized_hkl(candidate.get("hkl")),
                "q_group_key": _geometry_fit_cache_jsonable(candidate.get("q_group_key")),
                "source_coverage_key": _geometry_fit_source_coverage_key_payload(
                    normalize_new4_source_coverage_key(candidate)
                ),
                "simulated_point": [
                    _geometry_fit_metric_float(candidate.get("sim_col", np.nan)),
                    _geometry_fit_metric_float(candidate.get("sim_row", np.nan)),
                ],
                "source_reflection_namespace": candidate.get("source_reflection_namespace"),
                "source_reflection_is_full": bool(
                    candidate.get("source_reflection_is_full", False)
                ),
            }
        )
    return inventory


def _geometry_fit_compact_source_resolution_entry_payload(
    entry: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Return one compact diagnostics payload for source-resolution tracing."""

    if not isinstance(entry, Mapping):
        return None

    payload: dict[str, object] = {}
    for key in (
        "pair_id",
        "fit_run_id",
        "label",
        "x",
        "y",
        "detector_x",
        "detector_y",
        "caked_x",
        "caked_y",
        "raw_caked_x",
        "raw_caked_y",
        "refined_sim_x",
        "refined_sim_y",
        "refined_sim_native_x",
        "refined_sim_native_y",
        "refined_sim_caked_x",
        "refined_sim_caked_y",
        "sim_col",
        "sim_row",
        "sim_col_raw",
        "sim_row_raw",
        "two_theta_deg",
        "phi_deg",
        "simulated_phi_deg",
        "background_phi_deg",
        "source_table_index",
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
        "source_row_index",
        "source_branch_index",
        "source_peak_index",
        "legacy_source_reflection_index",
        "legacy_source_peak_index",
        "row_origin",
        "provider_backed_live_source_row",
        "is_00l_collapsed",
    ):
        if key in entry:
            payload[key] = _geometry_fit_cache_jsonable(entry.get(key))

    hkl = _geometry_fit_normalized_hkl(entry.get("hkl"))
    if hkl is not None:
        payload["hkl"] = [int(v) for v in hkl]

    if "q_group_key" in entry:
        payload["q_group_key"] = _geometry_fit_cache_jsonable(entry.get("q_group_key"))
    if "source_coverage_aliases" in entry:
        payload["source_coverage_aliases"] = _geometry_fit_cache_jsonable(
            entry.get("source_coverage_aliases")
        )

    return payload


def _geometry_fit_resolved_pair_trace_entry(
    *,
    pair_id: str,
    entry: Mapping[str, object],
    candidate: Mapping[str, object],
    resolution_kind: str,
) -> dict[str, object]:
    """Return the canonical pair-resolution trace for one validated pair."""

    source_peak_key = _geometry_fit_source_peak_key(candidate)
    target_coverage_key = normalize_new4_source_coverage_key(entry)
    candidate_coverage_key = normalize_new4_source_coverage_key(candidate)
    return {
        "pair_id": pair_id,
        "overlay_match_index": entry.get("overlay_match_index"),
        "resolution_kind": str(resolution_kind),
        "hkl": _geometry_fit_normalized_hkl(entry.get("hkl")),
        "source_reflection_index": candidate.get("source_reflection_index"),
        "source_branch_index": _geometry_fit_source_branch_index(candidate),
        "source_peak_index": candidate.get("source_peak_index"),
        "source_table_index": candidate.get("source_table_index"),
        "source_row_index": candidate.get("source_row_index"),
        "target_source_coverage_key": _geometry_fit_source_coverage_key_payload(
            target_coverage_key
        ),
        "candidate_source_coverage_key": _geometry_fit_source_coverage_key_payload(
            candidate_coverage_key
        ),
        "canonical_identity": (
            [int(source_peak_key[0]), int(source_peak_key[1])]
            if isinstance(source_peak_key, tuple)
            else None
        ),
        "q_group_key": _geometry_fit_cache_jsonable(entry.get("q_group_key")),
        "simulated_point": [
            _geometry_fit_metric_float(candidate.get("sim_col", np.nan)),
            _geometry_fit_metric_float(candidate.get("sim_row", np.nan)),
        ],
        "source_reflection_namespace": candidate.get("source_reflection_namespace"),
        "source_reflection_is_full": bool(candidate.get("source_reflection_is_full", False)),
    }


def _geometry_fit_required_pair_dual_path_diff(
    before_validation: Mapping[str, object] | None,
    after_validation: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Diff required-pair resolution between live-cache and rebuild paths."""

    before_map: dict[str, dict[str, object]] = {}
    after_map: dict[str, dict[str, object]] = {}
    for raw_entry in (
        before_validation.get("resolved_pairs", ())
        if isinstance(before_validation, Mapping)
        else ()
    ):
        if not isinstance(raw_entry, Mapping):
            continue
        pair_id = str(raw_entry.get("pair_id", "")).strip()
        if pair_id:
            before_map[pair_id] = {"status": "resolved", **dict(raw_entry)}
    for raw_entry in (
        before_validation.get("pair_failures", ()) if isinstance(before_validation, Mapping) else ()
    ):
        if not isinstance(raw_entry, Mapping):
            continue
        pair_id = str(raw_entry.get("pair_id", "")).strip()
        if pair_id:
            before_map[pair_id] = {"status": "failed", **dict(raw_entry)}
    for raw_entry in (
        after_validation.get("resolved_pairs", ()) if isinstance(after_validation, Mapping) else ()
    ):
        if not isinstance(raw_entry, Mapping):
            continue
        pair_id = str(raw_entry.get("pair_id", "")).strip()
        if pair_id:
            after_map[pair_id] = {"status": "resolved", **dict(raw_entry)}
    for raw_entry in (
        after_validation.get("pair_failures", ()) if isinstance(after_validation, Mapping) else ()
    ):
        if not isinstance(raw_entry, Mapping):
            continue
        pair_id = str(raw_entry.get("pair_id", "")).strip()
        if pair_id:
            after_map[pair_id] = {"status": "failed", **dict(raw_entry)}

    diffs: list[dict[str, object]] = []
    for pair_id in sorted(set(before_map) | set(after_map)):
        before_entry = before_map.get(pair_id, {"status": "missing"})
        after_entry = after_map.get(pair_id, {"status": "missing"})
        before_identity = before_entry.get("canonical_identity")
        after_identity = after_entry.get("canonical_identity")
        before_point = before_entry.get("simulated_point")
        after_point = after_entry.get("simulated_point")
        before_reason = before_entry.get("reason")
        after_reason = after_entry.get("reason")
        if (
            before_entry.get("status") == after_entry.get("status")
            and before_identity == after_identity
            and before_point == after_point
            and before_reason == after_reason
        ):
            continue
        diffs.append(
            {
                "pair_id": pair_id,
                "before_status": before_entry.get("status"),
                "after_status": after_entry.get("status"),
                "before_canonical_identity": before_identity,
                "after_canonical_identity": after_identity,
                "before_simulated_point": before_point,
                "after_simulated_point": after_point,
                "before_reason": before_reason,
                "after_reason": after_reason,
            }
        )
    return diffs


def validate_geometry_fit_live_source_rows(
    live_rows: Sequence[object] | None,
    *,
    required_pairs: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate whether live/source rows satisfy canonical identity for active pairs."""

    rows = [dict(entry) for entry in (live_rows or ()) if isinstance(entry, Mapping)]
    trusted_full_count = 0
    canonical_branch_count = 0
    canonical_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    canonical_id_counts: Counter[tuple[int, int]] = Counter()
    for row_index, entry in enumerate(rows):
        if _geometry_fit_trusted_full_reflection_identity(entry):
            trusted_full_count += 1
        branch_idx = _geometry_fit_source_branch_index(entry)
        if branch_idx in {0, 1}:
            canonical_branch_count += 1
        is_canonical, reason = _geometry_fit_is_canonical_live_source_entry(entry)
        if is_canonical:
            canonical_rows.append(dict(entry))
            peak_key = _geometry_fit_source_peak_key(entry)
            if peak_key is not None:
                canonical_id_counts[(int(peak_key[0]), int(peak_key[1]))] += 1
            continue
        invalid_rows.append(
            {
                "row_index": int(row_index),
                "reason": str(reason or "invalid"),
                "source_table_index": entry.get("source_table_index"),
                "source_reflection_index": entry.get("source_reflection_index"),
                "source_row_index": entry.get("source_row_index"),
                "source_peak_index": entry.get("source_peak_index"),
                "source_branch_index": entry.get("source_branch_index"),
                "hkl": _geometry_fit_normalized_hkl(entry.get("hkl")),
            }
        )

    canonical_by_row: dict[tuple[int, int], dict[str, object]] = {}
    canonical_by_reflection_row: dict[tuple[int, int], dict[str, object]] = {}
    canonical_by_peak: dict[tuple[int, int], dict[str, object]] = {}
    canonical_by_q_group: dict[object, list[dict[str, object]]] = defaultdict(list)
    canonical_by_hkl: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    canonical_by_coverage_key: dict[
        tuple[tuple[int, int, int], object | None, object | None],
        list[dict[str, object]],
    ] = defaultdict(list)
    for entry in canonical_rows:
        row_key = _geometry_fit_source_row_key(entry)
        if row_key is not None:
            canonical_by_row[row_key] = dict(entry)
        reflection_row_key = _geometry_fit_source_reflection_row_key(entry)
        if reflection_row_key is not None:
            canonical_by_reflection_row[reflection_row_key] = dict(entry)
        peak_key = _geometry_fit_source_peak_key(entry)
        if peak_key is not None:
            canonical_by_peak[peak_key] = dict(entry)
        q_group_key = _geometry_fit_group_identity(entry)
        if q_group_key is not None:
            canonical_by_q_group[q_group_key].append(dict(entry))
        hkl_key = _geometry_fit_normalized_hkl(entry.get("hkl"))
        if hkl_key is not None:
            canonical_by_hkl[hkl_key].append(dict(entry))
        for coverage_key in _geometry_fit_source_coverage_alias_keys(entry):
            canonical_by_coverage_key[coverage_key].append(dict(entry))

    pair_failures: list[dict[str, object]] = []
    resolved_pairs: list[dict[str, object]] = []
    validated_pair_count = 0
    for fallback_index, raw_entry in enumerate(required_pairs or ()):
        if not isinstance(raw_entry, Mapping):
            continue
        validated_pair_count += 1
        entry = dict(raw_entry)
        pair_id = str(
            entry.get("pair_id") or f"pair[{int(entry.get('overlay_match_index', fallback_index))}]"
        )
        trusted_identity = _geometry_fit_trusted_full_reflection_identity(entry)
        entry_branch_idx = _geometry_fit_source_branch_index(entry)
        entry_branch_reason = _geometry_fit_source_branch_reason(entry) or "missing_branch"
        candidate: dict[str, object] | None = None
        failure_reason: str | None = None
        resolution_kind: str | None = None
        branch_candidates: list[dict[str, object]] = []
        candidate_count_total = 0
        candidate_count_after_hkl_filter = 0
        candidate_count_after_branch_filter = 0
        target_coverage_key = normalize_new4_source_coverage_key(entry)
        deferred_trusted_failure_reason: str | None = None

        if trusted_identity and entry_branch_idx not in {0, 1}:
            failure_reason = str(entry_branch_reason)

        reflection_row_key = _geometry_fit_source_reflection_row_key(entry)
        if candidate is None and failure_reason is None and reflection_row_key is not None:
            candidate = canonical_by_reflection_row.get(reflection_row_key)
            hkl_matches = isinstance(candidate, Mapping) and _geometry_fit_source_entry_hkl_matches(
                entry, candidate
            )
            branch_matches = isinstance(
                candidate, Mapping
            ) and _geometry_fit_source_entry_branch_matches(entry, candidate)
            if not (hkl_matches and branch_matches):
                candidate = None
                if trusted_identity:
                    deferred_trusted_failure_reason = (
                        "branch_mismatch"
                        if hkl_matches and not branch_matches
                        else "missing_trusted_reflection_row"
                    )
            elif isinstance(candidate, Mapping):
                resolution_kind = "trusted_reflection_row"

        if candidate is None and failure_reason is None and target_coverage_key is not None:
            coverage_candidates = [
                dict(item)
                for item in canonical_by_coverage_key.get(target_coverage_key, ())
                if isinstance(item, Mapping)
                and _geometry_fit_source_entry_hkl_matches(entry, item)
                and _geometry_fit_source_entry_branch_matches(entry, item)
            ]
            if len(coverage_candidates) == 1:
                candidate = dict(coverage_candidates[0])
                resolution_kind = (
                    "source_coverage_alias"
                    if target_coverage_key
                    in _geometry_fit_source_coverage_alias_keys(coverage_candidates[0])
                    else "source_coverage_key"
                )
                deferred_trusted_failure_reason = None
            elif len(coverage_candidates) > 1 and not trusted_identity:
                candidate_pool = _geometry_fit_filter_branch_candidates(
                    entry,
                    coverage_candidates,
                )
                if len(candidate_pool) == 1:
                    candidate = dict(candidate_pool[0])
                    resolution_kind = "source_coverage_key"
                else:
                    failure_reason = "ambiguous_coverage_candidate"

        if (
            candidate is None
            and failure_reason is None
            and deferred_trusted_failure_reason is not None
        ):
            failure_reason = str(deferred_trusted_failure_reason)

        if candidate is None and not trusted_identity:
            row_key = _geometry_fit_source_row_key(entry)
            if row_key is not None:
                row_candidate = canonical_by_row.get(row_key)
                if (
                    isinstance(row_candidate, Mapping)
                    and _geometry_fit_source_entry_hkl_matches(entry, row_candidate)
                    and _geometry_fit_source_entry_branch_matches(entry, row_candidate)
                ):
                    candidate = dict(row_candidate)
                    resolution_kind = "source_row"

        if candidate is None and not trusted_identity:
            peak_key = _geometry_fit_source_peak_key(entry)
            if peak_key is not None:
                peak_candidate = canonical_by_peak.get(peak_key)
                if (
                    isinstance(peak_candidate, Mapping)
                    and _geometry_fit_source_entry_hkl_matches(entry, peak_candidate)
                    and _geometry_fit_source_entry_branch_matches(entry, peak_candidate)
                ):
                    candidate = dict(peak_candidate)
                    resolution_kind = "source_peak"

        if candidate is None:
            candidate_pool: list[dict[str, object]] = []
            q_group_key = _geometry_fit_group_identity(entry)
            if q_group_key is not None:
                candidate_pool = [
                    dict(item)
                    for item in canonical_by_q_group.get(q_group_key, ())
                    if isinstance(item, Mapping)
                ]
            if not candidate_pool:
                hkl_key = _geometry_fit_normalized_hkl(entry.get("hkl"))
                if hkl_key is not None:
                    candidate_pool = [
                        dict(item)
                        for item in canonical_by_hkl.get(hkl_key, ())
                        if isinstance(item, Mapping)
                    ]
            candidate_count_total = int(len(candidate_pool))
            candidate_pool = [
                dict(item)
                for item in candidate_pool
                if _geometry_fit_source_entry_hkl_matches(entry, item)
            ]
            candidate_count_after_hkl_filter = int(len(candidate_pool))
            candidate_pool = _geometry_fit_filter_branch_candidates(entry, candidate_pool)
            candidate_count_after_branch_filter = int(len(candidate_pool))
            branch_candidates = _geometry_fit_trace_candidate_inventory(candidate_pool)
        if candidate is None and not trusted_identity:
            if len(candidate_pool) == 1:
                candidate = dict(candidate_pool[0])
                resolution_kind = "q_group_or_hkl"
            elif len(candidate_pool) > 1:
                failure_reason = "ambiguous_canonical_candidate"

        if candidate is None:
            pair_failures.append(
                {
                    "pair_id": pair_id,
                    "overlay_match_index": entry.get("overlay_match_index"),
                    "reason": str(failure_reason or "missing_canonical_candidate"),
                    "hkl": _geometry_fit_normalized_hkl(entry.get("hkl")),
                    "source_table_index": entry.get("source_table_index"),
                    "source_reflection_index": entry.get("source_reflection_index"),
                    "source_row_index": entry.get("source_row_index"),
                    "source_peak_index": entry.get("source_peak_index"),
                    "source_branch_index": _geometry_fit_source_branch_index(entry),
                    "target_source_coverage_key": _geometry_fit_source_coverage_key_payload(
                        target_coverage_key
                    ),
                    "trusted_identity_required": bool(trusted_identity),
                    "candidate_count_total": int(candidate_count_total),
                    "candidate_count_after_hkl_filter": int(candidate_count_after_hkl_filter),
                    "candidate_count_after_branch_filter": int(candidate_count_after_branch_filter),
                    "branch_candidates": branch_candidates,
                }
            )
            continue

        entry_branch_idx = _geometry_fit_source_branch_index(entry)
        candidate_branch_idx = _geometry_fit_source_branch_index(candidate)
        if (
            _geometry_fit_is_zero_qr_00l(entry)
            and _geometry_fit_is_zero_qr_00l(candidate)
            and entry_branch_idx is not None
            and candidate_branch_idx is not None
            and int(entry_branch_idx) != int(candidate_branch_idx)
        ):
            resolution_kind = "zero_qr_00l_branch_unconstrained"
        resolved_pairs.append(
            _geometry_fit_resolved_pair_trace_entry(
                pair_id=pair_id,
                entry=entry,
                candidate=candidate,
                resolution_kind=str(resolution_kind or "canonical"),
            )
        )

    duplicate_canonical_ids = {
        f"{int(key[0])}:{int(key[1])}": int(count)
        for key, count in canonical_id_counts.items()
        if int(count) > 1
    }
    hkl_missing_candidate_count = 0
    branch_mismatch_count = 0
    for failure in pair_failures:
        reason = str(failure.get("reason", "") or "")
        if reason == "branch_mismatch":
            branch_mismatch_count += 1
            continue
        if reason != "missing_canonical_candidate":
            continue
        candidate_count_total = int(failure.get("candidate_count_total", 0) or 0)
        candidate_count_after_hkl_filter = int(
            failure.get("candidate_count_after_hkl_filter", 0) or 0
        )
        candidate_count_after_branch_filter = int(
            failure.get("candidate_count_after_branch_filter", 0) or 0
        )
        if candidate_count_total > 0 and candidate_count_after_hkl_filter == 0:
            hkl_missing_candidate_count += 1
        elif candidate_count_after_hkl_filter > 0 and candidate_count_after_branch_filter == 0:
            branch_mismatch_count += 1
    return {
        "valid": not pair_failures,
        "row_count": int(len(rows)),
        "canonical_row_count": int(len(canonical_rows)),
        "trusted_full_id_count": int(trusted_full_count),
        "canonical_branch_id_count": int(canonical_branch_count),
        "required_pair_count": int(validated_pair_count),
        "validated_pair_count": int(validated_pair_count),
        "missing_required_pair_count": int(len(pair_failures)),
        "branch_mismatch_count": int(branch_mismatch_count),
        "hkl_missing_candidate_count": int(hkl_missing_candidate_count),
        "pair_failures": pair_failures,
        "resolved_pairs": resolved_pairs,
        "invalid_rows": invalid_rows,
        "duplicate_canonical_ids": duplicate_canonical_ids,
    }


def rebuild_geometry_fit_source_rows(
    *,
    background_index: int,
    background_label: str | None = None,
    params_local: Mapping[str, object],
    consumer: str,
    prior_diagnostics: Mapping[str, object] | None,
    requested_signature: object,
    requested_signature_summary: object,
    can_use_live_runtime_cache: bool,
    build_live_rows: Callable[[], object] | None,
    get_memory_intersection_cache: Callable[[], Sequence[object]] | None,
    memory_cache_signature: object | None = None,
    load_logged_intersection_cache_metadata: (
        Callable[[], Mapping[str, object] | None] | None
    ) = None,
    load_logged_intersection_cache: Callable[
        [],
        tuple[Sequence[object], Mapping[str, object] | None],
    ]
    | None,
    logged_cache_matches_params: Callable[
        [Mapping[str, object] | None, Mapping[str, object]],
        bool | Mapping[str, object],
    ]
    | None,
    build_source_rows_from_hit_tables: Callable[
        [Sequence[object]],
        tuple[
            Sequence[object],
            Sequence[object] | None,
            Sequence[object] | None,
            Sequence[int] | None,
        ],
    ],
    simulate_hit_tables: Callable[..., Sequence[object]] | None,
    last_runtime_simulation_diagnostics: (Callable[[], Mapping[str, object]] | None) = None,
    project_rows: Callable[[Sequence[object] | None], Sequence[object]] | None = None,
    required_pairs: Sequence[Mapping[str, object]] | None = None,
    required_branch_group_keys: Sequence[tuple[tuple[int, int, int], int | None, object | None]]
    | None = None,
    required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
    preflight_mode: str = "full",
    projection_view_mode: str | None = None,
    projection_view_signature: object = None,
    projection_payload: object = None,
    project_rows_for_background_view: (
        Callable[[Sequence[object] | None], Sequence[object]] | None
    ) = None,
    live_cache_inventory: Mapping[str, object] | Callable[[], Mapping[str, object]] | None = None,
    get_targeted_projected_cache: Callable[[str], Mapping[str, object] | None] | None = None,
    store_targeted_projected_cache: (Callable[[str, Mapping[str, object]], None] | None) = None,
    stage_callback: GeometryFitStageCallback | None = None,
) -> GeometryFitSourceRowRebuildResult:
    """Rebuild source rows without mutating runtime state."""

    background_idx = int(background_index)
    lookup_context = str(consumer or "unspecified")
    resolved_background_label = (
        str(background_label).strip()
        if background_label is not None and str(background_label).strip()
        else f"background {int(background_idx) + 1}"
    )
    normalized_params = dict(params_local or {})
    normalized_preflight_mode = str(preflight_mode or "full").strip().lower() or "full"
    normalized_requested_signature = _geometry_fit_cache_jsonable(requested_signature)
    normalized_requested_signature_summary = _geometry_fit_cache_jsonable(
        requested_signature_summary
    )
    normalized_projection_view_mode = (
        str(projection_view_mode).strip().lower()
        if projection_view_mode is not None and str(projection_view_mode).strip()
        else None
    )
    normalized_projection_view_signature = (
        dict(projection_view_signature)
        if isinstance(projection_view_signature, Mapping)
        else _geometry_fit_cache_jsonable(projection_view_signature)
    )
    normalized_projection_payload = (
        copy.deepcopy(dict(projection_payload)) if isinstance(projection_payload, Mapping) else None
    )
    collected_required_manual_fit_targets = [
        dict(entry)
        for entry in (
            required_manual_fit_targets
            if required_manual_fit_targets is not None
            else collect_geometry_fit_required_manual_fit_targets(
                required_pairs,
                background_index=int(background_idx),
            )
        )
        if isinstance(entry, Mapping)
    ]
    collected_required_branch_group_keys = list(
        required_branch_group_keys
        if required_branch_group_keys is not None
        else _geometry_fit_required_branch_group_keys(collected_required_manual_fit_targets)
    )
    if (
        normalized_preflight_mode == "full"
        and collected_required_manual_fit_targets
        and collected_required_branch_group_keys
    ):
        normalized_preflight_mode = "manual_geometry_targeted"
    targeted_preflight_enabled = bool(
        normalized_preflight_mode == "manual_geometry_targeted"
        and collected_required_branch_group_keys
    )
    force_fresh_simulation = lookup_context == "geometry_fit_trial_source_rows"
    required_branch_group_keys_digest = _geometry_fit_required_branch_group_keys_digest(
        collected_required_branch_group_keys,
        background_index=int(background_idx),
        requested_signature=normalized_requested_signature,
        requested_signature_summary=normalized_requested_signature_summary,
        preflight_mode=normalized_preflight_mode,
        consumer=lookup_context,
        projection_view_mode=normalized_projection_view_mode,
        projection_view_signature=normalized_projection_view_signature,
    )
    manual_target_scoring_digest = _geometry_fit_manual_target_scoring_digest(
        collected_required_manual_fit_targets
    )
    rebuild_attempts: list[str] = []
    live_cache_validation: dict[str, object] | None = None
    live_runtime_cache_metadata: dict[str, object] = {}
    rebuild_started_at = perf_counter()
    stage_callback_failure_count = 0
    stage_callback_last_failed_stage: str | None = None
    projected_rows_failure_reason: str | None = None
    targeted_runtime_flags: dict[str, object] = {
        "preflight_mode": normalized_preflight_mode,
        "targeted_preflight_enabled": bool(targeted_preflight_enabled),
        "required_manual_pair_count": int(len(collected_required_manual_fit_targets)),
        "required_hkl_branch_group_count": int(len(collected_required_branch_group_keys)),
        "required_hkl_branch_keys_digest": str(required_branch_group_keys_digest),
        "manual_target_scoring_digest": str(manual_target_scoring_digest),
        "targeted_simulation_supported": False,
        "targeted_simulation_used": False,
        "targeted_simulation_fallback_reason": None,
        "targeted_cache_hit": False,
        "cache_source": None,
        "total_hit_tables_available": 0,
        "hit_tables_considered_for_rebinding": 0,
        "hit_tables_expanded_for_rebinding": 0,
        "total_source_rows_available": 0,
        "source_rows_considered_for_rebinding": 0,
        "source_rows_projected_for_rebinding": 0,
        "candidate_rows_after_hkl_filter": 0,
        "candidate_rows_after_branch_filter": 0,
        "candidate_rows_scored_for_background_distance": 0,
        "unrelated_available_row_count_for_rebinding": 0,
        "unrelated_projected_row_count_for_rebinding": 0,
        "unrelated_scored_row_count_for_rebinding": 0,
        "full_source_rows_built_for_rebinding": False,
        "full_source_rows_projected_for_rebinding": False,
        "full_fresh_simulation_fallback_used": False,
        "required_manual_fit_targets": copy.deepcopy(collected_required_manual_fit_targets),
        "required_branch_group_keys": copy.deepcopy(
            _geometry_fit_required_branch_group_key_payloads(collected_required_branch_group_keys)
        ),
        "required_hkl_inventory": _geometry_fit_required_hkl_inventory(
            collected_required_branch_group_keys
        ),
    }

    def _emit_rebuild_stage(
        stage: str,
        *,
        stage_started_at: float | None = None,
        **payload: object,
    ) -> None:
        nonlocal stage_callback_failure_count, stage_callback_last_failed_stage
        if not callable(stage_callback):
            return
        event_payload = {
            "background_index": int(background_idx),
            "background_label": resolved_background_label,
            "elapsed_s": float(max(0.0, perf_counter() - rebuild_started_at)),
            **payload,
        }
        if stage_started_at is not None:
            event_payload["stage_elapsed_s"] = float(max(0.0, perf_counter() - stage_started_at))
        try:
            stage_callback(str(stage), dict(event_payload))
        except Exception:
            stage_callback_failure_count += 1
            stage_callback_last_failed_stage = str(stage)

    def _targeted_gate_payload() -> dict[str, object]:
        payload = dict(targeted_runtime_flags)
        payload["preflight_mode"] = normalized_preflight_mode
        return _geometry_fit_targeted_performance_gate_payload(payload)

    def _update_targeted_runtime_flags(**fields: object) -> None:
        for key, value in fields.items():
            targeted_runtime_flags[str(key)] = copy.deepcopy(value)

    def _emit_target_collection_events() -> None:
        if not targeted_preflight_enabled:
            return
        _emit_rebuild_stage(
            "source_cache_target_collection_start",
            preflight_mode=normalized_preflight_mode,
            required_pair_count=int(len(collected_required_manual_fit_targets)),
        )
        _emit_rebuild_stage(
            "source_cache_target_collection_ready",
            preflight_mode=normalized_preflight_mode,
            targeted_preflight_enabled=True,
            required_pair_count=int(len(collected_required_manual_fit_targets)),
            required_hkl_branch_group_count=int(len(collected_required_branch_group_keys)),
            required_hkl_branch_keys_digest=str(required_branch_group_keys_digest),
        )

    def _accepted_optional_keywords(
        callback: Callable[..., object] | None,
        kwargs: Mapping[str, object],
    ) -> tuple[dict[str, object], bool]:
        if not callable(callback) or not kwargs:
            return {}, True
        try:
            signature = inspect.signature(callback)
        except Exception:
            return dict(kwargs), False
        has_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if has_var_keyword:
            return dict(kwargs), True
        return (
            {key: value for key, value in kwargs.items() if key in signature.parameters},
            True,
        )

    def _call_with_optional_keywords(
        callback: Callable[..., object] | None,
        /,
        *args: object,
        **kwargs: object,
    ) -> tuple[object, tuple[str, ...]]:
        if not callable(callback):
            return None, ()
        if not kwargs:
            return callback(*args), ()
        accepted_kwargs, _signature_known = _accepted_optional_keywords(
            callback,
            kwargs,
        )
        try:
            return callback(*args, **accepted_kwargs), tuple(accepted_kwargs)
        except TypeError as exc:
            error_text = str(exc)
            if kwargs and (
                "unexpected keyword" in error_text
                or "positional argument" in error_text
                or "got an unexpected keyword" in error_text
            ):
                return callback(*args), ()
            raise

    def _resolve_logged_cache_match_payload(
        metadata_local: Mapping[str, object] | None,
    ) -> dict[str, object]:
        payload: dict[str, object]
        if callable(logged_cache_matches_params):
            raw_payload = logged_cache_matches_params(metadata_local, normalized_params)
            if isinstance(raw_payload, Mapping):
                payload = dict(raw_payload)
            else:
                payload = {"matches": bool(raw_payload)}
        else:
            payload = {"matches": False}
        payload.setdefault("matches", False)
        payload.setdefault("heavy_hit_table_load_attempted", False)
        return payload

    def _filter_rows_for_rebinding(
        raw_rows: Sequence[object] | None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        copied_rows = _copy_rows(raw_rows)
        filtered_rows, counts, matched_keys = (
            _geometry_fit_filter_entries_for_required_branch_groups(
                copied_rows,
                collected_required_branch_group_keys,
            )
            if targeted_preflight_enabled and not force_fresh_simulation
            else (
                copied_rows,
                {
                    "total_count": int(len(copied_rows)),
                    "after_hkl_filter_count": int(len(copied_rows)),
                    "after_branch_filter_count": int(len(copied_rows)),
                    "unrelated_count": 0,
                },
                [
                    key
                    for key in (
                        _geometry_fit_required_branch_group_key(entry) for entry in copied_rows
                    )
                    if key is not None
                ],
            )
        )
        required_payloads = _geometry_fit_required_branch_group_key_payloads(
            collected_required_branch_group_keys
        )
        matched_payloads = _geometry_fit_required_branch_group_key_payloads(matched_keys)
        matched_payload_keys = {
            (tuple(item.get("hkl") or ()), item.get("branch_index"), item.get("q_group_key"))
            for item in matched_payloads
        }
        missing_required_payloads = [
            item
            for item in required_payloads
            if (
                tuple(item.get("hkl") or ()),
                item.get("branch_index"),
                item.get("q_group_key"),
            )
            not in matched_payload_keys
        ]
        diagnostics_local = {
            "total_source_rows_available": int(counts.get("total_count", 0) or 0),
            "source_rows_considered_for_rebinding": int(counts.get("total_count", 0) or 0),
            "candidate_rows_after_hkl_filter": int(counts.get("after_hkl_filter_count", 0) or 0),
            "candidate_rows_after_branch_filter": int(
                counts.get("after_branch_filter_count", 0) or 0
            ),
            "candidate_rows_scored_for_background_distance": int(
                counts.get("after_branch_filter_count", 0) or 0
            ),
            "unrelated_available_row_count_for_rebinding": int(
                counts.get("unrelated_count", 0) or 0
            ),
            "unrelated_scored_row_count_for_rebinding": 0,
            "unrelated_projected_row_count_for_rebinding": 0,
            "required_branch_group_keys_seen": copy.deepcopy(list(matched_keys)),
            "missing_required_branch_group_keys": missing_required_payloads,
            "missing_required_hkl_inventory": _geometry_fit_required_hkl_inventory(
                [
                    (
                        tuple(item.get("hkl") or ()),
                        item.get("branch_index"),
                        item.get("q_group_key"),
                    )
                    for item in missing_required_payloads
                ]
            ),
            "source_row_hkl_inventory_before_rebinding_filter": (
                _geometry_fit_hkl_inventory_from_entries(copied_rows)
            ),
            "source_row_hkl_branch_inventory_before_rebinding_filter": (
                _geometry_fit_hkl_branch_inventory_from_entries(copied_rows)
            ),
            "source_row_hkl_inventory_after_rebinding_filter": (
                _geometry_fit_hkl_inventory_from_entries(filtered_rows)
            ),
            "source_row_hkl_branch_inventory_after_rebinding_filter": (
                _geometry_fit_hkl_branch_inventory_from_entries(filtered_rows)
            ),
        }
        return filtered_rows, diagnostics_local

    def _finalize_diagnostics(
        diagnostics_local: Mapping[str, object] | None,
    ) -> dict[str, object]:
        finalized = dict(diagnostics_local or {})
        finalized.update(copy.deepcopy(targeted_runtime_flags))
        finalized["targeted_performance_gate"] = _targeted_gate_payload()
        if stage_callback_failure_count > 0:
            finalized["stage_callback_failure_count"] = int(stage_callback_failure_count)
            finalized["stage_callback_last_failed_stage"] = str(
                stage_callback_last_failed_stage or ""
            )
        return finalized

    def _copy_rows(raw_rows: Sequence[object] | None) -> list[dict[str, object]]:
        copied_rows: list[dict[str, object]] = []
        for raw_entry in raw_rows or ():
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            entry.setdefault("background_index", int(background_idx))
            copied_rows.append(entry)
        return copied_rows

    def _resolve_live_rows_payload(
        raw_payload: object,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        rows_source = raw_payload
        cache_metadata_local: dict[str, object] = {}
        if isinstance(raw_payload, Mapping) and "rows" in raw_payload:
            rows_source = raw_payload.get("rows")
            raw_cache_metadata = raw_payload.get("cache_metadata")
            if isinstance(raw_cache_metadata, Mapping):
                cache_metadata_local = copy.deepcopy(dict(raw_cache_metadata))
        if isinstance(rows_source, Mapping):
            rows_source = ()
        return _copy_rows(rows_source), cache_metadata_local

    def _copy_optional_sequence(
        raw_values: Sequence[object] | None,
    ) -> list[object] | None:
        if raw_values is None:
            return None
        return [copy.deepcopy(entry) for entry in raw_values]

    def _resolve_live_cache_inventory() -> dict[str, object]:
        if callable(live_cache_inventory):
            try:
                resolved = live_cache_inventory()
            except Exception:
                resolved = {}
        else:
            resolved = live_cache_inventory or {}
        copied = copy.deepcopy(resolved)
        return copied if isinstance(copied, dict) else {}

    def _safe_int_value(raw_value: object, default: int = 0) -> int:
        try:
            return int(raw_value)
        except Exception:
            return int(default)

    def _source_counts_from_rows(
        rows: Sequence[object] | None,
    ) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in rows or ():
            if not isinstance(row, Mapping):
                continue
            counts[str(_geometry_fit_entry_source_label(row))] += 1
        return dict(sorted(counts.items()))

    def _source_counts_from_metadata(
        metadata_local: Mapping[str, object],
        *keys: str,
    ) -> dict[str, int] | None:
        for key in keys:
            raw_counts = metadata_local.get(key)
            if isinstance(raw_counts, Mapping):
                return {
                    str(label): _safe_int_value(count, 0)
                    for label, count in sorted(raw_counts.items(), key=lambda item: str(item[0]))
                }
        return None

    def _required_pair_source_labels() -> list[str]:
        source_labels = {
            str(_geometry_fit_entry_source_label(entry))
            for entry in (
                collected_required_manual_fit_targets
                if collected_required_manual_fit_targets
                else required_pairs or ()
            )
            if isinstance(entry, Mapping)
        }
        return sorted(source_labels)

    def _live_runtime_cache_empty_reason(
        *,
        stored_hit_rows: int,
        built_rows: int,
        projected_rows: int,
        enabled_filter_rows: int,
        required_pair_source_filter_rows: int,
    ) -> str:
        if stored_hit_rows <= 0:
            return "stored_hit_tables_missing_or_empty"
        if built_rows <= 0:
            return "stored_hit_tables_built_zero_rows"
        if projected_rows <= 0:
            return "projection_dropped_all_rows"
        if enabled_filter_rows <= 0:
            return "enabled_q_group_filter_dropped_all_rows"
        if required_pair_source_filter_rows <= 0:
            return "required_pair_source_filter_dropped_all_rows"
        return "live_runtime_cache_rows_empty"

    def _live_runtime_cache_validation_diagnostics(
        rows: Sequence[object] | None,
        metadata_local: Mapping[str, object] | None,
    ) -> dict[str, object]:
        metadata_dict = dict(metadata_local or {})
        row_count = int(len(rows or ()))
        stored_tables = _safe_int_value(
            metadata_dict.get(
                "stored_max_positions_table_count",
                metadata_dict.get("stored_tables", metadata_dict.get("hit_table_count", 0)),
            ),
            0,
        )
        stored_hit_rows = _safe_int_value(
            metadata_dict.get(
                "stored_max_positions_hit_row_count",
                metadata_dict.get(
                    "stored_hit_rows",
                    metadata_dict.get("max_positions_row_count", row_count),
                ),
            ),
            row_count,
        )
        stored_lattice = _safe_int_value(
            metadata_dict.get(
                "stored_peak_table_lattice_count",
                metadata_dict.get("stored_peak_table_lattice", 0),
            ),
            0,
        )
        stored_source_indices = _safe_int_value(
            metadata_dict.get(
                "stored_source_reflection_indices_count",
                metadata_dict.get("stored_source_reflection_indices", 0),
            ),
            0,
        )
        built_rows = _safe_int_value(
            metadata_dict.get(
                "built_stored_hit_table_peak_count",
                metadata_dict.get("built_rows", metadata_dict.get("simulated_peak_count", row_count)),
            ),
            row_count,
        )
        projected_rows = _safe_int_value(
            metadata_dict.get("projected_row_count", metadata_dict.get("projection_rows", row_count)),
            row_count,
        )
        enabled_filter_rows = _safe_int_value(
            metadata_dict.get(
                "after_enabled_q_group_filter_count",
                metadata_dict.get("enabled_filter_rows", row_count),
            ),
            row_count,
        )
        required_pair_source_filter_rows = _safe_int_value(
            metadata_dict.get(
                "after_required_pair_source_filter_count",
                metadata_dict.get("required_pair_source_filter_rows", row_count),
            ),
            row_count,
        )
        live_rows_raw_count = _safe_int_value(
            metadata_dict.get("live_rows_raw_count", row_count),
            row_count,
        )
        live_rows_payload_count = _safe_int_value(
            metadata_dict.get("live_rows_payload_count", row_count),
            row_count,
        )
        live_rows_signature_match = metadata_dict.get("live_rows_signature_match")
        if live_rows_signature_match is None:
            live_rows_signature_match = True
        else:
            live_rows_signature_match = bool(live_rows_signature_match)
        live_rows_signature_reason = str(
            metadata_dict.get("live_rows_signature_reason", "") or ""
        ).strip()
        live_rows_cache_source = str(metadata_dict.get("live_rows_cache_source", "") or "").strip()
        live_rows_source_counts = _source_counts_from_metadata(
            metadata_dict,
            "live_rows_source_counts",
        )
        if live_rows_source_counts is None:
            live_rows_source_counts = _source_counts_from_rows(rows)
        source_counts_before = _source_counts_from_metadata(
            metadata_dict,
            "source_counts_before_filter",
            "source_counts",
        )
        if source_counts_before is None:
            source_counts_before = _source_counts_from_rows(rows)
        source_counts_after = _source_counts_from_metadata(
            metadata_dict,
            "source_counts_after_filter",
        )
        if source_counts_after is None:
            source_counts_after = _source_counts_from_rows(rows)
        required_sources = _required_pair_source_labels()
        reason = str(
            metadata_dict.get("reason")
            or metadata_dict.get("empty_reason")
            or _live_runtime_cache_empty_reason(
                stored_hit_rows=stored_hit_rows,
                built_rows=built_rows,
                projected_rows=projected_rows,
                enabled_filter_rows=enabled_filter_rows,
                required_pair_source_filter_rows=required_pair_source_filter_rows,
            )
        )
        return {
            "geometry_fit_live_handoff_patch_marker": str(
                metadata_dict.get("geometry_fit_live_handoff_patch_marker", "") or ""
            ),
            "stored_max_positions_table_count": int(stored_tables),
            "stored_max_positions_hit_row_count": int(stored_hit_rows),
            "stored_peak_table_lattice_count": int(stored_lattice),
            "stored_source_reflection_indices_count": int(stored_source_indices),
            "built_stored_hit_table_peak_count": int(built_rows),
            "projected_row_count": int(projected_rows),
            "after_enabled_q_group_filter_count": int(enabled_filter_rows),
            "after_required_pair_source_filter_count": int(required_pair_source_filter_rows),
            "stored_tables": int(stored_tables),
            "stored_hit_rows": int(stored_hit_rows),
            "stored_peak_table_lattice": int(stored_lattice),
            "stored_source_reflection_indices": int(stored_source_indices),
            "built_rows": int(built_rows),
            "projection_rows": int(projected_rows),
            "enabled_filter_rows": int(enabled_filter_rows),
            "required_pair_source_filter_rows": int(required_pair_source_filter_rows),
            "live_rows_raw_count": int(live_rows_raw_count),
            "live_rows_payload_count": int(live_rows_payload_count),
            "live_rows_signature_match": bool(live_rows_signature_match),
            "live_rows_signature_reason": str(live_rows_signature_reason),
            "live_rows_cache_source": str(live_rows_cache_source),
            "live_rows_source_counts": dict(live_rows_source_counts),
            "q_group_cached_entries": _safe_int_value(
                metadata_dict.get("q_group_cached_entries", 0),
                0,
            ),
            "manual_picker_candidates": _safe_int_value(
                metadata_dict.get("manual_picker_candidates", 0),
                0,
            ),
            "live_preview_rows_count": _safe_int_value(
                metadata_dict.get("live_preview_rows_count", row_count),
                row_count,
            ),
            "live_rows_by_background_current_count": _safe_int_value(
                metadata_dict.get("live_rows_by_background_current_count", row_count),
                row_count,
            ),
            "live_rows_by_background_keys": copy.deepcopy(
                metadata_dict.get("live_rows_by_background_keys", [])
            ),
            "requested_signature_keys": copy.deepcopy(
                metadata_dict.get("requested_signature_keys", [])
            ),
            "requested_signature_by_background_keys": copy.deepcopy(
                metadata_dict.get("requested_signature_by_background_keys", [])
            ),
            "live_rows_signature_by_background_keys": copy.deepcopy(
                metadata_dict.get("live_rows_signature_by_background_keys", [])
            ),
            "required_pair_count": int(len(required_pairs or ())),
            "required_pair_sources": list(required_sources),
            "source_counts_before_filter": dict(source_counts_before),
            "source_counts_after_filter": dict(source_counts_after),
            "reason": str(reason),
        }

    def _resolve_runtime_simulation_diagnostics() -> dict[str, object]:
        if not callable(last_runtime_simulation_diagnostics):
            return {}
        try:
            resolved = last_runtime_simulation_diagnostics()
        except Exception:
            resolved = {}
        copied = copy.deepcopy(resolved)
        return copied if isinstance(copied, dict) else {}

    def _merge_runtime_simulation_diagnostics(
        diagnostics_local: Mapping[str, object] | None,
        runtime_diag_local: Mapping[str, object] | None,
    ) -> dict[str, object]:
        merged = dict(diagnostics_local or {})
        runtime_diag = copy.deepcopy(runtime_diag_local)
        if not isinstance(runtime_diag, dict) or not runtime_diag:
            return merged
        merged["runtime_simulation"] = runtime_diag
        for key in (
            "miller_shape",
            "intensity_shape",
            "image_size",
            "missing_param_keys",
            "missing_mosaic_keys",
            "param_summary",
            "mosaic_array_sizes",
        ):
            if key in runtime_diag and key not in merged:
                merged[key] = copy.deepcopy(runtime_diag.get(key))
        return merged

    def _project_copied_rows(raw_rows: Sequence[object] | None) -> list[dict[str, object]]:
        nonlocal projected_rows_failure_reason
        row_count = int(len(raw_rows or ()))
        project_started_at = perf_counter()
        _emit_rebuild_stage(
            "source_cache_project_rows_start",
            cache_source="projection",
            row_count=int(row_count),
        )
        projected_rows: Sequence[object] | None = raw_rows
        project_status = "projected"
        projection_failure_reason: str | None = None
        projected_rows_failure_reason = None
        strict_projection_mode = normalized_projection_view_mode in {"caked", "q_space"}
        if strict_projection_mode:
            projected_rows = []
            signature_available = _geometry_fit_projection_signature_available(
                normalized_projection_view_signature
            )
            signature_background_index = (
                normalized_projection_view_signature.get("background_index")
                if isinstance(normalized_projection_view_signature, Mapping)
                else None
            )
            current_signature_background_index = (
                normalized_projection_view_signature.get("current_background_index")
                if isinstance(normalized_projection_view_signature, Mapping)
                else None
            )
            requires_owned_payload = (
                normalized_projection_view_mode == "caked"
                and signature_background_index is not None
                and current_signature_background_index is not None
                and int(signature_background_index) != int(current_signature_background_index)
            )
            if not signature_available:
                projection_failure_reason = "unavailable_projection_signature"
            elif signature_background_index is not None and int(signature_background_index) != int(
                background_idx
            ):
                projection_failure_reason = "projection_signature_background_mismatch"
            elif requires_owned_payload and not isinstance(
                normalized_projection_payload,
                Mapping,
            ):
                projection_failure_reason = "missing_background_caked_payload"
            else:
                projector = project_rows_for_background_view
                if not callable(projector):
                    projection_failure_reason = "projection_error"
                else:
                    try:
                        projected_rows = projector(raw_rows)
                    except Exception as exc:
                        projected_rows = []
                        projection_failure_reason = f"projection_error:{type(exc).__name__}"
            if projection_failure_reason is not None:
                project_status = str(projection_failure_reason)
        elif callable(project_rows):
            try:
                projected_rows = project_rows(raw_rows)
            except Exception as exc:
                projected_rows = raw_rows
                project_status = f"project_rows_exception:{type(exc).__name__}"
        copied_projected_rows = _copy_rows(projected_rows)
        projected_rows_failure_reason = (
            str(projection_failure_reason) if projection_failure_reason is not None else None
        )
        if targeted_preflight_enabled:
            _update_targeted_runtime_flags(
                source_rows_projected_for_rebinding=int(len(copied_projected_rows)),
                unrelated_projected_row_count_for_rebinding=0,
                full_source_rows_projected_for_rebinding=False,
            )
        _emit_rebuild_stage(
            "source_cache_project_rows_ready",
            stage_started_at=project_started_at,
            cache_source="projection",
            row_count=int(len(copied_projected_rows)),
            status=project_status,
        )
        return copied_projected_rows

    def _build_source_rows_result(
        hit_tables_local: Sequence[object],
        *,
        cache_source: str,
    ) -> tuple[
        Sequence[object],
        Sequence[object] | None,
        Sequence[object] | None,
        Sequence[int] | None,
    ]:
        build_kwargs: dict[str, object] = {}
        build_kwargs["consumer"] = lookup_context
        if targeted_preflight_enabled:
            build_kwargs.update(
                {
                    "required_branch_group_keys": list(collected_required_branch_group_keys),
                    "required_manual_fit_targets": copy.deepcopy(
                        collected_required_manual_fit_targets
                    ),
                    "preflight_mode": normalized_preflight_mode,
                }
            )
        build_used_targeted_filter = False
        build_started_at = perf_counter()
        _emit_rebuild_stage(
            (
                "source_cache_targeted_source_rows_start"
                if targeted_preflight_enabled
                else "source_cache_build_source_rows_start"
            ),
            cache_source=str(cache_source),
            hit_table_count=int(len(hit_tables_local or ())),
            preflight_mode=normalized_preflight_mode,
        )
        _emit_rebuild_stage(
            "source_cache_fresh_rebuild_consumer_wrapper",
            cache_source=str(cache_source),
            fresh_rebuild_consumer_wrapper="deduped",
            preflight_mode=normalized_preflight_mode,
        )
        raw_result, accepted_build_keywords = _call_with_optional_keywords(
            build_source_rows_from_hit_tables,
            hit_tables_local,
            **build_kwargs,
        )
        try:
            build_signature = str(inspect.signature(build_source_rows_from_hit_tables))
        except Exception:
            build_signature = "<unavailable>"
        _update_targeted_runtime_flags(
            build_source_rows_accepted_keywords=list(accepted_build_keywords),
            build_source_rows_requested_keywords=list(build_kwargs),
            build_source_rows_consumer_kwarg_passed=("consumer" in accepted_build_keywords),
            build_source_rows_callback_signature=build_signature,
        )
        trial_supplemental_diag = getattr(
            build_source_rows_from_hit_tables,
            "last_trial_supplemental_diag",
            None,
        )
        if isinstance(trial_supplemental_diag, Mapping):
            _update_targeted_runtime_flags(
                trial_source_row_supplemental_diagnostics=copy.deepcopy(
                    dict(trial_supplemental_diag)
                )
            )
        build_used_targeted_filter = "required_branch_group_keys" in accepted_build_keywords
        if targeted_preflight_enabled:
            source_rows_built_from_targeted_hit_tables = bool(
                str(cache_source) == "fresh_simulation"
                and targeted_simulation_used
                and not targeted_simulation_fallback_reason
            )
            _update_targeted_runtime_flags(
                total_hit_tables_available=int(len(hit_tables_local or ())),
                hit_tables_considered_for_rebinding=int(len(hit_tables_local or ())),
                hit_tables_expanded_for_rebinding=int(len(hit_tables_local or ())),
                full_source_rows_built_for_rebinding=not bool(
                    build_used_targeted_filter or source_rows_built_from_targeted_hit_tables
                ),
            )
        rows_for_count: Sequence[object] | None = raw_result
        if isinstance(raw_result, tuple):
            if len(raw_result) == 4:
                rows_for_count = raw_result[0]
                _emit_rebuild_stage(
                    (
                        "source_cache_targeted_source_rows_ready"
                        if targeted_preflight_enabled
                        else "source_cache_build_source_rows_ready"
                    ),
                    stage_started_at=build_started_at,
                    cache_source=str(cache_source),
                    hit_table_count=int(len(hit_tables_local or ())),
                    row_count=int(len(rows_for_count or ())),
                    preflight_mode=normalized_preflight_mode,
                )
                return raw_result
            if len(raw_result) == 3:
                rows, peak_table_lattice, returned_hit_tables = raw_result
                _emit_rebuild_stage(
                    (
                        "source_cache_targeted_source_rows_ready"
                        if targeted_preflight_enabled
                        else "source_cache_build_source_rows_ready"
                    ),
                    stage_started_at=build_started_at,
                    cache_source=str(cache_source),
                    hit_table_count=int(len(hit_tables_local or ())),
                    row_count=int(len(rows or ())),
                    preflight_mode=normalized_preflight_mode,
                )
                return rows, peak_table_lattice, returned_hit_tables, None
        _emit_rebuild_stage(
            (
                "source_cache_targeted_source_rows_ready"
                if targeted_preflight_enabled
                else "source_cache_build_source_rows_ready"
            ),
            stage_started_at=build_started_at,
            cache_source=str(cache_source),
            hit_table_count=int(len(hit_tables_local or ())),
            row_count=int(len(rows_for_count or ())),
            preflight_mode=normalized_preflight_mode,
        )
        return raw_result, None, None, None

    _emit_target_collection_events()

    def _success_result(
        raw_rows: Sequence[object] | None,
        *,
        rebuild_source: str,
        peak_table_lattice: Sequence[object] | None = None,
        hit_tables_local: Sequence[object] | None = None,
        source_reflection_indices: Sequence[int] | None = None,
        intersection_cache_local: Sequence[object] | None = None,
        metadata: Mapping[str, object] | None = None,
        runtime_simulation_diagnostics: Mapping[str, object] | None = None,
    ) -> GeometryFitSourceRowRebuildResult:
        stored_rows, row_filter_diagnostics = _filter_rows_for_rebinding(raw_rows)
        _update_targeted_runtime_flags(
            **{
                key: value
                for key, value in row_filter_diagnostics.items()
                if key
                in {
                    "total_source_rows_available",
                    "source_rows_considered_for_rebinding",
                    "candidate_rows_after_hkl_filter",
                    "candidate_rows_after_branch_filter",
                    "candidate_rows_scored_for_background_distance",
                    "unrelated_available_row_count_for_rebinding",
                    "unrelated_scored_row_count_for_rebinding",
                    "unrelated_projected_row_count_for_rebinding",
                    "required_branch_group_keys_seen",
                    "missing_required_branch_group_keys",
                    "missing_required_hkl_inventory",
                    "source_row_hkl_inventory_before_rebinding_filter",
                    "source_row_hkl_branch_inventory_before_rebinding_filter",
                    "source_row_hkl_inventory_after_rebinding_filter",
                    "source_row_hkl_branch_inventory_after_rebinding_filter",
                }
            }
        )
        projected_rows = _project_copied_rows(stored_rows)
        cache_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        cache_metadata.setdefault(
            "required_branch_group_keys_digest",
            str(required_branch_group_keys_digest),
        )
        cache_metadata.setdefault(
            "manual_target_scoring_digest",
            str(manual_target_scoring_digest),
        )
        cache_metadata.setdefault(
            "preflight_mode",
            str(normalized_preflight_mode),
        )
        if targeted_preflight_enabled:
            cache_metadata.setdefault(
                "required_manual_fit_targets",
                copy.deepcopy(collected_required_manual_fit_targets),
            )
            cache_metadata.setdefault(
                "required_branch_group_keys",
                copy.deepcopy(list(collected_required_branch_group_keys)),
            )
        if live_runtime_cache_metadata:
            cache_metadata.setdefault(
                "live_runtime_cache_metadata",
                copy.deepcopy(dict(live_runtime_cache_metadata)),
            )
        if isinstance(live_cache_validation, Mapping) and live_cache_validation:
            cache_metadata.setdefault(
                "live_runtime_cache_validation",
                copy.deepcopy(dict(live_cache_validation)),
            )
            if str(rebuild_source) != "live_runtime_cache":
                rebuilt_validation = validate_geometry_fit_live_source_rows(
                    stored_rows,
                    required_pairs=required_pairs,
                )
                dual_path_diff = _geometry_fit_required_pair_dual_path_diff(
                    live_cache_validation,
                    rebuilt_validation,
                )
                cache_metadata.setdefault(
                    "live_runtime_cache_rebuild_validation",
                    copy.deepcopy(dict(rebuilt_validation)),
                )
                if dual_path_diff:
                    cache_metadata.setdefault(
                        "live_runtime_cache_dual_path_diff",
                        copy.deepcopy(list(dual_path_diff)),
                    )
        if projected_rows_failure_reason is not None:
            cache_metadata.setdefault(
                "projection_failure_reason",
                str(projected_rows_failure_reason),
            )
        if (
            targeted_preflight_enabled
            and callable(store_targeted_projected_cache)
            and _geometry_fit_projected_rows_cacheable(
                background_index=int(background_idx),
                projected_rows=projected_rows,
                projection_view_signature=normalized_projection_view_signature,
                requested_projection_view_signature=normalized_projection_view_signature,
                consumer=lookup_context,
            )
        ):
            try:
                store_targeted_projected_cache(
                    str(required_branch_group_keys_digest),
                    {
                        "background_index": int(background_idx),
                        "requested_signature": normalized_requested_signature,
                        "requested_signature_summary": (normalized_requested_signature_summary),
                        "required_branch_group_keys_digest": str(required_branch_group_keys_digest),
                        "manual_target_scoring_digest": str(manual_target_scoring_digest),
                        "preflight_mode": str(normalized_preflight_mode),
                        "consumer": str(lookup_context),
                        "projection_view_mode": normalized_projection_view_mode,
                        "projection_view_signature": (normalized_projection_view_signature),
                        "stored_rows": _copy_rows(stored_rows),
                        "projected_rows": _copy_rows(projected_rows),
                        "cache_source": str(rebuild_source),
                        "diagnostics": copy.deepcopy(dict(targeted_runtime_flags)),
                    },
                )
            except Exception:
                pass
        diagnostics = _merge_runtime_simulation_diagnostics(
            {
                "source": "source_snapshot",
                "cache_family": "source_snapshot",
                "action": "rebuild",
                "consumer": lookup_context,
                "status": "snapshot_hit",
                "background_index": int(background_idx),
                "background_label": resolved_background_label,
                "requested_signature": requested_signature,
                "requested_signature_summary": requested_signature_summary,
                "snapshot_signature": requested_signature,
                "stored_signature_summary": requested_signature_summary,
                "raw_peak_count": int(len(stored_rows)),
                "projected_peak_count": int(len(projected_rows)),
                "created_from": str(rebuild_source),
                "cache_source": "source_snapshot_rebuild",
                "rebuild_source": str(rebuild_source),
                "signature_match": True,
                "rebuild_attempts": list(rebuild_attempts),
                "cache_metadata": cache_metadata,
                "projection_failure_reason": (
                    cache_metadata.get("projection_failure_reason")
                    if isinstance(cache_metadata, Mapping)
                    else None
                ),
                "live_cache_inventory": _resolve_live_cache_inventory(),
            },
            runtime_simulation_diagnostics,
        )
        diagnostics = _finalize_diagnostics(diagnostics)
        return GeometryFitSourceRowRebuildResult(
            background_index=int(background_idx),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            projected_rows=projected_rows,
            stored_rows=stored_rows,
            rebuild_source=str(rebuild_source),
            rebuild_attempts=list(rebuild_attempts),
            diagnostics=diagnostics,
            peak_table_lattice=_copy_optional_sequence(peak_table_lattice),
            hit_tables=_copy_optional_sequence(hit_tables_local),
            source_reflection_indices=(
                list(source_reflection_indices)
                if isinstance(source_reflection_indices, Sequence)
                and not isinstance(source_reflection_indices, (str, bytes))
                else None
            ),
            intersection_cache=_copy_optional_sequence(intersection_cache_local),
            metadata=cache_metadata,
        )

    if targeted_preflight_enabled and not force_fresh_simulation:
        rebuild_attempts.append("targeted_projected_cache")
        targeted_cache_started_at = perf_counter()
        _emit_rebuild_stage(
            "source_cache_targeted_projected_cache_start",
            cache_source="targeted_projected_cache",
            preflight_mode=normalized_preflight_mode,
            required_hkl_branch_keys_digest=str(required_branch_group_keys_digest),
        )
        targeted_cache_payload: Mapping[str, object] | None = None
        if callable(get_targeted_projected_cache):
            try:
                targeted_cache_payload = get_targeted_projected_cache(
                    str(required_branch_group_keys_digest)
                )
            except Exception:
                targeted_cache_payload = None
        targeted_cache_rows: list[dict[str, object]] = []
        if isinstance(targeted_cache_payload, Mapping):
            requested_signature_matches = (
                targeted_cache_payload.get("requested_signature") == normalized_requested_signature
            )
            projection_signature_matches = _geometry_fit_stable_projection_view_signature(
                targeted_cache_payload.get("projection_view_signature")
            ) == _geometry_fit_stable_projection_view_signature(
                normalized_projection_view_signature
            )
            projected_payload = targeted_cache_payload.get("projected_rows")
            targeted_cache_rows = _copy_rows(projected_payload)
            if (
                requested_signature_matches
                and projection_signature_matches
                and _geometry_fit_projected_rows_cacheable(
                    background_index=int(background_idx),
                    projected_rows=targeted_cache_rows,
                    projection_view_signature=targeted_cache_payload.get(
                        "projection_view_signature"
                    ),
                    requested_projection_view_signature=normalized_projection_view_signature,
                    consumer=str(targeted_cache_payload.get("consumer") or ""),
                )
            ):
                _update_targeted_runtime_flags(
                    targeted_cache_hit=True,
                    cache_source="targeted_projected_cache",
                    total_source_rows_available=int(len(targeted_cache_rows)),
                    source_rows_considered_for_rebinding=int(len(targeted_cache_rows)),
                    candidate_rows_after_hkl_filter=int(len(targeted_cache_rows)),
                    candidate_rows_after_branch_filter=int(len(targeted_cache_rows)),
                    candidate_rows_scored_for_background_distance=int(len(targeted_cache_rows)),
                    unrelated_scored_row_count_for_rebinding=0,
                    source_rows_projected_for_rebinding=0,
                    unrelated_projected_row_count_for_rebinding=0,
                    full_source_rows_built_for_rebinding=False,
                    full_source_rows_projected_for_rebinding=False,
                )
                _emit_rebuild_stage(
                    "source_cache_targeted_projected_cache_ready",
                    stage_started_at=targeted_cache_started_at,
                    cache_source="targeted_projected_cache",
                    preflight_mode=normalized_preflight_mode,
                    targeted_cache_hit=True,
                    row_count=int(len(targeted_cache_rows)),
                )
                diagnostics = _finalize_diagnostics(
                    {
                        "source": "source_snapshot",
                        "cache_family": "source_snapshot",
                        "action": "rebuild",
                        "consumer": lookup_context,
                        "status": "snapshot_hit",
                        "background_index": int(background_idx),
                        "background_label": resolved_background_label,
                        "requested_signature": requested_signature,
                        "requested_signature_summary": requested_signature_summary,
                        "snapshot_signature": requested_signature,
                        "stored_signature_summary": requested_signature_summary,
                        "raw_peak_count": int(len(targeted_cache_rows)),
                        "projected_peak_count": int(len(targeted_cache_rows)),
                        "created_from": "targeted_projected_cache",
                        "cache_source": "targeted_projected_cache",
                        "rebuild_source": "targeted_projected_cache",
                        "signature_match": True,
                        "rebuild_attempts": list(rebuild_attempts),
                        "cache_metadata": {
                            "required_branch_group_keys_digest": str(
                                required_branch_group_keys_digest
                            ),
                            "manual_target_scoring_digest": str(manual_target_scoring_digest),
                            "preflight_mode": str(normalized_preflight_mode),
                        },
                        "live_cache_inventory": _resolve_live_cache_inventory(),
                    }
                )
                return GeometryFitSourceRowRebuildResult(
                    background_index=int(background_idx),
                    requested_signature=requested_signature,
                    requested_signature_summary=requested_signature_summary,
                    projected_rows=_copy_rows(targeted_cache_rows),
                    stored_rows=_copy_rows(
                        targeted_cache_payload.get("stored_rows") or targeted_cache_rows
                    ),
                    rebuild_source="targeted_projected_cache",
                    rebuild_attempts=list(rebuild_attempts),
                    diagnostics=diagnostics,
                    metadata={
                        "required_branch_group_keys_digest": str(required_branch_group_keys_digest),
                        "manual_target_scoring_digest": str(manual_target_scoring_digest),
                        "preflight_mode": str(normalized_preflight_mode),
                    },
                )
        _emit_rebuild_stage(
            "source_cache_targeted_projected_cache_miss",
            stage_started_at=targeted_cache_started_at,
            cache_source="targeted_projected_cache",
            preflight_mode=normalized_preflight_mode,
            targeted_cache_hit=False,
            row_count=int(len(targeted_cache_rows)),
        )

    if can_use_live_runtime_cache and callable(build_live_rows) and not force_fresh_simulation:
        rebuild_attempts.append("live_runtime_cache")
        validation_started_at = perf_counter()
        _emit_rebuild_stage(
            "source_cache_live_runtime_cache_validation_start",
            cache_source="live_runtime_cache",
        )
        live_rows, live_runtime_cache_metadata = _resolve_live_rows_payload(build_live_rows())
        live_runtime_cache_diag = _live_runtime_cache_validation_diagnostics(
            live_rows,
            live_runtime_cache_metadata,
        )
        if live_rows:
            live_cache_validation = validate_geometry_fit_live_source_rows(
                live_rows,
                required_pairs=required_pairs,
            )
            _emit_rebuild_stage(
                "source_cache_live_runtime_cache_validation_ready",
                stage_started_at=validation_started_at,
                cache_source="live_runtime_cache",
                status=str(
                    live_cache_validation.get("status")
                    or ("valid" if live_cache_validation.get("valid") else "invalid")
                ),
                row_count=int(len(live_rows)),
                validated_pair_count=int(live_cache_validation.get("validated_pair_count", 0) or 0),
                missing_required_pair_count=int(
                    live_cache_validation.get("missing_required_pair_count", 0) or 0
                ),
                branch_mismatch_count=int(
                    live_cache_validation.get("branch_mismatch_count", 0) or 0
                ),
                hkl_missing_candidate_count=int(
                    live_cache_validation.get("hkl_missing_candidate_count", 0) or 0
                ),
                **live_runtime_cache_diag,
            )
            if bool(live_cache_validation.get("valid", False)):
                if targeted_preflight_enabled:
                    _update_targeted_runtime_flags(
                        targeted_cache_hit=True,
                        cache_source="live_runtime_cache",
                        full_source_rows_built_for_rebinding=False,
                    )
                _emit_rebuild_stage(
                    "source_cache_live_runtime_cache_accepted",
                    cache_source="live_runtime_cache",
                    status="accepted",
                    row_count=int(len(live_rows)),
                    required_pair_count=int(
                        live_cache_validation.get("required_pair_count", 0) or 0
                    ),
                    validated_pair_count=int(
                        live_cache_validation.get("validated_pair_count", 0) or 0
                    ),
                )
                return _success_result(
                    live_rows,
                    rebuild_source="live_runtime_cache",
                )
            _emit_rebuild_stage(
                "source_cache_live_runtime_cache_rejected",
                cache_source="live_runtime_cache",
                status=str(live_cache_validation.get("status") or "validation_failed"),
                row_count=int(len(live_rows)),
                required_pair_count=int(live_cache_validation.get("required_pair_count", 0) or 0),
                validated_pair_count=int(live_cache_validation.get("validated_pair_count", 0) or 0),
                missing_required_pair_count=int(
                    live_cache_validation.get("missing_required_pair_count", 0) or 0
                ),
                branch_mismatch_count=int(
                    live_cache_validation.get("branch_mismatch_count", 0) or 0
                ),
                hkl_missing_candidate_count=int(
                    live_cache_validation.get("hkl_missing_candidate_count", 0) or 0
                ),
            )
            rebuild_attempts.append("live_runtime_cache_validation_failed")
        else:
            _emit_rebuild_stage(
                "source_cache_live_runtime_cache_validation_ready",
                stage_started_at=validation_started_at,
                cache_source="live_runtime_cache",
                status="empty_live_runtime_cache",
                row_count=0,
                validated_pair_count=0,
                missing_required_pair_count=0,
                branch_mismatch_count=0,
                hkl_missing_candidate_count=0,
                **live_runtime_cache_diag,
            )
            _emit_rebuild_stage(
                "source_cache_live_runtime_cache_rejected",
                cache_source="live_runtime_cache",
                status="empty_live_runtime_cache",
                row_count=0,
                validated_pair_count=0,
                missing_required_pair_count=0,
                branch_mismatch_count=0,
                hkl_missing_candidate_count=0,
                **live_runtime_cache_diag,
            )

    if force_fresh_simulation:
        rebuild_attempts.append("fresh_simulation_required_for_trial_source_rows")
    elif memory_cache_signature is not None and memory_cache_signature != requested_signature:
        rebuild_attempts.append("last_intersection_cache_memory_signature_mismatch")
        _emit_rebuild_stage(
            "source_cache_memory_intersection_cache_start",
            cache_source="last_intersection_cache_memory",
        )
        _emit_rebuild_stage(
            "source_cache_memory_intersection_cache_miss",
            cache_source="last_intersection_cache_memory",
            status="signature_mismatch",
            row_count=0,
        )
    elif not force_fresh_simulation:
        rebuild_attempts.append("last_intersection_cache_memory")
        memory_cache_started_at = perf_counter()
        _emit_rebuild_stage(
            "source_cache_memory_intersection_cache_start",
            cache_source="last_intersection_cache_memory",
        )
        memory_intersection_cache = (
            list(get_memory_intersection_cache() or ())
            if callable(get_memory_intersection_cache)
            else []
        )
        memory_rows: list[dict[str, object]] = []
        memory_lattice: Sequence[object] | None = None
        memory_hit_tables: Sequence[object] | None = None
        memory_source_reflection_indices: Sequence[int] | None = None
        if memory_intersection_cache:
            memory_rows, memory_lattice, memory_hit_tables, memory_source_reflection_indices = (
                _build_source_rows_result(
                    memory_intersection_cache,
                    cache_source="last_intersection_cache_memory",
                )
            )
            _emit_rebuild_stage(
                "source_cache_memory_intersection_cache_ready",
                stage_started_at=memory_cache_started_at,
                cache_source="last_intersection_cache_memory",
                row_count=int(len(memory_rows)),
                hit_table_count=int(len(memory_intersection_cache)),
                status="ready",
            )
        else:
            _emit_rebuild_stage(
                "source_cache_memory_intersection_cache_miss",
                stage_started_at=memory_cache_started_at,
                cache_source="last_intersection_cache_memory",
                row_count=0,
                hit_table_count=0,
                status="empty_cache",
            )
        if memory_rows:
            if targeted_preflight_enabled:
                _update_targeted_runtime_flags(
                    targeted_cache_hit=True,
                    cache_source="last_intersection_cache_memory",
                )
            return _success_result(
                memory_rows,
                rebuild_source="last_intersection_cache_memory",
                peak_table_lattice=memory_lattice,
                hit_tables_local=memory_hit_tables,
                source_reflection_indices=memory_source_reflection_indices,
                intersection_cache_local=memory_intersection_cache,
            )

    if not force_fresh_simulation:
        rebuild_attempts.append("last_intersection_cache_log")
        logged_cache_started_at = perf_counter()
        _emit_rebuild_stage(
            "source_cache_logged_intersection_cache_start",
            cache_source="last_intersection_cache_log",
        )
        logged_metadata: Mapping[str, object] | None = None
        metadata_only_logged_lookup = callable(load_logged_intersection_cache_metadata)
        if callable(load_logged_intersection_cache_metadata):
            try:
                logged_metadata = load_logged_intersection_cache_metadata()
            except Exception:
                logged_metadata = None
        logged_match_payload = _resolve_logged_cache_match_payload(logged_metadata)
        if not bool(logged_match_payload.get("matches", False)):
            _emit_rebuild_stage(
                "source_cache_logged_intersection_cache_miss",
                stage_started_at=logged_cache_started_at,
                cache_source="last_intersection_cache_log",
                row_count=0,
                hit_table_count=0,
                status=str(
                    logged_match_payload.get("mismatch_reason")
                    or logged_match_payload.get("status")
                    or (
                        "params_mismatch" if isinstance(logged_metadata, Mapping) else "empty_cache"
                    )
                ),
                expected_signature_digest=logged_match_payload.get("expected_signature_digest"),
                actual_signature_digest=logged_match_payload.get("actual_signature_digest"),
                mismatch_reason=logged_match_payload.get("mismatch_reason"),
                heavy_hit_table_load_attempted=bool(
                    logged_match_payload.get("heavy_hit_table_load_attempted", False)
                ),
            )
        else:
            logged_intersection_cache: Sequence[object] = []
            if callable(load_logged_intersection_cache):
                try:
                    logged_intersection_cache, logged_metadata = load_logged_intersection_cache()
                    logged_match_payload["heavy_hit_table_load_attempted"] = True
                except Exception:
                    logged_intersection_cache, logged_metadata = [], logged_metadata
            if logged_intersection_cache:
                logged_rows, logged_lattice, logged_hit_tables, logged_source_reflection_indices = (
                    _build_source_rows_result(
                        logged_intersection_cache,
                        cache_source="last_intersection_cache_log",
                    )
                )
                _emit_rebuild_stage(
                    "source_cache_logged_intersection_cache_ready",
                    stage_started_at=logged_cache_started_at,
                    cache_source="last_intersection_cache_log",
                    row_count=int(len(logged_rows)),
                    hit_table_count=int(len(logged_intersection_cache)),
                    status="ready",
                )
                if logged_rows:
                    if targeted_preflight_enabled:
                        _update_targeted_runtime_flags(
                            targeted_cache_hit=True,
                            cache_source="last_intersection_cache_log",
                        )
                    return _success_result(
                        logged_rows,
                        rebuild_source="last_intersection_cache_log",
                        peak_table_lattice=logged_lattice,
                        hit_tables_local=logged_hit_tables,
                        source_reflection_indices=logged_source_reflection_indices,
                        intersection_cache_local=logged_intersection_cache,
                        metadata={
                            **(
                                dict(logged_metadata)
                                if isinstance(logged_metadata, Mapping)
                                else {}
                            ),
                            "expected_signature_digest": logged_match_payload.get(
                                "expected_signature_digest"
                            ),
                            "actual_signature_digest": logged_match_payload.get(
                                "actual_signature_digest"
                            ),
                        },
                    )
            _emit_rebuild_stage(
                "source_cache_logged_intersection_cache_miss",
                stage_started_at=logged_cache_started_at,
                cache_source="last_intersection_cache_log",
                row_count=0,
                hit_table_count=0,
                status="empty_cache",
                heavy_hit_table_load_attempted=bool(
                    logged_match_payload.get("heavy_hit_table_load_attempted", False)
                ),
            )
        if (
            not metadata_only_logged_lookup
            and callable(load_logged_intersection_cache)
            and not callable(load_logged_intersection_cache_metadata)
        ):
            logged_intersection_cache, logged_metadata = [], None
            try:
                logged_intersection_cache, logged_metadata = load_logged_intersection_cache()
            except Exception:
                logged_intersection_cache, logged_metadata = [], None
            if (
                logged_intersection_cache
                and callable(logged_cache_matches_params)
                and bool(_resolve_logged_cache_match_payload(logged_metadata).get("matches", False))
            ):
                logged_rows, logged_lattice, logged_hit_tables, logged_source_reflection_indices = (
                    _build_source_rows_result(
                        logged_intersection_cache,
                        cache_source="last_intersection_cache_log",
                    )
                )
                if logged_rows:
                    if targeted_preflight_enabled:
                        _update_targeted_runtime_flags(
                            targeted_cache_hit=True,
                            cache_source="last_intersection_cache_log",
                        )
                    return _success_result(
                        logged_rows,
                        rebuild_source="last_intersection_cache_log",
                        peak_table_lattice=logged_lattice,
                        hit_tables_local=logged_hit_tables,
                        source_reflection_indices=logged_source_reflection_indices,
                        intersection_cache_local=logged_intersection_cache,
                        metadata=logged_metadata,
                    )

    rebuild_attempts.append("fresh_simulation")
    fresh_started_at = perf_counter()
    fresh_stage_name = (
        "source_cache_targeted_fresh_simulation_start"
        if targeted_preflight_enabled
        else "source_cache_fresh_simulation_start"
    )
    _emit_rebuild_stage(
        fresh_stage_name,
        cache_source="fresh_simulation",
        preflight_mode=normalized_preflight_mode,
    )
    fresh_hit_tables: Sequence[object] = []
    fresh_simulation_exception: Exception | None = None
    targeted_simulation_used = False
    targeted_simulation_supported = False
    targeted_simulation_fallback_reason: str | None = None
    fresh_required_source_missing: list[str] = []
    if callable(simulate_hit_tables):
        try:
            if targeted_preflight_enabled and collected_required_branch_group_keys:
                targeted_simulation_kwargs = {
                    "required_branch_group_keys": list(collected_required_branch_group_keys),
                    "required_manual_fit_targets": copy.deepcopy(
                        collected_required_manual_fit_targets
                    ),
                    "preflight_mode": normalized_preflight_mode,
                }
                accepted_simulation_kwargs, _signature_known = _accepted_optional_keywords(
                    simulate_hit_tables,
                    targeted_simulation_kwargs,
                )
                targeted_keyword_support = (
                    "required_branch_group_keys" in accepted_simulation_kwargs
                )
                full_fallback_already_loaded = False
                if targeted_keyword_support:
                    try:
                        fresh_hit_tables = simulate_hit_tables(
                            normalized_params,
                            **accepted_simulation_kwargs,
                        )
                    except TypeError as exc:
                        error_text = str(exc)
                        if (
                            "unexpected keyword" in error_text
                            or "positional argument" in error_text
                            or "got an unexpected keyword" in error_text
                        ):
                            fresh_hit_tables = simulate_hit_tables(normalized_params)
                            full_fallback_already_loaded = True
                            targeted_keyword_support = False
                        else:
                            raise
                    runtime_simulation_diagnostics = _resolve_runtime_simulation_diagnostics()
                    targeted_simulation_supported = bool(
                        runtime_simulation_diagnostics.get(
                            "targeted_simulation_supported",
                            targeted_keyword_support,
                        )
                    )
                    targeted_simulation_used = bool(
                        runtime_simulation_diagnostics.get(
                            "targeted_simulation_used",
                            targeted_simulation_supported,
                        )
                    )
                    targeted_simulation_fallback_reason = (
                        str(
                            runtime_simulation_diagnostics.get(
                                "targeted_simulation_fallback_reason"
                            )
                            or ""
                        ).strip()
                        or None
                    )
                if (
                    targeted_simulation_supported
                    and not targeted_simulation_used
                    and targeted_simulation_fallback_reason is None
                ):
                    targeted_simulation_fallback_reason = "targeted_filter_not_applied"
                if not targeted_simulation_supported:
                    _update_targeted_runtime_flags(
                        targeted_simulation_supported=False,
                        targeted_simulation_used=False,
                        targeted_simulation_fallback_reason="simulator_filter_not_supported",
                        full_fresh_simulation_fallback_used=True,
                    )
                    fallback_started_at = perf_counter()
                    _emit_rebuild_stage(
                        "source_cache_targeted_fresh_simulation_unsupported",
                        cache_source="fresh_simulation",
                        preflight_mode=normalized_preflight_mode,
                        status="simulator_filter_not_supported",
                    )
                    _emit_rebuild_stage(
                        "source_cache_full_simulation_fallback_start",
                        cache_source="fresh_simulation",
                        preflight_mode=normalized_preflight_mode,
                    )
                    if not full_fallback_already_loaded:
                        fresh_hit_tables = simulate_hit_tables(normalized_params)
                    _emit_rebuild_stage(
                        "source_cache_full_simulation_fallback_ready",
                        stage_started_at=fallback_started_at,
                        cache_source="fresh_simulation",
                        preflight_mode=normalized_preflight_mode,
                        hit_table_count=int(len(fresh_hit_tables or ())),
                    )
                else:
                    _update_targeted_runtime_flags(
                        targeted_simulation_supported=True,
                        targeted_simulation_used=bool(targeted_simulation_used),
                        targeted_simulation_fallback_reason=(targeted_simulation_fallback_reason),
                        cache_source="fresh_simulation",
                    )
            else:
                fresh_hit_tables = simulate_hit_tables(normalized_params)
        except Exception as exc:
            fresh_simulation_exception = exc
            fresh_hit_tables = []
    runtime_simulation_diagnostics = _resolve_runtime_simulation_diagnostics()
    if targeted_preflight_enabled and isinstance(runtime_simulation_diagnostics, Mapping):
        _update_targeted_runtime_flags(
            **{
                key: runtime_simulation_diagnostics.get(key)
                for key in (
                    "targeted_miller_hkl_inventory_before_filter",
                    "targeted_miller_hkl_inventory_after_filter",
                    "fresh_hit_table_hkl_inventory_before_filter",
                    "fresh_hit_table_hkl_branch_inventory_before_filter",
                    "fresh_hit_table_hkl_inventory",
                    "fresh_hit_table_hkl_branch_inventory",
                )
                if key in runtime_simulation_diagnostics
            }
        )
    _emit_rebuild_stage(
        (
            "source_cache_targeted_fresh_simulation_failed"
            if targeted_preflight_enabled and fresh_simulation_exception is not None
            else "source_cache_targeted_fresh_simulation_ready"
            if targeted_preflight_enabled
            else "source_cache_fresh_simulation_failed"
            if fresh_simulation_exception is not None
            else "source_cache_fresh_simulation_ready"
        ),
        stage_started_at=fresh_started_at,
        cache_source="fresh_simulation",
        hit_table_count=int(len(fresh_hit_tables or ())),
        targeted_simulation_supported=bool(targeted_simulation_supported),
        targeted_simulation_used=bool(targeted_simulation_used),
        status=(
            f"exception:{type(fresh_simulation_exception).__name__}"
            if fresh_simulation_exception is not None
            else str(runtime_simulation_diagnostics.get("status") or "ready")
        ),
    )
    fresh_rows, fresh_lattice, fresh_hit_tables, fresh_source_reflection_indices = (
        _build_source_rows_result(
            fresh_hit_tables,
            cache_source="fresh_simulation",
        )
    )
    if fresh_rows:
        required_source_labels = _required_pair_source_labels()
        fresh_source_counts = _source_counts_from_rows(fresh_rows)
        fresh_required_source_missing = [
            source_label
            for source_label in required_source_labels
            if int(fresh_source_counts.get(source_label, 0) or 0) <= 0
        ]
        if fresh_required_source_missing:
            _emit_rebuild_stage(
                "source_cache_fresh_simulation_required_source_missing",
                cache_source="fresh_simulation",
                status="required_source_rows_unavailable",
                required_pair_sources=list(required_source_labels),
                source_counts_before_filter=dict(fresh_source_counts),
                missing_required_sources=list(fresh_required_source_missing),
                reason="required_source_rows_unavailable",
            )
            runtime_simulation_diagnostics = dict(runtime_simulation_diagnostics or {})
            runtime_simulation_diagnostics.setdefault(
                "status",
                "required_source_rows_unavailable",
            )
            runtime_simulation_diagnostics["required_pair_sources"] = list(
                required_source_labels
            )
            runtime_simulation_diagnostics["missing_required_sources"] = list(
                fresh_required_source_missing
            )
            runtime_simulation_diagnostics["fresh_source_counts"] = dict(fresh_source_counts)
        else:
            return _success_result(
                fresh_rows,
                rebuild_source="fresh_simulation",
                peak_table_lattice=fresh_lattice,
                hit_tables_local=fresh_hit_tables,
                source_reflection_indices=fresh_source_reflection_indices,
                runtime_simulation_diagnostics=runtime_simulation_diagnostics,
            )

    runtime_status = str(runtime_simulation_diagnostics.get("status", "")).strip()
    if fresh_simulation_exception is not None:
        failure_status = runtime_status or "fresh_simulation_exception"
    elif fresh_required_source_missing:
        failure_status = "required_source_rows_unavailable"
    elif fresh_hit_tables:
        failure_status = "empty_source_rows"
    elif runtime_status:
        failure_status = runtime_status
    else:
        failure_status = "snapshot_rebuild_failed"

    diagnostics = _merge_runtime_simulation_diagnostics(
        {
            "source": "source_snapshot",
            "cache_family": "source_snapshot",
            "action": "rebuild",
            "consumer": lookup_context,
            "status": failure_status,
            "background_index": int(background_idx),
            "background_label": resolved_background_label,
            "requested_signature": requested_signature,
            "requested_signature_summary": requested_signature_summary,
            "raw_peak_count": 0,
            "projected_peak_count": 0,
            "signature_match": False,
            "rebuild_attempts": list(rebuild_attempts),
            "prior_diagnostics": (
                dict(prior_diagnostics) if isinstance(prior_diagnostics, Mapping) else {}
            ),
            "live_cache_inventory": _resolve_live_cache_inventory(),
        },
        runtime_simulation_diagnostics,
    )
    diagnostics = _finalize_diagnostics(diagnostics)
    if isinstance(live_cache_validation, Mapping) and live_cache_validation:
        diagnostics["live_runtime_cache_validation"] = copy.deepcopy(dict(live_cache_validation))
    if fresh_simulation_exception is not None:
        diagnostics.setdefault(
            "exception_type",
            type(fresh_simulation_exception).__name__,
        )
        diagnostics.setdefault(
            "exception_message",
            str(fresh_simulation_exception),
        )
    return GeometryFitSourceRowRebuildResult(
        background_index=int(background_idx),
        requested_signature=requested_signature,
        requested_signature_summary=requested_signature_summary,
        projected_rows=[],
        stored_rows=[],
        rebuild_source=None,
        rebuild_attempts=list(rebuild_attempts),
        diagnostics=diagnostics,
    )


def build_runtime_geometry_fit_manual_dataset_bindings(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    image_size: int,
    display_rotate_k: int,
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]],
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None],
    geometry_manual_simulated_peaks_for_params: Callable[..., object],
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]],
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ],
    geometry_manual_project_peaks_to_current_view: (
        Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]] | None
    ) = None,
    geometry_manual_project_peaks_for_background_view: (
        Callable[[int, Sequence[dict[str, object]] | None], list[dict[str, object]]] | None
    ) = None,
    unrotate_display_peaks: Callable[..., list[dict[str, object]]],
    display_to_native_sim_coords: Callable[..., tuple[float, float]],
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]],
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]],
    orient_image_for_fit: Callable[..., object],
    backend_detector_coords_to_native_detector_coords: (
        Callable[
            [float, float, Sequence[object] | None],
            tuple[float | None, float | None],
        ]
        | None
    ) = None,
    native_detector_coords_to_bundle_detector_coords: (
        Callable[[float, float], tuple[float | None, float | None]] | None
    ) = None,
    native_detector_coords_to_detector_display_coords: (
        Callable[[float, float], tuple[float | None, float | None] | None] | None
    ) = None,
    native_detector_coords_to_detector_display_coords_for_background: (
        Callable[
            [int],
            Callable[[float, float], tuple[float | None, float | None] | None] | None,
        ]
        | None
    ) = None,
    geometry_manual_source_rows_for_background: Callable[..., object] | None = None,
    geometry_manual_rebuild_source_rows_for_background: (Callable[..., object] | None) = None,
    geometry_manual_last_source_snapshot_diagnostics: (
        Callable[[], Mapping[str, object]] | None
    ) = None,
    geometry_manual_last_simulation_diagnostics: (Callable[[], Mapping[str, object]] | None) = None,
    geometry_manual_match_config: Callable[[], Mapping[str, object]] | None = None,
    pick_uses_caked_space: Callable[[], bool] | None = None,
    geometry_manual_caked_view_for_index: Callable[[int], object] | None = None,
    geometry_manual_refresh_pair_entry: (
        Callable[[Mapping[str, object] | None], dict[str, object] | None] | None
    ) = None,
) -> GeometryFitRuntimeManualDatasetBindings:
    """Build the live manual-pair dataset bundle used during geometry-fit prep."""

    return GeometryFitRuntimeManualDatasetBindings(
        osc_files=osc_files,
        current_background_index=int(current_background_index),
        image_size=int(image_size),
        display_rotate_k=int(display_rotate_k),
        geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
        load_background_by_index=load_background_by_index,
        apply_background_backend_orientation=apply_background_backend_orientation,
        geometry_manual_simulated_peaks_for_params=(geometry_manual_simulated_peaks_for_params),
        geometry_manual_simulated_lookup=geometry_manual_simulated_lookup,
        geometry_manual_source_rows_for_background=(geometry_manual_source_rows_for_background),
        geometry_manual_rebuild_source_rows_for_background=(
            geometry_manual_rebuild_source_rows_for_background
        ),
        geometry_manual_last_source_snapshot_diagnostics=(
            geometry_manual_last_source_snapshot_diagnostics
        ),
        geometry_manual_last_simulation_diagnostics=(geometry_manual_last_simulation_diagnostics),
        geometry_manual_match_config=geometry_manual_match_config,
        geometry_manual_entry_display_coords=geometry_manual_entry_display_coords,
        geometry_manual_project_peaks_to_current_view=(
            geometry_manual_project_peaks_to_current_view
        ),
        geometry_manual_project_peaks_for_background_view=(
            geometry_manual_project_peaks_for_background_view
        ),
        unrotate_display_peaks=unrotate_display_peaks,
        display_to_native_sim_coords=display_to_native_sim_coords,
        select_fit_orientation=select_fit_orientation,
        apply_orientation_to_entries=apply_orientation_to_entries,
        orient_image_for_fit=orient_image_for_fit,
        backend_detector_coords_to_native_detector_coords=(
            backend_detector_coords_to_native_detector_coords
        ),
        native_detector_coords_to_bundle_detector_coords=(
            native_detector_coords_to_bundle_detector_coords
        ),
        native_detector_coords_to_detector_display_coords=(
            native_detector_coords_to_detector_display_coords
        ),
        native_detector_coords_to_detector_display_coords_for_background=(
            native_detector_coords_to_detector_display_coords_for_background
        ),
        pick_uses_caked_space=pick_uses_caked_space,
        geometry_manual_caked_view_for_index=geometry_manual_caked_view_for_index,
        geometry_manual_refresh_pair_entry=geometry_manual_refresh_pair_entry,
    )


def make_runtime_geometry_fit_manual_dataset_bindings_factory(
    *,
    osc_files_factory: Callable[[], Sequence[object]],
    current_background_index_factory: Callable[[], object],
    image_size: int,
    display_rotate_k: int,
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    load_background_by_index: Callable[[int], tuple[np.ndarray, np.ndarray]],
    apply_background_backend_orientation: Callable[[np.ndarray], np.ndarray | None],
    geometry_manual_simulated_peaks_for_params: Callable[..., object],
    geometry_manual_simulated_lookup: Callable[[object], Mapping[object, object]],
    geometry_manual_entry_display_coords: Callable[
        [Mapping[str, object]],
        Sequence[object] | None,
    ],
    geometry_manual_project_peaks_to_current_view: (
        Callable[[Sequence[dict[str, object]] | None], list[dict[str, object]]] | None
    ) = None,
    geometry_manual_project_peaks_for_background_view: (
        Callable[[int, Sequence[dict[str, object]] | None], list[dict[str, object]]] | None
    ) = None,
    unrotate_display_peaks: Callable[..., list[dict[str, object]]],
    display_to_native_sim_coords: Callable[..., tuple[float, float]],
    select_fit_orientation: Callable[..., tuple[dict[str, object], dict[str, object]]],
    apply_orientation_to_entries: Callable[..., list[dict[str, object]]],
    orient_image_for_fit: Callable[..., object],
    backend_detector_coords_to_native_detector_coords: (
        Callable[
            [float, float, Sequence[object] | None],
            tuple[float | None, float | None],
        ]
        | None
    ) = None,
    native_detector_coords_to_bundle_detector_coords: (
        Callable[[float, float], tuple[float | None, float | None]] | None
    ) = None,
    native_detector_coords_to_detector_display_coords: (
        Callable[[float, float], tuple[float | None, float | None] | None] | None
    ) = None,
    native_detector_coords_to_detector_display_coords_for_background: (
        Callable[
            [int],
            Callable[[float, float], tuple[float | None, float | None] | None] | None,
        ]
        | None
    ) = None,
    geometry_manual_source_rows_for_background: Callable[..., object] | None = None,
    geometry_manual_rebuild_source_rows_for_background: (Callable[..., object] | None) = None,
    geometry_manual_last_source_snapshot_diagnostics: (
        Callable[[], Mapping[str, object]] | None
    ) = None,
    geometry_manual_last_simulation_diagnostics: (Callable[[], Mapping[str, object]] | None) = None,
    geometry_manual_match_config: Callable[[], Mapping[str, object]] | None = None,
    pick_uses_caked_space: Callable[[], bool] | None = None,
    geometry_manual_caked_view_for_index: Callable[[int], object] | None = None,
    geometry_manual_refresh_pair_entry: (
        Callable[[Mapping[str, object] | None], dict[str, object] | None] | None
    ) = None,
) -> Callable[[], GeometryFitRuntimeManualDatasetBindings]:
    """Build a factory that resolves the live manual-pair dataset bundle on demand."""

    def _build() -> GeometryFitRuntimeManualDatasetBindings:
        return build_runtime_geometry_fit_manual_dataset_bindings(
            osc_files=osc_files_factory(),
            current_background_index=int(current_background_index_factory()),
            image_size=int(image_size),
            display_rotate_k=int(display_rotate_k),
            geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            load_background_by_index=load_background_by_index,
            apply_background_backend_orientation=(apply_background_backend_orientation),
            geometry_manual_simulated_peaks_for_params=(geometry_manual_simulated_peaks_for_params),
            geometry_manual_simulated_lookup=geometry_manual_simulated_lookup,
            geometry_manual_source_rows_for_background=(geometry_manual_source_rows_for_background),
            geometry_manual_rebuild_source_rows_for_background=(
                geometry_manual_rebuild_source_rows_for_background
            ),
            geometry_manual_last_source_snapshot_diagnostics=(
                geometry_manual_last_source_snapshot_diagnostics
            ),
            geometry_manual_last_simulation_diagnostics=(
                geometry_manual_last_simulation_diagnostics
            ),
            geometry_manual_match_config=geometry_manual_match_config,
            geometry_manual_entry_display_coords=(geometry_manual_entry_display_coords),
            geometry_manual_project_peaks_to_current_view=(
                geometry_manual_project_peaks_to_current_view
            ),
            geometry_manual_project_peaks_for_background_view=(
                geometry_manual_project_peaks_for_background_view
            ),
            unrotate_display_peaks=unrotate_display_peaks,
            display_to_native_sim_coords=display_to_native_sim_coords,
            select_fit_orientation=select_fit_orientation,
            apply_orientation_to_entries=apply_orientation_to_entries,
            orient_image_for_fit=orient_image_for_fit,
            backend_detector_coords_to_native_detector_coords=(
                backend_detector_coords_to_native_detector_coords
            ),
            native_detector_coords_to_bundle_detector_coords=(
                native_detector_coords_to_bundle_detector_coords
            ),
            native_detector_coords_to_detector_display_coords=(
                native_detector_coords_to_detector_display_coords
            ),
            native_detector_coords_to_detector_display_coords_for_background=(
                native_detector_coords_to_detector_display_coords_for_background
            ),
            pick_uses_caked_space=pick_uses_caked_space,
            geometry_manual_caked_view_for_index=(geometry_manual_caked_view_for_index),
            geometry_manual_refresh_pair_entry=(geometry_manual_refresh_pair_entry),
        )

    return _build


def make_runtime_geometry_fit_action_prepare_bindings_factory(
    *,
    fit_config: Mapping[str, object] | None,
    theta_initial: object,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
) -> Callable[[Sequence[str]], GeometryFitRuntimePreparationBindings]:
    """Build the live prepare-bundle factory for one geometry-fit action."""

    def _build(var_names: Sequence[str]) -> GeometryFitRuntimePreparationBindings:
        return GeometryFitRuntimePreparationBindings(
            fit_config=fit_config,
            theta_initial=theta_initial,
            apply_geometry_fit_background_selection=(apply_geometry_fit_background_selection),
            current_geometry_fit_background_indices=(current_geometry_fit_background_indices),
            geometry_fit_uses_shared_theta_offset=(geometry_fit_uses_shared_theta_offset),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
            build_runtime_config=(
                lambda fit_params: build_runtime_config_factory(
                    list(var_names),
                    fit_params,
                )
            ),
        )

    return _build


def build_runtime_geometry_fit_action_execution_bindings(
    *,
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> GeometryFitRuntimeActionExecutionBindings:
    """Build the live execution-bundle for one geometry-fit action."""

    return GeometryFitRuntimeActionExecutionBindings(
        downloads_dir=downloads_dir,
        log_dir=log_dir,
        simulation_runtime_state=simulation_runtime_state,
        background_runtime_state=background_runtime_state,
        theta_initial_var=theta_initial_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        current_ui_params=current_ui_params,
        var_map=var_map,
        background_theta_for_index=background_theta_for_index,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        capture_undo_state=capture_undo_state,
        push_undo_state=push_undo_state,
        replace_dataset_cache=replace_dataset_cache,
        request_preview_skip_once=request_preview_skip_once,
        schedule_update=schedule_update,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
        solver_inputs=solver_inputs,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
        simulate_and_compare_hkl=simulate_and_compare_hkl,
        aggregate_match_centers=aggregate_match_centers,
        build_overlay_records=build_overlay_records,
        compute_frame_diagnostics=compute_frame_diagnostics,
        live_update_callback=live_update_callback,
    )


def build_runtime_geometry_fit_action_bindings(
    *,
    value_callbacks: GeometryFitRuntimeValueCallbacks,
    fit_config: Mapping[str, object] | None,
    theta_initial: object,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
    solve_fit: Callable[..., object],
    stamp_factory: Callable[[], str],
    flush_ui: Callable[[], None] | None = None,
) -> GeometryFitRuntimeActionBindings:
    """Build the top-level live geometry-fit action bindings."""

    execution_binding_kwargs: dict[str, object] = {
        "downloads_dir": downloads_dir,
        "simulation_runtime_state": simulation_runtime_state,
        "background_runtime_state": background_runtime_state,
        "theta_initial_var": theta_initial_var,
        "geometry_theta_offset_var": geometry_theta_offset_var,
        "current_ui_params": current_ui_params,
        "var_map": var_map,
        "background_theta_for_index": background_theta_for_index,
        "refresh_status": refresh_status,
        "update_manual_pick_button_label": update_manual_pick_button_label,
        "capture_undo_state": capture_undo_state,
        "push_undo_state": push_undo_state,
        "replace_dataset_cache": replace_dataset_cache,
        "request_preview_skip_once": request_preview_skip_once,
        "schedule_update": schedule_update,
        "draw_overlay_records": draw_overlay_records,
        "draw_initial_pairs_overlay": draw_initial_pairs_overlay,
        "set_last_overlay_state": set_last_overlay_state,
        "set_progress_text": set_progress_text,
        "cmd_line": cmd_line,
        "solver_inputs": solver_inputs,
        "sim_display_rotate_k": int(sim_display_rotate_k),
        "background_display_rotate_k": int(background_display_rotate_k),
        "simulate_and_compare_hkl": simulate_and_compare_hkl,
        "aggregate_match_centers": aggregate_match_centers,
        "build_overlay_records": build_overlay_records,
        "compute_frame_diagnostics": compute_frame_diagnostics,
        "live_update_callback": live_update_callback,
    }
    if log_dir is not None:
        execution_binding_kwargs["log_dir"] = log_dir

    return GeometryFitRuntimeActionBindings(
        value_callbacks=value_callbacks,
        prepare_bindings_factory=make_runtime_geometry_fit_action_prepare_bindings_factory(
            fit_config=fit_config,
            theta_initial=theta_initial,
            apply_geometry_fit_background_selection=(apply_geometry_fit_background_selection),
            current_geometry_fit_background_indices=(current_geometry_fit_background_indices),
            geometry_fit_uses_shared_theta_offset=(geometry_fit_uses_shared_theta_offset),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
            build_runtime_config_factory=build_runtime_config_factory,
        ),
        execution_bindings=build_runtime_geometry_fit_action_execution_bindings(
            **execution_binding_kwargs
        ),
        solve_fit=solve_fit,
        stamp_factory=stamp_factory,
        flush_ui=flush_ui,
    )


def make_runtime_geometry_fit_action_bindings_factory(
    *,
    value_callbacks_factory: Callable[[], GeometryFitRuntimeValueCallbacks],
    fit_config: Mapping[str, object] | None,
    theta_initial_factory: Callable[[], object],
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    ensure_geometry_fit_caked_view: Callable[[], None],
    manual_dataset_bindings_factory: Callable[
        [],
        GeometryFitRuntimeManualDatasetBindings,
    ],
    build_runtime_config_factory: Callable[
        [Sequence[str], Mapping[str, object]],
        dict[str, object],
    ],
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs_factory: Callable[[], GeometryFitRuntimeSolverInputs],
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
    solve_fit: Callable[..., object],
    stamp_factory: Callable[[], str],
    flush_ui: Callable[[], None] | None = None,
) -> Callable[[], GeometryFitRuntimeActionBindings]:
    """Build a factory that resolves live geometry-fit action bindings on demand."""

    def _build() -> GeometryFitRuntimeActionBindings:
        return build_runtime_geometry_fit_action_bindings(
            value_callbacks=value_callbacks_factory(),
            fit_config=fit_config,
            theta_initial=theta_initial_factory(),
            apply_geometry_fit_background_selection=(apply_geometry_fit_background_selection),
            current_geometry_fit_background_indices=(current_geometry_fit_background_indices),
            geometry_fit_uses_shared_theta_offset=(geometry_fit_uses_shared_theta_offset),
            apply_background_theta_metadata=apply_background_theta_metadata,
            current_background_theta_values=current_background_theta_values,
            current_geometry_theta_offset=current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings_factory(),
            build_runtime_config_factory=build_runtime_config_factory,
            downloads_dir=downloads_dir,
            log_dir=log_dir,
            simulation_runtime_state=simulation_runtime_state,
            background_runtime_state=background_runtime_state,
            theta_initial_var=theta_initial_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            current_ui_params=current_ui_params,
            var_map=var_map,
            background_theta_for_index=background_theta_for_index,
            refresh_status=refresh_status,
            update_manual_pick_button_label=update_manual_pick_button_label,
            capture_undo_state=capture_undo_state,
            push_undo_state=push_undo_state,
            replace_dataset_cache=replace_dataset_cache,
            request_preview_skip_once=request_preview_skip_once,
            schedule_update=schedule_update,
            draw_overlay_records=draw_overlay_records,
            draw_initial_pairs_overlay=draw_initial_pairs_overlay,
            set_last_overlay_state=set_last_overlay_state,
            set_progress_text=set_progress_text,
            cmd_line=cmd_line,
            solver_inputs=solver_inputs_factory(),
            sim_display_rotate_k=int(sim_display_rotate_k),
            background_display_rotate_k=int(background_display_rotate_k),
            simulate_and_compare_hkl=simulate_and_compare_hkl,
            aggregate_match_centers=aggregate_match_centers,
            build_overlay_records=build_overlay_records,
            compute_frame_diagnostics=compute_frame_diagnostics,
            live_update_callback=live_update_callback,
            solve_fit=solve_fit,
            stamp_factory=stamp_factory,
            flush_ui=flush_ui,
        )

    return _build


def make_runtime_geometry_fit_action_callback(
    bindings_factory: Callable[[], GeometryFitRuntimeActionBindings],
    *,
    before_run: Callable[[], None] | None = None,
    run_action: Callable[..., GeometryFitRuntimeActionResult] | None = None,
    after_run: Callable[[GeometryFitRuntimeActionResult], None] | None = None,
) -> Callable[[], None]:
    """Build the zero-arg Tk-safe runtime callback for the top-level geometry-fit action."""

    def _run() -> None:
        if callable(before_run):
            before_run()
        action = run_action if callable(run_action) else run_runtime_geometry_fit_action
        result = action(bindings=bindings_factory())
        if callable(after_run):
            after_run(result)
        # Tkinter stringifies callback return values. Returning the rich action
        # result here can force a dataclass repr of SciPy OptimizeResult payloads.
        return None

    return _run


def _format_geometry_fit_notice_path(log_path: object) -> str:
    """Return one log path string formatted for user-facing notices."""

    try:
        formatted_path = os.fspath(log_path)
    except TypeError:
        formatted_path = str(log_path)
    if isinstance(formatted_path, bytes):
        formatted_path = os.fsdecode(formatted_path)
    else:
        formatted_path = str(formatted_path)
    windows_path = PureWindowsPath(formatted_path)
    if windows_path.drive and windows_path.is_absolute():
        return str(windows_path)
    return formatted_path


def build_geometry_fit_action_notice(
    action_result: GeometryFitRuntimeActionResult | None,
) -> GeometryFitActionNotice | None:
    """Return a user-facing notice for one failed or rejected geometry fit."""

    if action_result is None:
        return None

    error_text = str(action_result.error_text or "").strip()
    if error_text:
        lines = [error_text]
        log_path = None
        execution_result = action_result.execution_result
        if execution_result is not None:
            log_path = getattr(execution_result, "log_path", None)
        if log_path is None:
            prepare_result = action_result.prepare_result
            if prepare_result is not None:
                log_path = getattr(prepare_result, "log_path", None)
        if log_path is not None:
            lines.append(f"Fit log: {_format_geometry_fit_notice_path(log_path)}")
        return GeometryFitActionNotice(
            level="error",
            title="Geometry Fit Failed",
            message="\n".join(lines),
        )

    execution_result = action_result.execution_result
    if execution_result is None:
        return None

    apply_result = execution_result.apply_result
    if apply_result is None or bool(apply_result.accepted):
        return None

    lines = ["Geometry fit finished but the solution was rejected."]
    rejection_reason = str(apply_result.rejection_reason or "").strip()
    if rejection_reason:
        lines.append(rejection_reason)
    lines.append("The live geometry state was left unchanged.")
    log_path = getattr(execution_result, "log_path", None)
    if log_path is not None:
        lines.append(f"Fit log: {_format_geometry_fit_notice_path(log_path)}")
    return GeometryFitActionNotice(
        level="warning",
        title="Geometry Fit Rejected",
        message="\n".join(lines),
    )


def copy_geometry_fit_state_value(value):
    """Deep-copy simple geometry-fit GUI state."""

    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: copy_geometry_fit_state_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [copy_geometry_fit_state_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(copy_geometry_fit_state_value(val) for val in value)
    return value


def _copy_geometry_fit_dataset_spec_for_state(
    spec: Mapping[str, object] | None,
) -> dict[str, object]:
    copied = dict(copy_geometry_fit_state_value(dict(spec))) if isinstance(spec, Mapping) else {}
    copied.pop("fit_space_projector", None)
    return copied


def current_geometry_fit_ui_params(
    *,
    zb: float,
    zs: float,
    theta_initial: float,
    psi_z: float,
    chi: float,
    cor_angle: float,
    gamma: float,
    Gamma: float,
    corto_detector: float,
    a: float,
    c: float,
    center_x: float,
    center_y: float,
    theta_offset: float | None = None,
) -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    params = {
        "zb": float(zb),
        "zs": float(zs),
        "theta_initial": float(theta_initial),
        "psi_z": float(psi_z),
        "chi": float(chi),
        "cor_angle": float(cor_angle),
        "gamma": float(gamma),
        "Gamma": float(Gamma),
        "corto_detector": float(corto_detector),
        "a": float(a),
        "c": float(c),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center": [float(center_x), float(center_y)],
    }
    if theta_offset is not None:
        params["theta_offset"] = float(theta_offset)
    return params


def current_geometry_fit_var_names(
    *,
    fit_zb: bool,
    fit_zs: bool,
    fit_theta: bool,
    fit_psi_z: bool,
    fit_chi: bool,
    fit_cor: bool,
    fit_gamma: bool,
    fit_Gamma: bool,
    fit_dist: bool,
    fit_a: bool,
    fit_c: bool,
    fit_center_x: bool,
    fit_center_y: bool,
    use_shared_theta_offset: bool = False,
) -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    var_names: list[str] = []
    if fit_zb:
        var_names.append("zb")
    if fit_zs:
        var_names.append("zs")
    if fit_theta:
        var_names.append("theta_offset" if use_shared_theta_offset else "theta_initial")
    if fit_psi_z:
        var_names.append("psi_z")
    if fit_chi:
        var_names.append("chi")
    if fit_cor:
        var_names.append("cor_angle")
    if fit_gamma:
        var_names.append("gamma")
    if fit_Gamma:
        var_names.append("Gamma")
    if fit_dist:
        var_names.append("corto_detector")
    if fit_a:
        var_names.append("a")
    if fit_c:
        var_names.append("c")
    if fit_center_x:
        var_names.append("center_x")
    if fit_center_y:
        var_names.append("center_y")
    return var_names


def geometry_fit_constraint_source_name(name: str) -> str:
    """Map fitted parameter names back to the UI constraint control names."""

    if str(name) == "theta_offset":
        return "theta_initial"
    return str(name)


def geometry_fit_constraint_parameter_name(
    name: str,
    *,
    use_shared_theta_offset: bool = False,
) -> str:
    """Map UI constraint row names to the active fitted parameter names."""

    if str(name) == "theta_initial" and bool(use_shared_theta_offset):
        return "theta_offset"
    return str(name)


def read_runtime_geometry_fit_constraint_state(
    *,
    controls: Mapping[str, object],
    names: Sequence[str] | None = None,
    use_shared_theta_offset: bool = False,
) -> dict[str, dict[str, float]]:
    """Read the live geometry-fit constraint controls into normalized settings."""

    selected_names = list(names) if names is not None else list(controls)
    state: dict[str, dict[str, float]] = {}
    for name in selected_names:
        control = controls.get(
            geometry_fit_constraint_source_name(
                geometry_fit_constraint_parameter_name(
                    str(name),
                    use_shared_theta_offset=use_shared_theta_offset,
                )
            )
        )
        if not isinstance(control, dict):
            continue
        try:
            window = float(control["window_var"].get())
        except Exception:
            window = float("nan")
        try:
            pull = float(control["pull_var"].get())
        except Exception:
            pull = 0.0
        if not np.isfinite(window):
            continue
        window = max(0.0, float(window))
        if not np.isfinite(pull):
            pull = 0.0
        pull = min(max(float(pull), 0.0), 1.0)
        state[str(name)] = {
            "window": float(window),
            "pull": float(pull),
        }
    return state


def _geometry_fit_config_section(
    fit_config: Mapping[str, object] | None,
    section: str,
) -> Mapping[str, object]:
    """Return one normalized geometry-fit config mapping section."""

    if not isinstance(fit_config, Mapping):
        return {}
    fit_geometry_cfg = fit_config.get("geometry", {})
    if "geometry" in fit_config and isinstance(fit_geometry_cfg, Mapping):
        container_cfg = fit_geometry_cfg
    else:
        container_cfg = fit_config
    section_cfg = container_cfg.get(section, {}) or {}
    if not isinstance(section_cfg, Mapping):
        return {}
    return section_cfg


def read_geometry_fit_caked_roi_config(
    fit_config: Mapping[str, object] | None,
    *,
    enabled_override: object | None = None,
) -> dict[str, object]:
    """Return normalized branch-restricted caking settings for geometry fit."""

    section_cfg = _geometry_fit_config_section(fit_config, "caked_roi")

    enabled = enabled_override
    if enabled is None:
        enabled = section_cfg.get("enabled", False)

    half_width_px = _geometry_fit_cache_finite_float(section_cfg.get("half_width_px"))
    if half_width_px is None or half_width_px < 0.0:
        half_width_px = 15.0

    max_detector_fraction = _geometry_fit_cache_finite_float(
        section_cfg.get("max_detector_fraction")
    )
    if max_detector_fraction is None or max_detector_fraction <= 0.0 or max_detector_fraction > 1.0:
        max_detector_fraction = 0.35

    return {
        "enabled": bool(enabled),
        "half_width_px": float(half_width_px),
        "max_detector_fraction": float(max_detector_fraction),
    }


def _geometry_fit_caked_roi_branch_selection(
    validation: Mapping[str, object] | None,
) -> dict[str, set[tuple[object, int]]]:
    """Return the selected branch identities resolved from active manual pairs."""

    selected_q_groups: set[tuple[object, int]] = set()
    selected_reflections: set[tuple[object, int]] = set()
    selected_tables: set[tuple[object, int]] = set()
    selected_hkls: set[tuple[object, int]] = set()

    for raw_entry in (
        validation.get("resolved_pairs", ()) if isinstance(validation, Mapping) else ()
    ):
        if not isinstance(raw_entry, Mapping):
            continue
        branch_idx = _geometry_fit_coerce_nonnegative_index(raw_entry.get("source_branch_index"))
        if branch_idx not in {0, 1}:
            continue

        q_group_key = _geometry_fit_cache_jsonable(raw_entry.get("q_group_key"))
        if q_group_key is not None:
            try:
                q_group_token = json.dumps(q_group_key, sort_keys=True)
            except Exception:
                q_group_token = repr(q_group_key)
            selected_q_groups.add((q_group_token, int(branch_idx)))

        reflection_idx = _geometry_fit_coerce_nonnegative_index(
            raw_entry.get("source_reflection_index")
        )
        if reflection_idx is not None:
            selected_reflections.add((int(reflection_idx), int(branch_idx)))

        table_idx = _geometry_fit_coerce_nonnegative_index(raw_entry.get("source_table_index"))
        if table_idx is not None:
            selected_tables.add((int(table_idx), int(branch_idx)))

        hkl_key = _geometry_fit_normalized_hkl(raw_entry.get("hkl"))
        if hkl_key is not None:
            selected_hkls.add((tuple(int(v) for v in hkl_key), int(branch_idx)))

    return {
        "q_groups": selected_q_groups,
        "reflections": selected_reflections,
        "tables": selected_tables,
        "hkls": selected_hkls,
    }


def _geometry_fit_caked_roi_row_matches_selection(
    entry: Mapping[str, object] | None,
    selected_branches: Mapping[str, set[tuple[object, int]]] | None,
) -> bool:
    """Return whether one canonical source row belongs to the selected branches."""

    if not isinstance(entry, Mapping):
        return False
    is_canonical, _reason = _geometry_fit_is_canonical_live_source_entry(entry)
    if not is_canonical:
        return False
    branch_idx = _geometry_fit_source_branch_index(entry)
    if branch_idx not in {0, 1}:
        return False

    q_group_key = _geometry_fit_cache_jsonable(entry.get("q_group_key"))
    if q_group_key is not None:
        try:
            q_group_token = json.dumps(q_group_key, sort_keys=True)
        except Exception:
            q_group_token = repr(q_group_key)
        if (q_group_token, int(branch_idx)) in (
            selected_branches.get("q_groups", set())
            if isinstance(selected_branches, Mapping)
            else set()
        ):
            return True

    reflection_idx = _geometry_fit_coerce_nonnegative_index(entry.get("source_reflection_index"))
    if reflection_idx is not None and (int(reflection_idx), int(branch_idx)) in (
        selected_branches.get("reflections", set())
        if isinstance(selected_branches, Mapping)
        else set()
    ):
        return True

    table_idx = _geometry_fit_coerce_nonnegative_index(entry.get("source_table_index"))
    if table_idx is not None and (int(table_idx), int(branch_idx)) in (
        selected_branches.get("tables", set()) if isinstance(selected_branches, Mapping) else set()
    ):
        return True

    hkl_key = _geometry_fit_normalized_hkl(entry.get("hkl"))
    if hkl_key is not None and (tuple(int(v) for v in hkl_key), int(branch_idx)) in (
        selected_branches.get("hkls", set()) if isinstance(selected_branches, Mapping) else set()
    ):
        return True

    return False


def _geometry_fit_caked_roi_angle_point(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    """Return one fit-space ``(2theta_deg, phi_deg)`` point for ROI rasterization."""

    if not isinstance(entry, Mapping):
        return None
    candidate_keys = (
        ("background_two_theta_deg", "background_phi_deg"),
        ("simulated_two_theta_deg", "simulated_phi_deg"),
        ("two_theta_deg", "phi_deg"),
    )
    for two_theta_key, phi_key in candidate_keys:
        two_theta_val = _geometry_fit_cache_finite_float(entry.get(two_theta_key))
        phi_val = _geometry_fit_cache_finite_float(entry.get(phi_key))
        if two_theta_val is None or phi_val is None:
            continue
        return float(two_theta_val), float(phi_val)
    return None


def _geometry_fit_caked_roi_native_point(
    entry: Mapping[str, object] | None,
    *,
    fit_space_to_detector_point: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
) -> tuple[float, float] | None:
    """Return one detector-space ``(col, row)`` point for ROI rasterization."""

    if not isinstance(entry, Mapping):
        return None
    fit_space_point = _geometry_fit_caked_roi_angle_point(entry)
    if fit_space_point is not None:
        if not callable(fit_space_to_detector_point):
            return None
        try:
            projected_point = fit_space_to_detector_point(
                float(fit_space_point[0]),
                float(fit_space_point[1]),
            )
        except Exception:
            projected_point = None
        if (
            isinstance(projected_point, tuple)
            and len(projected_point) >= 2
            and projected_point[0] is not None
            and projected_point[1] is not None
            and np.isfinite(float(projected_point[0]))
            and np.isfinite(float(projected_point[1]))
        ):
            return float(projected_point[0]), float(projected_point[1])
        return None
    candidate_keys = (
        ("detector_x", "detector_y"),
        ("refined_sim_native_x", "refined_sim_native_y"),
        ("sim_col_raw", "sim_row_raw"),
    )
    for col_key, row_key in candidate_keys:
        col_val = _geometry_fit_cache_finite_float(entry.get(col_key))
        row_val = _geometry_fit_cache_finite_float(entry.get(row_key))
        if col_val is None or row_val is None:
            continue
        return float(col_val), float(row_val)
    return None


def _geometry_fit_caked_roi_disk_offsets(radius: float) -> list[tuple[int, int]]:
    """Return integer dilation offsets for one detector-space disk."""

    if not np.isfinite(radius) or radius <= 0.0:
        return [(0, 0)]
    pixel_radius = int(math.ceil(float(radius)))
    offsets: list[tuple[int, int]] = []
    radius_sq = float(radius) * float(radius)
    for drow in range(-pixel_radius, pixel_radius + 1):
        for dcol in range(-pixel_radius, pixel_radius + 1):
            if float(drow * drow + dcol * dcol) <= radius_sq + 1.0e-9:
                offsets.append((int(drow), int(dcol)))
    return offsets or [(0, 0)]


def build_geometry_fit_caked_roi_selection(
    source_rows: Sequence[object] | None,
    *,
    required_pairs: Sequence[Mapping[str, object]] | None = None,
    image_shape: Sequence[object] | None = None,
    fit_config: Mapping[str, object] | None = None,
    enabled_override: object | None = None,
    fit_space_to_detector_point: Callable[
        [float, float],
        tuple[float | None, float | None] | None,
    ]
    | None = None,
) -> dict[str, object]:
    """Build the detector-space ROI used for branch-restricted geometry-fit caking."""

    roi_cfg = read_geometry_fit_caked_roi_config(
        fit_config,
        enabled_override=enabled_override,
    )
    enabled = bool(roi_cfg.get("enabled", False))
    half_width_px = float(roi_cfg.get("half_width_px", 15.0))
    max_detector_fraction = float(roi_cfg.get("max_detector_fraction", 0.35))

    result: dict[str, object] = {
        "enabled": bool(enabled),
        "valid": False,
        "rows": np.empty((0,), dtype=np.int32),
        "cols": np.empty((0,), dtype=np.int32),
        "pixel_count": 0,
        "fraction": 0.0,
        "fallback_reason": None,
        "half_width_px": float(half_width_px),
        "max_detector_fraction": float(max_detector_fraction),
        "resolved_pair_count": 0,
        "selected_branch_count": 0,
    }
    if not enabled:
        result["fallback_reason"] = "disabled"
        return result

    try:
        height = int(image_shape[0]) if image_shape is not None else 0
        width = int(image_shape[1]) if image_shape is not None else 0
    except Exception:
        height = 0
        width = 0
    if height <= 0 or width <= 0:
        result["fallback_reason"] = "invalid_image_shape"
        return result

    source_rows_list = [dict(entry) for entry in (source_rows or ()) if isinstance(entry, Mapping)]
    if not source_rows_list:
        result["fallback_reason"] = "no_source_rows"
        return result

    validation = validate_geometry_fit_live_source_rows(
        source_rows_list,
        required_pairs=required_pairs,
    )
    resolved_pairs = [
        dict(entry)
        for entry in (validation.get("resolved_pairs", ()) or ())
        if isinstance(entry, Mapping)
    ]
    result["resolved_pair_count"] = int(len(resolved_pairs))
    if required_pairs and not validation.get("valid", False):
        result["fallback_reason"] = "pair_validation_failed"
        return result
    if required_pairs and not resolved_pairs:
        result["fallback_reason"] = "no_resolved_pairs"
        return result

    selected_branches = _geometry_fit_caked_roi_branch_selection(validation)
    selected_rows = [
        dict(entry)
        for entry in source_rows_list
        if _geometry_fit_caked_roi_row_matches_selection(entry, selected_branches)
    ]
    if not selected_rows:
        result["fallback_reason"] = "no_selected_branch_rows"
        return result

    grouped_points: dict[tuple[object, int], list[tuple[int, int, float, float]]] = defaultdict(
        list
    )
    for source_order, entry in enumerate(selected_rows):
        point = _geometry_fit_caked_roi_native_point(
            entry,
            fit_space_to_detector_point=fit_space_to_detector_point,
        )
        if point is None:
            continue
        col_val, row_val = point
        if not (np.isfinite(col_val) and np.isfinite(row_val)):
            continue
        branch_idx = _geometry_fit_source_branch_index(entry)
        if branch_idx not in {0, 1}:
            continue
        q_group_key = _geometry_fit_cache_jsonable(entry.get("q_group_key"))
        if q_group_key is not None:
            try:
                group_token = json.dumps(q_group_key, sort_keys=True)
            except Exception:
                group_token = repr(q_group_key)
        else:
            group_token = repr(
                (
                    _geometry_fit_coerce_nonnegative_index(entry.get("source_reflection_index")),
                    _geometry_fit_coerce_nonnegative_index(entry.get("source_table_index")),
                    _geometry_fit_normalized_hkl(entry.get("hkl")),
                )
            )
        row_index = _geometry_fit_coerce_nonnegative_index(entry.get("source_row_index"))
        if row_index is None:
            row_index = int(source_order)
        grouped_points[(group_token, int(branch_idx))].append(
            (int(row_index), int(source_order), float(col_val), float(row_val))
        )

    result["selected_branch_count"] = int(len(grouped_points))
    if not grouped_points:
        result["fallback_reason"] = "no_native_detector_points"
        return result

    offsets = _geometry_fit_caked_roi_disk_offsets(half_width_px)
    linear_indices: set[int] = set()

    for point_group in grouped_points.values():
        ordered_group = sorted(point_group, key=lambda item: (int(item[0]), int(item[1])))
        sampled_points: list[tuple[float, float]] = []
        for point_index, (_row_index, _source_order, col_val, row_val) in enumerate(ordered_group):
            sampled_points.append((float(col_val), float(row_val)))
            if point_index >= len(ordered_group) - 1:
                continue
            next_col = float(ordered_group[point_index + 1][2])
            next_row = float(ordered_group[point_index + 1][3])
            delta_col = next_col - float(col_val)
            delta_row = next_row - float(row_val)
            step_count = max(
                1,
                int(math.ceil(math.hypot(delta_col, delta_row) / 0.5)),
            )
            for step_index in range(1, step_count):
                frac = float(step_index) / float(step_count)
                sampled_points.append(
                    (
                        float(col_val) + delta_col * frac,
                        float(row_val) + delta_row * frac,
                    )
                )

        for col_val, row_val in sampled_points:
            if not (np.isfinite(col_val) and np.isfinite(row_val)):
                continue
            for drow, dcol in offsets:
                row_idx = int(round(float(row_val) + float(drow)))
                col_idx = int(round(float(col_val) + float(dcol)))
                if row_idx < 0 or row_idx >= int(height) or col_idx < 0 or col_idx >= int(width):
                    continue
                linear_indices.add(int(row_idx) * int(width) + int(col_idx))

    if not linear_indices:
        result["fallback_reason"] = "empty_detector_roi"
        return result

    ordered_indices = np.asarray(sorted(linear_indices), dtype=np.int64)
    rows = (ordered_indices // int(width)).astype(np.int32, copy=False)
    cols = (ordered_indices % int(width)).astype(np.int32, copy=False)
    pixel_count = int(rows.size)
    fraction = float(pixel_count) / float(max(1, int(height) * int(width)))

    result["rows"] = rows
    result["cols"] = cols
    result["pixel_count"] = int(pixel_count)
    result["fraction"] = float(fraction)
    if fraction > max_detector_fraction:
        result["fallback_reason"] = "roi_too_large"
        return result

    result["valid"] = True
    return result


def read_runtime_geometry_fit_parameter_domains(
    *,
    parameter_specs: Mapping[str, object],
    image_size: object,
    fit_config: Mapping[str, object] | None,
    names: Sequence[str] | None = None,
    use_shared_theta_offset: bool = False,
) -> dict[str, tuple[float, float]]:
    """Read live parameter domains from slider ranges and image geometry."""

    selected_names = list(names) if names is not None else list(parameter_specs)
    domains: dict[str, tuple[float, float]] = {}
    try:
        image_size_value = float(image_size)
    except Exception:
        image_size_value = 0.0

    for name in selected_names:
        parameter_name = geometry_fit_constraint_parameter_name(
            str(name),
            use_shared_theta_offset=use_shared_theta_offset,
        )
        control_name = geometry_fit_constraint_source_name(parameter_name)

        if parameter_name == "center_x" or parameter_name == "center_y":
            domains[str(name)] = (0.0, max(image_size_value - 1.0, 0.0))
            continue

        spec = parameter_specs.get(control_name)
        if not isinstance(spec, Mapping):
            continue
        slider_widget = spec.get("value_slider")
        if slider_widget is None:
            continue
        try:
            lo = float(slider_widget.cget("from"))
            hi = float(slider_widget.cget("to"))
        except Exception:
            continue
        if parameter_name == "theta_offset":
            span = max(abs(lo), abs(hi), 1.0)
            domains[str(name)] = (-float(span), float(span))
            continue
        if lo > hi:
            lo, hi = hi, lo
        domains[str(name)] = (float(lo), float(hi))
    return domains


def default_runtime_geometry_fit_constraint_window(
    *,
    name: str,
    parameter_specs: Mapping[str, object],
    fit_config: Mapping[str, object] | None,
    parameter_domains: Mapping[str, tuple[float, float]] | None = None,
    current_theta_offset: object = 0.0,
    use_shared_theta_offset: bool = False,
) -> float:
    """Compute the default live constraint window for one geometry-fit row."""

    parameter_name = geometry_fit_constraint_parameter_name(
        name,
        use_shared_theta_offset=use_shared_theta_offset,
    )
    control_name = geometry_fit_constraint_source_name(parameter_name)
    spec = parameter_specs.get(control_name, {})
    if parameter_name == "theta_offset":
        try:
            current_value = float(current_theta_offset)
        except Exception:
            current_value = 0.0
    else:
        try:
            current_value = float(spec["value_var"].get())
        except Exception:
            current_value = 0.0
    try:
        step = abs(float(spec.get("step", 0.01)))
    except Exception:
        step = 0.01
    step = max(step, 1.0e-6)
    resolved_domains = parameter_domains or {}
    domain = resolved_domains.get(parameter_name)
    if domain is None:
        domain = resolved_domains.get(str(name))
    domain_span = 0.0
    if isinstance(domain, tuple) and len(domain) >= 2:
        domain_span = max(0.0, float(domain[1]) - float(domain[0]))

    bounds_cfg = _geometry_fit_config_section(fit_config, "bounds")
    entry = bounds_cfg.get(parameter_name)

    default_window = float("nan")
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            lo = float(entry[0])
            hi = float(entry[1])
            default_window = max(abs(current_value - lo), abs(hi - current_value))
        except Exception:
            default_window = float("nan")
    elif isinstance(entry, Mapping):
        mode = str(entry.get("mode", "absolute")).strip().lower()
        try:
            min_raw = float(entry.get("min")) if entry.get("min") is not None else float("nan")
        except Exception:
            min_raw = float("nan")
        try:
            max_raw = float(entry.get("max")) if entry.get("max") is not None else float("nan")
        except Exception:
            max_raw = float("nan")
        if mode in {"relative", "rel", "relative_min0", "rel_min0"}:
            candidates = [abs(v) for v in (min_raw, max_raw) if np.isfinite(v)]
            if candidates:
                default_window = max(candidates)
        else:
            candidates = [abs(current_value - v) for v in (min_raw, max_raw) if np.isfinite(v)]
            if candidates:
                default_window = max(candidates)

    if not np.isfinite(default_window) or default_window <= 0.0:
        default_window = max(
            step * 10.0,
            0.02 * domain_span,
            0.1 * max(abs(current_value), 1.0),
        )

    if domain_span > 0.0:
        default_window = min(default_window, domain_span)

    return max(float(default_window), step)


def default_runtime_geometry_fit_constraint_pull(
    *,
    name: str,
    fit_config: Mapping[str, object] | None,
    window: float,
    use_shared_theta_offset: bool = False,
) -> float:
    """Compute the default live constraint pull for one geometry-fit row."""

    parameter_name = geometry_fit_constraint_parameter_name(
        name,
        use_shared_theta_offset=use_shared_theta_offset,
    )
    priors_cfg = _geometry_fit_config_section(fit_config, "priors")
    entry = priors_cfg.get(parameter_name)
    if not isinstance(entry, Mapping):
        return 0.0
    try:
        sigma = float(entry.get("sigma"))
    except Exception:
        sigma = float("nan")
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(window) or window <= 0.0:
        return 0.0
    inferred = (1.0 - min(max(sigma / window, 0.05), 1.0)) / 0.95
    if not np.isfinite(inferred):
        return 0.0
    return min(max(float(inferred), 0.0), 1.0)


def build_runtime_geometry_fit_value_callbacks(
    bindings: GeometryFitRuntimeValueBindings,
) -> GeometryFitRuntimeValueCallbacks:
    """Build bound live geometry-fit value readers for the runtime."""

    var_map = {
        "zb": bindings.zb_var,
        "zs": bindings.zs_var,
        "theta_initial": bindings.theta_initial_var,
        "psi_z": bindings.psi_z_var,
        "chi": bindings.chi_var,
        "cor_angle": bindings.cor_angle_var,
        "gamma": bindings.gamma_var,
        "Gamma": bindings.Gamma_var,
        "corto_detector": bindings.corto_detector_var,
        "a": bindings.a_var,
        "c": bindings.c_var,
        "center_x": bindings.center_x_var,
        "center_y": bindings.center_y_var,
    }

    def _current_var_names() -> list[str]:
        return current_geometry_fit_var_names(
            fit_zb=bool(bindings.fit_zb_var.get()),
            fit_zs=bool(bindings.fit_zs_var.get()),
            fit_theta=bool(bindings.fit_theta_var.get()),
            fit_psi_z=bool(bindings.fit_psi_z_var.get()),
            fit_chi=bool(bindings.fit_chi_var.get()),
            fit_cor=bool(bindings.fit_cor_var.get()),
            fit_gamma=bool(bindings.fit_gamma_var.get()),
            fit_Gamma=bool(bindings.fit_Gamma_var.get()),
            fit_dist=bool(bindings.fit_dist_var.get()),
            fit_a=bool(bindings.fit_a_var.get()),
            fit_c=bool(bindings.fit_c_var.get()),
            fit_center_x=bool(bindings.fit_center_x_var.get()),
            fit_center_y=bool(bindings.fit_center_y_var.get()),
            use_shared_theta_offset=bool(bindings.geometry_fit_uses_shared_theta_offset()),
        )

    def _current_params() -> dict[str, object]:
        use_theta_offset = bool(bindings.geometry_fit_uses_shared_theta_offset())
        theta_offset_current = (
            float(bindings.current_geometry_theta_offset(strict=False)) if use_theta_offset else 0.0
        )
        current_background_index_value = (
            bindings.current_background_index()
            if callable(bindings.current_background_index)
            else bindings.current_background_index
        )
        current_background_index = int(current_background_index_value)
        theta_current = (
            bindings.background_theta_for_index(
                current_background_index,
                strict_count=False,
            )
            if use_theta_offset
            else bindings.theta_initial_var.get()
        )
        n2_value = bindings.n2() if callable(bindings.n2) else bindings.n2
        pixel_size_value = (
            bindings.pixel_size_value()
            if callable(bindings.pixel_size_value)
            else bindings.pixel_size_value
        )
        return {
            "a": bindings.a_var.get(),
            "c": bindings.c_var.get(),
            "lambda": bindings.lambda_value,
            "psi": bindings.psi,
            "psi_z": bindings.psi_z_var.get(),
            "zs": bindings.zs_var.get(),
            "zb": bindings.zb_var.get(),
            "sample_width_m": bindings.sample_width_var.get(),
            "sample_length_m": bindings.sample_length_var.get(),
            "sample_depth_m": bindings.sample_depth_var.get(),
            "chi": bindings.chi_var.get(),
            "n2": n2_value,
            "mosaic_params": dict(bindings.build_mosaic_params() or {}),
            "debye_x": bindings.debye_x_var.get(),
            "debye_y": bindings.debye_y_var.get(),
            "center": [bindings.center_x_var.get(), bindings.center_y_var.get()],
            "center_x": bindings.center_x_var.get(),
            "center_y": bindings.center_y_var.get(),
            "theta_initial": theta_current,
            "theta_offset": theta_offset_current,
            "uv1": np.array([1.0, 0.0, 0.0]),
            "uv2": np.array([0.0, 1.0, 0.0]),
            "corto_detector": bindings.corto_detector_var.get(),
            "gamma": bindings.gamma_var.get(),
            "Gamma": bindings.Gamma_var.get(),
            "cor_angle": bindings.cor_angle_var.get(),
            "optics_mode": bindings.current_optics_mode_flag(),
            "pixel_size": pixel_size_value,
            "pixel_size_m": pixel_size_value,
        }

    def _current_ui_params() -> dict[str, object]:
        theta_offset = None
        if bindings.geometry_theta_offset_var is not None:
            theta_offset = float(bindings.current_geometry_theta_offset(strict=False))
        return current_geometry_fit_ui_params(
            zb=float(bindings.zb_var.get()),
            zs=float(bindings.zs_var.get()),
            theta_initial=float(bindings.theta_initial_var.get()),
            psi_z=float(bindings.psi_z_var.get()),
            chi=float(bindings.chi_var.get()),
            cor_angle=float(bindings.cor_angle_var.get()),
            gamma=float(bindings.gamma_var.get()),
            Gamma=float(bindings.Gamma_var.get()),
            corto_detector=float(bindings.corto_detector_var.get()),
            a=float(bindings.a_var.get()),
            c=float(bindings.c_var.get()),
            center_x=float(bindings.center_x_var.get()),
            center_y=float(bindings.center_y_var.get()),
            theta_offset=theta_offset,
        )

    return GeometryFitRuntimeValueCallbacks(
        current_var_names=_current_var_names,
        current_params=_current_params,
        current_ui_params=_current_ui_params,
        var_map=var_map,
        build_mosaic_params=bindings.build_mosaic_params,
    )


def build_runtime_geometry_fit_config_factory(
    *,
    base_config: Mapping[str, object] | None,
    current_constraint_state: Callable[[Sequence[str] | None], Mapping[str, object]],
    current_parameter_domains: Callable[[Sequence[str] | None], Mapping[str, object]],
    current_candidate_param_names: Callable[[], Sequence[str]] | None = None,
    current_caked_roi_enabled: Callable[[], object] | None = None,
) -> Callable[[Sequence[str], Mapping[str, object]], dict[str, object]]:
    """Build the live geometry-fit refinement-config factory from runtime readers."""

    def _build(
        var_names: Sequence[str],
        fit_params: Mapping[str, object],
    ) -> dict[str, object]:
        selected_names = [str(name) for name in var_names]
        if callable(current_candidate_param_names):
            candidate_names = [str(name) for name in current_candidate_param_names() or ()]
        else:
            candidate_names = []
        if not candidate_names:
            candidate_names = list(selected_names)
        candidate_names = list(dict.fromkeys(candidate_names))
        current_params = {name: fit_params.get(name) for name in candidate_names}
        caked_roi_enabled = None
        if callable(current_caked_roi_enabled):
            try:
                caked_roi_enabled = current_caked_roi_enabled()
            except Exception:
                caked_roi_enabled = None
        return build_geometry_fit_runtime_config(
            base_config,
            current_params,
            {},
            current_parameter_domains(candidate_names),
            candidate_param_names=candidate_names,
            caked_roi_enabled=caked_roi_enabled,
        )

    return _build


def build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
    *,
    candidate_param_names: Sequence[str] | None = None,
    caked_roi_enabled: object | None = None,
):
    runtime_cfg = copy.deepcopy(base_config) if isinstance(base_config, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    # GUI/headless geometry-fit runs always keep the unsafe runtime path off,
    # but the safe-wrapper Numba path can still stay enabled.
    optimizer_cfg_raw = runtime_cfg.get("optimizer", runtime_cfg.get("solver", {})) or {}
    optimizer_cfg = dict(optimizer_cfg_raw) if isinstance(optimizer_cfg_raw, Mapping) else {}
    runtime_cfg["optimizer"] = optimizer_cfg
    runtime_cfg["solver"] = optimizer_cfg

    gui_use_numba = runtime_cfg.pop("gui_use_numba", None)
    runtime_cfg.pop("gui_allow_unsafe_runtime", None)
    runtime_cfg["use_numba"] = bool(
        gui_use_numba if gui_use_numba is not None else runtime_cfg.get("use_numba", False)
    )
    runtime_cfg["allow_unsafe_runtime"] = False

    gui_workers = optimizer_cfg.pop("gui_workers", None)
    if gui_workers is None:
        gui_workers = optimizer_cfg.get("workers", "auto")
    optimizer_cfg["workers"] = gui_workers if gui_workers is not None else "auto"

    gui_parallel_mode = optimizer_cfg.pop("gui_parallel_mode", None)
    if gui_parallel_mode is None:
        gui_parallel_mode = optimizer_cfg.get("parallel_mode", "auto")
    optimizer_cfg["parallel_mode"] = (
        str(gui_parallel_mode).strip() if gui_parallel_mode is not None else "auto"
    )

    gui_worker_numba_threads = optimizer_cfg.pop("gui_worker_numba_threads", None)
    if gui_worker_numba_threads is None:
        gui_worker_numba_threads = optimizer_cfg.get("worker_numba_threads", 0)
    optimizer_cfg["worker_numba_threads"] = (
        gui_worker_numba_threads if gui_worker_numba_threads is not None else 0
    )

    bounds_cfg = runtime_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}
    runtime_cfg["bounds"] = bounds_cfg

    priors_cfg = runtime_cfg.get("priors", {}) or {}
    if not isinstance(priors_cfg, dict):
        priors_cfg = {}
    runtime_cfg["priors"] = priors_cfg

    active_names = [
        str(name)
        for name in (
            candidate_param_names
            if candidate_param_names is not None
            else list(current_params or {})
        )
    ]
    active_names = list(dict.fromkeys(active_names))

    # GUI geometry fits now expose the full live parameter domains instead of
    # constraining each variable to a current-value window or soft prior.
    for name in active_names:
        priors_cfg.pop(str(name), None)
        domain = (parameter_domains or {}).get(name)
        if not isinstance(domain, (list, tuple)) or len(domain) < 2:
            bounds_cfg.pop(str(name), None)
            continue
        try:
            lo = float(domain[0])
            hi = float(domain[1])
        except Exception:
            bounds_cfg.pop(str(name), None)
            continue
        if not np.isfinite(lo) or not np.isfinite(hi):
            bounds_cfg.pop(str(name), None)
            continue
        if lo > hi:
            lo, hi = hi, lo
        bounds_cfg[str(name)] = [float(lo), float(hi)]

    runtime_cfg["candidate_param_names"] = active_names
    runtime_cfg["caked_roi"] = read_geometry_fit_caked_roi_config(
        runtime_cfg,
        enabled_override=caked_roi_enabled,
    )

    return runtime_cfg


def geometry_fit_runtime_fit_sample_count(
    geometry_runtime_cfg: Mapping[str, object] | None,
) -> int | None:
    """Return the fit-only sample count when this run requests one."""

    cfg = geometry_runtime_cfg if isinstance(geometry_runtime_cfg, Mapping) else {}
    sampling_cfg_raw = cfg.get("sampling", {})
    sampling_cfg = sampling_cfg_raw if isinstance(sampling_cfg_raw, Mapping) else {}
    raw_value = sampling_cfg.get("fit_sample_count")
    if raw_value is None:
        return None
    try:
        resolved = int(raw_value)
    except Exception:
        return None
    return max(int(resolved), 1)


def build_geometry_fit_solver_mosaic_params(
    *,
    params: Mapping[str, object] | None,
    geometry_runtime_cfg: Mapping[str, object] | None,
    build_mosaic_params: Callable[..., Mapping[str, object] | None] | None = None,
) -> tuple[dict[str, object], int | None]:
    """Return the fit-only mosaic params used by the solver, if overridden."""

    resolved_params = params if isinstance(params, Mapping) else {}
    mosaic_params = dict(resolved_params.get("mosaic_params", {}) or {})
    fit_sample_count = geometry_fit_runtime_fit_sample_count(geometry_runtime_cfg)
    if fit_sample_count is None:
        return mosaic_params, None
    if callable(build_mosaic_params):
        rebuilt: Mapping[str, object] | None = None
        try:
            rebuilt = build_mosaic_params(sample_count=int(fit_sample_count))
        except TypeError:
            rebuilt = build_mosaic_params()
        except Exception:
            rebuilt = None
        if isinstance(rebuilt, Mapping):
            mosaic_params = dict(rebuilt)
    return mosaic_params, int(fit_sample_count)


def apply_geometry_fit_undo_state(
    state: dict[str, object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
):
    """Apply saved UI values and return copied cache/overlay state."""

    if not isinstance(state, dict):
        return {
            "profile_cache": {},
            "overlay_state": None,
        }

    ui_params = state.get("ui_params", {}) or {}
    for name, var in var_map.items():
        try:
            value = float(ui_params.get(name))
        except Exception:
            continue
        if np.isfinite(value):
            var.set(value)

    if geometry_theta_offset_var is not None:
        try:
            theta_offset = float(ui_params.get("theta_offset", 0.0))
        except Exception:
            theta_offset = 0.0
        if np.isfinite(theta_offset):
            geometry_theta_offset_var.set(f"{theta_offset:.6g}")

    overlay_state = copy_geometry_fit_state_value(state.get("overlay_state"))
    if not overlay_state:
        overlay_state = None

    return {
        "profile_cache": copy_geometry_fit_state_value(state.get("profile_cache", {})) or {},
        "overlay_state": overlay_state,
    }


def set_runtime_geometry_fit_history_button_state(
    *,
    can_undo: bool,
    can_redo: bool,
    set_button_state: Callable[[bool, bool], None] | None = None,
) -> None:
    """Apply the current geometry-fit undo/redo availability to the UI."""

    if callable(set_button_state):
        set_button_state(bool(can_undo), bool(can_redo))


def _resolve_runtime_value(value_or_factory):
    """Return one runtime value, calling it first when it is a factory."""

    if callable(value_or_factory):
        return value_or_factory()
    return value_or_factory


def _geometry_fit_sequence_has_items(value: object) -> bool:
    """Return whether one saved sequence-like payload contains any entries."""

    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return int(np.asarray(value).size) > 0
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except Exception:
        return True


def _geometry_fit_sequence_list(value: object) -> list[object]:
    """Normalize one saved sequence-like payload into a plain list."""

    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            item = arr.item()
            if item is None:
                return []
            if isinstance(item, list):
                return list(item)
            if isinstance(item, tuple):
                return list(item)
            return [item]
        return list(arr.tolist())
    try:
        return list(value)  # type: ignore[arg-type]
    except Exception:
        return [value]


def redraw_runtime_geometry_fit_overlay_state(
    overlay_state: Mapping[str, object] | None,
    *,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
) -> bool:
    """Redraw one saved geometry-fit overlay payload when it contains display data."""

    overlay_payload = dict(overlay_state) if isinstance(overlay_state, Mapping) else {}
    overlay_records = _geometry_fit_sequence_list(overlay_payload.get("overlay_records"))
    initial_pairs_display = _geometry_fit_sequence_list(
        overlay_payload.get("initial_pairs_display")
    )
    try:
        max_display_markers = int(overlay_payload.get("max_display_markers", 120))
    except Exception:
        max_display_markers = 120
    max_display_markers = max(1, max_display_markers)

    if overlay_records and callable(draw_overlay_records):
        draw_overlay_records(overlay_records, int(max_display_markers))
        return True
    if initial_pairs_display and callable(draw_initial_pairs_overlay):
        draw_initial_pairs_overlay(initial_pairs_display, int(max_display_markers))
        return True
    return False


def capture_runtime_geometry_fit_undo_state(
    *,
    current_ui_params: Callable[[], Mapping[str, object]] | Mapping[str, object],
    current_profile_cache: Callable[[], object] | object,
    copy_state_value: Callable[[object], object],
    last_overlay_state: Callable[[], Mapping[str, object] | None] | Mapping[str, object] | None,
    build_initial_pairs_display: Callable[
        ..., tuple[Sequence[dict[str, object]], Sequence[dict[str, object]]]
    ],
    current_background_index: Callable[[], object] | object,
    current_fit_params: Callable[[], Mapping[str, object]] | Mapping[str, object],
    pending_pairs_display: Callable[[], Sequence[dict[str, object]]] | Sequence[dict[str, object]],
) -> dict[str, object]:
    """Capture the current geometry-fit UI/profile/overlay state for undo."""

    overlay_state = copy_state_value(_resolve_runtime_value(last_overlay_state))
    overlay_records = (
        overlay_state.get("overlay_records") if isinstance(overlay_state, dict) else None
    )
    initial_pairs_display = (
        overlay_state.get("initial_pairs_display") if isinstance(overlay_state, dict) else None
    )
    if not (
        isinstance(overlay_state, dict)
        and (
            _geometry_fit_sequence_has_items(overlay_records)
            or _geometry_fit_sequence_has_items(initial_pairs_display)
        )
    ):
        try:
            _, initial_pairs_display = build_initial_pairs_display(
                int(_resolve_runtime_value(current_background_index)),
                param_set=_resolve_runtime_value(current_fit_params),
                prefer_cache=True,
            )
            pending_display = _resolve_runtime_value(pending_pairs_display)
            combined_pairs_display = list(initial_pairs_display) + list(pending_display)
            if combined_pairs_display:
                overlay_state = {
                    "overlay_records": [],
                    "initial_pairs_display": copy_state_value(combined_pairs_display),
                    "max_display_markers": max(1, len(combined_pairs_display)),
                }
        except Exception:
            pass

    return {
        "ui_params": copy_state_value(_resolve_runtime_value(current_ui_params)),
        "profile_cache": copy_state_value(_resolve_runtime_value(current_profile_cache)),
        "overlay_state": overlay_state,
    }


def restore_runtime_geometry_fit_undo_state(
    state: dict[str, object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
    replace_profile_cache: Callable[[dict[str, object]], None],
    set_last_overlay_state: Callable[[dict[str, object] | None], object],
    request_preview_skip_once: Callable[[], None] | None = None,
    mark_last_simulation_dirty: Callable[[], None] | None = None,
    cancel_pending_update: Callable[[], None] | None = None,
    run_update: Callable[[], None] | None = None,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    update_manual_pick_button_label: Callable[[], None] | None = None,
    apply_undo_state: Callable[..., Mapping[str, object]] = apply_geometry_fit_undo_state,
) -> Mapping[str, object]:
    """Apply one saved geometry-fit history state back onto the live runtime."""

    ui_params_raw = state.get("ui_params", {}) if isinstance(state, dict) else {}
    ui_params = dict(ui_params_raw) if isinstance(ui_params_raw, Mapping) else {}
    overlay_state_raw = (
        copy_geometry_fit_state_value(state.get("overlay_state"))
        if isinstance(state, dict)
        else None
    )
    overlay_state = dict(overlay_state_raw) if isinstance(overlay_state_raw, Mapping) else None
    profile_cache_raw = (
        copy_geometry_fit_state_value(state.get("profile_cache", {}))
        if isinstance(state, dict)
        else {}
    )
    profile_cache = dict(profile_cache_raw) if isinstance(profile_cache_raw, Mapping) else {}
    restored = apply_undo_state(
        {
            **(dict(state) if isinstance(state, dict) else {}),
            "ui_params": ui_params,
            "overlay_state": overlay_state,
            "profile_cache": profile_cache,
        },
        var_map=var_map,
        geometry_theta_offset_var=geometry_theta_offset_var,
    )
    restored_profile_cache = restored.get("profile_cache", {})
    replace_profile_cache(
        dict(restored_profile_cache) if isinstance(restored_profile_cache, Mapping) else {}
    )
    overlay_state = restored.get("overlay_state")
    set_last_overlay_state(dict(overlay_state) if isinstance(overlay_state, dict) else None)

    if callable(request_preview_skip_once):
        request_preview_skip_once()
    if callable(mark_last_simulation_dirty):
        mark_last_simulation_dirty()
    if callable(cancel_pending_update):
        cancel_pending_update()
    if callable(run_update):
        run_update()

    redraw_runtime_geometry_fit_overlay_state(
        overlay_state if isinstance(overlay_state, Mapping) else None,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
    )

    if callable(refresh_status):
        refresh_status()
    if callable(update_manual_pick_button_label):
        update_manual_pick_button_label()
    return restored


def build_runtime_geometry_fit_undo_restore_callback(
    *,
    var_map_factory: Mapping[str, object] | Callable[[], Mapping[str, object]],
    geometry_theta_offset_var_factory: object | Callable[[], object] | None = None,
    replace_profile_cache: Callable[[dict[str, object]], None],
    set_last_overlay_state: Callable[[dict[str, object] | None], object],
    request_preview_skip_once: Callable[[], None] | None = None,
    mark_last_simulation_dirty: Callable[[], None] | None = None,
    cancel_pending_update: Callable[[], None] | None = None,
    run_update: Callable[[], None] | None = None,
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    update_manual_pick_button_label: Callable[[], None] | None = None,
    apply_undo_state: Callable[..., Mapping[str, object]] = apply_geometry_fit_undo_state,
) -> Callable[[dict[str, object]], Mapping[str, object] | None]:
    """Build one undo-restore callback that resolves live runtime hooks lazily."""

    def _restore(state: dict[str, object]) -> Mapping[str, object] | None:
        if not isinstance(state, dict):
            return None
        return restore_runtime_geometry_fit_undo_state(
            state,
            var_map=_resolve_runtime_value(var_map_factory),
            geometry_theta_offset_var=_resolve_runtime_value(geometry_theta_offset_var_factory),
            replace_profile_cache=replace_profile_cache,
            set_last_overlay_state=set_last_overlay_state,
            request_preview_skip_once=request_preview_skip_once,
            mark_last_simulation_dirty=mark_last_simulation_dirty,
            cancel_pending_update=cancel_pending_update,
            run_update=run_update,
            draw_overlay_records=draw_overlay_records,
            draw_initial_pairs_overlay=draw_initial_pairs_overlay,
            refresh_status=refresh_status,
            update_manual_pick_button_label=update_manual_pick_button_label,
            apply_undo_state=apply_undo_state,
        )

    return _restore


def _run_runtime_geometry_fit_history_transition(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_transition: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
    empty_text: str,
    failure_prefix: str,
    success_text: str,
) -> bool:
    """Run one geometry-fit history transition against the live runtime."""

    history_available = bool(has_history()) if callable(has_history) else bool(has_history)
    if not history_available:
        if callable(set_progress_text):
            set_progress_text(empty_text)
        return False

    current_state = capture_current_state()
    state = read_state()
    if not isinstance(state, dict):
        if callable(set_progress_text):
            set_progress_text(empty_text)
        return False

    try:
        restore_state(state)
    except Exception as exc:
        if callable(set_progress_text):
            set_progress_text(f"{failure_prefix}: {exc}")
        return False

    commit_transition(current_state)
    if callable(update_button_state):
        update_button_state()
    if callable(set_progress_text):
        set_progress_text(success_text)
    return True


def undo_runtime_geometry_fit(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_undo_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_undo: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Restore the previous geometry-fit history state."""

    return _run_runtime_geometry_fit_history_transition(
        has_history=has_history,
        capture_current_state=capture_current_state,
        read_state=read_undo_state,
        restore_state=restore_state,
        commit_transition=commit_undo,
        update_button_state=update_button_state,
        set_progress_text=set_progress_text,
        empty_text="No geometry fit history available to undo.",
        failure_prefix="Failed to undo geometry fit",
        success_text="Restored the previous geometry-fit state.",
    )


def redo_runtime_geometry_fit(
    *,
    has_history: Callable[[], bool] | bool,
    capture_current_state: Callable[[], dict[str, object]],
    read_redo_state: Callable[[], dict[str, object] | None],
    restore_state: Callable[[dict[str, object]], object],
    commit_redo: Callable[[dict[str, object]], None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> bool:
    """Reapply the next geometry-fit history state."""

    return _run_runtime_geometry_fit_history_transition(
        has_history=has_history,
        capture_current_state=capture_current_state,
        read_state=read_redo_state,
        restore_state=restore_state,
        commit_transition=commit_redo,
        update_button_state=update_button_state,
        set_progress_text=set_progress_text,
        empty_text="No geometry fit history available to redo.",
        failure_prefix="Failed to redo geometry fit",
        success_text="Reapplied the next geometry-fit state.",
    )


def build_runtime_geometry_fit_history_callbacks(
    *,
    history_state: Any,
    capture_current_state: Callable[[], dict[str, object]],
    restore_state: Callable[[dict[str, object]], object],
    copy_state_value: Callable[[object], object],
    history_limit: Callable[[], object] | object,
    peek_last_undo_state: Callable[..., dict[str, object] | None],
    peek_last_redo_state: Callable[..., dict[str, object] | None],
    commit_undo: Callable[..., None],
    commit_redo: Callable[..., None],
    update_button_state: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> GeometryFitRuntimeHistoryCallbacks:
    """Build shared runtime undo/redo callbacks around one geometry-fit history store."""

    def _undo() -> bool:
        return undo_runtime_geometry_fit(
            has_history=lambda: bool(getattr(history_state, "undo_stack", [])),
            capture_current_state=capture_current_state,
            read_undo_state=(
                lambda: peek_last_undo_state(
                    history_state,
                    copy_state_value=copy_state_value,
                )
            ),
            restore_state=restore_state,
            commit_undo=(
                lambda current_state: commit_undo(
                    history_state,
                    current_state,
                    copy_state_value=copy_state_value,
                    limit=int(_resolve_runtime_value(history_limit)),
                )
            ),
            update_button_state=update_button_state,
            set_progress_text=set_progress_text,
        )

    def _redo() -> bool:
        return redo_runtime_geometry_fit(
            has_history=lambda: bool(getattr(history_state, "redo_stack", [])),
            capture_current_state=capture_current_state,
            read_redo_state=(
                lambda: peek_last_redo_state(
                    history_state,
                    copy_state_value=copy_state_value,
                )
            ),
            restore_state=restore_state,
            commit_redo=(
                lambda current_state: commit_redo(
                    history_state,
                    current_state,
                    copy_state_value=copy_state_value,
                    limit=int(_resolve_runtime_value(history_limit)),
                )
            ),
            update_button_state=update_button_state,
            set_progress_text=set_progress_text,
        )

    return GeometryFitRuntimeHistoryCallbacks(
        undo=_undo,
        redo=_redo,
    )


def make_runtime_geometry_tool_action_callbacks(
    *,
    geometry_fit_history_state: Any,
    manual_pick_armed: Callable[[], bool] | bool,
    set_manual_pick_armed: Callable[[bool], None],
    current_background_index: Callable[[], object] | object,
    current_pick_session: Callable[[], object] | object,
    manual_pick_session_active: Callable[[], bool] | bool,
    build_manual_pick_button_label: Callable[..., str],
    pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    pair_group_count: Callable[[int], int],
    set_manual_pick_text: Callable[[str], None] | None = None,
    set_history_button_state: Callable[[bool, bool], None] | None = None,
    show_caked_2d_var: Any = None,
    toggle_caked_2d: Callable[[], None] | None = None,
    ensure_geometry_fit_caked_view: Callable[[], None] | None = None,
    set_hkl_pick_mode: Callable[..., None] | None = None,
    set_geometry_preview_exclude_mode: Callable[..., None] | None = None,
    cancel_manual_pick_session: Callable[..., None] | None = None,
    canvas_widget: Callable[[], Any] | Any = None,
    push_manual_undo_state: Callable[[], None] | None = None,
    clear_pairs_for_current_background: Callable[[int], None] | None = None,
    clear_geometry_pick_artists: Callable[[], None] | None = None,
    refresh_status: Callable[[], None] | None = None,
    set_progress_text: Callable[[str], None] | None = None,
) -> GeometryToolActionRuntimeCallbacks:
    """Build the live geometry tool action callbacks around shared helpers."""

    def _current_background() -> int:
        return int(_resolve_runtime_value(current_background_index))

    def _manual_pick_armed() -> bool:
        return bool(_resolve_runtime_value(manual_pick_armed))

    def _manual_pick_session_is_active() -> bool:
        active = (
            manual_pick_session_active()
            if callable(manual_pick_session_active)
            else manual_pick_session_active
        )
        return bool(active)

    def _update_fit_history_button_state() -> None:
        set_runtime_geometry_fit_history_button_state(
            can_undo=bool(getattr(geometry_fit_history_state, "undo_stack", [])),
            can_redo=bool(getattr(geometry_fit_history_state, "redo_stack", [])),
            set_button_state=set_history_button_state,
        )

    def _update_manual_pick_button_label() -> None:
        label = build_manual_pick_button_label(
            armed=_manual_pick_armed(),
            current_background_index=_current_background(),
            pick_session=_resolve_runtime_value(current_pick_session),
            pairs_for_index=pairs_for_index,
            pair_group_count=pair_group_count,
        )
        if callable(set_manual_pick_text):
            set_manual_pick_text(str(label))

    def _set_manual_pick_mode(enabled: bool, message: str | None = None) -> None:
        armed = bool(enabled)
        set_manual_pick_armed(armed)

        if armed:
            if callable(set_hkl_pick_mode):
                set_hkl_pick_mode(False)
            if callable(set_geometry_preview_exclude_mode):
                set_geometry_preview_exclude_mode(False)
        elif callable(cancel_manual_pick_session):
            cancel_manual_pick_session(restore_view=True, redraw=True)

        _update_manual_pick_button_label()

        widget = _resolve_runtime_value(canvas_widget)
        configure = getattr(widget, "configure", None)
        if callable(configure):
            try:
                configure(cursor="crosshair" if armed else "")
            except Exception:
                pass

        if message and callable(set_progress_text):
            set_progress_text(message)

    def _toggle_manual_pick_mode() -> None:
        armed = _manual_pick_armed()
        _set_manual_pick_mode(
            not armed,
            message=(
                (
                    "Manual geometry picking armed. "
                    "Click a Qr/Qz set once, then click the matching background peaks "
                    "for each simulated member of that set."
                )
                if not armed
                else "Manual geometry picking disabled."
            ),
        )

    def _clear_current_manual_pairs() -> None:
        background_index = _current_background()
        if pairs_for_index(background_index) or _manual_pick_session_is_active():
            if callable(push_manual_undo_state):
                push_manual_undo_state()
        if callable(cancel_manual_pick_session):
            cancel_manual_pick_session(restore_view=True, redraw=False)
        if callable(clear_pairs_for_current_background):
            clear_pairs_for_current_background(background_index)
        if callable(clear_geometry_pick_artists):
            clear_geometry_pick_artists()
        _update_manual_pick_button_label()
        if callable(refresh_status):
            refresh_status()
        if callable(set_progress_text):
            set_progress_text("Cleared saved geometry pairs for the current background image.")

    return GeometryToolActionRuntimeCallbacks(
        update_fit_history_button_state=_update_fit_history_button_state,
        update_manual_pick_button_label=_update_manual_pick_button_label,
        set_manual_pick_mode=_set_manual_pick_mode,
        toggle_manual_pick_mode=_toggle_manual_pick_mode,
        clear_current_manual_pairs=_clear_current_manual_pairs,
    )


def geometry_manual_pair_enabled_for_geometry_fit(entry: object) -> bool:
    """Return whether a saved manual pair should contribute to geometry solving."""

    if not isinstance(entry, Mapping):
        return False
    if gui_manual_geometry.geometry_manual_entry_is_background_qr_reference(entry):
        return False
    return not bool(entry.get("geometry_fit_disabled", False))


def build_geometry_manual_fit_dataset(
    background_index: int,
    *,
    theta_base: float,
    base_fit_params: Mapping[str, object] | None,
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    orientation_cfg: Mapping[str, object] | None = None,
    stage_callback: GeometryFitStageCallback | None = None,
) -> dict[str, object]:
    """Build one saved-manual-pair geometry dataset for the optimizer."""

    background_idx = int(background_index)
    _emit_geometry_fit_stage_event(
        stage_callback,
        "dataset_start",
        background_index=int(background_idx),
        message=f"preflight: building dataset for background {int(background_idx) + 1}",
    )

    def _finite_float(value: object) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        if not np.isfinite(out):
            return None
        return float(out)

    def _caked_angle_pair(
        entry: Mapping[str, object] | None,
        *,
        x_keys: Sequence[str],
        y_keys: Sequence[str],
    ) -> tuple[float, float] | None:
        if not isinstance(entry, Mapping):
            return None
        two_theta_value: float | None = None
        phi_value: float | None = None
        for key in x_keys:
            two_theta_value = _finite_float(entry.get(key))
            if two_theta_value is not None:
                break
        for key in y_keys:
            phi_value = _finite_float(entry.get(key))
            if phi_value is not None:
                break
        if two_theta_value is None or phi_value is None:
            return None
        return float(two_theta_value), float(phi_value)

    def _reference_two_theta_deg(
        entry: Mapping[str, object] | None,
        *,
        a_lattice: float | None,
        c_lattice: float | None,
        wavelength: float | None,
    ) -> float | None:
        if not isinstance(entry, Mapping):
            return None
        hkl = entry.get("hkl")
        if (
            isinstance(hkl, (list, tuple, np.ndarray))
            and len(hkl) >= 3
            and a_lattice is not None
            and c_lattice is not None
            and wavelength is not None
        ):
            try:
                spacing = d_spacing(
                    int(hkl[0]),
                    int(hkl[1]),
                    int(hkl[2]),
                    float(a_lattice),
                    float(c_lattice),
                )
                if spacing is not None:
                    value = two_theta(float(spacing), float(wavelength))
                    finite_value = _finite_float(value)
                    if finite_value is not None:
                        return float(finite_value)
            except Exception:
                pass

        qr_value = _finite_float(entry.get("qr"))
        qz_value = _finite_float(entry.get("qz"))
        if qr_value is None or qz_value is None or wavelength is None:
            return None
        q_mag = math.hypot(float(qr_value), float(qz_value))
        if not (np.isfinite(q_mag) and q_mag > 0.0):
            return None
        arg = float(q_mag) * float(wavelength) / (4.0 * np.pi)
        if not np.isfinite(arg) or abs(arg) > 1.0:
            return None
        return float(2.0 * np.degrees(np.arcsin(arg)))

    use_caked_display = False
    if callable(manual_dataset_bindings.pick_uses_caked_space):
        try:
            use_caked_display = bool(manual_dataset_bindings.pick_uses_caked_space())
        except Exception:
            use_caked_display = False

    def _install_provider_simulated_point(
        initial_entry: dict[str, object],
        point: Sequence[object] | None,
        frame: object,
        point_source: object,
        *,
        overwrite_existing: bool = False,
    ) -> None:
        point_list = _geometry_fit_point_list(point)
        point_frame = _geometry_fit_normalize_point_frame(frame)
        if point_list is None or point_frame == "unknown":
            return
        source_text = str(point_source or "")
        picker_owned = source_text in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
        initial_entry["provider_simulated_frame"] = point_frame
        initial_entry["provider_simulated_point_source"] = source_text
        if point_frame == "display":
            if overwrite_existing or "sim_display" not in initial_entry:
                initial_entry["sim_display"] = (
                    float(point_list[0]),
                    float(point_list[1]),
                )
            if use_caked_display and (
                overwrite_existing or "sim_caked_display" not in initial_entry
            ):
                initial_entry["sim_caked_display"] = (
                    float(point_list[0]),
                    float(point_list[1]),
                )
            if picker_owned and (overwrite_existing or "sim_native" not in initial_entry):
                sim_native = None
                try:
                    sim_native = manual_dataset_bindings.display_to_native_sim_coords(
                        float(point_list[0]),
                        float(point_list[1]),
                        (
                            int(manual_dataset_bindings.image_size),
                            int(manual_dataset_bindings.image_size),
                        ),
                    )
                except Exception:
                    sim_native = None
                if isinstance(sim_native, (list, tuple, np.ndarray)) and len(sim_native) >= 2:
                    try:
                        sim_native_point = (
                            float(sim_native[0]),
                            float(sim_native[1]),
                        )
                    except Exception:
                        sim_native_point = None
                    if (
                        sim_native_point is not None
                        and np.isfinite(sim_native_point[0])
                        and np.isfinite(sim_native_point[1])
                    ):
                        initial_entry["sim_native"] = sim_native_point
                        initial_entry["sim_native_source"] = (
                            f"display_to_native_sim_coords({source_text or 'sim_display'})"
                        )
        elif point_frame == "caked_2theta_phi":
            initial_entry["simulated_two_theta_deg"] = float(point_list[0])
            initial_entry["simulated_phi_deg"] = float(point_list[1])
            if use_caked_display:
                initial_entry["sim_display"] = (
                    float(point_list[0]),
                    float(point_list[1]),
                )
                initial_entry["sim_caked_display"] = (
                    float(point_list[0]),
                    float(point_list[1]),
                )
            if picker_owned:
                initial_entry.pop("sim_native", None)
                initial_entry.pop("sim_native_source", None)
        elif point_frame == "detector_native":
            initial_entry["sim_native"] = (
                float(point_list[0]),
                float(point_list[1]),
            )
            initial_entry["sim_native_source"] = str(source_text or "provider_detector_native")

    raw_selected_entries = [
        entry
        for entry in (manual_dataset_bindings.geometry_manual_pairs_for_index(background_idx) or ())
        if geometry_manual_pair_enabled_for_geometry_fit(entry)
    ]
    if not raw_selected_entries:
        raise RuntimeError(f"background {background_idx + 1} has no saved manual geometry pairs")

    native_background, display_background = manual_dataset_bindings.load_background_by_index(
        background_idx
    )
    (
        native_to_display_callback,
        native_to_display_unavailable_reason,
        native_to_display_source,
    ) = _geometry_fit_dataset_native_to_display_callback(
        manual_dataset_bindings,
        background_idx,
    )
    selected_entry_inputs: list[dict[str, object]] = []
    if callable(manual_dataset_bindings.geometry_manual_refresh_pair_entry):
        for raw_entry in raw_selected_entries:
            raw_saved_entry = dict(raw_entry) if isinstance(raw_entry, Mapping) else None
            refreshed = manual_dataset_bindings.geometry_manual_refresh_pair_entry(raw_entry)
            normalized_entry = (
                dict(refreshed)
                if isinstance(refreshed, Mapping)
                else (dict(raw_entry) if isinstance(raw_entry, Mapping) else None)
            )
            if normalized_entry is None:
                continue
            background_detector_x = _finite_float(normalized_entry.get("background_detector_x"))
            background_detector_y = _finite_float(normalized_entry.get("background_detector_y"))
            if background_detector_x is not None and background_detector_y is not None:
                normalized_entry.setdefault(
                    "background_detector_frame_provenance",
                    "geometry_manual_refresh_pair_entry",
                )
                normalized_entry.setdefault(
                    "background_detector_input_frame",
                    "native_detector",
                )
            selected_entry_inputs.append(
                {
                    "raw_saved_entry": (
                        raw_saved_entry
                        if isinstance(raw_saved_entry, dict)
                        else dict(normalized_entry)
                    ),
                    "entry": dict(normalized_entry),
                }
            )
    else:
        for raw_entry in raw_selected_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            selected_entry_inputs.append(
                {
                    "raw_saved_entry": dict(raw_entry),
                    "entry": dict(raw_entry),
                }
            )
    selected_entries = [
        {
            **dict(item["entry"]),
            "source_label": _geometry_fit_entry_source_label(
                item["entry"] if isinstance(item.get("entry"), Mapping) else None
            ),
        }
        for item in selected_entry_inputs
        if isinstance(item.get("entry"), Mapping)
    ]
    manual_picker_truth_pairs = gui_manual_geometry.build_geometry_manual_picker_truth_pairs(
        background_idx,
        [
            dict(item["raw_saved_entry"])
            for item in selected_entry_inputs
            if isinstance(item.get("raw_saved_entry"), Mapping)
        ],
        refresh_pair_entry=manual_dataset_bindings.geometry_manual_refresh_pair_entry,
    )
    manual_picker_truth_by_order = _geometry_fit_truth_by_order_key(manual_picker_truth_pairs)

    baseline_fit_params_i = dict(base_fit_params or {})
    params_i = dict(base_fit_params or {})
    theta_offset = float(params_i.get("theta_offset", 0.0))
    params_i["theta_initial"] = float(theta_base + theta_offset)
    reference_a = _finite_float(params_i.get("a"))
    reference_c = _finite_float(params_i.get("c"))
    reference_lambda = _finite_float(params_i.get("lambda"))

    def _project_source_rows_for_current_view(
        rows: object,
    ) -> list[dict[str, object]]:
        normalized_rows = [dict(entry) for entry in (rows or ()) if isinstance(entry, Mapping)]
        if not normalized_rows:
            return normalized_rows
        caked_projection_required = bool(
            use_caked_display or geometry_manual_pairs_use_caked_fit_space(selected_entries)
        )

        per_background_projector = (
            manual_dataset_bindings.geometry_manual_project_peaks_for_background_view
        )
        if callable(per_background_projector):
            order_key = "__ra_sim_geometry_fit_projection_order__"
            grouped_rows: dict[int, list[dict[str, object]]] = {}
            ordered_backgrounds: list[int] = []
            for position, raw_entry in enumerate(normalized_rows):
                entry = dict(raw_entry)
                try:
                    row_background_idx = int(entry.get("background_index", background_idx))
                except Exception:
                    row_background_idx = int(background_idx)
                entry.setdefault("background_index", int(row_background_idx))
                entry[order_key] = int(position)
                if int(row_background_idx) not in grouped_rows:
                    ordered_backgrounds.append(int(row_background_idx))
                    grouped_rows[int(row_background_idx)] = []
                grouped_rows[int(row_background_idx)].append(entry)
            projected_rows = []
            for row_background_idx in ordered_backgrounds:
                try:
                    projected_rows.extend(
                        dict(entry)
                        for entry in (
                            per_background_projector(
                                int(row_background_idx),
                                grouped_rows[int(row_background_idx)],
                            )
                            or ()
                        )
                        if isinstance(entry, Mapping)
                    )
                except Exception:
                    if caked_projection_required:
                        raise
                    return normalized_rows
            projected_rows = sorted(
                projected_rows,
                key=lambda entry: (
                    int(entry.get(order_key))
                    if isinstance(entry, Mapping) and entry.get(order_key) is not None
                    else int(1e12)
                ),
            )
            for projected_entry in projected_rows:
                projected_entry.pop(order_key, None)
        else:
            if caked_projection_required:
                raise RuntimeError(
                    "exact caked projector unavailable for manual caked geometry fit"
                )
            projector = manual_dataset_bindings.geometry_manual_project_peaks_to_current_view
            if not callable(projector):
                return normalized_rows
            try:
                projected_rows = projector(normalized_rows)
            except Exception:
                if caked_projection_required:
                    raise
                return normalized_rows
        projected = [dict(entry) for entry in (projected_rows or ()) if isinstance(entry, Mapping)]
        if not use_caked_display:
            try:
                sim_shape = (
                    int(manual_dataset_bindings.image_size),
                    int(manual_dataset_bindings.image_size),
                )
                rotate_k = int(manual_dataset_bindings.display_rotate_k)
            except Exception:
                sim_shape = ()
                rotate_k = 0

            def _point(
                entry: Mapping[str, object],
                x_key: str,
                y_key: str,
            ) -> tuple[float, float] | None:
                try:
                    point = (float(entry.get(x_key)), float(entry.get(y_key)))
                except Exception:
                    return None
                if not (np.isfinite(point[0]) and np.isfinite(point[1])):
                    return None
                return point

            for source_entry, projected_entry in zip(normalized_rows, projected):
                raw_point = _point(source_entry, "sim_col_raw", "sim_row_raw")
                if raw_point is None or len(sim_shape) < 2 or min(sim_shape) <= 0:
                    continue
                projected_display = _point(projected_entry, "display_col", "display_row")
                projected_sim = _point(projected_entry, "sim_col", "sim_row")
                if projected_display != raw_point or projected_sim != raw_point:
                    continue
                try:
                    rotated = gui_manual_geometry._default_rotate_point(
                        float(raw_point[0]),
                        float(raw_point[1]),
                        sim_shape,
                        rotate_k,
                    )
                except Exception:
                    continue
                if (
                    isinstance(rotated, tuple)
                    and len(rotated) >= 2
                    and np.isfinite(float(rotated[0]))
                    and np.isfinite(float(rotated[1]))
                ):
                    projected_entry["display_col"] = float(rotated[0])
                    projected_entry["display_row"] = float(rotated[1])
                    projected_entry["sim_col"] = float(rotated[0])
                    projected_entry["sim_row"] = float(rotated[1])
        return projected

    def _project_source_entry_for_current_view(
        entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        if not isinstance(entry, Mapping):
            return None
        projected_rows = _project_source_rows_for_current_view([dict(entry)])
        if projected_rows:
            return dict(projected_rows[0])
        return dict(entry)

    def _current_simulation_diagnostics() -> dict[str, object]:
        if callable(manual_dataset_bindings.geometry_manual_last_source_snapshot_diagnostics):
            raw_value = manual_dataset_bindings.geometry_manual_last_source_snapshot_diagnostics()
        elif callable(manual_dataset_bindings.geometry_manual_last_simulation_diagnostics):
            raw_value = manual_dataset_bindings.geometry_manual_last_simulation_diagnostics()
        else:
            raw_value = {}
        copied = copy.deepcopy(raw_value)
        return copied if isinstance(copied, dict) else {}

    def _source_coverage_filter_diagnostics(
        rows: Sequence[object] | None,
    ) -> dict[str, object]:
        required_targets = collect_geometry_fit_required_manual_fit_targets(
            selected_entries,
            background_index=int(background_idx),
        )
        required_keys = _geometry_fit_required_branch_group_keys(required_targets)
        copied_rows = [dict(row) for row in (rows or ()) if isinstance(row, Mapping)]
        if not required_keys:
            return {}
        filtered_rows, counts, matched_keys = (
            _geometry_fit_filter_entries_for_required_branch_groups(
                copied_rows,
                required_keys,
            )
        )
        required_payloads = _geometry_fit_required_branch_group_key_payloads(required_keys)
        matched_payloads = _geometry_fit_required_branch_group_key_payloads(matched_keys)
        matched_payload_keys = {
            (tuple(item.get("hkl") or ()), item.get("branch_index"), item.get("q_group_key"))
            for item in matched_payloads
        }
        missing_required_payloads = [
            item
            for item in required_payloads
            if (
                tuple(item.get("hkl") or ()),
                item.get("branch_index"),
                item.get("q_group_key"),
            )
            not in matched_payload_keys
        ]
        return {
            "total_source_rows_available": int(counts.get("total_count", 0) or 0),
            "source_rows_considered_for_rebinding": int(counts.get("total_count", 0) or 0),
            "candidate_rows_after_hkl_filter": int(counts.get("after_hkl_filter_count", 0) or 0),
            "candidate_rows_after_branch_filter": int(
                counts.get("after_branch_filter_count", 0) or 0
            ),
            "candidate_rows_scored_for_background_distance": int(
                counts.get("after_branch_filter_count", 0) or 0
            ),
            "unrelated_available_row_count_for_rebinding": int(
                counts.get("unrelated_count", 0) or 0
            ),
            "missing_required_branch_group_keys": missing_required_payloads,
            "missing_required_hkl_inventory": _geometry_fit_required_hkl_inventory(
                [
                    (
                        tuple(item.get("hkl") or ()),
                        item.get("branch_index"),
                        item.get("q_group_key"),
                    )
                    for item in missing_required_payloads
                ]
            ),
            "source_row_hkl_inventory_after_rebinding_filter": (
                _geometry_fit_hkl_inventory_from_entries(filtered_rows)
            ),
            "source_row_hkl_branch_inventory_after_rebinding_filter": (
                _geometry_fit_hkl_branch_inventory_from_entries(filtered_rows)
            ),
        }

    def _provider_backed_source_row_for_target(
        *,
        pair_idx: int,
        entry: Mapping[str, object],
        raw_saved_entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        saved_entry = dict(raw_saved_entry) if isinstance(raw_saved_entry, Mapping) else dict(entry)
        truth_pair = manual_picker_truth_by_order.get((int(background_idx), int(pair_idx)), {})
        simulated_point = _geometry_fit_point_list(
            truth_pair.get("manual_selected_simulated_point")
        )
        simulated_frame = _geometry_fit_normalize_point_frame(
            truth_pair.get("manual_selected_simulated_frame")
        )
        simulated_source = str(truth_pair.get("manual_simulated_point_source") or "")
        if (
            simulated_point is None
            or simulated_frame == "unknown"
            or simulated_source not in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
        ):
            return None

        row = dict(saved_entry)
        row.update(
            {
                "background_index": int(background_idx),
                "overlay_match_index": int(pair_idx),
                "pair_id": str(entry.get("pair_id") or f"bg{int(background_idx)}:pair{pair_idx}"),
                "row_origin": "manual_picker_saved_source_coverage",
                "provider_backed_live_source_row": True,
                "provider_simulated_frame": simulated_frame,
                "provider_simulated_point_source": simulated_source,
            }
        )
        for key in (
            "hkl",
            "normalized_hkl",
            "q_group_key",
            "source_q_group_key",
            "branch_group_key",
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
            "legacy_source_reflection_index",
            "legacy_source_peak_index",
            "selected_source_identity_canonical",
        ):
            if entry.get(key) is not None:
                row[key] = entry.get(key)
        if row.get("source_branch_index") is None:
            branch_idx = _geometry_fit_source_branch_index(entry)
            if branch_idx in {0, 1}:
                row["source_branch_index"] = int(branch_idx)
        coverage_key = normalize_new4_source_coverage_key(row)
        coverage_payload = _geometry_fit_source_coverage_key_payload(coverage_key)
        if coverage_payload is not None:
            row["source_coverage_aliases"] = [coverage_payload]
            if coverage_payload.get("branch_slot") == _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT:
                row["is_00l_collapsed"] = True

        if simulated_frame == "display":
            row["sim_col"] = float(simulated_point[0])
            row["sim_row"] = float(simulated_point[1])
            row["display_col"] = float(simulated_point[0])
            row["display_row"] = float(simulated_point[1])
            row["sim_display"] = (float(simulated_point[0]), float(simulated_point[1]))
        elif simulated_frame == "detector_native":
            row["native_col"] = float(simulated_point[0])
            row["native_row"] = float(simulated_point[1])
            row["sim_native_x"] = float(simulated_point[0])
            row["sim_native_y"] = float(simulated_point[1])
            row["sim_col_raw"] = float(simulated_point[0])
            row["sim_row_raw"] = float(simulated_point[1])
        elif simulated_frame == "caked_2theta_phi":
            row["caked_x"] = float(simulated_point[0])
            row["caked_y"] = float(simulated_point[1])
            row["two_theta_deg"] = float(simulated_point[0])
            row["phi_deg"] = float(simulated_point[1])
            row["sim_caked_display"] = (
                float(simulated_point[0]),
                float(simulated_point[1]),
            )

        display_point = None
        for x_key, y_key in (
            ("refined_sim_x", "refined_sim_y"),
            ("sim_col", "sim_row"),
            ("display_col", "display_row"),
        ):
            try:
                display_point = (float(row.get(x_key)), float(row.get(y_key)))
            except Exception:
                display_point = None
            if (
                isinstance(display_point, tuple)
                and np.isfinite(display_point[0])
                and np.isfinite(display_point[1])
            ):
                break
            display_point = None
        if display_point is not None and not use_caked_display:
            row["sim_col"] = float(display_point[0])
            row["sim_row"] = float(display_point[1])
            row["display_col"] = float(display_point[0])
            row["display_row"] = float(display_point[1])
            row["sim_display"] = (float(display_point[0]), float(display_point[1]))

        caked_point = _caked_angle_pair(
            row,
            x_keys=("refined_sim_caked_x", "simulated_two_theta_deg", "two_theta_deg", "caked_x"),
            y_keys=("refined_sim_caked_y", "simulated_phi_deg", "phi_deg", "caked_y"),
        )
        if caked_point is not None:
            row["caked_x"] = float(caked_point[0])
            row["caked_y"] = float(caked_point[1])
            row["two_theta_deg"] = float(caked_point[0])
            row["phi_deg"] = float(caked_point[1])
            if use_caked_display:
                row["sim_col"] = float(caked_point[0])
                row["sim_row"] = float(caked_point[1])
                row["sim_caked_display"] = (float(caked_point[0]), float(caked_point[1]))

        native_point = None
        for x_key, y_key in (
            ("refined_sim_native_x", "refined_sim_native_y"),
            ("native_col", "native_row"),
            ("sim_native_x", "sim_native_y"),
        ):
            try:
                native_point = (float(row.get(x_key)), float(row.get(y_key)))
            except Exception:
                native_point = None
            if (
                isinstance(native_point, tuple)
                and np.isfinite(native_point[0])
                and np.isfinite(native_point[1])
            ):
                break
            native_point = None
        if native_point is not None:
            row["native_col"] = float(native_point[0])
            row["native_row"] = float(native_point[1])
            row["sim_native_x"] = float(native_point[0])
            row["sim_native_y"] = float(native_point[1])
        return row

    def _augment_source_rows_with_provider_coverage(
        rows: Sequence[object] | None,
        diagnostics: Mapping[str, object] | None,
        *,
        require_provider_backed_rows: bool = False,
        allow_saved_coordinate_materialization: bool = True,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        augmented_rows = [dict(item) for item in (rows or ()) if isinstance(item, Mapping)]
        augmented_diagnostics = dict(diagnostics or {})
        coverage_keys = {
            key for row in augmented_rows for key in _geometry_fit_source_coverage_alias_keys(row)
        }
        provider_backed_keys = {
            key
            for row in augmented_rows
            if bool(row.get("provider_backed_live_source_row", False))
            or str(row.get("row_origin", "") or "") == "manual_picker_saved_source_coverage"
            for key in _geometry_fit_source_coverage_alias_keys(row)
        }
        materialized_rows: list[dict[str, object]] = []
        promoted_rows: list[dict[str, object]] = []
        point_missing: list[dict[str, object]] = []

        def _source_identity_value(
            item: Mapping[str, object],
            *keys: str,
        ) -> int | None:
            for key in keys:
                value = _geometry_fit_coerce_nonnegative_index(item.get(key))
                if value is not None and value >= 0:
                    return int(value)
            return None

        def _finite_point_for_keys(
            item: Mapping[str, object],
            x_key: str,
            y_key: str,
        ) -> tuple[float, float] | None:
            try:
                point = (float(item.get(x_key)), float(item.get(y_key)))
            except Exception:
                return None
            if not (np.isfinite(point[0]) and np.isfinite(point[1])):
                return None
            return float(point[0]), float(point[1])

        def _points_for_tuple_keys(
            item: Mapping[str, object],
            keys: Sequence[str],
        ) -> list[tuple[float, float]]:
            points: list[tuple[float, float]] = []
            for key in keys:
                point = _geometry_fit_point_list(item.get(key))
                if point is None:
                    continue
                points.append((float(point[0]), float(point[1])))
            return points

        def _target_match_points(item: Mapping[str, object]) -> list[tuple[float, float]]:
            points: list[tuple[float, float]] = []
            points.extend(
                _points_for_tuple_keys(
                    item,
                    (
                        "manual_selected_simulated_point",
                        "provider_selected_simulated_point",
                        "selected_live_simulated_current_view_point",
                        "simulated_point",
                        "sim_display",
                        "sim_native",
                        "sim_caked_display",
                    ),
                )
            )
            for x_key, y_key in (
                ("refined_sim_native_x", "refined_sim_native_y"),
                ("refined_sim_x", "refined_sim_y"),
                ("refined_sim_caked_x", "refined_sim_caked_y"),
                ("simulated_two_theta_deg", "simulated_phi_deg"),
            ):
                point = _finite_point_for_keys(item, x_key, y_key)
                if point is None:
                    continue
                points.append((float(point[0]), float(point[1])))
            return points

        def _row_match_points(item: Mapping[str, object]) -> list[tuple[float, float]]:
            points: list[tuple[float, float]] = []
            points.extend(
                _points_for_tuple_keys(
                    item,
                    (
                        "sim_display",
                        "sim_native",
                        "sim_caked_display",
                    ),
                )
            )
            for x_key, y_key in (
                ("native_col", "native_row"),
                ("sim_native_x", "sim_native_y"),
                ("sim_col_raw", "sim_row_raw"),
                ("sim_col", "sim_row"),
                ("display_col", "display_row"),
                ("caked_x", "caked_y"),
                ("two_theta_deg", "phi_deg"),
                ("simulated_two_theta_deg", "simulated_phi_deg"),
            ):
                point = _finite_point_for_keys(item, x_key, y_key)
                if point is None:
                    continue
                points.append((float(point[0]), float(point[1])))
            return points

        def _fresh_row_match_distance(
            target: Mapping[str, object],
            row: Mapping[str, object],
        ) -> float | None:
            target_points = _target_match_points(target)
            row_points = _row_match_points(row)
            if not target_points or not row_points:
                return None
            best_distance: float | None = None
            for target_x, target_y in target_points:
                for row_x, row_y in row_points:
                    distance = math.hypot(
                        float(row_x) - float(target_x), float(row_y) - float(target_y)
                    )
                    if not np.isfinite(distance):
                        continue
                    if best_distance is None or float(distance) < float(best_distance):
                        best_distance = float(distance)
            return best_distance

        def _best_unique_match_by_score_and_distance(
            target: Mapping[str, object],
            scored_rows: Sequence[tuple[int, int, dict[str, object]]],
        ) -> list[tuple[int, dict[str, object]]]:
            if not scored_rows:
                return []
            best_score = max(score for score, _row_idx, _row in scored_rows)
            best_scored = [
                (row_idx, row)
                for score, row_idx, row in scored_rows
                if int(score) == int(best_score)
            ]
            if len(best_scored) <= 1:
                return best_scored
            distanced: list[tuple[float, int, dict[str, object]]] = []
            for row_idx, row in best_scored:
                distance = _fresh_row_match_distance(target, row)
                if distance is None:
                    continue
                distanced.append((float(distance), int(row_idx), row))
            if not distanced:
                return []
            best_distance = min(distance for distance, _row_idx, _row in distanced)
            best = [
                (row_idx, row)
                for distance, row_idx, row in distanced
                if abs(float(distance) - float(best_distance)) <= 1.0e-9
            ]
            return best if len(best) == 1 else []

        def _fresh_row_match_score(
            target: Mapping[str, object],
            target_key: object,
            row: Mapping[str, object],
        ) -> int | None:
            if bool(row.get("provider_backed_live_source_row", False)) or (
                str(row.get("row_origin", "") or "") == "manual_picker_saved_source_coverage"
            ):
                return None
            if not (
                isinstance(target_key, tuple)
                and len(target_key) >= 3
                and isinstance(target_key[0], tuple)
            ):
                return None
            row_hkl = _geometry_fit_normalized_hkl(
                row.get("normalized_hkl", row.get("hkl", row.get("source_hkl")))
            )
            target_hkl = tuple(int(v) for v in target_key[0])
            target_table = _source_identity_value(
                target,
                "source_table_index",
                "resolved_table_index",
            )
            target_row = _source_identity_value(
                target,
                "source_row_index",
                "resolved_source_row_index",
                "resolved_source_row_position",
            )
            target_peak = _source_identity_value(
                target,
                "source_peak_index",
                "resolved_peak_index",
            )
            target_reflection = _source_identity_value(
                target,
                "source_reflection_index",
                "legacy_source_reflection_index",
            )
            target_branch_slot = target_key[1]
            row_branch = _geometry_fit_source_branch_index(row)
            row_table = _source_identity_value(
                row,
                "source_table_index",
                "resolved_table_index",
            )
            row_index = _source_identity_value(
                row,
                "source_row_index",
                "resolved_source_row_index",
                "resolved_source_row_position",
            )
            row_peak = _source_identity_value(
                row,
                "source_peak_index",
                "resolved_peak_index",
            )
            row_reflection = _source_identity_value(
                row,
                "source_reflection_index",
                "legacy_source_reflection_index",
            )
            identity_score = 0
            target_group = (
                _geometry_fit_stable_group_identity(target_key[2])
                if isinstance(target_key, tuple) and len(target_key) >= 3
                else None
            )
            row_group = _geometry_fit_group_identity(row)
            group_matches = (
                _geometry_fit_group_identity_is_q_group(target_group)
                and target_group is not None
                and row_group == target_group
            )
            strong_identity_score = 0
            if (
                target_table is not None
                and row_table is not None
                and int(row_table) == int(target_table)
            ):
                identity_score += 4
                strong_identity_score += 4
            if (
                target_row is not None
                and row_index is not None
                and int(row_index) == int(target_row)
            ):
                identity_score += 4
                strong_identity_score += 4
            if (
                target_peak is not None
                and row_peak is not None
                and int(row_peak) == int(target_peak)
            ):
                identity_score += 2
            if (
                target_reflection is not None
                and row_reflection is not None
                and int(row_reflection) == int(target_reflection)
            ):
                identity_score += 6
                strong_identity_score += 6
            hkl_matches = row_hkl == target_hkl
            if not hkl_matches and not group_matches and strong_identity_score <= 0:
                return None

            score = int(identity_score)
            if hkl_matches:
                score += 8
            if group_matches:
                score += 6
            if target_branch_slot == _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT:
                score += 2
            elif target_branch_slot in {0, 1}:
                if row_branch != int(target_branch_slot):
                    return None
                score += 4
            else:
                return None
            return int(score)

        def _matching_fresh_rows(
            target: Mapping[str, object],
            target_key: object,
        ) -> list[tuple[int, dict[str, object]]]:
            matches: list[tuple[int, dict[str, object]]] = []
            for row_idx, row in enumerate(augmented_rows):
                if target_key not in _geometry_fit_source_coverage_alias_keys(row):
                    continue
                if bool(row.get("provider_backed_live_source_row", False)) or (
                    str(row.get("row_origin", "") or "") == "manual_picker_saved_source_coverage"
                ):
                    continue
                matches.append((int(row_idx), row))
            if len(matches) <= 1:
                if matches:
                    return matches
                scored_matches: list[tuple[int, int, dict[str, object]]] = []
                for row_idx, row in enumerate(augmented_rows):
                    score = _fresh_row_match_score(target, target_key, row)
                    if score is not None:
                        scored_matches.append((int(score), int(row_idx), row))
                return _best_unique_match_by_score_and_distance(target, scored_matches)

            target_table = _source_identity_value(
                target,
                "source_table_index",
                "resolved_table_index",
            )
            target_row = _source_identity_value(
                target,
                "source_row_index",
                "resolved_source_row_index",
                "resolved_source_row_position",
            )
            target_peak = _source_identity_value(
                target,
                "source_peak_index",
                "resolved_peak_index",
            )
            target_reflection = _source_identity_value(
                target,
                "source_reflection_index",
                "legacy_source_reflection_index",
            )

            scored: list[tuple[int, int, dict[str, object]]] = []
            for row_idx, row in matches:
                score = 0
                row_table = _source_identity_value(
                    row,
                    "source_table_index",
                    "resolved_table_index",
                )
                row_index = _source_identity_value(
                    row,
                    "source_row_index",
                    "resolved_source_row_index",
                    "resolved_source_row_position",
                )
                row_peak = _source_identity_value(
                    row,
                    "source_peak_index",
                    "resolved_peak_index",
                )
                row_reflection = _source_identity_value(
                    row,
                    "source_reflection_index",
                    "legacy_source_reflection_index",
                )
                if (
                    target_table is not None
                    and row_table is not None
                    and int(row_table) == int(target_table)
                ):
                    score += 4
                if (
                    target_row is not None
                    and row_index is not None
                    and int(row_index) == int(target_row)
                ):
                    score += 4
                if (
                    target_peak is not None
                    and row_peak is not None
                    and int(row_peak) == int(target_peak)
                ):
                    score += 2
                if (
                    target_reflection is not None
                    and row_reflection is not None
                    and int(row_reflection) == int(target_reflection)
                ):
                    score += 2
                scored.append((int(score), int(row_idx), row))
            if not scored:
                return matches
            best_score = max(score for score, _row_idx, _row in scored)
            if best_score <= 0:
                return _best_unique_match_by_score_and_distance(
                    target,
                    [(1, row_idx, row) for row_idx, row in matches],
                )
            return _best_unique_match_by_score_and_distance(target, scored)

        def _promote_fresh_source_row(
            *,
            pair_idx: int,
            target_key: object,
            entry: Mapping[str, object],
        ) -> bool:
            matches = _matching_fresh_rows(entry, target_key)
            if len(matches) != 1:
                return False
            row_index, row = matches[0]
            coverage_payload = _geometry_fit_source_coverage_key_payload(target_key)
            promoted = dict(row)
            aliases = list(promoted.get("source_coverage_aliases") or [])
            if coverage_payload is not None and coverage_payload not in aliases:
                aliases.append(coverage_payload)
            if aliases:
                promoted["source_coverage_aliases"] = aliases
            for key in (
                "hkl",
                "normalized_hkl",
                "q_group_key",
                "source_q_group_key",
                "branch_group_key",
                "source_table_index",
                "source_reflection_index",
                "source_reflection_namespace",
                "source_reflection_is_full",
                "source_row_index",
                "source_branch_index",
                "source_peak_index",
                "legacy_source_reflection_index",
                "legacy_source_peak_index",
                "branch_id",
                "best_sample_index",
                "mosaic_top_rank_key",
                "selection_reason",
                "selected_source_identity_canonical",
            ):
                locked_value = entry.get(key)
                if locked_value is None:
                    continue
                if promoted.get(key) is not None and promoted.get(key) != locked_value:
                    promoted.setdefault(f"trial_{key}", promoted.get(key))
                promoted[key] = locked_value
            promoted["provider_backed_live_source_row"] = True
            promoted["provider_backed_live_source_row_reason"] = (
                "geometry_fit_dataset_required_source_coverage"
            )
            if use_caked_display or geometry_manual_pairs_use_caked_fit_space(selected_entries):
                promoted["source_kind"] = "sim_visual_caked_deg"
                promoted["actual_source"] = "sim_visual_caked_deg"
                promoted["expected_source"] = "sim_visual_caked_deg"
                promoted["projection_frame"] = "caked_display"
                promoted["coordinate_provenance"] = "trial_geometry_projection"
                promoted["is_dynamic_trial_row"] = True
            promoted.setdefault(
                "row_origin",
                str(row.get("row_origin") or "geometry_fit_dataset_required_source_row"),
            )
            promoted["physical_branch_slot"] = (
                coverage_payload.get("branch_slot")
                if isinstance(coverage_payload, Mapping)
                else None
            )
            promoted["fit_qr_branch_key"] = {
                "q_group_key": _geometry_fit_cache_jsonable(promoted.get("q_group_key")),
                "hkl": _geometry_fit_cache_jsonable(
                    promoted.get("normalized_hkl", promoted.get("hkl"))
                ),
                "physical_branch_slot": promoted.get("physical_branch_slot"),
                "source_branch_index": promoted.get("source_branch_index"),
                "source_peak_index": promoted.get("source_peak_index"),
            }
            promoted["background_index"] = int(background_idx)
            promoted["overlay_match_index"] = int(pair_idx)
            promoted["pair_id"] = str(
                entry.get("pair_id") or f"bg{int(background_idx)}:pair{pair_idx}"
            )
            augmented_rows[int(row_index)] = promoted
            promoted_rows.append(promoted)
            provider_backed_keys.update(_geometry_fit_source_coverage_alias_keys(promoted))
            return True

        for pair_idx, selected_input in enumerate(selected_entry_inputs):
            raw_entry = selected_input.get("entry")
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            raw_saved_entry = (
                dict(selected_input.get("raw_saved_entry"))
                if isinstance(selected_input.get("raw_saved_entry"), Mapping)
                else None
            )
            truth_pair = manual_picker_truth_by_order.get((int(background_idx), int(pair_idx)), {})
            simulated_point = _geometry_fit_point_list(
                truth_pair.get("manual_selected_simulated_point")
            )
            simulated_frame = _geometry_fit_normalize_point_frame(
                truth_pair.get("manual_selected_simulated_frame")
            )
            if simulated_point is not None and simulated_frame != "unknown":
                entry["manual_selected_simulated_point"] = [
                    float(simulated_point[0]),
                    float(simulated_point[1]),
                ]
                entry["manual_selected_simulated_frame"] = simulated_frame
                entry["manual_simulated_point_source"] = str(
                    truth_pair.get("manual_simulated_point_source") or ""
                )
                _geometry_fit_put_simulated_point_fields(
                    entry,
                    simulated_point,
                    simulated_frame,
                )
            target_key = normalize_new4_source_coverage_key(entry)
            if target_key is None:
                continue
            if target_key in coverage_keys and (
                not require_provider_backed_rows or target_key in provider_backed_keys
            ):
                continue
            if require_provider_backed_rows:
                if _promote_fresh_source_row(
                    pair_idx=int(pair_idx),
                    target_key=target_key,
                    entry=entry,
                ):
                    continue
            if not allow_saved_coordinate_materialization:
                point_missing.append(
                    {
                        "pair_index": int(pair_idx),
                        "target_key": _geometry_fit_source_coverage_key_payload(target_key),
                        "reason": "missing_dynamic_trial_source_row",
                    }
                )
                continue
            provider_row = _provider_backed_source_row_for_target(
                pair_idx=int(pair_idx),
                entry=entry,
                raw_saved_entry=raw_saved_entry,
            )
            if provider_row is None:
                point_missing.append(
                    {
                        "pair_index": int(pair_idx),
                        "target_key": _geometry_fit_source_coverage_key_payload(target_key),
                        "reason": "coverage_source_present_point_missing",
                    }
                )
                continue
            materialized_rows.append(provider_row)
            coverage_keys.update(_geometry_fit_source_coverage_alias_keys(provider_row))
            provider_backed_keys.update(_geometry_fit_source_coverage_alias_keys(provider_row))
        if materialized_rows:
            augmented_rows.extend(materialized_rows)
        if materialized_rows or promoted_rows or point_missing:
            augmented_diagnostics["source_coverage_materialization"] = {
                "provider_backed_row_count": int(len(materialized_rows)),
                "provider_backed_fresh_row_count": int(len(promoted_rows)),
                "point_missing_count": int(len(point_missing)),
                "saved_coordinate_materialization_allowed": bool(
                    allow_saved_coordinate_materialization
                ),
                "provider_backed_keys": [
                    _geometry_fit_source_coverage_key_payload(
                        normalize_new4_source_coverage_key(row)
                    )
                    for row in materialized_rows
                ],
                "point_missing": point_missing,
            }
            augmented_diagnostics["provider_backed_fresh_source_coverage_row_count"] = int(
                len(promoted_rows)
            )
            augmented_diagnostics["provider_backed_source_coverage_row_count"] = int(
                len(materialized_rows)
            )
            augmented_diagnostics["coverage_source_present_point_missing_count"] = int(
                len(point_missing)
            )
            augmented_diagnostics["missing_dynamic_trial_source_row_count"] = int(
                sum(
                    1
                    for item in point_missing
                    if str(item.get("reason") or "") == "missing_dynamic_trial_source_row"
                )
            )
        coverage_diagnostics = _source_coverage_filter_diagnostics(augmented_rows)
        if coverage_diagnostics:
            augmented_diagnostics.update(copy.deepcopy(coverage_diagnostics))
            targeted_gate = dict(augmented_diagnostics.get("targeted_performance_gate") or {})
            targeted_gate.update(copy.deepcopy(coverage_diagnostics))
            augmented_diagnostics["targeted_performance_gate"] = targeted_gate
        return augmented_rows, augmented_diagnostics

    def _trial_source_rows_signature(rows: object) -> str:
        try:
            payload = repr(_geometry_fit_cache_jsonable(rows)).encode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            payload = repr(rows).encode("utf-8", errors="replace")
        return hashlib.sha1(payload).hexdigest()

    def _trial_row_is_dynamic(row: Mapping[str, object]) -> bool:
        if str(row.get("row_origin", "") or "") == "manual_picker_saved_source_coverage":
            return False
        actual_source = str(row.get("actual_source") or "").strip()
        source_kind = str(row.get("source_kind") or "").strip()
        projection_frame = str(row.get("projection_frame") or "").strip()
        provenance = str(row.get("coordinate_provenance") or "").strip()
        if actual_source != "sim_visual_caked_deg" or source_kind != "sim_visual_caked_deg":
            return False
        if projection_frame != "caked_display":
            return False
        if provenance and provenance != "trial_geometry_projection":
            return False
        if "is_dynamic_trial_row" in row and row.get("is_dynamic_trial_row") is not True:
            return False
        return True

    def _trial_stage_lineage(row: Mapping[str, object] | None) -> dict[str, object]:
        if not isinstance(row, Mapping):
            return {}
        return {
            "source_kind": row.get("source_kind"),
            "actual_source": row.get("actual_source"),
            "expected_source": row.get("expected_source"),
            "coordinate_provenance": row.get("coordinate_provenance"),
            "projection_frame": row.get("projection_frame"),
            "is_dynamic_trial_row": row.get("is_dynamic_trial_row"),
            "source_table_index": row.get("source_table_index"),
            "source_row_index": row.get("source_row_index"),
            "source_branch_index": row.get("source_branch_index"),
            "source_peak_index": row.get("source_peak_index"),
            "row_index_namespace": row.get("row_index_namespace"),
            "row_origin": row.get("row_origin"),
            "fit_qr_branch_key": _geometry_fit_cache_jsonable(row.get("fit_qr_branch_key")),
        }

    def _trial_row_matches_target_key(
        row: Mapping[str, object] | None,
        target_key: object,
    ) -> bool:
        if not isinstance(row, Mapping):
            return False
        row_keys = _geometry_fit_source_coverage_alias_keys(row)
        if target_key in row_keys:
            return True
        if not isinstance(target_key, tuple) or len(target_key) < 3 or target_key[2] is None:
            return False
        row_group = _geometry_fit_group_identity(row)
        target_group = _geometry_fit_stable_group_identity(target_key[2])
        if (
            row_group is None
            or target_group is None
            or not _geometry_fit_group_identity_is_q_group(target_group)
            or row_group != target_group
        ):
            return False
        target_branch = target_key[1]
        if target_branch == _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT:
            return True
        row_branch = _geometry_fit_source_branch_index(row)
        return target_branch in {0, 1} and row_branch == int(target_branch)

    def _trial_stage_summary(
        stage_name: str,
        rows: Sequence[object] | None,
    ) -> dict[str, object]:
        stage_rows = [dict(row) for row in (rows or ()) if isinstance(row, Mapping)]
        q_group_counts: dict[str, int] = {}
        hkl_counts: dict[str, int] = {}
        branch_counts: dict[str, int] = {}
        for row in stage_rows:
            payload = _geometry_fit_source_coverage_key_payload(
                normalize_new4_source_coverage_key(row)
            )
            if isinstance(payload, Mapping):
                q_label = json.dumps(
                    _geometry_fit_cache_jsonable(payload.get("q_group_key")),
                    sort_keys=True,
                )
                h_label = json.dumps(
                    _geometry_fit_cache_jsonable(payload.get("hkl")),
                    sort_keys=True,
                )
                b_label = json.dumps(
                    _geometry_fit_cache_jsonable(payload.get("branch_slot")),
                    sort_keys=True,
                )
            else:
                q_label = "<missing>"
                h_label = "<missing>"
                b_label = "<missing>"
            q_group_counts[q_label] = q_group_counts.get(q_label, 0) + 1
            hkl_counts[h_label] = hkl_counts.get(h_label, 0) + 1
            branch_counts[b_label] = branch_counts.get(b_label, 0) + 1

        per_pair: list[dict[str, object]] = []
        dynamic_count = 0
        stale_count = 0
        missing_count = 0
        for pair_idx, selected_input in enumerate(selected_entry_inputs):
            raw_entry = selected_input.get("entry")
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            target_key = normalize_new4_source_coverage_key(entry)
            target_payload = _geometry_fit_source_coverage_key_payload(target_key)
            matches: list[tuple[int, dict[str, object]]] = []
            for row_idx, row in enumerate(stage_rows):
                if _trial_row_matches_target_key(row, target_key):
                    matches.append((int(row_idx), row))
            dynamic_matches = [
                (row_idx, row) for row_idx, row in matches if _trial_row_is_dynamic(row)
            ]
            selected_index: int | None = None
            selected_row: dict[str, object] | None = None
            drop_reason: str | None = None
            if dynamic_matches:
                selected_index, selected_row = dynamic_matches[0]
                dynamic_count += 1
            elif matches:
                selected_index, selected_row = matches[0]
                stale_count += 1
                drop_reason = "stale_qr_coordinate_provenance"
            else:
                missing_count += 1
                drop_reason = "missing_dynamic_trial_source_row"
            per_pair.append(
                {
                    "pair_id": str(
                        entry.get("pair_id") or f"bg{int(background_idx)}:pair{pair_idx}"
                    ),
                    "pair_index": int(pair_idx),
                    "q_group_key": (
                        _geometry_fit_cache_jsonable(target_payload.get("q_group_key"))
                        if isinstance(target_payload, Mapping)
                        else None
                    ),
                    "normalized_hkl": (
                        _geometry_fit_cache_jsonable(target_payload.get("hkl"))
                        if isinstance(target_payload, Mapping)
                        else None
                    ),
                    "physical_branch_slot": (
                        _geometry_fit_cache_jsonable(target_payload.get("branch_slot"))
                        if isinstance(target_payload, Mapping)
                        else None
                    ),
                    "fit_qr_branch_key": _geometry_fit_cache_jsonable(
                        entry.get("fit_qr_branch_key")
                    ),
                    "present": bool(matches),
                    "matched_source_row_count": int(len(matches)),
                    "dynamic_match_count": int(len(dynamic_matches)),
                    "source_row_array_index": selected_index,
                    "drop_reason": drop_reason,
                    **_trial_stage_lineage(selected_row),
                }
            )
        return {
            "stage_name": str(stage_name),
            "row_count": int(len(stage_rows)),
            "required_pair_count": int(len(per_pair)),
            "dynamic_required_pair_count": int(dynamic_count),
            "missing_required_pair_count": int(missing_count),
            "stale_required_pair_count": int(stale_count),
            "q_group_counts": q_group_counts,
            "hkl_counts": hkl_counts,
            "branch_counts": branch_counts,
            "per_required_pair": per_pair,
        }

    def _mark_dynamic_trial_source_rows(rows: Sequence[object] | None) -> list[dict[str, object]]:
        marked: list[dict[str, object]] = []
        for item in rows or ():
            if not isinstance(item, Mapping):
                continue
            row = dict(item)
            if str(row.get("row_origin", "") or "") != "manual_picker_saved_source_coverage":
                row.setdefault("source_kind", "sim_visual_caked_deg")
                row.setdefault("actual_source", "sim_visual_caked_deg")
                row.setdefault("expected_source", "sim_visual_caked_deg")
                row.setdefault("projection_frame", "caked_display")
                row.setdefault("coordinate_provenance", "trial_geometry_projection")
                row.setdefault("is_dynamic_trial_row", True)
            marked.append(row)
        return marked

    def _trial_candidate_sort_key(
        target: Mapping[str, object],
        target_key: object,
        row: Mapping[str, object],
        row_idx: int,
    ) -> tuple[int, float, int]:
        score = 0
        if target_key in _geometry_fit_source_coverage_alias_keys(row):
            score += 64
        target_hkl = (
            tuple(int(v) for v in target_key[0])
            if isinstance(target_key, tuple)
            and len(target_key) >= 1
            and isinstance(target_key[0], tuple)
            else None
        )
        row_hkl = _geometry_fit_normalized_hkl(
            row.get("normalized_hkl", row.get("hkl", row.get("source_hkl")))
        )
        if target_hkl is not None and row_hkl == target_hkl:
            score += 16
        for target_field, row_field, weight in (
            ("source_reflection_index", "source_reflection_index", 32),
            ("source_table_index", "source_table_index", 24),
            ("source_row_index", "source_row_index", 8),
            ("source_peak_index", "source_peak_index", 4),
        ):
            target_value = _geometry_fit_coerce_nonnegative_index(target.get(target_field))
            row_value = _geometry_fit_coerce_nonnegative_index(row.get(row_field))
            if target_value is not None and row_value is not None and target_value == row_value:
                score += int(weight)

        target_points: list[tuple[float, float]] = []
        for key in (
            "manual_selected_simulated_point",
            "provider_selected_simulated_point",
            "selected_live_simulated_current_view_point",
            "sim_visual_caked_deg",
            "sim_visual_deg",
            "sim_refined_caked_deg",
            "simulated_point",
            "sim_caked_display",
            "sim_display",
        ):
            point = _geometry_fit_point_list(target.get(key))
            if point is not None:
                target_points.append((float(point[0]), float(point[1])))
        row_points: list[tuple[float, float]] = []
        for key in (
            "sim_visual_caked_deg",
            "sim_visual_deg",
            "sim_refined_caked_deg",
            "sim_caked_display",
            "sim_display",
        ):
            point = _geometry_fit_point_list(row.get(key))
            if point is not None:
                row_points.append((float(point[0]), float(point[1])))
        for x_key, y_key in (
            ("caked_x", "caked_y"),
            ("two_theta_deg", "phi_deg"),
            ("sim_col", "sim_row"),
            ("display_col", "display_row"),
        ):
            try:
                point = (float(row.get(x_key)), float(row.get(y_key)))
            except Exception:
                point = None
            if point is not None and np.isfinite(point[0]) and np.isfinite(point[1]):
                row_points.append((float(point[0]), float(point[1])))
        distance = float("inf")
        for target_x, target_y in target_points:
            for row_x, row_y in row_points:
                candidate_distance = math.hypot(
                    float(row_x) - float(target_x), float(row_y) - float(target_y)
                )
                if np.isfinite(candidate_distance):
                    distance = min(distance, float(candidate_distance))
        return (-int(score), float(distance), int(row_idx))

    def _trial_hkl_two_theta_deg(
        hkl_value: object,
        params_local: Mapping[str, object],
    ) -> float | None:
        hkl = _geometry_fit_normalized_hkl(hkl_value)
        if hkl is None:
            return None
        try:
            a_value = float(params_local.get("a"))
            c_value = float(params_local.get("c"))
            wavelength_value = float(
                params_local.get(
                    "lambda",
                    params_local.get("wavelength", params_local.get("wavelength_angstrom")),
                )
            )
        except Exception:
            return None
        if not (
            np.isfinite(a_value)
            and np.isfinite(c_value)
            and np.isfinite(wavelength_value)
            and a_value > 0.0
            and c_value > 0.0
            and wavelength_value > 0.0
        ):
            return None
        h_val, k_val, l_val = (int(hkl[0]), int(hkl[1]), int(hkl[2]))
        m_val = float(h_val * h_val + h_val * k_val + k_val * k_val)
        inv_d_sq = (4.0 / 3.0) * m_val / (a_value * a_value)
        inv_d_sq += float(l_val * l_val) / (c_value * c_value)
        if not (np.isfinite(inv_d_sq) and inv_d_sq > 0.0):
            return None
        d_spacing = 1.0 / math.sqrt(float(inv_d_sq))
        sin_theta = wavelength_value / (2.0 * d_spacing)
        if not np.isfinite(sin_theta) or sin_theta <= 0.0 or sin_theta >= 1.0:
            return None
        return float(2.0 * math.degrees(math.asin(float(sin_theta))))

    def _trial_row_caked_point(
        row: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        if not isinstance(row, Mapping):
            return None
        for x_key, y_key in (
            ("caked_x", "caked_y"),
            ("two_theta_deg", "phi_deg"),
            ("display_col", "display_row"),
            ("sim_col", "sim_row"),
        ):
            try:
                x_val = float(row.get(x_key))
                y_val = float(row.get(y_key))
            except Exception:
                continue
            if np.isfinite(x_val) and np.isfinite(y_val):
                return float(x_val), float(y_val)
        point = _geometry_fit_point_list(row.get("sim_caked_display"))
        if point is not None:
            return float(point[0]), float(point[1])
        point = _geometry_fit_point_list(row.get("sim_visual_caked_deg"))
        if point is not None:
            return float(point[0]), float(point[1])
        return None

    def _build_dynamic_trial_completion_row(
        target: Mapping[str, object],
        target_key: object,
        *,
        active_params: Mapping[str, object],
        candidate_rows: Sequence[Mapping[str, object]],
    ) -> tuple[dict[str, object] | None, str]:
        if (
            not isinstance(target_key, tuple)
            or len(target_key) < 3
            or not isinstance(target_key[0], tuple)
        ):
            return None, "invalid_target_key"
        target_hkl = tuple(int(v) for v in target_key[0])
        target_branch = target_key[1]
        target_group = _geometry_fit_stable_group_identity(target_key[2])
        if target_group is None:
            return None, "missing_q_group_key"
        two_theta = _trial_hkl_two_theta_deg(target_hkl, active_params)
        if two_theta is None:
            return None, "missing_analytic_two_theta"

        sibling_row: dict[str, object] | None = None
        sibling_point: tuple[float, float] | None = None
        if target_branch in {0, 1}:
            sibling_candidates: list[tuple[float, int, dict[str, object], tuple[float, float]]] = []
            for row_idx, raw_row in enumerate(candidate_rows or ()):
                if not isinstance(raw_row, Mapping) or not _trial_row_is_dynamic(raw_row):
                    continue
                row_group = _geometry_fit_group_identity(raw_row)
                if row_group != target_group:
                    continue
                row_branch = _geometry_fit_source_branch_index(raw_row)
                if row_branch == int(target_branch):
                    continue
                point = _trial_row_caked_point(raw_row)
                if point is None:
                    continue
                sibling_candidates.append(
                    (
                        abs(float(point[0]) - float(two_theta)),
                        int(row_idx),
                        dict(raw_row),
                        (float(point[0]), float(point[1])),
                    )
                )
            if not sibling_candidates:
                return None, "missing_dynamic_sibling_branch"
            _distance, _row_idx, sibling_row, sibling_point = min(
                sibling_candidates,
                key=lambda item: (float(item[0]), int(item[1])),
            )

        if target_branch == _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT:
            caked_point = (float(two_theta), 0.0)
            completion_reason = "analytic_00l_collapsed"
        elif target_branch in {0, 1} and sibling_point is not None:
            caked_point = (float(two_theta), -float(sibling_point[1]))
            completion_reason = "mirrored_dynamic_q_group_branch"
        else:
            return None, "unsupported_branch_slot"

        row: dict[str, object] = {
            "background_index": int(background_idx),
            "q_group_key": target_group,
            "hkl": target_hkl,
            "normalized_hkl": target_hkl,
            "label": f"{target_hkl[0]},{target_hkl[1]},{target_hkl[2]}",
            "source_kind": "sim_visual_caked_deg",
            "actual_source": "sim_visual_caked_deg",
            "expected_source": "sim_visual_caked_deg",
            "projection_frame": "caked_display",
            "coordinate_provenance": "trial_geometry_projection",
            "is_dynamic_trial_row": True,
            "row_origin": "geometry_fit_trial_dynamic_completion",
            "dynamic_completion_reason": completion_reason,
            "consumer": "geometry_fit_trial_source_rows",
            "caked_x": float(caked_point[0]),
            "caked_y": float(caked_point[1]),
            "two_theta_deg": float(caked_point[0]),
            "phi_deg": float(caked_point[1]),
            "display_col": float(caked_point[0]),
            "display_row": float(caked_point[1]),
            "sim_col": float(caked_point[0]),
            "sim_row": float(caked_point[1]),
            "sim_caked_display": (float(caked_point[0]), float(caked_point[1])),
            "sim_visual_caked_deg": (float(caked_point[0]), float(caked_point[1])),
        }
        for key in (
            "pair_id",
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
        ):
            if key in target:
                row[key] = copy.deepcopy(target.get(key))
        if isinstance(sibling_row, Mapping):
            row["dynamic_completion_sibling_source_table_index"] = sibling_row.get(
                "source_table_index"
            )
            row["dynamic_completion_sibling_source_row_index"] = sibling_row.get("source_row_index")
            row["dynamic_completion_sibling_branch"] = _geometry_fit_source_branch_index(
                sibling_row
            )
        if target_branch in {0, 1}:
            row["physical_branch_slot"] = int(target_branch)
            row["source_branch_index"] = int(target_branch)
            row["source_peak_index"] = int(target_branch)
            row["source_branch_index_namespace"] = "physical_branch_slot"
        else:
            row["physical_branch_slot"] = _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT
            row["source_branch_index"] = _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT
            row["source_peak_index"] = 0
            row["source_branch_index_namespace"] = "00l_collapsed"
            row["is_00l_collapsed"] = True
        coverage_payload = _geometry_fit_source_coverage_key_payload(target_key)
        if isinstance(coverage_payload, Mapping):
            row["source_coverage_aliases"] = [dict(coverage_payload)]
        return row, completion_reason

    def _supplement_dynamic_trial_rows_from_candidate_pool(
        existing_rows: Sequence[object] | None,
        *,
        active_params: Mapping[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        diag: dict[str, object] = {
            "attempted": False,
            "candidate_row_count": 0,
            "supplemental_row_count": 0,
            "missing_before_count": 0,
            "missing_after_count": 0,
            "per_pair": [],
        }
        if not callable(manual_dataset_bindings.geometry_manual_simulated_peaks_for_params):
            diag["skip_reason"] = "simulated_peaks_provider_unavailable"
            return [], diag

        current_rows = [dict(row) for row in (existing_rows or ()) if isinstance(row, Mapping)]
        missing_targets: list[tuple[int, dict[str, object], object]] = []
        for pair_idx, selected_input in enumerate(selected_entry_inputs):
            raw_entry = selected_input.get("entry")
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            target_key = normalize_new4_source_coverage_key(entry)
            if target_key is None:
                continue
            if any(
                _trial_row_is_dynamic(row) and _trial_row_matches_target_key(row, target_key)
                for row in current_rows
            ):
                continue
            missing_targets.append((int(pair_idx), entry, target_key))
        diag["missing_before_count"] = int(len(missing_targets))
        if not missing_targets:
            return [], diag

        diag["attempted"] = True
        supplemental: list[dict[str, object]] = []
        remaining_targets: list[tuple[int, dict[str, object], object]] = []
        for pair_idx, target, target_key in missing_targets:
            completion_row, completion_reason = _build_dynamic_trial_completion_row(
                target,
                target_key,
                active_params=active_params,
                candidate_rows=current_rows,
            )
            if completion_row is None:
                remaining_targets.append((pair_idx, target, target_key))
                continue
            pair_id = str(target.get("pair_id") or f"bg{int(background_idx)}:pair{pair_idx}")
            completion_row["background_index"] = int(background_idx)
            completion_row["overlay_match_index"] = int(pair_idx)
            completion_row["pair_id"] = str(pair_id)
            supplemental.append(dict(completion_row))
            diag["per_pair"].append(
                {
                    "pair_id": str(pair_id),
                    "pair_index": int(pair_idx),
                    "target_key": _geometry_fit_source_coverage_key_payload(target_key),
                    "matched_candidate_count": 0,
                    "selected": True,
                    "selected_candidate_index": -1,
                    "dynamic_completion_reason": completion_reason,
                    "dynamic_completion_source": "source_rows_before_rebinding",
                    **_trial_stage_lineage(completion_row),
                }
            )
        diag["dynamic_completion_before_candidate_pool_count"] = int(len(supplemental))
        if not remaining_targets:
            combined_rows = current_rows + supplemental
            remaining_missing = 0
            for _pair_idx, _target, target_key in missing_targets:
                if not any(
                    _trial_row_is_dynamic(row) and _trial_row_matches_target_key(row, target_key)
                    for row in combined_rows
                ):
                    remaining_missing += 1
            diag["raw_candidate_row_count"] = 0
            diag["projected_candidate_row_count"] = 0
            diag["candidate_row_count"] = 0
            diag["_candidate_pool_rows_for_stage"] = []
            diag["supplemental_row_count"] = int(len(supplemental))
            diag["missing_after_count"] = int(remaining_missing)
            return supplemental, diag

        try:
            raw_candidates = (
                manual_dataset_bindings.geometry_manual_simulated_peaks_for_params(
                    dict(active_params),
                    prefer_cache=False,
                )
                or []
            )
        except TypeError:
            try:
                raw_candidates = (
                    manual_dataset_bindings.geometry_manual_simulated_peaks_for_params(
                        dict(active_params),
                    )
                    or []
                )
            except Exception as exc:
                diag["skip_reason"] = f"simulated_peaks_provider_error:{type(exc).__name__}"
                return [], diag
        except Exception as exc:
            diag["skip_reason"] = f"simulated_peaks_provider_error:{type(exc).__name__}"
            return [], diag

        raw_candidate_rows = [
            dict(row) for row in (raw_candidates or ()) if isinstance(row, Mapping)
        ]
        diag["raw_candidate_row_count"] = int(len(raw_candidate_rows))
        if callable(manual_dataset_bindings.geometry_manual_last_simulation_diagnostics):
            try:
                provider_diag = (
                    manual_dataset_bindings.geometry_manual_last_simulation_diagnostics()
                )
            except Exception:
                provider_diag = None
            if isinstance(provider_diag, Mapping):
                diag["simulated_candidate_provider_diagnostics"] = copy.deepcopy(
                    dict(provider_diag)
                )
        projected_candidates = _project_source_rows_for_current_view(raw_candidates)
        diag["projected_candidate_row_count"] = int(len(projected_candidates or ()))
        candidate_rows = _mark_dynamic_trial_source_rows(projected_candidates)
        for row in candidate_rows:
            row.setdefault("row_origin", "geometry_fit_trial_caked_candidate_pool")
            row["consumer"] = "geometry_fit_trial_source_rows"
            row.setdefault("source_kind", "sim_visual_caked_deg")
            row.setdefault("actual_source", "sim_visual_caked_deg")
            row.setdefault("expected_source", "sim_visual_caked_deg")
            row.setdefault("projection_frame", "caked_display")
            row.setdefault("coordinate_provenance", "trial_geometry_projection")
            row.setdefault("is_dynamic_trial_row", True)
            branch_idx = _geometry_fit_source_branch_index(row)
            if branch_idx in {0, 1}:
                row.setdefault("physical_branch_slot", int(branch_idx))
                row.setdefault("source_branch_index_namespace", "physical_branch_slot")
        diag["candidate_row_count"] = int(len(candidate_rows))
        diag["_candidate_pool_rows_for_stage"] = [dict(row) for row in candidate_rows]

        selected_candidate_ids: set[int] = set()
        for pair_idx, target, target_key in remaining_targets:
            matches = [
                (row_idx, row)
                for row_idx, row in enumerate(candidate_rows)
                if row_idx not in selected_candidate_ids
                and _trial_row_is_dynamic(row)
                and _trial_row_matches_target_key(row, target_key)
            ]
            pair_diag = {
                "pair_id": str(target.get("pair_id") or f"bg{int(background_idx)}:pair{pair_idx}"),
                "pair_index": int(pair_idx),
                "target_key": _geometry_fit_source_coverage_key_payload(target_key),
                "matched_candidate_count": int(len(matches)),
                "selected": False,
            }
            if not matches:
                completion_row, completion_reason = _build_dynamic_trial_completion_row(
                    target,
                    target_key,
                    active_params=active_params,
                    candidate_rows=candidate_rows,
                )
                if completion_row is None:
                    pair_diag["drop_reason"] = "missing_from_caked_click_pick_candidate_inventory"
                    pair_diag["dynamic_completion_failure_reason"] = completion_reason
                    diag["per_pair"].append(pair_diag)
                    continue
                best_row_idx = -1
                best_row = completion_row
                pair_diag["dynamic_completion_reason"] = completion_reason
            else:
                best_row_idx, best_row = min(
                    matches,
                    key=lambda item: _trial_candidate_sort_key(
                        target,
                        target_key,
                        item[1],
                        item[0],
                    ),
                )
                selected_candidate_ids.add(int(best_row_idx))
            supplemental_row = dict(best_row)
            supplemental_row["background_index"] = int(background_idx)
            supplemental_row["overlay_match_index"] = int(pair_idx)
            supplemental_row["pair_id"] = str(pair_diag["pair_id"])
            coverage_payload = _geometry_fit_source_coverage_key_payload(target_key)
            if isinstance(coverage_payload, Mapping):
                aliases = list(supplemental_row.get("source_coverage_aliases") or [])
                if coverage_payload not in aliases:
                    aliases.append(dict(coverage_payload))
                supplemental_row["source_coverage_aliases"] = aliases
                supplemental_row["physical_branch_slot"] = coverage_payload.get("branch_slot")
                if (
                    coverage_payload.get("branch_slot")
                    == _GEOMETRY_FIT_ZERO_QR_COVERAGE_BRANCH_SLOT
                ):
                    supplemental_row["is_00l_collapsed"] = True
            supplemental_row["fit_qr_branch_key"] = {
                "q_group_key": _geometry_fit_cache_jsonable(supplemental_row.get("q_group_key")),
                "hkl": _geometry_fit_cache_jsonable(
                    supplemental_row.get("normalized_hkl", supplemental_row.get("hkl"))
                ),
                "physical_branch_slot": supplemental_row.get("physical_branch_slot"),
                "source_branch_index": supplemental_row.get("source_branch_index"),
                "source_peak_index": supplemental_row.get("source_peak_index"),
            }
            supplemental.append(supplemental_row)
            pair_diag.update(
                {
                    "selected": True,
                    "selected_candidate_index": int(best_row_idx),
                    **_trial_stage_lineage(supplemental_row),
                }
            )
            diag["per_pair"].append(pair_diag)

        combined_rows = current_rows + supplemental
        remaining_missing = 0
        for _pair_idx, _target, target_key in missing_targets:
            if not any(
                _trial_row_is_dynamic(row) and _trial_row_matches_target_key(row, target_key)
                for row in combined_rows
            ):
                remaining_missing += 1
        diag["supplemental_row_count"] = int(len(supplemental))
        diag["missing_after_count"] = int(remaining_missing)
        return supplemental, diag

    def _qr_fit_trial_source_rows_builder(
        *,
        local_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        active_params = dict(params_i)
        if isinstance(local_params, Mapping):
            active_params.update(dict(local_params))
        raw_rows: object = []
        source = "unavailable"
        rebuild_attempted = False
        detector_source_rows = False
        if callable(manual_dataset_bindings.geometry_manual_rebuild_source_rows_for_background):
            rebuild_attempted = True
            try:
                raw_rows = (
                    manual_dataset_bindings.geometry_manual_rebuild_source_rows_for_background(
                        int(background_idx),
                        active_params,
                        consumer="geometry_fit_trial_source_rows",
                        required_pairs=selected_entries,
                    )
                    or []
                )
                source = "geometry_manual_rebuild_source_rows_for_background"
                detector_source_rows = True
            except TypeError:
                try:
                    raw_rows = (
                        manual_dataset_bindings.geometry_manual_rebuild_source_rows_for_background(
                            int(background_idx),
                            active_params,
                        )
                        or []
                    )
                    source = "geometry_manual_rebuild_source_rows_for_background"
                except Exception:
                    raw_rows = []
            except Exception:
                raw_rows = []
        if not raw_rows and callable(
            manual_dataset_bindings.geometry_manual_source_rows_for_background
        ):
            try:
                raw_rows = (
                    manual_dataset_bindings.geometry_manual_source_rows_for_background(
                        int(background_idx),
                        active_params,
                        consumer="geometry_fit_dataset",
                        required_pairs=selected_entries,
                    )
                    or []
                )
                source = "geometry_manual_source_rows_for_background"
            except TypeError:
                try:
                    raw_rows = (
                        manual_dataset_bindings.geometry_manual_source_rows_for_background(
                            int(background_idx),
                            active_params,
                        )
                        or []
                    )
                    source = "geometry_manual_source_rows_for_background"
                except Exception:
                    raw_rows = []
            except Exception:
                raw_rows = []
        if not raw_rows:
            try:
                raw_rows = (
                    manual_dataset_bindings.geometry_manual_simulated_peaks_for_params(
                        active_params,
                        prefer_cache=False,
                    )
                    or []
                )
                source = "geometry_manual_simulated_peaks_for_params(prefer_cache=False)"
            except TypeError:
                try:
                    raw_rows = (
                        manual_dataset_bindings.geometry_manual_simulated_peaks_for_params(
                            active_params,
                        )
                        or []
                    )
                    source = "geometry_manual_simulated_peaks_for_params"
                except Exception:
                    raw_rows = []
            except Exception:
                raw_rows = []
        caked_trial_rows = bool(
            use_caked_display or geometry_manual_pairs_use_caked_fit_space(selected_entries)
        )
        projected_rows = (
            _project_source_rows_for_current_view(raw_rows)
            if caked_trial_rows or not detector_source_rows
            else [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]
        )
        if caked_trial_rows and source in {
            "geometry_manual_rebuild_source_rows_for_background",
            "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
            "geometry_manual_simulated_peaks_for_params",
        }:
            projected_rows = _mark_dynamic_trial_source_rows(projected_rows)
        supplemental_rows: list[dict[str, object]] = []
        supplemental_diag: dict[str, object] = {}
        candidate_pool_rows: list[dict[str, object]] = []
        if caked_trial_rows:
            supplemental_rows, supplemental_diag = (
                _supplement_dynamic_trial_rows_from_candidate_pool(
                    projected_rows,
                    active_params=active_params,
                )
            )
            if isinstance(supplemental_diag, Mapping):
                raw_candidate_rows = supplemental_diag.pop(
                    "_candidate_pool_rows_for_stage",
                    [],
                )
                candidate_pool_rows = [
                    dict(row) for row in (raw_candidate_rows or ()) if isinstance(row, Mapping)
                ]
            if supplemental_rows:
                projected_rows = [
                    dict(row) for row in (projected_rows or ()) if isinstance(row, Mapping)
                ] + [dict(row) for row in supplemental_rows]
        diagnostics = _current_simulation_diagnostics()
        trial_stages = {
            "required_pairs": _trial_stage_summary("required_pairs", selected_entries),
            "caked_click_pick_candidate_inventory": _trial_stage_summary(
                "caked_click_pick_candidate_inventory",
                candidate_pool_rows,
            ),
            "collector_input_rows": _trial_stage_summary(
                "collector_input_rows",
                candidate_pool_rows,
            ),
            "source_rows_raw": _trial_stage_summary("source_rows_raw", raw_rows),
            "source_rows_after_consumer_filter": _trial_stage_summary(
                "source_rows_after_consumer_filter",
                projected_rows,
            ),
            "source_rows_before_rebinding": _trial_stage_summary(
                "source_rows_before_rebinding",
                projected_rows,
            ),
        }
        rows, diagnostics = _augment_source_rows_with_provider_coverage(
            projected_rows,
            diagnostics,
            require_provider_backed_rows=True,
            allow_saved_coordinate_materialization=False,
        )
        trial_stages["source_rows_after_rebinding"] = _trial_stage_summary(
            "source_rows_after_rebinding",
            rows,
        )
        diagnostics = dict(diagnostics)
        diagnostics["geometry_fit_trial_source_rows_stages"] = trial_stages
        if supplemental_diag:
            diagnostics["caked_click_pick_candidate_inventory"] = copy.deepcopy(
                dict(supplemental_diag)
            )
        trial_geometry_hash = _geometry_fit_digest_payload(active_params)
        for row in rows:
            row["trial_geometry_hash"] = str(trial_geometry_hash)
            coverage_payload = _geometry_fit_source_coverage_key_payload(
                normalize_new4_source_coverage_key(row)
            )
            if coverage_payload is not None:
                row.setdefault("physical_branch_slot", coverage_payload.get("branch_slot"))
                row.setdefault(
                    "fit_qr_branch_key",
                    {
                        "q_group_key": _geometry_fit_cache_jsonable(row.get("q_group_key")),
                        "hkl": _geometry_fit_cache_jsonable(
                            row.get("normalized_hkl", row.get("hkl"))
                        ),
                        "physical_branch_slot": row.get("physical_branch_slot"),
                        "source_branch_index": row.get("source_branch_index"),
                        "source_peak_index": row.get("source_peak_index"),
                    },
                )
            if caked_trial_rows:
                row.setdefault("expected_source", "sim_visual_caked_deg")
                row.setdefault("projection_frame", "caked_display")
                if row.get("actual_source") is None:
                    if (
                        str(row.get("row_origin", "") or "")
                        == "manual_picker_saved_source_coverage"
                    ):
                        row["actual_source"] = str(
                            row.get("provider_simulated_point_source")
                            or row.get("sim_visual_source")
                            or "clicked_visual_candidate"
                        )
                    else:
                        row["actual_source"] = "sim_visual_caked_deg"
                row.setdefault("source_kind", row.get("actual_source"))
                if str(row.get("row_origin", "") or "") == "manual_picker_saved_source_coverage":
                    row.setdefault(
                        "coordinate_provenance",
                        "saved_manual_coordinate_materialization",
                    )
                    row.setdefault("is_dynamic_trial_row", False)
                elif (
                    str(row.get("actual_source") or "") == "sim_visual_caked_deg"
                    and str(row.get("source_kind") or "") == "sim_visual_caked_deg"
                ):
                    row.setdefault("coordinate_provenance", "trial_geometry_projection")
                    row.setdefault("is_dynamic_trial_row", True)
        return {
            "available": bool(rows),
            "rows": rows,
            "source": source,
            "source_rows_rebuilt_or_reused": "rebuilt_for_trial_params",
            "reuse_valid_for_same_params_signature": True,
            "rebuild_attempted": bool(rebuild_attempted),
            "row_count": int(len(rows)),
            "source_rows_signature": _trial_source_rows_signature(rows),
            "source_diagnostics": diagnostics,
        }

    qr_fit_trial_source_rows_builder = _qr_fit_trial_source_rows_builder
    qr_fit_trial_source_rows_builder_kind = "geometry_manual_trial_source_rows"

    if callable(manual_dataset_bindings.geometry_manual_source_rows_for_background):
        simulated_peaks = manual_dataset_bindings.geometry_manual_source_rows_for_background(
            int(background_idx),
            params_i,
            consumer="geometry_fit_dataset",
            required_pairs=selected_entries,
        )
    else:
        simulated_peaks = manual_dataset_bindings.geometry_manual_simulated_peaks_for_params(
            params_i,
            prefer_cache=True,
        )
    simulated_peaks = _project_source_rows_for_current_view(simulated_peaks)
    simulation_diagnostics = _current_simulation_diagnostics()
    if not simulated_peaks and callable(
        manual_dataset_bindings.geometry_manual_rebuild_source_rows_for_background
    ):
        _emit_geometry_fit_stage_event(
            stage_callback,
            "source_snapshot_rebuild",
            background_index=int(background_idx),
            message=(
                "preflight: rebuilding simulated source rows for "
                f"background {int(background_idx) + 1}"
            ),
        )
        simulated_peaks = (
            manual_dataset_bindings.geometry_manual_rebuild_source_rows_for_background(
                int(background_idx),
                params_i,
                consumer="geometry_fit_dataset",
                prior_diagnostics=dict(simulation_diagnostics),
                required_pairs=selected_entries,
            )
        )
        simulated_peaks = _project_source_rows_for_current_view(simulated_peaks)
        simulation_diagnostics = _current_simulation_diagnostics()
    simulated_peaks, simulation_diagnostics = _augment_source_rows_with_provider_coverage(
        simulated_peaks,
        simulation_diagnostics,
    )
    if not simulated_peaks:
        snapshot_status = str(simulation_diagnostics.get("status", "<unknown>"))
        raw_peak_count = int(simulation_diagnostics.get("raw_peak_count", 0) or 0)
        projected_peak_count = int(simulation_diagnostics.get("projected_peak_count", 0) or 0)
        final_returned_row_count = int(
            simulation_diagnostics.get("final_returned_row_count", 0) or 0
        )
        filter_reason = str(
            simulation_diagnostics.get(
                "snapshot_filter_reason",
                simulation_diagnostics.get(
                    "projection_failure_reason",
                    simulation_diagnostics.get("reason", "empty_returned_rows"),
                ),
            )
            or "empty_returned_rows"
        )
        if snapshot_status == "snapshot_hit":
            snapshot_status = "snapshot_hit_empty_returned_rows"
            simulation_diagnostics["status"] = snapshot_status
            simulation_diagnostics["stale_reason"] = (
                "snapshot_hit returned zero rows "
                f"(raw_peak_count={raw_peak_count}, "
                f"projected_peak_count={projected_peak_count}, "
                f"final_returned_row_count={final_returned_row_count}, "
                f"filter_reason={filter_reason})"
            )
        exception_type = str(simulation_diagnostics.get("exception_type", "")).strip()
        exception_message = str(simulation_diagnostics.get("exception_message", "")).strip()
        error_text = (
            "Geometry-fit source snapshot unavailable for "
            f"background {int(background_idx) + 1} (status={snapshot_status}; "
            f"raw_peak_count={raw_peak_count}; projected_peak_count={projected_peak_count}; "
            f"final_returned_row_count={final_returned_row_count}; "
            f"filter_reason={filter_reason})."
        )
        if exception_type or exception_message:
            error_text = (
                f"{error_text[:-1]} "
                f"Runtime={exception_type or '<unknown>'}: "
                f"{exception_message or '<no message>'}."
            )
        _emit_geometry_fit_stage_event(
            stage_callback,
            "dataset_failed",
            background_index=int(background_idx),
            status=snapshot_status,
            message=(
                f"preflight: source snapshot unavailable for background {int(background_idx) + 1}"
            ),
        )
        raise RuntimeError(error_text)
    simulated_lookup = manual_dataset_bindings.geometry_manual_simulated_lookup(simulated_peaks)

    def _source_row_key(
        entry: Mapping[str, object] | None,
    ) -> tuple[int, int] | None:
        return _geometry_fit_source_row_key(entry)

    def _source_reflection_row_key(
        entry: Mapping[str, object] | None,
    ) -> tuple[int, int] | None:
        return _geometry_fit_source_reflection_row_key(entry)

    def _coerce_nonnegative_index(value: object) -> int | None:
        return _geometry_fit_coerce_nonnegative_index(value)

    def _trusted_full_reflection_identity(
        entry: Mapping[str, object] | None,
    ) -> bool:
        return _geometry_fit_trusted_full_reflection_identity(entry)

    def _source_branch_resolution(
        entry: Mapping[str, object] | None,
    ) -> tuple[int | None, str | None]:
        return _geometry_fit_source_branch_resolution(entry)

    def _source_branch_index(
        entry: Mapping[str, object] | None,
    ) -> int | None:
        branch_idx, _branch_source = _source_branch_resolution(entry)
        return branch_idx

    def _source_peak_key(
        entry: Mapping[str, object] | None,
    ) -> tuple[int, int] | None:
        return _geometry_fit_source_peak_key(entry)

    def _source_locator_payload(
        entry: Mapping[str, object] | None,
    ) -> dict[str, object]:
        if not isinstance(entry, Mapping):
            return {
                "source_reflection_index": None,
                "source_reflection_namespace": None,
                "source_reflection_is_full": None,
                "source_table_index": None,
                "source_row_index": None,
                "source_branch_index": None,
                "source_peak_index": None,
            }
        return {
            "source_reflection_index": entry.get("source_reflection_index"),
            "source_reflection_namespace": entry.get("source_reflection_namespace"),
            "source_reflection_is_full": entry.get("source_reflection_is_full"),
            "source_table_index": entry.get("source_table_index"),
            "source_row_index": entry.get("source_row_index"),
            "source_branch_index": _source_branch_index(entry),
            "source_peak_index": entry.get("source_peak_index"),
        }

    def _source_locator_identity_match(
        saved_entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
    ) -> bool:
        saved_locator = _source_locator_payload(saved_entry)
        candidate_locator = _source_locator_payload(candidate)
        comparable_keys = [key for key, value in saved_locator.items() if value is not None]
        if not comparable_keys:
            return False
        return all(candidate_locator.get(key) == saved_locator.get(key) for key in comparable_keys)

    def _normalized_hkl(
        value: object,
    ) -> tuple[int, int, int] | None:
        return _geometry_fit_normalized_hkl(value)

    simulated_by_peak: dict[tuple[int, int], dict[str, object]] = {}
    simulated_by_reflection_row: dict[tuple[int, int], dict[str, object]] = {}
    simulated_by_group: dict[object, list[dict[str, object]]] = {}
    simulated_by_hkl: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    simulated_by_source_hkl: dict[tuple[str, int, int, int], list[dict[str, object]]] = {}
    for raw_entry in simulated_peaks or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        entry["source_label"] = _geometry_fit_entry_source_label(entry)
        reflection_row_key = _source_reflection_row_key(entry)
        if reflection_row_key is not None:
            simulated_by_reflection_row[reflection_row_key] = entry
        peak_key = _source_peak_key(entry)
        if peak_key is not None:
            simulated_by_peak[peak_key] = entry
        group_key = _geometry_fit_group_identity(entry)
        if group_key is not None:
            simulated_by_group.setdefault(group_key, []).append(entry)
        hkl_key = _normalized_hkl(entry.get("hkl"))
        if hkl_key is not None:
            simulated_by_hkl.setdefault(hkl_key, []).append(entry)
            simulated_by_source_hkl.setdefault(
                (_geometry_fit_entry_source_label(entry), *hkl_key),
                [],
            ).append(entry)

    def _entry_display_point(
        entry: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        coords = (
            manual_dataset_bindings.geometry_manual_entry_display_coords(entry)
            if isinstance(entry, Mapping)
            else None
        )
        if coords is None or len(coords) < 2:
            return None
        try:
            col = float(coords[0])
            row = float(coords[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _entry_saved_simulated_current_view_point(
        entry: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        if not isinstance(entry, Mapping):
            return None
        current_view_point = _entry_point(entry, "display_col", "display_row")
        if current_view_point is not None:
            return current_view_point
        if use_caked_display:
            return _entry_point(
                entry,
                "refined_sim_caked_x",
                "refined_sim_caked_y",
            )
        return _entry_point(entry, "refined_sim_x", "refined_sim_y")

    def _candidate_current_view_point(
        entry: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        if not isinstance(entry, Mapping):
            return None
        if use_caked_display:
            current_view_point = _caked_angle_pair(
                entry,
                x_keys=("caked_x", "two_theta_deg"),
                y_keys=("caked_y", "phi_deg"),
            )
            if current_view_point is not None:
                return current_view_point
            return None
        current_view_point = _entry_point(entry, "sim_col", "sim_row")
        if current_view_point is not None:
            return current_view_point
        has_measured_background_hint = any(
            key in entry for key in ("x", "y", "raw_x", "raw_y", "detector_x", "detector_y")
        )
        has_live_simulated_shape = any(
            key in entry
            for key in (
                "source_reflection_index",
                "source_table_index",
                "source_row_index",
                "source_peak_index",
                "sim_col",
                "sim_row",
                "sim_col_raw",
                "sim_row_raw",
                "native_col",
                "native_row",
            )
        )
        if has_live_simulated_shape and not has_measured_background_hint:
            current_view_point = _entry_point(entry, "display_col", "display_row")
            if current_view_point is not None:
                return current_view_point
        return None

    def _candidate_current_view_frame(
        entry: Mapping[str, object] | None,
    ) -> str | None:
        if not isinstance(entry, Mapping):
            return None
        if use_caked_display:
            return "caked_display" if _candidate_current_view_point(entry) is not None else None
        return "current_view_display" if _candidate_current_view_point(entry) is not None else None

    def _background_current_view_frame(
        entry: Mapping[str, object] | None,
    ) -> str | None:
        if use_caked_display:
            if _entry_has_stale_caked_fields(entry):
                return None
            return (
                "caked_display"
                if (
                    _entry_point(entry, "raw_caked_x", "raw_caked_y") is not None
                    or _entry_point(entry, "caked_x", "caked_y") is not None
                )
                else None
            )
        return "current_view_display" if _entry_display_point(entry) is not None else None

    def _source_entry_branch_matches(
        entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
    ) -> bool:
        return _geometry_fit_source_entry_branch_matches(entry, candidate)

    def _source_entry_group_matches(
        entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
    ) -> bool:
        return _candidate_matches_group_constraints(entry, candidate)

    def _source_entry_hkl_matches(
        entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
    ) -> bool:
        return _geometry_fit_source_entry_hkl_matches(entry, candidate)

    def _filter_hkl_candidates(
        entry: Mapping[str, object] | None,
        candidates: Sequence[dict[str, object]] | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        filtered: list[dict[str, object]] = []
        excluded_missing_hkl: list[dict[str, object]] = []
        excluded_mismatched_hkl: list[dict[str, object]] = []
        target_hkl = _normalized_hkl(entry.get("hkl") if isinstance(entry, Mapping) else None)
        for raw_candidate in candidates or ():
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = dict(raw_candidate)
            candidate_hkl = _normalized_hkl(candidate.get("hkl"))
            if candidate_hkl is None:
                excluded_missing_hkl.append(candidate)
                continue
            if (
                target_hkl is not None
                and tuple(candidate_hkl) != tuple(target_hkl)
                and not _geometry_fit_source_entry_hkl_matches(entry, candidate)
            ):
                excluded_mismatched_hkl.append(candidate)
                continue
            filtered.append(candidate)
        return filtered, excluded_missing_hkl, excluded_mismatched_hkl

    def _candidate_matches_group_constraints(
        entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
    ) -> bool:
        if not isinstance(entry, Mapping):
            return True
        if not isinstance(candidate, Mapping):
            return False
        if _geometry_fit_entry_source_label(candidate) != _geometry_fit_entry_source_label(entry):
            return False
        required_branch_group = _geometry_fit_stable_group_identity(entry.get("branch_group_key"))
        if required_branch_group is not None:
            candidate_branch_group = _geometry_fit_stable_group_identity(
                candidate.get("branch_group_key")
            )
            if candidate_branch_group != required_branch_group:
                return False

        required_q_groups: list[object] = []
        for key in ("q_group_key", "source_q_group_key"):
            required_q_group = _geometry_fit_stable_group_identity(entry.get(key))
            if required_q_group is not None and not any(
                existing == required_q_group for existing in required_q_groups
            ):
                required_q_groups.append(required_q_group)
        if required_q_groups:
            candidate_q_groups = [
                _geometry_fit_stable_group_identity(candidate.get(key))
                for key in ("q_group_key", "source_q_group_key")
            ]
            candidate_q_groups = [value for value in candidate_q_groups if value is not None]
            if not candidate_q_groups:
                return False
            for required_q_group in required_q_groups:
                if not any(
                    candidate_q_group == required_q_group
                    for candidate_q_group in candidate_q_groups
                ):
                    return False
        return True

    def _filter_group_candidates(
        entry: Mapping[str, object] | None,
        candidates: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        return [
            dict(candidate)
            for candidate in (candidates or ())
            if isinstance(candidate, Mapping)
            and _candidate_matches_group_constraints(entry, candidate)
        ]

    def _filter_source_branch_candidates(
        entry: Mapping[str, object] | None,
        candidates: Sequence[dict[str, object]] | None,
        *,
        required_branch_index: int | None = None,
    ) -> list[dict[str, object]]:
        candidate_pool = [
            dict(candidate) for candidate in (candidates or ()) if isinstance(candidate, Mapping)
        ]
        if not candidate_pool:
            return []
        if _geometry_fit_is_zero_qr_00l(entry):
            return candidate_pool
        branch_idx = (
            int(required_branch_index)
            if required_branch_index in {0, 1}
            else _geometry_fit_source_branch_index(entry)
        )
        if branch_idx in {0, 1}:
            return [
                dict(candidate)
                for candidate in candidate_pool
                if _geometry_fit_source_branch_index(candidate) == int(branch_idx)
            ]
        return candidate_pool

    def _source_candidate_filter_inventory(
        entry: Mapping[str, object] | None,
        candidates: Sequence[dict[str, object]] | None,
        *,
        required_branch_index: int | None = None,
    ) -> dict[str, object]:
        candidate_pool = [
            dict(candidate) for candidate in (candidates or ()) if isinstance(candidate, Mapping)
        ]
        hkl_candidates, missing_hkl_candidates, mismatched_hkl_candidates = _filter_hkl_candidates(
            entry,
            candidate_pool,
        )
        group_candidates = _filter_group_candidates(entry, hkl_candidates)
        branch_candidates = _filter_source_branch_candidates(
            entry,
            group_candidates,
            required_branch_index=required_branch_index,
        )
        return {
            "candidate_pool": candidate_pool,
            "hkl_candidates": hkl_candidates,
            "missing_hkl_candidates": missing_hkl_candidates,
            "mismatched_hkl_candidates": mismatched_hkl_candidates,
            "group_candidates": group_candidates,
            "branch_candidates": branch_candidates,
        }

    def _select_source_candidate(
        entry: Mapping[str, object] | None,
        candidates: Sequence[dict[str, object]] | None,
        *,
        saved_identity_entry: Mapping[str, object] | None = None,
        target_point: tuple[float, float] | None = None,
        frame_name: str | None = None,
        tie_tolerance: float = 0.0,
        required_branch_index: int | None = None,
        fail_on_ambiguous_tie: bool = False,
        require_all_candidate_points: bool = False,
        allow_missing_target_selection: bool = True,
        allow_single_candidate_without_score: bool = False,
    ) -> dict[str, object]:
        inventory = _source_candidate_filter_inventory(
            entry,
            candidates,
            required_branch_index=required_branch_index,
        )
        branch_candidates = [
            dict(candidate)
            for candidate in inventory.get("branch_candidates", [])
            if isinstance(candidate, Mapping)
        ]
        result: dict[str, object] = {
            **inventory,
            "selected": None,
            "selected_score": None,
            "selected_point": None,
            "selected_frame": frame_name,
            "scored_candidates": [],
            "score_inventory": [],
            "best_score": None,
            "second_best_score": None,
            "tied_candidates": [],
            "identity_tied_candidates": [],
            "selection_tie_breaker": None,
            "failure_reason": None,
        }
        if not inventory.get("hkl_candidates"):
            result["failure_reason"] = "missing_candidate_pool"
            return result
        if not inventory.get("group_candidates"):
            result["failure_reason"] = "group_constraint_no_match"
            return result
        if not branch_candidates:
            result["failure_reason"] = "branch_constraint_no_match"
            return result

        resolved_target_point = target_point
        if resolved_target_point is None:
            resolved_target_point = _entry_display_point(entry)
        if resolved_target_point is None:
            resolved_target_point = _entry_saved_simulated_current_view_point(entry)
        if resolved_target_point is None:
            if allow_missing_target_selection or (
                allow_single_candidate_without_score and len(branch_candidates) == 1
            ):
                selected = dict(branch_candidates[0])
                result["selected"] = selected
                result["selection_tie_breaker"] = "missing_target_point_first_candidate"
                return result
            result["failure_reason"] = "missing_target_point"
            return result

        scored_candidates: list[
            tuple[float, dict[str, object], tuple[float, float], str | None]
        ] = []
        for candidate in branch_candidates:
            candidate_point = (
                _candidate_point_for_frame(candidate, frame_name=str(frame_name))
                if frame_name is not None
                else _candidate_current_view_point(candidate)
            )
            if candidate_point is None:
                if require_all_candidate_points:
                    result["failure_reason"] = "missing_candidate_point"
                    return result
                continue
            score = float(
                math.hypot(
                    float(candidate_point[0]) - float(resolved_target_point[0]),
                    float(candidate_point[1]) - float(resolved_target_point[1]),
                )
            )
            if not np.isfinite(float(score)):
                result["failure_reason"] = "nonfinite_score"
                return result
            candidate_frame = (
                str(frame_name)
                if frame_name is not None
                else _candidate_current_view_frame(candidate)
            )
            scored_candidates.append(
                (
                    float(score),
                    dict(candidate),
                    (float(candidate_point[0]), float(candidate_point[1])),
                    candidate_frame,
                )
            )

        if not scored_candidates:
            if allow_single_candidate_without_score and len(branch_candidates) == 1:
                selected = dict(branch_candidates[0])
                result["selected"] = selected
                result["selection_tie_breaker"] = "single_candidate_without_geometry"
                return result
            result["failure_reason"] = "missing_candidate_point"
            return result

        scored_candidates.sort(
            key=lambda item: (
                float(item[0]),
                *_candidate_sort_identity(item[1]),
            )
        )
        score_inventory = [
            {
                **(_geometry_fit_compact_source_resolution_entry_payload(candidate) or {}),
                "score": float(score),
                "frame_name": str(candidate_frame or ""),
                "candidate_point": [
                    float(candidate_point[0]),
                    float(candidate_point[1]),
                ],
                "saved_target_point": [
                    float(resolved_target_point[0]),
                    float(resolved_target_point[1]),
                ],
            }
            for score, candidate, candidate_point, candidate_frame in scored_candidates
        ]
        best_score = float(scored_candidates[0][0])
        second_best_score = float(scored_candidates[1][0]) if len(scored_candidates) > 1 else None
        try:
            tie_window = float(tie_tolerance)
        except Exception:
            tie_window = 0.0
        if not np.isfinite(float(tie_window)) or tie_window < 0.0:
            tie_window = 0.0
        tied_scored = [
            item
            for item in scored_candidates
            if abs(float(item[0]) - float(best_score)) <= float(tie_window)
        ]
        identity_tied_scored = [
            item
            for item in tied_scored
            if _source_locator_identity_match(saved_identity_entry, item[1])
        ]
        chosen = tied_scored[0]
        selection_tie_breaker: str | None = None
        tied_candidates = [dict(item[1]) for item in tied_scored]
        identity_tied_candidates = [dict(item[1]) for item in identity_tied_scored]
        if fail_on_ambiguous_tie and len(tied_scored) > 1:
            deduped_tied_candidates = _dedupe_geometry_tied_candidates(
                tied_candidates,
                frame_name=str(frame_name or ""),
            )
            tied_candidates = [dict(candidate) for candidate in deduped_tied_candidates]
            identity_tied_candidates = [
                dict(candidate)
                for candidate in deduped_tied_candidates
                if _source_locator_identity_match(saved_identity_entry, candidate)
            ]
            if len(deduped_tied_candidates) == 1:
                selection_tie_breaker = "duplicate_live_rows_canonicalized"
                chosen_candidate = dict(deduped_tied_candidates[0])
                chosen = next(
                    (item for item in scored_candidates if dict(item[1]) == chosen_candidate),
                    scored_candidates[0],
                )
            elif len(identity_tied_candidates) == 1:
                selection_tie_breaker = "saved_source_identity"
                chosen_candidate = dict(identity_tied_candidates[0])
                chosen = next(
                    (item for item in scored_candidates if dict(item[1]) == chosen_candidate),
                    scored_candidates[0],
                )
            else:
                result.update(
                    {
                        "scored_candidates": scored_candidates,
                        "score_inventory": score_inventory,
                        "best_score": best_score,
                        "second_best_score": second_best_score,
                        "tied_candidates": tied_candidates,
                        "identity_tied_candidates": identity_tied_candidates,
                        "failure_reason": "ambiguous_geometry_tie",
                    }
                )
                return result
        elif len(identity_tied_scored) == 1:
            chosen = identity_tied_scored[0]
            if len(tied_scored) > 1:
                selection_tie_breaker = "saved_source_identity"

        result.update(
            {
                "selected": dict(chosen[1]),
                "selected_score": float(chosen[0]),
                "selected_point": [
                    float(chosen[2][0]),
                    float(chosen[2][1]),
                ],
                "selected_frame": chosen[3],
                "scored_candidates": scored_candidates,
                "score_inventory": score_inventory,
                "best_score": best_score,
                "second_best_score": second_best_score,
                "tied_candidates": tied_candidates,
                "identity_tied_candidates": identity_tied_candidates,
                "selection_tie_breaker": selection_tie_breaker,
            }
        )
        return result

    def _cached_source_entry_candidates(
        entry: Mapping[str, object] | None,
    ) -> list[dict[str, object]]:
        if not isinstance(entry, Mapping):
            return []
        candidates: list[dict[str, object]] = []
        reflection_row_key = _source_reflection_row_key(entry)
        if reflection_row_key is not None:
            candidate = simulated_by_reflection_row.get(reflection_row_key)
            if isinstance(candidate, Mapping):
                resolved = dict(candidate)
                resolved.setdefault("source_row_index", int(reflection_row_key[1]))
                candidates.append(resolved)
        row_key = _source_row_key(entry)
        if row_key is not None:
            candidate = simulated_lookup.get(row_key)
            if isinstance(candidate, Mapping):
                resolved = dict(candidate)
                resolved.setdefault("source_table_index", int(row_key[0]))
                resolved.setdefault("source_row_index", int(row_key[1]))
                candidates.append(resolved)
        peak_key = _source_peak_key(entry)
        if peak_key is not None:
            candidate = simulated_by_peak.get(peak_key)
            if candidate is not None:
                resolved = dict(candidate)
                resolved.setdefault("source_table_index", int(peak_key[0]))
                resolved.setdefault("source_branch_index", int(peak_key[1]))
                resolved.setdefault("source_peak_index", int(peak_key[1]))
                candidates.append(resolved)
        return candidates

    def _resolve_cached_source_entry(
        entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        for resolved in _cached_source_entry_candidates(entry):
            if (
                _source_entry_hkl_matches(entry, resolved)
                and _source_entry_branch_matches(entry, resolved)
                and _source_entry_group_matches(entry, resolved)
            ):
                return dict(resolved)
        return None

    def _resolve_source_entry_candidate_pool(
        entry: Mapping[str, object] | None,
    ) -> tuple[list[dict[str, object]], str | None]:
        if not isinstance(entry, Mapping):
            return [], None
        group_key = _geometry_fit_group_identity(entry)
        if group_key is not None:
            group_pool = [dict(item) for item in simulated_by_group.get(group_key, ())]
            return (
                group_pool,
                "q_group" if entry.get("q_group_key") is not None else "group",
            )
        hkl_key = _normalized_hkl(entry.get("hkl"))
        if hkl_key is not None:
            source_hkl_key = (_geometry_fit_entry_source_label(entry), *hkl_key)
            hkl_pool = [dict(item) for item in simulated_by_source_hkl.get(source_hkl_key, ())]
            if not hkl_pool:
                hkl_pool = [
                    dict(item)
                    for item in simulated_by_hkl.get(hkl_key, ())
                    if _geometry_fit_entry_source_label(item)
                    == _geometry_fit_entry_source_label(entry)
                ]
            if hkl_pool:
                return hkl_pool, "source_hkl"
        return [], None

    def _resolve_source_entry(
        entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        candidate_pool, _candidate_pool_source = _resolve_source_entry_candidate_pool(entry)
        candidate_pool = [
            *_cached_source_entry_candidates(entry),
            *candidate_pool,
        ]
        selection = _select_source_candidate(
            entry,
            candidate_pool,
            saved_identity_entry=entry,
            tie_tolerance=0.0,
            fail_on_ambiguous_tie=False,
            require_all_candidate_points=False,
            allow_missing_target_selection=True,
            allow_single_candidate_without_score=False,
        )
        selected = selection.get("selected")
        return dict(selected) if isinstance(selected, Mapping) else None

    def _normalized_q_group_key(
        value: object,
    ) -> tuple[object, ...] | None:
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return None

    def _resolved_source_distance_sq(
        entry: Mapping[str, object],
        resolved_source_entry: Mapping[str, object] | None,
    ) -> float:
        display_point = _entry_display_point(entry)
        if display_point is None or not isinstance(resolved_source_entry, Mapping):
            return float("inf")
        resolved_source_point = _candidate_current_view_point(resolved_source_entry)
        if resolved_source_point is None:
            return float("inf")
        return float(
            (float(resolved_source_point[0]) - float(display_point[0])) ** 2
            + (float(resolved_source_point[1]) - float(display_point[1])) ** 2
        )

    def _source_candidate_status(
        entry: Mapping[str, object] | None,
        candidate: Mapping[str, object] | None,
        *,
        key_kind: str,
    ) -> str:
        if key_kind == "row":
            if _source_row_key(entry) is None:
                return "no_saved_row"
        elif key_kind == "peak":
            if _source_peak_key(entry) is None:
                return "no_saved_peak"
        if not isinstance(candidate, Mapping):
            return "missing"
        if not _source_entry_branch_matches(entry, candidate):
            return "branch_mismatch"
        if not _source_entry_group_matches(entry, candidate):
            return "group_mismatch"
        if _source_entry_hkl_matches(entry, candidate):
            return "matched"
        return "hkl_mismatch"

    def _resolved_source_kind(
        entry: Mapping[str, object] | None,
        resolved_source_entry: Mapping[str, object] | None,
    ) -> str | None:
        if not (isinstance(entry, Mapping) and isinstance(resolved_source_entry, Mapping)):
            return None
        row_key = _source_row_key(entry)
        if row_key is not None and row_key == _source_row_key(resolved_source_entry):
            return "source_row"
        peak_key = _source_peak_key(entry)
        if peak_key is not None and peak_key == _source_peak_key(resolved_source_entry):
            return "source_peak"
        q_group_key = entry.get("q_group_key")
        if q_group_key is not None and q_group_key == resolved_source_entry.get("q_group_key"):
            return "q_group_fallback"
        group_key = _geometry_fit_group_identity(entry)
        if (
            group_key is not None
            and _geometry_fit_group_identity(resolved_source_entry) == group_key
        ):
            return "group_fallback"
        if _source_entry_hkl_matches(entry, resolved_source_entry):
            return "hkl_fallback"
        return "override"

    def _uses_legacy_dense_source_identity(
        entry: Mapping[str, object] | None,
    ) -> bool:
        row_key = _source_row_key(entry)
        if row_key is None:
            return False
        legacy_peak_index = _coerce_nonnegative_index(
            entry.get("source_peak_index") if isinstance(entry, Mapping) else None
        )
        if legacy_peak_index is None or legacy_peak_index in {0, 1}:
            return False
        reflection_idx = _coerce_nonnegative_index(
            entry.get("source_reflection_index") if isinstance(entry, Mapping) else None
        )
        table_idx = _coerce_nonnegative_index(
            entry.get("source_table_index") if isinstance(entry, Mapping) else None
        )
        anchor_idx = reflection_idx if reflection_idx is not None else table_idx
        if anchor_idx is None:
            return False
        return int(row_key[1]) == 0 and int(legacy_peak_index) == int(anchor_idx)

    def _legacy_dense_working_entry(
        entry: Mapping[str, object] | None,
        raw_saved_entry: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        if not isinstance(entry, Mapping):
            return None
        legacy_source = raw_saved_entry if isinstance(raw_saved_entry, Mapping) else entry
        if not _uses_legacy_dense_source_identity(legacy_source):
            return None
        working = dict(entry)
        if isinstance(legacy_source, Mapping):
            if "source_reflection_index" in legacy_source:
                working["legacy_source_reflection_index"] = legacy_source.get(
                    "source_reflection_index"
                )
            if "source_peak_index" in legacy_source:
                working["legacy_source_peak_index"] = legacy_source.get("source_peak_index")
        for key in (
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
        ):
            working.pop(key, None)
        return working

    def _entry_point(
        entry: Mapping[str, object] | None,
        x_key: str,
        y_key: str,
    ) -> tuple[float, float] | None:
        if not isinstance(entry, Mapping):
            return None
        try:
            col = float(entry.get(x_key, np.nan))
            row = float(entry.get(y_key, np.nan))
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _candidate_point_for_frame(
        candidate: Mapping[str, object] | None,
        *,
        frame_name: str,
    ) -> tuple[float, float] | None:
        if not isinstance(candidate, Mapping):
            return None
        if frame_name == "refined_sim_native":
            return _entry_point(candidate, "sim_col_raw", "sim_row_raw") or _entry_point(
                candidate,
                "sim_native_x",
                "sim_native_y",
            )
        if frame_name == "refined_sim_caked":
            return _caked_angle_pair(
                candidate,
                x_keys=("caked_x", "two_theta_deg"),
                y_keys=("caked_y", "phi_deg"),
            )
        if frame_name == "refined_sim_display":
            return _candidate_current_view_point(candidate)
        if frame_name == "measured_detector":
            return _entry_point(candidate, "sim_col_raw", "sim_row_raw") or _entry_point(
                candidate,
                "sim_native_x",
                "sim_native_y",
            )
        if frame_name == "measured_display":
            return _candidate_current_view_point(candidate)
        return None

    def _legacy_branch_hint_resolution(
        entry: Mapping[str, object] | None,
    ) -> tuple[int | None, str | None, str | None]:
        if not isinstance(entry, Mapping):
            return None, None, "missing_branch_hint"
        for key in (
            "refined_sim_caked_y",
            "caked_y",
            "raw_caked_y",
            "simulated_phi_deg",
            "background_phi_deg",
            "phi_deg",
            "phi",
        ):
            value = _finite_float(entry.get(key))
            if value is None:
                continue
            branch_idx = source_branch_index_from_phi_deg(
                value,
                zero_deadband_deg=SOURCE_BRANCH_PHI_ZERO_DEADBAND_DEG,
            )
            if branch_idx in {0, 1}:
                return int(branch_idx), str(key), None
            return None, str(key), "ambiguous_branch_deadband"
        return None, None, "missing_branch_hint"

    def _legacy_geometry_hint(
        entry: Mapping[str, object] | None,
        candidates: Sequence[Mapping[str, object]] | None,
    ) -> tuple[str | None, tuple[float, float] | None, float]:
        candidate_pool = [
            candidate for candidate in (candidates or ()) if isinstance(candidate, Mapping)
        ]
        for frame_name, point, tolerance in (
            (
                "measured_display",
                _entry_display_point(entry),
                float(GEOMETRY_FIT_LEGACY_REBIND_PIXEL_TIE_TOLERANCE_PX),
            ),
            (
                "refined_sim_display",
                _entry_point(entry, "refined_sim_x", "refined_sim_y"),
                float(GEOMETRY_FIT_LEGACY_REBIND_PIXEL_TIE_TOLERANCE_PX),
            ),
            (
                "refined_sim_native",
                _entry_point(entry, "refined_sim_native_x", "refined_sim_native_y"),
                float(GEOMETRY_FIT_LEGACY_REBIND_PIXEL_TIE_TOLERANCE_PX),
            ),
            (
                "refined_sim_caked",
                (
                    _entry_point(entry, "refined_sim_caked_x", "refined_sim_caked_y")
                    if use_caked_display
                    else None
                ),
                float(GEOMETRY_FIT_LEGACY_REBIND_CAKED_TIE_TOLERANCE),
            ),
            (
                "measured_detector",
                _entry_point(entry, "detector_x", "detector_y"),
                float(GEOMETRY_FIT_LEGACY_REBIND_PIXEL_TIE_TOLERANCE_PX),
            ),
        ):
            if point is None:
                continue
            if candidate_pool and all(
                _candidate_point_for_frame(candidate, frame_name=frame_name) is not None
                for candidate in candidate_pool
            ):
                return str(frame_name), (float(point[0]), float(point[1])), float(tolerance)
        return None, None, float("nan")

    def _candidate_sort_identity(candidate: Mapping[str, object]) -> tuple[int, int, int, int]:
        sentinel = 1 << 30
        reflection_idx = _geometry_fit_coerce_nonnegative_index(
            candidate.get("source_reflection_index") if isinstance(candidate, Mapping) else None
        )
        table_idx = _geometry_fit_coerce_nonnegative_index(
            candidate.get("source_table_index") if isinstance(candidate, Mapping) else None
        )
        row_idx = _geometry_fit_coerce_nonnegative_index(
            candidate.get("source_row_index") if isinstance(candidate, Mapping) else None
        )
        branch_idx = _geometry_fit_source_branch_index(candidate)
        return (
            int(reflection_idx) if reflection_idx is not None else sentinel,
            int(table_idx) if table_idx is not None else sentinel,
            int(row_idx) if row_idx is not None else sentinel,
            int(branch_idx) if branch_idx in {0, 1} else sentinel,
        )

    def _dedupe_geometry_tied_candidates(
        candidates: Sequence[Mapping[str, object]] | None,
        *,
        frame_name: str,
    ) -> list[dict[str, object]]:
        unique_candidates: dict[tuple[object, ...], dict[str, object]] = {}
        for raw_candidate in candidates or ():
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = dict(raw_candidate)
            dedupe_key = (
                _normalized_hkl(candidate.get("hkl")),
                _source_branch_index(candidate),
                str(frame_name),
                _candidate_point_for_frame(candidate, frame_name=str(frame_name)),
                tuple(sorted(_source_locator_payload(candidate).items())),
            )
            existing = unique_candidates.get(dedupe_key)
            if existing is None or _candidate_sort_identity(candidate) < _candidate_sort_identity(
                existing
            ):
                unique_candidates[dedupe_key] = candidate
        return [
            dict(candidate)
            for _key, candidate in sorted(
                unique_candidates.items(),
                key=lambda item: _candidate_sort_identity(item[1]),
            )
        ]

    def _resolve_legacy_dense_source_entry(
        entry: Mapping[str, object] | None,
        *,
        raw_saved_entry: Mapping[str, object] | None,
    ) -> tuple[dict[str, object] | None, str | None, dict[str, object]]:
        diagnostics: dict[str, object] = {
            "legacy_raw_saved_entry": _geometry_fit_compact_source_resolution_entry_payload(
                raw_saved_entry
            ),
            "legacy_normalized_saved_entry": _geometry_fit_compact_source_resolution_entry_payload(
                entry
            ),
        }
        working_entry = _legacy_dense_working_entry(entry, raw_saved_entry)
        saved_identity_entry = raw_saved_entry if isinstance(raw_saved_entry, Mapping) else entry
        diagnostics["legacy_working_entry"] = _geometry_fit_compact_source_resolution_entry_payload(
            working_entry
        )
        if not isinstance(working_entry, Mapping):
            return None, None, diagnostics

        candidate_pool, candidate_pool_source = _resolve_source_entry_candidate_pool(working_entry)
        candidate_inventory = _source_candidate_filter_inventory(
            working_entry,
            candidate_pool,
        )
        hkl_candidates = [
            dict(candidate)
            for candidate in candidate_inventory.get("hkl_candidates", [])
            if isinstance(candidate, Mapping)
        ]
        group_candidates = [
            dict(candidate)
            for candidate in candidate_inventory.get("group_candidates", [])
            if isinstance(candidate, Mapping)
        ]
        missing_hkl_candidates = [
            dict(candidate)
            for candidate in candidate_inventory.get("missing_hkl_candidates", [])
            if isinstance(candidate, Mapping)
        ]
        mismatched_hkl_candidates = [
            dict(candidate)
            for candidate in candidate_inventory.get("mismatched_hkl_candidates", [])
            if isinstance(candidate, Mapping)
        ]
        diagnostics["legacy_candidate_pool_source"] = candidate_pool_source
        diagnostics["legacy_candidate_count_initial"] = int(len(hkl_candidates))
        diagnostics["legacy_candidate_inventory"] = _geometry_fit_trace_candidate_inventory(
            hkl_candidates
        )
        diagnostics["legacy_candidate_count_after_hkl"] = int(len(hkl_candidates))
        diagnostics["legacy_excluded_missing_hkl_candidates"] = (
            _geometry_fit_trace_candidate_inventory(missing_hkl_candidates)
        )
        diagnostics["legacy_excluded_mismatched_hkl_candidates"] = (
            _geometry_fit_trace_candidate_inventory(mismatched_hkl_candidates)
        )
        diagnostics["legacy_candidate_count_after_group"] = int(len(group_candidates))
        diagnostics["legacy_candidate_inventory_after_group"] = (
            _geometry_fit_trace_candidate_inventory(group_candidates)
        )
        if not hkl_candidates or not group_candidates:
            diagnostics["legacy_failure_reason"] = "legacy_rebind_missing_candidate_pool"
            return None, None, diagnostics

        branch_idx, branch_source, branch_reason = _legacy_branch_hint_resolution(working_entry)
        diagnostics["legacy_branch_hint_source"] = branch_source
        diagnostics["legacy_branch_hint_reason"] = branch_reason
        if branch_idx not in {0, 1}:
            diagnostics["legacy_failure_reason"] = (
                f"legacy_rebind_{branch_reason or 'missing_branch_hint'}"
            )
            return None, None, diagnostics

        branch_inventory = _source_candidate_filter_inventory(
            working_entry,
            candidate_pool,
            required_branch_index=int(branch_idx),
        )
        branch_candidates = [
            dict(candidate)
            for candidate in branch_inventory.get("branch_candidates", [])
            if isinstance(candidate, Mapping)
        ]
        diagnostics["legacy_candidate_count_after_branch"] = int(len(branch_candidates))
        diagnostics["legacy_candidate_inventory_after_branch"] = (
            _geometry_fit_trace_candidate_inventory(branch_candidates)
        )
        if not branch_candidates:
            diagnostics["legacy_failure_reason"] = "legacy_rebind_no_candidate_on_branch"
            return None, None, diagnostics

        geometry_hint_source, geometry_target_point, tie_tolerance = _legacy_geometry_hint(
            working_entry,
            branch_candidates,
        )
        diagnostics["legacy_geometry_hint_source"] = geometry_hint_source
        diagnostics["legacy_tie_tolerance"] = (
            float(tie_tolerance) if np.isfinite(float(tie_tolerance)) else None
        )

        selection = _select_source_candidate(
            working_entry,
            candidate_pool,
            saved_identity_entry=saved_identity_entry,
            target_point=geometry_target_point,
            frame_name=geometry_hint_source,
            tie_tolerance=tie_tolerance,
            required_branch_index=int(branch_idx),
            fail_on_ambiguous_tie=True,
            require_all_candidate_points=True,
            allow_missing_target_selection=False,
            allow_single_candidate_without_score=True,
        )
        score_inventory = [
            dict(item) for item in selection.get("score_inventory", []) if isinstance(item, Mapping)
        ]
        diagnostics["legacy_candidate_count_after_geometry"] = int(
            len(score_inventory)
            if score_inventory
            else (1 if isinstance(selection.get("selected"), Mapping) else 0)
        )
        if score_inventory:
            diagnostics["legacy_geometry_candidate_scores"] = score_inventory
        if selection.get("best_score") is not None:
            diagnostics["legacy_best_score"] = float(selection["best_score"])
        if selection.get("second_best_score") is not None:
            diagnostics["legacy_second_best_score"] = float(selection["second_best_score"])
        else:
            diagnostics["legacy_second_best_score"] = None
        if selection.get("selection_tie_breaker"):
            diagnostics["legacy_selection_tie_breaker"] = str(
                selection.get("selection_tie_breaker")
            )
        chosen_live_row = (
            dict(selection.get("selected"))
            if isinstance(selection.get("selected"), Mapping)
            else None
        )
        if chosen_live_row is None:
            failure_reason = str(selection.get("failure_reason") or "")
            if failure_reason == "missing_target_point":
                diagnostics["legacy_failure_reason"] = "legacy_rebind_missing_geometry_hint"
            elif failure_reason == "missing_candidate_point":
                diagnostics["legacy_failure_reason"] = "legacy_rebind_missing_geometry_candidate"
            elif failure_reason == "nonfinite_score":
                diagnostics["legacy_failure_reason"] = "legacy_rebind_nonfinite_geometry_score"
            elif failure_reason == "branch_constraint_no_match":
                diagnostics["legacy_failure_reason"] = "legacy_rebind_no_candidate_on_branch"
            elif failure_reason == "ambiguous_geometry_tie":
                diagnostics["legacy_failure_reason"] = "legacy_rebind_ambiguous_geometry_tie"
                diagnostics["legacy_geometry_tied_candidates"] = (
                    _geometry_fit_trace_candidate_inventory(selection.get("tied_candidates", []))
                )
                diagnostics["legacy_geometry_identity_tied_candidates"] = (
                    _geometry_fit_trace_candidate_inventory(
                        selection.get("identity_tied_candidates", [])
                    )
                )
            else:
                diagnostics["legacy_failure_reason"] = (
                    "legacy_rebind_missing_candidate_pool"
                    if failure_reason in {"missing_candidate_pool", "group_constraint_no_match"}
                    else "legacy_rebind_ambiguous_geometry_tie"
                )
            return None, None, diagnostics

        canonical_live_row = gui_manual_geometry.geometry_manual_canonicalize_live_source_entry(
            chosen_live_row,
            allow_legacy_peak_fallback=False,
            preserve_existing_trusted_identity=True,
        )
        canonical_ok, canonical_reason = _geometry_fit_is_canonical_live_source_entry(
            canonical_live_row
        )
        if not canonical_ok or not isinstance(canonical_live_row, Mapping):
            diagnostics["legacy_failure_reason"] = (
                f"legacy_rebind_noncanonical_live_row:{canonical_reason or 'unknown'}"
            )
            return None, None, diagnostics

        fit_bound_entry = gui_manual_geometry.geometry_manual_apply_refined_simulated_override(
            dict(working_entry),
            dict(canonical_live_row),
            prefer_caked_display=use_caked_display,
        )
        canonical_fit_ok, canonical_fit_reason = _geometry_fit_is_canonical_live_source_entry(
            fit_bound_entry
        )
        if not canonical_fit_ok or not isinstance(fit_bound_entry, Mapping):
            diagnostics["legacy_failure_reason"] = (
                "legacy_rebind_fit_bound_identity_loss"
                if canonical_fit_reason is None
                else f"legacy_rebind_fit_bound_identity_loss:{canonical_fit_reason}"
            )
            return None, None, diagnostics

        diagnostics["legacy_chosen_live_row"] = (
            _geometry_fit_compact_source_resolution_entry_payload(canonical_live_row)
        )
        diagnostics["legacy_selected_source_identity_fields"] = _source_locator_payload(
            canonical_live_row
        )
        diagnostics["legacy_saved_background_current_view_point"] = _geometry_fit_cache_jsonable(
            _entry_display_point(entry)
        )
        diagnostics["legacy_saved_background_current_view_frame"] = _background_current_view_frame(
            entry
        )
        diagnostics["legacy_selected_live_simulated_current_view_point"] = (
            _geometry_fit_cache_jsonable(_candidate_current_view_point(canonical_live_row))
        )
        diagnostics["legacy_selected_live_simulated_current_view_frame"] = (
            _candidate_current_view_frame(canonical_live_row)
        )
        diagnostics["legacy_selected_to_background_distance_px"] = (
            float(
                math.hypot(
                    float(_candidate_current_view_point(canonical_live_row)[0])
                    - float(_entry_display_point(entry)[0]),
                    float(_candidate_current_view_point(canonical_live_row)[1])
                    - float(_entry_display_point(entry)[1]),
                )
            )
            if _candidate_current_view_frame(canonical_live_row)
            == _background_current_view_frame(entry)
            and _candidate_current_view_point(canonical_live_row) is not None
            and _entry_display_point(entry) is not None
            else None
        )
        diagnostics["legacy_saved_simulated_detector_hint"] = _geometry_fit_cache_jsonable(
            _entry_point(
                saved_identity_entry,
                "refined_sim_native_x",
                "refined_sim_native_y",
            )
            or _entry_saved_simulated_current_view_point(saved_identity_entry)
        )
        diagnostics["legacy_fit_bound_entry"] = (
            _geometry_fit_compact_source_resolution_entry_payload(fit_bound_entry)
        )
        resolution_kind = (
            "legacy_dense_q_group_rebind"
            if candidate_pool_source == "q_group"
            else "legacy_dense_hkl_rebind"
        )
        return dict(fit_bound_entry), resolution_kind, diagnostics

    def _strict_failure_reason(
        *,
        row_status: str,
        peak_status: str,
        overlay_kind: str | None,
    ) -> str:
        reasons: list[str] = []
        if row_status == "no_saved_row" and peak_status == "no_saved_peak":
            reasons.append("saved pair has no cached source row/peak identity")
        if row_status == "missing":
            reasons.append("saved source row is missing from the current simulated rows")
        elif row_status == "hkl_mismatch":
            reasons.append("saved source row HKL no longer matches the current row")
        elif row_status == "branch_mismatch":
            reasons.append("saved source row resolved to the opposite detector-side branch")
        elif row_status == "group_mismatch":
            reasons.append("saved source row resolved to a different branch group")
        if peak_status == "missing":
            reasons.append("saved source peak is missing from the current simulated peaks")
        elif peak_status == "hkl_mismatch":
            reasons.append("saved source peak HKL no longer matches the current peak")
        elif peak_status == "branch_mismatch":
            reasons.append("saved source peak resolved to the opposite detector-side branch")
        elif peak_status == "group_mismatch":
            reasons.append("saved source peak resolved to a different branch group")
        if overlay_kind == "q_group_fallback":
            reasons.append("only a q-group fallback candidate was available")
        elif overlay_kind == "hkl_fallback":
            reasons.append("only an HKL fallback candidate was available")
        elif overlay_kind == "override":
            reasons.append("only an override/refined fallback candidate was available")
        if not reasons:
            reasons.append("strict cached-source lookup returned no acceptable candidate")
        return "; ".join(reasons)

    selected_records: list[dict[str, object]] = []
    for selected_input in selected_entry_inputs:
        raw_saved_entry = (
            dict(selected_input.get("raw_saved_entry"))
            if isinstance(selected_input.get("raw_saved_entry"), Mapping)
            else None
        )
        raw_entry = selected_input.get("entry")
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        q_group_key = _normalized_q_group_key(entry.get("q_group_key"))
        if q_group_key is not None:
            entry["q_group_key"] = q_group_key
        if isinstance(raw_saved_entry, Mapping):
            raw_saved_q_group_key = _normalized_q_group_key(raw_saved_entry.get("q_group_key"))
            if raw_saved_q_group_key is not None:
                raw_saved_entry["q_group_key"] = raw_saved_q_group_key
        saved_identity_entry = raw_saved_entry if isinstance(raw_saved_entry, Mapping) else entry
        legacy_working_entry = _legacy_dense_working_entry(entry, raw_saved_entry)
        row_key = _source_row_key(saved_identity_entry)
        row_candidate = (
            dict(simulated_lookup.get(row_key))
            if row_key is not None and isinstance(simulated_lookup.get(row_key), Mapping)
            else None
        )
        peak_key = _source_peak_key(saved_identity_entry)
        peak_candidate = (
            dict(simulated_by_peak.get(peak_key))
            if peak_key is not None and isinstance(simulated_by_peak.get(peak_key), Mapping)
            else None
        )
        strict_source_entry = (
            None
            if isinstance(legacy_working_entry, Mapping)
            else _resolve_cached_source_entry(entry)
        )
        overlay_lookup_entry = (
            dict(legacy_working_entry) if isinstance(legacy_working_entry, Mapping) else dict(entry)
        )
        overlay_source_entry = _resolve_source_entry(overlay_lookup_entry)
        overlay_source_entry = gui_manual_geometry.geometry_manual_apply_refined_simulated_override(
            dict(overlay_lookup_entry),
            dict(overlay_source_entry) if isinstance(overlay_source_entry, Mapping) else None,
            prefer_caked_display=use_caked_display,
        )
        overlay_source_entry = _project_source_entry_for_current_view(overlay_source_entry)
        overlay_kind = _resolved_source_kind(overlay_lookup_entry, overlay_source_entry)
        fit_source_entry: dict[str, object] | None = None
        fit_resolution_kind: str | None = None
        legacy_resolution_trace: dict[str, object] | None = None
        if isinstance(legacy_working_entry, Mapping):
            promoted_source_entry, promoted_resolution_kind, legacy_resolution_trace = (
                _resolve_legacy_dense_source_entry(
                    entry,
                    raw_saved_entry=raw_saved_entry,
                )
            )
            if isinstance(promoted_source_entry, Mapping):
                fit_source_entry = dict(promoted_source_entry)
                fit_resolution_kind = str(promoted_resolution_kind or "legacy_dense_source_rebind")
                overlay_source_entry = dict(promoted_source_entry)
                overlay_kind = str(fit_resolution_kind)
        else:
            promoted_source_entry = _resolve_source_entry(entry)
            if isinstance(promoted_source_entry, Mapping):
                fit_source_entry = dict(promoted_source_entry)
                fit_resolution_kind = _resolved_source_kind(entry, fit_source_entry)
        selected_records.append(
            {
                "entry": entry,
                "raw_saved_entry": dict(raw_saved_entry)
                if isinstance(raw_saved_entry, Mapping)
                else None,
                "legacy_working_entry": dict(legacy_working_entry)
                if isinstance(legacy_working_entry, Mapping)
                else None,
                "strict_source_entry": dict(strict_source_entry)
                if isinstance(strict_source_entry, Mapping)
                else None,
                "fit_source_entry": dict(fit_source_entry)
                if isinstance(fit_source_entry, Mapping)
                else None,
                "fit_resolution_kind": fit_resolution_kind,
                "overlay_source_entry": dict(overlay_source_entry)
                if isinstance(overlay_source_entry, Mapping)
                else None,
                "overlay_resolution_kind": overlay_kind,
                "row_candidate": dict(row_candidate)
                if isinstance(row_candidate, Mapping)
                else None,
                "peak_candidate": dict(peak_candidate)
                if isinstance(peak_candidate, Mapping)
                else None,
                "row_candidate_status": _source_candidate_status(
                    saved_identity_entry,
                    row_candidate,
                    key_kind="row",
                ),
                "peak_candidate_status": _source_candidate_status(
                    saved_identity_entry,
                    peak_candidate,
                    key_kind="peak",
                ),
                "legacy_resolution_trace": dict(legacy_resolution_trace)
                if isinstance(legacy_resolution_trace, Mapping)
                else None,
            }
        )
    resolved_source_pair_count = int(
        sum(1 for record in selected_records if isinstance(record.get("fit_source_entry"), Mapping))
    )

    measured_display: list[dict[str, object]] = []
    initial_pairs_display: list[dict[str, object]] = []
    provider_pairs: list[dict[str, object]] = []
    dataset_manual_point_pairs: list[dict[str, object]] = []
    manual_truth_by_key = _geometry_fit_truth_by_order_key(manual_picker_truth_pairs)
    source_resolution_diagnostics: list[dict[str, object]] = []
    selected_entries = [
        dict(record["entry"])
        for record in selected_records
        if isinstance(record.get("entry"), Mapping)
    ]
    for pair_idx, record in enumerate(selected_records):
        entry = dict(record.get("entry")) if isinstance(record.get("entry"), Mapping) else None
        if entry is None:
            continue
        raw_saved_entry = (
            dict(record.get("raw_saved_entry"))
            if isinstance(record.get("raw_saved_entry"), Mapping)
            else None
        )
        saved_entry_for_diag = raw_saved_entry if isinstance(raw_saved_entry, Mapping) else entry
        legacy_resolution_trace = (
            dict(record.get("legacy_resolution_trace"))
            if isinstance(record.get("legacy_resolution_trace"), Mapping)
            else None
        )
        pair_id = str(entry.get("pair_id") or f"bg{int(background_idx)}:pair{int(pair_idx)}")
        entry["pair_id"] = pair_id
        measured_entry = dict(entry)
        measured_entry["overlay_match_index"] = int(pair_idx)
        measured_entry["pair_id"] = pair_id
        strict_source_entry = (
            dict(record.get("strict_source_entry"))
            if isinstance(record.get("strict_source_entry"), Mapping)
            else None
        )
        fit_source_entry = (
            dict(record.get("fit_source_entry"))
            if isinstance(record.get("fit_source_entry"), Mapping)
            else None
        )
        row_candidate = (
            dict(record.get("row_candidate"))
            if isinstance(record.get("row_candidate"), Mapping)
            else None
        )
        peak_candidate = (
            dict(record.get("peak_candidate"))
            if isinstance(record.get("peak_candidate"), Mapping)
            else None
        )
        overlay_source_entry = (
            dict(record.get("overlay_source_entry"))
            if isinstance(record.get("overlay_source_entry"), Mapping)
            else _resolve_source_entry(measured_entry)
        )
        overlay_source_entry = _project_source_entry_for_current_view(overlay_source_entry)
        row_candidate_status = str(record.get("row_candidate_status", "") or "")
        peak_candidate_status = str(record.get("peak_candidate_status", "") or "")
        overlay_distance_sq = _resolved_source_distance_sq(entry, overlay_source_entry)
        overlay_distance_px = (
            float(np.sqrt(overlay_distance_sq))
            if np.isfinite(float(overlay_distance_sq))
            else float("nan")
        )
        overlay_kind = str(
            record.get("overlay_resolution_kind", "") or ""
        ) or _resolved_source_kind(
            entry,
            overlay_source_entry,
        )
        legacy_selected_live_row = (
            dict(legacy_resolution_trace.get("legacy_chosen_live_row"))
            if isinstance(legacy_resolution_trace, Mapping)
            and isinstance(legacy_resolution_trace.get("legacy_chosen_live_row"), Mapping)
            else None
        )
        selected_source_entry_for_diag = (
            legacy_selected_live_row
            if isinstance(legacy_selected_live_row, Mapping)
            else (
                fit_source_entry if isinstance(fit_source_entry, Mapping) else overlay_source_entry
            )
        )
        saved_background_current_view_point = _entry_display_point(entry)
        saved_background_current_view_frame = _background_current_view_frame(entry)
        selected_live_simulated_current_view_point = _candidate_current_view_point(
            selected_source_entry_for_diag
        )
        selected_live_simulated_current_view_frame = _candidate_current_view_frame(
            selected_source_entry_for_diag
        )
        selected_to_background_distance_px = (
            float(
                math.hypot(
                    float(selected_live_simulated_current_view_point[0])
                    - float(saved_background_current_view_point[0]),
                    float(selected_live_simulated_current_view_point[1])
                    - float(saved_background_current_view_point[1]),
                )
            )
            if selected_live_simulated_current_view_point is not None
            and saved_background_current_view_point is not None
            and selected_live_simulated_current_view_frame == saved_background_current_view_frame
            else None
        )
        saved_branch_index, saved_branch_source = _source_branch_resolution(saved_entry_for_diag)
        strict_branch_index, strict_branch_source = _source_branch_resolution(strict_source_entry)
        fit_branch_index, fit_branch_source = _source_branch_resolution(fit_source_entry)
        overlay_branch_index, overlay_branch_source = _source_branch_resolution(
            overlay_source_entry
        )
        truth_pair = manual_truth_by_key.get((int(background_idx), int(pair_idx)), {})
        provider_background_point = _geometry_fit_point_list(
            truth_pair.get("manual_background_point")
        )
        provider_background_frame = _geometry_fit_normalize_point_frame(
            truth_pair.get("manual_background_frame")
        )
        provider_background_point_source = str(
            truth_pair.get("manual_background_point_source") or "manual_picker_saved"
        )
        if provider_background_point is None:
            provider_background_point = _geometry_fit_point_list(
                saved_background_current_view_point
            )
            provider_background_frame = _geometry_fit_normalize_point_frame(
                saved_background_current_view_frame
            )
            provider_background_point_source = "manual_picker_refresh"

        provider_simulated_point = _geometry_fit_point_list(
            truth_pair.get("manual_selected_simulated_point")
        )
        provider_simulated_frame = _geometry_fit_normalize_point_frame(
            truth_pair.get("manual_selected_simulated_frame")
        )
        provider_simulated_point_source = str(
            truth_pair.get("manual_simulated_point_source") or "manual_picker_saved"
        )
        if use_caked_display and not isinstance(fit_source_entry, Mapping):
            saved_caked_simulated_point = _entry_point(
                saved_entry_for_diag,
                "refined_sim_caked_x",
                "refined_sim_caked_y",
            )
            if saved_caked_simulated_point is not None:
                provider_simulated_point = [
                    float(saved_caked_simulated_point[0]),
                    float(saved_caked_simulated_point[1]),
                ]
                provider_simulated_frame = "display"
                provider_simulated_point_source = "manual_picker_saved"
        saved_identity_available = any(
            saved_entry_for_diag.get(key) is not None
            for key in (
                "source_table_index",
                "source_reflection_index",
                "source_reflection_namespace",
                "source_reflection_is_full",
                "source_row_index",
                "source_peak_index",
                "source_label",
            )
        )
        fit_resolution_kind_text = str(record.get("fit_resolution_kind", "") or "")
        source_locator_identity_match = bool(
            saved_identity_available
            and isinstance(fit_source_entry, Mapping)
            and _source_locator_identity_match(saved_entry_for_diag, fit_source_entry)
        )
        source_semantic_identity_match = bool(
            isinstance(fit_source_entry, Mapping)
            and _source_entry_hkl_matches(saved_entry_for_diag, fit_source_entry)
            and _source_entry_branch_matches(saved_entry_for_diag, fit_source_entry)
            and _source_entry_group_matches(saved_entry_for_diag, fit_source_entry)
        )
        provider_saved_simulated_point_available = bool(
            provider_simulated_point is not None
            and provider_simulated_point_source in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
        )
        fit_resolution_kind_text = str(record.get("fit_resolution_kind", "") or "")
        legacy_rebind_used = bool(
            isinstance(record.get("legacy_working_entry"), Mapping)
            and isinstance(fit_source_entry, Mapping)
            and fit_resolution_kind_text.startswith("legacy_dense_")
        )
        source_rebound_to_live_row = bool(
            isinstance(fit_source_entry, Mapping)
            and source_semantic_identity_match
            and not source_locator_identity_match
        )
        canonical_live_identity_required = bool(
            source_rebound_to_live_row
            and (legacy_rebind_used or _trusted_full_reflection_identity(saved_entry_for_diag))
        )
        saved_identity_matches_live_source = bool(
            saved_identity_available
            and source_semantic_identity_match
            and isinstance(fit_source_entry, Mapping)
        )
        saved_identity_resolved = bool(
            saved_identity_matches_live_source
            and (
                source_locator_identity_match
                or provider_saved_simulated_point_available
                or canonical_live_identity_required
            )
        )
        rebinding_fallback_used = bool(
            (saved_identity_available and not saved_identity_resolved)
            or (
                not saved_identity_available
                and fit_source_entry is not None
                and strict_source_entry is None
            )
        )
        stale_statuses = {"missing", "hkl_mismatch", "branch_mismatch", "group_mismatch"}
        if rebinding_fallback_used and saved_identity_available:
            if not provider_saved_simulated_point_available:
                fallback_reason = "missing_saved_simulated_point"
            elif not isinstance(fit_source_entry, Mapping):
                fallback_reason = "unresolved_source_identity"
            elif not source_semantic_identity_match:
                fallback_reason = "stale_source_identity"
            else:
                fallback_reason = (
                    "stale_source_identity"
                    if row_candidate_status in stale_statuses
                    or peak_candidate_status in stale_statuses
                    else fit_resolution_kind_text or "unresolved_source_identity"
                )
        elif rebinding_fallback_used:
            fallback_reason = fit_resolution_kind_text or "missing_source_identity"
        else:
            fallback_reason = None
        stale_source_identity_diagnostic = bool(
            saved_identity_available
            and saved_identity_resolved
            and source_semantic_identity_match
            and not source_locator_identity_match
        )
        if provider_simulated_point is None:
            provider_simulated_point = _geometry_fit_point_list(
                selected_live_simulated_current_view_point
            )
            provider_simulated_frame = _geometry_fit_normalize_point_frame(
                selected_live_simulated_current_view_frame
            )
            provider_simulated_point_source = (
                "fallback_rebind" if rebinding_fallback_used else "live_source_row_projection"
            )
        provider_distance = (
            float(
                math.hypot(
                    float(provider_simulated_point[0]) - float(provider_background_point[0]),
                    float(provider_simulated_point[1]) - float(provider_background_point[1]),
                )
            )
            if provider_simulated_point is not None
            and provider_background_point is not None
            and provider_simulated_frame == provider_background_frame
            and provider_simulated_frame != "unknown"
            else None
        )
        identity_source_entry = (
            fit_source_entry
            if canonical_live_identity_required and isinstance(fit_source_entry, Mapping)
            else (
                saved_entry_for_diag
                if saved_identity_resolved
                else (fit_source_entry if isinstance(fit_source_entry, Mapping) else None)
            )
        )
        provider_identity = gui_manual_geometry.canonical_geometry_source_identity(
            identity_source_entry
        )
        semantic_pair_key = {
            "background_index": int(background_idx),
            "q_group_key": _geometry_fit_jsonable(saved_entry_for_diag.get("q_group_key")),
            "branch_group_key": _geometry_fit_jsonable(
                saved_entry_for_diag.get("branch_group_key")
            ),
            "normalized_hkl": _geometry_fit_jsonable(
                _normalized_hkl(saved_entry_for_diag.get("hkl"))
            ),
            "source_branch_index": (
                int(saved_branch_index) if saved_branch_index is not None else None
            ),
        }
        parity_mode = (
            "picker_rule_fallback"
            if rebinding_fallback_used
            else (
                "picker_saved_value_preserved"
                if provider_simulated_point_source in {"manual_picker_saved", "manual_picker_cache"}
                and provider_background_point_source
                in {"manual_picker_saved", "manual_picker_cache"}
                else "picker_identity_preserved_with_recomputed_point"
            )
        )
        provider_pair = {
            "pair_index": int(pair_idx),
            "provider_pair_index": int(pair_idx),
            "dataset_pair_index": int(pair_idx),
            "background_index": int(background_idx),
            "manual_pair_order_key": [int(background_idx), int(pair_idx)],
            "semantic_pair_key": semantic_pair_key,
            "q_group_key": semantic_pair_key["q_group_key"],
            "source_q_group_key": _geometry_fit_jsonable(
                selected_source_entry_for_diag.get("q_group_key")
                if isinstance(selected_source_entry_for_diag, Mapping)
                else None
            ),
            "branch_group_key": semantic_pair_key["branch_group_key"],
            "normalized_hkl": semantic_pair_key["normalized_hkl"],
            "source_branch_index": semantic_pair_key["source_branch_index"],
            "selected_source_identity_canonical": provider_identity,
            "background_point": provider_background_point,
            "background_frame": provider_background_frame,
            "background_point_source": provider_background_point_source,
            "simulated_point": provider_simulated_point,
            "simulated_frame": provider_simulated_frame,
            "simulated_point_source": provider_simulated_point_source,
            "selected_to_background_distance_px": provider_distance,
            "parity_mode": parity_mode,
            "rebinding_fallback_used": rebinding_fallback_used,
            "fallback_reason": fallback_reason,
            "source_locator_identity_match": source_locator_identity_match,
            "source_semantic_identity_match": source_semantic_identity_match,
            "stale_source_identity_diagnostic": stale_source_identity_diagnostic,
            "stale_saved_source_identity": (
                gui_manual_geometry.canonical_geometry_source_identity(saved_entry_for_diag)
                if (
                    rebinding_fallback_used
                    or stale_source_identity_diagnostic
                    or canonical_live_identity_required
                )
                else None
            ),
        }
        _geometry_fit_apply_source_coverage_identity(provider_pair)
        provider_pairs.append(provider_pair)
        resolution_diag: dict[str, object] = {
            "pair_index": int(pair_idx),
            "pair_id": pair_id,
            "saved_source_table_index": saved_entry_for_diag.get("source_table_index"),
            "saved_source_reflection_index": saved_entry_for_diag.get("source_reflection_index"),
            "saved_source_row_index": saved_entry_for_diag.get("source_row_index"),
            "saved_source_branch_index": saved_branch_index,
            "saved_source_branch_source": saved_branch_source,
            "saved_source_peak_index": saved_entry_for_diag.get("source_peak_index"),
            "saved_source_row_key": _source_row_key(saved_entry_for_diag),
            "saved_source_peak_key": _source_peak_key(saved_entry_for_diag),
            "saved_hkl": _normalized_hkl(saved_entry_for_diag.get("hkl")),
            "saved_q_group_key": saved_entry_for_diag.get("q_group_key"),
            "saved_display_point": _entry_display_point(entry),
            "saved_background_current_view_point": saved_background_current_view_point,
            "saved_background_current_view_frame": saved_background_current_view_frame,
            "saved_simulated_detector_hint": (
                _entry_point(
                    saved_entry_for_diag,
                    "refined_sim_native_x",
                    "refined_sim_native_y",
                )
                or _entry_saved_simulated_current_view_point(saved_entry_for_diag)
            ),
            "strict_resolved": isinstance(strict_source_entry, Mapping),
            "fit_resolved": isinstance(fit_source_entry, Mapping),
            "fit_resolution_kind": str(record.get("fit_resolution_kind", "") or "") or None,
            "resolution_kind": str(record.get("fit_resolution_kind", "") or "") or None,
            "fit_source_row_key": _source_row_key(fit_source_entry),
            "fit_source_peak_key": _source_peak_key(fit_source_entry),
            "fit_source_branch_index": fit_branch_index,
            "fit_source_branch_source": fit_branch_source,
            "selected_candidate_source_identity_fields": _source_locator_payload(
                selected_source_entry_for_diag
            ),
            "selected_live_simulated_current_view_point": (
                selected_live_simulated_current_view_point
            ),
            "selected_live_simulated_current_view_frame": (
                selected_live_simulated_current_view_frame
            ),
            "selected_to_background_distance_px": selected_to_background_distance_px,
            "source_locator_identity_match": source_locator_identity_match,
            "source_semantic_identity_match": source_semantic_identity_match,
            "stale_source_identity_diagnostic": stale_source_identity_diagnostic,
            "fit_source_reflection_index": (
                fit_source_entry.get("source_reflection_index")
                if isinstance(fit_source_entry, Mapping)
                else None
            ),
            "strict_source_branch_index": strict_branch_index,
            "strict_source_branch_source": strict_branch_source,
            "strict_resolution_kind": _resolved_source_kind(entry, strict_source_entry),
            "row_candidate_status": row_candidate_status,
            "row_candidate_hkl": _normalized_hkl(
                row_candidate.get("hkl") if isinstance(row_candidate, Mapping) else None
            ),
            "peak_candidate_status": peak_candidate_status,
            "peak_candidate_hkl": _normalized_hkl(
                peak_candidate.get("hkl") if isinstance(peak_candidate, Mapping) else None
            ),
            "overlay_resolution_kind": overlay_kind,
            "overlay_source_row_key": _source_row_key(overlay_source_entry),
            "overlay_source_peak_key": _source_peak_key(overlay_source_entry),
            "overlay_source_branch_index": overlay_branch_index,
            "overlay_source_branch_source": overlay_branch_source,
            "overlay_source_reflection_index": (
                overlay_source_entry.get("source_reflection_index")
                if isinstance(overlay_source_entry, Mapping)
                else None
            ),
            "overlay_hkl": _normalized_hkl(
                overlay_source_entry.get("hkl")
                if isinstance(overlay_source_entry, Mapping)
                else None
            ),
            "overlay_distance_px": (
                float(overlay_distance_px) if np.isfinite(float(overlay_distance_px)) else None
            ),
            "raw_saved_entry": _geometry_fit_compact_source_resolution_entry_payload(
                raw_saved_entry
            ),
            "normalized_saved_entry": (
                _geometry_fit_compact_source_resolution_entry_payload(entry)
            ),
            "failure_reason": (
                None
                if isinstance(fit_source_entry, Mapping)
                else (
                    str(legacy_resolution_trace.get("legacy_failure_reason"))
                    if isinstance(legacy_resolution_trace, Mapping)
                    and legacy_resolution_trace.get("legacy_failure_reason") is not None
                    else _strict_failure_reason(
                        row_status=row_candidate_status,
                        peak_status=peak_candidate_status,
                        overlay_kind=overlay_kind,
                    )
                )
            ),
        }
        if isinstance(legacy_resolution_trace, Mapping):
            resolution_diag.update(_geometry_fit_cache_jsonable(legacy_resolution_trace))
        source_resolution_diagnostics.append(resolution_diag)
        for key in (
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
            "pair_id",
            "fit_run_id",
        ):
            measured_entry.pop(key, None)
        if isinstance(identity_source_entry, Mapping):
            for key in (
                "source_table_index",
                "source_reflection_index",
                "source_reflection_namespace",
                "source_reflection_is_full",
                "source_row_index",
                "source_branch_index",
                "source_peak_index",
                "source_label",
                "pair_id",
                "fit_run_id",
            ):
                if key in identity_source_entry:
                    measured_entry[key] = identity_source_entry.get(key)
        measured_entry["source_label"] = _geometry_fit_entry_source_label(entry)
        measured_entry["selected_source_identity_canonical"] = provider_identity
        measured_entry["fit_source_identity_only"] = True
        if record.get("fit_resolution_kind") is not None:
            measured_entry["fit_source_resolution_kind"] = record.get("fit_resolution_kind")
        if provider_background_point is not None and provider_background_frame == "display":
            measured_entry["x"] = float(provider_background_point[0])
            measured_entry["y"] = float(provider_background_point[1])
            measured_entry["display_col"] = float(provider_background_point[0])
            measured_entry["display_row"] = float(provider_background_point[1])
            measured_entry["provider_background_point_source"] = provider_background_point_source
            measured_entry["provider_background_frame"] = provider_background_frame
        measured_entry["provider_simulated_point_source"] = provider_simulated_point_source
        measured_entry["provider_simulated_frame"] = provider_simulated_frame
        _geometry_fit_put_simulated_point_fields(
            measured_entry,
            provider_simulated_point,
            provider_simulated_frame,
        )
        _geometry_fit_apply_source_coverage_identity(measured_entry)
        measured_display.append(measured_entry)
        if isinstance(legacy_resolution_trace, Mapping):
            source_resolution_diagnostics[-1]["legacy_fit_bound_entry"] = (
                _geometry_fit_compact_source_resolution_entry_payload(measured_entry)
            )

        initial_entry: dict[str, object] = {
            "overlay_match_index": int(pair_idx),
            "pair_id": pair_id,
            "hkl": entry.get("hkl", entry.get("label")),
        }
        raw_group_key = entry.get("q_group_key")
        if isinstance(raw_group_key, tuple):
            initial_entry["q_group_key"] = raw_group_key
        elif isinstance(raw_group_key, list):
            initial_entry["q_group_key"] = tuple(raw_group_key)
        initial_entry["source_label"] = _geometry_fit_entry_source_label(entry)
        try:
            bg_coords = manual_dataset_bindings.geometry_manual_entry_display_coords(entry)
        except Exception:
            bg_coords = None
        if (
            bg_coords is None
            and provider_background_point is not None
            and provider_background_frame == "display"
        ):
            bg_coords = tuple(provider_background_point)
        if bg_coords is not None and len(bg_coords) >= 2:
            initial_entry["bg_display"] = (float(bg_coords[0]), float(bg_coords[1]))
            if use_caked_display:
                initial_entry["bg_caked_display"] = (
                    float(bg_coords[0]),
                    float(bg_coords[1]),
                )
        background_angles = _caked_angle_pair(
            entry,
            x_keys=(
                "caked_x",
                "raw_caked_x",
                "background_two_theta_deg",
            ),
            y_keys=(
                "caked_y",
                "raw_caked_y",
                "background_phi_deg",
            ),
        )
        if background_angles is not None:
            initial_entry["background_two_theta_deg"] = float(background_angles[0])
            initial_entry["background_phi_deg"] = float(background_angles[1])
            if use_caked_display:
                initial_entry["bg_caked_display"] = (
                    float(background_angles[0]),
                    float(background_angles[1]),
                )
        reference_two_theta = _finite_float(
            entry.get("background_reference_two_theta_deg")
        ) or _reference_two_theta_deg(
            entry,
            a_lattice=reference_a,
            c_lattice=reference_c,
            wavelength=reference_lambda,
        )
        if reference_two_theta is not None:
            initial_entry["background_reference_two_theta_deg"] = float(reference_two_theta)
        if reference_a is not None:
            initial_entry["background_reference_a"] = float(reference_a)
        if reference_c is not None:
            initial_entry["background_reference_c"] = float(reference_c)
        if reference_lambda is not None:
            initial_entry["background_reference_lambda"] = float(reference_lambda)
        for source_key, target_key in (
            ("qr", "background_reference_qr"),
            ("qz", "background_reference_qz"),
        ):
            value = _finite_float(entry.get(source_key))
            if value is not None:
                initial_entry[target_key] = float(value)
        if isinstance(identity_source_entry, Mapping):
            for key in (
                "source_table_index",
                "source_reflection_index",
                "source_reflection_namespace",
                "source_reflection_is_full",
                "source_row_index",
                "source_branch_index",
                "source_peak_index",
                "source_label",
                "fit_run_id",
            ):
                if key in identity_source_entry:
                    initial_entry[key] = identity_source_entry.get(key)
        initial_entry["selected_source_identity_canonical"] = provider_identity
        if isinstance(overlay_source_entry, Mapping):
            display_point = (
                tuple(provider_simulated_point)
                if provider_simulated_point is not None and provider_simulated_frame == "display"
                else _candidate_current_view_point(overlay_source_entry)
            )
            if display_point is not None:
                sim_col = float(display_point[0])
                sim_row = float(display_point[1])
            else:
                sim_col = float("nan")
                sim_row = float("nan")
            if np.isfinite(sim_col) and np.isfinite(sim_row):
                initial_entry["sim_display"] = (float(sim_col), float(sim_row))
                if use_caked_display:
                    initial_entry["sim_caked_display"] = (
                        float(sim_col),
                        float(sim_row),
                    )
            simulated_angles = _caked_angle_pair(
                overlay_source_entry,
                x_keys=("two_theta_deg", "caked_x"),
                y_keys=("phi_deg", "caked_y"),
            )
            if simulated_angles is None:
                simulated_angles = _caked_angle_pair(
                    entry,
                    x_keys=(
                        "refined_sim_caked_x",
                        "simulated_two_theta_deg",
                        "two_theta_deg",
                        "caked_x",
                    ),
                    y_keys=(
                        "refined_sim_caked_y",
                        "simulated_phi_deg",
                        "phi_deg",
                        "caked_y",
                    ),
                )
            if simulated_angles is not None:
                initial_entry["simulated_two_theta_deg"] = float(simulated_angles[0])
                initial_entry["simulated_phi_deg"] = float(simulated_angles[1])
                if use_caked_display:
                    initial_entry["sim_display"] = (
                        float(simulated_angles[0]),
                        float(simulated_angles[1]),
                    )
                    initial_entry["sim_caked_display"] = (
                        float(simulated_angles[0]),
                        float(simulated_angles[1]),
                    )
            sim_native_point = _entry_point(
                entry,
                "refined_sim_native_x",
                "refined_sim_native_y",
            )
            sim_native_point_source = (
                "refined_sim_native_px" if sim_native_point is not None else ""
            )
            if sim_native_point is None:
                sim_native_point = _entry_point(
                    overlay_source_entry,
                    "native_col",
                    "native_row",
                )
                if sim_native_point is not None:
                    sim_native_point_source = "overlay_source_native_col_row"
            if sim_native_point is None:
                sim_native_point = _entry_point(
                    overlay_source_entry,
                    "sim_native_x",
                    "sim_native_y",
                )
                if sim_native_point is not None:
                    sim_native_point_source = "overlay_source_sim_native_xy"
            if sim_native_point is None:
                try:
                    sim_col_raw = float(overlay_source_entry.get("sim_col_raw", sim_col))
                    sim_row_raw = float(overlay_source_entry.get("sim_row_raw", sim_row))
                except Exception:
                    sim_col_raw = float("nan")
                    sim_row_raw = float("nan")
                if np.isfinite(sim_col_raw) and np.isfinite(sim_row_raw):
                    try:
                        sim_native = manual_dataset_bindings.display_to_native_sim_coords(
                            float(sim_col_raw),
                            float(sim_row_raw),
                            (
                                int(manual_dataset_bindings.image_size),
                                int(manual_dataset_bindings.image_size),
                            ),
                        )
                    except Exception:
                        sim_native = None
                    if (
                        isinstance(sim_native, tuple)
                        and len(sim_native) >= 2
                        and np.isfinite(float(sim_native[0]))
                        and np.isfinite(float(sim_native[1]))
                    ):
                        sim_native_point = (
                            float(sim_native[0]),
                            float(sim_native[1]),
                        )
                        sim_native_point_source = "display_to_native_sim_coords(sim_col_raw)"
            if sim_native_point is None and not use_caked_display:
                refined_display_point = _entry_point(
                    entry,
                    "refined_sim_x",
                    "refined_sim_y",
                )
                if refined_display_point is not None:
                    try:
                        inverse_projected = gui_manual_geometry._default_rotate_point(
                            float(refined_display_point[0]),
                            float(refined_display_point[1]),
                            (
                                int(manual_dataset_bindings.image_size),
                                int(manual_dataset_bindings.image_size),
                            ),
                            (-int(manual_dataset_bindings.display_rotate_k)) % 4,
                        )
                    except Exception:
                        inverse_projected = None
                    if (
                        isinstance(inverse_projected, tuple)
                        and len(inverse_projected) >= 2
                        and np.isfinite(float(inverse_projected[0]))
                        and np.isfinite(float(inverse_projected[1]))
                    ):
                        sim_native_point = (
                            float(inverse_projected[0]),
                            float(inverse_projected[1]),
                        )
                        sim_native_point_source = "default_unrotate_refined_sim_display"
            if sim_native_point is not None:
                initial_entry["sim_native"] = (
                    float(sim_native_point[0]),
                    float(sim_native_point[1]),
                )
                if sim_native_point_source:
                    initial_entry["sim_native_source"] = sim_native_point_source
        provider_overwrites_existing_sim = bool(
            provider_simulated_point_source in _GEOMETRY_FIT_PICKER_OWNED_POINT_SOURCES
        )
        _install_provider_simulated_point(
            initial_entry,
            provider_simulated_point,
            provider_simulated_frame,
            provider_simulated_point_source,
            overwrite_existing=provider_overwrites_existing_sim,
        )
        _geometry_fit_apply_source_coverage_identity(initial_entry)
        initial_pairs_display.append(initial_entry)

    measured_native = manual_dataset_bindings.unrotate_display_peaks(
        measured_display,
        display_background.shape,
        k=manual_dataset_bindings.display_rotate_k,
    )
    for original_entry, initial_entry, measured_entry in zip(
        measured_display,
        initial_pairs_display,
        measured_native,
    ):
        if not isinstance(measured_entry, dict):
            continue
        detector_anchor = None
        try:
            detector_anchor = (
                float(original_entry.get("detector_x")),
                float(original_entry.get("detector_y")),
            )
        except Exception:
            detector_anchor = None
        if (
            isinstance(detector_anchor, tuple)
            and len(detector_anchor) >= 2
            and np.isfinite(float(detector_anchor[0]))
            and np.isfinite(float(detector_anchor[1]))
        ):
            initial_entry["background_detector_x"] = float(detector_anchor[0])
            initial_entry["background_detector_y"] = float(detector_anchor[1])
            initial_entry["background_detector_input_frame"] = "native_detector"
            initial_entry["background_detector_frame_provenance"] = (
                "geometry_manual_refresh_pair_entry"
            )
            initial_entry["native_col"] = float(detector_anchor[0])
            initial_entry["native_row"] = float(detector_anchor[1])
            measured_entry["x"] = float(detector_anchor[0])
            measured_entry["y"] = float(detector_anchor[1])
        try:
            mx = float(measured_entry.get("x"))
            my = float(measured_entry.get("y"))
        except Exception:
            continue
        if np.isfinite(mx) and np.isfinite(my):
            measured_entry["detector_x"] = float(mx)
            measured_entry["detector_y"] = float(my)
            initial_entry["bg_native"] = (float(mx), float(my))

    sim_orientation_points: list[tuple[float, float]] = []
    meas_orientation_points: list[tuple[float, float]] = []
    sim_native_shape = (
        int(manual_dataset_bindings.image_size),
        int(manual_dataset_bindings.image_size),
    )
    for initial_entry, measured_entry in zip(initial_pairs_display, measured_native):
        if not isinstance(measured_entry, Mapping):
            continue
        sim_display = initial_entry.get("sim_display")
        if not isinstance(sim_display, (list, tuple, np.ndarray)) or len(sim_display) < 2:
            continue
        try:
            sim_native_raw = initial_entry.get("sim_native")
            sim_native_from_display = manual_dataset_bindings.display_to_native_sim_coords(
                float(sim_display[0]),
                float(sim_display[1]),
                sim_native_shape,
            )
            if isinstance(sim_native_raw, (list, tuple, np.ndarray)) and len(sim_native_raw) >= 2:
                sim_native = (
                    float(sim_native_raw[0]),
                    float(sim_native_raw[1]),
                )
            else:
                sim_native = sim_native_from_display
            mx = float(measured_entry.get("x"))
            my = float(measured_entry.get("y"))
        except Exception:
            continue
        if not (
            np.isfinite(sim_native[0])
            and np.isfinite(sim_native[1])
            and np.isfinite(mx)
            and np.isfinite(my)
        ):
            continue
        if not _geometry_fit_sim_native_source_is_display_to_native(
            initial_entry.get("sim_native_source")
        ) and _geometry_fit_points_match(sim_native, sim_native_from_display, tol=1.0e-6):
            initial_entry["sim_native_source"] = "display_to_native_sim_coords(sim_display)"
        sim_orientation_points.append((float(sim_native[0]), float(sim_native[1])))
        meas_orientation_points.append((float(mx), float(my)))

    orientation_choice, orientation_diag = manual_dataset_bindings.select_fit_orientation(
        sim_orientation_points,
        meas_orientation_points,
        tuple(int(v) for v in native_background.shape[:2]),
        cfg=orientation_cfg or {},
    )
    measured_for_fit = manual_dataset_bindings.apply_orientation_to_entries(
        measured_native,
        native_background.shape,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )
    for measured_entry in measured_for_fit:
        if not isinstance(measured_entry, dict):
            continue
        try:
            display_col = float(measured_entry.get("x"))
            display_row = float(measured_entry.get("y"))
        except Exception:
            continue
        if np.isfinite(display_col) and np.isfinite(display_row):
            measured_entry["display_col"] = float(display_col)
            measured_entry["display_row"] = float(display_row)
            measured_entry["fit_detector_x"] = float(display_col)
            measured_entry["fit_detector_y"] = float(display_row)
            measured_entry["detector_x"] = float(display_col)
            measured_entry["detector_y"] = float(display_row)
            measured_entry["detector_input_frame"] = "fit_detector"
            measured_entry["detector_input_frame_reason"] = "apply_orientation_to_entries"
        measured_entry["fit_source_identity_only"] = True
    for measured_entry, initial_entry in zip(measured_for_fit, initial_pairs_display):
        if not isinstance(measured_entry, dict) or not isinstance(initial_entry, Mapping):
            continue
        background_detector_x = _finite_float(measured_entry.get("background_detector_x"))
        background_detector_y = _finite_float(measured_entry.get("background_detector_y"))
        if background_detector_x is not None and background_detector_y is not None:
            measured_entry["background_detector_x"] = float(background_detector_x)
            measured_entry["background_detector_y"] = float(background_detector_y)
        for key in (
            "overlay_match_index",
            "pair_id",
            "fit_run_id",
            "q_group_key",
            "source_table_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
            "background_detector_x",
            "background_detector_y",
            "background_detector_input_frame",
            "background_detector_frame_provenance",
            "native_col",
            "native_row",
            "background_two_theta_deg",
            "background_phi_deg",
            "background_reference_two_theta_deg",
            "background_reference_a",
            "background_reference_c",
            "background_reference_lambda",
            "background_reference_qr",
            "background_reference_qz",
        ):
            if key in initial_entry:
                measured_entry[key] = initial_entry.get(key)
        native_detector_x = _finite_float(
            measured_entry.get(
                "background_detector_x",
                measured_entry.get("native_col"),
            )
        )
        native_detector_y = _finite_float(
            measured_entry.get(
                "background_detector_y",
                measured_entry.get("native_row"),
            )
        )
        if native_detector_x is not None and native_detector_y is not None:
            measured_entry["background_detector_x"] = float(native_detector_x)
            measured_entry["background_detector_y"] = float(native_detector_y)
            measured_entry["native_col"] = float(native_detector_x)
            measured_entry["native_row"] = float(native_detector_y)
            measured_entry.setdefault(
                "background_detector_input_frame",
                "native_detector",
            )
            measured_entry.setdefault(
                "background_detector_frame_provenance",
                "geometry_manual_refresh_pair_entry",
            )
        else:
            display_col = _finite_float(measured_entry.get("display_col", measured_entry.get("x")))
            display_row = _finite_float(measured_entry.get("display_row", measured_entry.get("y")))
            if display_col is not None and display_row is not None:
                measured_entry["detector_x"] = float(display_col)
                measured_entry["detector_y"] = float(display_row)
                measured_entry["fit_detector_x"] = float(display_col)
                measured_entry["fit_detector_y"] = float(display_row)
                measured_entry["detector_input_frame"] = "fit_detector"
                measured_entry["detector_input_frame_reason"] = "apply_orientation_to_entries"
        caked_two_theta = _finite_float(measured_entry.get("background_two_theta_deg"))
        caked_phi = _finite_float(measured_entry.get("background_phi_deg"))
        if caked_two_theta is not None and caked_phi is not None:
            measured_entry["background_two_theta_deg"] = float(caked_two_theta)
            measured_entry["background_phi_deg"] = float(caked_phi)
            measured_entry["fit_space_anchor_override"] = True
            measured_entry["fit_space_anchor_source"] = "manual_caked_background_angles"
        _geometry_fit_apply_source_coverage_identity(measured_entry)
    for pair_idx, provider_pair in enumerate(provider_pairs):
        measured_entry = (
            measured_for_fit[pair_idx]
            if pair_idx < len(measured_for_fit) and isinstance(measured_for_fit[pair_idx], Mapping)
            else {}
        )
        dataset_pair = {
            "pair_index": int(pair_idx),
            "provider_pair_index": int(provider_pair.get("provider_pair_index", pair_idx)),
            "dataset_pair_index": int(pair_idx),
            "background_index": int(background_idx),
            "manual_pair_order_key": provider_pair.get("manual_pair_order_key"),
            "semantic_pair_key": provider_pair.get("semantic_pair_key"),
            "q_group_key": provider_pair.get("q_group_key"),
            "source_q_group_key": provider_pair.get("source_q_group_key"),
            "branch_group_key": provider_pair.get("branch_group_key"),
            "normalized_hkl": provider_pair.get("normalized_hkl"),
            "source_branch_index": provider_pair.get("source_branch_index"),
            "selected_source_identity_canonical": provider_pair.get(
                "selected_source_identity_canonical"
            ),
            "background_point": provider_pair.get("background_point"),
            "background_frame": provider_pair.get("background_frame"),
            "background_point_source": provider_pair.get("background_point_source"),
            "simulated_point": provider_pair.get("simulated_point"),
            "simulated_frame": provider_pair.get("simulated_frame"),
            "simulated_point_source": provider_pair.get("simulated_point_source"),
            "selected_to_background_distance_px": provider_pair.get(
                "selected_to_background_distance_px"
            ),
            "parity_mode": provider_pair.get("parity_mode"),
            "rebinding_fallback_used": provider_pair.get("rebinding_fallback_used"),
            "fallback_reason": provider_pair.get("fallback_reason"),
            "solver_measured_point": _geometry_fit_point_list(
                (
                    measured_entry.get("x"),
                    measured_entry.get("y"),
                )
                if isinstance(measured_entry, Mapping)
                else None
            ),
            "solver_measured_frame": _geometry_fit_normalize_point_frame(
                measured_entry.get("detector_input_frame", "display")
                if isinstance(measured_entry, Mapping)
                else None
            ),
        }
        _geometry_fit_apply_source_coverage_identity(dataset_pair)
        dataset_manual_point_pairs.append(dataset_pair)
    backend_background = manual_dataset_bindings.apply_background_backend_orientation(
        native_background
    )
    if backend_background is None:
        backend_background = native_background
    experimental_image_for_fit = manual_dataset_bindings.orient_image_for_fit(
        backend_background,
        indexing_mode=orientation_choice["indexing_mode"],
        k=orientation_choice["k"],
        flip_x=orientation_choice["flip_x"],
        flip_y=orientation_choice["flip_y"],
        flip_order=orientation_choice["flip_order"],
    )
    dynamic_reanchor_callback = None
    dynamic_reanchor_match_cfg: dict[str, object] = {}
    dynamic_reanchor_detector_background_context: dict[str, object] | None = None
    dynamic_reanchor_caked_background_context: dict[str, object] | None = None
    dynamic_reanchor_detector_image: np.ndarray | None = None
    dynamic_reanchor_caked_background: np.ndarray | None = None
    dynamic_reanchor_radial_axis: np.ndarray | None = None
    dynamic_reanchor_azimuth_axis: np.ndarray | None = None
    dynamic_reanchor_raw_azimuth_axis: np.ndarray | None = None
    dynamic_reanchor_transform_bundle: object = None
    dynamic_reanchor_exact_bundle: CakeTransformBundle | None = None
    dynamic_reanchor_caked_view_ready = False
    dynamic_reanchor_enabled = (
        isinstance(experimental_image_for_fit, np.ndarray)
        and experimental_image_for_fit.ndim == 2
        and experimental_image_for_fit.size > 0
    )
    fit_space_projector = None
    fit_space_projector_kind: str | None = None
    fit_space_projector_unavailable_reason = "exact_caked_view_unavailable"
    sim_caked_image_builder = None
    sim_caked_image_builder_kind: str | None = None
    if callable(manual_dataset_bindings.geometry_manual_match_config):
        try:
            dynamic_reanchor_match_cfg = dict(
                manual_dataset_bindings.geometry_manual_match_config() or {}
            )
        except Exception:
            dynamic_reanchor_match_cfg = {}
    if callable(manual_dataset_bindings.geometry_manual_caked_view_for_index):
        try:
            raw_caked_view = manual_dataset_bindings.geometry_manual_caked_view_for_index(
                int(background_idx)
            )
        except Exception:
            raw_caked_view = None
        caked_background_local = None
        radial_axis_local = None
        azimuth_axis_local = None
        raw_azimuth_axis_local = None
        dynamic_reanchor_transform_bundle = None
        if isinstance(raw_caked_view, Mapping):
            caked_background_local = raw_caked_view.get(
                "background_image",
                raw_caked_view.get("background"),
            )
            radial_axis_local = raw_caked_view.get("radial_axis")
            azimuth_axis_local = raw_caked_view.get("azimuth_axis")
            raw_azimuth_axis_local = raw_caked_view.get("raw_azimuth_axis")
            dynamic_reanchor_transform_bundle = raw_caked_view.get("transform_bundle")
        elif isinstance(raw_caked_view, (list, tuple)) and len(raw_caked_view) >= 3:
            caked_background_local = raw_caked_view[0]
            radial_axis_local = raw_caked_view[1]
            azimuth_axis_local = raw_caked_view[2]
        exact_payload = {
            "background": caked_background_local,
            "detector_shape": np.asarray(native_background).shape[:2],
            "radial_axis": radial_axis_local,
            "azimuth_axis": azimuth_axis_local,
            "raw_azimuth_axis": raw_azimuth_axis_local,
            "transform_bundle": dynamic_reanchor_transform_bundle,
        }
        dynamic_reanchor_exact_bundle = _geometry_fit_caked_payload_exact_bundle(
            exact_payload,
            detector_shape=np.asarray(native_background).shape[:2],
            params=params_i,
            require_background=True,
        )
        if isinstance(dynamic_reanchor_exact_bundle, CakeTransformBundle):
            dynamic_reanchor_transform_bundle = dynamic_reanchor_exact_bundle
        try:
            if caked_background_local is not None:
                dynamic_reanchor_caked_background = np.asarray(
                    caked_background_local,
                    dtype=np.float64,
                )
        except Exception:
            dynamic_reanchor_caked_background = None
        try:
            if radial_axis_local is not None:
                dynamic_reanchor_radial_axis = np.asarray(
                    radial_axis_local,
                    dtype=np.float64,
                )
        except Exception:
            dynamic_reanchor_radial_axis = None
        try:
            if azimuth_axis_local is not None:
                dynamic_reanchor_azimuth_axis = np.asarray(
                    azimuth_axis_local,
                    dtype=np.float64,
                )
        except Exception:
            dynamic_reanchor_azimuth_axis = None
        try:
            if raw_azimuth_axis_local is not None:
                dynamic_reanchor_raw_azimuth_axis = np.asarray(
                    raw_azimuth_axis_local,
                    dtype=np.float64,
                )
        except Exception:
            dynamic_reanchor_raw_azimuth_axis = None
        if (
            isinstance(dynamic_reanchor_caked_background, np.ndarray)
            and dynamic_reanchor_caked_background.ndim == 2
            and dynamic_reanchor_caked_background.size > 0
            and isinstance(dynamic_reanchor_radial_axis, np.ndarray)
            and dynamic_reanchor_radial_axis.size > 0
            and isinstance(dynamic_reanchor_azimuth_axis, np.ndarray)
            and dynamic_reanchor_azimuth_axis.size > 0
            and isinstance(dynamic_reanchor_exact_bundle, CakeTransformBundle)
        ):
            dynamic_reanchor_caked_view_ready = True
        elif (
            isinstance(dynamic_reanchor_caked_background, np.ndarray)
            or isinstance(dynamic_reanchor_radial_axis, np.ndarray)
            or isinstance(dynamic_reanchor_azimuth_axis, np.ndarray)
        ):
            fit_space_projector_unavailable_reason = "missing_exact_caked_bundle"
    if dynamic_reanchor_enabled:
        dynamic_reanchor_detector_image = np.asarray(
            experimental_image_for_fit,
            dtype=np.float64,
        )
        try:
            built_background_context = build_background_peak_context(
                dynamic_reanchor_detector_image,
                dict(dynamic_reanchor_match_cfg),
            )
        except Exception:
            built_background_context = None
        if isinstance(built_background_context, Mapping):
            dynamic_reanchor_detector_background_context = dict(built_background_context)
        if dynamic_reanchor_caked_view_ready and isinstance(
            dynamic_reanchor_caked_background, np.ndarray
        ):
            try:
                built_caked_context = build_background_peak_context(
                    dynamic_reanchor_caked_background,
                    dict(dynamic_reanchor_match_cfg),
                )
            except Exception:
                built_caked_context = None
            if isinstance(built_caked_context, Mapping):
                dynamic_reanchor_caked_background_context = dict(built_caked_context)
        try:
            dynamic_reanchor_backend_shape = tuple(
                int(v) for v in np.asarray(backend_background).shape[:2]
            )
        except Exception:
            dynamic_reanchor_backend_shape = ()
        dynamic_reanchor_native_shape = _geometry_fit_detector_shape_2d(
            np.asarray(native_background).shape[:2]
        )

        def _native_detector_coords_to_caked_display_coords(
            detector_col: float,
            detector_row: float,
            *,
            active_caked_bundle: CakeTransformBundle | None,
        ) -> tuple[float | None, float | None]:
            projected = gui_manual_geometry.native_detector_coords_to_caked_display_coords(
                float(detector_col),
                float(detector_row),
                ai=None,
                get_detector_angular_maps=(lambda _ai: (None, None)),
                detector_pixel_to_scattering_angles=(lambda *_args, **_kwargs: (None, None)),
                center=None,
                detector_distance=float("nan"),
                pixel_size=float("nan"),
                wrap_phi_range=(lambda value: value),
                transform_bundle=active_caked_bundle,
                native_detector_coords_to_bundle_detector_coords=(
                    manual_dataset_bindings.native_detector_coords_to_bundle_detector_coords
                ),
            )
            if not (
                isinstance(projected, tuple)
                and len(projected) >= 2
                and projected[0] is not None
                and projected[1] is not None
            ):
                return None, None
            try:
                two_theta_deg = float(projected[0])
                phi_deg = float(projected[1])
            except Exception:
                return None, None
            if not (np.isfinite(two_theta_deg) and np.isfinite(phi_deg)):
                return None, None
            return float(two_theta_deg), float(phi_deg)

        def _effective_theta_for_projection(
            active_params: Mapping[str, object] | None,
        ) -> float:
            if isinstance(active_params, Mapping):
                try:
                    theta_value = float(active_params.get("theta_initial", np.nan))
                except Exception:
                    theta_value = float("nan")
                if np.isfinite(theta_value):
                    return float(theta_value)
                try:
                    theta_offset_value = float(active_params.get("theta_offset", np.nan))
                except Exception:
                    theta_offset_value = float("nan")
                if np.isfinite(theta_offset_value):
                    return float(theta_base + theta_offset_value)
            try:
                return float(params_i.get("theta_initial", theta_base))
            except Exception:
                return float(theta_base)

        dynamic_reanchor_exact_bundle_cache: dict[tuple[object, ...], CakeTransformBundle] = {}

        def _dynamic_reanchor_axis_cache_signature(axis: object) -> tuple[object, ...] | None:
            vec = _geometry_fit_float64_vector(axis)
            if vec is None:
                return None
            arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float64).reshape(-1))
            if arr.size <= 0:
                return (0, "empty")
            digest = hashlib.sha1(arr.tobytes()).hexdigest()
            return (
                int(arr.size),
                float(arr[0]),
                float(arr[-1]),
                str(arr.dtype),
                digest,
            )

        def _dynamic_reanchor_bundle_cache_key(
            active_params: Mapping[str, object] | None,
        ) -> tuple[object, ...]:
            return (
                "exact_caked_bundle",
                tuple(dynamic_reanchor_native_shape or ()),
                _dynamic_reanchor_axis_cache_signature(dynamic_reanchor_radial_axis),
                _dynamic_reanchor_axis_cache_signature(dynamic_reanchor_azimuth_axis),
                _dynamic_reanchor_axis_cache_signature(dynamic_reanchor_raw_azimuth_axis),
                _geometry_fit_projection_signature(
                    _geometry_fit_exact_caked_bundle_param_payload(active_params)
                ),
            )

        def _resolve_dynamic_reanchor_cached_caked_bundle(
            active_params: Mapping[str, object] | None,
            *,
            prefer_rebuild_bundle: bool,
        ) -> CakeTransformBundle | None:
            bundle_cache_key = _dynamic_reanchor_bundle_cache_key(active_params)
            active_bundle = dynamic_reanchor_exact_bundle_cache.get(bundle_cache_key)
            if isinstance(active_bundle, CakeTransformBundle):
                return active_bundle
            active_bundle = _geometry_fit_resolve_dynamic_reanchor_caked_bundle(
                detector_shape=dynamic_reanchor_native_shape,
                radial_axis=dynamic_reanchor_radial_axis,
                azimuth_axis=dynamic_reanchor_azimuth_axis,
                raw_azimuth_axis=dynamic_reanchor_raw_azimuth_axis,
                transform_bundle=(
                    None if prefer_rebuild_bundle else dynamic_reanchor_transform_bundle
                ),
                params=active_params,
            )
            if isinstance(active_bundle, CakeTransformBundle):
                dynamic_reanchor_exact_bundle_cache[bundle_cache_key] = active_bundle
                return active_bundle
            return None

        def _project_detector_points_with_active_caked_bundle(
            cols: object,
            rows: object,
            *,
            local_params: Mapping[str, object] | None,
            anchor_kind: str,
            input_frame: str,
            prefer_rebuild_bundle: bool,
        ) -> dict[str, object]:
            del anchor_kind
            try:
                col_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
                row_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
            except Exception:
                col_arr = np.asarray([], dtype=np.float64)
                row_arr = np.asarray([], dtype=np.float64)
            invalid_projection = {
                "two_theta_deg": np.full(col_arr.shape, np.nan, dtype=np.float64),
                "phi_deg": np.full(col_arr.shape, np.nan, dtype=np.float64),
                "native_cols": np.full(col_arr.shape, np.nan, dtype=np.float64),
                "native_rows": np.full(col_arr.shape, np.nan, dtype=np.float64),
                "fit_space_source": "invalid_dataset_fit_space_projector",
                "input_frame": str(input_frame or ""),
                "fit_space_projector_kind": "exact_caked_bundle",
                "cake_bundle_signature": None,
                "fit_space_local_params_signature": _geometry_fit_projection_signature(
                    _geometry_fit_transform_driven_param_payload(local_params)
                ),
                "valid": False,
                "invalid_reason": "",
                "native_frame_conversion_source": "",
                "native_frame_conversion_count": 0,
                "caked_projection_source": "fit_space_projector_native_detector",
            }
            if col_arr.shape != row_arr.shape or col_arr.size <= 0:
                invalid_projection["invalid_reason"] = "shape_mismatch"
                return invalid_projection
            if not np.all(np.isfinite(col_arr)) or not np.all(np.isfinite(row_arr)):
                invalid_projection["invalid_reason"] = "nonfinite_detector_coords"
                return invalid_projection
            input_frame_key = str(input_frame or "").strip().lower()
            native_cols = np.asarray(col_arr, dtype=np.float64).copy()
            native_rows = np.asarray(row_arr, dtype=np.float64).copy()
            native_frame_conversion_source = "identity_native_detector"
            native_frame_conversion_count = 0
            if input_frame_key == "native_detector":
                native_frame_conversion_source = "identity_native_detector"
            elif input_frame_key == "fit_detector":
                native_points: list[tuple[float, float]] = []
                for detector_col, detector_row in zip(col_arr, row_arr):
                    try:
                        native_point = _fit_detector_coords_to_native_detector_coords(
                            float(detector_col),
                            float(detector_row),
                            backend_shape=dynamic_reanchor_backend_shape,
                            orientation_choice=orientation_choice,
                            native_mapper=(
                                manual_dataset_bindings.backend_detector_coords_to_native_detector_coords
                                or manual_dataset_bindings.display_to_native_sim_coords
                            ),
                            native_shape=native_background.shape,
                        )
                    except TypeError:
                        native_point = _fit_detector_coords_to_native_detector_coords(
                            float(detector_col),
                            float(detector_row),
                        )
                    if (
                        not isinstance(native_point, tuple)
                        or len(native_point) < 2
                        or native_point[0] is None
                        or native_point[1] is None
                    ):
                        invalid_projection["invalid_reason"] = "fit_detector_to_native_failed"
                        invalid_projection["native_frame_conversion_source"] = (
                            "fit_detector_to_native_detector"
                        )
                        invalid_projection["native_frame_conversion_count"] = 1
                        return invalid_projection
                    try:
                        native_col = float(native_point[0])
                        native_row = float(native_point[1])
                    except Exception:
                        invalid_projection["invalid_reason"] = "fit_detector_to_native_failed"
                        invalid_projection["native_frame_conversion_source"] = (
                            "fit_detector_to_native_detector"
                        )
                        invalid_projection["native_frame_conversion_count"] = 1
                        return invalid_projection
                    if not (np.isfinite(native_col) and np.isfinite(native_row)):
                        invalid_projection["invalid_reason"] = "fit_detector_to_native_failed"
                        invalid_projection["native_frame_conversion_source"] = (
                            "fit_detector_to_native_detector"
                        )
                        invalid_projection["native_frame_conversion_count"] = 1
                        return invalid_projection
                    native_points.append((float(native_col), float(native_row)))
                native_cols = np.asarray(
                    [point[0] for point in native_points],
                    dtype=np.float64,
                )
                native_rows = np.asarray(
                    [point[1] for point in native_points],
                    dtype=np.float64,
                )
                native_frame_conversion_source = "fit_detector_to_native_detector"
                native_frame_conversion_count = 1
            else:
                invalid_projection["invalid_reason"] = (
                    f"unsupported_input_frame:{input_frame_key or 'missing'}"
                )
                return invalid_projection

            active_params = local_params if isinstance(local_params, Mapping) else params_i
            base_theta_value = _effective_theta_for_projection(params_i)
            active_theta_value = _effective_theta_for_projection(active_params)
            theta_adjustment_deg = 0.0
            active_bundle = _resolve_dynamic_reanchor_cached_caked_bundle(
                active_params,
                prefer_rebuild_bundle=prefer_rebuild_bundle,
            )
            if not isinstance(active_bundle, CakeTransformBundle):
                invalid_projection["invalid_reason"] = "missing_exact_caked_bundle"
                invalid_projection["native_frame_conversion_source"] = (
                    native_frame_conversion_source
                )
                invalid_projection["native_frame_conversion_count"] = int(
                    native_frame_conversion_count
                )
                return invalid_projection

            projected_two_theta: list[float] = []
            projected_phi: list[float] = []
            for native_col, native_row in zip(native_cols, native_rows):
                two_theta_deg, phi_deg = _native_detector_coords_to_caked_display_coords(
                    float(native_col),
                    float(native_row),
                    active_caked_bundle=active_bundle,
                )
                if two_theta_deg is None or phi_deg is None:
                    invalid_projection["invalid_reason"] = "native_detector_to_caked_display_failed"
                    invalid_projection["cake_bundle_signature"] = (
                        _geometry_fit_cake_bundle_signature(
                            active_bundle,
                            local_params=active_params,
                        )
                    )
                    invalid_projection["native_frame_conversion_source"] = (
                        native_frame_conversion_source
                    )
                    invalid_projection["native_frame_conversion_count"] = int(
                        native_frame_conversion_count
                    )
                    return invalid_projection
                projected_two_theta.append(float(two_theta_deg))
                projected_phi.append(float(phi_deg))

            return {
                "two_theta_deg": np.asarray(projected_two_theta, dtype=np.float64),
                "phi_deg": np.asarray(projected_phi, dtype=np.float64),
                "native_cols": np.asarray(native_cols, dtype=np.float64),
                "native_rows": np.asarray(native_rows, dtype=np.float64),
                "fit_space_source": "dataset_fit_space_projector",
                "input_frame": (
                    "fit_detector" if input_frame_key == "fit_detector" else "native_detector"
                ),
                "fit_space_projector_kind": "exact_caked_bundle",
                "cake_bundle_signature": _geometry_fit_cake_bundle_signature(
                    active_bundle,
                    local_params=active_params,
                ),
                "fit_space_local_params_signature": _geometry_fit_projection_signature(
                    _geometry_fit_transform_driven_param_payload(active_params)
                ),
                "valid": True,
                "invalid_reason": None,
                "native_frame_conversion_source": native_frame_conversion_source,
                "native_frame_conversion_count": int(native_frame_conversion_count),
                "caked_projection_source": "fit_space_projector_native_detector",
                "theta_initial_base_deg": float(base_theta_value),
                "theta_initial_active_deg": float(active_theta_value),
                "theta_initial_adjustment_applied_deg": float(theta_adjustment_deg),
            }

        def _fit_space_projector(
            cols: object,
            rows: object,
            *,
            local_params: Mapping[str, object] | None,
            anchor_kind: str,
            input_frame: str,
        ) -> dict[str, object]:
            return _project_detector_points_with_active_caked_bundle(
                cols,
                rows,
                local_params=local_params,
                anchor_kind=anchor_kind,
                input_frame=input_frame,
                prefer_rebuild_bundle=True,
            )

        def _trial_detector_image_signature(detector_image: object) -> str:
            try:
                arr = np.ascontiguousarray(np.asarray(detector_image, dtype=np.float64))
            except Exception:
                return "unavailable"
            payload = arr.tobytes() + repr((arr.shape, str(arr.dtype))).encode("utf-8")
            return hashlib.sha1(payload).hexdigest()

        def _sim_caked_image_builder(
            detector_image: object,
            *,
            local_params: Mapping[str, object] | None = None,
            axes_only: bool = False,
        ) -> dict[str, object] | None:
            active_params = local_params if isinstance(local_params, Mapping) else params_i
            try:
                detector_arr = np.asarray(detector_image, dtype=np.float64)
            except Exception:
                return {
                    "available": False,
                    "unavailable_reason": "detector_image_invalid",
                    "detector_simulation_signature": "unavailable",
                }
            active_bundle = _resolve_dynamic_reanchor_cached_caked_bundle(
                active_params,
                prefer_rebuild_bundle=True,
            )
            if not isinstance(active_bundle, CakeTransformBundle):
                return {
                    "available": False,
                    "unavailable_reason": "missing_exact_caked_bundle",
                    "detector_simulation_signature": _trial_detector_image_signature(detector_arr),
                    "fit_space_local_params_signature": _geometry_fit_projection_signature(
                        _geometry_fit_transform_driven_param_payload(active_params)
                    ),
                }
            if bool(axes_only):
                return {
                    "available": True,
                    "axes_only": True,
                    "image": None,
                    "radial_axis": np.asarray(active_bundle.radial_deg, dtype=np.float64),
                    "azimuth_axis": np.asarray(
                        active_bundle.gui_azimuth_deg,
                        dtype=np.float64,
                    ),
                    "raw_azimuth_axis": np.asarray(
                        active_bundle.raw_azimuth_deg,
                        dtype=np.float64,
                    ),
                    "detector_simulation_signature": "axes_only",
                    "caked_simulation_signature": "axes_only",
                    "fit_space_local_params_signature": _geometry_fit_projection_signature(
                        _geometry_fit_transform_driven_param_payload(active_params)
                    ),
                    "cake_bundle_signature": _geometry_fit_cake_bundle_signature(
                        active_bundle,
                        local_params=active_params,
                    ),
                    "source_rows_rebuilt_or_reused": "axes_reused_for_trial_params",
                    "reuse_valid_for_same_params_signature": True,
                }
            try:
                caked_result = integrate_detector_to_cake_lut(
                    detector_arr,
                    np.asarray(active_bundle.radial_deg, dtype=np.float64),
                    np.asarray(active_bundle.raw_azimuth_deg, dtype=np.float64),
                    active_bundle.lut,
                )
                caked_image, radial_axis, azimuth_axis = prepare_gui_phi_display(caked_result)
                caked_arr = np.asarray(caked_image, dtype=np.float64)
            except Exception as exc:
                return {
                    "available": False,
                    "unavailable_reason": f"sim_caked_integration_exception:{type(exc).__name__}",
                    "detector_simulation_signature": _trial_detector_image_signature(detector_arr),
                    "fit_space_local_params_signature": _geometry_fit_projection_signature(
                        _geometry_fit_transform_driven_param_payload(active_params)
                    ),
                    "cake_bundle_signature": _geometry_fit_cake_bundle_signature(
                        active_bundle,
                        local_params=active_params,
                    ),
                }
            return {
                "available": True,
                "image": caked_arr,
                "radial_axis": np.asarray(radial_axis, dtype=np.float64),
                "azimuth_axis": np.asarray(azimuth_axis, dtype=np.float64),
                "raw_azimuth_axis": np.asarray(active_bundle.raw_azimuth_deg, dtype=np.float64),
                "detector_simulation_signature": _trial_detector_image_signature(detector_arr),
                "caked_simulation_signature": _trial_detector_image_signature(caked_arr),
                "fit_space_local_params_signature": _geometry_fit_projection_signature(
                    _geometry_fit_transform_driven_param_payload(active_params)
                ),
                "cake_bundle_signature": _geometry_fit_cake_bundle_signature(
                    active_bundle,
                    local_params=active_params,
                ),
                "source_rows_rebuilt_or_reused": "rebuilt_for_trial_params",
                "reuse_valid_for_same_params_signature": True,
            }

        def _dynamic_reanchor_callback(
            measured_entry: Mapping[str, object] | None,
            simulated_detector_point: object,
            local_params: Mapping[str, object] | None = None,
            dataset_ctx: object = None,
        ) -> dict[str, object] | None:
            del dataset_ctx
            if not isinstance(measured_entry, Mapping):
                return None
            if (
                not isinstance(
                    simulated_detector_point,
                    (list, tuple, np.ndarray),
                )
                or len(simulated_detector_point) < 2
            ):
                return None
            try:
                sim_col = float(simulated_detector_point[0])
                sim_row = float(simulated_detector_point[1])
            except Exception:
                return None
            if not (np.isfinite(sim_col) and np.isfinite(sim_row)):
                return None

            seed_entry = dict(measured_entry)
            raw_col = None
            raw_row = None
            active_params = local_params if isinstance(local_params, Mapping) else params_i
            active_caked_bundle = (
                _resolve_dynamic_reanchor_cached_caked_bundle(
                    active_params,
                    prefer_rebuild_bundle=False,
                )
                if dynamic_reanchor_caked_view_ready
                else None
            )
            use_caked_reanchor = isinstance(active_caked_bundle, CakeTransformBundle)
            dynamic_reanchor_cache_data = {
                "match_config": dict(dynamic_reanchor_match_cfg),
                "background_context": (
                    dynamic_reanchor_caked_background_context
                    if use_caked_reanchor
                    else dynamic_reanchor_detector_background_context
                ),
            }
            dynamic_reanchor_image = (
                np.asarray(dynamic_reanchor_caked_background, dtype=np.float64)
                if use_caked_reanchor and isinstance(dynamic_reanchor_caked_background, np.ndarray)
                else np.asarray(dynamic_reanchor_detector_image, dtype=np.float64)
            )
            if use_caked_reanchor:
                sim_projection = _project_detector_points_with_active_caked_bundle(
                    np.asarray([sim_col], dtype=np.float64),
                    np.asarray([sim_row], dtype=np.float64),
                    local_params=active_params,
                    anchor_kind="simulated",
                    input_frame="fit_detector",
                    prefer_rebuild_bundle=False,
                )
                sim_two_theta_values = np.asarray(
                    sim_projection.get("two_theta_deg", []),
                    dtype=np.float64,
                ).reshape(-1)
                sim_phi_values = np.asarray(
                    sim_projection.get("phi_deg", []),
                    dtype=np.float64,
                ).reshape(-1)
                sim_two_theta = (
                    float(sim_two_theta_values[0])
                    if sim_two_theta_values.size >= 1
                    else float("nan")
                )
                sim_phi = float(sim_phi_values[0]) if sim_phi_values.size >= 1 else float("nan")
                sim_col_local = gui_manual_geometry.caked_axis_to_image_index(
                    float(sim_two_theta),
                    dynamic_reanchor_radial_axis,
                )
                sim_row_local = gui_manual_geometry.caked_axis_to_image_index(
                    float(sim_phi),
                    dynamic_reanchor_azimuth_axis,
                )
                if (
                    np.isfinite(sim_two_theta)
                    and np.isfinite(sim_phi)
                    and np.isfinite(sim_col_local)
                    and np.isfinite(sim_row_local)
                ):
                    seed_entry["sim_col"] = float(sim_two_theta)
                    seed_entry["sim_row"] = float(sim_phi)
                    seed_entry["sim_col_global"] = float(sim_two_theta)
                    seed_entry["sim_row_global"] = float(sim_phi)
                    seed_entry["sim_col_local"] = float(sim_col_local)
                    seed_entry["sim_row_local"] = float(sim_row_local)
                    raw_col = _finite_float(
                        measured_entry.get(
                            "caked_x",
                            measured_entry.get("background_two_theta_deg"),
                        )
                    )
                    raw_row = _finite_float(
                        measured_entry.get(
                            "caked_y",
                            measured_entry.get("background_phi_deg"),
                        )
                    )
                    if raw_col is None or raw_row is None:
                        measured_detector_col = _finite_float(
                            measured_entry.get("background_detector_x")
                        )
                        measured_detector_row = _finite_float(
                            measured_entry.get("background_detector_y")
                        )
                        measured_input_frame = "native_detector"
                        if measured_detector_col is None or measured_detector_row is None:
                            measured_detector_col = _finite_float(
                                measured_entry.get("fit_detector_x")
                            )
                            measured_detector_row = _finite_float(
                                measured_entry.get("fit_detector_y")
                            )
                            measured_input_frame = "fit_detector"
                        if measured_detector_col is not None and measured_detector_row is not None:
                            measured_projection = _project_detector_points_with_active_caked_bundle(
                                np.asarray(
                                    [measured_detector_col],
                                    dtype=np.float64,
                                ),
                                np.asarray(
                                    [measured_detector_row],
                                    dtype=np.float64,
                                ),
                                local_params=active_params,
                                anchor_kind="measured",
                                input_frame=measured_input_frame,
                                prefer_rebuild_bundle=False,
                            )
                            measured_two_theta_values = np.asarray(
                                measured_projection.get("two_theta_deg", []),
                                dtype=np.float64,
                            ).reshape(-1)
                            measured_phi_values = np.asarray(
                                measured_projection.get("phi_deg", []),
                                dtype=np.float64,
                            ).reshape(-1)
                            measured_two_theta = (
                                float(measured_two_theta_values[0])
                                if measured_two_theta_values.size >= 1
                                else float("nan")
                            )
                            measured_phi = (
                                float(measured_phi_values[0])
                                if measured_phi_values.size >= 1
                                else float("nan")
                            )
                            if np.isfinite(measured_two_theta) and np.isfinite(measured_phi):
                                raw_col = float(measured_two_theta)
                                raw_row = float(measured_phi)
            if raw_col is None or raw_row is None:
                seed_entry["sim_col"] = float(sim_col)
                seed_entry["sim_row"] = float(sim_row)
                seed_entry["sim_col_local"] = float(sim_col)
                seed_entry["sim_row_local"] = float(sim_row)
                seed_entry["sim_col_global"] = float(sim_col)
                seed_entry["sim_row_global"] = float(sim_row)
                raw_col = _finite_float(measured_entry.get("background_detector_x"))
                raw_row = _finite_float(measured_entry.get("background_detector_y"))
                if raw_col is None or raw_row is None:
                    raw_col = _finite_float(measured_entry.get("detector_x"))
                    raw_row = _finite_float(measured_entry.get("detector_y"))
                if raw_col is None or raw_row is None:
                    raw_col = float(sim_col)
                    raw_row = float(sim_row)
            try:
                refined_col, refined_row = gui_manual_geometry.geometry_manual_refine_preview_point(
                    seed_entry,
                    float(raw_col),
                    float(raw_row),
                    display_background=dynamic_reanchor_image,
                    cache_data=dynamic_reanchor_cache_data,
                    use_caked_space=bool(use_caked_reanchor),
                    radial_axis=(dynamic_reanchor_radial_axis if use_caked_reanchor else None),
                    azimuth_axis=(dynamic_reanchor_azimuth_axis if use_caked_reanchor else None),
                    match_simulated_peaks_to_peak_context=(match_simulated_peaks_to_peak_context),
                )
            except Exception:
                return None
            if not (np.isfinite(refined_col) and np.isfinite(refined_row)):
                return None
            if use_caked_reanchor:
                return {
                    "background_two_theta_deg": float(refined_col),
                    "background_phi_deg": float(refined_row),
                    "fit_space_anchor_override": True,
                    "measured_reanchor_motion_px": 0.0,
                }
            return {
                "x": float(refined_col),
                "y": float(refined_row),
                "detector_x": float(refined_col),
                "detector_y": float(refined_row),
                "background_two_theta_deg": float("nan"),
                "background_phi_deg": float("nan"),
            }

        dynamic_reanchor_callback = _dynamic_reanchor_callback
        if dynamic_reanchor_caked_view_ready and dynamic_reanchor_native_shape is not None:
            fit_space_projector = _fit_space_projector
            fit_space_projector_kind = "exact_caked_bundle"
            fit_space_projector_unavailable_reason = None
            sim_caked_image_builder = _sim_caked_image_builder
            sim_caked_image_builder_kind = "exact_caked_lut"
        elif fit_space_projector_unavailable_reason == "missing_exact_caked_bundle":
            pass
        elif not dynamic_reanchor_caked_view_ready:
            fit_space_projector_unavailable_reason = "exact_caked_view_unavailable"
        else:
            fit_space_projector_unavailable_reason = "native_detector_shape_unavailable"

    label = (
        Path(str(manual_dataset_bindings.osc_files[background_idx])).name
        if 0 <= background_idx < len(manual_dataset_bindings.osc_files)
        else f"background_{background_idx}"
    )
    group_count = len(
        {
            entry.get("q_group_key")
            for entry in selected_entries
            if entry.get("q_group_key") is not None
        }
    )
    cache_metadata = build_geometry_fit_dataset_cache_metadata(
        background_index=int(background_idx),
        current_background_index=int(manual_dataset_bindings.current_background_index),
        simulated_peaks=simulated_peaks,
        source_snapshot_diagnostics=simulation_diagnostics,
        source_resolution_diagnostics=source_resolution_diagnostics,
        pair_count=int(len(measured_display)),
        resolved_source_pair_count=int(resolved_source_pair_count),
    )
    _emit_geometry_fit_stage_event(
        stage_callback,
        "dataset_ready",
        background_index=int(background_idx),
        pair_count=int(len(measured_display)),
        resolved_source_pair_count=int(resolved_source_pair_count),
        message=(
            "preflight: dataset ready for "
            f"background {int(background_idx) + 1} "
            f"({int(resolved_source_pair_count)}/{int(len(selected_entries))} source pairs resolved)"
        ),
    )

    dataset_payload = {
        "dataset_index": int(background_idx),
        "label": label,
        "theta_base": float(theta_base),
        "theta_effective": float(theta_base + theta_offset),
        "image_size": int(manual_dataset_bindings.image_size),
        "display_rotate_k": int(manual_dataset_bindings.display_rotate_k),
        "native_detector_coords_to_detector_display_coords": native_to_display_callback,
        "native_detector_coords_to_detector_display_coords_source": native_to_display_source,
        "native_detector_coords_to_detector_display_coords_unavailable_reason": (
            native_to_display_unavailable_reason
        ),
        "group_count": int(group_count),
        "pair_count": int(len(measured_display)),
        "resolved_source_pair_count": int(resolved_source_pair_count),
        "simulated_peak_count": int(
            len([item for item in simulated_peaks or () if isinstance(item, Mapping)])
        ),
        "simulated_lookup_count": int(len(simulated_lookup)),
        "simulation_diagnostics": (
            simulation_diagnostics if isinstance(simulation_diagnostics, Mapping) else {}
        ),
        "cache_metadata": cache_metadata,
        "source_resolution_diagnostics": source_resolution_diagnostics,
        "source_rows_for_trace": [
            dict(entry) for entry in (simulated_peaks or ()) if isinstance(entry, Mapping)
        ],
        "manual_picker_truth_pairs": manual_picker_truth_pairs,
        "provider_pairs": provider_pairs,
        "manual_point_pairs": dataset_manual_point_pairs,
        "measured_display": measured_display,
        "measured_native": measured_native,
        "measured_for_fit": measured_for_fit,
        "initial_pairs_display": initial_pairs_display,
        "native_background": native_background,
        "orientation_choice": orientation_choice,
        "orientation_diag": orientation_diag,
        "summary_line": (
            "bg[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
            "groups={groups} points={points} orientation={orientation}"
        ).format(
            idx=background_idx,
            name=label,
            theta_base=float(theta_base),
            theta_eff=float(theta_base + theta_offset),
            groups=int(group_count),
            points=int(len(measured_display)),
            orientation=orientation_choice.get("label", "identity"),
        ),
        "spec": {
            "dataset_index": int(background_idx),
            "label": label,
            "theta_initial": float(theta_base),
            "measured_peaks": measured_for_fit,
            "experimental_image": experimental_image_for_fit,
            "dynamic_reanchor_callback": dynamic_reanchor_callback,
            "dynamic_reanchor_enabled": bool(dynamic_reanchor_enabled),
            "fit_space_projector": fit_space_projector,
            "fit_space_projector_kind": fit_space_projector_kind,
            "fit_space_projector_unavailable_reason": (fit_space_projector_unavailable_reason),
            "sim_caked_image_builder": sim_caked_image_builder,
            "sim_caked_image_builder_kind": sim_caked_image_builder_kind,
            "qr_fit_trial_source_rows_builder": qr_fit_trial_source_rows_builder,
            "qr_fit_trial_source_rows_builder_kind": (qr_fit_trial_source_rows_builder_kind),
            "baseline_fit_params": dict(baseline_fit_params_i),
        },
    }
    dataset_payload["manual_point_pairs"] = _geometry_fit_dataset_pairs_from_handoff(
        dataset_payload,
        provider_pairs,
    )
    dataset_payload["spec"]["manual_point_pairs"] = dataset_payload["manual_point_pairs"]
    fit_handoff_audit_rows = build_geometry_fit_qr_handoff_audit_rows(
        dataset_payload,
        base_fit_params=params_i,
    )
    fit_handoff_audit_lines = build_geometry_fit_qr_handoff_audit_lines(
        fit_handoff_audit_rows,
    )
    dataset_payload["fit_handoff_audit_rows"] = fit_handoff_audit_rows
    dataset_payload["fit_handoff_audit_lines"] = fit_handoff_audit_lines
    for audit_line in fit_handoff_audit_lines:
        _emit_geometry_fit_stage_event(stage_callback, "cmd_line", text=audit_line)
    dataset_payload["point_provider_report"] = build_geometry_fit_point_provider_report(
        dataset_payload
    )
    return dataset_payload


def prepare_geometry_fit_point_pairs(
    background_index: int,
    *,
    theta_base: float,
    base_fit_params: Mapping[str, object] | None,
    manual_dataset_bindings: GeometryFitRuntimeManualDatasetBindings,
    orientation_cfg: Mapping[str, object] | None = None,
    stage_callback: GeometryFitStageCallback | None = None,
) -> list[dict[str, object]]:
    """Build the pre-optimizer manual point-provider pairs only."""

    dataset = build_geometry_manual_fit_dataset(
        background_index,
        theta_base=theta_base,
        base_fit_params=base_fit_params,
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg=orientation_cfg,
        stage_callback=stage_callback,
    )
    return [
        dict(pair) for pair in dataset.get("provider_pairs", ()) or () if isinstance(pair, Mapping)
    ]


def _manual_geometry_fit_preflight_error(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> str | None:
    """Return a preflight error when saved manual pairs are no longer trustworthy."""

    if not dataset_infos:
        return None

    min_resolved_pairs_per_background = 3
    min_resolved_fraction_per_background = 0.9
    orientation_failures: list[str] = []
    partially_unresolved_sources: list[str] = []
    resolved_source_total = 0
    resolved_source_known = False
    total_pairs = 0
    for idx, dataset in enumerate(dataset_infos):
        if not isinstance(dataset, Mapping):
            continue
        label = str(dataset.get("label", f"background_{idx + 1}"))
        try:
            pair_count = max(0, int(dataset.get("pair_count", 0)))
        except Exception:
            pair_count = 0
        total_pairs += int(pair_count)

        orientation_diag = dataset.get("orientation_diag")
        if isinstance(orientation_diag, Mapping) and pair_count > 0:
            try:
                orientation_pairs = int(orientation_diag.get("pairs", 0))
            except Exception:
                orientation_pairs = 0
            if orientation_pairs <= 0:
                orientation_failures.append(label)

        if "resolved_source_pair_count" in dataset:
            resolved_source_known = True
            try:
                resolved_count = max(
                    0,
                    int(dataset.get("resolved_source_pair_count", 0)),
                )
                resolved_source_total += int(resolved_count)
                if pair_count > 0 and int(resolved_count) < int(pair_count):
                    resolved_fraction = float(resolved_count) / float(pair_count)
                    if int(resolved_count) >= int(
                        min_resolved_pairs_per_background
                    ) and resolved_fraction >= float(min_resolved_fraction_per_background):
                        continue
                    partially_unresolved_sources.append(
                        "{label} ({resolved}/{total})".format(
                            label=label,
                            resolved=int(resolved_count),
                            total=int(pair_count),
                        )
                    )
            except Exception:
                pass

    if orientation_failures:
        joined = ", ".join(orientation_failures)
        return (
            "Geometry fit unavailable: orientation preflight produced no usable "
            f"simulated/measured anchor pairs for {joined}. Refresh the picks "
            "before fitting."
        )

    if resolved_source_known and total_pairs > 0 and resolved_source_total <= 0:
        return (
            "Geometry fit unavailable: saved manual pairs no longer resolve to "
            "current simulated source rows on any selected background. Refresh "
            "the picks before fitting."
        )
    if partially_unresolved_sources:
        joined = ", ".join(partially_unresolved_sources)
        return (
            "Geometry fit unavailable: some saved manual pairs no longer "
            "resolve to current simulated source rows: "
            f"{joined}. Refresh the picks before fitting."
        )
    return None


def prepare_geometry_fit_run(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    fit_config: Mapping[str, object] | None,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial: object,
    preserve_live_theta: bool,
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., list[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., list[float]],
    current_geometry_theta_offset: Callable[..., float],
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    ensure_geometry_fit_caked_view: Callable[[], None],
    build_dataset: Callable[..., dict[str, object]],
    build_runtime_config: Callable[[Mapping[str, object]], dict[str, object]],
    require_selected_var_names: bool = True,
    require_active_background_in_selection: bool = True,
    include_all_selected_backgrounds: bool | None = None,
    manual_fit_pick_uses_caked_space: bool = False,
    stage_callback: GeometryFitStageCallback | None = None,
) -> GeometryFitPreparationResult:
    """Validate and assemble the manual-pair geometry-fit runtime inputs."""

    _ = ensure_geometry_fit_caked_view
    selected_var_names = [str(name) for name in (var_names or ())]
    fit_params = dict(params or {})
    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, Mapping) else {}
    if not isinstance(geometry_refine_cfg, Mapping):
        geometry_refine_cfg = {}

    current_index = int(current_background_index)
    _emit_geometry_fit_stage_event(
        stage_callback,
        "prepare_start",
        message="preflight: collecting geometry-fit datasets",
        selected_var_names=list(selected_var_names),
        current_background_index=int(current_index),
    )

    def _failure_result(
        error_text: str,
        *,
        dataset_infos: Sequence[Mapping[str, object]] | None = None,
        current_dataset: Mapping[str, object] | None = None,
        selected_background_indices: Sequence[object] = (),
        joint_background_mode: bool = False,
        geometry_runtime_cfg: Mapping[str, object] | None = None,
    ) -> GeometryFitPreparationResult:
        log_runtime_cfg = (
            geometry_runtime_cfg
            if isinstance(geometry_runtime_cfg, Mapping)
            else geometry_refine_cfg
        )
        _emit_geometry_fit_stage_event(
            stage_callback,
            "preflight_failure",
            error_text=str(error_text),
            selected_background_indices=[int(idx) for idx in (selected_background_indices or ())],
            joint_background_mode=bool(joint_background_mode),
            message=str(error_text),
        )
        return GeometryFitPreparationResult(
            error_text=str(error_text),
            failure_log_sections=build_geometry_fit_preflight_log_sections(
                error_text=str(error_text),
                params=fit_params,
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
                selected_background_indices=selected_background_indices,
                joint_background_mode=joint_background_mode,
                geometry_runtime_cfg=log_runtime_cfg,
            ),
        )

    if require_selected_var_names and not selected_var_names:
        return _failure_result("No geometry parameters are selected for fitting.")
    lattice_refinement_cfg = geometry_refine_cfg.get("lattice_refinement", {}) or {}
    if not isinstance(lattice_refinement_cfg, Mapping):
        lattice_refinement_cfg = {}
    lattice_refinement_enabled = bool(lattice_refinement_cfg.get("enabled", False))
    selected_lattice_vars = [name for name in selected_var_names if name in {"a", "c"}]
    if selected_lattice_vars and not lattice_refinement_enabled:
        joined = ", ".join(selected_lattice_vars)
        return _failure_result(
            (
                "Geometry fit unavailable: lattice parameters "
                f"({joined}) are frozen by default. Enable "
                "`fit.geometry.lattice_refinement.enabled` to refine them."
            )
        )
    orientation_cfg = geometry_refine_cfg.get("orientation", {}) or {}
    if not isinstance(orientation_cfg, Mapping):
        orientation_cfg = {}
    overlay_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(overlay_cfg, Mapping):
        overlay_cfg = {}
    max_display_markers = max(1, int(overlay_cfg.get("max_display_markers", 120)))

    if not osc_files:
        return _failure_result("Geometry fit unavailable: no background image is loaded.")

    if not apply_geometry_fit_background_selection(
        trigger_update=False,
        sync_live_theta=not preserve_live_theta,
    ):
        return GeometryFitPreparationResult()

    try:
        selected_background_indices = current_geometry_fit_background_indices(strict=True)
    except Exception as exc:
        return _failure_result(
            (f"Geometry fit unavailable: invalid fit background selection ({exc}).")
        )
    selected_background_indices = [int(idx) for idx in selected_background_indices]
    if not selected_background_indices:
        return _failure_result("Geometry fit unavailable: no fit backgrounds are selected.")

    if current_index in set(selected_background_indices):
        primary_index = int(current_index)
    elif require_active_background_in_selection:
        return _failure_result(
            (
                "Geometry fit unavailable: the active background must be part of "
                "the fit selection so the overlay can be drawn on the current image."
            ),
            selected_background_indices=selected_background_indices,
        )
    else:
        primary_index = int(selected_background_indices[0])

    joint_background_mode = False
    background_theta_values: list[float] = []
    if geometry_fit_uses_shared_theta_offset(selected_background_indices):
        if not apply_background_theta_metadata(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        ):
            return GeometryFitPreparationResult()
        try:
            background_theta_values = list(current_background_theta_values(strict_count=True))
            fit_params["theta_offset"] = float(current_geometry_theta_offset(strict=True))
        except Exception as exc:
            return _failure_result(
                (f"Geometry fit unavailable: failed to parse background theta settings ({exc})."),
                selected_background_indices=selected_background_indices,
                joint_background_mode=len(selected_background_indices) > 1,
            )
        joint_background_mode = len(selected_background_indices) > 1
        if background_theta_values:
            fit_params["theta_initial"] = float(background_theta_values[primary_index])
    else:
        fit_params["theta_offset"] = 0.0
        theta_default = float(fit_params.get("theta_initial", theta_initial))
        build_all_selected_backgrounds = bool(include_all_selected_backgrounds)
        if build_all_selected_backgrounds:
            try:
                background_theta_values = list(current_background_theta_values(strict_count=True))
            except Exception:
                try:
                    background_theta_values = list(
                        current_background_theta_values(strict_count=False)
                    )
                except Exception:
                    background_theta_values = []
            required_theta_count = max(selected_background_indices) + 1
            if len(background_theta_values) < required_theta_count:
                background_theta_values.extend(
                    [float(theta_default)] * (required_theta_count - len(background_theta_values))
                )
            normalized_theta_values: list[float] = []
            for raw_value in background_theta_values:
                try:
                    theta_value = float(raw_value)
                except Exception:
                    theta_value = float(theta_default)
                if not np.isfinite(theta_value):
                    theta_value = float(theta_default)
                normalized_theta_values.append(float(theta_value))
            background_theta_values = normalized_theta_values
            fit_params["theta_initial"] = float(background_theta_values[primary_index])
        else:
            fit_params["theta_initial"] = float(theta_default)
            background_theta_values = [float(fit_params["theta_initial"])]

    build_all_selected_backgrounds = (
        bool(joint_background_mode)
        if include_all_selected_backgrounds is None
        else bool(include_all_selected_backgrounds)
    )

    required_indices = (
        list(selected_background_indices)
        if build_all_selected_backgrounds
        else [int(primary_index)]
    )
    missing_indices = []
    for idx in selected_background_indices:
        enabled_pairs = [
            entry
            for entry in (geometry_manual_pairs_for_index(int(idx)) or ())
            if geometry_manual_pair_enabled_for_geometry_fit(entry)
        ]
        if not enabled_pairs:
            missing_indices.append(idx)
    if missing_indices:
        missing_names = [
            Path(str(osc_files[idx])).name for idx in missing_indices if 0 <= idx < len(osc_files)
        ]
        return _failure_result(
            (
                "Geometry fit unavailable: save manual Qr/Qz pairs first for "
                + ", ".join(missing_names or [f"background {idx + 1}" for idx in missing_indices])
                + "."
            ),
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
        )

    selected_manual_fit_space_by_background = geometry_manual_fit_space_by_background(
        selected_background_indices,
        geometry_manual_pairs_for_index,
        pick_uses_caked_space=bool(manual_fit_pick_uses_caked_space),
        current_background_index=int(current_index),
    )
    fit_space_error = manual_geometry_fit_space_preflight_error(
        selected_manual_fit_space_by_background,
        osc_files=osc_files,
    )
    if fit_space_error:
        return _failure_result(
            fit_space_error,
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
        )
    selected_caked_background_indices = sorted(
        {
            int(idx)
            for idx in selected_background_indices
            if str(selected_manual_fit_space_by_background.get(int(idx), "")).strip().lower()
            == "caked"
        }
    )
    manual_pairs_use_caked_space = bool(selected_caked_background_indices)
    if manual_pairs_use_caked_space:
        try:
            ensure_geometry_fit_caked_view()
        except Exception as exc:
            unavailable_backgrounds = ", ".join(
                f"background {int(idx) + 1}" for idx in selected_caked_background_indices
            )
            error_text = (
                "Geometry fit unavailable: exact caked fit-space projector could not be "
                "prepared. Rebuild the caked/source cache and rerun the fit. "
                f"exact caked projector unavailable for {unavailable_backgrounds}."
            )
            detail = str(exc).strip()
            if detail:
                error_text = f"{error_text} Details: {detail}"
            return _failure_result(
                error_text,
                selected_background_indices=selected_background_indices,
                joint_background_mode=joint_background_mode,
            )

    def _theta_base_for_index(dataset_index: int) -> float:
        if build_all_selected_backgrounds:
            return float(background_theta_values[int(dataset_index)])
        if joint_background_mode:
            return float(background_theta_values[int(dataset_index)])
        return float(fit_params.get("theta_initial", theta_initial))

    def _build_dataset_entry(
        dataset_index: int,
        *,
        theta_base_value: float,
    ) -> dict[str, object]:
        build_kwargs = {
            "theta_base": float(theta_base_value),
            "base_fit_params": fit_params,
            "orientation_cfg": dict(orientation_cfg),
        }
        if callable(stage_callback):
            try:
                signature = inspect.signature(build_dataset)
            except (TypeError, ValueError):
                signature = None
            accepts_var_kwargs = bool(
                signature is not None
                and any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if signature is None or "stage_callback" in signature.parameters or accepts_var_kwargs:
                build_kwargs["stage_callback"] = stage_callback
        return build_dataset(
            int(dataset_index),
            **build_kwargs,
        )

    current_theta_base = _theta_base_for_index(primary_index)
    _emit_geometry_fit_stage_event(
        stage_callback,
        "current_dataset_start",
        background_index=int(primary_index),
        message=(f"preflight: building active dataset for background {int(primary_index) + 1}"),
    )
    current_dataset = _build_dataset_entry(
        int(primary_index),
        theta_base_value=float(current_theta_base),
    )
    dataset_infos = [current_dataset]
    if build_all_selected_backgrounds:
        for bg_idx in selected_background_indices:
            idx = int(bg_idx)
            if idx == primary_index:
                continue
            _emit_geometry_fit_stage_event(
                stage_callback,
                "additional_dataset_start",
                background_index=int(idx),
                message=(f"preflight: building additional dataset for background {int(idx) + 1}"),
            )
            dataset_infos.append(
                _build_dataset_entry(
                    idx,
                    theta_base_value=float(_theta_base_for_index(idx)),
                )
            )

    preflight_error = _manual_geometry_fit_preflight_error(dataset_infos)
    if preflight_error:
        return _failure_result(
            preflight_error,
            dataset_infos=dataset_infos,
            current_dataset=current_dataset,
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
        )

    dataset_specs = build_geometry_fit_dataset_specs(dataset_infos)
    manual_fit_uses_caked_space = bool(
        manual_pairs_use_caked_space or geometry_fit_datasets_use_caked_fit_space(dataset_infos)
    )
    base_runtime_cfg = apply_joint_geometry_fit_runtime_safety_overrides(
        build_runtime_config(fit_params),
        joint_background_mode=joint_background_mode,
    )
    if manual_fit_uses_caked_space:
        geometry_runtime_cfg = apply_manual_caked_point_geometry_fit_runtime_overrides(
            base_runtime_cfg,
            joint_background_mode=joint_background_mode,
        )
        projector_error = manual_caked_geometry_fit_projector_preflight_error(dataset_specs)
        if projector_error:
            return _failure_result(
                projector_error,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
                selected_background_indices=selected_background_indices,
                joint_background_mode=joint_background_mode,
                geometry_runtime_cfg=geometry_runtime_cfg,
            )
    else:
        geometry_runtime_cfg = apply_manual_point_geometry_fit_runtime_overrides(
            base_runtime_cfg,
            joint_background_mode=joint_background_mode,
        )
    _emit_geometry_fit_stage_event(
        stage_callback,
        "prepare_ready",
        dataset_count=int(len(dataset_infos)),
        joint_background_mode=bool(joint_background_mode),
        message=(
            "preflight: ready to solve geometry fit "
            f"({int(len(dataset_infos))} dataset{'s' if len(dataset_infos) != 1 else ''})"
        ),
    )
    return GeometryFitPreparationResult(
        prepared_run=GeometryFitPreparedRun(
            fit_params=fit_params,
            selected_background_indices=[int(idx) for idx in selected_background_indices],
            background_theta_values=[float(value) for value in background_theta_values],
            joint_background_mode=bool(joint_background_mode),
            current_dataset=current_dataset,
            dataset_infos=dataset_infos,
            dataset_specs=dataset_specs,
            start_cmd_line=build_geometry_fit_start_cmd_line(
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
            ),
            start_log_sections=build_geometry_fit_start_log_sections(
                params=fit_params,
                var_names=selected_var_names,
                dataset_infos=dataset_infos,
                current_dataset=current_dataset,
                selected_background_indices=selected_background_indices,
                joint_background_mode=joint_background_mode,
                geometry_runtime_cfg=geometry_runtime_cfg,
            ),
            max_display_markers=int(max_display_markers),
            geometry_runtime_cfg=geometry_runtime_cfg,
        )
    )


def prepare_runtime_geometry_fit_run(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    preserve_live_theta: bool,
    bindings: GeometryFitRuntimePreparationBindings,
    stage_callback: GeometryFitStageCallback | None = None,
) -> GeometryFitPreparationResult:
    """Prepare one geometry fit from the live runtime value/callback sources."""

    fit_config = bindings.fit_config if isinstance(bindings.fit_config, Mapping) else {}
    manual_dataset_bindings = bindings.manual_dataset_bindings
    try:
        manual_fit_pick_uses_caked_space = (
            bool(manual_dataset_bindings.pick_uses_caked_space())
            if callable(manual_dataset_bindings.pick_uses_caked_space)
            else False
        )
    except Exception:
        manual_fit_pick_uses_caked_space = False

    return prepare_geometry_fit_run(
        params=params,
        var_names=var_names,
        fit_config=fit_config,
        osc_files=manual_dataset_bindings.osc_files,
        current_background_index=int(manual_dataset_bindings.current_background_index),
        theta_initial=bindings.theta_initial,
        preserve_live_theta=preserve_live_theta,
        apply_geometry_fit_background_selection=(bindings.apply_geometry_fit_background_selection),
        current_geometry_fit_background_indices=(bindings.current_geometry_fit_background_indices),
        geometry_fit_uses_shared_theta_offset=(bindings.geometry_fit_uses_shared_theta_offset),
        apply_background_theta_metadata=bindings.apply_background_theta_metadata,
        current_background_theta_values=bindings.current_background_theta_values,
        current_geometry_theta_offset=bindings.current_geometry_theta_offset,
        geometry_manual_pairs_for_index=(manual_dataset_bindings.geometry_manual_pairs_for_index),
        ensure_geometry_fit_caked_view=bindings.ensure_geometry_fit_caked_view,
        build_dataset=(
            lambda background_index, *, theta_base, base_fit_params, orientation_cfg, stage_callback=None: (
                build_geometry_manual_fit_dataset(
                    background_index,
                    theta_base=theta_base,
                    base_fit_params=base_fit_params,
                    manual_dataset_bindings=manual_dataset_bindings,
                    orientation_cfg=orientation_cfg,
                    stage_callback=stage_callback,
                )
            )
        ),
        build_runtime_config=(
            lambda fit_params: bindings.build_runtime_config(dict(fit_params or {}))
        ),
        manual_fit_pick_uses_caked_space=bool(manual_fit_pick_uses_caked_space),
        stage_callback=stage_callback,
    )


def build_geometry_fit_dataset_specs(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    """Copy optimizer dataset specs from the prepared dataset metadata."""

    dataset_specs: list[dict[str, object]] = []
    for info in dataset_infos or ():
        if not isinstance(info, Mapping):
            continue
        spec = info.get("spec")
        if isinstance(spec, Mapping):
            dataset_specs.append(dict(spec))
    return dataset_specs


def _geometry_fit_finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _geometry_manual_entry_uses_caked_fit_space(entry: object) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if _entry_has_stale_caked_fields(entry):
        return False
    two_theta = _geometry_fit_finite_float(entry.get("background_two_theta_deg"))
    phi = _geometry_fit_finite_float(entry.get("background_phi_deg"))
    if two_theta is not None and phi is not None:
        return True
    caked_x = _geometry_fit_finite_float(entry.get("caked_x"))
    caked_y = _geometry_fit_finite_float(entry.get("caked_y"))
    if caked_x is not None and caked_y is not None:
        return True
    raw_caked_x = _geometry_fit_finite_float(entry.get("raw_caked_x"))
    raw_caked_y = _geometry_fit_finite_float(entry.get("raw_caked_y"))
    return raw_caked_x is not None and raw_caked_y is not None


def geometry_manual_pairs_use_caked_fit_space(
    manual_pairs: object,
) -> bool:
    """Return whether saved manual pairs carry caked fit-space anchors."""

    if not isinstance(manual_pairs, Sequence) or isinstance(manual_pairs, (str, bytes)):
        return False
    return any(_geometry_manual_entry_uses_caked_fit_space(entry) for entry in manual_pairs)


def _geometry_manual_pair_fit_space_kinds(manual_pairs: object) -> set[str]:
    if not isinstance(manual_pairs, Sequence) or isinstance(manual_pairs, (str, bytes)):
        return set()
    kinds: set[str] = set()
    for entry in manual_pairs:
        if not isinstance(entry, Mapping):
            continue
        if _geometry_manual_entry_uses_caked_fit_space(entry):
            kinds.add("caked")
        else:
            kinds.add("detector")
    return kinds


def geometry_manual_pairs_fit_space_kind(
    manual_pairs: object,
    *,
    pick_uses_caked_space: bool = False,
    pick_applies_to_background: bool = False,
) -> str:
    """Classify one manual-pair set as detector-pixel or caked fit-space."""

    pair_kinds = _geometry_manual_pair_fit_space_kinds(manual_pairs)
    if len(pair_kinds) > 1:
        return "mixed"
    if pair_kinds == {"caked"}:
        return "caked"
    if bool(pick_uses_caked_space) and bool(pick_applies_to_background):
        return "caked"
    return "detector"


def geometry_manual_fit_space_by_background(
    background_indices: Sequence[object] | None,
    pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]] | Mapping[object, object],
    *,
    pick_uses_caked_space: bool = False,
    current_background_index: int | None = None,
) -> dict[int, str]:
    """Return detector/caked manual fit-space classification per background."""

    indices = [int(idx) for idx in (background_indices or ())]
    result: dict[int, str] = {}
    for idx in indices:
        if callable(pairs_for_index):
            pairs = pairs_for_index(int(idx))
        elif isinstance(pairs_for_index, Mapping):
            pairs = pairs_for_index.get(int(idx), pairs_for_index.get(str(int(idx)), ()))
        else:
            pairs = ()
        pairs = [
            entry for entry in (pairs or ()) if geometry_manual_pair_enabled_for_geometry_fit(entry)
        ]
        pick_applies = bool(pick_uses_caked_space) and (
            len(indices) == 1
            or (current_background_index is not None and int(idx) == int(current_background_index))
        )
        result[int(idx)] = geometry_manual_pairs_fit_space_kind(
            pairs,
            pick_uses_caked_space=bool(pick_uses_caked_space),
            pick_applies_to_background=bool(pick_applies),
        )
    return result


def manual_geometry_fit_space_preflight_error(
    fit_space_by_background: Mapping[object, object] | None,
    *,
    osc_files: Sequence[object] | None = None,
) -> str | None:
    """Reject mixed detector/caked manual-pair selections before overrides run."""

    normalized: dict[int, str] = {}
    for raw_idx, raw_kind in (fit_space_by_background or {}).items():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        kind = str(raw_kind or "detector").strip().lower()
        normalized[idx] = kind if kind in {"caked", "mixed"} else "detector"
    mixed_indices = [idx for idx, kind in normalized.items() if kind == "mixed"]
    if mixed_indices:
        labels: list[str] = []
        osc_list = list(osc_files or ())
        for idx in sorted(mixed_indices):
            if 0 <= int(idx) < len(osc_list):
                label = Path(str(osc_list[int(idx)])).name
            else:
                label = f"background {int(idx) + 1}"
            labels.append(f"{label}=mixed")
        return (
            "Geometry fit unavailable: saved manual Qr/Qz pairs mix detector-pixel "
            "and caked fit-space coordinates within the same background. Clear and "
            "re-pick the selected groups in one view before fitting. Fit spaces: "
            + ", ".join(labels)
            + "."
        )
    if len(set(normalized.values())) <= 1:
        return None

    labels: list[str] = []
    osc_list = list(osc_files or ())
    for idx in sorted(normalized):
        if 0 <= int(idx) < len(osc_list):
            label = Path(str(osc_list[int(idx)])).name
        else:
            label = f"background {int(idx) + 1}"
        labels.append(f"{label}={normalized[idx]}")
    return (
        "Geometry fit unavailable: selected manual Qr/Qz pairs mix detector-pixel "
        "and caked fit-space coordinates. Clear and re-pick the selected groups in "
        "one view before fitting. Fit spaces: " + ", ".join(labels) + "."
    )


def geometry_fit_datasets_use_caked_fit_space(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> bool:
    """Classify prepared manual datasets before applying runtime overrides."""

    for info in dataset_infos or ():
        if not isinstance(info, Mapping):
            continue
        for key in (
            "measured_for_fit",
            "measured_display",
            "measured_native",
            "initial_pairs_display",
            "provider_pairs",
            "manual_picker_truth_pairs",
            "manual_point_pairs",
        ):
            if geometry_manual_pairs_use_caked_fit_space(info.get(key)):
                return True
        spec = info.get("spec")
        if isinstance(spec, Mapping):
            if geometry_manual_pairs_use_caked_fit_space(spec.get("measured_peaks")):
                return True
    return False


def manual_caked_geometry_fit_projector_preflight_error(
    dataset_specs: Sequence[Mapping[str, object]] | None,
) -> str | None:
    """Fail closed when caked manual fits cannot use exact caked projection."""

    missing_labels: list[str] = []
    for idx, spec in enumerate(dataset_specs or ()):
        if not isinstance(spec, Mapping):
            missing_labels.append(f"dataset {idx + 1}")
            continue
        projector = spec.get("fit_space_projector")
        projector_kind = str(spec.get("fit_space_projector_kind") or "")
        if projector_kind != "exact_caked_bundle" or not callable(projector):
            label = str(spec.get("label") or f"dataset {idx + 1}")
            reason = str(spec.get("fit_space_projector_unavailable_reason") or "missing")
            missing_labels.append(f"{label} ({reason})")
    if not missing_labels:
        return None
    return (
        "Geometry fit unavailable: caked manual Qr/Qz pairs require an exact "
        "caked fit-space projector for every selected background. Rebuild the "
        "caked/source cache and rerun the fit. Missing exact projector for "
        + ", ".join(missing_labels)
        + "."
    )


def apply_joint_geometry_fit_runtime_safety_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Apply interactive multi-background runtime safeguards without serializing."""

    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    if not joint_background_mode:
        return cfg
    if bool(cfg.get("allow_unsafe_runtime", False)):
        return cfg

    solver_cfg_raw = cfg.get("solver", {})
    solver_cfg = dict(solver_cfg_raw) if isinstance(solver_cfg_raw, Mapping) else {}
    solver_cfg["stagnation_probe"] = False
    solver_cfg["stagnation_probe_pairwise"] = False
    solver_cfg["stagnation_probe_random_directions"] = 0
    if "restarts" in solver_cfg:
        try:
            solver_cfg["restarts"] = min(max(int(solver_cfg["restarts"]), 0), 1)
        except Exception:
            solver_cfg["restarts"] = 1
    cfg["solver"] = solver_cfg

    identifiability_cfg_raw = cfg.get("identifiability", {})
    identifiability_cfg = (
        dict(identifiability_cfg_raw) if isinstance(identifiability_cfg_raw, Mapping) else {}
    )
    identifiability_cfg["enabled"] = False
    cfg["identifiability"] = identifiability_cfg
    return cfg


def apply_manual_point_geometry_fit_runtime_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Build the lean runtime profile for raw detector-pixel manual point fitting."""

    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    cfg.pop("projection_view_mode", None)
    unsafe_runtime_enabled = bool(cfg.get("allow_unsafe_runtime", False))

    optimizer_cfg_raw = cfg.get("solver", cfg.get("optimizer", {}))
    optimizer_cfg = dict(optimizer_cfg_raw) if isinstance(optimizer_cfg_raw, Mapping) else {}
    for key in (
        "dynamic_point_geometry_fit",
        "restarts",
        "restart_jitter",
        "stagnation_probe",
        "stagnation_probe_pairwise",
        "stagnation_probe_pair_limit",
        "stagnation_probe_random_directions",
        "staged_release",
        "reparameterize_pairs",
    ):
        optimizer_cfg.pop(key, None)
    optimizer_cfg["manual_point_fit_mode"] = True
    optimizer_cfg["missing_pair_penalty_px"] = float(
        optimizer_cfg.get("missing_pair_penalty_px", 20.0)
    )
    optimizer_cfg["missing_pair_penalty_deg"] = float(
        optimizer_cfg.get("missing_pair_penalty_deg", 5.0)
    )
    optimizer_cfg["q_group_line_constraints"] = bool(
        optimizer_cfg.get("q_group_line_constraints", True)
    )
    optimizer_cfg["q_group_line_angle_weight"] = float(
        optimizer_cfg.get("q_group_line_angle_weight", 0.6)
    )
    optimizer_cfg["q_group_line_offset_weight"] = float(
        optimizer_cfg.get("q_group_line_offset_weight", 1.0)
    )
    optimizer_cfg["q_group_line_missing_penalty_scale"] = float(
        optimizer_cfg.get("q_group_line_missing_penalty_scale", 0.35)
    )
    optimizer_cfg["hk0_peak_priority_weight"] = float(
        optimizer_cfg.get("hk0_peak_priority_weight", 6.0)
    )
    manual_max_nfev = 30
    try:
        configured_max_nfev = int(optimizer_cfg.get("max_nfev", manual_max_nfev))
    except (TypeError, ValueError):
        configured_max_nfev = manual_max_nfev
    optimizer_cfg["max_nfev"] = max(1, min(configured_max_nfev, manual_max_nfev))

    if unsafe_runtime_enabled:
        optimizer_cfg["workers"] = optimizer_cfg.get("workers", "auto")
        optimizer_cfg["parallel_mode"] = str(optimizer_cfg.get("parallel_mode", "auto")).strip()
        optimizer_cfg["worker_numba_threads"] = optimizer_cfg.get(
            "worker_numba_threads",
            0,
        )
    else:
        optimizer_cfg["workers"] = 1
        optimizer_cfg["parallel_mode"] = "off"
        optimizer_cfg["worker_numba_threads"] = 0
    cfg["optimizer"] = optimizer_cfg
    cfg["solver"] = optimizer_cfg

    seed_search_cfg_raw = cfg.get("seed_search", {})
    seed_search_cfg = dict(seed_search_cfg_raw) if isinstance(seed_search_cfg_raw, Mapping) else {}
    # Manual point fits are interactive; keep the normalized-u solver on one
    # trusted seed instead of inheriting the heavier global multistart budget.
    seed_search_cfg["prescore_top_k"] = 1
    seed_search_cfg["n_global"] = 0
    seed_search_cfg["n_jitter"] = 0
    cfg["seed_search"] = seed_search_cfg
    cfg["use_numba"] = bool(cfg.get("use_numba", False))
    cfg["allow_unsafe_runtime"] = bool(unsafe_runtime_enabled)

    sampling_cfg_raw = cfg.get("sampling", {})
    sampling_cfg = dict(sampling_cfg_raw) if isinstance(sampling_cfg_raw, Mapping) else {}
    try:
        fit_sample_count = max(int(sampling_cfg.get("fit_sample_count", 8)), 1)
    except Exception:
        fit_sample_count = 8
    sampling_cfg["fit_sample_count"] = int(fit_sample_count)
    cfg["sampling"] = sampling_cfg

    discrete_modes_cfg_raw = cfg.get("discrete_modes", {})
    discrete_modes_cfg = (
        dict(discrete_modes_cfg_raw) if isinstance(discrete_modes_cfg_raw, Mapping) else {}
    )
    # Manual GUI fits already resolve detector orientation from the saved
    # point pairs before the solver runs. Repeating the full solver-side
    # rot/flip sweep multiplies the work by up to 16x and has been able to
    # stall or crash interactive runs on large datasets.
    discrete_modes_cfg["enabled"] = False
    cfg["discrete_modes"] = discrete_modes_cfg

    # Manual detector-pixel fits should react to real peak offsets, not to a
    # heavily downweighted residual that can remain small even when the
    # geometry is still hundreds of pixels off.
    optimizer_cfg["loss"] = "linear"
    optimizer_cfg["f_scale_px"] = 1.0
    optimizer_cfg["weighted_matching"] = False
    optimizer_cfg["use_measurement_uncertainty"] = False
    optimizer_cfg["anisotropic_measurement_uncertainty"] = False

    cfg.pop("full_beam_polish", None)
    cfg.pop("ridge_refinement", None)
    cfg.pop("image_refinement", None)

    identifiability_cfg_raw = cfg.get("identifiability", {})
    identifiability_cfg = (
        dict(identifiability_cfg_raw) if isinstance(identifiability_cfg_raw, Mapping) else {}
    )
    identifiability_cfg["enabled"] = False
    identifiability_cfg.pop("auto_freeze", None)
    identifiability_cfg.pop("selective_thaw", None)
    identifiability_cfg.pop("adaptive_regularization", None)
    cfg["identifiability"] = identifiability_cfg
    return cfg


def apply_manual_caked_point_geometry_fit_runtime_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Build the manual profile for caked fit-space point fitting."""

    cfg = apply_manual_point_geometry_fit_runtime_overrides(
        runtime_cfg,
        joint_background_mode=joint_background_mode,
    )
    optimizer_cfg_raw = cfg.get("optimizer", cfg.get("solver", {}))
    optimizer_cfg = dict(optimizer_cfg_raw) if isinstance(optimizer_cfg_raw, Mapping) else {}
    optimizer_cfg["manual_point_fit_mode"] = True
    optimizer_cfg["dynamic_point_geometry_fit"] = True
    cfg["optimizer"] = optimizer_cfg
    cfg["solver"] = optimizer_cfg
    cfg["projection_view_mode"] = "caked"
    return cfg


def apply_dynamic_point_geometry_fit_runtime_overrides(
    runtime_cfg: Mapping[str, object] | None,
    *,
    joint_background_mode: bool,
) -> dict[str, object]:
    """Build the richer runtime profile for source-bound dynamic point fitting."""

    del joint_background_mode

    cfg = copy.deepcopy(dict(runtime_cfg or {}))
    unsafe_runtime_enabled = bool(cfg.get("allow_unsafe_runtime", False))

    optimizer_cfg_raw = cfg.get("optimizer", cfg.get("solver", {}))
    optimizer_cfg = dict(optimizer_cfg_raw) if isinstance(optimizer_cfg_raw, Mapping) else {}
    optimizer_cfg.pop("manual_point_fit_mode", None)
    optimizer_cfg["dynamic_point_geometry_fit"] = True
    optimizer_cfg["workers"] = optimizer_cfg.get("workers", "auto")

    parallel_mode = str(optimizer_cfg.get("parallel_mode", "auto")).strip().lower()
    if parallel_mode in {"false", "none", "disabled"}:
        parallel_mode = "off"
    if parallel_mode not in {"auto", "off", "datasets", "restarts"}:
        parallel_mode = "auto"
    optimizer_cfg["parallel_mode"] = parallel_mode
    optimizer_cfg["worker_numba_threads"] = optimizer_cfg.get(
        "worker_numba_threads",
        0,
    )
    cfg["optimizer"] = optimizer_cfg
    cfg["solver"] = optimizer_cfg
    cfg["use_numba"] = bool(cfg.get("use_numba", False))
    cfg["allow_unsafe_runtime"] = bool(unsafe_runtime_enabled)
    return cfg


def _geometry_fit_cache_normalized_hkl(
    value: object,
) -> tuple[int, int, int] | None:
    """Return one normalized HKL triplet when the value is usable."""

    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        return int(value[0]), int(value[1]), int(value[2])
    except Exception:
        return None


def _geometry_fit_cache_jsonable(
    value: object,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> object:
    """Convert one cache metadata value into a stable JSON-safe shape."""

    if _seen is None:
        _seen = set()
    if int(_depth) > 20:
        return "<max_depth>"
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in _seen:
            return "<cycle>"
        _seen.add(marker)
        try:
            return [
                (
                    str(key),
                    _geometry_fit_cache_jsonable(item, _seen, int(_depth) + 1),
                )
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
        finally:
            _seen.discard(marker)
    if isinstance(value, (tuple, list)):
        marker = id(value)
        if marker in _seen:
            return "<cycle>"
        _seen.add(marker)
        try:
            return [_geometry_fit_cache_jsonable(item, _seen, int(_depth) + 1) for item in value]
        finally:
            _seen.discard(marker)
    if isinstance(value, np.ndarray):
        marker = id(value)
        if marker in _seen:
            return "<cycle>"
        _seen.add(marker)
        try:
            return [
                _geometry_fit_cache_jsonable(item, _seen, int(_depth) + 1)
                for item in value.tolist()
            ]
        finally:
            _seen.discard(marker)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _geometry_fit_cache_canonical_tuple(value: object) -> object:
    canonical = _geometry_fit_cache_jsonable(value)
    if isinstance(canonical, list):
        return tuple(_geometry_fit_cache_canonical_tuple(item) for item in canonical)
    if isinstance(canonical, tuple):
        return tuple(_geometry_fit_cache_canonical_tuple(item) for item in canonical)
    return canonical


def _geometry_fit_cache_finite_float(
    value: object,
) -> float | None:
    """Return one finite float cache metadata value when possible."""

    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


def build_geometry_fit_dataset_cache_metadata(
    *,
    background_index: int,
    current_background_index: int,
    simulated_peaks: Sequence[object] | None,
    source_snapshot_diagnostics: Mapping[str, object] | None = None,
    source_resolution_diagnostics: Sequence[object] | None,
    pair_count: int,
    resolved_source_pair_count: int,
) -> dict[str, object]:
    """Build structured cache metadata for one geometry-fit dataset payload."""

    raw_by_table: dict[int, list[dict[str, object]]] = defaultdict(list)
    diag_by_table: dict[int, list[dict[str, object]]] = defaultdict(list)
    raw_by_row: dict[tuple[int, int], dict[str, object]] = {}
    raw_by_peak: dict[tuple[int, int], dict[str, object]] = {}

    for raw_entry in simulated_peaks or ():
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            table_idx = int(
                raw_entry.get("source_reflection_index", raw_entry.get("source_table_index"))
            )
        except Exception:
            continue
        entry = dict(raw_entry)
        raw_by_table[table_idx].append(entry)
        try:
            row_idx = int(entry.get("source_row_index"))
        except Exception:
            row_idx = -1
        if row_idx >= 0:
            raw_by_row[(int(table_idx), int(row_idx))] = entry
        try:
            peak_idx = int(entry.get("source_peak_index"))
        except Exception:
            peak_idx = -1
        if peak_idx >= 0:
            raw_by_peak[(int(table_idx), int(peak_idx))] = entry

    for raw_diag in source_resolution_diagnostics or ():
        if not isinstance(raw_diag, Mapping):
            continue
        diag = dict(raw_diag)
        table_idx = None
        for key_name in (
            "fit_source_row_key",
            "fit_source_peak_key",
            "overlay_source_row_key",
            "overlay_source_peak_key",
            "saved_source_row_key",
            "saved_source_peak_key",
        ):
            key_value = diag.get(key_name)
            if isinstance(key_value, (list, tuple, np.ndarray)) and len(key_value) >= 2:
                try:
                    table_idx = int(key_value[0])
                    break
                except Exception:
                    continue
        if table_idx is None:
            raw_table_idx = diag.get("saved_source_table_index")
            try:
                table_idx = int(raw_table_idx)
            except Exception:
                continue
        diag_by_table[int(table_idx)].append(diag)

    table_summaries: list[dict[str, object]] = []
    for table_idx in sorted(set(raw_by_table) | set(diag_by_table)):
        raw_entries = list(raw_by_table.get(table_idx, ()))
        table_diags = list(diag_by_table.get(table_idx, ()))
        reference_entry: dict[str, object] | None = None
        representative_row_indices: set[int] = set()
        kept_keys: set[tuple[str, int]] = set()
        nominal_hkl_recovery_count = 0

        for diag in table_diags:
            fit_kind = str(diag.get("fit_resolution_kind", "") or "")
            if fit_kind in {"legacy_dense_hkl_rebind", "hkl_fallback"}:
                nominal_hkl_recovery_count += 1
            if not bool(diag.get("fit_resolved", False)):
                continue
            row_key_added = False
            fit_row_key = diag.get("fit_source_row_key")
            if isinstance(fit_row_key, (list, tuple, np.ndarray)) and len(fit_row_key) >= 2:
                try:
                    row_idx = int(fit_row_key[1])
                except Exception:
                    row_idx = -1
                if row_idx >= 0:
                    representative_row_indices.add(int(row_idx))
                    kept_keys.add(("row", int(row_idx)))
                    row_key_added = True
                    reference_entry = (
                        raw_by_row.get((int(table_idx), int(row_idx))) or reference_entry
                    )
            fit_peak_key = diag.get("fit_source_peak_key")
            if isinstance(fit_peak_key, (list, tuple, np.ndarray)) and len(fit_peak_key) >= 2:
                try:
                    peak_idx = int(fit_peak_key[1])
                except Exception:
                    peak_idx = -1
                if peak_idx >= 0 and not row_key_added:
                    kept_keys.add(("peak", int(peak_idx)))
                    if reference_entry is None:
                        reference_entry = raw_by_peak.get((int(table_idx), int(peak_idx)))
            if reference_entry is None:
                overlay_row_key = diag.get("overlay_source_row_key")
                if (
                    isinstance(overlay_row_key, (list, tuple, np.ndarray))
                    and len(overlay_row_key) >= 2
                ):
                    try:
                        overlay_row_idx = int(overlay_row_key[1])
                    except Exception:
                        overlay_row_idx = -1
                    if overlay_row_idx >= 0:
                        reference_entry = raw_by_row.get((int(table_idx), int(overlay_row_idx)))
            if reference_entry is None:
                overlay_peak_key = diag.get("overlay_source_peak_key")
                if (
                    isinstance(overlay_peak_key, (list, tuple, np.ndarray))
                    and len(overlay_peak_key) >= 2
                ):
                    try:
                        overlay_peak_idx = int(overlay_peak_key[1])
                    except Exception:
                        overlay_peak_idx = -1
                    if overlay_peak_idx >= 0:
                        reference_entry = raw_by_peak.get((int(table_idx), int(overlay_peak_idx)))

        if reference_entry is None and raw_entries:
            reference_entry = dict(raw_entries[0])

        dropped_nonfinite = 0
        for entry in raw_entries:
            sim_col = _geometry_fit_cache_finite_float(entry.get("sim_col"))
            sim_row = _geometry_fit_cache_finite_float(entry.get("sim_row"))
            if sim_col is None or sim_row is None:
                dropped_nonfinite += 1
        row_count_before = int(len(raw_entries))
        row_count_after = int(len(kept_keys))
        merged_group_count = max(
            0,
            int(row_count_before) - int(row_count_after) - int(dropped_nonfinite),
        )
        reference = dict(reference_entry or {})
        nominal_hkl = _geometry_fit_cache_normalized_hkl(reference.get("hkl"))
        table_summaries.append(
            {
                "source_table_index": int(table_idx),
                "nominal_hkl": (list(nominal_hkl) if isinstance(nominal_hkl, tuple) else None),
                "q_group_key": _geometry_fit_cache_jsonable(reference.get("q_group_key")),
                "qr": _geometry_fit_cache_finite_float(reference.get("qr")),
                "qz": _geometry_fit_cache_finite_float(reference.get("qz")),
                "row_count_before_grouping": int(row_count_before),
                "row_count_after_grouping": int(row_count_after),
                "dropped_nonfinite_row_count": int(dropped_nonfinite),
                "nominal_hkl_recovery_count": int(nominal_hkl_recovery_count),
                "merged_group_count": int(merged_group_count),
                "representative_row_indices_kept": [
                    int(idx) for idx in sorted(representative_row_indices)
                ],
            }
        )

    snapshot_diag = (
        dict(source_snapshot_diagnostics)
        if isinstance(source_snapshot_diagnostics, Mapping)
        else {}
    )
    if snapshot_diag:
        snapshot_status = str(
            snapshot_diag.get(
                "status",
                "snapshot_hit" if simulated_peaks else "snapshot_empty",
            )
        )
        simulated_peak_total = int(sum(len(entries) for entries in raw_by_table.values()))
        snapshot_filter_reason = str(
            snapshot_diag.get(
                "snapshot_filter_reason",
                snapshot_diag.get(
                    "projection_failure_reason",
                    snapshot_diag.get("reason", "empty_returned_rows"),
                ),
            )
            or "empty_returned_rows"
        )
        if snapshot_status == "snapshot_hit" and simulated_peak_total <= 0:
            snapshot_status = "snapshot_hit_empty_returned_rows"
            snapshot_diag["status"] = snapshot_status
            snapshot_diag["stale_reason"] = (
                "snapshot_hit returned zero simulated rows "
                f"(raw_peak_count={int(snapshot_diag.get('raw_peak_count', 0) or 0)}, "
                f"projected_peak_count={int(snapshot_diag.get('projected_peak_count', 0) or 0)}, "
                f"filter_reason={snapshot_filter_reason})"
            )
        snapshot_hit = snapshot_status == "snapshot_hit"
        snapshot_cache_source = str(
            snapshot_diag.get(
                "cache_source",
                snapshot_diag.get("source", "source_snapshot"),
            )
            or "source_snapshot"
        )
        snapshot_rebuild_source = str(
            snapshot_diag.get("rebuild_source", snapshot_diag.get("created_from", "")) or ""
        )
        stale_reason = (
            None
            if snapshot_hit
            else str(
                snapshot_diag.get("stale_reason")
                or f"source snapshot status={snapshot_status}; filter_reason={snapshot_filter_reason}"
            )
        )
        cache_provenance = [
            f"source_snapshot:{snapshot_status}",
            *([f"rebuild_source:{snapshot_rebuild_source}"] if snapshot_rebuild_source else []),
            "build_geometry_manual_fit_dataset",
        ]
    else:
        snapshot_status = "rebuilt"
        snapshot_hit = False
        snapshot_cache_source = "geometry_manual_simulated_peaks_for_params(prefer_cache=False)"
        stale_reason = (
            "geometry-fit dataset prep rebuilds from fresh simulation rows (prefer_cache=False)."
        )
        cache_provenance = [
            "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
            "build_geometry_manual_fit_dataset",
        ]

    return {
        "cache_action": ("reused" if snapshot_hit else "rebuilt"),
        "reused": bool(snapshot_hit),
        "rebuilt": bool(not snapshot_hit),
        "stale_reason": stale_reason,
        "cache_source": snapshot_cache_source,
        "cache_provenance": cache_provenance,
        "background_index": int(background_index),
        "current_background_index": int(current_background_index),
        "prefer_cache": False,
        "pair_count": int(pair_count),
        "resolved_source_pair_count": int(resolved_source_pair_count),
        "source_snapshot_status": snapshot_status,
        "source_snapshot_created_from": snapshot_diag.get("created_from"),
        "source_snapshot_signature_match": bool(snapshot_diag.get("signature_match", snapshot_hit)),
        "source_snapshot_consumer": snapshot_diag.get("consumer"),
        "source_snapshot_cache_family": snapshot_diag.get("cache_family"),
        "source_snapshot_action": snapshot_diag.get("action"),
        "source_snapshot_filter_reason": snapshot_diag.get(
            "snapshot_filter_reason",
            snapshot_diag.get("projection_failure_reason", snapshot_diag.get("reason")),
        ),
        "source_snapshot_requested_signature_summary": snapshot_diag.get(
            "requested_signature_summary",
        ),
        "source_snapshot_stored_signature_summary": snapshot_diag.get(
            "stored_signature_summary",
        ),
        "source_snapshot_raw_peak_count": int(
            snapshot_diag.get(
                "raw_peak_count",
                sum(len(entries) for entries in raw_by_table.values()),
            )
            or 0
        ),
        "source_snapshot_row_count": int(
            snapshot_diag.get(
                "projected_peak_count",
                sum(len(entries) for entries in raw_by_table.values()),
            )
            or 0
        ),
        "simulated_peak_count": int(sum(len(entries) for entries in raw_by_table.values())),
        "table_count": int(len(table_summaries)),
        "table_summaries": table_summaries,
    }


def build_geometry_fit_dataset_cache_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[str]:
    """Format geometry-fit dataset cache metadata for one debug log section."""

    lines: list[str] = []
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        cache_metadata = raw_dataset.get("cache_metadata")
        if not isinstance(cache_metadata, Mapping):
            continue
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        lines.append(
            (
                "{label}: cache_action={action} reused={reused} rebuilt={rebuilt} "
                "stale_reason={stale} source={source}"
            ).format(
                label=label,
                action=str(cache_metadata.get("cache_action", "<unknown>")),
                reused=_geometry_fit_debug_value_text(cache_metadata.get("reused", False)),
                rebuilt=_geometry_fit_debug_value_text(cache_metadata.get("rebuilt", True)),
                stale=str(cache_metadata.get("stale_reason", "<none>") or "<none>"),
                source=str(cache_metadata.get("cache_source", "<unknown>")),
            )
        )
        provenance = cache_metadata.get("cache_provenance")
        if isinstance(provenance, Sequence) and not isinstance(
            provenance,
            (str, bytes, bytearray),
        ):
            lines.append(
                "{label}: provenance={provenance}".format(
                    label=label,
                    provenance=_geometry_fit_debug_value_text(
                        list(provenance),
                        float_digits=3,
                    ),
                )
            )
        lines.append(
            (
                "{label}: pair_count={pairs} resolved_source_pairs={resolved} "
                "simulated_peaks={peaks} tables={tables}"
            ).format(
                label=label,
                pairs=_geometry_fit_debug_value_text(
                    cache_metadata.get("pair_count", raw_dataset.get("pair_count", 0)),
                    float_digits=0,
                ),
                resolved=_geometry_fit_debug_value_text(
                    cache_metadata.get(
                        "resolved_source_pair_count",
                        raw_dataset.get("resolved_source_pair_count", 0),
                    ),
                    float_digits=0,
                ),
                peaks=_geometry_fit_debug_value_text(
                    cache_metadata.get(
                        "simulated_peak_count",
                        raw_dataset.get("simulated_peak_count", 0),
                    ),
                    float_digits=0,
                ),
                tables=_geometry_fit_debug_value_text(
                    cache_metadata.get("table_count", 0),
                    float_digits=0,
                ),
            )
        )
        has_snapshot_metadata = any(
            key in cache_metadata
            for key in (
                "source_snapshot_status",
                "source_snapshot_action",
                "source_snapshot_cache_family",
                "source_snapshot_consumer",
                "source_snapshot_created_from",
                "source_snapshot_signature_match",
                "source_snapshot_row_count",
                "source_snapshot_raw_peak_count",
                "source_snapshot_requested_signature_summary",
                "source_snapshot_stored_signature_summary",
            )
        )
        if has_snapshot_metadata:
            lines.append(
                (
                    "{label}: source_snapshot family={family} action={action} "
                    "status={status} consumer={consumer} created_from={created_from} "
                    "signature_match={match} rows={rows} raw_peaks={raw}"
                ).format(
                    label=label,
                    family=str(cache_metadata.get("source_snapshot_cache_family", "<unknown>")),
                    action=str(cache_metadata.get("source_snapshot_action", "<unknown>")),
                    status=str(cache_metadata.get("source_snapshot_status", "<unknown>")),
                    consumer=str(cache_metadata.get("source_snapshot_consumer", "<unknown>")),
                    created_from=str(
                        cache_metadata.get("source_snapshot_created_from", "<unknown>")
                    ),
                    match=_geometry_fit_debug_value_text(
                        cache_metadata.get("source_snapshot_signature_match", False)
                    ),
                    rows=_geometry_fit_debug_value_text(
                        cache_metadata.get("source_snapshot_row_count", 0),
                        float_digits=0,
                    ),
                    raw=_geometry_fit_debug_value_text(
                        cache_metadata.get("source_snapshot_raw_peak_count", 0),
                        float_digits=0,
                    ),
                )
            )
            lines.append(
                (
                    "{label}: source_snapshot requested_signature={requested} "
                    "stored_signature={stored}"
                ).format(
                    label=label,
                    requested=_geometry_fit_debug_value_text(
                        cache_metadata.get("source_snapshot_requested_signature_summary")
                    ),
                    stored=_geometry_fit_debug_value_text(
                        cache_metadata.get("source_snapshot_stored_signature_summary")
                    ),
                )
            )
        table_summaries = cache_metadata.get("table_summaries")
        if not isinstance(table_summaries, Sequence) or isinstance(
            table_summaries,
            (str, bytes, bytearray),
        ):
            continue
        for raw_summary in table_summaries:
            if not isinstance(raw_summary, Mapping):
                continue
            table_index = raw_summary.get(
                "source_table_index",
                raw_summary.get("table_index", 0),
            )
            lines.append(
                (
                    "{label}: table[{table}] nominal_hkl={hkl} q_group_key={q_group} "
                    "qr={qr} qz={qz} rows_before={before} rows_after={after} "
                    "dropped_nonfinite={dropped} nominal_hkl_recovery={recovery} "
                    "merged_groups={merged} representative_rows={rows}"
                ).format(
                    label=label,
                    table=_geometry_fit_debug_value_text(table_index, float_digits=0),
                    hkl=_geometry_fit_debug_value_text(
                        raw_summary.get("nominal_hkl"),
                        float_digits=3,
                    ),
                    q_group=_geometry_fit_debug_value_text(
                        raw_summary.get("q_group_key"),
                        float_digits=3,
                    ),
                    qr=_geometry_fit_debug_value_text(
                        raw_summary.get("qr"),
                        float_digits=6,
                    ),
                    qz=_geometry_fit_debug_value_text(
                        raw_summary.get("qz"),
                        float_digits=6,
                    ),
                    before=_geometry_fit_debug_value_text(
                        raw_summary.get("row_count_before_grouping", 0),
                        float_digits=0,
                    ),
                    after=_geometry_fit_debug_value_text(
                        raw_summary.get("row_count_after_grouping", 0),
                        float_digits=0,
                    ),
                    dropped=_geometry_fit_debug_value_text(
                        raw_summary.get("dropped_nonfinite_row_count", 0),
                        float_digits=0,
                    ),
                    recovery=_geometry_fit_debug_value_text(
                        raw_summary.get("nominal_hkl_recovery_count", 0),
                        float_digits=0,
                    ),
                    merged=_geometry_fit_debug_value_text(
                        raw_summary.get("merged_group_count", 0),
                        float_digits=0,
                    ),
                    rows=_geometry_fit_debug_value_text(
                        raw_summary.get("representative_row_indices_kept"),
                        float_digits=0,
                    ),
                )
            )
    return lines


def build_geometry_fit_live_cache_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[str]:
    """Format live-cache inventory and source-snapshot behavior for debug logs."""

    lines: list[str] = []
    inventory: Mapping[str, object] | None = None
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        diagnostics_raw = raw_dataset.get("simulation_diagnostics")
        if not isinstance(diagnostics_raw, Mapping):
            continue
        diagnostics = dict(diagnostics_raw)
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        lines.append(
            (
                "{label}: source_snapshot status={status} consumer={consumer} "
                "created_from={created_from} signature_match={match} "
                "rows={rows} raw_peaks={raw}"
            ).format(
                label=label,
                status=str(diagnostics.get("status", "<unknown>")),
                consumer=str(diagnostics.get("consumer", "<unknown>")),
                created_from=str(diagnostics.get("created_from", "<unknown>")),
                match=_geometry_fit_debug_value_text(diagnostics.get("signature_match", False)),
                rows=_geometry_fit_debug_value_text(
                    diagnostics.get(
                        "projected_peak_count",
                        raw_dataset.get("simulated_peak_count", 0),
                    ),
                    float_digits=0,
                ),
                raw=_geometry_fit_debug_value_text(
                    diagnostics.get("raw_peak_count", 0),
                    float_digits=0,
                ),
            )
        )
        requested_signature = diagnostics.get("requested_signature_summary")
        stored_signature = diagnostics.get("stored_signature_summary")
        if requested_signature is not None or stored_signature is not None:
            lines.append(
                (
                    "{label}: source_snapshot requested_signature={requested} "
                    "stored_signature={stored}"
                ).format(
                    label=label,
                    requested=_geometry_fit_debug_value_text(requested_signature),
                    stored=_geometry_fit_debug_value_text(stored_signature),
                )
            )
        inventory_raw = diagnostics.get("live_cache_inventory")
        if inventory is None and isinstance(inventory_raw, Mapping):
            inventory = dict(inventory_raw)
    if not isinstance(inventory, Mapping):
        return lines
    lines.append(
        (
            "inventory: preview_active={preview_active} preview_sample_count={preview_samples} "
            "stored_hit_table_signature_present={stored_present} "
            "stored_hit_table_signature={stored_sig}"
        ).format(
            preview_active=_geometry_fit_debug_value_text(inventory.get("preview_active", False)),
            preview_samples=_geometry_fit_debug_value_text(
                inventory.get("preview_sample_count"),
                float_digits=0,
            ),
            stored_present=_geometry_fit_debug_value_text(
                inventory.get("stored_hit_table_signature_present", False)
            ),
            stored_sig=_geometry_fit_debug_value_text(
                inventory.get("stored_hit_table_signature_summary")
            ),
        )
    )
    lines.append(
        (
            "inventory: last_simulation_signature={last_sig} "
            "primary_contribution_signature={primary_sig} primary_source_mode={source_mode} "
            "primary_active_keys={active_keys} primary_hit_table_entries={entries}"
        ).format(
            last_sig=_geometry_fit_debug_value_text(
                inventory.get("last_simulation_signature_summary")
            ),
            primary_sig=_geometry_fit_debug_value_text(
                inventory.get("primary_contribution_cache_signature_summary")
            ),
            source_mode=_geometry_fit_debug_value_text(inventory.get("primary_source_mode")),
            active_keys=_geometry_fit_debug_value_text(
                inventory.get("primary_active_contribution_key_count", 0),
                float_digits=0,
            ),
            entries=_geometry_fit_debug_value_text(
                inventory.get("primary_hit_table_cache_entry_count", 0),
                float_digits=0,
            ),
        )
    )
    source_snapshots_raw = inventory.get("source_snapshots")
    if isinstance(source_snapshots_raw, Sequence) and not isinstance(
        source_snapshots_raw,
        (str, bytes, bytearray),
    ):
        source_snapshots = [
            dict(item) for item in source_snapshots_raw if isinstance(item, Mapping)
        ]
    else:
        source_snapshots = []
    if not source_snapshots:
        lines.append("inventory: source_snapshots=<none>")
    else:
        lines.append(
            "inventory: source_snapshots={count}".format(
                count=_geometry_fit_debug_value_text(
                    len(source_snapshots),
                    float_digits=0,
                )
            )
        )
    return lines


def build_geometry_fit_dataset_cache_payload(
    prepared_run: GeometryFitPreparedRun,
    *,
    current_background_index: int | None = None,
) -> dict[str, object]:
    """Copy the reusable dataset bundle from a successful geometry fit."""

    dataset_index = current_background_index
    if dataset_index is None:
        current_dataset = (
            prepared_run.current_dataset
            if isinstance(prepared_run.current_dataset, Mapping)
            else {}
        )
        try:
            dataset_index = int(current_dataset.get("dataset_index", 0))
        except Exception:
            dataset_index = 0

    dataset_cache_metadata: list[dict[str, object]] = []
    for info in prepared_run.dataset_infos or []:
        if not isinstance(info, Mapping):
            continue
        raw_metadata = info.get("cache_metadata")
        if not isinstance(raw_metadata, Mapping):
            continue
        copied_metadata = copy_geometry_fit_state_value(dict(raw_metadata))
        if isinstance(copied_metadata, Mapping):
            copied_metadata = dict(copied_metadata)
            copied_metadata.setdefault(
                "dataset_index",
                int(info.get("dataset_index", copied_metadata.get("dataset_index", 0)) or 0),
            )
            label = str(info.get("label", "") or "").strip()
            if label:
                copied_metadata.setdefault("label", label)
            dataset_cache_metadata.append(copied_metadata)

    return {
        "selected_background_indices": [
            int(idx) for idx in prepared_run.selected_background_indices
        ],
        "current_background_index": int(dataset_index),
        "joint_background_mode": bool(prepared_run.joint_background_mode),
        "background_theta_values": [float(value) for value in prepared_run.background_theta_values],
        "dataset_specs": [
            _copy_geometry_fit_dataset_spec_for_state(spec)
            for spec in (prepared_run.dataset_specs or [])
            if isinstance(spec, Mapping)
        ],
        "cache_metadata": {
            "cache_action": "rebuilt",
            "reused": False,
            "rebuilt": True,
            "stale_reason": None,
            "cache_source": "build_geometry_fit_dataset_cache_payload",
            "cache_provenance": [
                "build_geometry_manual_fit_dataset",
                "build_geometry_fit_dataset_cache_payload",
            ],
            "dataset_count": int(len(prepared_run.dataset_infos or [])),
            "dataset_cache_metadata": dataset_cache_metadata,
        },
    }


def geometry_fit_dataset_cache_stale_reason(
    cache_payload: Mapping[str, object] | None,
    *,
    selected_background_indices: Sequence[object],
    current_background_index: int,
    joint_background_mode: bool,
    background_theta_values: Sequence[object],
) -> str | None:
    """Return a human-readable stale-cache reason, or ``None`` when valid."""

    if not isinstance(cache_payload, Mapping):
        return "Run geometry fit first."

    raw_specs = cache_payload.get("dataset_specs")
    if not isinstance(raw_specs, Sequence) or len(raw_specs) <= 0:
        return "Run geometry fit first."

    try:
        cached_selected = [int(idx) for idx in cache_payload.get("selected_background_indices", [])]
    except Exception:
        return "Run geometry fit first."
    current_selected = [int(idx) for idx in selected_background_indices]
    if cached_selected != current_selected:
        return "Geometry-fit background selection changed. Rerun geometry fit."

    try:
        cached_index = int(cache_payload.get("current_background_index", -1))
    except Exception:
        return "Run geometry fit first."
    if int(cached_index) != int(current_background_index):
        return "Active background changed since geometry fit. Rerun geometry fit."

    cached_joint = bool(cache_payload.get("joint_background_mode", False))
    if cached_joint != bool(joint_background_mode):
        return "Shared-theta mode changed since geometry fit. Rerun geometry fit."

    try:
        cached_theta_values = [
            float(value) for value in cache_payload.get("background_theta_values", [])
        ]
        current_theta_values = [float(value) for value in background_theta_values]
    except Exception:
        return "Background theta values are unavailable. Rerun geometry fit."
    if len(cached_theta_values) != len(current_theta_values):
        return "Background theta values changed since geometry fit. Rerun geometry fit."
    for cached_value, current_value in zip(
        cached_theta_values,
        current_theta_values,
    ):
        if not np.isfinite(cached_value) or not np.isfinite(current_value):
            return "Background theta values changed since geometry fit. Rerun geometry fit."
        if not np.isclose(
            float(cached_value),
            float(current_value),
            rtol=0.0,
            atol=1.0e-9,
        ):
            return "Background theta values changed since geometry fit. Rerun geometry fit."

    return None


def build_geometry_fit_start_cmd_line(
    *,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
) -> str:
    """Build the console start line for one geometry-fit run."""

    dataset_list = list(dataset_infos or ())
    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    return (
        "start: "
        f"vars={','.join(str(name) for name in var_names)} "
        f"datasets={len(dataset_list)} "
        f"current_groups={int(dataset.get('group_count', 0) or 0)} "
        f"current_points={int(dataset.get('pair_count', 0) or 0)}"
    )


def _build_geometry_fit_run_request_lines(
    *,
    selected_background_indices: Sequence[object],
    joint_background_mode: bool,
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
) -> list[str]:
    """Summarize the prepared runtime request for one geometry-fit run."""

    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    lines = [
        f"joint_background_mode={bool(joint_background_mode)}",
        "selected_background_indices=[{indices}]".format(
            indices=", ".join(str(int(idx)) for idx in selected_background_indices)
        ),
        f"dataset_count={int(len(list(dataset_infos or ())))}",
        f"current_dataset_index={int(dataset.get('dataset_index', 0) or 0)}",
        f"current_groups={int(dataset.get('group_count', 0) or 0)}",
        f"current_points={int(dataset.get('pair_count', 0) or 0)}",
    ]
    label = str(dataset.get("label", "") or "").strip()
    if label:
        lines.append(f"current_label={label}")
    if "resolved_source_pair_count" in dataset:
        lines.append(
            "resolved_source_pairs={count}".format(
                count=int(dataset.get("resolved_source_pair_count", 0) or 0)
            )
        )
    if "theta_base" in dataset:
        lines.append(
            "theta_base={theta}".format(
                theta=_geometry_fit_debug_value_text(
                    dataset.get("theta_base", np.nan),
                )
            )
        )
    if "theta_effective" in dataset:
        lines.append(
            "theta_effective={theta}".format(
                theta=_geometry_fit_debug_value_text(
                    dataset.get("theta_effective", np.nan),
                )
            )
        )
    return lines


def _geometry_fit_flag_enabled(value: object) -> bool:
    """Return whether one user-facing debug flag value is enabled."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def geometry_fit_debug_logging_enabled(
    geometry_runtime_cfg: Mapping[str, object] | None,
) -> bool:
    """Return whether extra geometry-fit logging is enabled for this run."""

    if isinstance(geometry_runtime_cfg, Mapping) and not geometry_runtime_cfg:
        return False
    return geometry_fit_extra_sections_enabled(
        geometry_runtime_cfg=geometry_runtime_cfg,
    )


def _build_geometry_fit_runtime_config_lines(
    geometry_runtime_cfg: Mapping[str, object] | None,
) -> list[str]:
    """Summarize the active runtime/solver configuration for one fit run."""

    cfg = geometry_runtime_cfg if isinstance(geometry_runtime_cfg, Mapping) else {}
    optimizer_cfg_raw = cfg.get("optimizer", cfg.get("solver", {}))
    optimizer_cfg = optimizer_cfg_raw if isinstance(optimizer_cfg_raw, Mapping) else {}
    solver_cfg_raw = cfg.get("solver", optimizer_cfg)
    solver_cfg = solver_cfg_raw if isinstance(solver_cfg_raw, Mapping) else {}
    discrete_cfg_raw = cfg.get("discrete_modes", {})
    discrete_cfg = discrete_cfg_raw if isinstance(discrete_cfg_raw, Mapping) else {}
    identifiability_cfg_raw = cfg.get("identifiability", {})
    identifiability_cfg = (
        identifiability_cfg_raw if isinstance(identifiability_cfg_raw, Mapping) else {}
    )
    if not cfg and not optimizer_cfg and not solver_cfg:
        return []
    lines = [
        "use_numba={use_numba} allow_unsafe_runtime={unsafe}".format(
            use_numba=_geometry_fit_debug_value_text(cfg.get("use_numba", "<default>")),
            unsafe=_geometry_fit_debug_value_text(cfg.get("allow_unsafe_runtime", False)),
        ),
    ]
    fit_sample_count = geometry_fit_runtime_fit_sample_count(cfg)
    if fit_sample_count is not None:
        lines.append(f"sampling fit_sample_count={int(fit_sample_count)}")
    lines.extend(
        [
            (
                "optimizer loss={loss} f_scale_px={f_scale} manual_point_fit_mode={manual} "
                "weighted_matching={weighted} q_group_line_constraints={line_constraints}"
            ).format(
                loss=str(optimizer_cfg.get("loss", "<default>")),
                f_scale=_geometry_fit_debug_value_text(optimizer_cfg.get("f_scale_px", np.nan)),
                manual=_geometry_fit_debug_value_text(
                    optimizer_cfg.get("manual_point_fit_mode", False)
                ),
                weighted=_geometry_fit_debug_value_text(
                    optimizer_cfg.get("weighted_matching", "<default>")
                ),
                line_constraints=_geometry_fit_debug_value_text(
                    optimizer_cfg.get("q_group_line_constraints", "<default>")
                ),
            ),
            (
                "solver parallel_mode={mode} workers={workers} "
                "worker_numba_threads={threads} discrete_modes_enabled={discrete} "
                "identifiability_enabled={ident}"
            ).format(
                mode=str(solver_cfg.get("parallel_mode", "<default>")),
                workers=_geometry_fit_debug_value_text(
                    solver_cfg.get("workers", "<default>"),
                    float_digits=0,
                ),
                threads=_geometry_fit_debug_value_text(
                    solver_cfg.get("worker_numba_threads", "<default>"),
                    float_digits=0,
                ),
                discrete=_geometry_fit_debug_value_text(discrete_cfg.get("enabled", "<default>")),
                ident=_geometry_fit_debug_value_text(
                    identifiability_cfg.get("enabled", "<default>")
                ),
            ),
        ]
    )
    return lines


def build_geometry_fit_start_log_sections(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
    selected_background_indices: Sequence[object],
    joint_background_mode: bool,
    geometry_runtime_cfg: Mapping[str, object] | None,
) -> list[tuple[str, list[str]]]:
    """Build the start-log sections for one geometry-fit run."""

    fit_params = params if isinstance(params, Mapping) else {}
    dataset = current_dataset if isinstance(current_dataset, Mapping) else {}
    debug_logging = geometry_fit_debug_logging_enabled(geometry_runtime_cfg)
    orientation_diag = dataset.get("orientation_diag") or {}
    if not isinstance(orientation_diag, Mapping):
        orientation_diag = {}
    orientation_choice = dataset.get("orientation_choice") or {}
    if not isinstance(orientation_choice, Mapping):
        orientation_choice = {}
    dataset_lines = [
        str(info.get("summary_line", ""))
        for info in (dataset_infos or ())
        if isinstance(info, Mapping)
    ] or ["<none>"]

    sections: list[tuple[str, list[str]]] = [
        (
            "Fitting variables (start values):",
            [f"{name}={float(fit_params.get(str(name), np.nan)):.6f}" for name in var_names],
        ),
        (
            "Manual geometry datasets:",
            dataset_lines,
        ),
        (
            "Current orientation diagnostics:",
            [
                f"pairs={orientation_diag.get('pairs', 0)}",
                f"chosen={orientation_choice.get('label', 'identity')}",
                f"identity_rms_px={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}",
                f"best_rms_px={float(orientation_diag.get('best_rms_px', np.nan)):.4f}",
                f"reason={orientation_diag.get('reason', 'n/a')}",
            ],
        ),
    ]

    if debug_logging:
        dataset_cache_lines = build_geometry_fit_dataset_cache_log_lines(dataset_infos)
        if dataset_cache_lines:
            sections.append(
                (
                    "Geometry-fit dataset cache diagnostics:",
                    dataset_cache_lines,
                )
            )
        live_cache_lines = build_geometry_fit_live_cache_log_lines(dataset_infos)
        if live_cache_lines:
            sections.append(("Live simulation cache:", live_cache_lines))
        simulation_diagnostic_lines = build_geometry_fit_simulation_diagnostic_log_lines(
            dataset_infos
        )
        if simulation_diagnostic_lines:
            sections.append(("Fresh simulation diagnostics:", simulation_diagnostic_lines))
        source_resolution_lines = build_geometry_fit_source_resolution_log_lines(dataset_infos)
        if source_resolution_lines:
            sections.append(("Cached source-row diagnostics:", source_resolution_lines))

    return sections


def _geometry_fit_source_resolution_key_text(value: object) -> str:
    """Format one cached-source identity tuple for log output."""

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        try:
            return f"({int(value[0])}, {int(value[1])})"
        except Exception:
            return _geometry_fit_debug_value_text(value, float_digits=3)
    return "<none>"


def _geometry_fit_source_branch_text(
    branch_value: object,
    branch_source: object = None,
) -> str:
    """Format one branch label plus provenance for log output."""

    try:
        branch_idx = int(branch_value)
    except Exception:
        branch_idx = -1
    source_text = str(branch_source or "").strip()
    if branch_idx in {0, 1}:
        if source_text:
            return f"{branch_idx}({source_text})"
        return str(branch_idx)
    return "<none>"


def build_geometry_fit_source_resolution_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
    *,
    max_unresolved_pairs_per_dataset: int = 8,
) -> list[str]:
    """Format cached-source resolution diagnostics for log output."""

    lines: list[str] = []
    max_pairs = max(1, int(max_unresolved_pairs_per_dataset))
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        diagnostics_raw = raw_dataset.get("source_resolution_diagnostics")
        if not isinstance(diagnostics_raw, Sequence) or isinstance(
            diagnostics_raw,
            (str, bytes, bytearray),
        ):
            continue
        diagnostics = [dict(item) for item in diagnostics_raw if isinstance(item, Mapping)]
        if not diagnostics:
            continue
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        pair_count = int(raw_dataset.get("pair_count", len(diagnostics)) or 0)
        resolved_count = int(raw_dataset.get("resolved_source_pair_count", 0) or 0)
        simulated_peak_count = int(raw_dataset.get("simulated_peak_count", 0) or 0)
        simulated_lookup_count = int(raw_dataset.get("simulated_lookup_count", 0) or 0)
        lines.append(
            (
                "{label}: resolved_source_pairs={resolved}/{pairs} "
                "simulated_peaks={peaks} simulated_source_rows={rows}"
            ).format(
                label=label,
                resolved=resolved_count,
                pairs=pair_count,
                peaks=simulated_peak_count,
                rows=simulated_lookup_count,
            )
        )
        unresolved = [
            item
            for item in diagnostics
            if not bool(
                item.get(
                    "fit_resolved",
                    item.get("strict_resolved", False),
                )
            )
        ]
        if not unresolved:
            if any(
                bool(item.get("fit_resolved", False))
                and not bool(item.get("strict_resolved", False))
                for item in diagnostics
            ):
                lines.append(
                    f"{label}: all saved manual pairs resolved for fit "
                    "(including legacy source-id remaps)."
                )
            else:
                lines.append(f"{label}: all saved manual pairs strictly resolved.")
            for item in diagnostics[:max_pairs]:
                pair_index = int(item.get("pair_index", 0) or 0)
                lines.append(
                    (
                        "{label} pair#{pair}: saved_branch={saved_branch} "
                        "strict_branch={strict_branch} fit_branch={fit_branch} "
                        "fallback_branch={fallback_branch} fit_kind={fit_kind}"
                    ).format(
                        label=label,
                        pair=pair_index,
                        saved_branch=_geometry_fit_source_branch_text(
                            item.get("saved_source_branch_index"),
                            item.get("saved_source_branch_source"),
                        ),
                        strict_branch=_geometry_fit_source_branch_text(
                            item.get("strict_source_branch_index"),
                            item.get("strict_source_branch_source"),
                        ),
                        fit_branch=_geometry_fit_source_branch_text(
                            item.get("fit_source_branch_index"),
                            item.get("fit_source_branch_source"),
                        ),
                        fallback_branch=_geometry_fit_source_branch_text(
                            item.get("overlay_source_branch_index"),
                            item.get("overlay_source_branch_source"),
                        ),
                        fit_kind=str(item.get("fit_resolution_kind", "<none>") or "<none>"),
                    )
                )
            extra_count = len(diagnostics) - max_pairs
            if extra_count > 0:
                lines.append(f"{label}: ... {int(extra_count)} more resolved pair(s) not shown")
            continue
        for item in unresolved[:max_pairs]:
            pair_index = int(item.get("pair_index", 0) or 0)
            lines.append(
                (
                    "{label} pair#{pair}: saved_row={saved_row} saved_peak={saved_peak} "
                    "saved_hkl={saved_hkl} q_group={q_group} display={display}"
                ).format(
                    label=label,
                    pair=pair_index,
                    saved_row=_geometry_fit_source_resolution_key_text(
                        item.get("saved_source_row_key")
                    ),
                    saved_peak=_geometry_fit_source_resolution_key_text(
                        item.get("saved_source_peak_key")
                    ),
                    saved_hkl=_geometry_fit_debug_value_text(
                        item.get("saved_hkl"),
                        float_digits=3,
                    ),
                    q_group=_geometry_fit_debug_value_text(
                        item.get("saved_q_group_key"),
                        float_digits=3,
                    ),
                    display=_geometry_fit_debug_value_text(
                        item.get("saved_display_point"),
                        float_digits=3,
                    ),
                )
            )
            lines.append(
                (
                    "{label} pair#{pair}: strict_row={row_status} strict_peak={peak_status} "
                    "fallback={fallback} fallback_row={fallback_row} "
                    "fallback_peak={fallback_peak} fallback_hkl={fallback_hkl} "
                    "fallback_distance_px={distance}"
                ).format(
                    label=label,
                    pair=pair_index,
                    row_status=str(item.get("row_candidate_status", "<unknown>")),
                    peak_status=str(item.get("peak_candidate_status", "<unknown>")),
                    fallback=str(item.get("overlay_resolution_kind", "<none>") or "<none>"),
                    fallback_row=_geometry_fit_source_resolution_key_text(
                        item.get("overlay_source_row_key")
                    ),
                    fallback_peak=_geometry_fit_source_resolution_key_text(
                        item.get("overlay_source_peak_key")
                    ),
                    fallback_hkl=_geometry_fit_debug_value_text(
                        item.get("overlay_hkl"),
                        float_digits=3,
                    ),
                    distance=_geometry_fit_debug_value_text(
                        item.get("overlay_distance_px"),
                        float_digits=3,
                    ),
                )
            )
            reason = str(item.get("failure_reason", "") or "").strip()
            if reason:
                lines.append(f"{label} pair#{pair_index}: reason={reason}")
            lines.append(
                (
                    "{label} pair#{pair}: saved_branch={saved_branch} "
                    "strict_branch={strict_branch} fit_branch={fit_branch} "
                    "fallback_branch={fallback_branch} fit_kind={fit_kind}"
                ).format(
                    label=label,
                    pair=pair_index,
                    saved_branch=_geometry_fit_source_branch_text(
                        item.get("saved_source_branch_index"),
                        item.get("saved_source_branch_source"),
                    ),
                    strict_branch=_geometry_fit_source_branch_text(
                        item.get("strict_source_branch_index"),
                        item.get("strict_source_branch_source"),
                    ),
                    fit_branch=_geometry_fit_source_branch_text(
                        item.get("fit_source_branch_index"),
                        item.get("fit_source_branch_source"),
                    ),
                    fallback_branch=_geometry_fit_source_branch_text(
                        item.get("overlay_source_branch_index"),
                        item.get("overlay_source_branch_source"),
                    ),
                    fit_kind=str(item.get("fit_resolution_kind", "<none>") or "<none>"),
                )
            )
        extra_count = len(unresolved) - max_pairs
        if extra_count > 0:
            lines.append(f"{label}: ... {int(extra_count)} more unresolved pair(s) not shown")
    return lines


def _geometry_fit_simulation_diag_int(
    value: object,
) -> int | None:
    """Return one diagnostics count as an integer when possible."""

    try:
        return int(value)
    except Exception:
        return None


def _geometry_fit_short_count_list(
    values: object,
    *,
    limit: int = 8,
) -> str:
    """Return one compact count-list preview for diagnostics logs."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return "<none>"
    ints: list[str] = []
    for raw_value in values[: max(1, int(limit))]:
        count = _geometry_fit_simulation_diag_int(raw_value)
        ints.append(str(count) if count is not None else "?")
    suffix = ""
    try:
        extra_count = len(values) - max(1, int(limit))
    except Exception:
        extra_count = 0
    if extra_count > 0:
        suffix = f", ... +{int(extra_count)}"
    return "[" + ", ".join(ints) + suffix + "]"


def build_geometry_fit_simulation_diagnostic_log_lines(
    dataset_infos: Sequence[Mapping[str, object]] | None,
) -> list[str]:
    """Format source-snapshot diagnostics for geometry-fit preflight logs."""

    lines: list[str] = []
    for raw_dataset in dataset_infos or ():
        if not isinstance(raw_dataset, Mapping):
            continue
        diagnostics_raw = raw_dataset.get("simulation_diagnostics")
        if not isinstance(diagnostics_raw, Mapping):
            continue
        diagnostics = dict(diagnostics_raw)
        label = str(
            raw_dataset.get(
                "label",
                f"bg[{raw_dataset.get('dataset_index', '?')}]",
            )
        )
        runtime_diag_raw = diagnostics.get("runtime_simulation")
        runtime_diag = dict(runtime_diag_raw) if isinstance(runtime_diag_raw, Mapping) else {}
        lines.append(
            (
                "{label}: source={source} status={status} consumer={consumer} "
                "created_from={created_from} signature_match={signature_match} "
                "projected_rows={projected} raw_peaks={raw}"
            ).format(
                label=label,
                source=str(diagnostics.get("source", "<unknown>")),
                status=str(diagnostics.get("status", "<unknown>")),
                consumer=str(diagnostics.get("consumer", "<unknown>")),
                created_from=str(diagnostics.get("created_from", "<unknown>")),
                signature_match=_geometry_fit_debug_value_text(
                    diagnostics.get("signature_match", False)
                ),
                projected=_geometry_fit_debug_value_text(
                    diagnostics.get(
                        "projected_peak_count",
                        raw_dataset.get("simulated_peak_count", np.nan),
                    ),
                    float_digits=0,
                ),
                raw=_geometry_fit_debug_value_text(
                    diagnostics.get("raw_peak_count", np.nan),
                    float_digits=0,
                ),
            )
        )
        lines.append(
            (
                "{label}: inputs miller_shape={miller} intensity_shape={intensity} "
                "image_size={image_size}"
            ).format(
                label=label,
                miller=_geometry_fit_debug_value_text(
                    diagnostics.get("miller_shape"),
                    float_digits=0,
                ),
                intensity=_geometry_fit_debug_value_text(
                    diagnostics.get("intensity_shape"),
                    float_digits=0,
                ),
                image_size=_geometry_fit_debug_value_text(
                    diagnostics.get("image_size", np.nan),
                    float_digits=0,
                ),
            )
        )
        lines.append(
            ("{label}: missing_params={params} missing_mosaic={mosaic}").format(
                label=label,
                params=_geometry_fit_debug_value_text(
                    diagnostics.get("missing_param_keys", []),
                    float_digits=0,
                ),
                mosaic=_geometry_fit_debug_value_text(
                    diagnostics.get("missing_mosaic_keys", []),
                    float_digits=0,
                ),
            )
        )
        param_summary_raw = diagnostics.get("param_summary")
        if isinstance(param_summary_raw, Mapping) and param_summary_raw:
            lines.append(
                (
                    "{label}: params a={a} c={c} lambda={lam} theta_initial={theta} "
                    "center={center} n2={n2}"
                ).format(
                    label=label,
                    a=_geometry_fit_debug_value_text(param_summary_raw.get("a", np.nan)),
                    c=_geometry_fit_debug_value_text(param_summary_raw.get("c", np.nan)),
                    lam=_geometry_fit_debug_value_text(param_summary_raw.get("lambda", np.nan)),
                    theta=_geometry_fit_debug_value_text(
                        param_summary_raw.get("theta_initial", np.nan)
                    ),
                    center=_geometry_fit_debug_value_text(param_summary_raw.get("center", None)),
                    n2=_geometry_fit_debug_value_text(param_summary_raw.get("n2", None)),
                )
            )
        mosaic_sizes_raw = diagnostics.get("mosaic_array_sizes")
        if isinstance(mosaic_sizes_raw, Mapping) and mosaic_sizes_raw:
            lines.append(
                (
                    "{label}: mosaic_sizes beam_x={beam_x} beam_y={beam_y} "
                    "theta={theta} phi={phi} wavelength={wave} wavelength_i={wave_i} "
                    "sample_weights={weights}"
                ).format(
                    label=label,
                    beam_x=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("beam_x_array", None),
                        float_digits=0,
                    ),
                    beam_y=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("beam_y_array", None),
                        float_digits=0,
                    ),
                    theta=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("theta_array", None),
                        float_digits=0,
                    ),
                    phi=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("phi_array", None),
                        float_digits=0,
                    ),
                    wave=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("wavelength_array", None),
                        float_digits=0,
                    ),
                    wave_i=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("wavelength_i_array", None),
                        float_digits=0,
                    ),
                    weights=_geometry_fit_debug_value_text(
                        mosaic_sizes_raw.get("sample_weights", None),
                        float_digits=0,
                    ),
                )
            )
        if runtime_diag:
            lines.append(
                (
                    "{label}: runtime stage={stage} status={status} "
                    "hit_tables={tables} nonempty_hit_tables={nonempty} "
                    "finite_hit_rows={rows} peak_count={peaks}"
                ).format(
                    label=label,
                    stage=str(runtime_diag.get("stage", "<unknown>")),
                    status=str(runtime_diag.get("status", "<unknown>")),
                    tables=_geometry_fit_debug_value_text(
                        runtime_diag.get("hit_table_count", None),
                        float_digits=0,
                    ),
                    nonempty=_geometry_fit_debug_value_text(
                        runtime_diag.get("nonempty_hit_table_count", None),
                        float_digits=0,
                    ),
                    rows=_geometry_fit_debug_value_text(
                        runtime_diag.get("finite_hit_row_total", None),
                        float_digits=0,
                    ),
                    peaks=_geometry_fit_debug_value_text(
                        runtime_diag.get(
                            "peak_count",
                            runtime_diag.get("peak_center_count", None),
                        ),
                        float_digits=0,
                    ),
                )
            )
            if "hit_row_counts" in runtime_diag:
                lines.append(
                    ("{label}: runtime hit_row_counts={counts}").format(
                        label=label,
                        counts=_geometry_fit_short_count_list(runtime_diag.get("hit_row_counts")),
                    )
                )
        exception_type = diagnostics.get(
            "exception_type",
            runtime_diag.get("exception_type"),
        )
        exception_message = diagnostics.get(
            "exception_message",
            runtime_diag.get("exception_message"),
        )
        if exception_type or exception_message:
            lines.append(
                ("{label}: exception={exc_type}: {exc_message}").format(
                    label=label,
                    exc_type=str(exception_type or "<unknown>"),
                    exc_message=str(exception_message or "<no message>"),
                )
            )
    return lines


def build_geometry_fit_preflight_log_sections(
    *,
    error_text: str,
    params: Mapping[str, object] | None,
    var_names: Sequence[object],
    dataset_infos: Sequence[Mapping[str, object]] | None,
    current_dataset: Mapping[str, object] | None,
    selected_background_indices: Sequence[object],
    joint_background_mode: bool,
    geometry_runtime_cfg: Mapping[str, object] | None = None,
) -> list[tuple[str, list[str]]]:
    """Build the log sections for a geometry-fit failure before solver start."""

    debug_logging = geometry_fit_debug_logging_enabled(geometry_runtime_cfg)
    sections: list[tuple[str, list[str]]] = [
        (
            "Failure:",
            [
                str(error_text).strip(),
                "stage=preflight",
            ],
        )
    ]
    sections.extend(
        build_geometry_fit_start_log_sections(
            params=params,
            var_names=var_names,
            dataset_infos=dataset_infos,
            current_dataset=current_dataset,
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
            geometry_runtime_cfg=geometry_runtime_cfg,
        )
    )
    if debug_logging:
        run_request_lines = _build_geometry_fit_run_request_lines(
            selected_background_indices=selected_background_indices,
            joint_background_mode=joint_background_mode,
            dataset_infos=dataset_infos,
            current_dataset=current_dataset,
        )
        if run_request_lines:
            sections.append(("Run request:", run_request_lines))
        runtime_cfg_lines = _build_geometry_fit_runtime_config_lines(geometry_runtime_cfg)
        if runtime_cfg_lines:
            sections.append(("Runtime configuration:", runtime_cfg_lines))
    return sections


def write_geometry_fit_preflight_failure_log(
    *,
    stamp: str,
    error_text: str,
    log_path: Path | str,
    log_sections: Sequence[tuple[str, Sequence[str]]] | None = None,
) -> Path:
    """Persist one geometry-fit preflight failure log."""

    resolved_log_path = Path(log_path)
    if geometry_fit_all_logging_disabled():
        return resolved_log_path
    sections = list(log_sections or ())
    if not sections:
        sections = [("Failure:", [str(error_text).strip(), "stage=preflight"])]
    resolved_log_path, log_line, log_section = _build_geometry_fit_log_writers(resolved_log_path)
    log_line(f"Geometry fit aborted before solver start: {stamp}")
    log_line("")
    for title, lines in sections:
        log_section(str(title), [str(line) for line in lines or ()])
    return resolved_log_path


def write_geometry_fit_run_start_log(
    *,
    stamp: str,
    prepared_run: GeometryFitPreparedRun,
    cmd_line: Callable[[str], None],
    log_line: Callable[[str], None],
    log_section: Callable[[str, Sequence[str]], None],
) -> None:
    """Emit the runtime console/log prelude for one prepared geometry-fit run."""

    cmd_line(str(prepared_run.start_cmd_line))
    log_line(f"Geometry fit started: {stamp}")
    log_line("")
    for title, lines in prepared_run.start_log_sections:
        log_section(title, list(lines))


def should_apply_geometry_fit_runtime_safety_overrides(
    *,
    platform_name: str | None = None,
    version_info: Sequence[object] | None = None,
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether GUI geometry fitting should keep the unsafe runtime path off."""

    if platform_name is None:
        platform_name = os.name
    if version_info is None:
        version_info = sys.version_info
    if env is None:
        env = os.environ

    opt_out = str(env.get("RA_SIM_ALLOW_UNSAFE_GEOMETRY_FIT_RUNTIME", "")).strip().lower()
    if opt_out in {"1", "true", "yes", "on"}:
        return False

    if str(platform_name).strip().lower() != "nt":
        return False

    try:
        major = int(version_info[0])
        minor = int(version_info[1])
    except Exception:
        return False
    return (major, minor) >= (3, 13)


def apply_geometry_fit_runtime_safety_overrides(
    refinement_config: Mapping[str, object] | None,
    *,
    platform_name: str | None = None,
    version_info: Sequence[object] | None = None,
    env: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str | None]:
    """Return one copied refinement config with GUI runtime safety overrides."""

    if isinstance(refinement_config, Mapping):
        resolved = copy.deepcopy(dict(refinement_config))
    else:
        resolved = {}

    if bool(resolved.get("allow_unsafe_runtime", False)):
        return resolved, None

    if not should_apply_geometry_fit_runtime_safety_overrides(
        platform_name=platform_name,
        version_info=version_info,
        env=env,
    ):
        return resolved, None

    optimizer_cfg_raw = resolved.get("optimizer", None)
    solver_cfg_raw = resolved.get("solver", {})
    if isinstance(optimizer_cfg_raw, Mapping):
        solver_cfg = dict(optimizer_cfg_raw)
    elif isinstance(solver_cfg_raw, Mapping):
        solver_cfg = dict(solver_cfg_raw)
    else:
        solver_cfg = {}

    if isinstance(optimizer_cfg_raw, Mapping):
        resolved["optimizer"] = solver_cfg
    resolved["solver"] = solver_cfg

    return (
        resolved,
        (
            "Windows/Python 3.13 runtime guard enabled: "
            "unsafe runtime disabled, safe-wrapper Numba allowed."
        ),
    )


def _geometry_fit_assign_fit_run_id_to_entries(
    entries: object,
    *,
    fit_run_id: str,
) -> None:
    """Attach one stable run id to mutable entry collections in place."""

    if not isinstance(entries, list):
        return
    for fallback_index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            continue
        raw_entry.setdefault("fit_run_id", str(fit_run_id))
        if "pair_id" not in raw_entry:
            overlay_index = _geometry_fit_coerce_nonnegative_index(
                raw_entry.get("overlay_match_index")
            )
            if overlay_index is not None:
                raw_entry["pair_id"] = f"pair[{int(overlay_index)}]"
            elif "pair_index" in raw_entry:
                try:
                    raw_entry["pair_id"] = f"pair[{int(raw_entry['pair_index'])}]"
                except Exception:
                    raw_entry["pair_id"] = f"pair[{int(fallback_index)}]"


def _geometry_fit_assign_fit_run_id(
    prepared_run: GeometryFitPreparedRun,
    *,
    fit_run_id: str,
) -> None:
    """Propagate one run id across mutable prepared-run payloads."""

    dataset_payloads: list[dict[str, object]] = []
    if isinstance(prepared_run.current_dataset, dict):
        dataset_payloads.append(prepared_run.current_dataset)
    dataset_payloads.extend(
        dataset for dataset in (prepared_run.dataset_infos or []) if isinstance(dataset, dict)
    )
    for dataset in dataset_payloads:
        dataset.setdefault("fit_run_id", str(fit_run_id))
        for key in (
            "measured_for_fit",
            "initial_pairs_display",
            "source_resolution_diagnostics",
            "source_rows_for_trace",
        ):
            _geometry_fit_assign_fit_run_id_to_entries(
                dataset.get(key),
                fit_run_id=str(fit_run_id),
            )
    for raw_spec in prepared_run.dataset_specs or []:
        if isinstance(raw_spec, dict):
            raw_spec.setdefault("fit_run_id", str(fit_run_id))


def _geometry_fit_trace_namespace_label(
    entry: Mapping[str, object] | None,
    field_name: str,
    *,
    phase: str,
) -> str:
    """Return the namespace label for one identity field in trace output."""

    if not isinstance(entry, Mapping):
        return "unset"
    if field_name == "source_reflection_index":
        return str(entry.get("source_reflection_namespace", "unset") or "unset")
    if field_name in {"source_branch_index", "source_peak_index"}:
        branch_idx = (
            _geometry_fit_source_branch_index(entry)
            if field_name == "source_branch_index"
            else _geometry_fit_coerce_nonnegative_index(entry.get("source_peak_index"))
        )
        return "branch_index" if branch_idx in {0, 1} else "unset"
    if field_name in {"source_table_index", "source_row_index"}:
        if phase == "live_source_rows":
            return (
                "full_hit_table"
                if _geometry_fit_trusted_full_reflection_identity(entry)
                else "live_cache_row"
            )
        if str(entry.get("resolution_kind", "") or "").strip() == "hkl_fallback":
            return "subset_hit_table"
        return (
            "full_hit_table"
            if _geometry_fit_trusted_full_reflection_identity(entry)
            else "subset_hit_table"
        )
    if field_name in {"resolved_table_index", "resolved_peak_index"}:
        if phase == "seed_correspondence":
            return "subset_hit_table"
        if phase in {"full_beam_polish_correspondence", "acceptance_residuals"}:
            return "full_hit_table"
        return "resolved"
    return "unset"


def _geometry_fit_trace_simulated_point(
    entry: Mapping[str, object] | None,
) -> list[float | None] | None:
    """Extract one simulated detector point from any supported trace entry."""

    if not isinstance(entry, Mapping):
        return None
    for x_key, y_key in (
        ("simulated_x", "simulated_y"),
        ("sim_col", "sim_row"),
    ):
        x_value = _geometry_fit_metric_float(entry.get(x_key, np.nan))
        y_value = _geometry_fit_metric_float(entry.get(y_key, np.nan))
        if np.isfinite(x_value) and np.isfinite(y_value):
            return [float(x_value), float(y_value)]
    return None


def _geometry_fit_trace_measured_point(
    entry: Mapping[str, object] | None,
) -> list[float | None] | None:
    """Extract one measured point from any supported trace entry."""

    if not isinstance(entry, Mapping):
        return None
    for x_key, y_key in (
        ("measured_x", "measured_y"),
        ("x", "y"),
    ):
        x_value = _geometry_fit_metric_float(entry.get(x_key, np.nan))
        y_value = _geometry_fit_metric_float(entry.get(y_key, np.nan))
        if np.isfinite(x_value) and np.isfinite(y_value):
            return [float(x_value), float(y_value)]
    return None


def _geometry_fit_trace_optimizer_residual_px(
    entry: Mapping[str, object] | None,
) -> float | None:
    """Return the optimizer-space residual magnitude when available."""

    if not isinstance(entry, Mapping):
        return None
    weighted_dx = _geometry_fit_metric_float(entry.get("weighted_dx_px", np.nan))
    weighted_dy = _geometry_fit_metric_float(entry.get("weighted_dy_px", np.nan))
    if np.isfinite(weighted_dx) and np.isfinite(weighted_dy):
        return float(np.hypot(weighted_dx, weighted_dy))
    return None


def _geometry_fit_trace_pair_record(
    *,
    phase: str,
    dataset_info: Mapping[str, object] | None,
    entry: Mapping[str, object],
    fit_run_id: str,
    record_type: str = "pair",
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one structured phase record for a pair or source-row entry."""

    dataset = dict(dataset_info or {})
    simulation_diag = (
        dict(dataset.get("simulation_diagnostics", {}))
        if isinstance(dataset.get("simulation_diagnostics"), Mapping)
        else {}
    )
    cache_metadata = (
        dict(dataset.get("cache_metadata", {}))
        if isinstance(dataset.get("cache_metadata"), Mapping)
        else {}
    )
    pair_id = str(entry.get("pair_id") or f"pair[{int(entry.get('overlay_match_index', 0) or 0)}]")
    simulated_point = _geometry_fit_trace_simulated_point(entry)
    measured_point = _geometry_fit_trace_measured_point(entry)
    dx_px = _geometry_fit_metric_float(entry.get("dx_px", np.nan))
    dy_px = _geometry_fit_metric_float(entry.get("dy_px", np.nan))
    if (
        (not np.isfinite(dx_px) or not np.isfinite(dy_px))
        and isinstance(simulated_point, list)
        and len(simulated_point) == 2
        and isinstance(measured_point, list)
        and len(measured_point) == 2
        and simulated_point[0] is not None
        and simulated_point[1] is not None
        and measured_point[0] is not None
        and measured_point[1] is not None
    ):
        dx_px = _geometry_fit_metric_float(float(simulated_point[0]) - float(measured_point[0]))
        dy_px = _geometry_fit_metric_float(float(simulated_point[1]) - float(measured_point[1]))
    record: dict[str, object] = {
        "record_type": str(record_type),
        "fit_run_id": str(fit_run_id),
        "pair_id": pair_id if record_type == "pair" else None,
        "phase": str(phase),
        "dataset_index": int(dataset.get("dataset_index", 0) or 0),
        "background_index": int(dataset.get("dataset_index", 0) or 0),
        "background_label": str(dataset.get("label", f"bg[{dataset.get('dataset_index', 0)}]")),
        "overlay_match_index": _geometry_fit_coerce_nonnegative_index(
            entry.get("overlay_match_index")
        ),
        "source_cache_origin": str(
            simulation_diag.get(
                "created_from",
                cache_metadata.get("cache_source", "<unknown>"),
            )
            or "<unknown>"
        ),
        "requested_signature_summary": _geometry_fit_cache_jsonable(
            simulation_diag.get(
                "requested_signature_summary",
                cache_metadata.get("source_snapshot_requested_signature_summary"),
            )
        ),
        "simulation_revision": _geometry_fit_cache_jsonable(
            simulation_diag.get(
                "simulation_signature",
                cache_metadata.get("source_snapshot_stored_signature_summary"),
            )
        ),
        "hkl": _geometry_fit_normalized_hkl(entry.get("hkl")),
        "source_reflection_index": entry.get("source_reflection_index"),
        "source_branch_index": _geometry_fit_source_branch_index(entry),
        "source_peak_index": entry.get("source_peak_index"),
        "source_table_index": entry.get("source_table_index"),
        "source_row_index": entry.get("source_row_index"),
        "resolved_table_index": entry.get("resolved_table_index"),
        "resolved_peak_index": entry.get("resolved_peak_index"),
        "source_reflection_namespace": entry.get("source_reflection_namespace"),
        "source_reflection_is_full": bool(entry.get("source_reflection_is_full", False)),
        "source_reflection_index_namespace": _geometry_fit_trace_namespace_label(
            entry,
            "source_reflection_index",
            phase=phase,
        ),
        "source_table_index_namespace": _geometry_fit_trace_namespace_label(
            entry,
            "source_table_index",
            phase=phase,
        ),
        "source_row_index_namespace": _geometry_fit_trace_namespace_label(
            entry,
            "source_row_index",
            phase=phase,
        ),
        "source_peak_index_namespace": _geometry_fit_trace_namespace_label(
            entry,
            "source_peak_index",
            phase=phase,
        ),
        "source_branch_index_namespace": _geometry_fit_trace_namespace_label(
            entry,
            "source_branch_index",
            phase=phase,
        ),
        "resolved_table_index_namespace": _geometry_fit_trace_namespace_label(
            entry,
            "resolved_table_index",
            phase=phase,
        ),
        "resolved_peak_index_namespace": _geometry_fit_trace_namespace_label(
            entry,
            "resolved_peak_index",
            phase=phase,
        ),
        "fit_source_resolution_kind": entry.get("fit_source_resolution_kind"),
        "resolution_kind": entry.get("resolution_kind"),
        "resolution_reason": entry.get(
            "correspondence_resolution_reason",
            entry.get("resolution_reason"),
        ),
        "match_status": entry.get("match_status", "selected"),
        "canonical_identity": (
            [
                int(_geometry_fit_source_peak_key(entry)[0]),
                int(_geometry_fit_source_peak_key(entry)[1]),
            ]
            if _geometry_fit_source_peak_key(entry) is not None
            else None
        ),
        "simulated_point": simulated_point,
        "measured_point": measured_point,
        "dx_px": dx_px,
        "dy_px": dy_px,
        "optimizer_residual_px": _geometry_fit_trace_optimizer_residual_px(entry),
        "detector_residual_px": _geometry_fit_metric_float(entry.get("distance_px", np.nan)),
        "placement_error_px": _geometry_fit_metric_float(entry.get("placement_error_px", np.nan)),
    }
    for key in (
        "measured_detector_field_name",
        "measured_detector_input_frame",
        "measured_detector_frame_reason",
        "simulated_detector_field_name",
        "simulated_detector_input_frame",
        "simulated_detector_frame_reason",
        "measured_native_col",
        "measured_native_row",
        "simulated_native_col",
        "simulated_native_row",
        "measured_two_theta_deg",
        "measured_phi_deg",
        "simulated_two_theta_deg",
        "simulated_phi_deg",
        "measured_fit_space_source",
        "simulated_fit_space_source",
        "fit_space_anchor_override",
        "fit_space_projector_kind",
        "cake_bundle_signature",
        "measured_native_frame_conversion_source",
        "simulated_native_frame_conversion_source",
        "measured_native_frame_conversion_count",
        "simulated_native_frame_conversion_count",
        "valid",
        "invalid_projection_reason",
        "measured_invalid_projection_reason",
        "simulated_invalid_projection_reason",
    ):
        if key in entry:
            record[str(key)] = _geometry_fit_cache_jsonable(entry.get(key))
    if isinstance(extra, Mapping):
        record.update(
            {str(key): _geometry_fit_cache_jsonable(value) for key, value in extra.items()}
        )
    return record


def _geometry_fit_phase_summary_record(
    *,
    phase: str,
    dataset_info: Mapping[str, object] | None,
    entries: Sequence[Mapping[str, object]] | None,
    fit_run_id: str,
    dropped_pair_count: int = 0,
    trust_marker_drop_count: int = 0,
) -> dict[str, object]:
    """Build one compact phase summary for the structured trace."""

    normalized_entries = [dict(entry) for entry in (entries or ()) if isinstance(entry, Mapping)]
    canonical_counts: Counter[tuple[int, int]] = Counter()
    simulated_point_counts: Counter[tuple[float, float]] = Counter()
    trusted_full_ids = 0
    canonical_branch_ids = 0
    legacy_fallbacks = 0
    hkl_fallbacks = 0
    for entry in normalized_entries:
        if _geometry_fit_trusted_full_reflection_identity(entry):
            trusted_full_ids += 1
        if _geometry_fit_source_branch_index(entry) in {0, 1}:
            canonical_branch_ids += 1
        fit_kind = str(entry.get("fit_source_resolution_kind", "") or "")
        resolution_kind = str(entry.get("resolution_kind", "") or "")
        if "legacy_dense" in fit_kind or "legacy_dense" in resolution_kind:
            legacy_fallbacks += 1
        if fit_kind == "hkl_fallback" or resolution_kind == "hkl_fallback":
            hkl_fallbacks += 1
        canonical_key = _geometry_fit_source_peak_key(entry)
        if canonical_key is not None:
            canonical_counts[(int(canonical_key[0]), int(canonical_key[1]))] += 1
        simulated_point = _geometry_fit_trace_simulated_point(entry)
        if (
            isinstance(simulated_point, list)
            and len(simulated_point) == 2
            and simulated_point[0] is not None
            and simulated_point[1] is not None
        ):
            simulated_point_counts[
                (round(float(simulated_point[0]), 6), round(float(simulated_point[1]), 6))
            ] += 1
    dataset = dict(dataset_info or {})
    return {
        "record_type": "summary",
        "fit_run_id": str(fit_run_id),
        "phase": str(phase),
        "dataset_index": int(dataset.get("dataset_index", 0) or 0),
        "background_index": int(dataset.get("dataset_index", 0) or 0),
        "background_label": str(dataset.get("label", f"bg[{dataset.get('dataset_index', 0)}]")),
        "entry_count": int(len(normalized_entries)),
        "trusted_full_id_count": int(trusted_full_ids),
        "canonical_branch_id_count": int(canonical_branch_ids),
        "legacy_fallback_count": int(legacy_fallbacks),
        "hkl_fallback_count": int(hkl_fallbacks),
        "dropped_pair_count": int(max(0, dropped_pair_count)),
        "duplicate_canonical_id_count": int(
            sum(1 for count in canonical_counts.values() if int(count) > 1)
        ),
        "duplicate_simulated_target_count": int(
            sum(1 for count in simulated_point_counts.values() if int(count) > 1)
        ),
        "trust_marker_drop_count": int(max(0, trust_marker_drop_count)),
    }


def _geometry_fit_duplicate_target_records(
    diagnostics: Sequence[Mapping[str, object]] | None,
    *,
    fit_run_id: str,
    phase: str,
) -> list[dict[str, object]]:
    """Return duplicate-target diagnostics for final correspondence phases."""

    pair_entries = [
        dict(entry)
        for entry in (diagnostics or ())
        if isinstance(entry, Mapping) and str(entry.get("match_status", "") or "") == "matched"
    ]
    by_identity: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    by_point: defaultdict[tuple[float, float], list[str]] = defaultdict(list)
    for entry in pair_entries:
        pair_id = str(entry.get("pair_id", "") or "").strip()
        if not pair_id:
            continue
        identity = _geometry_fit_source_peak_key(entry)
        if identity is not None:
            by_identity[(int(identity[0]), int(identity[1]))].append(pair_id)
        simulated_point = _geometry_fit_trace_simulated_point(entry)
        if (
            isinstance(simulated_point, list)
            and len(simulated_point) == 2
            and simulated_point[0] is not None
            and simulated_point[1] is not None
        ):
            by_point[
                (round(float(simulated_point[0]), 6), round(float(simulated_point[1]), 6))
            ].append(pair_id)
    records: list[dict[str, object]] = []
    for identity, pair_ids in sorted(by_identity.items()):
        if len(pair_ids) <= 1:
            continue
        records.append(
            {
                "record_type": "duplicate_target",
                "fit_run_id": str(fit_run_id),
                "phase": str(phase),
                "duplicate_kind": "canonical_identity",
                "canonical_identity": [int(identity[0]), int(identity[1])],
                "pair_ids": list(pair_ids),
            }
        )
    for point, pair_ids in sorted(by_point.items()):
        if len(pair_ids) <= 1:
            continue
        records.append(
            {
                "record_type": "duplicate_target",
                "fit_run_id": str(fit_run_id),
                "phase": str(phase),
                "duplicate_kind": "simulated_point",
                "simulated_point": [float(point[0]), float(point[1])],
                "pair_ids": list(pair_ids),
            }
        )
    return records


def _build_geometry_fit_trace_records(
    *,
    fit_run_id: str,
    prepared_run: GeometryFitPreparedRun,
    result: object,
    apply_result: GeometryFitRuntimeApplyResult,
    log_path: Path,
) -> list[dict[str, object]]:
    """Build the JSONL phase trace for one geometry-fit run."""

    full_beam_summary = getattr(result, "full_beam_polish_summary", None)
    point_match_summary = getattr(result, "point_match_summary", None)
    debug_summary = getattr(result, "geometry_fit_debug_summary", None)
    debug_vars = (
        [str(name) for name in debug_summary.get("var_names", ()) or ()]
        if isinstance(debug_summary, Mapping)
        else []
    )
    run_stage_timings: dict[str, object] = {}
    if isinstance(prepared_run.stage_timing_s, Mapping):
        run_stage_timings.update(
            {
                str(key): _geometry_fit_cache_jsonable(value)
                for key, value in prepared_run.stage_timing_s.items()
            }
        )
    solver_stage_timings = getattr(result, "geometry_fit_stage_timings", None)
    if isinstance(solver_stage_timings, Mapping):
        run_stage_timings.update(
            {
                str(key): _geometry_fit_cache_jsonable(value)
                for key, value in solver_stage_timings.items()
            }
        )

    records: list[dict[str, object]] = [
        {
            "record_type": "run",
            "fit_run_id": str(fit_run_id),
            "phase": "run",
            "accepted": bool(apply_result.accepted),
            "rejection_reason": apply_result.rejection_reason,
            "log_path": str(log_path),
            "dataset_count": int(len(prepared_run.dataset_infos or [])),
            "joint_background_mode": bool(prepared_run.joint_background_mode),
            "weighted_residual_rms_px": _geometry_fit_metric_float(
                getattr(result, "weighted_residual_rms_px", np.nan)
            ),
            "detector_rms_px": _geometry_fit_metric_float(geometry_fit_result_rms(result)),
            "fit_quality_passed": bool(
                full_beam_summary.get("fit_quality_passed", False)
                if isinstance(full_beam_summary, Mapping)
                else False
            ),
            "final_metric_name": str(getattr(result, "final_metric_name", "") or ""),
            "selection_status": (
                str(full_beam_summary.get("selection_status", "") or "")
                if isinstance(full_beam_summary, Mapping)
                else ""
            ),
            "selected_candidate_name": (
                str(full_beam_summary.get("selected_candidate_name", "") or "")
                if isinstance(full_beam_summary, Mapping)
                else ""
            ),
            "selected_candidate_source": (
                str(full_beam_summary.get("selected_candidate_source", "") or "")
                if isinstance(full_beam_summary, Mapping)
                else ""
            ),
            "best_valid_raw_detector_candidate_name": (
                str(full_beam_summary.get("best_valid_raw_detector_candidate_name", "") or "")
                if isinstance(full_beam_summary, Mapping)
                else ""
            ),
            "best_valid_raw_detector_candidate_source": (
                str(full_beam_summary.get("best_valid_raw_detector_candidate_source", "") or "")
                if isinstance(full_beam_summary, Mapping)
                else ""
            ),
            "constraint_count": int(
                full_beam_summary.get("constraint_count", 0)
                if isinstance(full_beam_summary, Mapping)
                else 0
            ),
            "active_fit_variable_count": int(
                full_beam_summary.get("active_fit_variable_count", len(debug_vars))
                if isinstance(full_beam_summary, Mapping)
                else len(debug_vars)
            ),
            "active_fit_variables": _geometry_fit_cache_jsonable(
                full_beam_summary.get("active_fit_variables", debug_vars)
                if isinstance(full_beam_summary, Mapping)
                else debug_vars
            ),
            "candidate_ledger": _geometry_fit_cache_jsonable(
                full_beam_summary.get("candidate_ledger")
                if isinstance(full_beam_summary, Mapping)
                else None
            ),
            "dynamic_point_geometry_fit": bool(
                getattr(result, "geometry_fit_debug_summary", {}).get(
                    "dynamic_point_geometry_fit",
                    False,
                )
                if isinstance(getattr(result, "geometry_fit_debug_summary", None), Mapping)
                else False
            ),
            "full_beam_polish_enabled": bool(
                full_beam_summary.get("enabled", False)
                if isinstance(full_beam_summary, Mapping)
                else False
            ),
            "full_beam_polish_accepted": bool(
                full_beam_summary.get("accepted", False)
                if isinstance(full_beam_summary, Mapping)
                else False
            ),
            "full_beam_start_vector_source": (
                str(full_beam_summary.get("start_vector_source", "") or "")
                if isinstance(full_beam_summary, Mapping)
                else ""
            ),
            "seed_correspondence_count": int(
                full_beam_summary.get("seed_correspondence_count", 0)
                if isinstance(full_beam_summary, Mapping)
                else 0
            ),
            "nfev": int(getattr(result, "nfev", 0) or 0),
            "stage_timing_s": run_stage_timings,
            "point_match_summary": _geometry_fit_cache_jsonable(point_match_summary),
            "full_beam_polish_summary": _geometry_fit_cache_jsonable(full_beam_summary),
        }
    ]

    for raw_dataset in prepared_run.dataset_infos or []:
        if not isinstance(raw_dataset, Mapping):
            continue
        dataset = dict(raw_dataset)
        saved_trust_by_pair: dict[str, bool] = {}
        preflight_trust_by_pair: dict[str, bool] = {}
        saved_pairs = [
            dict(entry)
            for entry in dataset.get("initial_pairs_display", ())
            if isinstance(entry, Mapping)
        ]
        source_rows = [
            dict(entry)
            for entry in dataset.get("source_rows_for_trace", ())
            if isinstance(entry, Mapping)
        ]
        preflight_pairs = [
            dict(entry)
            for entry in dataset.get("measured_for_fit", ())
            if isinstance(entry, Mapping)
        ]
        source_resolution = [
            dict(entry)
            for entry in dataset.get("source_resolution_diagnostics", ())
            if isinstance(entry, Mapping)
        ]

        for entry in saved_pairs:
            pair_id = str(entry.get("pair_id", "") or "").strip()
            if pair_id:
                saved_trust_by_pair[pair_id] = _geometry_fit_trusted_full_reflection_identity(entry)
            records.append(
                _geometry_fit_trace_pair_record(
                    phase="saved_pairs",
                    dataset_info=dataset,
                    entry=entry,
                    fit_run_id=str(fit_run_id),
                )
            )
        records.append(
            _geometry_fit_phase_summary_record(
                phase="saved_pairs",
                dataset_info=dataset,
                entries=saved_pairs,
                fit_run_id=str(fit_run_id),
            )
        )

        for entry in source_rows:
            records.append(
                _geometry_fit_trace_pair_record(
                    phase="live_source_rows",
                    dataset_info=dataset,
                    entry=entry,
                    fit_run_id=str(fit_run_id),
                    record_type="source_row",
                )
            )
        records.append(
            _geometry_fit_phase_summary_record(
                phase="live_source_rows",
                dataset_info=dataset,
                entries=source_rows,
                fit_run_id=str(fit_run_id),
            )
        )

        for entry in preflight_pairs:
            pair_id = str(entry.get("pair_id", "") or "").strip()
            if pair_id:
                preflight_trust_by_pair[pair_id] = _geometry_fit_trusted_full_reflection_identity(
                    entry
                )
            records.append(
                _geometry_fit_trace_pair_record(
                    phase="preflight_normalized_pairs",
                    dataset_info=dataset,
                    entry=entry,
                    fit_run_id=str(fit_run_id),
                )
            )
        trust_drop_count = sum(
            1
            for pair_id, was_trusted in saved_trust_by_pair.items()
            if was_trusted and not bool(preflight_trust_by_pair.get(pair_id, False))
        )
        dropped_pair_count = sum(
            1
            for entry in source_resolution
            if not bool(entry.get("fit_resolved", entry.get("strict_resolved", False)))
        )
        records.append(
            _geometry_fit_phase_summary_record(
                phase="preflight_normalized_pairs",
                dataset_info=dataset,
                entries=preflight_pairs,
                fit_run_id=str(fit_run_id),
                dropped_pair_count=int(dropped_pair_count),
                trust_marker_drop_count=int(trust_drop_count),
            )
        )

        simulation_diag = (
            dict(dataset.get("simulation_diagnostics", {}))
            if isinstance(dataset.get("simulation_diagnostics"), Mapping)
            else {}
        )
        if simulation_diag:
            records.append(
                {
                    "record_type": "summary",
                    "fit_run_id": str(fit_run_id),
                    "phase": "subset_mapping",
                    "dataset_index": int(dataset.get("dataset_index", 0) or 0),
                    "background_index": int(dataset.get("dataset_index", 0) or 0),
                    "background_label": str(
                        dataset.get("label", f"bg[{dataset.get('dataset_index', 0)}]")
                    ),
                    "point_match_summary": _geometry_fit_cache_jsonable(
                        getattr(result, "point_match_summary", None)
                    ),
                    "source_snapshot_status": simulation_diag.get("status"),
                    "live_runtime_cache_validation": _geometry_fit_cache_jsonable(
                        simulation_diag.get("live_runtime_cache_validation")
                    ),
                }
            )
            validation = simulation_diag.get("live_runtime_cache_validation")
            if isinstance(validation, Mapping):
                for raw_failure in validation.get("pair_failures", ()) or ():
                    if not isinstance(raw_failure, Mapping):
                        continue
                    records.append(
                        {
                            "record_type": "validation_failure",
                            "fit_run_id": str(fit_run_id),
                            "phase": "live_source_rows",
                            "dataset_index": int(dataset.get("dataset_index", 0) or 0),
                            "background_index": int(dataset.get("dataset_index", 0) or 0),
                            **_geometry_fit_cache_jsonable(dict(raw_failure)),
                        }
                    )
            dual_path_diff = simulation_diag.get("cache_metadata", {})
            if isinstance(dual_path_diff, Mapping):
                diff_entries = dual_path_diff.get("live_runtime_cache_dual_path_diff")
                if isinstance(diff_entries, Sequence) and not isinstance(
                    diff_entries,
                    (str, bytes, bytearray),
                ):
                    for raw_diff in diff_entries:
                        if not isinstance(raw_diff, Mapping):
                            continue
                        records.append(
                            {
                                "record_type": "dual_path_diff",
                                "fit_run_id": str(fit_run_id),
                                "phase": "live_source_rows",
                                "dataset_index": int(dataset.get("dataset_index", 0) or 0),
                                "background_index": int(dataset.get("dataset_index", 0) or 0),
                                **_geometry_fit_cache_jsonable(dict(raw_diff)),
                            }
                        )

    seed_records = []
    if isinstance(full_beam_summary, Mapping):
        raw_seed_records = full_beam_summary.get("seed_correspondence_records", ())
        if isinstance(raw_seed_records, Sequence) and not isinstance(
            raw_seed_records,
            (str, bytes, bytearray),
        ):
            seed_records = [dict(entry) for entry in raw_seed_records if isinstance(entry, Mapping)]
    for entry in seed_records:
        dataset_index = int(entry.get("dataset_index", 0) or 0)
        dataset = next(
            (
                dict(item)
                for item in prepared_run.dataset_infos or []
                if isinstance(item, Mapping)
                and int(item.get("dataset_index", -1)) == int(dataset_index)
            ),
            {"dataset_index": int(dataset_index)},
        )
        records.append(
            _geometry_fit_trace_pair_record(
                phase="seed_correspondence",
                dataset_info=dataset,
                entry=entry,
                fit_run_id=str(fit_run_id),
            )
        )
    if seed_records:
        for raw_dataset in prepared_run.dataset_infos or []:
            if not isinstance(raw_dataset, Mapping):
                continue
            dataset = dict(raw_dataset)
            dataset_seed_records = [
                dict(entry)
                for entry in seed_records
                if int(entry.get("dataset_index", 0) or 0)
                == int(dataset.get("dataset_index", 0) or 0)
            ]
            records.append(
                _geometry_fit_phase_summary_record(
                    phase="seed_correspondence",
                    dataset_info=dataset,
                    entries=dataset_seed_records,
                    fit_run_id=str(fit_run_id),
                    dropped_pair_count=max(
                        0,
                        int(dataset.get("pair_count", 0) or 0) - len(dataset_seed_records),
                    ),
                )
            )

    start_point_records = []
    if isinstance(full_beam_summary, Mapping):
        raw_start_point_records = full_beam_summary.get("start_point_match_diagnostics", ())
        if isinstance(raw_start_point_records, Sequence) and not isinstance(
            raw_start_point_records,
            (str, bytes, bytearray),
        ):
            start_point_records = [
                dict(entry) for entry in raw_start_point_records if isinstance(entry, Mapping)
            ]
    for entry in start_point_records:
        dataset_index = int(entry.get("dataset_index", 0) or 0)
        dataset = next(
            (
                dict(item)
                for item in prepared_run.dataset_infos or []
                if isinstance(item, Mapping)
                and int(item.get("dataset_index", -1)) == int(dataset_index)
            ),
            {"dataset_index": int(dataset_index)},
        )
        records.append(
            _geometry_fit_trace_pair_record(
                phase="requested_start_correspondence",
                dataset_info=dataset,
                entry=entry,
                fit_run_id=str(fit_run_id),
            )
        )
    if start_point_records:
        for raw_dataset in prepared_run.dataset_infos or []:
            if not isinstance(raw_dataset, Mapping):
                continue
            dataset = dict(raw_dataset)
            dataset_start_records = [
                dict(entry)
                for entry in start_point_records
                if int(entry.get("dataset_index", 0) or 0)
                == int(dataset.get("dataset_index", 0) or 0)
            ]
            records.append(
                _geometry_fit_phase_summary_record(
                    phase="requested_start_correspondence",
                    dataset_info=dataset,
                    entries=dataset_start_records,
                    fit_run_id=str(fit_run_id),
                    dropped_pair_count=max(
                        0,
                        int(dataset.get("pair_count", 0) or 0) - len(dataset_start_records),
                    ),
                )
            )

    polish_records = []
    if isinstance(full_beam_summary, Mapping):
        raw_polish_records = full_beam_summary.get("point_match_diagnostics", ())
        if isinstance(raw_polish_records, Sequence) and not isinstance(
            raw_polish_records,
            (str, bytes, bytearray),
        ):
            polish_records = [
                dict(entry) for entry in raw_polish_records if isinstance(entry, Mapping)
            ]
    for entry in polish_records:
        dataset_index = int(entry.get("dataset_index", 0) or 0)
        dataset = next(
            (
                dict(item)
                for item in prepared_run.dataset_infos or []
                if isinstance(item, Mapping)
                and int(item.get("dataset_index", -1)) == int(dataset_index)
            ),
            {"dataset_index": int(dataset_index)},
        )
        records.append(
            _geometry_fit_trace_pair_record(
                phase="full_beam_polish_correspondence",
                dataset_info=dataset,
                entry=entry,
                fit_run_id=str(fit_run_id),
            )
        )
    if polish_records:
        records.extend(
            _geometry_fit_duplicate_target_records(
                polish_records,
                fit_run_id=str(fit_run_id),
                phase="full_beam_polish_correspondence",
            )
        )

    final_records = [
        dict(entry)
        for entry in (getattr(result, "point_match_diagnostics", None) or ())
        if isinstance(entry, Mapping)
    ]
    for entry in final_records:
        dataset_index = int(entry.get("dataset_index", 0) or 0)
        dataset = next(
            (
                dict(item)
                for item in prepared_run.dataset_infos or []
                if isinstance(item, Mapping)
                and int(item.get("dataset_index", -1)) == int(dataset_index)
            ),
            {"dataset_index": int(dataset_index)},
        )
        records.append(
            _geometry_fit_trace_pair_record(
                phase="acceptance_residuals",
                dataset_info=dataset,
                entry=entry,
                fit_run_id=str(fit_run_id),
                extra={
                    "accepted": bool(apply_result.accepted),
                    "rejection_reason": apply_result.rejection_reason,
                },
            )
        )
    if final_records:
        for raw_dataset in prepared_run.dataset_infos or []:
            if not isinstance(raw_dataset, Mapping):
                continue
            dataset = dict(raw_dataset)
            dataset_records = [
                dict(entry)
                for entry in final_records
                if int(entry.get("dataset_index", 0) or 0)
                == int(dataset.get("dataset_index", 0) or 0)
            ]
            records.append(
                _geometry_fit_phase_summary_record(
                    phase="acceptance_residuals",
                    dataset_info=dataset,
                    entries=dataset_records,
                    fit_run_id=str(fit_run_id),
                    dropped_pair_count=sum(
                        1
                        for entry in dataset_records
                        if str(entry.get("match_status", "") or "") != "matched"
                    ),
                )
            )
        records.extend(
            _geometry_fit_duplicate_target_records(
                final_records,
                fit_run_id=str(fit_run_id),
                phase="acceptance_residuals",
            )
        )

    return [
        _geometry_fit_cache_jsonable(record)  # type: ignore[arg-type]
        for record in records
    ]


def _write_geometry_fit_trace_file(
    *,
    fit_run_id: str,
    prepared_run: GeometryFitPreparedRun,
    result: object,
    apply_result: GeometryFitRuntimeApplyResult,
    log_path: Path,
    trace_path: Path,
) -> Path | None:
    """Persist the structured geometry-fit trace as JSONL."""

    records = _build_geometry_fit_trace_records(
        fit_run_id=str(fit_run_id),
        prepared_run=prepared_run,
        result=result,
        apply_result=apply_result,
        log_path=log_path,
    )
    if not records:
        return None
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    register_run_output_path(trace_path)
    return trace_path


_GEOMETRY_FIT_PROVIDER_IDENTITY_KEYS = (
    "normalized_hkl",
    "source_table_index",
    "resolved_table_index",
    "source_reflection_index",
    "source_row_index",
    "source_peak_index",
    "resolved_peak_index",
    "q_group_key",
    "source_q_group_key",
    "branch_group_key",
    "source_branch_index",
    "source_label",
    "label",
    "source_reflection_namespace",
    "source_reflection_is_full",
    "source_table_index_namespace",
    "resolved_table_index_namespace",
)


def _geometry_fit_source_identity_from_pair(
    *pairs: Mapping[str, object] | None,
) -> dict[str, object]:
    for pair in pairs:
        if not isinstance(pair, Mapping):
            continue
        identity = pair.get("selected_source_identity_canonical")
        if isinstance(identity, Mapping) and identity:
            return copy.deepcopy(dict(identity))
    return {}


def _geometry_fit_optimizer_point_from_pair(
    *pairs: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    for pair in pairs:
        if not isinstance(pair, Mapping):
            continue
        for key in ("solver_measured_point", "background_point"):
            raw_point = pair.get(key)
            if (
                isinstance(raw_point, Sequence)
                and not isinstance(raw_point, (str, bytes))
                and len(raw_point) >= 2
            ):
                try:
                    x_val = float(raw_point[0])
                    y_val = float(raw_point[1])
                except Exception:
                    continue
                if np.isfinite(x_val) and np.isfinite(y_val):
                    return float(x_val), float(y_val)
    return None


def _geometry_fit_optimizer_hkl_from_identity(
    identity: Mapping[str, object],
) -> tuple[int, int, int] | None:
    raw_hkl = identity.get("normalized_hkl", identity.get("hkl"))
    if (
        isinstance(raw_hkl, Sequence)
        and not isinstance(raw_hkl, (str, bytes))
        and len(raw_hkl) >= 3
    ):
        try:
            return (int(raw_hkl[0]), int(raw_hkl[1]), int(raw_hkl[2]))
        except Exception:
            return None
    return None


def _geometry_fit_optimizer_row_has_full_source(row: Mapping[str, object]) -> bool:
    try:
        reflection_index = int(row.get("source_reflection_index", -1))
    except Exception:
        reflection_index = -1
    if reflection_index < 0:
        return False
    namespace = str(row.get("source_reflection_namespace", "") or "").strip().lower()
    if namespace in {"full", "full_reflection", "miller"}:
        return True
    return bool(row.get("source_reflection_is_full", False))


def _geometry_fit_full_source_hkl_matches(
    row: Mapping[str, object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
) -> bool:
    if not _geometry_fit_optimizer_row_has_full_source(row):
        return False
    raw_hkl = row.get("hkl")
    if not (
        isinstance(raw_hkl, Sequence)
        and not isinstance(raw_hkl, (str, bytes))
        and len(raw_hkl) >= 3
    ):
        return False
    try:
        measured_hkl = (int(raw_hkl[0]), int(raw_hkl[1]), int(raw_hkl[2]))
        reflection_index = int(row.get("source_reflection_index"))
        miller = np.asarray(solver_inputs.miller)
        source_hkl = tuple(int(v) for v in miller[reflection_index][:3])
    except Exception:
        return False
    return tuple(measured_hkl) == tuple(source_hkl)


def _geometry_fit_solver_miller_hkl(
    solver_inputs: GeometryFitRuntimeSolverInputs,
    index: int | None,
) -> tuple[int, int, int] | None:
    if index is None:
        return None
    try:
        miller = np.asarray(solver_inputs.miller)
        if int(index) < 0 or int(index) >= int(miller.shape[0]):
            return None
        return tuple(int(v) for v in miller[int(index)][:3])
    except Exception:
        return None


def _geometry_fit_local_source_hkl_is_compatible(
    row: Mapping[str, object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
) -> tuple[bool, str | None]:
    hkl = _geometry_fit_normalized_hkl(row.get("hkl"))
    if hkl is None:
        return False, "missing_provider_hkl"
    for key in ("source_table_index", "resolved_table_index"):
        namespace = str(row.get(f"{key}_namespace", "") or "").strip().lower()
        if namespace not in {"full", "full_reflection", "miller", "full_hit_table"}:
            continue
        table_idx = _geometry_fit_coerce_nonnegative_index(row.get(key))
        if table_idx is None:
            continue
        source_hkl = _geometry_fit_solver_miller_hkl(solver_inputs, table_idx)
        if source_hkl is None:
            continue
        if tuple(source_hkl) != tuple(hkl):
            return False, f"{key}_hkl_mismatch"
        return True, None
    return True, None


def _geometry_fit_optimizer_row_has_local_source(row: Mapping[str, object]) -> bool:
    table_idx = _geometry_fit_coerce_nonnegative_index(row.get("source_table_index"))
    if table_idx is None:
        table_idx = _geometry_fit_coerce_nonnegative_index(row.get("resolved_table_index"))
    if table_idx is None:
        return False
    if _geometry_fit_normalized_hkl(row.get("hkl")) is None:
        return False
    branch_idx = _geometry_fit_coerce_nonnegative_index(row.get("source_branch_index"))
    peak_idx = _geometry_fit_coerce_nonnegative_index(row.get("source_peak_index"))
    row_idx = _geometry_fit_coerce_nonnegative_index(row.get("source_row_index"))
    return bool(branch_idx in {0, 1} or peak_idx in {0, 1} or row_idx is not None)


def _geometry_fit_optimizer_fixed_source_kind(
    row: Mapping[str, object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    *,
    provider_owned: bool,
) -> tuple[str | None, list[str]]:
    if _geometry_fit_optimizer_row_has_full_source(row):
        if _geometry_fit_full_source_hkl_matches(row, solver_inputs):
            return "provider_fixed_source", []
        if not _geometry_fit_optimizer_row_has_local_source(row):
            return None, ["provider_full_reflection_hkl_mismatch"]
    if not provider_owned:
        return None, ["missing_provider_source_identity"]
    if not _geometry_fit_optimizer_row_has_local_source(row):
        return None, ["missing_provider_local_source_identity"]
    compatible, reason = _geometry_fit_local_source_hkl_is_compatible(
        row,
        solver_inputs,
    )
    if not compatible:
        return None, [str(reason or "provider_local_source_hkl_mismatch")]
    return "provider_fixed_source_local", []


def _geometry_fit_optimizer_identity_matches(
    identity: Mapping[str, object],
    row: Mapping[str, object],
) -> bool:
    for key, expected in identity.items():
        if key not in _GEOMETRY_FIT_PROVIDER_IDENTITY_KEYS:
            continue
        actual = row.get(key)
        if _geometry_fit_jsonable(actual) != _geometry_fit_jsonable(expected):
            return False
    return True


def _geometry_fit_optimizer_point_matches(
    point: tuple[float, float] | None,
    row: Mapping[str, object],
) -> bool:
    if point is None:
        return False
    try:
        row_point = (float(row.get("x")), float(row.get("y")))
    except Exception:
        return False
    if not (np.isfinite(row_point[0]) and np.isfinite(row_point[1])):
        return False
    return bool(
        abs(float(point[0]) - float(row_point[0])) <= 1.0e-6
        and abs(float(point[1]) - float(row_point[1])) <= 1.0e-6
    )


def _build_geometry_fit_optimizer_request_rows(
    *,
    prepared_run: GeometryFitPreparedRun,
    solver_inputs: GeometryFitRuntimeSolverInputs,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    dataset = (
        prepared_run.current_dataset if isinstance(prepared_run.current_dataset, Mapping) else {}
    )
    provider_pairs = [
        dict(pair) for pair in dataset.get("provider_pairs", ()) or () if isinstance(pair, Mapping)
    ]
    manual_pairs = [
        dict(pair)
        for pair in dataset.get("manual_point_pairs", ()) or ()
        if isinstance(pair, Mapping)
    ]
    measured_rows = [
        copy.deepcopy(dict(row))
        for row in dataset.get("measured_for_fit", ()) or ()
        if isinstance(row, Mapping)
    ]
    initial_rows = [
        dict(row)
        for row in dataset.get("initial_pairs_display", ()) or ()
        if isinstance(row, Mapping)
    ]
    if not provider_pairs and not manual_pairs:
        return [], {}
    pair_count = max(
        len(provider_pairs),
        len(manual_pairs),
        len(measured_rows),
        len(initial_rows),
    )
    rows: list[dict[str, object]] = []
    row_reports: list[dict[str, object]] = []
    provider_identity_match = True
    provider_point_match = True
    fixed_source_pair_count = 0
    fallback_row_count = 0
    fixed_source_resolution_fallback_count = 0
    missing_fixed_source_count = 0

    for pair_index in range(pair_count):
        measured_row = measured_rows[pair_index] if pair_index < len(measured_rows) else {}
        initial_row = initial_rows[pair_index] if pair_index < len(initial_rows) else {}
        provider_pair = provider_pairs[pair_index] if pair_index < len(provider_pairs) else {}
        manual_pair = manual_pairs[pair_index] if pair_index < len(manual_pairs) else {}
        identity = _geometry_fit_source_identity_from_pair(provider_pair, manual_pair)
        point = _geometry_fit_optimizer_point_from_pair(provider_pair, manual_pair)
        row = copy.deepcopy(dict(measured_row))
        sim_display = _geometry_fit_point_list(initial_row.get("sim_display"))
        sim_native = _geometry_fit_point_list(initial_row.get("sim_native"))
        sim_native_source = str(initial_row.get("sim_native_source") or "").strip()
        if sim_display is not None:
            row["fit_prediction_detector_display_px"] = (
                float(sim_display[0]),
                float(sim_display[1]),
            )
            row["fit_prediction_detector_display_px_source"] = "sim_display"
        if sim_native is not None:
            canonical_native = (float(sim_native[0]), float(sim_native[1]))
            row["fit_prediction_detector_native_px"] = canonical_native
            if sim_native_source:
                row["fit_prediction_detector_native_px_source"] = sim_native_source
            if sim_display is not None and _geometry_fit_sim_native_source_is_display_to_native(
                sim_native_source
            ):
                row["sim_visual_detector_canonical_native_px"] = canonical_native
                row["sim_visual_detector_canonical_native_source"] = sim_native_source
        projection_theta = _geometry_fit_finite_float(
            dataset.get("theta_effective", dataset.get("theta_base"))
        )
        if projection_theta is not None:
            row["sim_visual_detector_projection_theta_initial_deg"] = float(projection_theta)
        for sim_key in (
            "sim_display",
            "sim_native",
            "sim_native_source",
            "simulated_two_theta_deg",
            "simulated_phi_deg",
        ):
            if sim_key in initial_row:
                row.setdefault(sim_key, copy.deepcopy(initial_row.get(sim_key)))
        for stale_key in (
            "fit_source_resolution_kind",
            "resolution_kind",
            "rebinding_fallback_used",
            "fallback_reason",
            "optimizer_request_fallback_row",
            "optimizer_request_fallback_reason",
            "optimizer_request_has_fixed_source",
            "resolved_table_index",
            "resolved_peak_index",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_table_index",
            "source_table_index_namespace",
            "source_row_index",
            "source_branch_index",
            "source_peak_index",
            "resolved_table_index_namespace",
        ):
            row.pop(stale_key, None)
        if identity:
            for key in _GEOMETRY_FIT_PROVIDER_IDENTITY_KEYS:
                if key in identity:
                    row[key] = copy.deepcopy(identity[key])
            hkl = _geometry_fit_optimizer_hkl_from_identity(identity)
            if hkl is not None:
                row["hkl"] = tuple(int(value) for value in hkl)
                row["label"] = str(row.get("label") or f"{hkl[0]},{hkl[1]},{hkl[2]}")
        if point is not None:
            row["x"] = float(point[0])
            row["y"] = float(point[1])
        row["optimizer_request_pair_index"] = int(pair_index)
        row["optimizer_request_source"] = "provider_pair"
        row["provider_selected_source_identity_canonical"] = copy.deepcopy(identity)
        _geometry_fit_apply_source_coverage_identity(row)

        identity_matches = bool(identity) and _geometry_fit_optimizer_identity_matches(
            identity, row
        )
        point_matches = _geometry_fit_optimizer_point_matches(point, row)
        provider_identity_match = provider_identity_match and identity_matches
        provider_point_match = provider_point_match and point_matches

        fallback_reasons: list[str] = []
        if bool(provider_pair.get("rebinding_fallback_used", False)):
            fallback_reasons.append("provider_rebinding_fallback")
        if not identity:
            fallback_reasons.append("missing_provider_source_identity")
        fixed_source_kind: str | None = None
        if identity:
            fixed_source_kind, fixed_source_reasons = _geometry_fit_optimizer_fixed_source_kind(
                row,
                solver_inputs,
                provider_owned=True,
            )
            fallback_reasons.extend(fixed_source_reasons)

        if fallback_reasons:
            row["fit_source_resolution_kind"] = "provider_fixed_source_unavailable"
            row["optimizer_request_fallback_row"] = True
            row["optimizer_request_fallback_reason"] = ",".join(fallback_reasons)
            fallback_row_count += 1
            fixed_source_resolution_fallback_count += 1
            missing_fixed_source_count += 1
        else:
            row["fit_source_resolution_kind"] = str(fixed_source_kind or "provider_fixed_source")
            row["optimizer_request_fallback_row"] = False
            row["optimizer_request_has_fixed_source"] = True
            fixed_source_pair_count += 1

        rows.append(row)
        row_reports.append(
            {
                "pair_index": int(pair_index),
                "identity_match": bool(identity_matches),
                "point_match": bool(point_matches),
                "fallback_row": bool(row.get("optimizer_request_fallback_row", False)),
                "fallback_reason": row.get("optimizer_request_fallback_reason"),
                "hkl": row.get("hkl"),
                "source_reflection_index": row.get("source_reflection_index"),
                "source_table_index": row.get("source_table_index"),
                "source_row_index": row.get("source_row_index"),
                "source_branch_index": row.get("source_branch_index"),
                "source_peak_index": row.get("source_peak_index"),
                "fit_source_resolution_kind": row.get("fit_source_resolution_kind"),
                "optimizer_request_has_fixed_source": bool(
                    row.get("optimizer_request_has_fixed_source", False)
                ),
            }
        )

    summary = {
        "provider_pair_count": int(len(provider_pairs)),
        "dataset_pair_count": int(
            dataset.get("pair_count", len(manual_pairs) or len(measured_rows)) or 0
        ),
        "optimizer_request_pair_count": int(len(rows)),
        "fixed_source_pair_count": int(fixed_source_pair_count),
        "fallback_row_count": int(fallback_row_count),
        "fixed_source_resolution_fallback_count": int(fixed_source_resolution_fallback_count),
        "missing_fixed_source_count": int(missing_fixed_source_count),
        "provider_to_optimizer_identity_match": bool(provider_identity_match),
        "provider_to_optimizer_point_match": bool(provider_point_match),
        "optimizer_request_pair_handoff": row_reports,
    }
    return rows, summary


def build_geometry_fit_solver_request(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
) -> GeometryFitSolverRequest:
    """Build one concrete solver request from a prepared geometry-fit run."""

    refinement_config, runtime_safety_note = apply_geometry_fit_runtime_safety_overrides(
        prepared_run.geometry_runtime_cfg,
    )
    refinement_config = (
        copy.deepcopy(dict(refinement_config)) if isinstance(refinement_config, Mapping) else {}
    )
    measured_peaks, request_handoff_summary = _build_geometry_fit_optimizer_request_rows(
        prepared_run=prepared_run,
        solver_inputs=solver_inputs,
    )
    if not measured_peaks:
        measured_peaks = copy.deepcopy(list(prepared_run.current_dataset["measured_for_fit"]))
    dataset_specs = [
        copy.deepcopy(dict(spec))
        for spec in prepared_run.dataset_specs
        if isinstance(spec, Mapping)
    ]
    if measured_peaks and len(dataset_specs) == 1:
        dataset_specs[0]["measured_peaks"] = copy.deepcopy(measured_peaks)
    if request_handoff_summary:
        refinement_config["optimizer_request_handoff_summary"] = request_handoff_summary

    return GeometryFitSolverRequest(
        miller=solver_inputs.miller,
        intensities=solver_inputs.intensities,
        image_size=int(solver_inputs.image_size),
        params=dict(prepared_run.fit_params),
        measured_peaks=measured_peaks,
        var_names=[str(name) for name in var_names],
        candidate_param_names=(
            [
                str(name)
                for name in (
                    refinement_config.get("candidate_param_names", [])
                    if isinstance(refinement_config, Mapping)
                    else []
                )
            ]
            or None
        ),
        dataset_specs=dataset_specs,
        refinement_config=refinement_config,
        runtime_safety_note=runtime_safety_note,
    )


def solve_geometry_fit_request(
    request: GeometryFitSolverRequest,
    *,
    solve_fit: Callable[..., object],
    status_callback: Callable[[str], None] | None = None,
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> object:
    """Invoke the live geometry-fit solver for one prepared request."""

    refinement_config = (
        copy.deepcopy(dict(request.refinement_config))
        if isinstance(request.refinement_config, Mapping)
        else {}
    )
    optimizer_cfg_raw = refinement_config.get("optimizer", None)
    solver_cfg_raw = refinement_config.get("solver", optimizer_cfg_raw)
    solver_cfg = dict(solver_cfg_raw) if isinstance(solver_cfg_raw, Mapping) else {}
    if solver_cfg.get("workers") is None:
        solver_cfg["workers"] = "auto"
    if solver_cfg.get("parallel_mode") is None:
        solver_cfg["parallel_mode"] = "auto"
    if solver_cfg.get("worker_numba_threads") is None:
        solver_cfg["worker_numba_threads"] = 0
    refinement_config["solver"] = solver_cfg
    if refinement_config.get("use_numba") is None:
        refinement_config["use_numba"] = False
    if isinstance(optimizer_cfg_raw, Mapping):
        optimizer_cfg = dict(optimizer_cfg_raw)
        if optimizer_cfg.get("workers") is None:
            optimizer_cfg["workers"] = solver_cfg["workers"]
        if optimizer_cfg.get("parallel_mode") is None:
            optimizer_cfg["parallel_mode"] = solver_cfg["parallel_mode"]
        if optimizer_cfg.get("worker_numba_threads") is None:
            optimizer_cfg["worker_numba_threads"] = solver_cfg["worker_numba_threads"]
        refinement_config["optimizer"] = optimizer_cfg

    solve_kwargs: dict[str, object] = {
        "pixel_tol": float("inf"),
        "experimental_image": None,
        "dataset_specs": request.dataset_specs,
        "refinement_config": refinement_config,
    }
    signature = None
    accepts_var_kwargs = False
    try:
        signature = inspect.signature(solve_fit)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if request.candidate_param_names is not None and (
            "candidate_param_names" in signature.parameters or accepts_var_kwargs
        ):
            solve_kwargs["candidate_param_names"] = request.candidate_param_names
    if callable(status_callback):
        if signature is not None:
            accepts_status_callback = (
                "status_callback" in signature.parameters or accepts_var_kwargs
            )
            if accepts_status_callback:
                solve_kwargs["status_callback"] = status_callback
    if callable(live_update_callback):
        if signature is not None:
            accepts_live_update_callback = (
                "live_update_callback" in signature.parameters or accepts_var_kwargs
            )
            if accepts_live_update_callback:
                solve_kwargs["live_update_callback"] = live_update_callback

    return solve_fit(
        request.miller,
        request.intensities,
        request.image_size,
        request.params,
        request.measured_peaks,
        request.var_names,
        **solve_kwargs,
    )


def apply_geometry_fit_result_values(
    var_names: Sequence[object],
    values: Sequence[object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var: object | None = None,
) -> None:
    """Apply fitted geometry values back to the live UI variables."""

    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        if not np.isfinite(value):
            continue

        param_name = str(name)
        if param_name == "theta_offset":
            if geometry_theta_offset_var is not None:
                geometry_theta_offset_var.set(f"{value:.6g}")
            continue

        var = var_map.get(param_name)
        if var is None:
            continue
        try:
            var.set(value)
        except Exception:
            continue


def _geometry_fit_normalize_hkl(raw_hkl: object) -> tuple[int, int, int] | object:
    """Normalize one HKL triplet when possible."""

    if (
        isinstance(raw_hkl, Sequence)
        and not isinstance(raw_hkl, (str, bytes))
        and len(raw_hkl) == 3
    ):
        try:
            return tuple(int(v) for v in raw_hkl)
        except Exception:
            return raw_hkl
    return raw_hkl


def build_geometry_fit_export_records(
    point_match_diagnostics: Sequence[object] | None = None,
    *,
    agg_millers: Sequence[Sequence[object]] | None = None,
    agg_sim_coords: Sequence[Sequence[object]] | None = None,
    agg_meas_coords: Sequence[Sequence[object]] | None = None,
    pixel_offsets: Sequence[Sequence[object]] | None = None,
) -> list[dict[str, object]]:
    """Build one export row per manual point from final diagnostics."""

    if (
        agg_millers is not None
        and agg_sim_coords is not None
        and agg_meas_coords is not None
        and pixel_offsets is not None
    ):
        export_recs: list[dict[str, object]] = []
        for source_label, coords in (("sim", agg_sim_coords), ("meas", agg_meas_coords)):
            for hkl, (x, y), offset in zip(agg_millers, coords, pixel_offsets):
                try:
                    hkl_triplet = tuple(int(v) for v in hkl[:3])
                except Exception:
                    continue
                try:
                    dist = float(offset[3])
                except Exception:
                    dist = float("nan")
                export_recs.append(
                    {
                        "source": str(source_label),
                        "hkl": hkl_triplet,
                        "x": int(x),
                        "y": int(y),
                        "dist_px": dist,
                    }
                )
        return export_recs

    export_recs: list[dict[str, object]] = []
    for raw_entry in point_match_diagnostics or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        resolution_reason = entry.get(
            "correspondence_resolution_reason",
            entry.get("resolution_reason"),
        )
        export_recs.append(
            {
                "dataset_index": int(entry.get("dataset_index", 0)),
                "overlay_match_index": int(entry.get("overlay_match_index", -1)),
                "match_status": str(entry.get("match_status", "")),
                "hkl": _geometry_fit_normalize_hkl(entry.get("hkl")),
                "measured_x": float(entry.get("measured_x", np.nan)),
                "measured_y": float(entry.get("measured_y", np.nan)),
                "simulated_x": float(entry.get("simulated_x", np.nan)),
                "simulated_y": float(entry.get("simulated_y", np.nan)),
                "dx_px": float(entry.get("dx_px", np.nan)),
                "dy_px": float(entry.get("dy_px", np.nan)),
                "distance_px": float(entry.get("distance_px", np.nan)),
                "source_table_index": entry.get("source_table_index"),
                "source_row_index": entry.get("source_row_index"),
                "source_peak_index": entry.get("source_peak_index"),
                "resolution_kind": str(entry.get("resolution_kind", "")),
                "resolution_reason": (
                    None if resolution_reason is None else str(resolution_reason)
                ),
            }
        )
    return export_recs


def build_geometry_fit_optimizer_diagnostics_lines(result: object) -> list[str]:
    """Format optimizer diagnostics for one geometry-fit result."""

    lines = [
        f"success={getattr(result, 'success', False)}",
        f"status={getattr(result, 'status', '')}",
        f"message={(getattr(result, 'message', '') or '').strip()}",
        f"nfev={getattr(result, 'nfev', '<unknown>')}",
        f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
        f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
        f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
        f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
        f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
        f"active_mask={list(getattr(result, 'active_mask', []))}",
    ]
    optimizer_method = str(getattr(result, "optimizer_method", "") or "").strip()
    if optimizer_method:
        lines.append(f"optimizer_method={optimizer_method}")
    weighted_rms = _geometry_fit_metric_float(getattr(result, "weighted_residual_rms_px", np.nan))
    if np.isfinite(weighted_rms):
        lines.append(f"weighted_residual_rms_px={weighted_rms:.6f}")
    display_rms = geometry_fit_result_rms(result)
    if np.isfinite(display_rms):
        lines.append(f"display_rms_px={display_rms:.6f}")
    final_metric_name = str(getattr(result, "final_metric_name", "") or "").strip()
    if final_metric_name:
        lines.append(f"final_metric_name={final_metric_name}")
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        lines.append(f"solver_discrete_mode={mode_label}")
    bound_hits = getattr(result, "bound_hits", None)
    if isinstance(bound_hits, Sequence) and not isinstance(bound_hits, (str, bytes, bytearray)):
        bound_hit_names = [str(name) for name in bound_hits if str(name).strip()]
        if bound_hit_names:
            lines.append(f"bound_hits=[{', '.join(bound_hit_names)}]")
    boundary_warning = str(getattr(result, "boundary_warning", "") or "").strip()
    if boundary_warning:
        lines.append(f"boundary_warning={boundary_warning}")
    for entry in getattr(result, "restart_history", []) or []:
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                idx=int(entry.get("restart", -1)),
                cost=float(entry.get("cost", np.nan)),
                success=bool(entry.get("success", False)),
                msg=str(entry.get("message", "")).strip(),
            )
        )
    return lines


def geometry_fit_result_rms(result: object) -> float:
    """Resolve the displayed RMS residual from one geometry-fit result."""

    try:
        direct_rms = float(getattr(result, "rms_px", np.nan))
    except Exception:
        direct_rms = float("nan")
    if np.isfinite(direct_rms):
        return float(direct_rms)

    fun = getattr(result, "fun", None)
    if fun is None:
        return 0.0
    try:
        residuals = np.asarray(fun, dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    finite_residuals = residuals[np.isfinite(residuals)]
    if finite_residuals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite_residuals**2)))


def _geometry_fit_metric_float(value: object, *, default: float = np.nan) -> float:
    """Return one finite float metric, or ``default`` when unavailable."""

    try:
        resolved = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(resolved):
        return float(default)
    return float(resolved)


def _geometry_fit_debug_value_text(
    value: object,
    *,
    float_digits: int = 6,
) -> str:
    """Format one geometry-fit debug value for log output."""

    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        if np.isposinf(numeric):
            return "inf"
        if np.isneginf(numeric):
            return "-inf"
        return f"{numeric:.{int(float_digits)}f}"
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(
                f"{key}={_geometry_fit_debug_value_text(val, float_digits=float_digits)}"
                for key, val in value.items()
            )
            + "}"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        preview = [
            _geometry_fit_debug_value_text(item, float_digits=float_digits) for item in items[:6]
        ]
        if len(items) > 6:
            preview.append("...")
        return "[" + ", ".join(preview) + "]"
    return str(value)


def build_geometry_fit_debug_lines(result: object) -> list[str]:
    """Format solver setup/progress details for one geometry-fit result."""

    debug_summary = getattr(result, "geometry_fit_debug_summary", None)
    if not isinstance(debug_summary, Mapping):
        return []

    lines: list[str] = [
        "point_match_mode={mode} datasets={datasets} vars={vars}".format(
            mode=bool(debug_summary.get("point_match_mode", False)),
            datasets=int(debug_summary.get("dataset_count", 0) or 0),
            vars=",".join(str(name) for name in debug_summary.get("var_names", []) or ()),
        )
    ]

    solver = debug_summary.get("solver", None)
    if isinstance(solver, Mapping):
        lines.append(
            "solver loss={loss} f_scale_px={f_scale} max_nfev={max_nfev} "
            "restarts={restarts} weighted_matching={weighted} "
            "missing_pair_penalty_px={missing} hk0_peak_priority_weight={hk0} "
            "measurement_uncertainty={unc} "
            "anisotropic_uncertainty={anisotropic} full_beam_polish={full_beam} "
            "full_beam_radius_px={full_beam_radius}".format(
                loss=str(solver.get("loss", "<unknown>")),
                f_scale=_geometry_fit_debug_value_text(solver.get("f_scale_px", np.nan)),
                max_nfev=_geometry_fit_debug_value_text(
                    solver.get("max_nfev", "<unknown>"), float_digits=0
                ),
                restarts=_geometry_fit_debug_value_text(
                    solver.get("restarts", "<unknown>"), float_digits=0
                ),
                weighted=_geometry_fit_debug_value_text(
                    solver.get("weighted_matching", False), float_digits=0
                ),
                missing=_geometry_fit_debug_value_text(
                    solver.get("missing_pair_penalty_px", np.nan)
                ),
                hk0=_geometry_fit_debug_value_text(solver.get("hk0_peak_priority_weight", np.nan)),
                unc=_geometry_fit_debug_value_text(
                    solver.get("use_measurement_uncertainty", False), float_digits=0
                ),
                anisotropic=_geometry_fit_debug_value_text(
                    solver.get("anisotropic_measurement_uncertainty", False),
                    float_digits=0,
                ),
                full_beam=_geometry_fit_debug_value_text(
                    solver.get("full_beam_polish_enabled", False),
                    float_digits=0,
                ),
                full_beam_radius=_geometry_fit_debug_value_text(
                    solver.get("full_beam_polish_match_radius_px", np.nan),
                ),
            )
        )

    parallel = debug_summary.get("parallelization", None)
    if isinstance(parallel, Mapping):
        lines.append(
            "parallel mode={mode} configured_workers={configured} "
            "dataset_workers={datasets} restart_workers={restarts} "
            "worker_numba_threads={threads} thread_budget={budget}".format(
                mode=str(parallel.get("mode", "<unknown>")),
                configured=_geometry_fit_debug_value_text(
                    parallel.get("configured_workers", "<unknown>"), float_digits=0
                ),
                datasets=_geometry_fit_debug_value_text(
                    parallel.get("dataset_workers", "<unknown>"), float_digits=0
                ),
                restarts=_geometry_fit_debug_value_text(
                    parallel.get("restart_workers", "<unknown>"), float_digits=0
                ),
                threads=_geometry_fit_debug_value_text(
                    parallel.get("worker_numba_threads", "None"), float_digits=0
                ),
                budget=_geometry_fit_debug_value_text(
                    parallel.get("numba_thread_budget", "<unknown>"), float_digits=0
                ),
            )
        )

    seed_search = debug_summary.get("seed_search", None)
    if isinstance(seed_search, Mapping):
        lines.append(
            "seed_search prescore_top_k={top_k} n_global={n_global} "
            "n_jitter={n_jitter} jitter_sigma_u={jitter} "
            "min_seed_separation_u={separation} "
            "trusted_prior_fraction_of_span={trusted_prior}".format(
                top_k=_geometry_fit_debug_value_text(
                    seed_search.get("prescore_top_k", 0),
                    float_digits=0,
                ),
                n_global=_geometry_fit_debug_value_text(
                    seed_search.get("n_global", 0),
                    float_digits=0,
                ),
                n_jitter=_geometry_fit_debug_value_text(
                    seed_search.get("n_jitter", 0),
                    float_digits=0,
                ),
                jitter=_geometry_fit_debug_value_text(seed_search.get("jitter_sigma_u", np.nan)),
                separation=_geometry_fit_debug_value_text(
                    seed_search.get("min_seed_separation_u", np.nan)
                ),
                trusted_prior=_geometry_fit_debug_value_text(
                    seed_search.get("trusted_prior_fraction_of_span", np.nan)
                ),
            )
        )

    discrete_modes = debug_summary.get("discrete_modes", None)
    if isinstance(discrete_modes, Sequence) and not isinstance(
        discrete_modes, (str, bytes, bytearray)
    ):
        mode_labels = [
            _geometry_fit_discrete_mode_label(mode)
            for mode in discrete_modes
            if isinstance(mode, Mapping)
        ]
        mode_labels = [label for label in mode_labels if label]
        selected_label = str(debug_summary.get("selected_mode_label", "") or "").strip()
        line = f"discrete_modes count={int(len(list(discrete_modes)))}"
        if selected_label:
            line += f" selected={selected_label}"
        if mode_labels:
            line += " labels=[{labels}]".format(labels=", ".join(mode_labels[:6]))
            if len(mode_labels) > 6:
                line += ", ..."
        lines.append(line)

    for idx, entry in enumerate(debug_summary.get("mode_seed_summary", []) or []):
        if not isinstance(entry, Mapping):
            continue
        mode_label = _geometry_fit_discrete_mode_label(entry.get("mode"))
        selected_seeds = entry.get("selected_seeds", [])
        best_seed_cost = float("nan")
        best_seed_label = ""
        if isinstance(selected_seeds, Sequence) and not isinstance(
            selected_seeds, (str, bytes, bytearray)
        ):
            for raw_seed in selected_seeds:
                if not isinstance(raw_seed, Mapping):
                    continue
                best_seed_cost = _geometry_fit_metric_float(raw_seed.get("cost", np.nan))
                best_seed_label = "{kind}:{label}".format(
                    kind=str(raw_seed.get("seed_kind", "")).strip() or "?",
                    label=str(raw_seed.get("seed_label", "")).strip() or "?",
                )
                break
        lines.append(
            "mode_seed[{idx}] label={label} total_seeds={seed_count} "
            "selected_seeds={selected} best_selected={best_label} "
            "best_selected_cost={best_cost}".format(
                idx=int(idx),
                label=mode_label or "<unknown>",
                seed_count=_geometry_fit_debug_value_text(
                    entry.get("seed_count", 0),
                    float_digits=0,
                ),
                selected=_geometry_fit_debug_value_text(
                    entry.get("selected_seed_count", 0),
                    float_digits=0,
                ),
                best_label=best_seed_label or "<none>",
                best_cost=_geometry_fit_debug_value_text(best_seed_cost),
            )
        )

    main_seed = debug_summary.get("main_solve_seed", None)
    if isinstance(main_seed, Mapping):
        lines.append(
            "main_seed kind={kind} label={label} cost={cost} weighted_rms_px={rms}".format(
                kind=str(main_seed.get("seed_kind", "") or "?"),
                label=str(main_seed.get("seed_label", "") or "?"),
                cost=_geometry_fit_debug_value_text(main_seed.get("cost", np.nan)),
                rms=_geometry_fit_debug_value_text(main_seed.get("weighted_rms_px", np.nan)),
            )
        )
        point_seed = main_seed.get("point_match_summary", None)
        if isinstance(point_seed, Mapping):
            lines.append(
                "main_seed_point_match matched={matched} missing={missing} "
                "seed_rms_px={peak_rms} peak_max_px={peak_max}".format(
                    matched=_geometry_fit_debug_value_text(
                        point_seed.get("matched_pair_count", 0), float_digits=0
                    ),
                    missing=_geometry_fit_debug_value_text(
                        point_seed.get("missing_pair_count", 0), float_digits=0
                    ),
                    peak_rms=_geometry_fit_debug_value_text(
                        point_seed.get("unweighted_peak_rms_px", np.nan)
                    ),
                    peak_max=_geometry_fit_debug_value_text(
                        point_seed.get("unweighted_peak_max_px", np.nan)
                    ),
                )
            )

    for entry in debug_summary.get("dataset_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        lines.append(
            "dataset[{idx}] label={label} theta_initial_deg={theta} measured={measured} "
            "subset_reflections={subset}/{total} fixed_source_reflections={fixed} "
            "fallback_hkls={fallback} reduced={reduced}".format(
                idx=_geometry_fit_debug_value_text(entry.get("dataset_index", -1), float_digits=0),
                label=str(entry.get("label", "")),
                theta=_geometry_fit_debug_value_text(entry.get("theta_initial_deg", np.nan)),
                measured=_geometry_fit_debug_value_text(
                    entry.get("measured_count", 0), float_digits=0
                ),
                subset=_geometry_fit_debug_value_text(
                    entry.get("subset_reflection_count", 0), float_digits=0
                ),
                total=_geometry_fit_debug_value_text(
                    entry.get("total_reflection_count", 0), float_digits=0
                ),
                fixed=_geometry_fit_debug_value_text(
                    entry.get("fixed_source_reflection_count", 0), float_digits=0
                ),
                fallback=_geometry_fit_debug_value_text(
                    entry.get("fallback_hkl_count", 0), float_digits=0
                ),
                reduced=_geometry_fit_debug_value_text(
                    entry.get("subset_reduced", False), float_digits=0
                ),
            )
        )

    for entry in debug_summary.get("parameter_entries", []) or []:
        if not isinstance(entry, Mapping):
            continue
        line = (
            "param[{name}] group={group} start={start} final={final} delta={delta} "
            "bounds=[{lower}, {upper}] scale={scale}".format(
                name=str(entry.get("name", "")),
                group=str(entry.get("group", "other")),
                start=_geometry_fit_debug_value_text(entry.get("start", np.nan)),
                final=_geometry_fit_debug_value_text(entry.get("final", np.nan)),
                delta=_geometry_fit_debug_value_text(entry.get("delta", np.nan)),
                lower=_geometry_fit_debug_value_text(entry.get("lower_bound", np.nan)),
                upper=_geometry_fit_debug_value_text(entry.get("upper_bound", np.nan)),
                scale=_geometry_fit_debug_value_text(entry.get("scale", np.nan)),
            )
        )
        if bool(entry.get("prior_enabled", False)):
            line += " prior_center={center} prior_sigma={sigma}".format(
                center=_geometry_fit_debug_value_text(entry.get("prior_center", np.nan)),
                sigma=_geometry_fit_debug_value_text(entry.get("prior_sigma", np.nan)),
            )
        lines.append(line)

    final_summary = debug_summary.get("final", None)
    if isinstance(final_summary, Mapping):
        lines.append(
            "final metric={metric} cost={cost} robust_cost={robust} "
            "weighted_rms_px={weighted_rms} final_full_beam_rms_px={display_rms}".format(
                metric=str(final_summary.get("metric_name", "") or "<unknown>"),
                cost=_geometry_fit_debug_value_text(final_summary.get("cost", np.nan)),
                robust=_geometry_fit_debug_value_text(final_summary.get("robust_cost", np.nan)),
                weighted_rms=_geometry_fit_debug_value_text(
                    final_summary.get("weighted_rms_px", np.nan)
                ),
                display_rms=_geometry_fit_debug_value_text(
                    final_summary.get(
                        "final_full_beam_rms_px",
                        final_summary.get("display_rms_px", np.nan),
                    )
                ),
            )
        )

    solve_counts = debug_summary.get("solve_counts", None)
    if isinstance(solve_counts, Mapping):
        lines.append(
            "solve_counts prescored={prescored} solved={solved}".format(
                prescored=_geometry_fit_debug_value_text(
                    solve_counts.get("prescored", 0),
                    float_digits=0,
                ),
                solved=_geometry_fit_debug_value_text(
                    solve_counts.get("solved", 0),
                    float_digits=0,
                ),
            )
        )

    solve_progress = debug_summary.get("solve_progress", None)
    if isinstance(solve_progress, Mapping):
        lines.append(
            "solve_progress label={label} evaluations={evals} best_cost={best_cost} "
            "last_cost={last_cost} best_weighted_rms_px={best_rms} "
            "last_weighted_rms_px={last_rms} status_updates={updates} "
            "aborted_early={aborted}".format(
                label=str(solve_progress.get("label", "")),
                evals=_geometry_fit_debug_value_text(
                    solve_progress.get("evaluation_count", 0), float_digits=0
                ),
                best_cost=_geometry_fit_debug_value_text(
                    solve_progress.get("best_cost_seen", np.nan)
                ),
                last_cost=_geometry_fit_debug_value_text(
                    solve_progress.get("last_cost_seen", np.nan)
                ),
                best_rms=_geometry_fit_debug_value_text(
                    solve_progress.get("best_weighted_rms_px", np.nan)
                ),
                last_rms=_geometry_fit_debug_value_text(
                    solve_progress.get("last_weighted_rms_px", np.nan)
                ),
                updates=_geometry_fit_debug_value_text(
                    solve_progress.get("status_emit_count", 0), float_digits=0
                ),
                aborted=_geometry_fit_debug_value_text(
                    solve_progress.get("aborted_early", False),
                    float_digits=0,
                ),
            )
        )
        early_stop_reason = str(solve_progress.get("early_stop_reason", "") or "").strip()
        if early_stop_reason:
            lines.append(f"solve_progress early_stop_reason={early_stop_reason}")
        for idx, event in enumerate(solve_progress.get("trace", []) or []):
            if not isinstance(event, Mapping):
                continue
            lines.append(
                "solve_progress[{idx}] eval={eval} reason={reason} cost={cost} "
                "best_cost={best_cost} weighted_rms_px={rms}".format(
                    idx=int(idx),
                    eval=_geometry_fit_debug_value_text(event.get("eval", 0), float_digits=0),
                    reason=str(event.get("reason", "")),
                    cost=_geometry_fit_debug_value_text(event.get("current_cost", np.nan)),
                    best_cost=_geometry_fit_debug_value_text(event.get("best_cost", np.nan)),
                    rms=_geometry_fit_debug_value_text(event.get("weighted_rms_px", np.nan)),
                )
            )

    return lines


def _geometry_fit_discrete_mode_label(mode: object) -> str:
    """Return one compact discrete-mode label from a mode mapping."""

    if not isinstance(mode, Mapping):
        return ""
    explicit = str(mode.get("label", "") or "").strip()
    if explicit:
        return explicit
    parts: list[str] = []
    try:
        rot90 = int(mode.get("rot90", mode.get("k", 0))) % 4
    except Exception:
        rot90 = 0
    if rot90:
        parts.append(f"rot90={rot90}")
    if bool(mode.get("flip_x", False)):
        parts.append("flip_x")
    if bool(mode.get("flip_y", False)):
        parts.append("flip_y")
    return "+".join(parts) if parts else "identity"


def build_geometry_fit_stage_summary_lines(result: object) -> list[str]:
    """Format the stage-by-stage geometry-fit workflow summaries."""

    stage_specs = [
        ("reparameterization", getattr(result, "reparameterization_summary", None)),
        ("staged_release", getattr(result, "staged_release_summary", None)),
        ("adaptive_regularization", getattr(result, "adaptive_regularization_summary", None)),
        ("full_beam_polish", getattr(result, "full_beam_polish_summary", None)),
        ("ridge_refinement", getattr(result, "ridge_refinement_summary", None)),
        ("image_refinement", getattr(result, "image_refinement_summary", None)),
        ("auto_freeze", getattr(result, "auto_freeze_summary", None)),
        ("selective_thaw", getattr(result, "selective_thaw_summary", None)),
    ]
    preferred_keys = (
        "status",
        "reason",
        "accepted",
        "success",
        "start_cost",
        "final_cost",
        "regularized_cost",
        "release_cost",
        "seed_correspondence_count",
        "matched_pair_count_before",
        "matched_pair_count_after",
        "fixed_source_resolved_count",
        "start_rms_px",
        "final_rms_px",
        "accepted_stage_count",
        "release_accepted",
        "fixed_parameters",
        "thawed_parameters",
        "applied_parameters",
        "remaining_fixed_parameters",
        "nfev",
        "stage_nfev",
        "stage_success",
        "stage_message",
        "max_nfev",
        "point_cost_before",
        "point_cost_after",
        "point_cost_limit",
        "point_rms_before_px",
        "point_rms_after_px",
        "point_rms_limit_px",
        "ridge_cost_before",
        "ridge_cost_after",
        "min_rois",
    )

    lines: list[str] = []
    for stage_name, summary in stage_specs:
        if not isinstance(summary, Mapping):
            continue
        parts = [f"{stage_name}:"]
        for key in preferred_keys:
            if key not in summary:
                continue
            value = summary.get(key)
            if value in ("", None):
                continue
            parts.append(f"{key}={_geometry_fit_debug_value_text(value)}")
        if len(parts) == 1:
            parts.append("summary=<empty>")
        matched_before = _geometry_fit_metric_float(
            summary.get("matched_pair_count_before", np.nan)
        )
        matched_after = _geometry_fit_metric_float(summary.get("matched_pair_count_after", np.nan))
        if np.isfinite(matched_before) and np.isfinite(matched_after):
            parts.append(
                "matched_pair_delta={delta}".format(
                    delta=int(round(matched_after - matched_before))
                )
            )
        start_rms = _geometry_fit_metric_float(summary.get("start_rms_px", np.nan))
        final_rms = _geometry_fit_metric_float(summary.get("final_rms_px", np.nan))
        if np.isfinite(start_rms) and np.isfinite(final_rms):
            parts.append(
                "rms_delta_px={delta}".format(
                    delta=_geometry_fit_debug_value_text(final_rms - start_rms)
                )
            )
        point_cost_before = _geometry_fit_metric_float(summary.get("point_cost_before", np.nan))
        point_cost_after = _geometry_fit_metric_float(summary.get("point_cost_after", np.nan))
        if np.isfinite(point_cost_before) and np.isfinite(point_cost_after):
            parts.append(
                "point_cost_delta={delta}".format(
                    delta=_geometry_fit_debug_value_text(point_cost_after - point_cost_before)
                )
            )
        ridge_cost_before = _geometry_fit_metric_float(summary.get("ridge_cost_before", np.nan))
        ridge_cost_after = _geometry_fit_metric_float(summary.get("ridge_cost_after", np.nan))
        if np.isfinite(ridge_cost_before) and np.isfinite(ridge_cost_after):
            parts.append(
                "ridge_cost_delta={delta}".format(
                    delta=_geometry_fit_debug_value_text(ridge_cost_after - ridge_cost_before)
                )
            )
        lines.append(" ".join(parts))
    return lines


def _geometry_fit_selected_discrete_mode_label(result: object) -> str | None:
    """Return the selected solver-side discrete mode label, if any."""

    discrete_summary = getattr(result, "discrete_mode_summary", None)
    if isinstance(discrete_summary, Mapping):
        label = str(discrete_summary.get("selected_label", "") or "").strip()
        if label:
            return label

    chosen_mode = getattr(result, "chosen_discrete_mode", None)
    label = _geometry_fit_discrete_mode_label(chosen_mode)
    return label or None


def _geometry_fit_combo_text(combo: object) -> str:
    if not isinstance(combo, Mapping):
        return "<none>"
    pieces: list[str] = []
    for name, raw_weight in combo.items():
        try:
            weight = float(raw_weight)
        except Exception:
            continue
        pieces.append(f"{name}={weight:+.3f}")
    return ", ".join(pieces) if pieces else "<none>"


def _geometry_fit_effective_orientation_choice(
    *,
    native_shape: tuple[int, int],
    orientation_choice: Mapping[str, object] | None,
    result: object | None = None,
) -> dict[str, object]:
    """Compose the GUI-selected orientation with any solver-side discrete mode."""

    base_choice = (
        dict(orientation_choice)
        if isinstance(orientation_choice, Mapping)
        else {
            "indexing_mode": "xy",
            "k": 0,
            "flip_x": False,
            "flip_y": False,
            "flip_order": "yx",
        }
    )
    chosen_mode = getattr(result, "chosen_discrete_mode", None)
    if not isinstance(chosen_mode, Mapping):
        return base_choice

    solver_mode = {
        "indexing_mode": str(chosen_mode.get("indexing_mode", "xy")),
        "k": int(chosen_mode.get("k", 0)),
        "flip_x": bool(chosen_mode.get("flip_x", False)),
        "flip_y": bool(chosen_mode.get("flip_y", False)),
        "flip_order": str(chosen_mode.get("flip_order", "yx")),
    }
    if (
        solver_mode["indexing_mode"] == "xy"
        and int(solver_mode["k"]) % 4 == 0
        and not bool(solver_mode["flip_x"])
        and not bool(solver_mode["flip_y"])
    ):
        return base_choice

    from ra_sim.gui import geometry_overlay as gui_geometry_overlay

    return gui_geometry_overlay.compose_orientation_transforms(
        native_shape,
        base_choice,
        solver_mode,
    )


def _build_geometry_fit_identifiability_scope_lines(
    label: str,
    summary: Mapping[str, object],
) -> list[str]:
    status = str(summary.get("status", "unknown"))
    lines = [
        (
            f"{label}: status={status} "
            f"rank={_geometry_fit_debug_value_text(summary.get('rank', '?'), float_digits=0)}/"
            f"{_geometry_fit_debug_value_text(summary.get('num_parameters', '?'), float_digits=0)} "
            f"residuals={_geometry_fit_debug_value_text(summary.get('residual_count', '?'), float_digits=0)} "
            f"underconstrained={_geometry_fit_debug_value_text(summary.get('underconstrained', False), float_digits=0)} "
            f"priors={_geometry_fit_debug_value_text(summary.get('includes_priors', False), float_digits=0)}"
        )
    ]

    singular_values = summary.get("singular_values", [])
    if isinstance(singular_values, Sequence) and not isinstance(
        singular_values, (str, bytes, bytearray)
    ):
        singular_array = np.asarray(list(singular_values), dtype=float).reshape(-1)
        if singular_array.size:
            preview = ", ".join(
                f"{float(value):.3e}" for value in singular_array[: min(5, singular_array.size)]
            )
            if singular_array.size > 5:
                preview += ", ..."
            lines.append(f"{label}: singular_values=[{preview}]")

    weak_parameters = summary.get("weak_parameters", [])
    if isinstance(weak_parameters, Sequence) and not isinstance(
        weak_parameters, (str, bytes, bytearray)
    ):
        names = [
            str(entry.get("name", "")).strip()
            for entry in weak_parameters
            if isinstance(entry, Mapping) and str(entry.get("name", "")).strip()
        ]
        if names:
            lines.append(f"{label}: weak_parameters={', '.join(names)}")

    weak_combinations = summary.get("weak_combinations", [])
    if isinstance(weak_combinations, Sequence) and not isinstance(
        weak_combinations, (str, bytes, bytearray)
    ):
        for idx, entry in enumerate(list(weak_combinations)[:3]):
            if not isinstance(entry, Mapping):
                continue
            lines.append(
                f"{label}: weak_combo[{idx}] "
                f"sv_rel={_geometry_fit_debug_value_text(entry.get('sv_rel', np.nan))} "
                f"combo={_geometry_fit_combo_text(entry.get('combo', {}))}"
            )

    return lines


def build_geometry_fit_identifiability_lines(result: object) -> list[str]:
    """Format solver-conditioned and data-only identifiability diagnostics."""

    lines: list[str] = []
    solver_summary = getattr(result, "identifiability_summary", None)
    if isinstance(solver_summary, Mapping):
        lines.extend(
            _build_geometry_fit_identifiability_scope_lines(
                "solver_conditioned",
                solver_summary,
            )
        )

    data_summary = getattr(result, "data_only_identifiability_summary", None)
    if isinstance(data_summary, Mapping):
        lines.extend(
            _build_geometry_fit_identifiability_scope_lines(
                "data_only",
                data_summary,
            )
        )

    recommendations = getattr(result, "next_stage_recommendations", None)
    if not isinstance(recommendations, Sequence) or isinstance(
        recommendations, (str, bytes, bytearray)
    ):
        recommendations = []
    if recommendations:
        for idx, entry in enumerate(list(recommendations)[:3]):
            if not isinstance(entry, Mapping):
                continue
            params = [str(name) for name in entry.get("params", []) if str(name).strip()]
            lines.append(
                "next_stage[{idx}] params={params} rank_gain={rank_gain} "
                "min_sv_gain={min_sv_gain} block_independence={indep} "
                "use_soft_prior={use_soft_prior} reason={reason}".format(
                    idx=int(idx),
                    params=",".join(params) if params else "<none>",
                    rank_gain=_geometry_fit_debug_value_text(
                        entry.get("rank_gain", np.nan),
                        float_digits=0,
                    ),
                    min_sv_gain=_geometry_fit_debug_value_text(entry.get("min_sv_gain", np.nan)),
                    indep=_geometry_fit_debug_value_text(entry.get("block_independence", np.nan)),
                    use_soft_prior=_geometry_fit_debug_value_text(
                        entry.get("use_soft_prior", False),
                        float_digits=0,
                    ),
                    reason=str(entry.get("reason", "")).strip(),
                )
            )
    elif isinstance(data_summary, Mapping) and bool(data_summary.get("enabled", False)):
        lines.append("next_stage: no thaw recommendation")

    return lines


def build_geometry_fit_rejection_reason_lines(
    result: object,
    *,
    rms: float,
) -> list[str]:
    """Return human-readable rejection reasons for one geometry-fit result."""

    reasons: list[str] = []
    point_match_summary = getattr(result, "point_match_summary", None)
    headless_caked_angular_acceptance = False
    if isinstance(point_match_summary, Mapping):
        try:
            manual_caked_rows = int(
                point_match_summary.get("manual_caked_residual_row_count", 0) or 0
            )
            sim_caked_rows = int(
                point_match_summary.get("sim_visual_caked_source_row_count", 0) or 0
            )
            fixed_source_rows = int(point_match_summary.get("fixed_source_resolved_count", 0) or 0)
            matched_pair_count = int(point_match_summary.get("matched_pair_count", 0) or 0)
            missing_pair_count = int(point_match_summary.get("missing_pair_count", 0) or 0)
            fallback_entry_count = int(point_match_summary.get("fallback_entry_count", 0) or 0)
            fallback_row_count = int(point_match_summary.get("fallback_row_count", 0) or 0)
        except Exception:
            manual_caked_rows = 0
            sim_caked_rows = 0
            fixed_source_rows = 0
            matched_pair_count = 0
            missing_pair_count = 0
            fallback_entry_count = 0
            fallback_row_count = 0
        headless_caked_angular_acceptance = bool(
            point_match_summary.get(
                "_headless_accept_caked_angular_metric_without_pixel_threshold",
                False,
            )
            and str(point_match_summary.get("metric_unit", "") or "").strip().lower() == "deg"
            and manual_caked_rows > 0
            and sim_caked_rows == manual_caked_rows
            and fixed_source_rows == manual_caked_rows
            and matched_pair_count == manual_caked_rows
            and missing_pair_count == 0
            and fallback_entry_count == 0
            and fallback_row_count == 0
        )

    early_stop_reason = getattr(result, "early_stop_reason", None)
    if not early_stop_reason:
        geometry_fit_progress = getattr(result, "geometry_fit_progress", None)
        if isinstance(geometry_fit_progress, Mapping):
            early_stop_reason = geometry_fit_progress.get("early_stop_reason")
    if (
        isinstance(early_stop_reason, str)
        and early_stop_reason.strip()
        and not headless_caked_angular_acceptance
    ):
        reasons.append(str(early_stop_reason).strip())

    if not bool(getattr(result, "success", True)) and not headless_caked_angular_acceptance:
        reasons.append("Optimizer did not report success.")

    if not np.isfinite(rms):
        reasons.append("RMS residual is not finite.")
    elif float(rms) > GEOMETRY_FIT_ACCEPT_MAX_RMS_PX and not headless_caked_angular_acceptance:
        reasons.append(
            "RMS residual {rms:.2f} px exceeds the acceptance limit of {limit:.2f} px.".format(
                rms=float(rms),
                limit=float(GEOMETRY_FIT_ACCEPT_MAX_RMS_PX),
            )
        )

    matched_pair_count = 0
    has_matched_pair_count = False
    max_offset = float("nan")
    if isinstance(point_match_summary, Mapping):
        if "matched_pair_count" in point_match_summary:
            has_matched_pair_count = True
            try:
                matched_pair_count = int(point_match_summary.get("matched_pair_count", 0))
            except Exception:
                matched_pair_count = 0
        max_offset = _geometry_fit_metric_float(
            point_match_summary.get("unweighted_peak_max_px", np.nan)
        )

    if has_matched_pair_count and matched_pair_count <= 0:
        reasons.append("No matched peak pairs were available for the fitted solution.")
    if (
        np.isfinite(max_offset)
        and float(max_offset) > GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX
        and not headless_caked_angular_acceptance
    ):
        reasons.append(
            "Largest matched-peak offset {offset:.2f} px exceeds the acceptance "
            "limit of {limit:.2f} px.".format(
                offset=float(max_offset),
                limit=float(GEOMETRY_FIT_ACCEPT_MAX_PEAK_OFFSET_PX),
            )
        )

    identifiability_summary = getattr(result, "identifiability_summary", None)
    underconstrained = isinstance(identifiability_summary, Mapping) and bool(
        identifiability_summary.get("underconstrained", False)
    )
    if underconstrained:
        reasons.append(
            "Fit is underconstrained according to the final identifiability diagnostics."
        )

    bound_proximity_summary = getattr(result, "bound_proximity_summary", None)
    if underconstrained and isinstance(bound_proximity_summary, Mapping):
        near_bound_entries = bound_proximity_summary.get("near_bound_parameters", [])
        if isinstance(near_bound_entries, Sequence) and near_bound_entries:
            joined = ", ".join(
                "{name}({side})".format(
                    name=str(entry.get("name", "")),
                    side=str(entry.get("side", "")),
                )
                for entry in near_bound_entries
                if isinstance(entry, Mapping) and entry.get("name")
            )
            if joined:
                reasons.append(
                    "Parameters finished within 1% of a finite bound span from a bound: "
                    + joined
                    + "."
                )

    return reasons


def build_geometry_fit_result_lines(
    var_names: Sequence[object],
    values: Sequence[object],
    *,
    rms: float,
) -> list[str]:
    """Format fitted parameter values plus the RMS residual line."""

    lines = []
    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        lines.append(f"{name} = {value:.6f}")
    lines.append(f"RMS residual = {float(rms):.6f} px")
    return lines


def build_geometry_fit_fitted_params(
    base_params: Mapping[str, object] | None,
    *,
    zb: object,
    zs: object,
    theta_initial: object,
    theta_offset: object,
    chi: object,
    cor_angle: object,
    psi_z: object,
    gamma: object,
    Gamma: object,
    corto_detector: object,
    a: object,
    c: object,
    center_x: object,
    center_y: object,
) -> dict[str, object]:
    """Merge the current fitted UI values into one simulation parameter dict."""

    fitted = dict(base_params or {})
    fitted.update(
        current_geometry_fit_ui_params(
            zb=float(zb),
            zs=float(zs),
            theta_initial=float(theta_initial),
            theta_offset=float(theta_offset),
            psi_z=float(psi_z),
            chi=float(chi),
            cor_angle=float(cor_angle),
            gamma=float(gamma),
            Gamma=float(Gamma),
            corto_detector=float(corto_detector),
            a=float(a),
            c=float(c),
            center_x=float(center_x),
            center_y=float(center_y),
        )
    )
    return fitted


def build_geometry_fit_fitted_params_from_ui(
    base_params: Mapping[str, object] | None,
    ui_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build fitted simulation params from one current UI snapshot mapping."""

    params = dict(ui_params or {})
    return build_geometry_fit_fitted_params(
        base_params,
        zb=params.get("zb", np.nan),
        zs=params.get("zs", np.nan),
        theta_initial=params.get("theta_initial", np.nan),
        theta_offset=params.get("theta_offset", 0.0),
        chi=params.get("chi", np.nan),
        cor_angle=params.get("cor_angle", np.nan),
        psi_z=params.get("psi_z", np.nan),
        gamma=params.get("gamma", np.nan),
        Gamma=params.get("Gamma", np.nan),
        corto_detector=params.get("corto_detector", np.nan),
        a=params.get("a", np.nan),
        c=params.get("c", np.nan),
        center_x=params.get("center_x", np.nan),
        center_y=params.get("center_y", np.nan),
    )


def build_geometry_fit_profile_cache(
    base_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    *,
    theta_initial: object,
    theta_offset: object,
    cor_angle: object,
    chi: object,
    zs: object,
    zb: object,
    gamma: object,
    Gamma: object,
    corto_detector: object,
    a: object,
    c: object,
    center_x: object,
    center_y: object,
) -> dict[str, object]:
    """Build the geometry-fit profile-cache payload after one successful fit."""

    profile_cache = dict(base_cache or {})
    profile_cache.update(dict(mosaic_params or {}))
    profile_cache.update(
        {
            "theta_initial": theta_initial,
            "theta_offset": theta_offset,
            "cor_angle": cor_angle,
            "chi": chi,
            "zs": zs,
            "zb": zb,
            "gamma": gamma,
            "Gamma": Gamma,
            "corto_detector": corto_detector,
            "a": a,
            "c": c,
            "center_x": center_x,
            "center_y": center_y,
        }
    )
    return profile_cache


def build_geometry_fit_profile_cache_from_ui(
    base_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    ui_params: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the post-fit profile-cache payload from one UI snapshot mapping."""

    params = dict(ui_params or {})
    return build_geometry_fit_profile_cache(
        base_cache,
        mosaic_params,
        theta_initial=params.get("theta_initial", np.nan),
        theta_offset=params.get("theta_offset", 0.0),
        cor_angle=params.get("cor_angle", np.nan),
        chi=params.get("chi", np.nan),
        zs=params.get("zs", np.nan),
        zb=params.get("zb", np.nan),
        gamma=params.get("gamma", np.nan),
        Gamma=params.get("Gamma", np.nan),
        corto_detector=params.get("corto_detector", np.nan),
        a=params.get("a", np.nan),
        c=params.get("c", np.nan),
        center_x=params.get("center_x", np.nan),
        center_y=params.get("center_y", np.nan),
    )


def build_geometry_fit_pixel_offsets(
    point_match_diagnostics: Sequence[object] | None = None,
    agg_millers: Sequence[Sequence[object]] | None = None,
    agg_sim_coords: Sequence[Sequence[object]] | None = None,
    agg_meas_coords: Sequence[Sequence[object]] | None = None,
) -> list[dict[str, object]] | list[tuple[tuple[int, int, int], float, float, float]]:
    """Build one per-point native-frame offset record from final diagnostics."""

    if (
        agg_meas_coords is None
        and point_match_diagnostics is not None
        and agg_millers is not None
        and agg_sim_coords is not None
    ):
        agg_meas_coords = agg_sim_coords
        agg_sim_coords = agg_millers
        agg_millers = point_match_diagnostics  # type: ignore[assignment]
        point_match_diagnostics = None

    if agg_millers is not None and agg_sim_coords is not None and agg_meas_coords is not None:
        legacy_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
        for hkl_key, sim_center, meas_center in zip(
            agg_millers,
            agg_sim_coords,
            agg_meas_coords,
        ):
            try:
                hkl = tuple(int(v) for v in hkl_key[:3])
                dx = float(sim_center[0]) - float(meas_center[0])
                dy = float(sim_center[1]) - float(meas_center[1])
            except Exception:
                continue
            legacy_offsets.append((hkl, dx, dy, float(np.hypot(dx, dy))))
        return legacy_offsets

    pixel_offsets: list[dict[str, object]] = []
    for raw_entry in point_match_diagnostics or ():
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        pixel_offsets.append(
            {
                "dataset_index": int(entry.get("dataset_index", 0)),
                "overlay_match_index": int(entry.get("overlay_match_index", -1)),
                "match_status": str(entry.get("match_status", "")),
                "hkl": _geometry_fit_normalize_hkl(entry.get("hkl")),
                "dx_px": float(entry.get("dx_px", np.nan)),
                "dy_px": float(entry.get("dy_px", np.nan)),
                "distance_px": float(entry.get("distance_px", np.nan)),
            }
        )
    return pixel_offsets


def filter_geometry_fit_overlay_point_match_diagnostics(
    point_match_diagnostics: object,
    *,
    joint_background_mode: bool,
    current_background_index: int,
) -> list[dict[str, object]] | object:
    """Keep only the current-background overlay diagnostics in joint mode."""

    if not (joint_background_mode and isinstance(point_match_diagnostics, list)):
        return point_match_diagnostics
    filtered: list[dict[str, object]] = []
    for entry in point_match_diagnostics:
        if not isinstance(entry, Mapping):
            continue
        if int(entry.get("dataset_index", -1)) != int(current_background_index):
            continue
        filtered.append(dict(entry))
    return filtered


def build_geometry_fit_point_match_summary_lines(
    point_match_summary: Mapping[str, object] | None,
) -> list[str]:
    """Format the optional point-match summary section."""

    if not isinstance(point_match_summary, Mapping):
        return []
    lines: list[str] = []
    try:
        fixed_source_count = int(point_match_summary.get("fixed_source_resolved_count", 0))
    except Exception:
        fixed_source_count = 0
    try:
        measured_count = int(point_match_summary.get("measured_count", 0))
    except Exception:
        measured_count = 0
    if measured_count > 0 and fixed_source_count == 0:
        lines.append(
            "WARNING: fit used only HKL-fallback correspondences; no fixed source-row anchors resolved."
        )
    lines.extend(f"{key}={value}" for key, value in sorted(point_match_summary.items()))
    return lines


def _geometry_fit_pixel_size_provenance(
    params: Mapping[str, object] | None,
) -> tuple[str, float, dict[str, float]]:
    """Resolve the pixel-size source together with the raw candidate values."""

    cfg = params if isinstance(params, Mapping) else {}

    def _value(key: str) -> float:
        try:
            return float(cfg.get(key, np.nan))
        except Exception:
            return float("nan")

    raw_values = {
        "pixel_size": _value("pixel_size"),
        "pixel_size_m": _value("pixel_size_m"),
        "debye_x": _value("debye_x"),
        "debye_y": _value("debye_y"),
        "corto_detector": _value("corto_detector"),
    }
    for key in ("pixel_size", "pixel_size_m", "debye_x"):
        value = float(raw_values.get(key, np.nan))
        if np.isfinite(value) and value > 0.0:
            return key, float(value), raw_values
    fallback = raw_values.get("corto_detector", np.nan) / 4096.0
    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0e-6
    return "corto_detector/4096", float(fallback), raw_values


def build_geometry_fit_calibration_lines(
    params: Mapping[str, object] | None,
) -> list[str]:
    """Format pixel-size provenance for debug geometry-fit logs."""

    if not isinstance(params, Mapping):
        return []
    source, pixel_size_value, raw_values = _geometry_fit_pixel_size_provenance(params)
    lines = [
        "pixel_size_source={source} value={value}".format(
            source=source,
            value=_geometry_fit_debug_value_text(pixel_size_value),
        ),
        (
            "pixel_size={pixel_size} pixel_size_m={pixel_size_m} "
            "debye_x={debye_x} debye_y={debye_y}"
        ).format(
            pixel_size=_geometry_fit_debug_value_text(raw_values.get("pixel_size")),
            pixel_size_m=_geometry_fit_debug_value_text(raw_values.get("pixel_size_m")),
            debye_x=_geometry_fit_debug_value_text(raw_values.get("debye_x")),
            debye_y=_geometry_fit_debug_value_text(raw_values.get("debye_y")),
        ),
    ]
    debye_x = float(raw_values.get("debye_x", np.nan))
    if source != "debye_x" and np.isfinite(debye_x) and debye_x <= 0.0:
        lines.append(f"warning=debye_x <= 0; using {source} instead")
    return lines


def _geometry_fit_reason_counter_lines(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    key: str,
) -> list[str]:
    """Return one count summary for a diagnostics reason/status key."""

    counts: Counter[str] = Counter()
    for item in diagnostics:
        raw_value = item.get(key)
        if raw_value in (None, ""):
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        counts[text] += 1
    if not counts:
        return []
    summary = ", ".join(
        f"{name}={count}"
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return [f"{key}: {summary}"]


def _geometry_fit_preferred_failure_reason(
    entry: Mapping[str, object],
) -> str:
    """Choose the most actionable failure-reason field for one point match."""

    for key in (
        "correspondence_resolution_reason",
        "detector_resolution_reason",
        "measured_resolution_reason",
        "resolution_reason",
    ):
        value = entry.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def _geometry_fit_overlay_match_index(
    entry: Mapping[str, object],
) -> int:
    """Return one stable overlay-match index for diagnostics output."""

    for key in ("overlay_match_index", "match_input_index"):
        try:
            return int(entry.get(key, -1))
        except Exception:
            continue
    return -1


def _geometry_fit_point_text(
    x_value: object,
    y_value: object,
    *,
    float_digits: int = 3,
) -> str:
    """Format one detector or fit-space point for debug logs."""

    try:
        x_numeric = float(x_value)
    except Exception:
        x_numeric = float("nan")
    try:
        y_numeric = float(y_value)
    except Exception:
        y_numeric = float("nan")
    if not np.isfinite(x_numeric) and not np.isfinite(y_numeric):
        return "<none>"
    return _geometry_fit_debug_value_text(
        [x_numeric, y_numeric],
        float_digits=float_digits,
    )


def build_geometry_fit_point_match_failure_reason_lines(
    point_match_diagnostics: Sequence[object] | None,
    *,
    max_unresolved_pairs_per_dataset: int = 5,
) -> list[str]:
    """Format one post-solve point-match diagnostics summary for debug logs."""

    diagnostics = [
        dict(entry) for entry in (point_match_diagnostics or ()) if isinstance(entry, Mapping)
    ]
    if not diagnostics:
        return []

    lines: list[str] = []
    for key in (
        "match_status",
        "resolution_reason",
        "measured_resolution_reason",
        "detector_resolution_reason",
        "correspondence_resolution_reason",
        "fit_source_resolution_kind",
    ):
        lines.extend(_geometry_fit_reason_counter_lines(diagnostics, key=key))

    unresolved_by_dataset: dict[tuple[int, str], list[dict[str, object]]] = {}
    dataset_order: list[tuple[int, str]] = []
    for entry in diagnostics:
        if str(entry.get("match_status", "")).strip().lower() == "matched":
            continue
        try:
            dataset_index = int(entry.get("dataset_index", 0))
        except Exception:
            dataset_index = 0
        label = str(
            entry.get(
                "dataset_label",
                f"bg[{dataset_index}]",
            )
        )
        dataset_key = (int(dataset_index), label)
        if dataset_key not in unresolved_by_dataset:
            dataset_order.append(dataset_key)
        unresolved_by_dataset.setdefault(dataset_key, []).append(entry)

    max_pairs = max(1, int(max_unresolved_pairs_per_dataset))
    for dataset_index, label in dataset_order:
        unresolved = unresolved_by_dataset.get((dataset_index, label), [])
        if not unresolved:
            continue
        lines.append(f"dataset[{dataset_index}] {label}: unresolved_pairs={len(unresolved)}")
        for entry in unresolved[:max_pairs]:
            pair_index = _geometry_fit_overlay_match_index(entry)
            lines.append(
                (
                    "dataset[{dataset_index}] {label} overlay_index={pair} "
                    "hkl={hkl} q_group={q_group} measured_detector={measured_detector} "
                    "measured_angles={measured_angles}"
                ).format(
                    dataset_index=dataset_index,
                    label=label,
                    pair=pair_index,
                    hkl=_geometry_fit_debug_value_text(
                        entry.get("hkl"),
                        float_digits=3,
                    ),
                    q_group=_geometry_fit_debug_value_text(
                        entry.get("q_group_key"),
                        float_digits=3,
                    ),
                    measured_detector=_geometry_fit_point_text(
                        entry.get("measured_x"),
                        entry.get("measured_y"),
                        float_digits=3,
                    ),
                    measured_angles=_geometry_fit_point_text(
                        entry.get("measured_two_theta_deg"),
                        entry.get("measured_phi_deg"),
                        float_digits=3,
                    ),
                )
            )
            lines.append(
                (
                    "dataset[{dataset_index}] {label} overlay_index={pair} "
                    "simulated={simulated} distance_px={distance} "
                    "delta_two_theta_deg={delta_tth} delta_phi_deg={delta_phi} "
                    "resolution_reason={reason}"
                ).format(
                    dataset_index=dataset_index,
                    label=label,
                    pair=pair_index,
                    simulated=_geometry_fit_point_text(
                        entry.get("simulated_x"),
                        entry.get("simulated_y"),
                        float_digits=3,
                    ),
                    distance=_geometry_fit_debug_value_text(
                        entry.get("distance_px"),
                        float_digits=3,
                    ),
                    delta_tth=_geometry_fit_debug_value_text(
                        entry.get("delta_two_theta_deg"),
                        float_digits=3,
                    ),
                    delta_phi=_geometry_fit_debug_value_text(
                        entry.get("delta_phi_deg"),
                        float_digits=3,
                    ),
                    reason=_geometry_fit_preferred_failure_reason(entry),
                )
            )
            lines.append(
                (
                    "dataset[{dataset_index}] {label} overlay_index={pair} "
                    "source_branch={source_branch} resolved_branch={resolved_branch} "
                    "fit_kind={fit_kind}"
                ).format(
                    dataset_index=dataset_index,
                    label=label,
                    pair=pair_index,
                    source_branch=_geometry_fit_source_branch_text(
                        entry.get("source_branch_index"),
                        entry.get("source_branch_resolution_source"),
                    ),
                    resolved_branch=_geometry_fit_source_branch_text(
                        entry.get("resolved_peak_index"),
                        "resolved_peak_index"
                        if entry.get("resolved_peak_index") is not None
                        else None,
                    ),
                    fit_kind=str(entry.get("fit_source_resolution_kind", "<none>") or "<none>"),
                )
            )
        extra_count = len(unresolved) - max_pairs
        if extra_count > 0:
            lines.append(
                "dataset[{dataset_index}] {label}: ... {extra_count} more unresolved "
                "pair(s) not shown".format(
                    dataset_index=dataset_index,
                    label=label,
                    extra_count=int(extra_count),
                )
            )

    return lines


def build_geometry_fit_overlay_diagnostic_lines(
    frame_diag: Mapping[str, object] | None,
    *,
    overlay_record_count: int,
    result: object | None = None,
) -> list[str]:
    """Format overlay frame diagnostics for the geometry-fit log."""

    diag = frame_diag if isinstance(frame_diag, Mapping) else {}
    lines = [
        "transform_rule=sim:native_to_overlay_display; bg:inverse_orientation_then_overlay_display",
        f"overlay_records={int(overlay_record_count)}",
        f"paired_records={int(diag.get('paired_records', 0))}",
        f"sim_display_med_px={float(diag.get('sim_display_med_px', np.nan)):.3f}",
        f"bg_display_med_px={float(diag.get('bg_display_med_px', np.nan)):.3f}",
        f"sim_display_p90_px={float(diag.get('sim_display_p90_px', np.nan)):.3f}",
        f"bg_display_p90_px={float(diag.get('bg_display_p90_px', np.nan)):.3f}",
    ]
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        lines.insert(1, f"solver_discrete_mode={mode_label}")
    return lines


def count_geometry_fit_matched_overlay_records(
    overlay_records: Sequence[Mapping[str, object]] | None,
) -> int:
    """Return the number of overlay records that represent matched fitted points."""

    matched_count = 0
    for entry in overlay_records or ():
        if not isinstance(entry, Mapping):
            continue
        status = str(entry.get("match_status", "")).strip().lower()
        if status:
            if status == "matched":
                matched_count += 1
            continue
        matched_count += 1
    return int(matched_count)


def build_geometry_fit_pixel_offset_lines(
    pixel_offsets: Sequence[Sequence[object]],
) -> list[str]:
    """Format one pixel-offset section for the geometry-fit log/status."""

    lines = []
    for entry in pixel_offsets:
        if isinstance(entry, Mapping):
            hkl = _geometry_fit_normalize_hkl(entry.get("hkl"))
            dataset_index = int(entry.get("dataset_index", 0))
            overlay_match_index = int(entry.get("overlay_match_index", -1))
            match_status = str(entry.get("match_status", ""))
            dx = float(entry.get("dx_px", np.nan))
            dy = float(entry.get("dy_px", np.nan))
            dist = float(entry.get("distance_px", np.nan))
            prefix = f"dataset={dataset_index} idx={overlay_match_index} HKL={hkl}"
            if match_status.lower() != "matched" or not np.isfinite(dist):
                lines.append(f"{prefix}: status={match_status or 'unknown'}")
                continue
            lines.append(f"{prefix}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px")
            continue
        if not isinstance(entry, Mapping):
            try:
                hkl = tuple(int(v) for v in entry[0][:3])
                dx = float(entry[1])
                dy = float(entry[2])
                dist = float(entry[3])
            except Exception:
                continue
            lines.append(f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px")
    return lines or ["No matched peaks"]


def build_geometry_fit_summary_lines(
    *,
    current_dataset: Mapping[str, object],
    overlay_record_count: int,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    save_path: object,
    result: object | None = None,
) -> list[str]:
    """Format the fit-summary section written to the geometry-fit log."""

    lines = [
        f"manual_groups={int(current_dataset.get('group_count', 0) or 0)}",
        f"manual_points={int(current_dataset.get('pair_count', 0) or 0)}",
        f"overlay_records={int(overlay_record_count)}",
        "orientation={orientation}".format(
            orientation=str(
                (current_dataset.get("orientation_choice") or {}).get(
                    "label",
                    "identity",
                )
            )
        ),
    ]
    if "resolved_source_pair_count" in current_dataset:
        lines.append(
            "resolved_source_pairs={count}".format(
                count=int(current_dataset.get("resolved_source_pair_count", 0) or 0)
            )
        )
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        lines.append(f"solver_discrete_mode={mode_label}")
    final_metric_name = str(getattr(result, "final_metric_name", "") or "").strip()
    if final_metric_name:
        lines.append(f"final_metric={final_metric_name}")
    point_match_summary = getattr(result, "point_match_summary", None)
    if isinstance(point_match_summary, Mapping):
        if "matched_pair_count" in point_match_summary:
            lines.append(
                "matched_pairs={count}".format(
                    count=int(point_match_summary.get("matched_pair_count", 0) or 0)
                )
            )
        if "fixed_source_resolved_count" in point_match_summary:
            lines.append(
                "fixed_source_resolved={count}".format(
                    count=int(point_match_summary.get("fixed_source_resolved_count", 0) or 0)
                )
            )
        peak_rms = _geometry_fit_metric_float(
            point_match_summary.get("unweighted_peak_rms_px", np.nan)
        )
        if np.isfinite(peak_rms):
            lines.append(f"matched_peak_rms_px={peak_rms:.6f}")
        peak_max = _geometry_fit_metric_float(
            point_match_summary.get("unweighted_peak_max_px", np.nan)
        )
        if np.isfinite(peak_max):
            lines.append(f"matched_peak_max_px={peak_max:.6f}")
    identifiability_summary = getattr(result, "identifiability_summary", None)
    if isinstance(identifiability_summary, Mapping):
        lines.append(
            "underconstrained={flag}".format(
                flag=bool(identifiability_summary.get("underconstrained", False))
            )
        )
    boundary_warning = str(getattr(result, "boundary_warning", "") or "").strip()
    if boundary_warning:
        lines.append(f"boundary_warning={boundary_warning}")
    next_stage_recommendation = getattr(result, "next_stage_recommendation", None)
    if isinstance(next_stage_recommendation, Mapping):
        next_params = [
            str(name) for name in next_stage_recommendation.get("params", []) if str(name).strip()
        ]
        if next_params:
            lines.append("next_stage_params={params}".format(params=",".join(next_params)))
    lines.extend(
        build_geometry_fit_result_lines(
            var_names,
            values,
            rms=rms,
        )[:-1]
    )
    lines.append(f"RMS residual = {float(rms):.6f} px")
    lines.append(f"Matched peaks saved to: {save_path}")
    return lines


def build_geometry_fit_progress_text(
    *,
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    pixel_offsets: Sequence[Sequence[object]],
    export_record_count: int,
    save_path: object,
    log_path: object,
    frame_warning: str | None,
    result: object | None = None,
) -> str:
    """Build the final geometry-fit status text shown in the GUI."""

    base_summary_lines = ["Manual geometry fit complete:"]
    for name, raw_value in zip(var_names, values):
        try:
            value = float(raw_value)
        except Exception:
            continue
        base_summary_lines.append(f"{name} = {value:.4f}")
    base_summary_lines.append(f"RMS residual = {float(rms):.2f} px")
    base_summary_lines.append(
        "Orientation = {orientation}".format(
            orientation=str(
                (current_dataset.get("orientation_choice") or {}).get(
                    "label",
                    "identity",
                )
            )
        )
    )
    mode_label = _geometry_fit_selected_discrete_mode_label(result)
    if mode_label:
        base_summary_lines.append(f"Solver discrete mode = {mode_label}")
    base_summary = "\n".join(base_summary_lines)
    overlay_hint = (
        "Overlay: blue squares=selected simulated points, amber triangles=saved "
        "background points, green circles=fitted simulated peaks, dashed "
        "arrows=initial->fitted sim shifts."
    )
    dist_report_lines = build_geometry_fit_pixel_offset_lines(pixel_offsets)
    dist_report = "\n".join(dist_report_lines)
    return (
        f"{base_summary}\nManual pairs: {{points}} points across {{groups}} groups".format(
            points=int(current_dataset.get("pair_count", 0) or 0),
            groups=int(current_dataset.get("group_count", 0) or 0),
        )
        + (f" | joint backgrounds={int(dataset_count)}" if joint_background_mode else "")
        + "\n"
        + overlay_hint
        + (f"\n{frame_warning}" if frame_warning else "")
        + f"\nSaved {int(export_record_count)} peak records → {save_path}"
        + f"\nPixel offsets:\n{dist_report}"
        + f"\nFit log → {log_path}"
    )


def build_geometry_fit_rejected_progress_text(
    *,
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    rms: float,
    rejection_reasons: Sequence[object],
) -> str:
    """Build the GUI status text for one rejected manual geometry fit."""

    lines = ["Manual geometry fit rejected:"]
    for reason in rejection_reasons:
        text = str(reason).strip()
        if text:
            lines.append(text)
    if np.isfinite(rms):
        lines.append(f"RMS residual = {float(rms):.2f} px")
    lines.append(
        "Manual pairs: {points} points across {groups} groups".format(
            points=int(current_dataset.get("pair_count", 0) or 0),
            groups=int(current_dataset.get("group_count", 0) or 0),
        )
        + (f" | joint backgrounds={int(dataset_count)}" if joint_background_mode else "")
    )
    lines.append("Add more manual points or remove outliers before rerunning the fit.")
    return "\n".join(lines)


def postprocess_geometry_fit_result(
    *,
    fitted_params: Mapping[str, object],
    result: object,
    current_dataset: Mapping[str, object],
    joint_background_mode: bool,
    current_background_index: int,
    dataset_count: int,
    var_names: Sequence[object],
    values: Sequence[object],
    rms: float,
    miller: object,
    intensities: object,
    image_size: int,
    max_display_markers: int,
    downloads_dir: Path | str,
    stamp: str,
    log_path: Path | str,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
) -> GeometryFitPostprocessResult:
    """Build the post-fit analysis artifacts used by the live GUI."""

    point_match_diagnostics = getattr(result, "point_match_diagnostics", None)
    if not isinstance(point_match_diagnostics, list):
        point_match_diagnostics = []
    pixel_offsets = build_geometry_fit_pixel_offsets(point_match_diagnostics)

    overlay_point_match_diagnostics = filter_geometry_fit_overlay_point_match_diagnostics(
        point_match_diagnostics,
        joint_background_mode=joint_background_mode,
        current_background_index=int(current_background_index),
    )
    effective_orientation_choice = _geometry_fit_effective_orientation_choice(
        native_shape=tuple(int(v) for v in current_dataset["native_background"].shape[:2]),
        orientation_choice=(
            current_dataset["orientation_choice"] if isinstance(current_dataset, Mapping) else None
        ),
        result=result,
    )
    overlay_records = build_overlay_records(
        current_dataset["initial_pairs_display"],
        overlay_point_match_diagnostics,
        native_shape=tuple(int(v) for v in current_dataset["native_background"].shape[:2]),
        orientation_choice=effective_orientation_choice,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
    )
    matched_overlay_record_count = count_geometry_fit_matched_overlay_records(overlay_records)
    frame_diag, frame_warning = compute_frame_diagnostics(overlay_records)
    overlay_diagnostic_lines = build_geometry_fit_overlay_diagnostic_lines(
        frame_diag,
        overlay_record_count=int(matched_overlay_record_count),
        result=result,
    )

    export_records = build_geometry_fit_export_records(point_match_diagnostics)
    save_path = Path(downloads_dir) / f"matched_peaks_{stamp}.npy"
    register_run_output_path(save_path)
    fit_summary_lines = build_geometry_fit_summary_lines(
        current_dataset=current_dataset,
        overlay_record_count=int(matched_overlay_record_count),
        var_names=var_names,
        values=values,
        rms=rms,
        save_path=save_path,
        result=result,
    )
    progress_text = build_geometry_fit_progress_text(
        current_dataset=current_dataset,
        dataset_count=int(dataset_count),
        joint_background_mode=joint_background_mode,
        var_names=var_names,
        values=values,
        rms=rms,
        pixel_offsets=pixel_offsets,
        export_record_count=len(export_records),
        save_path=save_path,
        log_path=log_path,
        frame_warning=frame_warning,
        result=result,
    )
    overlay_state = {
        "overlay_records": copy_geometry_fit_state_value(overlay_records),
        "initial_pairs_display": copy_geometry_fit_state_value(
            current_dataset["initial_pairs_display"]
        ),
        "max_display_markers": int(max_display_markers),
    }

    return GeometryFitPostprocessResult(
        fitted_params=fitted_params,
        point_match_summary_lines=build_geometry_fit_point_match_summary_lines(
            getattr(result, "point_match_summary", None)
        ),
        pixel_offsets=pixel_offsets,
        overlay_records=overlay_records,
        overlay_state=overlay_state,
        overlay_diagnostic_lines=overlay_diagnostic_lines,
        frame_warning=frame_warning,
        export_records=export_records,
        save_path=save_path,
        fit_summary_lines=fit_summary_lines,
        progress_text=progress_text,
    )


def apply_runtime_geometry_fit_result(
    *,
    result: object,
    var_names: Sequence[object],
    current_dataset: Mapping[str, object],
    dataset_count: int,
    joint_background_mode: bool,
    preserve_live_theta: bool,
    max_display_markers: int,
    bindings: GeometryFitRuntimeResultBindings,
) -> GeometryFitRuntimeApplyResult:
    """Apply one successful geometry-fit result through runtime callbacks."""

    debug_logging = geometry_fit_debug_logging_enabled(bindings.geometry_runtime_cfg)
    point_match_summary = getattr(result, "point_match_summary", None)
    has_fit_space_summary = bool(
        isinstance(point_match_summary, Mapping)
        and any(str(key).startswith("fit_space_") for key in point_match_summary.keys())
    )
    bindings.log_section(
        "Optimizer diagnostics:",
        build_geometry_fit_optimizer_diagnostics_lines(result),
    )
    debug_lines = build_geometry_fit_debug_lines(result)
    if debug_lines:
        bindings.log_section(
            "Fit mechanics:",
            debug_lines,
        )
    stage_summary_lines = build_geometry_fit_stage_summary_lines(result)
    if stage_summary_lines:
        bindings.log_section(
            "Solver stages:",
            stage_summary_lines,
        )
    identifiability_lines = build_geometry_fit_identifiability_lines(result)
    if identifiability_lines:
        bindings.log_section(
            "Identifiability diagnostics:",
            identifiability_lines,
        )

    result_vector = getattr(result, "x", None)
    result_values = [] if result_vector is None else list(result_vector)
    preview_fitted_params = (
        bindings.preview_fitted_params(var_names, result_values)
        if callable(bindings.preview_fitted_params)
        else None
    )
    rms = geometry_fit_result_rms(result)
    rejection_reasons = build_geometry_fit_rejection_reason_lines(
        result,
        rms=rms,
    )
    if rejection_reasons:
        bindings.log_section(
            "Optimization result:",
            build_geometry_fit_result_lines(
                var_names,
                result_values,
                rms=rms,
            ),
        )
        point_match_summary_lines = build_geometry_fit_point_match_summary_lines(
            getattr(result, "point_match_summary", None)
        )
        if point_match_summary_lines:
            bindings.log_section(
                "Point-match summary:",
                point_match_summary_lines,
            )
        if debug_logging:
            diagnostic_lines = build_geometry_fit_point_match_failure_reason_lines(
                getattr(result, "point_match_diagnostics", None)
            )
            if diagnostic_lines:
                bindings.log_section(
                    "Point-match diagnostics:",
                    diagnostic_lines,
                )
            if has_fit_space_summary:
                calibration_lines = build_geometry_fit_calibration_lines(preview_fitted_params)
                if calibration_lines:
                    bindings.log_section(
                        "Fit-space calibration:",
                        calibration_lines,
                    )
        bindings.log_section("Fit rejected:", rejection_reasons)
        bindings.set_progress_text(
            build_geometry_fit_rejected_progress_text(
                current_dataset=current_dataset,
                dataset_count=dataset_count,
                joint_background_mode=joint_background_mode,
                rms=rms,
                rejection_reasons=rejection_reasons,
            )
        )
        bindings.cmd_line(
            "rejected: "
            f"datasets={int(dataset_count)} "
            f"groups={int(current_dataset.get('group_count', 0) or 0)} "
            f"points={int(current_dataset.get('pair_count', 0) or 0)} "
            f"rms={float(rms):.4f}px "
            f"reason={rejection_reasons[0]}"
        )
        return GeometryFitRuntimeApplyResult(
            accepted=False,
            rejection_reason=" ".join(str(reason) for reason in rejection_reasons),
            rms=float(rms),
            fitted_params=None,
            postprocess=None,
        )

    undo_state = bindings.capture_undo_state()
    bindings.apply_result_values(var_names, result_values)

    if joint_background_mode and not preserve_live_theta:
        sync_theta = bindings.sync_joint_background_theta
        if sync_theta is not None:
            sync_theta()

    bindings.refresh_status()
    bindings.update_manual_pick_button_label()
    bindings.replace_profile_cache(bindings.build_profile_cache())
    bindings.push_undo_state(undo_state)
    bindings.request_preview_skip_once()
    bindings.mark_last_simulation_dirty()
    bindings.schedule_update()

    bindings.log_section(
        "Optimization result:",
        build_geometry_fit_result_lines(
            var_names,
            result_values,
            rms=rms,
        ),
    )

    fitted_params = bindings.build_fitted_params()
    postprocess = bindings.postprocess_result(fitted_params, rms)

    if postprocess.point_match_summary_lines:
        bindings.log_section(
            "Point-match summary:",
            postprocess.point_match_summary_lines,
        )
    if debug_logging:
        if has_fit_space_summary:
            calibration_lines = build_geometry_fit_calibration_lines(
                preview_fitted_params or fitted_params
            )
            if calibration_lines:
                bindings.log_section(
                    "Fit-space calibration:",
                    calibration_lines,
                )
        diagnostic_lines = build_geometry_fit_point_match_failure_reason_lines(
            getattr(result, "point_match_diagnostics", None)
        )
        if diagnostic_lines:
            bindings.log_section(
                "Point-match diagnostics:",
                diagnostic_lines,
            )

    bindings.log_section(
        "Overlay frame diagnostics:",
        postprocess.overlay_diagnostic_lines,
    )

    if postprocess.overlay_records:
        bindings.draw_overlay_records(
            postprocess.overlay_records,
            int(max_display_markers),
        )
    else:
        bindings.draw_initial_pairs_overlay(
            current_dataset["initial_pairs_display"],
            int(max_display_markers),
        )

    bindings.set_last_overlay_state(postprocess.overlay_state)
    bindings.save_export_records(postprocess.save_path, postprocess.export_records)
    bindings.log_section(
        "Pixel offsets (native frame):",
        build_geometry_fit_pixel_offset_lines(postprocess.pixel_offsets),
    )
    bindings.log_section(
        "Fit summary:",
        postprocess.fit_summary_lines,
    )
    bindings.set_progress_text(postprocess.progress_text)
    bindings.cmd_line(
        "done: "
        f"datasets={int(dataset_count)} "
        f"groups={int(current_dataset.get('group_count', 0) or 0)} "
        f"points={int(current_dataset.get('pair_count', 0) or 0)} "
        f"rms={float(rms):.4f}px"
    )

    return GeometryFitRuntimeApplyResult(
        accepted=True,
        rejection_reason=None,
        rms=float(rms),
        fitted_params=fitted_params,
        postprocess=postprocess,
    )


def build_runtime_geometry_fit_result_bindings(
    *,
    fit_params: Mapping[str, object] | None,
    base_profile_cache: Mapping[str, object] | None,
    mosaic_params: Mapping[str, object] | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    geometry_runtime_cfg: Mapping[str, object] | None = None,
    geometry_theta_offset_var=None,
    log_section: Callable[[str, Sequence[str]], None],
    capture_undo_state: Callable[[], dict[str, object]],
    sync_joint_background_theta: Callable[[], None] | None,
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    replace_profile_cache: Callable[[dict[str, object]], None],
    push_undo_state: Callable[[dict[str, object] | None], None],
    request_preview_skip_once: Callable[[], None],
    mark_last_simulation_dirty: Callable[[], None],
    schedule_update: Callable[[], None],
    postprocess_result: Callable[[dict[str, object], float], GeometryFitPostprocessResult],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    save_export_records: Callable[[Path, Sequence[dict[str, object]]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
) -> GeometryFitRuntimeResultBindings:
    """Build the runtime success-path callback bundle for one geometry fit."""

    def _snapshot_ui_params() -> dict[str, object]:
        try:
            params = current_ui_params()
        except Exception:
            params = {}
        return dict(params) if isinstance(params, Mapping) else {}

    def _preview_fitted_params(
        names: Sequence[object],
        values: Sequence[object],
    ) -> dict[str, object]:
        ui_params = _snapshot_ui_params()
        for name, value in zip(names, values):
            key = str(name)
            try:
                ui_params[key] = float(value)
            except Exception:
                ui_params[key] = value
        return build_geometry_fit_fitted_params_from_ui(
            fit_params,
            ui_params,
        )

    return GeometryFitRuntimeResultBindings(
        log_section=log_section,
        capture_undo_state=capture_undo_state,
        apply_result_values=(
            lambda names, values: apply_geometry_fit_result_values(
                names,
                values,
                var_map=var_map,
                geometry_theta_offset_var=geometry_theta_offset_var,
            )
        ),
        sync_joint_background_theta=sync_joint_background_theta,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        build_profile_cache=(
            lambda: build_geometry_fit_profile_cache_from_ui(
                base_profile_cache,
                mosaic_params,
                _snapshot_ui_params(),
            )
        ),
        replace_profile_cache=replace_profile_cache,
        push_undo_state=push_undo_state,
        request_preview_skip_once=request_preview_skip_once,
        mark_last_simulation_dirty=mark_last_simulation_dirty,
        schedule_update=schedule_update,
        build_fitted_params=(
            lambda: build_geometry_fit_fitted_params_from_ui(
                fit_params,
                _snapshot_ui_params(),
            )
        ),
        postprocess_result=postprocess_result,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        save_export_records=save_export_records,
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
        geometry_runtime_cfg=(
            dict(geometry_runtime_cfg) if isinstance(geometry_runtime_cfg, Mapping) else None
        ),
        preview_fitted_params=_preview_fitted_params,
    )


def build_runtime_geometry_fit_execution_result_bindings(
    *,
    result: object,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    ui_bindings: GeometryFitRuntimeUiBindings,
    postprocess_config: GeometryFitRuntimePostprocessConfig,
    log_section: Callable[[str, Sequence[str]], None],
) -> GeometryFitRuntimeResultBindings:
    """Build the runtime apply-bindings for one executed geometry-fit result."""

    solver_inputs = postprocess_config.solver_inputs
    return build_runtime_geometry_fit_result_bindings(
        fit_params=ui_bindings.fit_params,
        base_profile_cache=ui_bindings.base_profile_cache,
        mosaic_params=ui_bindings.mosaic_params,
        current_ui_params=ui_bindings.current_ui_params,
        var_map=ui_bindings.var_map,
        geometry_runtime_cfg=prepared_run.geometry_runtime_cfg,
        geometry_theta_offset_var=ui_bindings.geometry_theta_offset_var,
        log_section=log_section,
        capture_undo_state=ui_bindings.capture_undo_state,
        sync_joint_background_theta=ui_bindings.sync_joint_background_theta,
        refresh_status=ui_bindings.refresh_status,
        update_manual_pick_button_label=ui_bindings.update_manual_pick_button_label,
        replace_profile_cache=ui_bindings.replace_profile_cache,
        push_undo_state=ui_bindings.push_undo_state,
        request_preview_skip_once=ui_bindings.request_preview_skip_once,
        mark_last_simulation_dirty=ui_bindings.mark_last_simulation_dirty,
        schedule_update=ui_bindings.schedule_update,
        postprocess_result=(
            lambda fitted_params, rms: postprocess_geometry_fit_result(
                fitted_params=fitted_params,
                result=result,
                current_dataset=prepared_run.current_dataset,
                joint_background_mode=prepared_run.joint_background_mode,
                current_background_index=int(postprocess_config.current_background_index),
                dataset_count=len(prepared_run.dataset_infos),
                var_names=var_names,
                values=getattr(result, "x", []),
                rms=rms,
                miller=solver_inputs.miller,
                intensities=solver_inputs.intensities,
                image_size=int(solver_inputs.image_size),
                max_display_markers=int(prepared_run.max_display_markers),
                downloads_dir=postprocess_config.downloads_dir,
                stamp=postprocess_config.stamp,
                log_path=postprocess_config.log_path,
                sim_display_rotate_k=int(postprocess_config.sim_display_rotate_k),
                background_display_rotate_k=int(postprocess_config.background_display_rotate_k),
                simulate_and_compare_hkl=postprocess_config.simulate_and_compare_hkl,
                aggregate_match_centers=postprocess_config.aggregate_match_centers,
                build_overlay_records=postprocess_config.build_overlay_records,
                compute_frame_diagnostics=postprocess_config.compute_frame_diagnostics,
            )
        ),
        draw_overlay_records=ui_bindings.draw_overlay_records,
        draw_initial_pairs_overlay=ui_bindings.draw_initial_pairs_overlay,
        set_last_overlay_state=ui_bindings.set_last_overlay_state,
        save_export_records=ui_bindings.save_export_records,
        set_progress_text=ui_bindings.set_progress_text,
        cmd_line=ui_bindings.cmd_line,
    )


def build_runtime_geometry_fit_execution_setup(
    *,
    prepared_run: GeometryFitPreparedRun,
    mosaic_params: Mapping[str, object] | None,
    stamp: str,
    downloads_dir: Path | str,
    log_dir: Path | str | None = None,
    simulation_runtime_state: Any,
    background_runtime_state: Any,
    theta_initial_var: Any,
    geometry_theta_offset_var: Any | None,
    current_ui_params: Callable[[], Mapping[str, object]],
    var_map: Mapping[str, object],
    background_theta_for_index: Callable[..., object],
    refresh_status: Callable[[], None],
    update_manual_pick_button_label: Callable[[], None],
    capture_undo_state: Callable[[], dict[str, object]],
    push_undo_state: Callable[[dict[str, object] | None], None],
    replace_dataset_cache: Callable[[dict[str, object] | None], None] | None,
    request_preview_skip_once: Callable[[], None],
    schedule_update: Callable[[], None],
    draw_overlay_records: Callable[[Sequence[dict[str, object]], int], None],
    draw_initial_pairs_overlay: Callable[[Sequence[dict[str, object]], int], None],
    set_last_overlay_state: Callable[[dict[str, object]], None],
    set_progress_text: Callable[[str], None],
    cmd_line: Callable[[str], None],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    sim_display_rotate_k: int,
    background_display_rotate_k: int,
    simulate_and_compare_hkl: Callable[..., Any],
    aggregate_match_centers: Callable[..., tuple[object, object, object]],
    build_overlay_records: Callable[..., list[dict[str, object]]],
    compute_frame_diagnostics: Callable[..., tuple[Mapping[str, object], str | None]],
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> GeometryFitRuntimeExecutionSetup:
    """Build the runtime execution setup for one prepared geometry-fit run."""

    def _sync_joint_background_theta() -> None:
        background_index = int(getattr(background_runtime_state, "current_background_index", 0))
        if 0 <= background_index < len(prepared_run.background_theta_values):
            theta_initial_var.set(float(prepared_run.background_theta_values[background_index]))
            return
        theta_initial_var.set(
            background_theta_for_index(
                background_index,
                strict_count=False,
            )
        )

    ui_bindings = GeometryFitRuntimeUiBindings(
        fit_params=prepared_run.fit_params,
        base_profile_cache=getattr(simulation_runtime_state, "profile_cache", {}),
        mosaic_params=mosaic_params,
        current_ui_params=current_ui_params,
        var_map=var_map,
        geometry_theta_offset_var=geometry_theta_offset_var,
        capture_undo_state=capture_undo_state,
        sync_joint_background_theta=_sync_joint_background_theta,
        refresh_status=refresh_status,
        update_manual_pick_button_label=update_manual_pick_button_label,
        replace_profile_cache=(
            lambda profile_cache: setattr(
                simulation_runtime_state,
                "profile_cache",
                dict(profile_cache),
            )
        ),
        replace_dataset_cache=replace_dataset_cache,
        push_undo_state=push_undo_state,
        request_preview_skip_once=request_preview_skip_once,
        mark_last_simulation_dirty=(
            lambda: setattr(
                simulation_runtime_state,
                "last_simulation_signature",
                None,
            )
        ),
        schedule_update=schedule_update,
        draw_overlay_records=draw_overlay_records,
        draw_initial_pairs_overlay=draw_initial_pairs_overlay,
        set_last_overlay_state=set_last_overlay_state,
        save_export_records=(
            lambda save_path, export_records: np.save(
                save_path,
                np.array(export_records, dtype=object),
                allow_pickle=True,
            )
        ),
        set_progress_text=set_progress_text,
        cmd_line=cmd_line,
        live_update_callback=live_update_callback,
    )

    postprocess_config = GeometryFitRuntimePostprocessConfig(
        current_background_index=int(
            getattr(background_runtime_state, "current_background_index", 0)
        ),
        downloads_dir=downloads_dir,
        stamp=stamp,
        log_path=build_geometry_fit_log_path(
            stamp=stamp,
            log_dir=log_dir,
            downloads_dir=downloads_dir,
        ),
        solver_inputs=solver_inputs,
        sim_display_rotate_k=int(sim_display_rotate_k),
        background_display_rotate_k=int(background_display_rotate_k),
        simulate_and_compare_hkl=simulate_and_compare_hkl,
        aggregate_match_centers=aggregate_match_centers,
        build_overlay_records=build_overlay_records,
        compute_frame_diagnostics=compute_frame_diagnostics,
        log_dir=log_dir,
    )

    return GeometryFitRuntimeExecutionSetup(
        ui_bindings=ui_bindings,
        postprocess_config=postprocess_config,
    )


def build_runtime_geometry_fit_execution_setup_from_bindings(
    *,
    prepared_run: GeometryFitPreparedRun,
    mosaic_params: Mapping[str, object] | None,
    stamp: str,
    bindings: GeometryFitRuntimeActionExecutionBindings,
) -> GeometryFitRuntimeExecutionSetup:
    """Build one runtime execution setup from a bound action bundle."""

    return build_runtime_geometry_fit_execution_setup(
        prepared_run=prepared_run,
        mosaic_params=mosaic_params,
        stamp=stamp,
        downloads_dir=bindings.downloads_dir,
        log_dir=bindings.log_dir,
        simulation_runtime_state=bindings.simulation_runtime_state,
        background_runtime_state=bindings.background_runtime_state,
        theta_initial_var=bindings.theta_initial_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        current_ui_params=bindings.current_ui_params,
        var_map=bindings.var_map,
        background_theta_for_index=bindings.background_theta_for_index,
        refresh_status=bindings.refresh_status,
        update_manual_pick_button_label=bindings.update_manual_pick_button_label,
        capture_undo_state=bindings.capture_undo_state,
        push_undo_state=bindings.push_undo_state,
        replace_dataset_cache=bindings.replace_dataset_cache,
        request_preview_skip_once=bindings.request_preview_skip_once,
        schedule_update=bindings.schedule_update,
        draw_overlay_records=bindings.draw_overlay_records,
        draw_initial_pairs_overlay=bindings.draw_initial_pairs_overlay,
        set_last_overlay_state=bindings.set_last_overlay_state,
        set_progress_text=bindings.set_progress_text,
        cmd_line=bindings.cmd_line,
        solver_inputs=bindings.solver_inputs,
        sim_display_rotate_k=bindings.sim_display_rotate_k,
        background_display_rotate_k=bindings.background_display_rotate_k,
        simulate_and_compare_hkl=bindings.simulate_and_compare_hkl,
        aggregate_match_centers=bindings.aggregate_match_centers,
        build_overlay_records=bindings.build_overlay_records,
        compute_frame_diagnostics=bindings.compute_frame_diagnostics,
        live_update_callback=bindings.live_update_callback,
    )


def execute_runtime_geometry_fit_solver_phase(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    solve_fit: Callable[..., object],
    solver_inputs: GeometryFitRuntimeSolverInputs,
    stamp: str,
    log_path: Path | str,
    start_progress_text: str = "Running geometry fit from saved manual Qr/Qz pairs...",
    event_callback: GeometryFitStageCallback | None = None,
    live_update_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> GeometryFitRuntimeExecutionResult:
    """Run the worker-safe logging and solver phase before any UI-state apply."""

    resolved_log_path = Path(log_path)
    logging_disabled = geometry_fit_all_logging_disabled()
    _, _log_line, _log_section = _build_geometry_fit_log_writers(resolved_log_path)

    def _emit_execution_event(kind: str, **payload: object) -> None:
        _emit_geometry_fit_stage_event(event_callback, kind, **payload)

    def _emit_cmd_line(text: str) -> None:
        text_value = str(text).strip()
        if text_value:
            _emit_execution_event("cmd_line", text=text_value)

    def _emit_progress_text(text: str) -> None:
        text_value = str(text)
        if text_value:
            _emit_execution_event("progress_text", text=text_value)

    def _solver_status_callback(text: str) -> None:
        status_text = str(text).strip()
        if not status_text:
            return
        _log_line(status_text)
        _emit_cmd_line(status_text)
        _emit_progress_text(status_text)

    try:
        _geometry_fit_assign_fit_run_id(
            prepared_run,
            fit_run_id=str(stamp),
        )
        write_geometry_fit_run_start_log(
            stamp=str(stamp),
            prepared_run=prepared_run,
            cmd_line=_emit_cmd_line,
            log_line=_log_line,
            log_section=_log_section,
        )
        try:
            point_provider_report_path = resolved_log_path.with_name(
                "geometry_fit_point_provider_report.json"
            )
            write_geometry_fit_point_provider_report(
                prepared_run.current_dataset,
                point_provider_report_path,
            )
            _log_line(f"Point-provider report: {point_provider_report_path}")
        except Exception as report_exc:
            _log_line(f"Point-provider report failed: {report_exc}")

        _emit_progress_text(str(start_progress_text))

        solver_request = build_geometry_fit_solver_request(
            prepared_run=prepared_run,
            var_names=var_names,
            solver_inputs=solver_inputs,
        )
        if solver_request.runtime_safety_note:
            _log_line(f"Runtime safety: {solver_request.runtime_safety_note}")
            _log_line()
        result = solve_geometry_fit_request(
            solver_request,
            solve_fit=solve_fit,
            status_callback=_solver_status_callback,
            live_update_callback=live_update_callback,
        )
        _emit_execution_event(
            "solver_finished",
            status="success",
            log_path=str(resolved_log_path),
        )
        return GeometryFitRuntimeExecutionResult(
            log_path=resolved_log_path,
            solver_request=solver_request,
            solver_result=result,
        )
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        _emit_cmd_line(f"failed: {exc}")
        _log_line(error_text)
        _emit_progress_text(error_text)
        _emit_execution_event(
            "solver_finished",
            status="error",
            error_text=error_text,
            log_path=str(resolved_log_path),
        )
        return GeometryFitRuntimeExecutionResult(
            log_path=resolved_log_path,
            error_text=error_text,
        )


def finalize_runtime_geometry_fit_execution(
    *,
    execution_result: GeometryFitRuntimeExecutionResult,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    preserve_live_theta: bool,
    setup: GeometryFitRuntimeExecutionSetup,
) -> GeometryFitRuntimeExecutionResult:
    """Apply one completed geometry-fit solver result on the runtime/UI thread."""

    if execution_result.error_text or execution_result.solver_result is None:
        return execution_result

    ui_bindings = setup.ui_bindings
    postprocess_config = setup.postprocess_config
    resolved_log_path = Path(execution_result.log_path)
    _, _, _log_section = _build_geometry_fit_log_writers(resolved_log_path)
    result = execution_result.solver_result
    apply_result = apply_runtime_geometry_fit_result(
        result=result,
        var_names=var_names,
        current_dataset=prepared_run.current_dataset,
        dataset_count=len(prepared_run.dataset_infos),
        joint_background_mode=prepared_run.joint_background_mode,
        preserve_live_theta=preserve_live_theta,
        max_display_markers=prepared_run.max_display_markers,
        bindings=build_runtime_geometry_fit_execution_result_bindings(
            result=result,
            prepared_run=prepared_run,
            var_names=var_names,
            ui_bindings=ui_bindings,
            postprocess_config=postprocess_config,
            log_section=_log_section,
        ),
    )
    trace_path: Path | None = None
    debug_logging = geometry_fit_debug_logging_enabled(prepared_run.geometry_runtime_cfg)
    should_write_trace = bool(debug_logging) or not bool(apply_result.accepted)
    if should_write_trace:
        candidate_trace_path = build_geometry_fit_trace_path(
            stamp=str(setup.postprocess_config.stamp),
            log_dir=setup.postprocess_config.log_dir,
            downloads_dir=setup.postprocess_config.downloads_dir,
        )
        trace_path = _write_geometry_fit_trace_file(
            fit_run_id=str(setup.postprocess_config.stamp),
            prepared_run=prepared_run,
            result=result,
            apply_result=apply_result,
            log_path=resolved_log_path,
            trace_path=candidate_trace_path,
        )
        if trace_path is not None:
            _log_section(
                "Phase trace:",
                [
                    f"fit_run_id={setup.postprocess_config.stamp}",
                    f"trace_path={trace_path}",
                    f"accepted={bool(apply_result.accepted)}",
                ],
            )
    replace_dataset_cache = ui_bindings.replace_dataset_cache
    if apply_result.accepted:
        if callable(replace_dataset_cache):
            replace_dataset_cache(
                build_geometry_fit_dataset_cache_payload(
                    prepared_run,
                    current_background_index=int(postprocess_config.current_background_index),
                )
            )
        play_completion_chime(
            prepared_run.geometry_runtime_cfg.get("completion_chime")
            if isinstance(prepared_run.geometry_runtime_cfg, Mapping)
            else None
        )
    return GeometryFitRuntimeExecutionResult(
        log_path=resolved_log_path,
        trace_path=trace_path,
        solver_request=execution_result.solver_request,
        solver_result=result,
        apply_result=apply_result,
    )


def run_runtime_geometry_fit_action(
    *,
    bindings: GeometryFitRuntimeActionBindings,
    prepare_run: Callable[..., GeometryFitPreparationResult] | None = None,
    build_execution_setup: Callable[..., GeometryFitRuntimeExecutionSetup] | None = None,
    execute_run: Callable[..., GeometryFitRuntimeExecutionResult] | None = None,
) -> GeometryFitRuntimeActionResult:
    """Run the full top-level geometry-fit action from live runtime bindings."""

    if prepare_run is None:
        prepare_run = prepare_runtime_geometry_fit_run
    if build_execution_setup is None:
        build_execution_setup = build_runtime_geometry_fit_execution_setup_from_bindings
    if execute_run is None:
        execute_run = execute_runtime_geometry_fit

    params = bindings.value_callbacks.current_params()
    mosaic_params = dict(params.get("mosaic_params", {}))
    var_names = list(bindings.value_callbacks.current_var_names())
    preserve_live_theta = "theta_initial" not in var_names and "theta_offset" not in var_names
    prepare_bindings = bindings.prepare_bindings_factory(var_names)
    preflight_runtime_cfg = (
        prepare_bindings.fit_config
        if isinstance(getattr(prepare_bindings, "fit_config", None), Mapping)
        else None
    )

    def _persist_preflight_failure_log(
        error_text: str,
        failure_log_sections: Sequence[tuple[str, Sequence[str]]] | None,
    ) -> Path | None:
        if geometry_fit_all_logging_disabled():
            return None
        try:
            stamp = str(bindings.stamp_factory())
            log_path = build_geometry_fit_log_path(
                stamp=stamp,
                log_dir=bindings.execution_bindings.log_dir,
                downloads_dir=bindings.execution_bindings.downloads_dir,
            )
            return write_geometry_fit_preflight_failure_log(
                stamp=stamp,
                error_text=str(error_text),
                log_path=log_path,
                log_sections=failure_log_sections,
            )
        except Exception:
            return None

    try:
        prepare_result = prepare_run(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            bindings=prepare_bindings,
        )
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        failure_log_sections = build_geometry_fit_preflight_log_sections(
            error_text=error_text,
            params=params,
            var_names=var_names,
            dataset_infos=None,
            current_dataset=None,
            selected_background_indices=(),
            joint_background_mode=False,
            geometry_runtime_cfg=preflight_runtime_cfg,
        )
        log_path = _persist_preflight_failure_log(error_text, failure_log_sections)
        prepare_result = GeometryFitPreparationResult(
            error_text=error_text,
            failure_log_sections=failure_log_sections,
            log_path=log_path,
        )
        bindings.execution_bindings.cmd_line(f"failed: {exc}")
        bindings.execution_bindings.set_progress_text(error_text)
        return GeometryFitRuntimeActionResult(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            prepare_result=prepare_result,
            error_text=error_text,
        )

    if prepare_result.prepared_run is None:
        should_force_preflight_log = bool(prepare_result.error_text) or bool(
            geometry_fit_debug_logging_enabled(preflight_runtime_cfg)
        )
        if should_force_preflight_log:
            preflight_error_text = str(
                prepare_result.error_text or "Geometry fit aborted before solver start."
            )
            failure_log_sections = list(
                prepare_result.failure_log_sections or ()
            ) or build_geometry_fit_preflight_log_sections(
                error_text=preflight_error_text,
                params=params,
                var_names=var_names,
                dataset_infos=None,
                current_dataset=None,
                selected_background_indices=(),
                joint_background_mode=False,
                geometry_runtime_cfg=preflight_runtime_cfg,
            )
            log_path = _persist_preflight_failure_log(
                preflight_error_text,
                failure_log_sections,
            )
            prepare_result = GeometryFitPreparationResult(
                prepared_run=prepare_result.prepared_run,
                error_text=prepare_result.error_text,
                failure_log_sections=[
                    (str(title), [str(line) for line in (lines or ())])
                    for title, lines in failure_log_sections
                ],
                log_path=log_path,
            )
            if prepare_result.error_text:
                bindings.execution_bindings.set_progress_text(str(prepare_result.error_text))
        return GeometryFitRuntimeActionResult(
            params=params,
            var_names=var_names,
            preserve_live_theta=preserve_live_theta,
            prepare_result=prepare_result,
            error_text=prepare_result.error_text,
        )

    solver_mosaic_params, fit_sample_count = build_geometry_fit_solver_mosaic_params(
        params=params,
        geometry_runtime_cfg=prepare_result.prepared_run.geometry_runtime_cfg,
        build_mosaic_params=bindings.value_callbacks.build_mosaic_params,
    )
    if fit_sample_count is not None:
        prepare_result.prepared_run.fit_params["mosaic_params"] = dict(solver_mosaic_params)
        bindings.execution_bindings.cmd_line(
            f"Geometry fit: solver sample count={int(fit_sample_count)}"
        )

    execution_setup = build_execution_setup(
        prepared_run=prepare_result.prepared_run,
        mosaic_params=mosaic_params,
        stamp=str(bindings.stamp_factory()),
        bindings=bindings.execution_bindings,
    )
    execution_result = execute_run(
        prepared_run=prepare_result.prepared_run,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        solve_fit=bindings.solve_fit,
        setup=execution_setup,
        flush_ui=bindings.flush_ui,
    )
    return GeometryFitRuntimeActionResult(
        params=params,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        prepare_result=prepare_result,
        execution_result=execution_result,
        error_text=execution_result.error_text,
    )


def execute_runtime_geometry_fit(
    *,
    prepared_run: GeometryFitPreparedRun,
    var_names: Sequence[object],
    preserve_live_theta: bool,
    solve_fit: Callable[..., object],
    setup: GeometryFitRuntimeExecutionSetup,
    start_progress_text: str = "Running geometry fit from saved manual Qr/Qz pairs…",
    flush_ui: Callable[[], None] | None = None,
) -> GeometryFitRuntimeExecutionResult:
    """Run one prepared geometry fit through the live solver and callbacks."""

    ui_bindings = setup.ui_bindings
    postprocess_config = setup.postprocess_config
    execution_phase = execute_runtime_geometry_fit_solver_phase(
        prepared_run=prepared_run,
        var_names=var_names,
        solve_fit=solve_fit,
        solver_inputs=postprocess_config.solver_inputs,
        stamp=str(postprocess_config.stamp),
        log_path=postprocess_config.log_path,
        start_progress_text=str(start_progress_text),
        event_callback=(
            lambda kind, payload: (
                ui_bindings.cmd_line(str(payload.get("text", "")))
                if str(kind) == "cmd_line"
                else (
                    ui_bindings.set_progress_text(str(payload.get("text", "")))
                    if str(kind) == "progress_text"
                    else None
                ),
                flush_ui()
                if callable(flush_ui) and str(kind) in {"cmd_line", "progress_text"}
                else None,
            )
        ),
        live_update_callback=ui_bindings.live_update_callback,
    )
    return finalize_runtime_geometry_fit_execution(
        execution_result=execution_phase,
        prepared_run=prepared_run,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        setup=setup,
    )
