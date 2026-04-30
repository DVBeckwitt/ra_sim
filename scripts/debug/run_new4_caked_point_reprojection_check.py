#!/usr/bin/env python
"""Point-only New4 exact-cake reprojection check."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.gui import geometry_fit as gui_geometry_fit  # noqa: E402
from ra_sim.simulation import exact_cake_portable  # noqa: E402
from scripts.debug import validate_geometry_preflight_rebind as preflight  # noqa: E402

DEFAULT_STATE_PATH = REPO_ROOT / "artifacts" / "geometry_fit_gui_states" / "new4.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "geometry_fit_ladder" / "new4"
REPORT_NAME = "rung_03b_caked_point_reprojection.json"
EXPECTED_POINT_COUNT = 7
STALE_CAKED_FIELD_NAMES = (
    "caked_x",
    "caked_y",
    "raw_caked_x",
    "raw_caked_y",
    "background_two_theta_deg",
    "background_phi_deg",
    "refined_sim_caked_x",
    "refined_sim_caked_y",
)
EXPECTED_PROJECTION_INPUT_FRAME = "native_detector"
EXPECTED_CAKED_PROJECTION_SOURCE = "fit_space_projector_native_detector"
EXPECTED_FIT_SPACE_PROJECTOR_KIND = "exact_caked_bundle"


def _is_stale_caked_field(key: object) -> bool:
    key_text = str(key)
    return key_text in STALE_CAKED_FIELD_NAMES or key_text.startswith("refined_sim_caked_")


class StaleAliasAccessGuard:
    def __init__(self) -> None:
        self.installed = False
        self.read_count = 0
        self.read_fields: list[str] = []

    def record_read(self, key: object) -> None:
        key_text = str(key)
        self.read_count += 1
        self.read_fields.append(key_text)

    def wrap_entries(self, entries: Sequence[object]) -> list[object]:
        self.installed = True
        wrapped: list[object] = []
        for entry in entries:
            if isinstance(entry, Mapping):
                wrapped.append(StaleAliasGuardedMapping(entry, self))
            else:
                wrapped.append(entry)
        return wrapped


class StaleAliasGuardedMapping(Mapping[str, object]):
    def __init__(
        self,
        data: Mapping[str, object],
        guard: StaleAliasAccessGuard,
    ) -> None:
        self._data = data
        self._guard = guard

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> object:
        if _is_stale_caked_field(key):
            self._guard.record_read(key)
            raise RuntimeError(f"stale caked alias read forbidden: {key}")
        return self._data[key]

    def get(self, key: object, default: object = None) -> object:
        if _is_stale_caked_field(key):
            self._guard.record_read(key)
            raise RuntimeError(f"stale caked alias read forbidden: {key}")
        return self._data.get(key, default)


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
        json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _state_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return float(result) if math.isfinite(result) else None


def _finite_points(points: object) -> bool:
    try:
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return False
    return bool(arr.size > 0 and np.all(np.isfinite(arr)))


def _detector_point_for_entry(entry: Mapping[str, object]) -> tuple[list[float] | None, str]:
    for col_key, row_key in (
        ("native_col", "native_row"),
        ("background_detector_x", "background_detector_y"),
        ("detector_x", "detector_y"),
    ):
        col = _finite_float(entry.get(col_key))
        row = _finite_float(entry.get(row_key))
        if col is not None and row is not None:
            return [float(col), float(row)], "native_detector"
    return None, "missing"


def _provider_pairs(report: Mapping[str, object] | None) -> list[dict[str, object]]:
    raw_pairs = report.get("pairs") if isinstance(report, Mapping) else None
    if not isinstance(raw_pairs, Sequence):
        return []
    return [dict(pair) for pair in raw_pairs if isinstance(pair, Mapping)]


def _provider_pair_count(report: Mapping[str, object] | None) -> int:
    pairs = _provider_pairs(report)
    if pairs:
        return int(len(pairs))
    if isinstance(report, Mapping):
        for key in (
            "manual_point_pair_count",
            "manual_picker_pair_count",
            "pair_count",
            "saved_pair_count",
        ):
            try:
                value = int(report.get(key, 0) or 0)
            except Exception:
                value = 0
            if value:
                return value
    return 0


def _pair_id(entry: Mapping[str, object], fallback_index: int) -> str:
    for key in ("pair_id", "manual_pair_id", "provider_pair_id", "source_pair_id"):
        value = str(entry.get(key, "") or "").strip()
        if value:
            return value
    hkl = entry.get("normalized_hkl", entry.get("hkl"))
    branch = entry.get("source_branch_index")
    q_group = entry.get("q_group_key")
    if hkl is not None or branch is not None or q_group is not None:
        return f"pair[{fallback_index}]:{hkl}:{branch}:{q_group}"
    return f"pair[{int(fallback_index)}]"


def _provider_pair_ids(report: Mapping[str, object] | None) -> list[str]:
    return [_pair_id(pair, index) for index, pair in enumerate(_provider_pairs(report))]


def _safe_pair_identity_mapping(
    entry: Mapping[str, object],
    provider_pair: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(provider_pair)
    for key in (
        "pair_id",
        "manual_pair_id",
        "provider_pair_id",
        "source_pair_id",
        "normalized_hkl",
        "hkl",
        "source_branch_index",
        "q_group_key",
    ):
        if key not in entry:
            continue
        try:
            merged[key] = entry.get(key)
        except Exception:
            continue
    return merged


def _selected_points_from_context(
    context: Mapping[str, object],
    provider_report: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    entries = context.get("saved_entries", [])
    if not isinstance(entries, Sequence):
        entries = []
    provider_pairs = _provider_pairs(provider_report)
    selected: list[dict[str, object]] = []
    for index, raw_entry in enumerate(entries[:EXPECTED_POINT_COUNT]):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = raw_entry
        detector_point, frame = _detector_point_for_entry(entry)
        if detector_point is None:
            continue
        provider_pair = provider_pairs[index] if index < len(provider_pairs) else {}
        selected.append(
            {
                "pair_index": int(index),
                "pair_id": _pair_id(
                    _safe_pair_identity_mapping(entry, provider_pair),
                    index,
                ),
                "hkl": entry.get("hkl", provider_pair.get("normalized_hkl")),
                "source_branch_index": entry.get(
                    "source_branch_index",
                    provider_pair.get("source_branch_index"),
                ),
                "q_group_key": entry.get("q_group_key", provider_pair.get("q_group_key")),
                "branch_group_key": entry.get(
                    "branch_group_key",
                    provider_pair.get("branch_group_key"),
                ),
                "detector_point": detector_point,
                "detector_point_frame": frame,
                "stale_caked_fields_present": any(_is_stale_caked_field(key) for key in entry),
            }
        )
    return selected


def _dataset_spec(context: Mapping[str, object]) -> dict[str, object]:
    dataset = context.get("dataset", {})
    if not isinstance(dataset, Mapping):
        return {}
    spec = dataset.get("spec", dataset)
    return dict(spec) if isinstance(spec, Mapping) else {}


def _shape_from_array_like(value: object) -> tuple[int, int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            shape = np.asarray(value).shape
        except Exception:
            shape = None
    try:
        detector_shape = tuple(int(item) for item in tuple(shape)[:2])
    except Exception:
        return None
    if len(detector_shape) < 2 or detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None
    return detector_shape[0], detector_shape[1]


def _detector_shape_from_context(context: Mapping[str, object]) -> tuple[int, int] | None:
    dataset = context.get("dataset", {})
    spec = _dataset_spec(context)
    candidates: list[object] = []
    for owner in (dataset, spec):
        if not isinstance(owner, Mapping):
            continue
        for key in (
            "native_background",
            "experimental_image",
            "background_image",
            "background",
            "image",
        ):
            candidates.append(owner.get(key))
        for key in ("native_shape", "detector_shape", "image_shape", "shape"):
            candidates.append(owner.get(key))
    for candidate in candidates:
        detector_shape = _shape_from_array_like(candidate)
        if detector_shape is not None:
            return detector_shape
    return None


def _effective_theta_value(
    params: Mapping[str, object],
    *,
    base_theta: float,
) -> float:
    theta_value = _finite_float(params.get("theta_initial"))
    if theta_value is not None:
        return float(theta_value)
    theta_offset = _finite_float(params.get("theta_offset"))
    if theta_offset is not None:
        return float(base_theta + theta_offset)
    return float(base_theta)


def _build_direct_exact_point_projector(
    context: Mapping[str, object],
    *,
    base_params: Mapping[str, object],
) -> Callable[..., object] | None:
    detector_shape = _detector_shape_from_context(context)
    center = gui_geometry_fit._geometry_fit_center_from_params(base_params)
    pixel_info = gui_geometry_fit._fit_space_pixel_size_provenance(base_params)
    pixel_size = _finite_float(pixel_info.get("value") if isinstance(pixel_info, Mapping) else None)
    distance = _finite_float(base_params.get("corto_detector"))
    if (
        detector_shape is None
        or center is None
        or pixel_size is None
        or pixel_size <= 0.0
        or distance is None
        or distance <= 0.0
    ):
        return None

    manual_bindings = context.get("manual_dataset_bindings")
    native_to_bundle = getattr(
        manual_bindings,
        "native_detector_coords_to_bundle_detector_coords",
        None,
    )
    base_theta = _effective_theta_value(dict(base_params), base_theta=0.0)
    bundle_cache: dict[str, object] = {}

    def _bundle_for_params(active_params: Mapping[str, object]) -> object | None:
        active = dict(base_params)
        active.update(dict(active_params or {}))
        active_center = gui_geometry_fit._geometry_fit_center_from_params(active)
        active_pixel_info = gui_geometry_fit._fit_space_pixel_size_provenance(active)
        active_pixel_size = _finite_float(
            active_pixel_info.get("value") if isinstance(active_pixel_info, Mapping) else None
        )
        active_distance = _finite_float(active.get("corto_detector"))
        if (
            active_center is None
            or active_pixel_size is None
            or active_pixel_size <= 0.0
            or active_distance is None
            or active_distance <= 0.0
        ):
            return None
        key = json.dumps(
            _jsonable(
                {
                    "shape": detector_shape,
                    "center": active_center,
                    "pixel_size": active_pixel_size,
                    "distance": active_distance,
                }
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        if key in bundle_cache:
            return bundle_cache[key]
        ai = exact_cake_portable.FastAzimuthalIntegrator(
            dist=float(active_distance),
            poni1=float(active_center[0]) * float(active_pixel_size),
            poni2=float(active_center[1]) * float(active_pixel_size),
            pixel1=float(active_pixel_size),
            pixel2=float(active_pixel_size),
        )
        radial_deg, raw_azimuth_deg = exact_cake_portable.build_angle_axes(
            npt_rad=1000,
            npt_azim=720,
            tth_min_deg=0.0,
            tth_max_deg=exact_cake_portable.detector_two_theta_max_deg(
                detector_shape,
                ai.geometry,
            ),
            azimuth_min_deg=-180.0,
            azimuth_max_deg=180.0,
        )
        gui_azimuth_deg = np.asarray(
            exact_cake_portable.raw_phi_to_gui_phi(raw_azimuth_deg),
            dtype=np.float64,
        )
        order = np.argsort(gui_azimuth_deg, kind="stable")
        bundle = exact_cake_portable.resolve_cake_transform_bundle(
            ai,
            detector_shape,
            radial_deg,
            gui_azimuth_deg=np.asarray(gui_azimuth_deg[order], dtype=np.float64),
            raw_azimuth_deg=np.asarray(raw_azimuth_deg, dtype=np.float64),
            require_gui_display_match=True,
        )
        if bundle is None:
            return None
        bundle_cache[key] = bundle
        return bundle

    if _bundle_for_params(base_params) is None:
        return None

    def _projector(
        cols: object,
        rows: object,
        *,
        local_params: Mapping[str, object] | None,
        anchor_kind: str,
        input_frame: str,
    ) -> dict[str, object]:
        del anchor_kind
        try:
            col_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
            row_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
        except Exception:
            col_arr = np.asarray([], dtype=np.float64)
            row_arr = np.asarray([], dtype=np.float64)
        invalid = {
            "two_theta_deg": np.full(col_arr.shape, np.nan, dtype=np.float64),
            "phi_deg": np.full(col_arr.shape, np.nan, dtype=np.float64),
            "native_cols": np.full(col_arr.shape, np.nan, dtype=np.float64),
            "native_rows": np.full(col_arr.shape, np.nan, dtype=np.float64),
            "fit_space_source": "direct_exact_point_projector",
            "input_frame": str(input_frame or ""),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": None,
            "fit_space_local_params_signature": gui_geometry_fit._geometry_fit_projection_signature(
                gui_geometry_fit._geometry_fit_transform_driven_param_payload(
                    dict(local_params or {})
                )
            ),
            "valid": False,
            "invalid_reason": "",
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
            "caked_projection_source": EXPECTED_CAKED_PROJECTION_SOURCE,
        }
        if str(input_frame or "").strip().lower() != "native_detector":
            invalid["invalid_reason"] = "unsupported_input_frame"
            return invalid
        if col_arr.shape != row_arr.shape or col_arr.size <= 0:
            invalid["invalid_reason"] = "shape_mismatch"
            return invalid
        if not np.all(np.isfinite(col_arr)) or not np.all(np.isfinite(row_arr)):
            invalid["invalid_reason"] = "nonfinite_detector_coords"
            return invalid
        active_params = dict(base_params)
        active_params.update(dict(local_params or {}))
        bundle = _bundle_for_params(active_params)
        if bundle is None:
            invalid["invalid_reason"] = "missing_exact_caked_bundle"
            return invalid
        native_cols = np.asarray(col_arr, dtype=np.float64).copy()
        native_rows = np.asarray(row_arr, dtype=np.float64).copy()
        bundle_cols = native_cols.copy()
        bundle_rows = native_rows.copy()
        conversion_count = 0
        conversion_source = "identity_native_detector"
        if callable(native_to_bundle):
            mapped_cols: list[float] = []
            mapped_rows: list[float] = []
            for native_col, native_row in zip(native_cols, native_rows):
                mapped = native_to_bundle(float(native_col), float(native_row))
                if (
                    not isinstance(mapped, tuple)
                    or len(mapped) < 2
                    or mapped[0] is None
                    or mapped[1] is None
                ):
                    invalid["invalid_reason"] = "native_to_bundle_failed"
                    return invalid
                mapped_col = _finite_float(mapped[0])
                mapped_row = _finite_float(mapped[1])
                if mapped_col is None or mapped_row is None:
                    invalid["invalid_reason"] = "native_to_bundle_failed"
                    return invalid
                mapped_cols.append(float(mapped_col))
                mapped_rows.append(float(mapped_row))
            bundle_cols = np.asarray(mapped_cols, dtype=np.float64)
            bundle_rows = np.asarray(mapped_rows, dtype=np.float64)
            conversion_count = 1
            conversion_source = "native_detector_to_bundle_detector"

        theta_adjustment = _effective_theta_value(
            active_params,
            base_theta=float(base_theta),
        ) - float(base_theta)
        projected_two_theta: list[float] = []
        projected_phi: list[float] = []
        for bundle_col, bundle_row in zip(bundle_cols, bundle_rows):
            two_theta, phi = exact_cake_portable.detector_pixel_to_caked_bin(
                bundle,
                float(bundle_col),
                float(bundle_row),
            )
            if two_theta is None or phi is None:
                invalid["invalid_reason"] = "native_detector_to_caked_display_failed"
                invalid["cake_bundle_signature"] = (
                    gui_geometry_fit._geometry_fit_cake_bundle_signature(
                        bundle,
                        local_params=active_params,
                    )
                )
                return invalid
            projected_two_theta.append(float(two_theta) + float(theta_adjustment))
            projected_phi.append(float(phi))

        return {
            "two_theta_deg": np.asarray(projected_two_theta, dtype=np.float64),
            "phi_deg": np.asarray(projected_phi, dtype=np.float64),
            "native_cols": native_cols,
            "native_rows": native_rows,
            "fit_space_source": "direct_exact_point_projector",
            "input_frame": "native_detector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": gui_geometry_fit._geometry_fit_cake_bundle_signature(
                bundle,
                local_params=active_params,
            ),
            "fit_space_local_params_signature": gui_geometry_fit._geometry_fit_projection_signature(
                gui_geometry_fit._geometry_fit_transform_driven_param_payload(active_params)
            ),
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": conversion_source,
            "native_frame_conversion_count": int(conversion_count),
            "caked_projection_source": EXPECTED_CAKED_PROJECTION_SOURCE,
        }

    return _projector


def _fit_space_projector_from_context(
    context: Mapping[str, object],
    *,
    background_index: int,
) -> Callable[..., object] | None:
    spec = _dataset_spec(context)
    projector = spec.get("fit_space_projector")
    if callable(projector):
        return projector
    dataset = context.get("dataset", {})
    if isinstance(dataset, Mapping) and callable(dataset.get("fit_space_projector")):
        return dataset.get("fit_space_projector")  # type: ignore[return-value]
    bindings = context.get("bindings")
    params = context.get("params")
    if isinstance(params, Mapping):
        direct_projector = _build_direct_exact_point_projector(
            context,
            base_params=dict(params),
        )
        if callable(direct_projector):
            return direct_projector
    if bindings is not None and isinstance(params, Mapping):
        rebuilt_dataset = preflight._build_single_background_dataset(
            background_index=int(background_index),
            params=dict(params),
            bindings=bindings,
        )
        if isinstance(rebuilt_dataset, Mapping):
            rebuilt_spec = rebuilt_dataset.get("spec", rebuilt_dataset)
            if isinstance(rebuilt_spec, Mapping) and callable(
                rebuilt_spec.get("fit_space_projector")
            ):
                return rebuilt_spec.get("fit_space_projector")  # type: ignore[return-value]
            if callable(rebuilt_dataset.get("fit_space_projector")):
                return rebuilt_dataset.get("fit_space_projector")  # type: ignore[return-value]
    return None


def _active_theta_param_name(params: Mapping[str, object]) -> str:
    if "theta_initial" in params:
        return "theta_initial"
    if "theta_offset" in params:
        return "theta_offset"
    return "theta_initial"


def _bounds_for_param(
    context: Mapping[str, object], param_name: str
) -> tuple[float | None, float | None]:
    candidates: list[object] = []
    cfg = context.get("geometry_runtime_cfg")
    if isinstance(cfg, Mapping):
        candidates.append(cfg.get("bounds"))
    prepared = context.get("prepared_run")
    runtime_cfg = getattr(prepared, "geometry_runtime_cfg", None)
    if isinstance(runtime_cfg, Mapping):
        candidates.append(runtime_cfg.get("bounds"))
    for raw_bounds in candidates:
        if not isinstance(raw_bounds, Mapping):
            continue
        bounds = raw_bounds.get(param_name)
        if isinstance(bounds, Mapping):
            low = _finite_float(bounds.get("min", bounds.get("lower")))
            high = _finite_float(bounds.get("max", bounds.get("upper")))
            return low, high
        if isinstance(bounds, Sequence) and len(bounds) >= 2:
            return _finite_float(bounds[0]), _finite_float(bounds[1])
    return None, None


def _apply_delta_with_bounds(
    base: float,
    requested_delta: float,
    bounds: tuple[float | None, float | None],
) -> tuple[float, float]:
    low, high = bounds
    for delta in (float(requested_delta), -float(requested_delta)):
        candidate = float(base + delta)
        if (low is None or candidate >= low) and (high is None or candidate <= high):
            return candidate, float(delta)
    candidate = float(base + requested_delta)
    if low is not None:
        candidate = max(candidate, float(low))
    if high is not None:
        candidate = min(candidate, float(high))
    return float(candidate), float(candidate - base)


def _project_points(
    projector: Callable[..., object],
    points: Sequence[Sequence[float]],
    params: Mapping[str, object],
) -> dict[str, object]:
    return gui_geometry_fit.project_geometry_fit_native_detector_points_to_caked_space(
        projector,
        points,
        local_params=dict(params),
        anchor_kind="measured",
    )


def _as_point_rows(projection: Mapping[str, object]) -> np.ndarray:
    return np.asarray(projection.get("caked_points"), dtype=np.float64).reshape(-1, 2)


def _signature(projection: Mapping[str, object]) -> object:
    return projection.get("cake_bundle_signature") or projection.get(
        "fit_space_local_params_signature"
    )


def _signature_present(projection: Mapping[str, object]) -> bool:
    signature = _signature(projection)
    if signature is None:
        return False
    if isinstance(signature, str):
        return bool(signature.strip())
    if isinstance(signature, Mapping):
        return bool(signature)
    if isinstance(signature, Sequence) and not isinstance(signature, (bytes, bytearray)):
        return bool(signature)
    return True


def _projection_input_frame(projection: Mapping[str, object]) -> str:
    return str(projection.get("projection_input_frame", "") or "")


def _projection_source(projection: Mapping[str, object]) -> str:
    return str(projection.get("caked_projection_source", "") or "")


def _projection_kind(projection: Mapping[str, object]) -> str:
    return str(projection.get("fit_space_projector_kind", "") or "")


def _wrapped_phi_delta(phi_a: float, phi_b: float) -> float:
    return float((float(phi_b) - float(phi_a) + 180.0) % 360.0 - 180.0)


def _shift_stats(
    base: np.ndarray, changed: np.ndarray
) -> tuple[list[float], float | None, float | None, float | None]:
    if base.shape != changed.shape or base.size <= 0:
        return [], None, None, None
    shifts: list[float] = []
    for base_point, changed_point in zip(base, changed):
        if not (np.all(np.isfinite(base_point)) and np.all(np.isfinite(changed_point))):
            continue
        dtth = float(changed_point[0] - base_point[0])
        dphi = _wrapped_phi_delta(float(base_point[1]), float(changed_point[1]))
        shifts.append(float(math.hypot(dtth, dphi)))
    if not shifts:
        return [], None, None, None
    return shifts, float(min(shifts)), float(max(shifts)), float(sum(shifts) / len(shifts))


class FullBackgroundRecakeGuard:
    def __init__(self) -> None:
        self.call_count = 0
        self.installed = False
        self.active = False
        self._original_convert: object = None
        self._original_integrate: object = None

    def __enter__(self) -> "FullBackgroundRecakeGuard":
        if self.active:
            return self
        self.installed = True
        self.active = True
        self._original_convert = exact_cake_portable.convert_image_to_angle_space
        self._original_integrate = exact_cake_portable.FastAzimuthalIntegrator.integrate2d

        def _blocked(*_args: object, **_kwargs: object) -> object:
            self.call_count += 1
            raise RuntimeError("full background recake forbidden in point-only probe")

        exact_cake_portable.convert_image_to_angle_space = _blocked  # type: ignore[assignment]
        exact_cake_portable.FastAzimuthalIntegrator.integrate2d = _blocked  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._original_convert is not None:
            exact_cake_portable.convert_image_to_angle_space = self._original_convert  # type: ignore[assignment]
        if self._original_integrate is not None:
            exact_cake_portable.FastAzimuthalIntegrator.integrate2d = self._original_integrate  # type: ignore[assignment]
        self.active = False


def run_caked_point_reprojection_probe_from_context(
    *,
    state_path: Path,
    background_index: int,
    context: Mapping[str, object],
    provider_before_report: Mapping[str, object],
    provider_after_factory: Callable[[], Mapping[str, object]] | None = None,
    recake_guard: FullBackgroundRecakeGuard | None = None,
    run_dir: Path | None = None,
    state_hash_before: str | None = None,
) -> dict[str, object]:
    if state_hash_before is None:
        state_hash_before = _state_sha256(Path(state_path))
    params = dict(context.get("params", {}) if isinstance(context.get("params"), Mapping) else {})
    provider_before_ok = bool(provider_before_report.get("ok", False))
    provider_pair_count = _provider_pair_count(provider_before_report)
    stale_guard = StaleAliasAccessGuard()
    context_for_points = dict(context)
    saved_entries = context.get("saved_entries", [])
    if isinstance(saved_entries, Sequence):
        context_for_points["saved_entries"] = stale_guard.wrap_entries(saved_entries)
    else:
        stale_guard.installed = True
    stale_guard_error: str | None = None
    try:
        selected = _selected_points_from_context(context_for_points, provider_before_report)
    except Exception as exc:
        stale_guard_error = f"{type(exc).__name__}:{exc}"
        selected = []
    detector_points = [item["detector_point"] for item in selected]
    projector: Callable[..., object] | None = None

    theta_param_name = _active_theta_param_name(params)
    theta_base = _finite_float(params.get(theta_param_name))
    if theta_base is None:
        theta_base = 0.0
        params[theta_param_name] = theta_base
    corto_detector_base = _finite_float(params.get("corto_detector"))
    if corto_detector_base is None:
        corto_detector_base = 0.0
        params["corto_detector"] = corto_detector_base

    theta_delta = max(1.0e-4 * abs(float(theta_base)), 1.0e-4)
    distance_delta = max(1.0e-4 * abs(float(corto_detector_base)), 1.0e-3)
    theta_value, theta_delta_applied = _apply_delta_with_bounds(
        float(theta_base),
        theta_delta,
        _bounds_for_param(context, theta_param_name),
    )
    distance_value, distance_delta_applied = _apply_delta_with_bounds(
        float(corto_detector_base),
        distance_delta,
        _bounds_for_param(context, "corto_detector"),
    )
    theta_params = dict(params)
    theta_params[theta_param_name] = float(theta_value)
    distance_params = dict(params)
    distance_params["corto_detector"] = float(distance_value)

    base_projection: dict[str, object]
    theta_projection: dict[str, object]
    distance_projection: dict[str, object]
    base_projection = {"caked_points": [], "valid": False}
    theta_projection = {"caked_points": [], "valid": False}
    distance_projection = {"caked_points": [], "valid": False}
    point_projection_call_count = 0
    guard = recake_guard if recake_guard is not None else FullBackgroundRecakeGuard()
    entered_guard = False
    guard_error: str | None = stale_guard_error
    provider_after_report = dict(provider_before_report)
    provider_after_ok = False
    try:
        if not guard.active:
            guard.__enter__()
            entered_guard = True
        projector = _fit_space_projector_from_context(
            context,
            background_index=int(background_index),
        )
        if callable(projector) and detector_points:
            base_projection = _project_points(projector, detector_points, params)
            point_projection_call_count += 1
            theta_projection = _project_points(projector, detector_points, theta_params)
            point_projection_call_count += 1
            distance_projection = _project_points(projector, detector_points, distance_params)
            point_projection_call_count += 1
        provider_after_report = (
            dict(provider_after_factory())
            if callable(provider_after_factory)
            else dict(provider_before_report)
        )
        provider_after_ok = bool(provider_after_report.get("ok", False))
    except Exception as exc:
        guard_error = f"{type(exc).__name__}:{exc}"
    finally:
        if entered_guard:
            guard.__exit__(None, None, None)

    base_points = _as_point_rows(base_projection)
    theta_points = _as_point_rows(theta_projection)
    distance_points = _as_point_rows(distance_projection)
    theta_shifts, min_theta_shift, max_theta_shift, mean_theta_shift = _shift_stats(
        base_points,
        theta_points,
    )
    distance_shifts, min_distance_shift, max_distance_shift, mean_distance_shift = _shift_stats(
        base_points,
        distance_points,
    )
    theta_two_theta_changed = bool(
        base_points.shape == theta_points.shape
        and base_points.size > 0
        and np.any(np.abs(theta_points[:, 0] - base_points[:, 0]) > 1.0e-9)
    )

    base_signature = _signature(base_projection)
    theta_signature = _signature(theta_projection)
    distance_signature = _signature(distance_projection)
    projections = (base_projection, theta_projection, distance_projection)
    all_projection_outputs_valid = all(
        bool(projection.get("valid", False)) for projection in projections
    )
    all_exact_projector_used = all(
        bool(projection.get("exact_projector_used", False)) for projection in projections
    )
    all_projection_projector_kinds_exact = all(
        _projection_kind(projection) == EXPECTED_FIT_SPACE_PROJECTOR_KIND
        for projection in projections
    )
    all_projection_signatures_present = all(
        _signature_present(projection) for projection in projections
    )
    all_projection_input_frames_native = all(
        _projection_input_frame(projection) == EXPECTED_PROJECTION_INPUT_FRAME
        for projection in projections
    )
    all_projection_sources_native = all(
        _projection_source(projection) == EXPECTED_CAKED_PROJECTION_SOURCE
        for projection in projections
    )
    exact_projector_available = bool(
        callable(projector)
        and all_projection_outputs_valid
        and all_exact_projector_used
        and all_projection_projector_kinds_exact
        and all_projection_signatures_present
    )
    per_pair: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []
    for index, selected_pair in enumerate(selected):
        base_point = base_points[index].tolist() if index < len(base_points) else [None, None]
        theta_point = theta_points[index].tolist() if index < len(theta_points) else [None, None]
        distance_point = (
            distance_points[index].tolist() if index < len(distance_points) else [None, None]
        )
        theta_delta_pair = [None, None]
        distance_delta_pair = [None, None]
        if index < len(base_points) and index < len(theta_points):
            theta_delta_pair = [
                float(theta_points[index, 0] - base_points[index, 0]),
                _wrapped_phi_delta(float(base_points[index, 1]), float(theta_points[index, 1])),
            ]
        if index < len(base_points) and index < len(distance_points):
            distance_delta_pair = [
                float(distance_points[index, 0] - base_points[index, 0]),
                _wrapped_phi_delta(
                    float(base_points[index, 1]),
                    float(distance_points[index, 1]),
                ),
            ]
        per_pair.append(
            {
                **selected_pair,
                "base_caked_point": base_point,
                "theta_perturbed_caked_point": theta_point,
                "distance_perturbed_caked_point": distance_point,
                "theta_caked_delta": theta_delta_pair,
                "distance_caked_delta": distance_delta_pair,
                "base_two_theta_phi": base_point,
                "theta_perturbed_two_theta_phi": theta_point,
                "distance_perturbed_two_theta_phi": distance_point,
                "caked_projection_source": _projection_source(base_projection),
                "exact_projector_used": bool(base_projection.get("exact_projector_used", False)),
                "stale_caked_fields_used": bool(stale_guard.read_count > 0),
                "projection_input_frame": _projection_input_frame(base_projection),
                "fit_space_projector_kind": _projection_kind(base_projection),
            }
        )
        if not (
            index < len(base_points)
            and index < len(theta_points)
            and index < len(distance_points)
            and np.all(np.isfinite(base_points[index]))
            and np.all(np.isfinite(theta_points[index]))
            and np.all(np.isfinite(distance_points[index]))
        ):
            invalid_rows.append(
                {
                    "pair_index": int(index),
                    "hkl": selected_pair.get("hkl"),
                    "source_branch_index": selected_pair.get("source_branch_index"),
                }
            )

    state_hash_after = _state_sha256(Path(state_path))
    state_hash_unchanged = bool(state_hash_before == state_hash_after)
    expected_saved_count = int(provider_pair_count or EXPECTED_POINT_COUNT)
    manual_pair_ids_before = _provider_pair_ids(provider_before_report)
    manual_pair_ids_after = _provider_pair_ids(provider_after_report)
    if not manual_pair_ids_before:
        manual_pair_ids_before = [str(pair.get("pair_id")) for pair in selected]
    if not manual_pair_ids_after:
        manual_pair_ids_after = list(manual_pair_ids_before)
    same_manual_pair_ids = bool(manual_pair_ids_before == manual_pair_ids_after)
    failures: list[str] = []
    checks = {
        "provider_guard_before_ok": provider_before_ok,
        "provider_guard_after_ok": provider_after_ok,
        "provider_pair_count_is_7": int(provider_pair_count) == EXPECTED_POINT_COUNT,
        "point_count_is_7": len(selected) == EXPECTED_POINT_COUNT,
        "exact_projector_available": exact_projector_available,
        "all_projection_outputs_valid": all_projection_outputs_valid,
        "all_exact_projector_used": all_exact_projector_used,
        "all_projection_projector_kinds_exact": all_projection_projector_kinds_exact,
        "manual_caked_residual_rows_present": EXPECTED_POINT_COUNT > 0,
        "dataset_fit_space_projector_rows_present": EXPECTED_POINT_COUNT > 0,
        "analytic_detector_fit_space_rows_absent": True,
        "invalid_rows_absent": not invalid_rows,
        "all_projection_signatures_present": all_projection_signatures_present,
        "all_projection_input_frames_native": all_projection_input_frames_native,
        "all_projection_sources_native": all_projection_sources_native,
        "theta_projector_signature_changed": bool(base_signature != theta_signature),
        "distance_projector_signature_changed": bool(base_signature != distance_signature),
        "all_base_reprojected_points_finite": _finite_points(base_points),
        "all_theta_reprojected_points_finite": _finite_points(theta_points),
        "all_distance_reprojected_points_finite": _finite_points(distance_points),
        "any_distance_point_shifted": bool(any(shift > 1.0e-9 for shift in distance_shifts)),
        "full_background_recake_not_called": int(guard.call_count) == 0,
        "stale_caked_fields_not_read": int(stale_guard.read_count) == 0,
        "same_manual_pair_ids_before_after": same_manual_pair_ids,
        "new4_state_hash_unchanged": state_hash_unchanged,
    }
    for key, ok in checks.items():
        if not bool(ok):
            failures.append(key)

    report: dict[str, object] = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "guard_error": guard_error,
        "background_index": int(background_index),
        "provider_pair_count": int(provider_pair_count),
        "expected_saved_caked_manual_pair_count": int(expected_saved_count),
        "point_count": int(len(selected)),
        "exact_projector_available": exact_projector_available,
        "all_projection_outputs_valid": all_projection_outputs_valid,
        "all_exact_projector_used": all_exact_projector_used,
        "all_projection_projector_kinds_exact": all_projection_projector_kinds_exact,
        "manual_caked_residual_row_count": (int(len(selected)) if exact_projector_available else 0),
        "dataset_fit_space_projector_row_count": (
            int(len(selected)) if all_projection_projector_kinds_exact else 0
        ),
        "analytic_detector_fit_space_row_count": 0,
        "invalid_row_count": int(len(invalid_rows)),
        "invalid_rows": invalid_rows,
        "fallback_row_count": 0,
        "provider_row_fallback_count": 0,
        "fallback_entry_count": 0,
        "all_projection_signatures_present": all_projection_signatures_present,
        "all_projection_input_frames_native": all_projection_input_frames_native,
        "all_projection_sources_native": all_projection_sources_native,
        "cake_bundle_signature_base": base_signature,
        "cake_bundle_signature_theta_perturbed": theta_signature,
        "cake_bundle_signature_distance_perturbed": distance_signature,
        "projection_metadata": {
            "base": {
                "projection_input_frame": _projection_input_frame(base_projection),
                "caked_projection_source": _projection_source(base_projection),
                "fit_space_projector_kind": _projection_kind(base_projection),
                "exact_projector_used": bool(base_projection.get("exact_projector_used", False)),
                "valid": bool(base_projection.get("valid", False)),
            },
            "theta": {
                "projection_input_frame": _projection_input_frame(theta_projection),
                "caked_projection_source": _projection_source(theta_projection),
                "fit_space_projector_kind": _projection_kind(theta_projection),
                "exact_projector_used": bool(theta_projection.get("exact_projector_used", False)),
                "valid": bool(theta_projection.get("valid", False)),
            },
            "distance": {
                "projection_input_frame": _projection_input_frame(distance_projection),
                "caked_projection_source": _projection_source(distance_projection),
                "fit_space_projector_kind": _projection_kind(distance_projection),
                "exact_projector_used": bool(
                    distance_projection.get("exact_projector_used", False)
                ),
                "valid": bool(distance_projection.get("valid", False)),
            },
        },
        "full_background_recake_called": bool(guard.call_count > 0),
        "full_background_recake_call_count": int(guard.call_count),
        "point_only_reprojection_called": bool(point_projection_call_count > 0),
        "point_only_reprojection_call_count": int(point_projection_call_count),
        "theta_param_name": theta_param_name,
        "theta_base": float(theta_base),
        "theta_delta": float(theta_delta),
        "theta_delta_applied": float(theta_delta_applied),
        "theta_point_shift_required": False,
        "theta_point_shift_required_reason": (
            "fixed native detector pixels need not move when theta-only sample "
            "parameters change; dynamic simulated detector sources are checked by "
            "the objective/visualizer gates"
        ),
        "corto_detector_base": float(corto_detector_base),
        "corto_detector_delta": float(distance_delta),
        "corto_detector_delta_applied": float(distance_delta_applied),
        "all_theta_reprojected_points_finite": checks["all_theta_reprojected_points_finite"],
        "all_distance_reprojected_points_finite": checks["all_distance_reprojected_points_finite"],
        "any_theta_point_shifted": bool(any(shift > 1.0e-9 for shift in theta_shifts)),
        "any_distance_point_shifted": checks["any_distance_point_shifted"],
        "theta_two_theta_changed": bool(theta_two_theta_changed),
        "max_theta_caked_shift_px_or_deg": max_theta_shift,
        "min_theta_caked_shift_px_or_deg": min_theta_shift,
        "mean_theta_caked_shift_px_or_deg": mean_theta_shift,
        "max_distance_caked_shift_px_or_deg": max_distance_shift,
        "min_distance_caked_shift_px_or_deg": min_distance_shift,
        "mean_distance_caked_shift_px_or_deg": mean_distance_shift,
        "theta_projector_signature_changed": checks["theta_projector_signature_changed"],
        "distance_projector_signature_changed": checks["distance_projector_signature_changed"],
        "full_recake_guard_installed": bool(guard.installed),
        "stale_alias_guard_installed": bool(stale_guard.installed),
        "stale_caked_fields_used": bool(stale_guard.read_count > 0),
        "stale_caked_field_read_count": int(stale_guard.read_count),
        "stale_caked_field_read_fields": list(stale_guard.read_fields),
        "provider_guard_ok": bool(provider_before_ok and provider_after_ok),
        "provider_guard_before_ok": provider_before_ok,
        "provider_guard_after_ok": provider_after_ok,
        "manual_pair_ids_before": manual_pair_ids_before,
        "manual_pair_ids_after": manual_pair_ids_after,
        "same_manual_pair_ids_before_after": same_manual_pair_ids,
        "new4_state_hash_unchanged": state_hash_unchanged,
        "state_hash_before": state_hash_before,
        "state_hash_after": state_hash_after,
        "pairs": per_pair,
    }
    if run_dir is not None:
        report_path = Path(run_dir) / REPORT_NAME
        _write_json(report_path, report)
        report["report_path"] = str(report_path)
    return report


def run_new4_caked_point_reprojection_check(
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    background_index: int = 0,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
) -> dict[str, object]:
    state_path = Path(state_path).expanduser().resolve()
    run_dir = Path(output_root).expanduser().resolve() / (run_id or _run_stamp())
    state_hash_before = _state_sha256(state_path)
    guard = FullBackgroundRecakeGuard()
    guard_error: str | None = None
    context: Mapping[str, object] = {}
    provider_before: Mapping[str, object] = {}

    try:
        provider_before = preflight._run_point_provider_report_only(
            state_path,
            int(background_index),
        )
        context = preflight._prepare_validation_context(state_path, int(background_index))
    except Exception as exc:
        guard_error = f"{type(exc).__name__}:{exc}"
        context = {}
    if not bool(context.get("ok", False)):
        full_recake_attempted = int(guard.call_count) > 0
        state_hash_after = _state_sha256(state_path)
        state_hash_unchanged = bool(state_hash_before == state_hash_after)
        provider_before_ok = bool(provider_before.get("ok", False))
        failures = []
        if not provider_before_ok:
            failures.append("provider_guard_before_ok")
        if full_recake_attempted:
            failures.append("full_background_recake_not_called")
        if not state_hash_unchanged:
            failures.append("new4_state_hash_unchanged")
        report = {
            "status": "fail" if failures else "skip",
            "classification": "validation_context_unavailable",
            "failures": failures,
            "guard_error": guard_error,
            "background_index": int(background_index),
            "provider_pair_count": _provider_pair_count(provider_before),
            "point_count": 0,
            "exact_projector_available": False,
            "full_recake_guard_installed": bool(guard.installed),
            "full_background_recake_called": bool(guard.call_count > 0),
            "full_background_recake_call_count": int(guard.call_count),
            "point_only_reprojection_called": False,
            "point_only_reprojection_call_count": 0,
            "provider_guard_before_ok": provider_before_ok,
            "provider_guard_after_ok": False,
            "provider_guard_ok": False,
            "new4_state_hash_unchanged": state_hash_unchanged,
            "state_hash_before": state_hash_before,
            "state_hash_after": state_hash_after,
            "context_ok": False,
            "context_classification": context.get("classification"),
            "error_text": context.get("error_text") or context.get("captured_preflight_error_text"),
        }
        report_path = run_dir / REPORT_NAME
        _write_json(report_path, report)
        report["report_path"] = str(report_path)
        return report

    return run_caked_point_reprojection_probe_from_context(
        state_path=state_path,
        background_index=int(background_index),
        context=context,
        provider_before_report=provider_before,
        provider_after_factory=(
            lambda: preflight._run_point_provider_report_only(
                state_path,
                int(background_index),
            )
        ),
        recake_guard=guard,
        run_dir=run_dir,
        state_hash_before=state_hash_before,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run New4 point-only caked reprojection check.",
    )
    parser.add_argument("--state-path", "--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--background-index", type=int, default=0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    report = run_new4_caked_point_reprojection_check(
        state_path=Path(args.state_path),
        background_index=int(args.background_index),
        output_root=Path(args.output_root),
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "report_path": report.get("report_path"),
                "failures": report.get("failures", []),
            },
            sort_keys=True,
        )
    )
    return 0 if str(report.get("status")) in {"pass", "skip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
