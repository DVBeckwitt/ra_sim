"""Internal geometry-fit worker context helpers."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GeometryFitWorkerCakedPayloadDeps:
    normalize_caked_view_payload: Callable[..., object]
    hydrate_exact_caked_payload: Callable[..., object]
    caked_projection_payload: Callable[..., object]
    projection_payload_storage_copy: Callable[..., object]
    projection_payload_digest: Callable[..., object]
    detector_shape_2d: Callable[..., object]
    resolve_targeted_caked_projection_payload: Callable[..., object]
    caking_integrator: Callable[..., object]
    emit_stage_event: Callable[..., object]
    is_transform_bundle: Callable[[object], bool]


@dataclass(frozen=True)
class GeometryFitWorkerSourceProjectionDeps:
    rows_for_background: Callable[..., object]
    group_rows_by_background: Callable[..., object]
    make_projection_callbacks: Callable[..., object]
    native_detector_coords_to_bundle_detector_coords: Callable[..., object]
    raw_phi_to_gui_phi: Callable[..., object]
    rotate_point_for_display: Callable[..., object]
    native_sim_to_display_coords: Callable[..., object] | None
    get_detector_angular_maps: Callable[..., object] | None
    detector_pixel_to_scattering_angles: Callable[..., object] | None
    backend_detector_coords_to_native_detector_coords: Callable[..., object]
    scattering_angles_to_detector_pixel: Callable[..., object] | None
    profile_cache: Callable[..., object]
    default_pixel_size_m: float
    default_image_size: int
    default_display_rotate_k: int


@dataclass(frozen=True)
class GeometryFitWorkerCacheBundleDeps:
    is_background_cache_bundle: Callable[[object], bool]
    copy_source_rows: Callable[..., object]
    make_background_cache_bundle: Callable[..., object]
    copy_optional_values: Callable[..., object]


@dataclass
class GeometryFitWorkerContext:
    job_data: dict[str, object]
    job_id: int
    event_queue: object | None
    worker_source_row_snapshots: dict[int, dict[str, object]]
    worker_source_snapshot_diagnostics: dict[str, object]
    worker_simulation_diagnostics: dict[str, object]
    worker_background_cache_by_index: dict[int, object]
    source_cache_generation_by_background: dict[int, object]
    source_cache_generation_lock: threading.Lock = field(default_factory=threading.Lock)
    caked_payload_deps: GeometryFitWorkerCakedPayloadDeps | None = None
    source_projection_deps: GeometryFitWorkerSourceProjectionDeps | None = None
    cache_bundle_deps: GeometryFitWorkerCacheBundleDeps | None = None

    @classmethod
    def from_job(cls, job: Mapping[str, object]) -> GeometryFitWorkerContext:
        job_data = dict(job or {})
        source_cache_generation_by_background = dict(
            job_data.get("source_cache_generation_by_background", {}) or {}
        )
        job_data["source_cache_generation_by_background"] = dict(
            source_cache_generation_by_background
        )
        return cls(
            job_data=job_data,
            job_id=int(job_data.get("job_id", -1)),
            event_queue=job_data.get("event_queue"),
            worker_source_row_snapshots={
                int(idx): copy.deepcopy(snapshot)
                for idx, snapshot in dict(job_data.get("source_snapshots", {}) or {}).items()
            },
            worker_source_snapshot_diagnostics=copy.deepcopy(
                job_data.get("source_snapshot_diagnostics") or {}
            ),
            worker_simulation_diagnostics=copy.deepcopy(
                job_data.get("simulation_diagnostics") or {}
            ),
            worker_background_cache_by_index={},
            source_cache_generation_by_background=source_cache_generation_by_background,
        )

    def emit_event(self, kind: str, payload: object = None) -> None:
        if self.event_queue is None:
            return
        try:
            self.event_queue.put(
                {
                    "job_id": int(self.job_id),
                    "kind": str(kind),
                    "payload": copy.deepcopy(payload),
                }
            )
        except Exception:
            return

    def current_source_cache_generation(self, background_index: int) -> int:
        with self.source_cache_generation_lock:
            return int(
                self.source_cache_generation_by_background.get(int(background_index), 0)
            )

    def advance_source_cache_generation(self, background_index: int) -> int:
        with self.source_cache_generation_lock:
            next_generation = (
                int(self.source_cache_generation_by_background.get(int(background_index), 0)) + 1
            )
            self.source_cache_generation_by_background[int(background_index)] = int(
                next_generation
            )
            self.job_data["source_cache_generation_by_background"] = dict(
                self.source_cache_generation_by_background
            )
            return int(next_generation)

    def source_cache_generation_matches(
        self,
        background_index: int,
        generation_id: int | None,
    ) -> bool:
        if generation_id is None:
            return True
        return int(self.current_source_cache_generation(background_index)) == int(generation_id)

    def set_worker_source_snapshot_diagnostics(self, **kwargs: object) -> None:
        self.worker_source_snapshot_diagnostics.clear()
        self.worker_source_snapshot_diagnostics.update(kwargs)

    def last_worker_source_snapshot_diagnostics(self) -> dict[str, object]:
        return dict(self.worker_source_snapshot_diagnostics)

    def last_worker_simulation_diagnostics(self) -> dict[str, object]:
        return dict(self.worker_simulation_diagnostics)

    def load_background_by_index_snapshot(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        background_payload = dict(
            dict(self.job_data.get("background_images", {}) or {}).get(int(index)) or {}
        )
        return (
            np.asarray(background_payload.get("native"), dtype=np.float64).copy(),
            np.asarray(background_payload.get("display"), dtype=np.float64).copy(),
        )

    def _require_caked_payload_deps(self) -> GeometryFitWorkerCakedPayloadDeps:
        if self.caked_payload_deps is None:
            raise RuntimeError("geometry-fit worker caked payload deps are not configured")
        return self.caked_payload_deps

    def _require_source_projection_deps(self) -> GeometryFitWorkerSourceProjectionDeps:
        if self.source_projection_deps is None:
            raise RuntimeError(
                "geometry-fit worker source projection deps are not configured"
            )
        return self.source_projection_deps

    def _require_cache_bundle_deps(self) -> GeometryFitWorkerCacheBundleDeps:
        if self.cache_bundle_deps is None:
            raise RuntimeError("geometry-fit worker cache bundle deps are not configured")
        return self.cache_bundle_deps

    def caked_projection_payload_status(
        self,
        payload: Mapping[str, object] | None,
    ) -> str:
        deps = self._require_caked_payload_deps()
        if not isinstance(payload, Mapping):
            return "projection_payload_missing_axes"
        try:
            radial_axis = np.asarray(
                payload.get("radial_axis", payload.get("radial")),
                dtype=np.float64,
            ).reshape(-1)
            azimuth_axis = np.asarray(
                payload.get("azimuth_axis", payload.get("azimuth")),
                dtype=np.float64,
            ).reshape(-1)
            raw_azimuth_axis = np.asarray(
                payload.get("raw_azimuth_axis", payload.get("raw_azimuth")),
                dtype=np.float64,
            ).reshape(-1)
        except Exception:
            return "projection_payload_missing_axes"
        if radial_axis.size <= 0 or azimuth_axis.size <= 0 or raw_azimuth_axis.size <= 0:
            return "projection_payload_missing_axes"
        if (
            raw_azimuth_axis.shape != azimuth_axis.shape
            or not np.all(np.isfinite(radial_axis))
            or not np.all(np.isfinite(azimuth_axis))
            or not np.all(np.isfinite(raw_azimuth_axis))
        ):
            return "projection_payload_axis_mismatch"
        if not deps.is_transform_bundle(payload.get("transform_bundle")):
            return "missing_exact_caked_bundle"
        if deps.projection_payload_digest(payload) is None:
            return "missing_exact_caked_bundle"
        return "projection_payload_ready"

    def projection_candidate_state(
        self,
        payload: Mapping[str, object] | None,
        *,
        detector_shape: Sequence[object] | None = None,
    ) -> tuple[dict[str, object] | None, str]:
        deps = self._require_caked_payload_deps()
        projection = deps.caked_projection_payload(payload)
        if not isinstance(projection, Mapping):
            return None, "absent"
        requested_shape = deps.detector_shape_2d(detector_shape)
        candidate_shape = deps.detector_shape_2d(projection.get("detector_shape"))
        transform_bundle = projection.get("transform_bundle")
        if candidate_shape is None and deps.is_transform_bundle(transform_bundle):
            candidate_shape = deps.detector_shape_2d(
                getattr(transform_bundle, "detector_shape", None)
            )
        if (
            requested_shape is not None
            and candidate_shape is not None
            and tuple(requested_shape) != tuple(candidate_shape)
        ):
            return dict(projection), "invalid"
        status = self.caked_projection_payload_status(projection)
        if status == "projection_payload_ready":
            return dict(projection), "ready"
        return dict(projection), "invalid"

    def load_caked_view_by_index_snapshot(
        self,
        index: int,
    ) -> dict[str, object] | None:
        deps = self._require_caked_payload_deps()
        caked_payload = dict(
            dict(self.job_data.get("caked_views_by_background", {}) or {}).get(int(index))
            or {}
        )
        if not caked_payload:
            return None
        normalized_payload = deps.normalize_caked_view_payload(
            caked_payload,
            detector_shape=caked_payload.get("detector_shape"),
            ai=deps.caking_integrator(),
        )
        hydrated_payload = deps.hydrate_exact_caked_payload(
            normalized_payload,
            detector_shape=(
                normalized_payload.get("detector_shape")
                if isinstance(normalized_payload, Mapping)
                else caked_payload.get("detector_shape")
            ),
            params=dict(self.job_data.get("params", {}) or {}),
            require_background=True,
        )
        if not isinstance(hydrated_payload, dict):
            return None
        self.job_data.setdefault("caked_views_by_background", {})[int(index)] = (
            hydrated_payload
        )
        if str(self.job_data.get("projection_view_mode") or "").strip().lower() == "caked":
            projection_payload = deps.caked_projection_payload(hydrated_payload)
            stored_projection_payload = deps.projection_payload_storage_copy(
                projection_payload
            )
            if isinstance(stored_projection_payload, Mapping):
                self.job_data.setdefault("projection_payload_by_background", {})[
                    int(index)
                ] = stored_projection_payload
        return hydrated_payload

    def load_caked_projection_by_index_snapshot(
        self,
        index: int,
        *,
        detector_shape: Sequence[object] | None = None,
        allow_generated_payload: bool = False,
    ) -> dict[str, object] | None:
        deps = self._require_caked_payload_deps()
        background_idx = int(index)
        payload_map = self.job_data.setdefault("projection_payload_by_background", {})

        def _hydrate_store_return(
            projection_payload: Mapping[str, object],
        ) -> dict[str, object] | None:
            normalized_payload = deps.normalize_caked_view_payload(
                projection_payload,
                detector_shape=detector_shape or projection_payload.get("detector_shape"),
                ai=deps.caking_integrator(),
            )
            if not isinstance(normalized_payload, Mapping):
                return None
            hydrated_payload = deps.hydrate_exact_caked_payload(
                deps.caked_projection_payload(normalized_payload),
                detector_shape=(
                    detector_shape
                    or normalized_payload.get("detector_shape")
                    or projection_payload.get("detector_shape")
                ),
                params=dict(self.job_data.get("params", {}) or {}),
                require_background=False,
            )
            if not isinstance(hydrated_payload, Mapping) or not deps.is_transform_bundle(
                hydrated_payload.get("transform_bundle")
            ):
                return None
            stored_payload = deps.caked_projection_payload(hydrated_payload)
            stored_payload = deps.projection_payload_storage_copy(stored_payload)
            if not isinstance(stored_payload, Mapping) or not deps.is_transform_bundle(
                stored_payload.get("transform_bundle")
            ):
                return None
            if deps.projection_payload_digest(stored_payload) is None:
                return None
            payload_map[background_idx] = stored_payload
            return dict(stored_payload)

        projection_payload, state = self.projection_candidate_state(
            payload_map.get(background_idx),
            detector_shape=detector_shape,
        )
        if state == "ready" and isinstance(projection_payload, Mapping):
            return _hydrate_store_return(projection_payload)
        if state == "invalid":
            return None

        caked_payload = dict(
            dict(self.job_data.get("caked_views_by_background", {}) or {}).get(
                background_idx
            )
            or {}
        )
        projection_payload, state = self.projection_candidate_state(
            caked_payload,
            detector_shape=detector_shape,
        )
        if state == "ready" and isinstance(projection_payload, Mapping):
            return _hydrate_store_return(projection_payload)
        if state == "invalid":
            return None

        if bool(allow_generated_payload):
            generated_payload = deps.resolve_targeted_caked_projection_payload(
                background_idx,
                detector_shape=detector_shape,
                ai=deps.caking_integrator(),
                analysis_preview_bins=self.job_data.get("analysis_bins"),
                allow_generated_payload=True,
            )
            projection_payload, state = self.projection_candidate_state(
                generated_payload,
                detector_shape=detector_shape,
            )
            if state == "ready" and isinstance(projection_payload, Mapping):
                return _hydrate_store_return(projection_payload)
        return None

    def ensure_worker_caked_projection_payload(
        self,
        background_index: int,
        *,
        detector_shape: Sequence[object] | None = None,
        stage_callback: object | None = None,
        emit_event: bool = False,
    ) -> dict[str, object] | None:
        deps = self._require_caked_payload_deps()
        projection_payload = self.load_caked_projection_by_index_snapshot(
            int(background_index),
            detector_shape=detector_shape,
            allow_generated_payload=True,
        )
        status = self.caked_projection_payload_status(projection_payload)
        if bool(emit_event):
            deps.emit_stage_event(
                stage_callback,
                str(status),
                background_index=int(background_index),
                background_label=str(
                    dict(self.job_data.get("background_labels", {}) or {}).get(
                        int(background_index),
                        f"background {int(background_index) + 1}",
                    )
                ),
                status=str(status),
                payload_kind="projection",
                message=(
                    f"preflight: exact caked projection payload ready for background {int(background_index) + 1}"
                    if status == "projection_payload_ready"
                    else f"preflight: exact caked projection payload failed for background {int(background_index) + 1} (status={status})"
                ),
            )
        if status != "projection_payload_ready":
            return None
        return projection_payload

    def project_source_rows_for_background(
        self,
        background_index: int,
        raw_rows: Sequence[object] | None,
        *,
        mode_override: str | None = None,
        strict_caked_projection: bool = True,
        params_override: Mapping[str, object] | None = None,
    ) -> Sequence[object]:
        deps = self._require_source_projection_deps()
        normalized_rows = deps.rows_for_background(background_index, raw_rows)
        if not normalized_rows:
            return []
        normalized_mode = (
            str(
                mode_override
                if mode_override is not None
                else self.job_data.get("projection_view_mode") or "detector"
            )
            .strip()
            .lower()
        )
        if normalized_mode not in {"detector", "caked", "q_space"}:
            normalized_mode = "detector"
        current_background_index = int(self.job_data.get("current_background_index", 0))
        is_current_background = int(background_index) == int(current_background_index)
        if normalized_mode == "q_space":
            if not is_current_background:
                return []
            projector = self.job_data.get("project_rows")
            if not callable(projector):
                return []
            try:
                return deps.rows_for_background(
                    background_index,
                    projector(normalized_rows),
                )
            except Exception:
                return []

        background_payload = dict(
            dict(self.job_data.get("background_images", {}) or {}).get(
                int(background_index)
            )
            or {}
        )
        native_background = background_payload.get("native")
        display_background = background_payload.get("display")
        try:
            detector_shape = tuple(
                int(v) for v in np.asarray(native_background, dtype=np.float64).shape[:2]
            )
        except Exception:
            detector_shape = None
        params_local = dict(self.job_data.get("params", {}) or {})
        if isinstance(params_override, Mapping):
            params_local.update(dict(params_override))
        resolved_caked_payload = None
        exact_caked_bundle = None
        caked_deps = self.caked_payload_deps
        if normalized_mode == "caked":
            caked_deps = self._require_caked_payload_deps()
            payload_map = self.job_data.setdefault("projection_payload_by_background", {})
            resolved_caked_payload = self.load_caked_projection_by_index_snapshot(
                int(background_index),
                detector_shape=detector_shape,
                allow_generated_payload=True,
            )
            if not isinstance(resolved_caked_payload, Mapping):
                if not bool(strict_caked_projection):
                    return []
                raise RuntimeError(
                    "exact caked projector unavailable for background "
                    f"{int(background_index) + 1}"
                )
            hydrated_caked_payload = caked_deps.hydrate_exact_caked_payload(
                resolved_caked_payload,
                detector_shape=detector_shape,
                params=params_local,
                require_background=False,
            )
            if (
                not isinstance(hydrated_caked_payload, Mapping)
                or not caked_deps.is_transform_bundle(
                    hydrated_caked_payload.get("transform_bundle")
                )
            ):
                if not bool(strict_caked_projection):
                    return []
                raise RuntimeError(
                    "exact caked projector unavailable for background "
                    f"{int(background_index) + 1}"
                )
            resolved_caked_payload = hydrated_caked_payload
            exact_caked_bundle = hydrated_caked_payload.get("transform_bundle")
            stored_projection_payload = caked_deps.caked_projection_payload(
                hydrated_caked_payload
            )
            stored_projection_payload = caked_deps.projection_payload_storage_copy(
                stored_projection_payload
            )
            if isinstance(stored_projection_payload, Mapping):
                payload_map[int(background_index)] = stored_projection_payload

        center_value = params_local.get("center")
        if isinstance(center_value, Sequence) and len(center_value) >= 2:
            try:
                center_pair = [float(center_value[0]), float(center_value[1])]
            except Exception:
                center_pair = None
        else:
            center_pair = None

        def _background_native_detector_coords_to_bundle_detector_coords(
            col: float,
            row: float,
        ) -> tuple[float | None, float | None]:
            return deps.native_detector_coords_to_bundle_detector_coords(
                float(col),
                float(row),
                detector_shape,
            )

        def _has_live_caked_handoff_rows(rows: Sequence[Mapping[str, object]]) -> bool:
            if not rows:
                return False
            for row in rows:
                if row.get("source_label") is None:
                    return False
                try:
                    caked_x = float(row.get("caked_x", row.get("background_two_theta_deg")))
                    caked_y = float(row.get("caked_y", row.get("background_phi_deg")))
                except Exception:
                    return False
                if not (np.isfinite(caked_x) and np.isfinite(caked_y)):
                    return False
            return True

        projection_callbacks = deps.make_projection_callbacks(
            caked_view_enabled=lambda: bool(
                normalized_mode == "caked" and isinstance(resolved_caked_payload, Mapping)
            ),
            last_caked_background_image_unscaled=lambda: (
                resolved_caked_payload.get("background")
                if isinstance(resolved_caked_payload, Mapping)
                else None
            ),
            last_caked_radial_values=lambda: (
                resolved_caked_payload.get("radial_axis")
                if isinstance(resolved_caked_payload, Mapping)
                else None
            ),
            last_caked_azimuth_values=lambda: (
                resolved_caked_payload.get("azimuth_axis")
                if isinstance(resolved_caked_payload, Mapping)
                else None
            ),
            current_background_display=lambda: display_background,
            current_background_native=lambda: native_background,
            ai=(
                caked_deps.caking_integrator
                if caked_deps is not None
                else (lambda: None)
            ),
            center=lambda: center_pair,
            detector_distance=lambda: float(
                params_local.get("corto_detector", 0.0) or 0.0
            ),
            pixel_size=float(
                params_local.get("pixel_size_m", deps.default_pixel_size_m)
                or deps.default_pixel_size_m
            ),
            caked_transform_bundle=lambda: (
                exact_caked_bundle
                if caked_deps is not None
                and caked_deps.is_transform_bundle(
                    exact_caked_bundle
                )
                else None
            ),
            wrap_phi_range=deps.raw_phi_to_gui_phi,
            rotate_point_for_display=deps.rotate_point_for_display,
            display_rotate_k=int(
                self.job_data.get("display_rotate_k", deps.default_display_rotate_k)
            ),
            current_background_index=lambda: int(background_index),
            caked_projection_payload=lambda: resolved_caked_payload,
            current_geometry_fit_params=lambda: dict(params_local),
            build_live_preview_simulated_peaks_from_cache=lambda: [],
            ensure_peak_overlay_data=lambda **_kwargs: False,
            miller=lambda: self.job_data["solver_inputs"].miller,
            intensities=lambda: self.job_data["solver_inputs"].intensities,
            image_size=int(self.job_data.get("image_size", deps.default_image_size)),
            display_to_native_sim_coords=self.job_data.get("display_to_native_sim_coords"),
            native_sim_to_display_coords=deps.native_sim_to_display_coords,
            native_detector_coords_to_detector_display_coords=(
                _background_native_detector_coords_to_bundle_detector_coords
            ),
            get_detector_angular_maps=(
                (lambda ai_value: deps.get_detector_angular_maps(ai_value))
                if callable(deps.get_detector_angular_maps)
                else (lambda _ai_value: None)
            ),
            detector_pixel_to_scattering_angles=(
                deps.detector_pixel_to_scattering_angles
            ),
            backend_detector_coords_to_native_detector_coords=(
                deps.backend_detector_coords_to_native_detector_coords
            ),
            native_detector_coords_to_bundle_detector_coords=(
                _background_native_detector_coords_to_bundle_detector_coords
            ),
            bundle_detector_coords_to_background_display_coords=(
                lambda col, row: (float(col), float(row))
            ),
            scattering_angles_to_detector_pixel=deps.scattering_angles_to_detector_pixel,
            filter_simulated_peaks=(
                lambda peaks, *_args, **_kwargs: (list(peaks or ()), [], 0)
            ),
            collapse_simulated_peaks=(
                lambda peaks, *_args, **_kwargs: (list(peaks or ()), 0)
            ),
            profile_cache=deps.profile_cache,
        )
        try:
            projected_rows = deps.rows_for_background(
                background_index,
                projection_callbacks.project_peaks_to_current_view(normalized_rows),
            )
            # Current-background live rows may already carry exact caked anchors.
            # Keep those rows when detector projection has nothing to rebuild.
            if (
                not projected_rows
                and normalized_mode == "detector"
                and _has_live_caked_handoff_rows(normalized_rows)
            ):
                return normalized_rows
            return projected_rows
        except Exception:
            if normalized_mode == "q_space":
                return []
            if normalized_mode == "caked":
                if bool(strict_caked_projection):
                    raise
                return []
            return normalized_rows

    def project_source_rows_by_row_background(
        self,
        raw_rows: Sequence[object] | None,
    ) -> list[dict[str, object]]:
        deps = self._require_source_projection_deps()
        order_key = "__ra_sim_projection_row_order__"
        grouped_rows = deps.group_rows_by_background(
            raw_rows,
            default_background_index=int(self.job_data.get("current_background_index", 0)),
            order_key=order_key,
        )
        if not grouped_rows:
            return []
        projected_rows: list[dict[str, object]] = []
        for background_index, rows_for_background in grouped_rows:
            projected_rows.extend(
                dict(entry)
                for entry in (
                    self.project_source_rows_for_background(
                        int(background_index),
                        rows_for_background,
                    )
                    or ()
                )
                if isinstance(entry, Mapping)
            )
        sorted_rows = sorted(
            projected_rows,
            key=lambda entry: (
                int(entry.get(order_key)) if entry.get(order_key) is not None else int(1e12)
            ),
        )
        cleaned_rows: list[dict[str, object]] = []
        for raw_entry in sorted_rows:
            entry = dict(raw_entry)
            entry.pop(order_key, None)
            cleaned_rows.append(entry)
        return cleaned_rows

    def mark_worker_cached_projection_rows(
        self,
        rows: list[dict[str, object]],
        *,
        background_index: int,
        mode: str,
    ) -> list[dict[str, object]]:
        if mode not in {"caked", "q_space"}:
            return rows
        for row in rows:
            row["_geometry_fit_worker_cached_projection"] = True
            row["_geometry_fit_worker_projection_mode"] = mode
            row["_geometry_fit_worker_projection_background_index"] = int(
                background_index
            )
        return rows

    def worker_cached_projection_rows_match(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        background_index: int,
        mode: str,
    ) -> bool:
        if mode not in {"caked", "q_space"} or not rows:
            return False
        expected_background_index = int(background_index)
        for row in rows:
            try:
                row_background_index = int(
                    row.get("_geometry_fit_worker_projection_background_index")
                )
            except Exception:
                return False
            if (
                row.get("_geometry_fit_worker_cached_projection") is not True
                or row_background_index != expected_background_index
                or str(row.get("_geometry_fit_worker_projection_mode") or "").lower()
                != mode
            ):
                return False
        return True

    def bundle_rows(
        self,
        bundle: object | None,
        *,
        mode_override: str | None = None,
        params_override: Mapping[str, object] | None = None,
    ) -> list[dict[str, object]]:
        cache_deps = self._require_cache_bundle_deps()
        source_deps = self._require_source_projection_deps()
        if not cache_deps.is_background_cache_bundle(bundle):
            return []
        background_index = int(bundle.background_index)
        base_mode = str(self.job_data.get("projection_view_mode") or "detector").strip().lower()
        normalized_mode = str(mode_override or base_mode or "detector").strip().lower()
        if normalized_mode not in {"detector", "caked", "q_space"}:
            normalized_mode = "detector"
        rows = source_deps.rows_for_background(
            background_index,
            bundle.projected_rows,
        )
        if rows and self.worker_cached_projection_rows_match(
            rows,
            background_index=background_index,
            mode=normalized_mode,
        ):
            return [dict(entry) for entry in rows if isinstance(entry, Mapping)]
        if rows and params_override is None and normalized_mode == base_mode:
            return self.mark_worker_cached_projection_rows(
                rows,
                background_index=background_index,
                mode=normalized_mode,
            )
        if normalized_mode in {"caked", "q_space"}:
            projected_rows = source_deps.rows_for_background(
                background_index,
                self.project_source_rows_for_background(
                    background_index,
                    bundle.stored_rows,
                    mode_override=normalized_mode,
                    params_override=params_override,
                ),
            )
            if projected_rows:
                return self.mark_worker_cached_projection_rows(
                    projected_rows,
                    background_index=background_index,
                    mode=normalized_mode,
                )
            return []
        return source_deps.rows_for_background(background_index, bundle.stored_rows)

    def store_worker_background_cache_bundle(self, bundle: object) -> int:
        cache_deps = self._require_cache_bundle_deps()
        background_index = int(bundle.background_index)
        self.worker_background_cache_by_index[background_index] = bundle
        projection_signatures = dict(
            self.job_data.get("projection_view_signature_by_background", {}) or {}
        )
        self.worker_source_row_snapshots[background_index] = {
            "background_index": background_index,
            "simulation_signature": bundle.requested_signature,
            "rows": cache_deps.copy_source_rows(bundle.stored_rows),
            "stored_rows": cache_deps.copy_source_rows(bundle.stored_rows),
            "projected_rows": cache_deps.copy_source_rows(bundle.projected_rows),
            "row_count": int(len(bundle.stored_rows or ())),
            "projected_row_count": int(len(bundle.projected_rows or ())),
            "created_from": str(bundle.cache_source or "geometry_fit_background_cache"),
            "requested_signature": bundle.requested_signature,
            "requested_signature_summary": bundle.requested_signature_summary,
            "diagnostics": copy.deepcopy(dict(bundle.diagnostics or {})),
            "projection_view_signature": copy.deepcopy(
                projection_signatures.get(background_index)
            ),
            "valid_for_picker": bool(bundle.stored_rows),
            "valid_for_geometry_fit_dataset": bool(
                bundle.stored_rows or bundle.projected_rows
            ),
        }
        return self.advance_source_cache_generation(background_index)

    def build_geometry_fit_background_cache_bundle(
        self,
        *,
        background_index: int,
        background_label: str,
        requested_signature: object,
        requested_signature_summary: object,
        theta_base: float,
        theta_initial: float,
        stored_rows: Sequence[object] | None,
        projected_rows: Sequence[object] | None = None,
        cache_source: str,
        diagnostics: Mapping[str, object] | None = None,
        peak_table_lattice: Sequence[object] | None = None,
        hit_tables: Sequence[object] | None = None,
        intersection_cache: Sequence[object] | None = None,
        cache_metadata: Mapping[str, object] | None = None,
    ) -> object:
        cache_deps = self._require_cache_bundle_deps()
        source_deps = self._require_source_projection_deps()
        copied_stored_rows = source_deps.rows_for_background(
            int(background_index),
            stored_rows,
        )
        copied_projected_rows = source_deps.rows_for_background(
            int(background_index),
            projected_rows,
        )
        normalized_mode = str(
            self.job_data.get("projection_view_mode") or "detector"
        ).strip().lower()
        strict_projection_mode = normalized_mode in {"caked", "q_space"}
        projection_failure_reason: str | None = None
        projected_from_stored_rows = False
        if not copied_projected_rows:
            if strict_projection_mode:
                try:
                    copied_projected_rows = source_deps.rows_for_background(
                        int(background_index),
                        self.project_source_rows_for_background(
                            int(background_index),
                            copied_stored_rows,
                            mode_override=normalized_mode,
                            strict_caked_projection=True,
                        ),
                    )
                    projected_from_stored_rows = bool(copied_projected_rows)
                except Exception as exc:
                    copied_projected_rows = []
                    projection_failure_reason = f"projection_error:{type(exc).__name__}"
            else:
                copied_projected_rows = source_deps.rows_for_background(
                    int(background_index),
                    self.project_source_rows_for_background(
                        int(background_index),
                        copied_stored_rows,
                    ),
                )
        resolved_diagnostics = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        resolved_diagnostics.setdefault("source", "geometry_fit_background_cache")
        resolved_diagnostics.setdefault(
            "cache_family",
            "geometry_fit_background_cache",
        )
        resolved_diagnostics.setdefault("action", "prepare")
        resolved_diagnostics.setdefault("status", "background_cache_ready")
        resolved_diagnostics.setdefault("background_index", int(background_index))
        resolved_diagnostics.setdefault("background_label", str(background_label))
        resolved_diagnostics.setdefault("requested_signature", requested_signature)
        resolved_diagnostics.setdefault(
            "requested_signature_summary",
            requested_signature_summary,
        )
        resolved_diagnostics.setdefault("snapshot_signature", requested_signature)
        resolved_diagnostics.setdefault(
            "stored_signature_summary",
            requested_signature_summary,
        )
        resolved_diagnostics.setdefault("theta_base", float(theta_base))
        resolved_diagnostics.setdefault("theta_initial", float(theta_initial))
        resolved_diagnostics.setdefault("raw_peak_count", int(len(copied_stored_rows)))
        resolved_diagnostics.setdefault(
            "projected_peak_count",
            int(len(copied_projected_rows)),
        )
        if projected_from_stored_rows:
            resolved_diagnostics["projected_peak_count"] = int(
                len(copied_projected_rows)
            )
            resolved_diagnostics.setdefault(
                "projected_rows_generated_from_stored_rows",
                True,
            )
        if projection_failure_reason is not None:
            resolved_diagnostics.setdefault(
                "projection_failure_reason",
                str(projection_failure_reason),
            )
        resolved_diagnostics.setdefault("cache_source", str(cache_source))
        resolved_diagnostics.setdefault("signature_match", True)
        resolved_diagnostics.setdefault(
            "live_cache_inventory",
            copy.deepcopy(self.job_data.get("live_cache_inventory", {})),
        )

        return cache_deps.make_background_cache_bundle(
            background_index=int(background_index),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            background_label=str(background_label),
            theta_base=float(theta_base),
            theta_initial=float(theta_initial),
            projected_rows=copied_projected_rows,
            stored_rows=copied_stored_rows,
            cache_source=str(cache_source),
            diagnostics=resolved_diagnostics,
            peak_table_lattice=cache_deps.copy_optional_values(peak_table_lattice),
            hit_tables=cache_deps.copy_optional_values(hit_tables),
            intersection_cache=cache_deps.copy_optional_values(intersection_cache),
            cache_metadata=(
                dict(cache_metadata) if isinstance(cache_metadata, Mapping) else None
            ),
        )
