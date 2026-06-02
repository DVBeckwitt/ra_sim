"""Internal geometry-fit worker context helpers."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter

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
class GeometryFitWorkerCakedViewStorageDeps:
    emit_geometry_fit_stage_event: Callable[..., object]
    build_caked_roi_selection: Callable[..., object]
    caked_roi_fit_space_to_detector_point: Callable[..., object]
    worker_projection_analysis_bins: Callable[..., object]
    geometry_fit_worker_caked_projection_view: Callable[..., object]
    temporary_numba_thread_limit: Callable[..., object]
    default_reserved_cpu_worker_count: Callable[..., object]
    caking: Callable[..., object]
    prepare_caked_display_payload: Callable[..., object]
    sanitize_caked_display_payload: Callable[..., object]
    normalize_projection_view_signature: Callable[..., object]
    targeted_projection_view_signature: Callable[..., object]
    digest_payload: Callable[..., object]
    replace_bundle: Callable[..., object]


@dataclass(frozen=True)
class GeometryFitWorkerSourceProjectionDeps:
    rows_for_background: Callable[..., object]
    group_rows_by_background: Callable[..., object]
    overlay_state_finite_pair: Callable[..., object]
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


@dataclass(frozen=True)
class GeometryFitWorkerPrebuildDeps:
    theta_initial_for_background: Callable[..., object]
    int_keyed_mapping: Callable[..., object]
    live_cache_signature_summary: Callable[..., object]
    cache_jsonable: Callable[..., object]
    digest_payload: Callable[..., object]
    collect_required_manual_fit_targets: Callable[..., object]
    required_branch_group_keys: Callable[..., object]
    live_row_source_counts: Callable[..., object]
    validate_required_source_rows_for_fit_space: Callable[..., object]
    projection_view_signature_for_background: Callable[..., object]
    logged_intersection_cache_loaders: Callable[..., object]
    copy_intersection_cache_tables: Callable[..., object]
    logged_cache_matches_params: Callable[..., object]
    forward_source_rows_for_rebuild: Callable[..., object]
    build_source_rows_for_rebuild: Callable[..., object]
    simulate_hit_tables_for_fit: Callable[..., object]
    load_targeted_projected_cache_entry: Callable[..., object]
    store_targeted_projected_cache_entry: Callable[..., object]
    rebuild_geometry_fit_source_rows: Callable[..., object]
    hydrate_exact_caked_payload: Callable[..., object]
    projection_payload_storage_copy: Callable[..., object]
    is_transform_bundle: Callable[[object], bool]
    live_handoff_patch_marker: str


@dataclass(frozen=True)
class GeometryFitWorkerSourceRowsDeps:
    manual_caked_fit_space_required_for_background: Callable[..., bool]
    theta_base_for_background: Callable[..., object]


@dataclass(frozen=True)
class GeometryFitWorkerRequiredCacheDeps:
    locked_qr_fit_space_projection_readiness: Callable[..., object]
    normalize_optics_mode_label: Callable[..., str]
    start_non_gating_caked_view_task: Callable[..., object]
    emit_source_cache_caked_view_event: Callable[..., object]


@dataclass(frozen=True)
class GeometryFitWorkerDatasetDeps:
    make_manual_dataset_bindings: Callable[..., object]
    prepare_geometry_fit_run: Callable[..., object]
    build_geometry_manual_fit_dataset: Callable[..., object]


@dataclass(frozen=True)
class GeometryFitWorkerManualFitSpaceDeps:
    geometry_manual_fit_space_by_background: Callable[..., object]
    geometry_manual_caked_fit_space_required_from_context: Callable[..., object]
    validate_geometry_fit_live_source_rows: Callable[..., object]
    collect_required_manual_fit_targets: Callable[..., object]


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
    caked_view_storage_deps: GeometryFitWorkerCakedViewStorageDeps | None = None
    source_projection_deps: GeometryFitWorkerSourceProjectionDeps | None = None
    cache_bundle_deps: GeometryFitWorkerCacheBundleDeps | None = None
    prebuild_deps: GeometryFitWorkerPrebuildDeps | None = None
    source_rows_deps: GeometryFitWorkerSourceRowsDeps | None = None
    required_cache_deps: GeometryFitWorkerRequiredCacheDeps | None = None
    dataset_deps: GeometryFitWorkerDatasetDeps | None = None
    manual_fit_space_deps: GeometryFitWorkerManualFitSpaceDeps | None = None

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

    def _require_caked_view_storage_deps(self) -> GeometryFitWorkerCakedViewStorageDeps:
        if self.caked_view_storage_deps is None:
            raise RuntimeError(
                "geometry-fit worker caked view storage deps are not configured"
            )
        return self.caked_view_storage_deps

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

    def _require_prebuild_deps(self) -> GeometryFitWorkerPrebuildDeps:
        if self.prebuild_deps is None:
            raise RuntimeError("geometry-fit worker prebuild deps are not configured")
        return self.prebuild_deps

    def _require_source_rows_deps(self) -> GeometryFitWorkerSourceRowsDeps:
        if self.source_rows_deps is None:
            raise RuntimeError("geometry-fit worker source-row deps are not configured")
        return self.source_rows_deps

    def _require_required_cache_deps(self) -> GeometryFitWorkerRequiredCacheDeps:
        if self.required_cache_deps is None:
            raise RuntimeError("geometry-fit worker required-cache deps are not configured")
        return self.required_cache_deps

    def _require_dataset_deps(self) -> GeometryFitWorkerDatasetDeps:
        if self.dataset_deps is None:
            raise RuntimeError("geometry-fit worker dataset deps are not configured")
        return self.dataset_deps

    def _require_manual_fit_space_deps(self) -> GeometryFitWorkerManualFitSpaceDeps:
        if self.manual_fit_space_deps is None:
            raise RuntimeError(
                "geometry-fit worker manual fit-space deps are not configured"
            )
        return self.manual_fit_space_deps

    def prepare_geometry_fit_run_for_worker(
        self,
        *,
        ensure_geometry_fit_caked_view: Callable[[], None],
        stage_callback: Callable[[str, Mapping[str, object]], None] | None,
        default_image_size: object = 0,
        default_display_rotate_k: object = 0,
    ) -> object:
        deps = self._require_dataset_deps()
        job_data = self.job_data

        def _manual_pairs_for_index(idx: int) -> list[dict[str, object]]:
            return [
                dict(entry)
                for entry in (
                    dict(job_data.get("manual_pairs_by_background", {}) or {}).get(
                        int(idx),
                        (),
                    )
                    or ()
                )
                if isinstance(entry, Mapping)
            ]

        image_size_value = (
            job_data["image_size"] if "image_size" in job_data else default_image_size
        )
        manual_dataset_bindings = deps.make_manual_dataset_bindings(
            osc_files=list(job_data.get("osc_files", ()) or ()),
            current_background_index=int(job_data.get("current_background_index", 0)),
            image_size=int(image_size_value),
            display_rotate_k=int(
                job_data.get("display_rotate_k", default_display_rotate_k)
            ),
            geometry_manual_pairs_for_index=_manual_pairs_for_index,
            load_background_by_index=self.load_background_by_index_snapshot,
            apply_background_backend_orientation=job_data.get(
                "apply_background_backend_orientation"
            ),
            geometry_manual_simulated_peaks_for_params=lambda *_args, **_kwargs: [],
            geometry_manual_simulated_lookup=job_data.get(
                "geometry_manual_simulated_lookup"
            ),
            geometry_manual_source_rows_for_background=(
                self.source_rows_for_background_worker
            ),
            geometry_manual_rebuild_source_rows_for_background=(
                self.rebuild_source_rows_for_background_worker
            ),
            geometry_manual_last_source_snapshot_diagnostics=(
                self.last_worker_source_snapshot_diagnostics
            ),
            geometry_manual_last_simulation_diagnostics=(
                self.last_worker_simulation_diagnostics
            ),
            geometry_manual_match_config=(
                lambda: copy.deepcopy(job_data.get("manual_match_config", {}))
            ),
            geometry_manual_entry_display_coords=(
                self.worker_geometry_manual_entry_display_coords
            ),
            geometry_manual_refresh_pair_entry=None,
            geometry_manual_caked_view_for_index=self.load_caked_view_by_index_snapshot,
            geometry_manual_project_peaks_to_current_view=(
                self.project_source_rows_by_row_background
            ),
            geometry_manual_project_peaks_for_background_view=(
                self.project_source_rows_for_background_view_worker
            ),
            geometry_manual_caked_projection_for_index=(
                self.load_caked_projection_by_index_snapshot
            ),
            unrotate_display_peaks=job_data.get("unrotate_display_peaks"),
            display_to_native_sim_coords=job_data.get("display_to_native_sim_coords"),
            native_detector_coords_to_detector_display_coords=(
                job_data.get("native_detector_coords_to_detector_display_coords")
            ),
            native_detector_coords_to_detector_display_coords_for_background=(
                self.worker_native_detector_coords_to_detector_display_coords_for_background
            ),
            select_fit_orientation=job_data.get("select_fit_orientation"),
            apply_orientation_to_entries=job_data.get("apply_orientation_to_entries"),
            orient_image_for_fit=job_data.get("orient_image_for_fit"),
            pick_uses_caked_space=(
                lambda: bool(job_data.get("pick_uses_caked_space", False))
            ),
        )

        def _selected_background_indices(**_kwargs: object) -> list[object]:
            if job_data.get("selection_error"):
                raise RuntimeError(str(job_data.get("selection_error")))
            return list(job_data.get("selected_background_indices", ()) or ())

        def _background_theta_values(**_kwargs: object) -> list[object]:
            if job_data.get("background_theta_error"):
                raise RuntimeError(str(job_data.get("background_theta_error")))
            return list(job_data.get("background_theta_values", ()) or ())

        def _build_dataset(
            background_index: int,
            *,
            theta_base: object,
            base_fit_params: object,
            orientation_cfg: object,
            manual_fit_requires_caked_space: bool = False,
            stage_callback: object = None,
        ) -> object:
            return deps.build_geometry_manual_fit_dataset(
                background_index,
                theta_base=theta_base,
                base_fit_params=base_fit_params,
                manual_dataset_bindings=manual_dataset_bindings,
                orientation_cfg=orientation_cfg,
                manual_fit_requires_caked_space=manual_fit_requires_caked_space,
                stage_callback=stage_callback,
            )

        return deps.prepare_geometry_fit_run(
            params=dict(job_data.get("params", {}) or {}),
            var_names=list(job_data.get("var_names", ()) or ()),
            fit_config=dict(job_data.get("fit_config", {}) or {}),
            osc_files=list(job_data.get("osc_files", ()) or ()),
            current_background_index=int(job_data.get("current_background_index", 0)),
            theta_initial=float(job_data.get("theta_initial", 0.0)),
            preserve_live_theta=bool(job_data.get("preserve_live_theta", False)),
            apply_geometry_fit_background_selection=(
                lambda **_kwargs: bool(job_data.get("selection_applied", True))
            ),
            current_geometry_fit_background_indices=_selected_background_indices,
            geometry_fit_uses_shared_theta_offset=(
                lambda _indices: bool(job_data.get("uses_shared_theta", False))
            ),
            apply_background_theta_metadata=(
                lambda **_kwargs: bool(job_data.get("theta_metadata_applied", True))
            ),
            current_background_theta_values=_background_theta_values,
            current_geometry_theta_offset=(
                lambda **_kwargs: float(job_data.get("theta_offset", 0.0))
            ),
            geometry_manual_pairs_for_index=(
                manual_dataset_bindings.geometry_manual_pairs_for_index
            ),
            ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
            build_dataset=_build_dataset,
            build_runtime_config=(
                lambda _fit_params: copy.deepcopy(
                    dict(job_data.get("geometry_runtime_cfg", {}) or {})
                )
            ),
            manual_fit_pick_uses_caked_space=bool(
                job_data.get("pick_uses_caked_space", False)
            ),
            manual_fit_requires_caked_space=any(
                bool(value)
                for value in dict(
                    job_data.get("manual_caked_fit_space_required_by_background", {})
                    or {}
                ).values()
            ),
            stage_callback=stage_callback,
        )

    def worker_manual_pairs_for_background(
        self,
        background_index: int,
    ) -> list[dict[str, object]]:
        pairs_by_background = self.job_data.get("manual_pairs_by_background")
        if not isinstance(pairs_by_background, Mapping):
            return []
        pairs = pairs_by_background.get(int(background_index), ())
        return [dict(entry) for entry in pairs or () if isinstance(entry, Mapping)]

    def worker_manual_fit_space_by_background(self) -> dict[int, str]:
        deps = self._require_manual_fit_space_deps()
        required_indices = [
            int(idx) for idx in (self.job_data.get("required_indices", ()) or ())
        ]
        stored_spaces = self.job_data.get("manual_fit_space_by_background")
        normalized: dict[int, str] = {}
        if isinstance(stored_spaces, Mapping):
            for background_idx in required_indices:
                raw_kind = stored_spaces.get(
                    int(background_idx),
                    stored_spaces.get(str(int(background_idx))),
                )
                kind = str(raw_kind or "detector").strip().lower()
                normalized[int(background_idx)] = (
                    kind if kind in {"caked", "mixed"} else "detector"
                )
            if len(normalized) == len(required_indices):
                return normalized
        return deps.geometry_manual_fit_space_by_background(
            required_indices,
            self.worker_manual_pairs_for_background,
            pick_uses_caked_space=bool(self.job_data.get("pick_uses_caked_space", False)),
            current_background_index=int(self.job_data.get("current_background_index", 0)),
        )

    def worker_manual_caked_fit_space_required_for_background(
        self,
        background_index: int,
    ) -> bool:
        deps = self._require_manual_fit_space_deps()
        background_idx = int(background_index)
        required_by_background = self.job_data.get(
            "manual_caked_fit_space_required_by_background"
        )
        if isinstance(required_by_background, Mapping):
            raw_value = required_by_background.get(
                background_idx,
                required_by_background.get(str(background_idx)),
            )
            if raw_value is not None:
                return bool(raw_value)
        manual_spaces = self.worker_manual_fit_space_by_background()
        manual_space = str(manual_spaces.get(background_idx, "")).strip().lower()
        if manual_space == "caked":
            return True
        pairs = self.worker_manual_pairs_for_background(background_idx)
        try:
            pick_applies_to_background = background_idx == int(
                self.job_data.get("current_background_index", background_idx),
            )
        except Exception:
            pick_applies_to_background = True
        return bool(
            deps.geometry_manual_caked_fit_space_required_from_context(
                pairs,
                manual_fit_space_kind=manual_space,
                projection_view_mode=self.job_data.get("projection_view_mode"),
                pick_uses_caked_space=bool(
                    self.job_data.get("pick_uses_caked_space", False)
                ),
                pick_applies_to_background=pick_applies_to_background,
            )
        )

    def worker_validate_required_source_rows_for_fit_space(
        self,
        rows: Sequence[object] | None,
        *,
        required_pairs: Sequence[Mapping[str, object]] | None,
        background_index: int,
    ) -> dict[str, object]:
        deps = self._require_manual_fit_space_deps()
        caked_required = self.worker_manual_caked_fit_space_required_for_background(
            int(background_index)
        )
        validation = dict(
            deps.validate_geometry_fit_live_source_rows(
                rows,
                required_pairs=required_pairs,
                require_caked_fit_space=bool(caked_required),
            )
        )
        validation["manual_caked_fit_space_required"] = bool(caked_required)
        if caked_required:
            validation["manual_caked_required_pair_count"] = int(
                validation.get("required_pair_count")
                or len(
                    deps.collect_required_manual_fit_targets(
                        required_pairs,
                        background_index=int(background_index),
                    )
                )
                or len(required_pairs or ())
            )
        return validation

    def reject_worker_mixed_manual_fit_spaces(
        self,
        manual_spaces: Mapping[int, str],
    ) -> None:
        mixed_backgrounds = [
            int(background_idx)
            for background_idx, kind in manual_spaces.items()
            if str(kind) == "mixed"
            and not self.worker_manual_caked_fit_space_required_for_background(
                int(background_idx)
            )
        ]
        if not mixed_backgrounds:
            return
        labels = ", ".join(str(idx + 1) for idx in sorted(mixed_backgrounds))
        raise RuntimeError(
            "mixed detector/caked manual fit spaces are not supported "
            f"for background(s) {labels}; rebuild manual pairs in one fit space"
        )

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

    def worker_caked_view_payload_ready(self, background_index: int) -> bool:
        deps = self._require_caked_payload_deps()
        background_payload = dict(
            dict(self.job_data.get("background_images", {}) or {}).get(
                int(background_index)
            )
            or {}
        )
        try:
            detector_shape = tuple(
                int(v)
                for v in np.asarray(
                    background_payload.get("native"),
                    dtype=np.float64,
                ).shape[:2]
            )
        except Exception:
            detector_shape = None
        payload = self.load_caked_projection_by_index_snapshot(
            int(background_index),
            detector_shape=detector_shape,
            allow_generated_payload=True,
        )
        hydrated_payload = deps.hydrate_exact_caked_payload(
            payload,
            detector_shape=(
                payload.get("detector_shape") if isinstance(payload, Mapping) else None
            ),
            params=dict(self.job_data.get("params", {}) or {}),
            require_background=False,
        )
        return isinstance(hydrated_payload, Mapping) and deps.is_transform_bundle(
            hydrated_payload.get("transform_bundle")
        )

    def ensure_worker_geometry_fit_caked_view(self) -> None:
        manual_spaces = self.worker_manual_fit_space_by_background()
        self.reject_worker_mixed_manual_fit_spaces(manual_spaces)
        caked_backgrounds = [
            int(background_idx)
            for background_idx in (self.job_data.get("required_indices", ()) or ())
            if self.worker_manual_caked_fit_space_required_for_background(
                int(background_idx)
            )
        ]
        for background_idx in caked_backgrounds:
            if self.worker_caked_view_payload_ready(int(background_idx)):
                continue
            raise RuntimeError(
                f"exact caked projector unavailable for background {int(background_idx) + 1}"
            )

    def store_worker_caked_view_for_background(
        self,
        bundle: object,
        *,
        stage_callback: object | None = None,
        source_cache_generation_id: int | None = None,
    ) -> dict[str, object]:
        storage_deps = self._require_caked_view_storage_deps()
        caked_deps = self._require_caked_payload_deps()
        helper_started_at = perf_counter()
        background_idx = int(getattr(bundle, "background_index"))
        resolved_background_label = str(
            getattr(bundle, "background_label", None)
            or f"background {background_idx + 1}"
        )
        stored_rows = getattr(bundle, "stored_rows", None)
        projected_rows = getattr(bundle, "projected_rows", None)
        row_count = int(len(projected_rows or stored_rows or ()))

        def _emit_caked_stage(
            stage: str,
            *,
            stage_started_at: float | None = None,
            **payload: object,
        ) -> None:
            event_payload = {
                "background_index": int(background_idx),
                "background_label": resolved_background_label,
                "source_cache_generation_id": (
                    int(source_cache_generation_id)
                    if source_cache_generation_id is not None
                    else None
                ),
                "row_count": int(row_count),
                "elapsed_s": float(max(0.0, perf_counter() - helper_started_at)),
                **payload,
            }
            if stage_started_at is not None:
                event_payload["stage_elapsed_s"] = float(
                    max(0.0, perf_counter() - stage_started_at)
                )
            storage_deps.emit_geometry_fit_stage_event(
                stage_callback,
                str(stage),
                **event_payload,
            )

        def _caked_result(
            status: str,
            *,
            caked_view_stored: bool,
            roi_enabled: bool = False,
            roi_used_restricted_cake: bool = False,
            roi_pixel_count: int = 0,
            roi_fraction: float = 0.0,
            roi_fallback_reason: str | None = None,
            roi_half_width_px: float = 0.0,
        ) -> dict[str, object]:
            return {
                "background_index": int(background_idx),
                "background_label": resolved_background_label,
                "source_cache_generation_id": (
                    int(source_cache_generation_id)
                    if source_cache_generation_id is not None
                    else None
                ),
                "row_count": int(row_count),
                "caked_view_stored": bool(caked_view_stored),
                "caked_view_status": str(status),
                "roi_enabled": bool(roi_enabled),
                "roi_used_restricted_cake": bool(roi_used_restricted_cake),
                "roi_pixel_count": int(roi_pixel_count),
                "roi_fraction": float(roi_fraction),
                "roi_fallback_reason": (
                    str(roi_fallback_reason) if roi_fallback_reason is not None else None
                ),
                "roi_half_width_px": float(roi_half_width_px),
                "elapsed_s": float(max(0.0, perf_counter() - helper_started_at)),
            }

        caked_views_by_background = self.job_data.setdefault(
            "caked_views_by_background",
            {},
        )
        projection_payload_by_background = self.job_data.setdefault(
            "projection_payload_by_background",
            {},
        )
        background_images = dict(self.job_data.get("background_images", {}) or {})
        background_payload = dict(background_images.get(int(background_idx)) or {})
        native_background = background_payload.get("native")
        if native_background is None:
            return _caked_result(
                "missing_native_background",
                caked_view_stored=False,
                roi_fallback_reason="missing_native_background",
            )

        backend_background = np.asarray(native_background, dtype=np.float64)
        apply_backend_orientation = self.job_data.get(
            "apply_background_backend_orientation"
        )
        if callable(apply_backend_orientation):
            try:
                oriented_background = apply_backend_orientation(backend_background)
            except Exception:
                oriented_background = backend_background
            if oriented_background is not None:
                backend_background = np.asarray(oriented_background, dtype=np.float64)

        worker_ai = caked_deps.caking_integrator()
        background_caked_view = dict(caked_views_by_background.get(background_idx) or {})
        if not caked_deps.is_transform_bundle(
            background_caked_view.get("transform_bundle")
        ):
            # Non-current worker backgrounds need same-axes exact-cake metadata
            # before the first caked payload exists, otherwise ROI projection drops
            # every angle-backed source row on the first pass.
            precompute_npt_rad, precompute_npt_azim = (
                storage_deps.worker_projection_analysis_bins()
            )
            precomputed_caked_view = storage_deps.geometry_fit_worker_caked_projection_view(
                detector_shape=backend_background.shape[:2],
                ai=worker_ai,
                npt_rad=precompute_npt_rad,
                npt_azim=precompute_npt_azim,
            )
            if isinstance(precomputed_caked_view, dict):
                background_caked_view.update(precomputed_caked_view)
        roi_selection_started_at = perf_counter()
        _emit_caked_stage("source_cache_caked_roi_selection_start")
        roi_selection = storage_deps.build_caked_roi_selection(
            stored_rows,
            required_pairs=list(
                dict(self.job_data.get("manual_pairs_by_background", {}) or {}).get(
                    background_idx,
                    (),
                )
                or ()
            ),
            image_shape=backend_background.shape[:2],
            fit_config=dict(self.job_data.get("geometry_runtime_cfg", {}) or {}),
            fit_space_to_detector_point=storage_deps.caked_roi_fit_space_to_detector_point(
                detector_shape=backend_background.shape[:2],
                radial_axis=background_caked_view.get("radial_axis"),
                azimuth_axis=background_caked_view.get("azimuth_axis"),
                ai=worker_ai,
                transform_bundle=background_caked_view.get("transform_bundle"),
            ),
        )
        roi_enabled = bool(roi_selection.get("enabled", False))
        roi_pixel_count = int(roi_selection.get("pixel_count", 0) or 0)
        roi_fraction = float(roi_selection.get("fraction", 0.0) or 0.0)
        roi_half_width_px = float(roi_selection.get("half_width_px", 0.0) or 0.0)
        roi_fallback_reason = roi_selection.get("fallback_reason")
        roi_used_restricted_cake = bool(roi_selection.get("valid", False))
        _emit_caked_stage(
            "source_cache_caked_roi_selection_ready",
            stage_started_at=roi_selection_started_at,
            status="ready",
            roi_enabled=bool(roi_enabled),
            roi_used_restricted_cake=bool(roi_used_restricted_cake),
            roi_pixel_count=int(roi_pixel_count),
            roi_fraction=float(roi_fraction),
            roi_fallback_reason=(
                str(roi_fallback_reason) if roi_fallback_reason is not None else None
            ),
            roi_half_width_px=float(roi_half_width_px),
        )

        rows = cols = None
        if roi_used_restricted_cake:
            try:
                rows = np.asarray(roi_selection.get("rows"), dtype=np.int32)
                cols = np.asarray(roi_selection.get("cols"), dtype=np.int32)
            except Exception:
                rows = cols = None
                roi_used_restricted_cake = False
                roi_fallback_reason = "invalid_roi_pixels"

        ai = worker_ai
        if ai is None:
            return _caked_result(
                "missing_integrator",
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason="missing_integrator",
                roi_half_width_px=float(roi_half_width_px),
            )

        res2 = None
        if roi_used_restricted_cake:
            restricted_started_at = perf_counter()
            _emit_caked_stage(
                "source_cache_restricted_cake_start",
                roi_pixel_count=int(roi_pixel_count),
            )
            try:
                with storage_deps.temporary_numba_thread_limit(
                    storage_deps.default_reserved_cpu_worker_count()
                ):
                    res2 = storage_deps.caking(
                        backend_background,
                        ai,
                        rows=rows,
                        cols=cols,
                    )
                _emit_caked_stage(
                    "source_cache_restricted_cake_ready",
                    stage_started_at=restricted_started_at,
                    status="ready",
                    roi_pixel_count=int(roi_pixel_count),
                )
            except Exception as exc:
                _emit_caked_stage(
                    "source_cache_restricted_cake_failed",
                    stage_started_at=restricted_started_at,
                    status=f"exception:{type(exc).__name__}",
                    roi_pixel_count=int(roi_pixel_count),
                )
                roi_used_restricted_cake = False
                roi_fallback_reason = f"restricted_cake_exception:{type(exc).__name__}"

        if res2 is None:
            full_started_at = perf_counter()
            _emit_caked_stage("source_cache_full_cake_start")
            try:
                with storage_deps.temporary_numba_thread_limit(
                    storage_deps.default_reserved_cpu_worker_count()
                ):
                    res2 = storage_deps.caking(backend_background, ai)
                _emit_caked_stage(
                    "source_cache_full_cake_ready",
                    stage_started_at=full_started_at,
                    status="ready",
                )
            except Exception as exc:
                _emit_caked_stage(
                    "source_cache_full_cake_failed",
                    stage_started_at=full_started_at,
                    status=f"exception:{type(exc).__name__}",
                )
                return _caked_result(
                    f"full_cake_exception:{type(exc).__name__}",
                    caked_view_stored=False,
                    roi_enabled=bool(roi_enabled),
                    roi_used_restricted_cake=False,
                    roi_pixel_count=int(roi_pixel_count),
                    roi_fraction=float(roi_fraction),
                    roi_fallback_reason=f"full_cake_exception:{type(exc).__name__}",
                    roi_half_width_px=float(roi_half_width_px),
                )

        caked_payload = storage_deps.prepare_caked_display_payload(
            res2,
            ai=ai,
            detector_shape=backend_background.shape,
        )
        if not isinstance(caked_payload, Mapping):
            return _caked_result(
                "invalid_caked_payload",
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason="invalid_caked_payload",
                roi_half_width_px=float(roi_half_width_px),
            )
        projection_payload = caked_deps.caked_projection_payload(caked_payload)
        projection_status = self.caked_projection_payload_status(projection_payload)
        if projection_status != "projection_payload_ready":
            return _caked_result(
                projection_status,
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason=projection_status,
                roi_half_width_px=float(roi_half_width_px),
            )
        hydrated_projection_payload = caked_deps.hydrate_exact_caked_payload(
            projection_payload,
            detector_shape=backend_background.shape[:2],
            params=dict(self.job_data.get("params", {}) or {}),
            require_background=False,
        )
        projection_status = self.caked_projection_payload_status(
            hydrated_projection_payload
        )
        if not isinstance(hydrated_projection_payload, Mapping) or not (
            caked_deps.is_transform_bundle(
                hydrated_projection_payload.get("transform_bundle")
            )
        ):
            return _caked_result(
                projection_status,
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason=projection_status,
                roi_half_width_px=float(roi_half_width_px),
            )
        projection_payload = caked_deps.caked_projection_payload(
            hydrated_projection_payload
        )
        if not isinstance(projection_payload, Mapping):
            return _caked_result(
                "missing_exact_caked_bundle",
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason="missing_exact_caked_bundle",
                roi_half_width_px=float(roi_half_width_px),
            )
        sanitized_caked_payload, sanitize_diag = (
            storage_deps.sanitize_caked_display_payload(caked_payload)
        )
        sanitize_status = str(sanitize_diag.get("status", "invalid_caked_payload"))
        if not isinstance(sanitized_caked_payload, Mapping):
            return _caked_result(
                sanitize_status,
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason=sanitize_status,
                roi_half_width_px=float(roi_half_width_px),
            )
        hydrated_caked_payload = caked_deps.hydrate_exact_caked_payload(
            sanitized_caked_payload,
            detector_shape=backend_background.shape[:2],
            params=dict(self.job_data.get("params", {}) or {}),
            require_background=True,
        )
        if not isinstance(hydrated_caked_payload, Mapping):
            return _caked_result(
                "missing_exact_caked_bundle",
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason="missing_exact_caked_bundle",
                roi_half_width_px=float(roi_half_width_px),
            )
        caked_payload = hydrated_caked_payload

        if not self.source_cache_generation_matches(
            background_idx,
            source_cache_generation_id,
        ):
            return _caked_result(
                "stale_generation",
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason=(
                    str(roi_fallback_reason) if roi_fallback_reason is not None else None
                ),
                roi_half_width_px=float(roi_half_width_px),
            )

        stored_projection_payload = caked_deps.projection_payload_storage_copy(
            projection_payload
        )
        if not isinstance(stored_projection_payload, Mapping):
            return _caked_result(
                "missing_exact_caked_bundle",
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason="missing_exact_caked_bundle",
                roi_half_width_px=float(roi_half_width_px),
            )
        projection_payload_by_background[background_idx] = stored_projection_payload
        _emit_caked_stage(
            "projection_payload_ready",
            status="projection_payload_ready",
            payload_kind="projection",
        )

        try:
            caked_views_by_background[background_idx] = {
                "background": np.asarray(
                    caked_payload.get("image"),
                    dtype=np.float64,
                ).copy(),
                "radial_axis": np.asarray(
                    caked_payload.get("radial"),
                    dtype=np.float64,
                ).copy(),
                "azimuth_axis": np.asarray(
                    caked_payload.get("azimuth"),
                    dtype=np.float64,
                ).copy(),
                "raw_azimuth_axis": np.asarray(
                    caked_payload.get(
                        "raw_azimuth_axis",
                        caked_payload.get("raw_azimuth"),
                    ),
                    dtype=np.float64,
                ).copy(),
                "raw_to_gui_row_permutation": np.asarray(
                    caked_payload.get("raw_to_gui_row_permutation"),
                    dtype=np.int32,
                ).copy(),
                "transform_bundle": caked_payload.get("transform_bundle"),
                "detector_shape": tuple(
                    int(v) for v in tuple(caked_payload.get("detector_shape", ()))[:2]
                ),
                "roi_enabled": bool(roi_enabled),
                "roi_used_restricted_cake": bool(roi_used_restricted_cake),
                "roi_pixel_count": int(roi_pixel_count),
                "roi_fraction": float(roi_fraction),
                "roi_fallback_reason": (
                    str(roi_fallback_reason) if roi_fallback_reason is not None else None
                ),
                "roi_half_width_px": float(roi_half_width_px),
                "caked_display_sanitize_status": sanitize_status,
                "sanitized_empty_bin_count": int(
                    sanitize_diag.get("sanitized_empty_bin_count", 0) or 0
                ),
                "nonfinite_supported_bin_count": int(
                    sanitize_diag.get("nonfinite_supported_bin_count", 0) or 0
                ),
            }
            if str(self.job_data.get("projection_view_mode") or "").strip().lower() == (
                "caked"
            ):
                projection_signature_map = self.job_data.setdefault(
                    "projection_view_signature_by_background",
                    {},
                )
                projection_signature_map[background_idx] = (
                    storage_deps.normalize_projection_view_signature(
                        storage_deps.targeted_projection_view_signature(
                            background_idx,
                            mode_override="caked",
                            caked_payload=dict(projection_payload),
                            detector_shape=backend_background.shape[:2],
                            ai=worker_ai,
                            analysis_preview_bins=self.job_data.get("analysis_bins"),
                        ),
                        background_idx,
                    )
                )
                if background_idx == int(
                    self.job_data.get("current_background_index", -1)
                ):
                    self.job_data["projection_view_signature"] = copy.deepcopy(
                        projection_signature_map[background_idx]
                    )
                source_deps = self._require_source_projection_deps()
                refreshed_projected_rows = source_deps.rows_for_background(
                    background_idx,
                    self.project_source_rows_for_background(
                        background_idx,
                        stored_rows,
                    ),
                )
                if refreshed_projected_rows:
                    refreshed_diagnostics = dict(getattr(bundle, "diagnostics", {}) or {})
                    refreshed_diagnostics["projected_peak_count"] = int(
                        len(refreshed_projected_rows)
                    )
                    refreshed_diagnostics["projection_view_signature"] = copy.deepcopy(
                        projection_signature_map.get(background_idx)
                    )
                    refreshed_diagnostics["projection_view_signature_digest"] = (
                        storage_deps.digest_payload(
                            refreshed_diagnostics.get("projection_view_signature")
                        )
                    )
                    refreshed_bundle = storage_deps.replace_bundle(
                        bundle,
                        projected_rows=[
                            dict(entry) for entry in refreshed_projected_rows
                        ],
                        diagnostics=refreshed_diagnostics,
                    )
                    self.worker_background_cache_by_index[background_idx] = (
                        refreshed_bundle
                    )
                    cache_deps = self._require_cache_bundle_deps()
                    self.worker_source_row_snapshots[background_idx] = {
                        "background_index": int(
                            getattr(refreshed_bundle, "background_index")
                        ),
                        "simulation_signature": getattr(
                            refreshed_bundle,
                            "requested_signature",
                        ),
                        "requested_signature": getattr(
                            refreshed_bundle,
                            "requested_signature",
                        ),
                        "requested_signature_summary": getattr(
                            refreshed_bundle,
                            "requested_signature_summary",
                        ),
                        "rows": cache_deps.copy_source_rows(
                            getattr(refreshed_bundle, "stored_rows", None)
                        ),
                        "stored_rows": cache_deps.copy_source_rows(
                            getattr(refreshed_bundle, "stored_rows", None)
                        ),
                        "projected_rows": cache_deps.copy_source_rows(
                            getattr(refreshed_bundle, "projected_rows", None)
                        ),
                        "row_count": int(
                            len(getattr(refreshed_bundle, "stored_rows", None) or ())
                        ),
                        "projected_row_count": int(
                            len(getattr(refreshed_bundle, "projected_rows", None) or ())
                        ),
                        "created_from": str(
                            getattr(refreshed_bundle, "cache_source", None)
                            or "geometry_fit_background_cache"
                        ),
                        "diagnostics": copy.deepcopy(
                            dict(getattr(refreshed_bundle, "diagnostics", {}) or {})
                        ),
                        "projection_view_signature": copy.deepcopy(
                            refreshed_diagnostics.get("projection_view_signature")
                        ),
                        "valid_for_picker": bool(
                            getattr(refreshed_bundle, "stored_rows", None)
                        ),
                        "valid_for_geometry_fit_dataset": bool(
                            getattr(refreshed_bundle, "stored_rows", None)
                            or getattr(refreshed_bundle, "projected_rows", None)
                        ),
                    }
        except Exception as exc:
            failure_status = f"store_caked_payload_failed:{type(exc).__name__}:{exc}"
            return _caked_result(
                failure_status,
                caked_view_stored=False,
                roi_enabled=bool(roi_enabled),
                roi_used_restricted_cake=bool(roi_used_restricted_cake),
                roi_pixel_count=int(roi_pixel_count),
                roi_fraction=float(roi_fraction),
                roi_fallback_reason=failure_status,
                roi_half_width_px=float(roi_half_width_px),
            )

        return _caked_result(
            sanitize_status if sanitize_status != "stored" else "stored",
            caked_view_stored=True,
            roi_enabled=bool(roi_enabled),
            roi_used_restricted_cake=bool(roi_used_restricted_cake),
            roi_pixel_count=int(roi_pixel_count),
            roi_fraction=float(roi_fraction),
            roi_fallback_reason=(
                str(roi_fallback_reason) if roi_fallback_reason is not None else None
            ),
            roi_half_width_px=float(roi_half_width_px),
        )

    def worker_native_detector_coords_to_detector_display_coords_for_background(
        self,
        background_index: int,
    ) -> Callable[[float, float], tuple[float | None, float | None]] | None:
        try:
            bg_idx = int(background_index)
        except Exception:
            return None
        background_payload = dict(
            dict(self.job_data.get("background_images", {}) or {}).get(bg_idx) or {}
        )
        native_background = background_payload.get("native")
        try:
            shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
        except Exception:
            return None
        if len(shape) < 2 or min(shape) <= 0:
            return None
        deps = self._require_source_projection_deps()
        rotate_k = int(
            self.job_data.get("display_rotate_k", deps.default_display_rotate_k)
        )

        def _to_display(col: float, row: float) -> tuple[float | None, float | None]:
            return deps.rotate_point_for_display(float(col), float(row), shape, rotate_k)

        _to_display.__name__ = (
            f"_worker_native_detector_coords_to_detector_display_coords_bg_{bg_idx}"
        )
        return _to_display

    def worker_geometry_manual_entry_display_coords(
        self,
        entry: Mapping[str, object] | None,
    ) -> tuple[float, float] | None:
        if not isinstance(entry, Mapping):
            return None

        def _entry_pair(
            pair_keys: Sequence[tuple[str, str]],
        ) -> tuple[float, float] | None:
            for key_x, key_y in pair_keys:
                try:
                    col = float(entry.get(key_x, np.nan))
                    row = float(entry.get(key_y, np.nan))
                except Exception:
                    continue
                if np.isfinite(col) and np.isfinite(row):
                    return float(col), float(row)
            return None

        def _tuple_pair(keys: Sequence[str]) -> tuple[float, float] | None:
            for key in keys:
                point = deps.overlay_state_finite_pair(entry.get(key))
                if point is not None:
                    return float(point[0]), float(point[1])
            return None

        if bool(self.job_data.get("pick_uses_caked_space", False)):
            return _entry_pair(
                (
                    ("caked_x", "caked_y"),
                    ("raw_caked_x", "raw_caked_y"),
                    ("background_two_theta_deg", "background_phi_deg"),
                    ("two_theta_deg", "phi_deg"),
                    ("display_col", "display_row"),
                    ("x", "y"),
                )
            )

        deps = self._require_source_projection_deps()
        display_point = _tuple_pair(("geometry_detector_display_px", "raw_detector_display_px"))
        if display_point is not None:
            return display_point

        display_point = _entry_pair(
            (
                ("x", "y"),
                ("display_col", "display_row"),
                ("detector_display_x", "detector_display_y"),
            )
        )
        if display_point is not None:
            return display_point

        try:
            background_index = int(
                entry.get(
                    "background_index",
                    self.job_data.get("current_background_index", 0),
                )
            )
        except Exception:
            background_index = int(self.job_data.get("current_background_index", 0))
        native_to_display = (
            self.worker_native_detector_coords_to_detector_display_coords_for_background(
                int(background_index)
            )
        )
        if not callable(native_to_display):
            return None
        native_point = _tuple_pair(("geometry_detector_native_px", "raw_detector_native_px"))
        if native_point is None:
            native_point = _entry_pair(
                (
                    ("detector_x", "detector_y"),
                    ("background_detector_x", "background_detector_y"),
                    ("detector_native_x", "detector_native_y"),
                    ("refined_detector_native_col", "refined_detector_native_row"),
                    ("native_col", "native_row"),
                    ("refined_sim_native_x", "refined_sim_native_y"),
                )
            )
        if native_point is None:
            return None
        display_point = deps.overlay_state_finite_pair(
            native_to_display(float(native_point[0]), float(native_point[1]))
        )
        if display_point is None:
            return None
        return float(display_point[0]), float(display_point[1])

    def project_source_rows_for_background_view_worker(
        self,
        background_index: int,
        rows: Sequence[object] | None,
        **kwargs: object,
    ) -> Sequence[object]:
        deps = self._require_source_projection_deps()
        normalized_rows = deps.rows_for_background(int(background_index), rows)
        normalized_mode = (
            str(
                kwargs.get("mode_override")
                if kwargs.get("mode_override") is not None
                else self.job_data.get("projection_view_mode") or "detector"
            )
            .strip()
            .lower()
        )
        if self.worker_cached_projection_rows_match(
            normalized_rows,
            background_index=int(background_index),
            mode=normalized_mode,
        ):
            return normalized_rows
        return self.project_source_rows_for_background(
            int(background_index),
            normalized_rows,
            mode_override=kwargs.get("mode_override"),
            strict_caked_projection=bool(kwargs.get("strict_caked_projection", True)),
        )

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

    def prebuild_background_cache_bundle_worker(
        self,
        background_index: int,
        *,
        theta_base: float,
        param_set: Mapping[str, object] | None = None,
        consumer: str = "geometry_fit_preflight_cache",
        prior_diagnostics: Mapping[str, object] | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
        stage_callback: object | None = None,
    ) -> object | None:
        cache_deps = self._require_cache_bundle_deps()
        prebuild_deps = self._require_prebuild_deps()
        background_idx = int(background_index)
        params_local = dict(self.job_data.get("params", {}) or {})
        params_local["theta_initial"] = float(
            prebuild_deps.theta_initial_for_background(int(background_idx))
        )
        if isinstance(param_set, Mapping):
            params_local.update(dict(param_set))

        requested_signature_map = prebuild_deps.int_keyed_mapping(
            self.job_data.get("requested_signatures", {})
        )
        requested_signature_summary_map = prebuild_deps.int_keyed_mapping(
            self.job_data.get("requested_signature_summaries", {})
        )
        base_requested_signature = requested_signature_map.get(int(background_idx))
        base_requested_signature_summary = requested_signature_summary_map.get(
            int(background_idx)
        )
        if base_requested_signature_summary is None:
            base_requested_signature_summary = prebuild_deps.live_cache_signature_summary(
                base_requested_signature
            )
        if isinstance(param_set, Mapping):
            requested_signature = (
                "geometry_fit_worker_trial_source_rows",
                int(background_idx),
                prebuild_deps.digest_payload(
                    prebuild_deps.cache_jsonable(params_local)
                ),
                prebuild_deps.digest_payload(
                    prebuild_deps.cache_jsonable(base_requested_signature)
                ),
            )
            requested_signature_summary = prebuild_deps.live_cache_signature_summary(
                requested_signature
            )
        else:
            requested_signature = base_requested_signature
            requested_signature_summary = base_requested_signature_summary
        background_label = dict(self.job_data.get("background_labels", {}) or {}).get(
            int(background_idx),
            f"background {int(background_idx) + 1}",
        )
        required_manual_fit_targets = (
            prebuild_deps.collect_required_manual_fit_targets(
                required_pairs,
                background_index=int(background_idx),
            )
            if required_pairs
            else []
        )
        required_branch_group_keys = prebuild_deps.required_branch_group_keys(
            required_manual_fit_targets
        )
        preflight_mode = "manual_geometry_targeted" if required_branch_group_keys else "full"
        live_rows_signature_map = prebuild_deps.int_keyed_mapping(
            self.job_data.get("live_rows_signature_by_background", {})
        )
        live_rows_signature = live_rows_signature_map.get(
            int(background_idx),
            self.job_data.get("live_rows_signature"),
        )
        live_rows_cache_metadata = dict(
            copy.deepcopy(
                prebuild_deps.int_keyed_mapping(
                    self.job_data.get("live_rows_cache_metadata_by_background", {})
                ).get(int(background_idx), {})
            )
            or {}
        )
        live_rows = cache_deps.copy_source_rows(
            prebuild_deps.int_keyed_mapping(
                self.job_data.get("live_rows_by_background", {})
            ).get(int(background_idx), ())
        )
        live_rows_cache_metadata.setdefault("live_rows_raw_count", int(len(live_rows)))
        live_rows_cache_metadata.setdefault("live_rows_payload_count", int(len(live_rows)))
        live_rows_cache_metadata.setdefault(
            "live_rows_source_counts",
            prebuild_deps.live_row_source_counts(live_rows),
        )
        live_rows_cache_metadata.setdefault(
            "live_rows_cache_source",
            str(
                live_rows_cache_metadata.get("cache_source", "live_preview_cache")
                or "live_preview_cache"
            ),
        )
        live_rows_cache_metadata.setdefault(
            "geometry_fit_live_handoff_patch_marker",
            str(prebuild_deps.live_handoff_patch_marker),
        )
        handoff_diag = dict(self.job_data.get("live_rows_handoff_diagnostics", {}) or {})
        for diag_key in (
            "q_group_cached_entries",
            "manual_picker_candidates",
            "live_preview_rows_count",
            "live_rows_by_background_keys",
            "live_rows_by_background_current_count",
            "requested_signature_keys",
            "requested_signature_by_background_keys",
            "live_rows_signature_by_background_keys",
        ):
            if diag_key in handoff_diag:
                live_rows_cache_metadata.setdefault(diag_key, handoff_diag.get(diag_key))

        snapshot = dict(self.worker_source_row_snapshots.get(int(background_idx)) or {})
        snapshot_signature = snapshot.get("simulation_signature")
        snapshot_rows = cache_deps.copy_source_rows(snapshot.get("rows"))
        snapshot_validation = (
            prebuild_deps.validate_required_source_rows_for_fit_space(
                snapshot_rows,
                required_pairs=required_pairs,
                background_index=int(background_idx),
            )
            if required_pairs
            else {}
        )
        if (
            snapshot_signature == requested_signature
            and snapshot_rows
            and (not required_pairs or bool(snapshot_validation.get("valid", False)))
        ):
            bundle = self.build_geometry_fit_background_cache_bundle(
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                theta_base=float(theta_base),
                theta_initial=float(params_local.get("theta_initial", 0.0)),
                stored_rows=snapshot_rows,
                cache_source=str(snapshot.get("created_from") or "source_snapshot"),
                diagnostics={
                    "created_from": snapshot.get("created_from"),
                    "cache_source": str(snapshot.get("created_from") or "source_snapshot"),
                    "live_runtime_cache_validation": snapshot_validation,
                },
            )
            self.store_worker_background_cache_bundle(bundle)
            self.set_worker_source_snapshot_diagnostics(**dict(bundle.diagnostics))
            self.worker_simulation_diagnostics.clear()
            self.worker_simulation_diagnostics.update(dict(bundle.diagnostics))
            return bundle
        if required_pairs and snapshot_signature == requested_signature and snapshot_rows:
            self.set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=str(consumer or "geometry_fit_preflight_cache"),
                status="background_cache_pair_validation_failed",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=snapshot_signature,
                stored_signature_summary=prebuild_deps.live_cache_signature_summary(
                    snapshot_signature
                ),
                raw_peak_count=int(len(snapshot_rows)),
                projected_peak_count=0,
                created_from=snapshot.get("created_from"),
                signature_match=True,
                live_cache_inventory=copy.deepcopy(
                    self.job_data.get("live_cache_inventory", {})
                ),
                live_runtime_cache_validation=snapshot_validation,
            )

        def worker_live_rows_payload() -> dict[str, object] | list[dict[str, object]]:
            signature_match = requested_signature == live_rows_signature
            payload_metadata = dict(live_rows_cache_metadata)
            payload_metadata.update(
                {
                    "live_rows_raw_count": int(len(live_rows)),
                    "live_rows_payload_count": int(
                        len(live_rows) if signature_match else 0
                    ),
                    "live_rows_signature_match": bool(signature_match),
                    "live_rows_signature_reason": (
                        "matched" if signature_match else "requested_signature_mismatch"
                    ),
                    "requested_signature_summary": requested_signature_summary,
                    "live_rows_signature_summary": (
                        prebuild_deps.live_cache_signature_summary(live_rows_signature)
                    ),
                    "live_rows_cache_source": str(
                        payload_metadata.get("live_rows_cache_source")
                        or payload_metadata.get("cache_source")
                        or "live_preview_cache"
                    ),
                    "live_rows_source_counts": prebuild_deps.live_row_source_counts(
                        live_rows
                    ),
                    "geometry_fit_live_handoff_patch_marker": (
                        str(prebuild_deps.live_handoff_patch_marker)
                    ),
                }
            )
            if not signature_match:
                payload_metadata["reason"] = "requested_signature_mismatch"
                return {
                    "rows": [],
                    "cache_metadata": payload_metadata,
                }
            return {
                "rows": [
                    dict(entry) for entry in (live_rows or ()) if isinstance(entry, Mapping)
                ],
                "cache_metadata": payload_metadata,
            }

        def worker_current_hit_table_cache_payload() -> dict[str, object]:
            cache_payload = prebuild_deps.int_keyed_mapping(
                self.job_data.get("current_hit_table_cache_by_background", {})
            ).get(int(background_idx))
            if not isinstance(cache_payload, Mapping):
                return {
                    "hit_tables": [],
                    "cache_metadata": {
                        "background_index": int(background_idx),
                        "reason": "missing_current_hit_table_cache",
                    },
                }
            raw_metadata = cache_payload.get("cache_metadata")
            metadata = (
                copy.deepcopy(dict(raw_metadata)) if isinstance(raw_metadata, Mapping) else {}
            )
            metadata.setdefault("background_index", int(background_idx))
            return {
                "hit_tables": copy.deepcopy(list(cache_payload.get("hit_tables") or ())),
                "cache_metadata": metadata,
            }

        projection_view_mode = (
            str(self.job_data.get("projection_view_mode") or "detector").strip().lower()
            or "detector"
        )
        try:
            detector_shape_for_projection = tuple(
                int(v)
                for v in np.asarray(
                    dict(self.job_data.get("background_images", {}) or {})
                    .get(int(background_idx), {})
                    .get("native"),
                    dtype=np.float64,
                ).shape[:2]
            )
        except Exception:
            detector_shape_for_projection = None
        ensured_projection_payload: dict[str, object] | None = None
        if projection_view_mode == "caked":
            ensured_projection_payload = self.ensure_worker_caked_projection_payload(
                int(background_idx),
                detector_shape=detector_shape_for_projection,
                stage_callback=stage_callback,
                emit_event=True,
            )
        projection_view_signature = prebuild_deps.projection_view_signature_for_background(
            int(background_idx)
        )
        projection_payload = (
            dict(
                self.job_data.setdefault("projection_payload_by_background", {}).get(
                    int(background_idx)
                )
                or {}
            )
            or None
        )
        candidate_state = "absent"
        if projection_view_mode == "caked":
            _projection_candidate, candidate_state = self.projection_candidate_state(
                projection_payload,
                detector_shape=detector_shape_for_projection,
            )
            if candidate_state != "ready":
                projection_payload = None
        if (
            projection_view_mode == "caked"
            and candidate_state == "absent"
            and isinstance(ensured_projection_payload, Mapping)
        ):
            projection_payload = prebuild_deps.projection_payload_storage_copy(
                ensured_projection_payload
            )
            if isinstance(projection_payload, Mapping):
                self.job_data.setdefault("projection_payload_by_background", {})[
                    int(background_idx)
                ] = projection_payload
        if (
            projection_view_mode == "caked"
            and candidate_state == "absent"
            and not isinstance(projection_payload, Mapping)
        ):
            projection_payload = self.load_caked_projection_by_index_snapshot(
                int(background_idx),
                detector_shape=detector_shape_for_projection,
                allow_generated_payload=True,
            )
        if projection_view_mode == "caked" and isinstance(projection_payload, Mapping):
            projection_payload = prebuild_deps.hydrate_exact_caked_payload(
                projection_payload,
                detector_shape=detector_shape_for_projection,
                params=params_local,
                require_background=False,
            )
            if isinstance(projection_payload, Mapping):
                stored_projection_payload = prebuild_deps.projection_payload_storage_copy(
                    projection_payload
                )
                if isinstance(stored_projection_payload, Mapping):
                    self.job_data.setdefault("projection_payload_by_background", {})[
                        int(background_idx)
                    ] = stored_projection_payload
        if projection_view_mode == "caked" and (
            not isinstance(projection_payload, Mapping)
            or not prebuild_deps.is_transform_bundle(
                projection_payload.get("transform_bundle")
            )
        ):
            raise RuntimeError(
                f"exact caked projector unavailable for background {int(background_idx) + 1}"
            )
        logged_cache_metadata_loader, logged_cache_loader = (
            prebuild_deps.logged_intersection_cache_loaders()
        )
        solver_inputs = self.job_data.get("solver_inputs")
        rebuild_result = prebuild_deps.rebuild_geometry_fit_source_rows(
            background_index=int(background_idx),
            background_label=str(background_label),
            params_local=params_local,
            consumer=str(consumer or "geometry_fit_preflight_cache"),
            prior_diagnostics=(
                dict(prior_diagnostics)
                if isinstance(prior_diagnostics, Mapping)
                else self.last_worker_source_snapshot_diagnostics()
            ),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            projection_view_mode=projection_view_mode,
            projection_view_signature=projection_view_signature,
            projection_payload=projection_payload,
            can_use_live_runtime_cache=(
                int(background_idx)
                == int(self.job_data.get("current_background_index", -1))
                or bool(live_rows)
            ),
            build_live_rows=worker_live_rows_payload,
            get_memory_intersection_cache=(
                lambda: prebuild_deps.copy_intersection_cache_tables(
                    self.job_data.get("memory_intersection_cache", [])
                )
            ),
            memory_cache_signature=self.job_data.get("memory_intersection_cache_signature"),
            load_logged_intersection_cache_metadata=logged_cache_metadata_loader,
            load_logged_intersection_cache=logged_cache_loader,
            logged_cache_matches_params=prebuild_deps.logged_cache_matches_params,
            build_source_rows_from_hit_tables=(
                lambda source_tables, **kwargs: prebuild_deps.forward_source_rows_for_rebuild(
                    prebuild_deps.build_source_rows_for_rebuild,
                    source_tables,
                    params_local=params_local,
                    fallback_consumer=str(consumer or "geometry_fit_preflight_cache"),
                    kwargs=kwargs,
                )
            ),
            simulate_hit_tables=(
                lambda normalized_params, **kwargs: prebuild_deps.simulate_hit_tables_for_fit(
                    getattr(solver_inputs, "miller"),
                    getattr(solver_inputs, "intensities"),
                    int(getattr(solver_inputs, "image_size")),
                    normalized_params,
                    **kwargs,
                )
            ),
            last_runtime_simulation_diagnostics=self.last_worker_simulation_diagnostics,
            project_rows=(
                lambda rows: self.project_source_rows_for_background(
                    int(background_idx),
                    rows,
                )
            ),
            project_rows_for_background_view=(
                lambda rows: self.project_source_rows_for_background(
                    int(background_idx),
                    rows,
                )
            ),
            required_pairs=required_pairs,
            required_branch_group_keys=required_branch_group_keys,
            required_manual_fit_targets=required_manual_fit_targets,
            preflight_mode=preflight_mode,
            live_cache_inventory=self.job_data.get("live_cache_inventory", {}),
            get_targeted_projected_cache=(
                lambda key_digest: prebuild_deps.load_targeted_projected_cache_entry(
                    background_index=int(background_idx),
                    key_digest=str(key_digest),
                )
            ),
            store_targeted_projected_cache=(
                lambda key_digest, payload: prebuild_deps.store_targeted_projected_cache_entry(
                    background_index=int(background_idx),
                    key_digest=str(key_digest),
                    payload=payload,
                )
            ),
            get_current_hit_table_cache=worker_current_hit_table_cache_payload,
            current_background_index=int(self.job_data.get("current_background_index", -1)),
            stage_callback=stage_callback,
        )
        self.set_worker_source_snapshot_diagnostics(
            **dict(rebuild_result.diagnostics or {})
        )
        self.worker_simulation_diagnostics.clear()
        self.worker_simulation_diagnostics.update(dict(rebuild_result.diagnostics or {}))
        if rebuild_result.stored_rows:
            bundle = self.build_geometry_fit_background_cache_bundle(
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=rebuild_result.requested_signature,
                requested_signature_summary=rebuild_result.requested_signature_summary,
                theta_base=float(theta_base),
                theta_initial=float(params_local.get("theta_initial", 0.0)),
                stored_rows=rebuild_result.stored_rows,
                projected_rows=rebuild_result.projected_rows,
                cache_source=str(rebuild_result.rebuild_source or "unknown"),
                diagnostics=dict(rebuild_result.diagnostics or {}),
                peak_table_lattice=rebuild_result.peak_table_lattice,
                hit_tables=rebuild_result.hit_tables,
                intersection_cache=rebuild_result.intersection_cache,
                cache_metadata=rebuild_result.metadata,
            )
            self.store_worker_background_cache_bundle(bundle)
            return bundle
        return None

    def rebuild_source_rows_for_background_worker(
        self,
        background_index: int,
        param_set: dict[str, object] | None = None,
        *,
        consumer: str | None = None,
        prior_diagnostics: Mapping[str, object] | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        cache_deps = self._require_cache_bundle_deps()
        source_deps = self._require_source_projection_deps()
        source_rows_deps = self._require_source_rows_deps()
        background_idx = int(background_index)
        lookup_context = str(consumer or "geometry_fit_dataset")

        def trial_source_row_projection_mode() -> str:
            base_mode = str(
                self.job_data.get("projection_view_mode") or "detector"
            ).strip().lower()
            if base_mode not in {"detector", "caked", "q_space"}:
                base_mode = "detector"
            if bool(
                source_rows_deps.manual_caked_fit_space_required_for_background(
                    int(background_idx)
                )
            ):
                return "caked"
            manual_spaces = self.job_data.get("manual_fit_space_by_background")
            manual_space = ""
            if isinstance(manual_spaces, Mapping):
                manual_space = (
                    str(
                        manual_spaces.get(
                            int(background_idx),
                            manual_spaces.get(str(int(background_idx)), ""),
                        )
                        or ""
                    )
                    .strip()
                    .lower()
                )
            if bool(self.job_data.get("pick_uses_caked_space", False)) or manual_space == "caked":
                return "caked"
            return base_mode

        def trial_source_rows_from_prebuilt_cache() -> list[dict[str, object]]:
            if lookup_context != "geometry_fit_trial_source_rows":
                return []
            cached_bundle = self.worker_background_cache_by_index.get(int(background_idx))
            if not cache_deps.is_background_cache_bundle(cached_bundle):
                return []
            normalized_mode = trial_source_row_projection_mode()
            caked_required = bool(
                source_rows_deps.manual_caked_fit_space_required_for_background(
                    int(background_idx)
                )
            )
            projected_rows = source_deps.rows_for_background(
                int(background_idx),
                self.project_source_rows_for_background(
                    int(background_idx),
                    cached_bundle.stored_rows,
                    mode_override=normalized_mode,
                    strict_caked_projection=bool(caked_required),
                    params_override=param_set,
                ),
            )
            if projected_rows:
                return self.mark_worker_cached_projection_rows(
                    projected_rows,
                    background_index=int(background_idx),
                    mode=normalized_mode,
                )
            if caked_required:
                return []
            return self.bundle_rows(
                cached_bundle,
                mode_override=normalized_mode,
                params_override=param_set,
            )

        cached_trial_rows = trial_source_rows_from_prebuilt_cache()
        if cached_trial_rows:
            return cached_trial_rows

        bundle = self.prebuild_background_cache_bundle_worker(
            background_idx,
            theta_base=float(source_rows_deps.theta_base_for_background(int(background_idx))),
            param_set=param_set,
            consumer=lookup_context,
            prior_diagnostics=prior_diagnostics,
            required_pairs=required_pairs,
        )
        if not cache_deps.is_background_cache_bundle(bundle):
            return trial_source_rows_from_prebuilt_cache()
        rows = self.bundle_rows(
            bundle,
            mode_override=(
                trial_source_row_projection_mode()
                if lookup_context == "geometry_fit_trial_source_rows"
                else None
            ),
            params_override=param_set,
        )
        if rows:
            return rows
        return trial_source_rows_from_prebuilt_cache()

    def source_rows_for_background_worker(
        self,
        background_index: int,
        param_set: dict[str, object] | None = None,
        *,
        consumer: str | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        cache_deps = self._require_cache_bundle_deps()
        prebuild_deps = self._require_prebuild_deps()
        source_rows_deps = self._require_source_rows_deps()
        background_idx = int(background_index)
        lookup_context = str(consumer or "unspecified")
        requested_signature = dict(
            self.job_data.get("requested_signatures", {}) or {}
        ).get(int(background_idx))
        requested_signature_summary = dict(
            self.job_data.get("requested_signature_summaries", {}) or {}
        ).get(int(background_idx))
        if requested_signature_summary is None:
            requested_signature_summary = prebuild_deps.live_cache_signature_summary(
                requested_signature
            )
        background_label = dict(self.job_data.get("background_labels", {}) or {}).get(
            int(background_idx),
            f"background {int(background_idx) + 1}",
        )
        live_cache_inventory = copy.deepcopy(
            self.job_data.get("live_cache_inventory", {})
        )
        bundle = self.worker_background_cache_by_index.get(int(background_idx))
        if not cache_deps.is_background_cache_bundle(bundle):
            self.set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_missing",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                raw_peak_count=0,
                projected_peak_count=0,
                signature_match=False,
                live_cache_inventory=live_cache_inventory,
            )
            return []

        stored_signature_summary = prebuild_deps.live_cache_signature_summary(
            bundle.requested_signature
        )
        if bundle.requested_signature != requested_signature:
            self.set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_signature_mismatch",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=bundle.requested_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(bundle.stored_rows or ())),
                projected_peak_count=0,
                created_from=bundle.cache_source,
                signature_match=False,
                live_cache_inventory=live_cache_inventory,
            )
            return []

        projected_rows = self.bundle_rows(bundle)
        if not projected_rows:
            self.set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_empty",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=bundle.requested_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(bundle.stored_rows or ())),
                projected_peak_count=0,
                created_from=bundle.cache_source,
                signature_match=True,
                live_cache_inventory=live_cache_inventory,
            )
            return []
        validation_rows = (
            projected_rows
            if bool(
                source_rows_deps.manual_caked_fit_space_required_for_background(
                    int(background_idx)
                )
            )
            else bundle.stored_rows
        )
        bundle_validation = (
            prebuild_deps.validate_required_source_rows_for_fit_space(
                validation_rows,
                required_pairs=required_pairs,
                background_index=int(background_idx),
            )
            if required_pairs
            else {}
        )
        if required_pairs and not bool(bundle_validation.get("valid", False)):
            self.set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_pair_validation_failed",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=bundle.requested_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(bundle.stored_rows or ())),
                projected_peak_count=0,
                created_from=bundle.cache_source,
                cache_source=bundle.cache_source,
                signature_match=True,
                theta_base=float(bundle.theta_base),
                theta_initial=float(bundle.theta_initial),
                live_cache_inventory=live_cache_inventory,
                live_runtime_cache_validation=bundle_validation,
            )
            rebuilt_rows = self.rebuild_source_rows_for_background_worker(
                background_idx,
                param_set,
                consumer=lookup_context,
                prior_diagnostics=self.last_worker_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
            return []

        self.set_worker_source_snapshot_diagnostics(
            source="geometry_fit_background_cache",
            cache_family="geometry_fit_background_cache",
            action="lookup",
            consumer=lookup_context,
            status="background_cache_hit",
            background_index=int(background_idx),
            background_label=str(background_label),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            snapshot_signature=bundle.requested_signature,
            stored_signature_summary=stored_signature_summary,
            raw_peak_count=int(len(bundle.stored_rows or ())),
            projected_peak_count=int(len(projected_rows)),
            created_from=bundle.cache_source,
            cache_source=bundle.cache_source,
            signature_match=True,
            theta_base=float(bundle.theta_base),
            theta_initial=float(bundle.theta_initial),
            live_cache_inventory=live_cache_inventory,
            live_runtime_cache_validation=bundle_validation,
        )
        return projected_rows

    def prebuild_required_background_caches(
        self,
        *,
        stage_callback: object = None,
    ) -> None:
        required_cache_deps = self._require_required_cache_deps()
        source_projection_deps = self._require_source_projection_deps()
        source_rows_deps = self._require_source_rows_deps()
        cache_bundle_deps = self._require_cache_bundle_deps()
        job_data = self.job_data
        caked_view_timeout_s = 5.0

        def _manual_fit_space_kind_for_background(background_index: int) -> str:
            stored_spaces = job_data.get("manual_fit_space_by_background")
            if isinstance(stored_spaces, Mapping):
                raw_kind = stored_spaces.get(
                    int(background_index),
                    stored_spaces.get(str(int(background_index))),
                )
                kind = str(raw_kind or "").strip().lower()
                if kind in {"caked", "mixed"}:
                    return kind
            if bool(job_data.get("pick_uses_caked_space", False)):
                return "caked"
            return "detector"

        required_indices = [int(idx) for idx in (job_data.get("required_indices", ()) or ())]
        for background_idx in required_indices:
            prebuild_started_at = perf_counter()
            background_label = dict(job_data.get("background_labels", {}) or {}).get(
                int(background_idx),
                f"background {int(background_idx) + 1}",
            )
            required_pairs = list(
                dict(job_data.get("manual_pairs_by_background", {}) or {}).get(
                    int(background_idx),
                    (),
                )
                or ()
            )
            self.emit_event(
                "source_cache_build_start",
                {
                    "background_index": int(background_idx),
                    "background_label": str(background_label),
                    "elapsed_s": 0.0,
                    "message": (
                        f"preflight: building source cache for background {int(background_idx) + 1}"
                    ),
                },
            )
            bundle_started_at = perf_counter()
            self.emit_event(
                "source_cache_bundle_start",
                {
                    "background_index": int(background_idx),
                    "background_label": str(background_label),
                    "elapsed_s": float(max(0.0, perf_counter() - prebuild_started_at)),
                    "status": "starting",
                    "required_pair_count": int(len(required_pairs)),
                    "message": (
                        "preflight: source cache bundle start for "
                        f"background {int(background_idx) + 1}"
                    ),
                },
            )
            bundle = self.prebuild_background_cache_bundle_worker(
                int(background_idx),
                theta_base=float(source_rows_deps.theta_base_for_background(int(background_idx))),
                required_pairs=required_pairs,
                stage_callback=stage_callback,
            )
            if not cache_bundle_deps.is_background_cache_bundle(bundle):
                failure_status = str(
                    self.last_worker_source_snapshot_diagnostics().get(
                        "status",
                        "background_cache_build_failed",
                    )
                )
                self.emit_event(
                    "source_cache_bundle_failed",
                    {
                        "background_index": int(background_idx),
                        "background_label": str(background_label),
                        "status": failure_status,
                        "elapsed_s": float(max(0.0, perf_counter() - prebuild_started_at)),
                        "stage_elapsed_s": float(max(0.0, perf_counter() - bundle_started_at)),
                        "message": (
                            "preflight: source cache bundle failed for "
                            f"background {int(background_idx) + 1} "
                            f"(status={failure_status})"
                        ),
                    },
                )
                if failure_status in {
                    "targeted_fresh_simulation_timeout",
                    "fresh_simulation_timeout",
                }:
                    raise RuntimeError(
                        "Geometry fit preflight timed out while rebuilding source cache "
                        f"for background {int(background_idx) + 1}. Refresh the "
                        "caked/source cache or reduce the fit simulation grid before rerunning."
                    )
                continue
            projected_rows_len = int(len(bundle.projected_rows or ()))
            stored_rows_len = int(len(bundle.stored_rows or ()))
            bundle_row_count = int(projected_rows_len or stored_rows_len)
            source_cache_generation_id = int(self.current_source_cache_generation(int(background_idx)))
            bundle_payload = {
                "background_index": int(background_idx),
                "background_label": str(background_label),
                "source_cache_generation_id": int(source_cache_generation_id),
                "cache_source": str(bundle.cache_source or "unknown"),
                "row_count": int(bundle_row_count),
                "required_pair_count": int(len(required_pairs)),
                "elapsed_s": float(max(0.0, perf_counter() - prebuild_started_at)),
                "stage_elapsed_s": float(max(0.0, perf_counter() - bundle_started_at)),
            }
            manual_fit_space_kind = _manual_fit_space_kind_for_background(int(background_idx))
            locked_qr_readiness = dict(
                required_cache_deps.locked_qr_fit_space_projection_readiness(
                    bundle.projected_rows,
                    required_pairs=required_pairs,
                    source_rows=bundle.stored_rows,
                )
            )
            locked_qr_readiness.update(
                {
                    "readiness_input_source": (
                        "projected_rows" if projected_rows_len > 0 else "none"
                    ),
                    "projected_rows_len": int(projected_rows_len),
                    "stored_rows_len": int(stored_rows_len),
                }
            )
            locked_qr_expected_rows = int(
                locked_qr_readiness.get("expected_locked_qr_rows", 0) or 0
            )
            locked_qr_projection_ready = bool(
                locked_qr_readiness.get("fit_space_projection_ready", False)
            )

            def _readiness_projection_payload_available() -> bool:
                payload_map = job_data.get("projection_payload_by_background")
                caked_view_map = job_data.get("caked_views_by_background")
                return bool(
                    (isinstance(payload_map, Mapping) and int(background_idx) in payload_map)
                    or (
                        isinstance(caked_view_map, Mapping)
                        and int(background_idx) in caked_view_map
                    )
                )

            def _refresh_locked_qr_readiness_from_caked_projection() -> None:
                nonlocal locked_qr_readiness
                nonlocal locked_qr_expected_rows
                nonlocal locked_qr_projection_ready
                if (
                    manual_fit_space_kind == "caked"
                    or locked_qr_expected_rows <= 0
                    or locked_qr_projection_ready
                    or stored_rows_len <= 0
                    or not _readiness_projection_payload_available()
                ):
                    return
                readiness_rows = source_projection_deps.rows_for_background(
                    int(background_idx),
                    self.project_source_rows_for_background(
                        int(background_idx),
                        bundle.stored_rows,
                        mode_override="caked",
                        strict_caked_projection=False,
                    ),
                )
                if not readiness_rows:
                    return
                readiness_candidate = dict(
                    required_cache_deps.locked_qr_fit_space_projection_readiness(
                        readiness_rows,
                        required_pairs=required_pairs,
                        source_rows=bundle.stored_rows,
                    )
                )
                if int(readiness_candidate.get("expected_locked_qr_rows", 0) or 0) <= 0:
                    return
                readiness_candidate.update(
                    {
                        "readiness_input_source": "readiness_caked_projection",
                        "projected_rows_len": int(len(readiness_rows)),
                        "stored_rows_len": int(stored_rows_len),
                    }
                )
                locked_qr_readiness = readiness_candidate
                locked_qr_expected_rows = int(
                    locked_qr_readiness.get("expected_locked_qr_rows", 0) or 0
                )
                locked_qr_projection_ready = bool(
                    locked_qr_readiness.get("fit_space_projection_ready", False)
                )

            def _locked_qr_projection_list_row_keys(
                readiness: Mapping[str, object],
                key_name: str,
            ) -> list[dict[str, object]]:
                return [
                    dict(entry)
                    for entry in readiness.get(key_name) or ()
                    if isinstance(entry, Mapping)
                ]

            def _locked_qr_projection_is_nonfinite_failure(
                readiness: Mapping[str, object],
            ) -> bool:
                reason = str(readiness.get("failure_reason") or "").strip().lower()
                return reason.replace("-", "_") in {
                    "nonfinite",
                    "non_finite",
                    "locked_qr_fit_space_projection_nonfinite",
                    "locked_qr_fit_space_projection_non_finite",
                }

            def _locked_qr_projection_issue_row_keys(
                readiness: Mapping[str, object],
            ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
                return (
                    _locked_qr_projection_list_row_keys(
                        readiness,
                        "missing_locked_qr_row_keys",
                    ),
                    _locked_qr_projection_list_row_keys(
                        readiness,
                        "nonfinite_locked_qr_row_keys",
                    ),
                )

            def _locked_qr_projection_primary_row_keys(
                readiness: Mapping[str, object],
            ) -> list[dict[str, object]]:
                missing_keys, nonfinite_keys = _locked_qr_projection_issue_row_keys(readiness)
                if _locked_qr_projection_is_nonfinite_failure(readiness):
                    return nonfinite_keys or missing_keys
                return missing_keys or nonfinite_keys

            def _locked_qr_projection_summary_row_keys(
                readiness: Mapping[str, object],
            ) -> list[dict[str, object]]:
                missing_keys, nonfinite_keys = _locked_qr_projection_issue_row_keys(readiness)
                issue_keys = [*missing_keys, *nonfinite_keys]
                if issue_keys:
                    return issue_keys
                return _locked_qr_projection_list_row_keys(readiness, "locked_qr_row_keys")

            def _locked_qr_projection_row_key_text(
                row_keys: Sequence[Mapping[str, object]],
            ) -> str:
                resolved_row_keys = [
                    dict(entry) for entry in row_keys if isinstance(entry, Mapping)
                ]
                if not resolved_row_keys:
                    return "none"
                labels: list[str] = []
                for key in resolved_row_keys[:4]:
                    labels.append(
                        f"pair_id={key.get('pair_id') or 'unknown'} "
                        f"hkl={key.get('hkl')} "
                        f"branch={key.get('branch')} "
                        f"table={key.get('table')} "
                        f"row={key.get('row')} "
                        f"peak={key.get('peak')} "
                        f"stage={key.get('first_missing_stage') or 'unknown'}"
                    )
                if len(resolved_row_keys) > len(labels):
                    labels.append(f"{len(resolved_row_keys) - len(labels)} more")
                return "; ".join(labels)

            def _locked_qr_projection_row_key_sections(
                readiness: Mapping[str, object],
            ) -> str:
                missing_keys, nonfinite_keys = _locked_qr_projection_issue_row_keys(readiness)
                nonfinite_failure = _locked_qr_projection_is_nonfinite_failure(readiness)
                if nonfinite_failure:
                    sections = [
                        ("Nonfinite row keys", nonfinite_keys),
                        ("Missing row keys", missing_keys),
                    ]
                else:
                    sections = [
                        ("Missing row keys", missing_keys),
                        ("Nonfinite row keys", nonfinite_keys),
                    ]
                labeled_sections = [
                    f"{label}: {_locked_qr_projection_row_key_text(row_keys)}."
                    for label, row_keys in sections
                    if row_keys
                ]
                if labeled_sections:
                    return " ".join(labeled_sections)
                fallback_label = "Nonfinite row keys" if nonfinite_failure else "Missing row keys"
                fallback_keys = _locked_qr_projection_primary_row_keys(readiness)
                return f"{fallback_label}: {_locked_qr_projection_row_key_text(fallback_keys)}."

            def _locked_qr_projection_yes_no_partial(
                readiness: Mapping[str, object],
                key_name: str,
            ) -> str:
                row_keys = _locked_qr_projection_summary_row_keys(readiness)
                if not row_keys:
                    return "unknown"
                values = [bool(entry.get(key_name, False)) for entry in row_keys]
                if all(values):
                    return "yes"
                if any(values):
                    return "partial"
                return "no"

            def _locked_qr_projection_diagnostic_summary(
                readiness: Mapping[str, object],
            ) -> dict[str, str]:
                optics_mode = required_cache_deps.normalize_optics_mode_label(
                    dict(job_data.get("params", {}) or {}).get(
                        "optics_mode",
                        job_data.get("optics_mode"),
                    )
                )
                return {
                    "source_rows_existed": _locked_qr_projection_yes_no_partial(
                        readiness,
                        "source_exists",
                    ),
                    "projected_rows_existed": _locked_qr_projection_yes_no_partial(
                        readiness,
                        "projected_exists",
                    ),
                    "projection_payload_status": (
                        "ready" if _readiness_projection_payload_available() else "missing"
                    ),
                    "optics_mode": optics_mode,
                }

            def _locked_qr_projection_error_text(
                readiness: Mapping[str, object],
                *,
                storage_status: str | None = None,
                storage_timeout_fatal: bool | None = None,
            ) -> str:
                expected_rows = int(readiness.get("expected_locked_qr_rows", 0) or 0)
                projected_rows = int(readiness.get("projected_locked_qr_rows", 0) or 0)
                reason = str(
                    readiness.get("failure_reason") or "locked_qr_fit_space_projection_missing"
                )
                resolved_storage_status = str(
                    storage_status or readiness.get("caked_view_storage_status") or "unknown"
                )
                diagnostic_summary = _locked_qr_projection_diagnostic_summary(readiness)
                timeout_text = (
                    "unknown"
                    if storage_timeout_fatal is None
                    else str(bool(storage_timeout_fatal)).lower()
                )
                detail_text = (
                    f" Source rows existed: {diagnostic_summary['source_rows_existed']}. "
                    "Projected caked rows existed: "
                    f"{diagnostic_summary['projected_rows_existed']}. "
                    "Exact caked projection payload: "
                    f"{diagnostic_summary['projection_payload_status']}. "
                    f"Caked view storage: {resolved_storage_status}. "
                    f"Storage timeout fatal: {timeout_text}. "
                    f"Optics mode: {diagnostic_summary['optics_mode']}. "
                    f"{_locked_qr_projection_row_key_sections(readiness)}"
                )
                if reason == "locked_qr_fit_space_projection_nonfinite":
                    return (
                        "exact caked fit-space projection has non-finite locked Qr/Qz "
                        f"rows. Expected {expected_rows} finite projected rows, found "
                        f"{projected_rows}." + detail_text
                    )
                return (
                    "exact caked fit-space projection is missing for locked Qr/Qz "
                    f"rows. Expected {expected_rows} projected rows, found {projected_rows}."
                    + detail_text
                )

            def _emit_locked_qr_projection_readiness(
                *,
                storage_status: str,
                storage_timeout_fatal: bool,
                last_chance_poll_ready: bool | None = None,
            ) -> None:
                expected_rows = int(locked_qr_readiness.get("expected_locked_qr_rows", 0) or 0)
                if expected_rows <= 0:
                    return
                projected_rows = int(locked_qr_readiness.get("projected_locked_qr_rows", 0) or 0)
                finite_rows = int(locked_qr_readiness.get("finite_locked_qr_rows", 0) or 0)
                projection_ready = bool(
                    locked_qr_readiness.get("fit_space_projection_ready", False)
                )
                projection_degenerate = bool(
                    locked_qr_readiness.get("projection_degenerate", False)
                )
                storage_required = bool(
                    locked_qr_readiness.get("caked_view_storage_required_for_fit", True)
                )
                diagnostic_summary = _locked_qr_projection_diagnostic_summary(locked_qr_readiness)
                message = (
                    "locked_qr_projection_readiness "
                    f"background={int(background_idx) + 1} "
                    f"expected_rows={expected_rows} projected_rows={projected_rows} "
                    f"finite_rows={finite_rows} "
                    f"projection_ready={str(projection_ready).lower()} "
                    f"projection_degenerate={str(projection_degenerate).lower()} "
                    f"storage_required_for_fit={str(storage_required).lower()} "
                    f"storage_status={str(storage_status)} "
                    f"storage_timeout_fatal={str(bool(storage_timeout_fatal)).lower()} "
                    f"source_rows_existed={diagnostic_summary['source_rows_existed']} "
                    f"projected_rows_existed={diagnostic_summary['projected_rows_existed']} "
                    f"projection_payload_status={diagnostic_summary['projection_payload_status']} "
                    f"optics={diagnostic_summary['optics_mode']}"
                )
                payload = {
                    **bundle_payload,
                    **locked_qr_readiness,
                    "projection_ready": bool(projection_ready),
                    "caked_view_storage_required_for_fit": bool(storage_required),
                    "caked_view_storage_status": str(storage_status),
                    "storage_timeout_fatal": bool(storage_timeout_fatal),
                    "source_rows_existed_for_locked_qr": diagnostic_summary["source_rows_existed"],
                    "projected_rows_existed_for_locked_qr": diagnostic_summary[
                        "projected_rows_existed"
                    ],
                    "exact_caked_projection_payload_status": diagnostic_summary[
                        "projection_payload_status"
                    ],
                    "optics_mode": diagnostic_summary["optics_mode"],
                    "full_cake_started": True,
                    "storage_timeout_s": float(caked_view_timeout_s),
                    "message": message,
                }
                if last_chance_poll_ready is not None:
                    payload["last_chance_poll_ready"] = bool(last_chance_poll_ready)
                self.emit_event("locked_qr_projection_readiness", payload)

            self.emit_event(
                "source_cache_bundle_ready",
                {
                    **bundle_payload,
                    "status": "ready",
                    "message": (
                        "preflight: source cache bundle ready for "
                        f"background {int(background_idx) + 1} "
                        f"({int(bundle_row_count)} rows via {str(bundle.cache_source or 'unknown')})"
                    ),
                },
            )
            self.emit_event(
                "source_cache_rows_ready",
                {
                    **bundle_payload,
                    "status": "rows_ready",
                    "message": (
                        "preflight: source cache rows ready for "
                        f"background {int(background_idx) + 1} "
                        f"({int(bundle_row_count)} rows via {str(bundle.cache_source or 'unknown')})"
                    ),
                },
            )
            self.emit_event(
                "source_cache_build_ready",
                {
                    **bundle_payload,
                    "status": "ready",
                    "message": (
                        "preflight: source cache ready for "
                        f"background {int(background_idx) + 1} "
                        f"({int(bundle_row_count)} rows via "
                        f"{str(bundle.cache_source or 'unknown')})"
                    ),
                },
            )
            caked_started_at = perf_counter()
            self.emit_event(
                "source_cache_caked_view_start",
                {
                    **bundle_payload,
                    "status": "starting",
                    "message": (
                        "preflight: caked view storage start for "
                        f"background {int(background_idx) + 1}"
                    ),
                },
            )
            await_caked_result, announce_caked_timeout = required_cache_deps.start_non_gating_caked_view_task(
                bundle,
                source_cache_generation_id=int(source_cache_generation_id),
                started_at=caked_started_at,
                stage_callback=stage_callback,
            )
            caked_outcome = await_caked_result(float(caked_view_timeout_s))
            caked_required = source_rows_deps.manual_caked_fit_space_required_for_background(
                int(background_idx)
            )
            locked_qr_projection_gate_required = (
                locked_qr_expected_rows > 0 and manual_fit_space_kind != "caked"
            )
            locked_qr_caked_origin_baseline = bool(
                locked_qr_expected_rows > 0 and manual_fit_space_kind == "caked"
            )

            def _locked_qr_projection_missing() -> bool:
                return bool(
                    locked_qr_projection_gate_required
                    and locked_qr_expected_rows > 0
                    and not locked_qr_projection_ready
                )

            def _caked_storage_timeout_fatal(caked_stored: bool) -> bool:
                return bool(
                    _locked_qr_projection_missing()
                    or (
                        caked_required
                        and not caked_stored
                        and not locked_qr_projection_ready
                        and not locked_qr_caked_origin_baseline
                    )
                )

            def _caked_storage_status(outcome: Mapping[str, object]) -> str:
                if bool(outcome.get("caked_view_stored", False)):
                    return "ready"
                return str(outcome.get("status") or outcome.get("caked_view_status") or "failed")

            def _handle_caked_storage_outcome(
                outcome: Mapping[str, object],
                *,
                last_chance_poll_ready: bool | None = None,
            ) -> bool:
                resolved_outcome = dict(outcome)
                caked_stored = bool(resolved_outcome.get("caked_view_stored", False))
                if caked_stored or _readiness_projection_payload_available():
                    _refresh_locked_qr_readiness_from_caked_projection()
                required_cache_deps.emit_source_cache_caked_view_event(
                    "source_cache_caked_view_ready"
                    if caked_stored
                    else "source_cache_caked_view_failed",
                    bundle,
                    source_cache_generation_id=int(source_cache_generation_id),
                    started_at=caked_started_at,
                    payload=resolved_outcome,
                )
                storage_status = _caked_storage_status(resolved_outcome)
                locked_qr_projection_missing = _locked_qr_projection_missing()
                storage_timeout_fatal = _caked_storage_timeout_fatal(caked_stored)
                _emit_locked_qr_projection_readiness(
                    storage_status=storage_status,
                    storage_timeout_fatal=storage_timeout_fatal,
                    last_chance_poll_ready=last_chance_poll_ready,
                )
                if locked_qr_projection_missing:
                    raise RuntimeError(
                        _locked_qr_projection_error_text(
                            locked_qr_readiness,
                            storage_status=storage_status,
                            storage_timeout_fatal=storage_timeout_fatal,
                        )
                    )
                if not caked_required or caked_stored:
                    return False
                if locked_qr_projection_ready or locked_qr_caked_origin_baseline:
                    return True
                if locked_qr_expected_rows > 0:
                    raise RuntimeError(
                        _locked_qr_projection_error_text(
                            locked_qr_readiness,
                            storage_status=storage_status,
                            storage_timeout_fatal=storage_timeout_fatal,
                        )
                    )
                raise RuntimeError(
                    "exact caked fit-space projection/storage unavailable for "
                    f"background {int(background_idx) + 1} "
                    f"(status={str(resolved_outcome.get('status', 'failed'))})"
                )

            if isinstance(caked_outcome, Mapping):
                if _handle_caked_storage_outcome(caked_outcome):
                    continue
            else:
                last_chance_caked_outcome = await_caked_result(0.0)
                if isinstance(last_chance_caked_outcome, Mapping):
                    if _handle_caked_storage_outcome(
                        last_chance_caked_outcome,
                        last_chance_poll_ready=bool(
                            last_chance_caked_outcome.get("caked_view_stored", False)
                        ),
                    ):
                        continue
                    continue
                required_cache_deps.emit_source_cache_caked_view_event(
                    "source_cache_caked_view_timeout",
                    bundle,
                    source_cache_generation_id=int(source_cache_generation_id),
                    started_at=caked_started_at,
                    payload={"status": "timeout"},
                )
                announce_caked_timeout()
                _refresh_locked_qr_readiness_from_caked_projection()
                locked_qr_projection_missing = _locked_qr_projection_missing()
                storage_status = (
                    "deferred"
                    if locked_qr_projection_ready or locked_qr_caked_origin_baseline
                    else "timeout"
                )
                storage_timeout_fatal = _caked_storage_timeout_fatal(False)
                _emit_locked_qr_projection_readiness(
                    storage_status=storage_status,
                    storage_timeout_fatal=storage_timeout_fatal,
                    last_chance_poll_ready=False,
                )
                if locked_qr_projection_missing:
                    raise RuntimeError(
                        _locked_qr_projection_error_text(
                            locked_qr_readiness,
                            storage_status=storage_status,
                            storage_timeout_fatal=storage_timeout_fatal,
                        )
                    )
                if caked_required:
                    if locked_qr_projection_ready or locked_qr_caked_origin_baseline:
                        continue
                    if locked_qr_expected_rows > 0:
                        raise RuntimeError(
                            _locked_qr_projection_error_text(
                                locked_qr_readiness,
                                storage_status="timeout",
                                storage_timeout_fatal=storage_timeout_fatal,
                            )
                        )
                    raise RuntimeError(
                        "exact caked fit-space projection/storage timed out for "
                        f"background {int(background_idx) + 1}"
                    )
