from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from ra_sim.gui._runtime.geometry_fit_worker import (
    GeometryFitWorkerCacheBundleDeps,
    GeometryFitWorkerCakedPayloadDeps,
    GeometryFitWorkerCakedViewStorageDeps,
    GeometryFitWorkerContext,
    GeometryFitWorkerDatasetDeps,
    GeometryFitWorkerManualFitSpaceDeps,
    GeometryFitWorkerPrebuildDeps,
    GeometryFitWorkerRequiredCacheDeps,
    GeometryFitWorkerResultDeps,
    GeometryFitWorkerSolverDeps,
    GeometryFitWorkerSourceProjectionDeps,
    GeometryFitWorkerSourceRowsDeps,
)


class RecordingQueue:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, item: object) -> None:
        self.items.append(item)


class FailingQueue:
    def put(self, _item: object) -> None:
        raise RuntimeError("queue unavailable")


class FakeBundle:
    def __init__(self, detector_shape: tuple[int, int] = (2, 3)) -> None:
        self.detector_shape = detector_shape


class FakeContextManager:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> bool:
        return False


class FakeBackgroundCacheBundle:
    def __init__(
        self,
        *,
        background_index: int = 0,
        background_label: str = "background 1",
        theta_base: float = 0.0,
        theta_initial: float = 0.0,
        stored_rows: list[dict[str, object]] | None = None,
        projected_rows: list[dict[str, object]] | None = None,
        cache_source: str = "fake_cache",
        requested_signature: object = ("signature", 0),
        requested_signature_summary: object = {"summary": 0},
        diagnostics: dict[str, object] | None = None,
        peak_table_lattice: list[object] | None = None,
        hit_tables: list[object] | None = None,
        intersection_cache: list[object] | None = None,
        cache_metadata: dict[str, object] | None = None,
    ) -> None:
        self.background_index = int(background_index)
        self.background_label = str(background_label)
        self.theta_base = float(theta_base)
        self.theta_initial = float(theta_initial)
        self.stored_rows = stored_rows if stored_rows is not None else []
        self.projected_rows = projected_rows if projected_rows is not None else []
        self.cache_source = cache_source
        self.requested_signature = requested_signature
        self.requested_signature_summary = requested_signature_summary
        self.diagnostics = diagnostics if diagnostics is not None else {}
        self.peak_table_lattice = peak_table_lattice
        self.hit_tables = hit_tables
        self.intersection_cache = intersection_cache
        self.cache_metadata = cache_metadata


class FakeRebuildResult:
    def __init__(
        self,
        *,
        background_index: int = 0,
        requested_signature: object = ("rebuilt", 0),
        requested_signature_summary: object | None = None,
        stored_rows: list[dict[str, object]] | None = None,
        projected_rows: list[dict[str, object]] | None = None,
        rebuild_source: str = "fresh_rebuild",
        diagnostics: dict[str, object] | None = None,
        peak_table_lattice: list[object] | None = None,
        hit_tables: list[object] | None = None,
        intersection_cache: list[object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.background_index = int(background_index)
        self.requested_signature = requested_signature
        self.requested_signature_summary = (
            {"summary": "rebuilt"}
            if requested_signature_summary is None
            else requested_signature_summary
        )
        self.stored_rows = stored_rows if stored_rows is not None else []
        self.projected_rows = projected_rows if projected_rows is not None else []
        self.rebuild_source = rebuild_source
        self.diagnostics = diagnostics if diagnostics is not None else {}
        self.peak_table_lattice = peak_table_lattice
        self.hit_tables = hit_tables
        self.intersection_cache = intersection_cache
        self.metadata = metadata


class FakeCakedPayloadDeps:
    def __init__(self) -> None:
        self.generated_payload: dict[str, object] | None = None
        self.generated_calls: list[dict[str, object]] = []
        self.stage_events: list[tuple[object, str, dict[str, object]]] = []
        self.hydrate_calls: list[dict[str, object]] = []
        self.normalize_calls: list[dict[str, object]] = []
        self.integrator: object | None = "fake-ai"

    @property
    def deps(self) -> GeometryFitWorkerCakedPayloadDeps:
        return GeometryFitWorkerCakedPayloadDeps(
            normalize_caked_view_payload=self.normalize_caked_view_payload,
            hydrate_exact_caked_payload=self.hydrate_exact_caked_payload,
            caked_projection_payload=self.caked_projection_payload,
            projection_payload_storage_copy=self.projection_payload_storage_copy,
            projection_payload_digest=self.projection_payload_digest,
            detector_shape_2d=self.detector_shape_2d,
            resolve_targeted_caked_projection_payload=(
                self.resolve_targeted_caked_projection_payload
            ),
            caking_integrator=self.caking_integrator,
            emit_stage_event=self.emit_stage_event,
            is_transform_bundle=self.is_transform_bundle,
        )

    def normalize_caked_view_payload(
        self,
        payload: object,
        *,
        detector_shape: object = None,
        ai: object = None,
    ) -> object:
        self.normalize_calls.append({"detector_shape": detector_shape, "ai": ai})
        return dict(payload) if isinstance(payload, Mapping) else payload

    def hydrate_exact_caked_payload(
        self,
        payload: object,
        *,
        detector_shape: object = None,
        params: object = None,
        require_background: bool = False,
    ) -> object:
        self.hydrate_calls.append(
            {
                "detector_shape": detector_shape,
                "params": params,
                "require_background": require_background,
            }
        )
        if not isinstance(payload, Mapping):
            return None
        hydrated = dict(payload)
        hydrated.setdefault("transform_bundle", FakeBundle())
        hydrated.setdefault("digest", "digest")
        return hydrated

    def caked_projection_payload(self, payload: object) -> object:
        if not isinstance(payload, Mapping):
            return None
        projection = payload.get("projection")
        if isinstance(projection, Mapping):
            return dict(projection)
        meaningful_keys = (
            "radial_axis",
            "azimuth_axis",
            "raw_azimuth_axis",
            "raw_to_gui_row_permutation",
            "transform_bundle",
        )
        if all(payload.get(key) is None for key in meaningful_keys):
            return None
        return payload

    def projection_payload_storage_copy(self, payload: object) -> object:
        return dict(payload) if isinstance(payload, Mapping) else payload

    def projection_payload_digest(self, payload: object) -> object:
        return payload.get("digest") if isinstance(payload, Mapping) else None

    def detector_shape_2d(self, detector_shape: object) -> tuple[int, int] | None:
        if not (isinstance(detector_shape, (tuple, list)) and len(detector_shape) >= 2):
            return None
        try:
            return (int(detector_shape[0]), int(detector_shape[1]))
        except Exception:
            return None

    def resolve_targeted_caked_projection_payload(
        self,
        background_index: int,
        *,
        detector_shape: object = None,
        ai: object = None,
        analysis_preview_bins: object = None,
        allow_generated_payload: bool = False,
    ) -> object:
        self.generated_calls.append(
            {
                "background_index": background_index,
                "detector_shape": detector_shape,
                "ai": ai,
                "analysis_preview_bins": analysis_preview_bins,
                "allow_generated_payload": allow_generated_payload,
            }
        )
        return self.generated_payload

    def caking_integrator(self) -> object:
        return self.integrator

    def emit_stage_event(
        self,
        stage_callback: object,
        kind: str,
        **payload: object,
    ) -> None:
        self.stage_events.append((stage_callback, kind, payload))

    def is_transform_bundle(self, value: object) -> bool:
        return isinstance(value, FakeBundle)


def _valid_caked_display_payload() -> dict[str, object]:
    return {
        "image": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "radial": np.array([1.0, 2.0], dtype=np.float64),
        "azimuth": np.array([3.0, 4.0], dtype=np.float64),
        "raw_azimuth_axis": np.array([3.0, 4.0], dtype=np.float64),
        "raw_to_gui_row_permutation": np.array([0, 1], dtype=np.int32),
        "transform_bundle": FakeBundle(detector_shape=(2, 2)),
        "detector_shape": (2, 2),
        "digest": "projection-digest",
    }


class FakeCakedViewStorageDeps:
    def __init__(self) -> None:
        self.stage_events: list[tuple[object, str, dict[str, object]]] = []
        self.roi_selection: dict[str, object] = {
            "enabled": False,
            "valid": False,
            "pixel_count": 0,
            "fraction": 0.0,
            "half_width_px": 0.0,
            "fallback_reason": None,
        }
        self.caking_result: object = {"cake": "result"}
        self.prepared_payload: object = _valid_caked_display_payload()
        self.sanitized_payload: object | None = None
        self.sanitize_diag: dict[str, object] = {
            "status": "stored",
            "sanitized_empty_bin_count": 1,
            "nonfinite_supported_bin_count": 2,
        }
        self.caking_calls: list[dict[str, object]] = []

    @property
    def deps(self) -> GeometryFitWorkerCakedViewStorageDeps:
        return GeometryFitWorkerCakedViewStorageDeps(
            emit_geometry_fit_stage_event=self.emit_geometry_fit_stage_event,
            build_caked_roi_selection=self.build_caked_roi_selection,
            caked_roi_fit_space_to_detector_point=(
                self.caked_roi_fit_space_to_detector_point
            ),
            worker_projection_analysis_bins=self.worker_projection_analysis_bins,
            geometry_fit_worker_caked_projection_view=(
                self.geometry_fit_worker_caked_projection_view
            ),
            temporary_numba_thread_limit=self.temporary_numba_thread_limit,
            default_reserved_cpu_worker_count=self.default_reserved_cpu_worker_count,
            caking=self.caking,
            prepare_caked_display_payload=self.prepare_caked_display_payload,
            sanitize_caked_display_payload=self.sanitize_caked_display_payload,
            normalize_projection_view_signature=self.normalize_projection_view_signature,
            targeted_projection_view_signature=self.targeted_projection_view_signature,
            digest_payload=self.digest_payload,
            replace_bundle=self.replace_bundle,
        )

    def emit_geometry_fit_stage_event(
        self,
        stage_callback: object,
        kind: str,
        **payload: object,
    ) -> None:
        self.stage_events.append((stage_callback, str(kind), dict(payload)))

    def build_caked_roi_selection(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return dict(self.roi_selection)

    def caked_roi_fit_space_to_detector_point(self, **_kwargs: object) -> object:
        return lambda point: point

    def worker_projection_analysis_bins(self) -> tuple[int, int]:
        return 4, 5

    def geometry_fit_worker_caked_projection_view(self, **_kwargs: object) -> dict[str, object]:
        return {
            "radial_axis": np.array([1.0, 2.0], dtype=np.float64),
            "azimuth_axis": np.array([3.0, 4.0], dtype=np.float64),
            "transform_bundle": FakeBundle(detector_shape=(2, 2)),
        }

    def temporary_numba_thread_limit(self, _count: object) -> FakeContextManager:
        return FakeContextManager()

    def default_reserved_cpu_worker_count(self) -> int:
        return 0

    def caking(
        self,
        background: object,
        ai: object,
        **kwargs: object,
    ) -> object:
        self.caking_calls.append(
            {
                "background_shape": tuple(np.asarray(background).shape),
                "ai": ai,
                **kwargs,
            }
        )
        return self.caking_result

    def prepare_caked_display_payload(self, *_args: object, **_kwargs: object) -> object:
        return self.prepared_payload

    def sanitize_caked_display_payload(
        self,
        payload: object,
    ) -> tuple[object, dict[str, object]]:
        return (
            self.sanitized_payload if self.sanitized_payload is not None else payload,
            dict(self.sanitize_diag),
        )

    def normalize_projection_view_signature(
        self,
        signature: object,
        background_index: int,
    ) -> dict[str, object]:
        return dict(signature, normalized_background_index=int(background_index))

    def targeted_projection_view_signature(
        self,
        background_index: int,
        **kwargs: object,
    ) -> dict[str, object]:
        return {"background_index": int(background_index), **kwargs}

    def digest_payload(self, payload: object) -> tuple[str, str]:
        return ("digest", repr(payload))

    def replace_bundle(
        self,
        bundle: FakeBackgroundCacheBundle,
        **changes: object,
    ) -> FakeBackgroundCacheBundle:
        values = {
            "background_index": bundle.background_index,
            "background_label": bundle.background_label,
            "theta_base": bundle.theta_base,
            "theta_initial": bundle.theta_initial,
            "stored_rows": bundle.stored_rows,
            "projected_rows": bundle.projected_rows,
            "cache_source": bundle.cache_source,
            "requested_signature": bundle.requested_signature,
            "requested_signature_summary": bundle.requested_signature_summary,
            "diagnostics": bundle.diagnostics,
            "peak_table_lattice": bundle.peak_table_lattice,
            "hit_tables": bundle.hit_tables,
            "intersection_cache": bundle.intersection_cache,
            "cache_metadata": bundle.cache_metadata,
        }
        values.update(changes)
        return FakeBackgroundCacheBundle(**values)


class FakeProjectionCallbacks:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.projected_rows: list[dict[str, object]] | None = None
        self.error: Exception | None = None

    def project_peaks_to_current_view(
        self,
        rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        self.calls.append([dict(row) for row in rows])
        if self.error is not None:
            raise self.error
        if self.projected_rows is not None:
            return [dict(row) for row in self.projected_rows]
        return [dict(row, projected=True) for row in rows]


class FakeSourceProjectionDeps:
    def __init__(self) -> None:
        self.callbacks = FakeProjectionCallbacks()
        self.rotate_calls: list[dict[str, object]] = []

    @property
    def deps(self) -> GeometryFitWorkerSourceProjectionDeps:
        return GeometryFitWorkerSourceProjectionDeps(
            rows_for_background=self.rows_for_background,
            group_rows_by_background=self.group_rows_by_background,
            overlay_state_finite_pair=self.overlay_state_finite_pair,
            make_projection_callbacks=self.make_projection_callbacks,
            native_detector_coords_to_bundle_detector_coords=(
                self.native_detector_coords_to_bundle_detector_coords
            ),
            raw_phi_to_gui_phi=lambda value: float(value),
            rotate_point_for_display=self.rotate_point_for_display,
            native_sim_to_display_coords=None,
            get_detector_angular_maps=lambda _ai: None,
            detector_pixel_to_scattering_angles=None,
            backend_detector_coords_to_native_detector_coords=(
                lambda col, row, *_args, **_kwargs: (float(col), float(row))
            ),
            scattering_angles_to_detector_pixel=None,
            profile_cache=lambda: {},
            default_pixel_size_m=1.0e-4,
            default_image_size=256,
            default_display_rotate_k=0,
        )

    def rows_for_background(
        self,
        background_index: int,
        raw_rows: object,
    ) -> list[dict[str, object]]:
        normalized_rows: list[dict[str, object]] = []
        for raw_entry in raw_rows or ():
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            entry.setdefault("background_index", int(background_index))
            normalized_rows.append(entry)
        return normalized_rows

    def group_rows_by_background(
        self,
        raw_rows: object,
        *,
        default_background_index: int,
        order_key: str | None = None,
    ) -> list[tuple[int, list[dict[str, object]]]]:
        grouped_rows: dict[int, list[dict[str, object]]] = {}
        ordered_backgrounds: list[int] = []
        for position, raw_entry in enumerate(raw_rows or ()):
            if not isinstance(raw_entry, Mapping):
                continue
            entry = dict(raw_entry)
            try:
                background_idx = int(entry.get("background_index"))
            except Exception:
                background_idx = int(default_background_index)
            entry.setdefault("background_index", int(background_idx))
            if order_key is not None:
                entry[order_key] = int(position)
            if int(background_idx) not in grouped_rows:
                ordered_backgrounds.append(int(background_idx))
                grouped_rows[int(background_idx)] = []
            grouped_rows[int(background_idx)].append(entry)
        return [
            (int(background_idx), list(grouped_rows.get(int(background_idx), ())))
            for background_idx in ordered_backgrounds
        ]

    def make_projection_callbacks(self, **_kwargs: object) -> FakeProjectionCallbacks:
        return self.callbacks

    def overlay_state_finite_pair(self, value: object) -> tuple[float, float] | None:
        if not (isinstance(value, (tuple, list)) and len(value) >= 2):
            return None
        try:
            col = float(value[0])
            row = float(value[1])
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return col, row

    def rotate_point_for_display(
        self,
        col: float,
        row: float,
        shape: object = None,
        rotate_k: object = None,
    ) -> tuple[float, float]:
        self.rotate_calls.append(
            {
                "col": float(col),
                "row": float(row),
                "shape": shape,
                "rotate_k": rotate_k,
            }
        )
        return float(col) + 10.0, float(row) + 20.0

    def native_detector_coords_to_bundle_detector_coords(
        self,
        col: float,
        row: float,
        _detector_shape: object,
    ) -> tuple[float, float]:
        return float(col), float(row)


class FakeManualDatasetBindings:
    def __init__(self, **kwargs: object) -> None:
        self.geometry_manual_pairs_for_index = kwargs["geometry_manual_pairs_for_index"]


class FakeDatasetDeps:
    def __init__(self) -> None:
        self.binding_calls: list[dict[str, object]] = []
        self.bindings: list[FakeManualDatasetBindings] = []
        self.prepare_calls: list[dict[str, object]] = []
        self.dataset_calls: list[dict[str, object]] = []
        self.prepare_result: object = {"prepared": True}

    @property
    def deps(self) -> GeometryFitWorkerDatasetDeps:
        return GeometryFitWorkerDatasetDeps(
            make_manual_dataset_bindings=self.make_manual_dataset_bindings,
            prepare_geometry_fit_run=self.prepare_geometry_fit_run,
            build_geometry_manual_fit_dataset=self.build_geometry_manual_fit_dataset,
        )

    def make_manual_dataset_bindings(self, **kwargs: object) -> FakeManualDatasetBindings:
        self.binding_calls.append(dict(kwargs))
        bindings = FakeManualDatasetBindings(**kwargs)
        self.bindings.append(bindings)
        return bindings

    def prepare_geometry_fit_run(self, **kwargs: object) -> object:
        self.prepare_calls.append(dict(kwargs))
        return self.prepare_result

    def build_geometry_manual_fit_dataset(
        self,
        background_index: int,
        **kwargs: object,
    ) -> dict[str, object]:
        self.dataset_calls.append(
            {
                "background_index": int(background_index),
                "kwargs": dict(kwargs),
            }
        )
        return {"dataset_background_index": int(background_index)}


class FakeSolverDeps:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result: object = SimpleNamespace(error_text="solver-error")

    @property
    def deps(self) -> GeometryFitWorkerSolverDeps:
        return GeometryFitWorkerSolverDeps(
            execute_solver_phase=self.execute_solver_phase,
        )

    def execute_solver_phase(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self.result


class FakeResultDeps:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.result: object = {"action_result": True}

    @property
    def deps(self) -> GeometryFitWorkerResultDeps:
        return GeometryFitWorkerResultDeps(
            build_action_result=self.build_action_result,
        )

    def build_action_result(self, job: object, **kwargs: object) -> object:
        self.calls.append({"job": job, "kwargs": dict(kwargs)})
        return self.result


class FakePrebuildDeps:
    def __init__(self) -> None:
        self.validation_result: dict[str, object] = {"valid": True}
        self.rebuild_result: FakeRebuildResult = FakeRebuildResult(
            diagnostics={"status": "rebuilt"}
        )
        self.rebuild_calls: list[dict[str, object]] = []
        self.live_rows_payloads: list[object] = []
        self.targeted_loads: list[str] = []
        self.targeted_stores: list[tuple[str, object]] = []

    @property
    def deps(self) -> GeometryFitWorkerPrebuildDeps:
        return GeometryFitWorkerPrebuildDeps(
            theta_initial_for_background=self.theta_initial_for_background,
            int_keyed_mapping=self.int_keyed_mapping,
            live_cache_signature_summary=self.live_cache_signature_summary,
            cache_jsonable=self.cache_jsonable,
            digest_payload=self.digest_payload,
            collect_required_manual_fit_targets=(self.collect_required_manual_fit_targets),
            required_branch_group_keys=self.required_branch_group_keys,
            live_row_source_counts=self.live_row_source_counts,
            validate_required_source_rows_for_fit_space=(
                self.validate_required_source_rows_for_fit_space
            ),
            projection_view_signature_for_background=(
                self.projection_view_signature_for_background
            ),
            logged_intersection_cache_loaders=self.logged_intersection_cache_loaders,
            copy_intersection_cache_tables=self.copy_intersection_cache_tables,
            logged_cache_matches_params=self.logged_cache_matches_params,
            forward_source_rows_for_rebuild=self.forward_source_rows_for_rebuild,
            build_source_rows_for_rebuild=self.build_source_rows_for_rebuild,
            simulate_hit_tables_for_fit=self.simulate_hit_tables_for_fit,
            load_targeted_projected_cache_entry=self.load_targeted_projected_cache_entry,
            store_targeted_projected_cache_entry=self.store_targeted_projected_cache_entry,
            rebuild_geometry_fit_source_rows=self.rebuild_geometry_fit_source_rows,
            hydrate_exact_caked_payload=self.hydrate_exact_caked_payload,
            projection_payload_storage_copy=self.projection_payload_storage_copy,
            is_transform_bundle=self.is_transform_bundle,
            live_handoff_patch_marker="phase4d1",
        )

    def theta_initial_for_background(self, background_index: int) -> float:
        return float(background_index) + 10.0

    def int_keyed_mapping(self, raw_mapping: object) -> dict[int, object]:
        normalized: dict[int, object] = {}
        if not isinstance(raw_mapping, Mapping):
            return normalized
        for key, value in raw_mapping.items():
            try:
                normalized[int(key)] = value
            except Exception:
                continue
        return normalized

    def live_cache_signature_summary(self, signature: object) -> dict[str, object]:
        return {"signature": signature}

    def cache_jsonable(self, payload: object) -> object:
        return copy.deepcopy(payload)

    def digest_payload(self, payload: object) -> tuple[str, str]:
        return ("digest", repr(payload))

    def collect_required_manual_fit_targets(
        self,
        required_pairs: object,
        *,
        background_index: int,
    ) -> list[dict[str, object]]:
        return [
            dict(pair, background_index=int(background_index))
            for pair in (required_pairs or ())
            if isinstance(pair, Mapping)
        ]

    def required_branch_group_keys(
        self,
        required_manual_fit_targets: object,
    ) -> list[tuple[tuple[int, int, int], int | None, object | None]]:
        if not required_manual_fit_targets:
            return []
        return [((1, 0, 0), None, None)]

    def live_row_source_counts(self, raw_rows: object) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in raw_rows or ():
            if not isinstance(row, Mapping):
                continue
            source_label = str(row.get("source_label", "primary") or "primary")
            counts[source_label] = int(counts.get(source_label, 0)) + 1
        return dict(sorted(counts.items()))

    def validate_required_source_rows_for_fit_space(
        self,
        _rows: object,
        *,
        required_pairs: object,
        background_index: int,
    ) -> dict[str, object]:
        return dict(
            self.validation_result,
            background_index=int(background_index),
            required_pair_count=len(required_pairs or ()),
        )

    def projection_view_signature_for_background(
        self,
        background_index: int,
    ) -> dict[str, object]:
        return {"background_index": int(background_index), "mode": "detector"}

    def logged_intersection_cache_loaders(self) -> tuple[None, None]:
        return None, None

    def copy_intersection_cache_tables(self, raw_tables: object) -> list[object]:
        return copy.deepcopy(list(raw_tables or ()))

    def logged_cache_matches_params(self, *_args: object, **_kwargs: object) -> bool:
        return True

    def forward_source_rows_for_rebuild(
        self,
        rebuild_callback: object,
        source_tables: object,
        *,
        params_local: Mapping[str, object],
        fallback_consumer: object = "geometry_fit_preflight_cache",
        kwargs: Mapping[str, object] | None = None,
    ) -> object:
        return rebuild_callback(
            source_tables,
            params_local=params_local,
            consumer=str(fallback_consumer or "geometry_fit_preflight_cache"),
            **dict(kwargs or {}),
        )

    def build_source_rows_for_rebuild(
        self,
        _source_tables: object,
        *,
        params_local: Mapping[str, object],
        consumer: str,
        **_kwargs: object,
    ) -> tuple[list[dict[str, object]], None, None, None]:
        return (
            [
                _source_row(
                    background_index=0,
                    source_label=str(consumer),
                    theta=float(params_local.get("theta_initial", 0.0)),
                )
            ],
            None,
            None,
            None,
        )

    def simulate_hit_tables_for_fit(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> list[object]:
        return []

    def load_targeted_projected_cache_entry(
        self,
        *,
        background_index: int,
        key_digest: str,
    ) -> object:
        self.targeted_loads.append(f"{int(background_index)}:{key_digest}")
        return None

    def store_targeted_projected_cache_entry(
        self,
        *,
        background_index: int,
        key_digest: str,
        payload: object,
    ) -> None:
        self.targeted_stores.append((f"{int(background_index)}:{key_digest}", payload))

    def rebuild_geometry_fit_source_rows(self, **kwargs: object) -> FakeRebuildResult:
        self.rebuild_calls.append(dict(kwargs))
        build_live_rows = kwargs.get("build_live_rows")
        if callable(build_live_rows):
            self.live_rows_payloads.append(build_live_rows())
        return self.rebuild_result

    def hydrate_exact_caked_payload(self, payload: object, **_kwargs: object) -> object:
        if not isinstance(payload, Mapping):
            return None
        hydrated = dict(payload)
        hydrated.setdefault("transform_bundle", FakeBundle())
        return hydrated

    def projection_payload_storage_copy(self, payload: object) -> object:
        return dict(payload) if isinstance(payload, Mapping) else payload

    def is_transform_bundle(self, value: object) -> bool:
        return isinstance(value, FakeBundle)


class FakeRequiredCacheDeps:
    def __init__(self) -> None:
        self.readiness: dict[str, object] = {
            "expected_locked_qr_rows": 0,
            "projected_locked_qr_rows": 0,
            "finite_locked_qr_rows": 0,
            "fit_space_projection_ready": True,
        }
        self.caked_outcome: dict[str, object] | None = {
            "caked_view_stored": True,
            "caked_view_status": "stored",
            "status": "stored",
        }
        self.announced_timeouts: int = 0

    @property
    def deps(self) -> GeometryFitWorkerRequiredCacheDeps:
        return GeometryFitWorkerRequiredCacheDeps(
            locked_qr_fit_space_projection_readiness=(
                self.locked_qr_fit_space_projection_readiness
            ),
            normalize_optics_mode_label=self.normalize_optics_mode_label,
        )

    def locked_qr_fit_space_projection_readiness(
        self,
        _projected_rows: object,
        *,
        required_pairs: object,
        source_rows: object,
    ) -> dict[str, object]:
        return dict(
            self.readiness,
            required_pair_count=len(required_pairs or ()),
            source_row_count=len(source_rows or ()),
        )

    def normalize_optics_mode_label(self, value: object) -> str:
        return str(value or "exact").strip().lower() or "exact"


class FakeManualFitSpaceDeps:
    def __init__(self) -> None:
        self.space_map: dict[int, str] = {}
        self.caked_required_result = False
        self.validation_result: dict[str, object] = {"valid": True}
        self.required_targets: list[dict[str, object]] = []
        self.fit_space_calls: list[dict[str, object]] = []
        self.caked_required_calls: list[dict[str, object]] = []
        self.validation_calls: list[dict[str, object]] = []
        self.required_target_calls: list[dict[str, object]] = []

    @property
    def deps(self) -> GeometryFitWorkerManualFitSpaceDeps:
        return GeometryFitWorkerManualFitSpaceDeps(
            geometry_manual_fit_space_by_background=(
                self.geometry_manual_fit_space_by_background
            ),
            geometry_manual_caked_fit_space_required_from_context=(
                self.geometry_manual_caked_fit_space_required_from_context
            ),
            validate_geometry_fit_live_source_rows=(
                self.validate_geometry_fit_live_source_rows
            ),
            collect_required_manual_fit_targets=(
                self.collect_required_manual_fit_targets
            ),
        )

    def geometry_manual_fit_space_by_background(
        self,
        required_indices: object,
        pairs_for_background: object,
        *,
        pick_uses_caked_space: bool,
        current_background_index: int,
    ) -> dict[int, str]:
        self.fit_space_calls.append(
            {
                "required_indices": [int(idx) for idx in (required_indices or ())],
                "pairs": {
                    int(idx): pairs_for_background(int(idx))
                    for idx in (required_indices or ())
                    if callable(pairs_for_background)
                },
                "pick_uses_caked_space": bool(pick_uses_caked_space),
                "current_background_index": int(current_background_index),
            }
        )
        return dict(self.space_map)

    def geometry_manual_caked_fit_space_required_from_context(
        self,
        pairs: object,
        *,
        manual_fit_space_kind: str,
        projection_view_mode: object,
        pick_uses_caked_space: bool,
        pick_applies_to_background: bool,
    ) -> bool:
        self.caked_required_calls.append(
            {
                "pairs": [dict(entry) for entry in (pairs or ())],
                "manual_fit_space_kind": str(manual_fit_space_kind),
                "projection_view_mode": projection_view_mode,
                "pick_uses_caked_space": bool(pick_uses_caked_space),
                "pick_applies_to_background": bool(pick_applies_to_background),
            }
        )
        return bool(self.caked_required_result)

    def validate_geometry_fit_live_source_rows(
        self,
        rows: object,
        *,
        required_pairs: object,
        require_caked_fit_space: bool,
    ) -> dict[str, object]:
        self.validation_calls.append(
            {
                "rows": [dict(row) for row in (rows or ()) if isinstance(row, Mapping)],
                "required_pairs": [
                    dict(pair) for pair in (required_pairs or ()) if isinstance(pair, Mapping)
                ],
                "require_caked_fit_space": bool(require_caked_fit_space),
            }
        )
        return dict(self.validation_result)

    def collect_required_manual_fit_targets(
        self,
        required_pairs: object,
        *,
        background_index: int,
    ) -> list[dict[str, object]]:
        self.required_target_calls.append(
            {
                "required_pairs": [
                    dict(pair) for pair in (required_pairs or ()) if isinstance(pair, Mapping)
                ],
                "background_index": int(background_index),
            }
        )
        return [dict(entry) for entry in self.required_targets]


def _ready_projection_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "radial_axis": [1.0, 2.0],
        "azimuth_axis": [0.0, 1.0],
        "raw_azimuth_axis": [0.0, 1.0],
        "detector_shape": (2, 3),
        "transform_bundle": FakeBundle((2, 3)),
        "digest": "digest",
    }
    payload.update(overrides)
    return payload


def _context_with_caked_deps(
    job: dict[str, object] | None = None,
    fake_deps: FakeCakedPayloadDeps | None = None,
) -> tuple[GeometryFitWorkerContext, FakeCakedPayloadDeps]:
    fake_deps = fake_deps or FakeCakedPayloadDeps()
    context = GeometryFitWorkerContext.from_job(job or {})
    context.caked_payload_deps = fake_deps.deps
    return context, fake_deps


def _context_with_source_projection_deps(
    job: dict[str, object] | None = None,
    fake_deps: FakeSourceProjectionDeps | None = None,
) -> tuple[GeometryFitWorkerContext, FakeSourceProjectionDeps]:
    fake_deps = fake_deps or FakeSourceProjectionDeps()
    context = GeometryFitWorkerContext.from_job(
        {
            "current_background_index": 0,
            "projection_view_mode": "detector",
            "background_images": {
                0: {"native": [[1.0]], "display": [[1.0]]},
                1: {"native": [[2.0]], "display": [[2.0]]},
            },
            "params": {"center": (0.0, 0.0), "pixel_size_m": 1.0e-4},
            "solver_inputs": type(
                "FakeSolverInputs",
                (),
                {"miller": [1], "intensities": [1.0]},
            )(),
            **(job or {}),
        }
    )
    context.source_projection_deps = fake_deps.deps
    return context, fake_deps


def _fake_cache_bundle_deps() -> GeometryFitWorkerCacheBundleDeps:
    return GeometryFitWorkerCacheBundleDeps(
        is_background_cache_bundle=lambda value: isinstance(
            value,
            FakeBackgroundCacheBundle,
        ),
        copy_source_rows=lambda raw_rows: [
            dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)
        ],
        make_background_cache_bundle=FakeBackgroundCacheBundle,
        copy_optional_values=lambda raw_values: (
            None
            if raw_values is None
            else [copy.deepcopy(entry) for entry in raw_values]
        ),
    )


def _context_with_cache_bundle_deps(
    job: dict[str, object] | None = None,
) -> tuple[GeometryFitWorkerContext, FakeSourceProjectionDeps]:
    source_deps = FakeSourceProjectionDeps()
    context, _source_deps = _context_with_source_projection_deps(
        job,
        fake_deps=source_deps,
    )
    context.cache_bundle_deps = _fake_cache_bundle_deps()
    return context, source_deps


def _context_with_caked_view_storage_deps(
    job: dict[str, object] | None = None,
    *,
    caked_payload_deps: FakeCakedPayloadDeps | None = None,
) -> tuple[GeometryFitWorkerContext, FakeCakedViewStorageDeps]:
    context, _source_deps = _context_with_cache_bundle_deps(
        {
            "current_background_index": 0,
            "params": {"center": (0.0, 0.0)},
            **(job or {}),
        }
    )
    caked_payload_deps = caked_payload_deps or FakeCakedPayloadDeps()
    context.caked_payload_deps = caked_payload_deps.deps
    fake_storage_deps = FakeCakedViewStorageDeps()
    context.caked_view_storage_deps = fake_storage_deps.deps
    return context, fake_storage_deps


def _context_with_prebuild_deps(
    job: dict[str, object] | None = None,
) -> tuple[GeometryFitWorkerContext, FakePrebuildDeps]:
    context, _source_deps = _context_with_cache_bundle_deps(job)
    fake_prebuild_deps = FakePrebuildDeps()
    context.prebuild_deps = fake_prebuild_deps.deps
    context.caked_payload_deps = FakeCakedPayloadDeps().deps
    context.source_rows_deps = GeometryFitWorkerSourceRowsDeps(
        manual_caked_fit_space_required_for_background=lambda _idx: False,
        theta_base_for_background=lambda idx: float(idx) + 20.0,
    )
    return context, fake_prebuild_deps


def _context_with_required_cache_deps(
    job: dict[str, object] | None = None,
) -> tuple[GeometryFitWorkerContext, FakeRequiredCacheDeps]:
    context, _prebuild_deps = _context_with_prebuild_deps(
        {
            "required_indices": [0],
            "background_labels": {0: "background 1"},
            "manual_pairs_by_background": {0: [{"pair_id": "pair-1"}]},
            "source_cache_generation_by_background": {0: 4},
            "params": {"optics_mode": "exact"},
            **(job or {}),
        }
    )
    fake_required_deps = FakeRequiredCacheDeps()
    context.required_cache_deps = fake_required_deps.deps
    bundle = FakeBackgroundCacheBundle(
        background_index=0,
        background_label="background 1",
        stored_rows=[_source_row(background_index=0)],
        projected_rows=[_source_row(background_index=0, projected=True)],
        cache_source="prebuilt",
    )

    def _prebuild_background_cache_bundle_worker(
        background_index: int,
        **_kwargs: object,
    ) -> FakeBackgroundCacheBundle:
        bundle.background_index = int(background_index)
        return bundle

    context.prebuild_background_cache_bundle_worker = (  # type: ignore[method-assign]
        _prebuild_background_cache_bundle_worker
    )

    def _store_worker_caked_view_for_background(
        _bundle: object,
        *,
        stage_callback: object = None,
        source_cache_generation_id: object = None,
    ) -> dict[str, object] | None:
        if fake_required_deps.caked_outcome is None:
            return None
        return dict(
            fake_required_deps.caked_outcome,
            background_index=int(bundle.background_index),
            source_cache_generation_id=int(source_cache_generation_id or 0),
            stage_callback=stage_callback,
        )

    context.store_worker_caked_view_for_background = (  # type: ignore[method-assign]
        _store_worker_caked_view_for_background
    )
    return context, fake_required_deps


def _context_with_manual_fit_space_deps(
    job: dict[str, object] | None = None,
) -> tuple[GeometryFitWorkerContext, FakeManualFitSpaceDeps]:
    context = GeometryFitWorkerContext.from_job(
        {
            "required_indices": [0, 1],
            "current_background_index": 0,
            "projection_view_mode": "detector",
            "manual_pairs_by_background": {
                0: [{"pair_id": "pair-0", "background_index": 0}],
                1: [{"pair_id": "pair-1", "background_index": 1}],
            },
            **(job or {}),
        }
    )
    fake_deps = FakeManualFitSpaceDeps()
    context.manual_fit_space_deps = fake_deps.deps
    return context, fake_deps


def _context_with_dataset_deps(
    job: dict[str, object] | None = None,
) -> tuple[GeometryFitWorkerContext, FakeDatasetDeps]:
    context = GeometryFitWorkerContext.from_job(
        {
            "params": {"center": (1.0, 2.0), "theta_initial": 10.0},
            "var_names": ["center_x", "theta_initial"],
            "fit_config": {"geometry": {"auto_match": {"max_display_markers": 8}}},
            "geometry_runtime_cfg": {"runtime": "cfg"},
            "osc_files": ["bg0.osc", "bg1.osc"],
            "current_background_index": 1,
            "image_size": 64,
            "display_rotate_k": 2,
            "manual_pairs_by_background": {
                1: [{"pair_id": "pair-1", "background_index": 1}]
            },
            "selected_background_indices": [1],
            "theta_initial": 12.5,
            "preserve_live_theta": True,
            "selection_applied": True,
            "uses_shared_theta": False,
            "theta_metadata_applied": True,
            "background_theta_values": [10.0, 12.5],
            "theta_offset": 1.25,
            "pick_uses_caked_space": True,
            "manual_caked_fit_space_required_by_background": {1: True},
            "apply_background_backend_orientation": lambda image: image,
            "geometry_manual_simulated_lookup": lambda entry: {"lookup": entry},
            "unrotate_display_peaks": lambda peaks: list(peaks or ()),
            "display_to_native_sim_coords": lambda col, row: (col, row),
            "native_detector_coords_to_detector_display_coords": (
                lambda col, row: (col, row)
            ),
            "select_fit_orientation": lambda *_args, **_kwargs: ({}, {}),
            "apply_orientation_to_entries": (
                lambda entries, *_args, **_kwargs: list(entries or ())
            ),
            "orient_image_for_fit": lambda image, *_args, **_kwargs: image,
            **(job or {}),
        }
    )
    fake_deps = FakeDatasetDeps()
    context.dataset_deps = fake_deps.deps
    return context, fake_deps


def _source_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "background_index": 0,
        "source_label": "primary",
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "normalized_hkl": (-1, 0, 10),
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 42,
        "locked_qr": 1.25,
        "locked_qz": 2.5,
        "sim_col": 11.0,
        "sim_row": 12.0,
    }
    row.update(overrides)
    return row


def _event_payloads(queue: RecordingQueue, kind: str) -> list[dict[str, object]]:
    return [
        dict(item.get("payload") or {})
        for item in queue.items
        if isinstance(item, Mapping) and item.get("kind") == kind
    ]


def test_emit_source_cache_caked_view_event_ready_payload_fields() -> None:
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job({"event_queue": queue, "job_id": 12})
    bundle = FakeBackgroundCacheBundle(
        background_index=2,
        background_label="sample bg",
        stored_rows=[_source_row(background_index=2)],
        projected_rows=[_source_row(background_index=2, projected=True)],
    )

    context.emit_source_cache_caked_view_event(
        "source_cache_caked_view_ready",
        bundle,
        source_cache_generation_id=9,
        started_at=0.0,
        payload={"caked_view_status": "stored"},
    )

    payload = _event_payloads(queue, "source_cache_caked_view_ready")[0]
    assert payload["background_index"] == 2
    assert payload["background_label"] == "sample bg"
    assert payload["source_cache_generation_id"] == 9
    assert payload["row_count"] == 1
    assert payload["caked_view_status"] == "stored"
    assert payload["elapsed_s"] >= 0.0
    assert payload["message"] == (
        "preflight: caked view storage ready for background 3 (status=stored)"
    )


def test_emit_source_cache_caked_view_event_timeout_payload_fields() -> None:
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job({"event_queue": queue})
    bundle = FakeBackgroundCacheBundle(
        background_index=0,
        background_label="background 1",
        stored_rows=[_source_row(background_index=0)],
        projected_rows=[],
    )

    context.emit_source_cache_caked_view_event(
        "source_cache_caked_view_timeout",
        bundle,
        source_cache_generation_id=4,
        started_at=0.0,
        payload={"status": "custom"},
    )

    payload = _event_payloads(queue, "source_cache_caked_view_timeout")[0]
    assert payload["background_index"] == 0
    assert payload["background_label"] == "background 1"
    assert payload["source_cache_generation_id"] == 4
    assert payload["row_count"] == 1
    assert payload["status"] == "custom"
    assert payload["message"] == (
        "preflight: caked view storage timeout for background 1 (waited 5.0s)"
    )


def test_emit_source_cache_caked_view_event_failed_payload_fields() -> None:
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job({"event_queue": queue})
    bundle = FakeBackgroundCacheBundle(
        background_index=1,
        background_label="background 2",
        stored_rows=[],
        projected_rows=[_source_row(background_index=1, projected=True)],
    )

    context.emit_source_cache_caked_view_event(
        "source_cache_caked_view_failed",
        bundle,
        source_cache_generation_id=5,
        started_at=0.0,
        payload={"status": "missing_projection"},
    )

    payload = _event_payloads(queue, "source_cache_caked_view_failed")[0]
    assert payload["background_index"] == 1
    assert payload["background_label"] == "background 2"
    assert payload["source_cache_generation_id"] == 5
    assert payload["row_count"] == 1
    assert payload["status"] == "missing_projection"
    assert payload["message"] == (
        "preflight: caked view storage failed for background 2 "
        "(status=missing_projection)"
    )


def test_emit_source_cache_caked_view_event_late_flag() -> None:
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job({"event_queue": queue})
    bundle = FakeBackgroundCacheBundle(background_index=0)

    context.emit_source_cache_caked_view_event(
        "source_cache_caked_view_ready",
        bundle,
        source_cache_generation_id=4,
        started_at=0.0,
        payload={"caked_view_stored": True},
        late=True,
    )

    payload = _event_payloads(queue, "source_cache_caked_view_ready")[0]
    assert payload["late"] is True


def test_start_non_gating_caked_view_task_returns_initial_ready_outcome() -> None:
    context = GeometryFitWorkerContext.from_job({"job_id": 3})
    bundle = FakeBackgroundCacheBundle(
        background_index=0,
        background_label="background 1",
        stored_rows=[_source_row(background_index=0)],
    )
    stage_callbacks: list[object] = []

    def _store(
        _bundle: object,
        *,
        stage_callback: object = None,
        source_cache_generation_id: object = None,
    ) -> dict[str, object]:
        stage_callbacks.append(stage_callback)
        return {
            "caked_view_stored": True,
            "caked_view_status": "stored",
            "status": "stored",
            "source_cache_generation_id": source_cache_generation_id,
        }

    context.store_worker_caked_view_for_background = _store  # type: ignore[method-assign]

    await_result, _announce_timeout = context.start_non_gating_caked_view_task(
        bundle,
        source_cache_generation_id=4,
        started_at=0.0,
        stage_callback="stage-callback",
    )

    outcome = await_result(1.0)
    assert isinstance(outcome, Mapping)
    assert outcome["caked_view_stored"] is True
    assert outcome["caked_view_status"] == "stored"
    assert outcome["source_cache_generation_id"] == 4
    assert stage_callbacks == ["stage-callback"]


def _install_blocking_caked_store(
    context: GeometryFitWorkerContext,
    *,
    release_storage: threading.Event,
    storage_finished: threading.Event,
) -> None:
    def _store(
        _bundle: object,
        *,
        stage_callback: object = None,
        source_cache_generation_id: object = None,
    ) -> dict[str, object]:
        release_storage.wait(1.0)
        storage_finished.set()
        return {
            "caked_view_stored": True,
            "caked_view_status": "stored",
            "status": "stored",
            "source_cache_generation_id": source_cache_generation_id,
        }

    context.store_worker_caked_view_for_background = _store  # type: ignore[method-assign]


def test_start_non_gating_caked_view_task_timeout_emits_late_ready_when_generation_matches() -> None:
    queue = RecordingQueue()
    release_storage = threading.Event()
    storage_finished = threading.Event()
    context = GeometryFitWorkerContext.from_job(
        {
            "event_queue": queue,
            "job_id": 3,
            "source_cache_generation_by_background": {0: 4},
        }
    )
    bundle = FakeBackgroundCacheBundle(
        background_index=0,
        background_label="background 1",
        stored_rows=[_source_row(background_index=0)],
    )
    _install_blocking_caked_store(
        context,
        release_storage=release_storage,
        storage_finished=storage_finished,
    )

    await_result, announce_timeout = context.start_non_gating_caked_view_task(
        bundle,
        source_cache_generation_id=4,
        started_at=0.0,
        stage_callback="stage-callback",
    )

    assert await_result(0.01) is None
    announce_timeout()
    release_storage.set()
    assert storage_finished.wait(1.0)
    for _ in range(100):
        if _event_payloads(queue, "source_cache_caked_view_ready"):
            break
        threading.Event().wait(0.01)
    payload = _event_payloads(queue, "source_cache_caked_view_ready")[0]
    assert payload["background_index"] == 0
    assert payload["source_cache_generation_id"] == 4
    assert payload["late"] is True
    assert payload["message"] == (
        "preflight: caked view storage ready for background 1 (status=stored)"
    )


def test_start_non_gating_caked_view_task_suppresses_late_event_when_generation_stale() -> None:
    queue = RecordingQueue()
    release_storage = threading.Event()
    storage_finished = threading.Event()
    context = GeometryFitWorkerContext.from_job(
        {
            "event_queue": queue,
            "job_id": 3,
            "source_cache_generation_by_background": {0: 4},
        }
    )
    bundle = FakeBackgroundCacheBundle(
        background_index=0,
        background_label="background 1",
        stored_rows=[_source_row(background_index=0)],
    )
    _install_blocking_caked_store(
        context,
        release_storage=release_storage,
        storage_finished=storage_finished,
    )

    await_result, announce_timeout = context.start_non_gating_caked_view_task(
        bundle,
        source_cache_generation_id=4,
        started_at=0.0,
        stage_callback="stage-callback",
    )

    assert await_result(0.01) is None
    context.advance_source_cache_generation(0)
    announce_timeout()
    release_storage.set()
    assert storage_finished.wait(1.0)
    threading.Event().wait(0.05)
    assert _event_payloads(queue, "source_cache_caked_view_ready") == []


def test_start_non_gating_caked_view_task_maps_storage_exception_to_failed_outcome() -> None:
    context = GeometryFitWorkerContext.from_job({"job_id": 3})
    bundle = FakeBackgroundCacheBundle(
        background_index=1,
        background_label="background 2",
        stored_rows=[_source_row(background_index=1)],
    )

    def _store(
        _bundle: object,
        *,
        stage_callback: object = None,
        source_cache_generation_id: object = None,
    ) -> dict[str, object]:
        raise RuntimeError("storage failed")

    context.store_worker_caked_view_for_background = _store  # type: ignore[method-assign]

    await_result, _announce_timeout = context.start_non_gating_caked_view_task(
        bundle,
        source_cache_generation_id=8,
        started_at=0.0,
        stage_callback="stage-callback",
    )

    outcome = await_result(1.0)
    assert isinstance(outcome, Mapping)
    assert outcome["caked_view_stored"] is False
    assert outcome["caked_view_status"] == "exception:RuntimeError"
    assert outcome["status"] == "exception:RuntimeError"
    assert outcome["background_index"] == 1
    assert outcome["background_label"] == "background 2"
    assert outcome["source_cache_generation_id"] == 8
    assert outcome["row_count"] == 1


# D2 contract map: caked payload helpers return ready/invalid/absent status
# strings, mutate only job-local caked/projection payload maps, short-circuit
# invalid existing payloads before generated fallback, and emit stage diagnostics
# with the same status/kind/message fields as the old nested helpers.
#
# D3.1 contract map: source-row projection helpers normalize source rows for a
# background, project them through runtime manual projection callbacks, preserve
# source/reflection/locked-Q identity fields, return [] for empty rows, group
# mixed-background rows with current background as the fallback, and restore the
# original cross-background row ordering after projection.
#
# D3.3a contract map: background cache bundle construction copies stored and
# projected rows for one background, projects missing rows in the current
# detector/caked/q-space mode, records projection failure diagnostics without
# raising for bundle construction, fills default cache diagnostics, and copies
# optional table/cache metadata into the bundle constructor.
#
# D3.3c contract map: source-row cache lookup returns job-local background cache
# rows when signatures match, reports exact source snapshot diagnostics for
# missing/mismatched/empty caches and validation failures, leaves source-cache
# generation unchanged on lookup, and delegates rebuilds through the existing
# prebuild helper with the same background, params, signature, projection mode,
# required-pair, and prior-diagnostic arguments.
#
# D3.3d contract map: required-cache prebuild iterates job required_indices in
# order, emits source-cache start/bundle-start/ready/failure/caked-view events
# with stable payload keys, preserves current source-cache generation ids,
# derives locked-Qr readiness from projected/stored rows, treats timed-out caked
# storage as fatal only when exact caked fit-space or locked-Qr readiness needs
# it, and does not move manual validation, dataset, solver, or result packaging.
#
# E1 contract map: worker manual fit-space helpers read only job-local manual
# pair/space maps and injected manual-fit callbacks, copy manual-pair dicts,
# preserve required-index order, honor explicit caked-required overrides before
# fallback classification, add manual caked validation diagnostics, and reject
# unsupported mixed detector/caked backgrounds with one-based labels.
#
# E2 contract map: caked-view ensure first rejects unsupported mixed manual
# spaces, then checks only required backgrounds that need caked fit-space. Caked
# readiness derives detector shape from the job-local native background image,
# loads projection payloads with generated fallback enabled, hydrates exact
# caked payloads without requiring background image data, and raises the existing
# one-based exact-projector RuntimeError when a required payload is unavailable.
#
# E6 contract map: worker solver-phase input assembly mutates only the prepared
# run fit-params mosaic_params copy, forwards job-local solver inputs/var names/
# solve_fit/stamp/log_path into the existing solver phase call, emits worker
# events through the established event shape, gates live-cache update events from
# enable_live_update_events, and does not execute the solver or package results.


def test_worker_context_from_job_copies_source_snapshots() -> None:
    source_snapshots = {0: {"rows": [{"source": "cached"}]}}

    context = GeometryFitWorkerContext.from_job({"source_snapshots": source_snapshots})

    source_snapshots[0]["rows"][0]["source"] = "mutated"
    assert context.worker_source_row_snapshots[0]["rows"][0]["source"] == "cached"


def test_worker_prepare_geometry_fit_run_passes_job_snapshots() -> None:
    context, fake_deps = _context_with_dataset_deps()

    result = context.prepare_geometry_fit_run_for_worker(
        ensure_geometry_fit_caked_view=lambda: None,
        stage_callback=lambda _stage, _payload: None,
    )

    assert result is fake_deps.prepare_result
    prepare_call = fake_deps.prepare_calls[0]
    assert prepare_call["params"] == {"center": (1.0, 2.0), "theta_initial": 10.0}
    assert prepare_call["var_names"] == ["center_x", "theta_initial"]
    assert prepare_call["fit_config"] == {
        "geometry": {"auto_match": {"max_display_markers": 8}}
    }
    assert prepare_call["osc_files"] == ["bg0.osc", "bg1.osc"]
    assert prepare_call["current_background_index"] == 1
    assert prepare_call["theta_initial"] == 12.5
    assert prepare_call["preserve_live_theta"] is True
    assert prepare_call["manual_fit_pick_uses_caked_space"] is True
    assert prepare_call["manual_fit_requires_caked_space"] is True


def test_worker_prepare_geometry_fit_run_uses_manual_dataset_bindings_callbacks() -> None:
    context, fake_deps = _context_with_dataset_deps()

    context.prepare_geometry_fit_run_for_worker(
        ensure_geometry_fit_caked_view=lambda: None,
        stage_callback=lambda _stage, _payload: None,
    )

    binding_call = fake_deps.binding_calls[0]
    assert binding_call["osc_files"] == ["bg0.osc", "bg1.osc"]
    assert binding_call["current_background_index"] == 1
    assert binding_call["image_size"] == 64
    assert binding_call["display_rotate_k"] == 2
    assert (
        binding_call["geometry_manual_source_rows_for_background"]
        == context.source_rows_for_background_worker
    )
    assert (
        binding_call["geometry_manual_rebuild_source_rows_for_background"]
        == context.rebuild_source_rows_for_background_worker
    )
    pairs = binding_call["geometry_manual_pairs_for_index"](1)
    pairs[0]["pair_id"] = "mutated"
    assert context.job_data["manual_pairs_by_background"][1][0]["pair_id"] == "pair-1"


def test_worker_prepare_geometry_fit_run_build_dataset_uses_manual_builder() -> None:
    context, fake_deps = _context_with_dataset_deps()

    context.prepare_geometry_fit_run_for_worker(
        ensure_geometry_fit_caked_view=lambda: None,
        stage_callback=lambda _stage, _payload: None,
    )
    prepare_call = fake_deps.prepare_calls[0]
    dataset = prepare_call["build_dataset"](
        1,
        theta_base=20.0,
        base_fit_params={"center": (1.0, 2.0)},
        orientation_cfg={"mode": "detector"},
        manual_fit_requires_caked_space=True,
        stage_callback="stage-callback",
    )

    assert dataset == {"dataset_background_index": 1}
    dataset_call = fake_deps.dataset_calls[0]
    assert dataset_call["background_index"] == 1
    assert dataset_call["kwargs"] == {
        "theta_base": 20.0,
        "base_fit_params": {"center": (1.0, 2.0)},
        "manual_dataset_bindings": fake_deps.bindings[0],
        "orientation_cfg": {"mode": "detector"},
        "manual_fit_requires_caked_space": True,
        "stage_callback": "stage-callback",
    }


def test_worker_prepare_geometry_fit_run_preserves_callbacks_and_error_lambdas() -> None:
    context, fake_deps = _context_with_dataset_deps({"selection_error": "bad selection"})
    stage_callback = lambda _stage, _payload: None
    ensure_calls: list[str] = []

    context.prepare_geometry_fit_run_for_worker(
        ensure_geometry_fit_caked_view=lambda: ensure_calls.append("ensure"),
        stage_callback=stage_callback,
    )

    prepare_call = fake_deps.prepare_calls[0]
    assert prepare_call["stage_callback"] is stage_callback
    prepare_call["ensure_geometry_fit_caked_view"]()
    assert ensure_calls == ["ensure"]
    try:
        prepare_call["current_geometry_fit_background_indices"](strict=True)
    except RuntimeError as exc:
        assert str(exc) == "bad selection"
    else:
        raise AssertionError("expected selection error")


def test_worker_solver_phase_kwargs_passes_job_solver_inputs() -> None:
    solve_fit = object()
    solver_inputs = object()
    prepared_run = SimpleNamespace(fit_params={})
    context = GeometryFitWorkerContext.from_job(
        {
            "var_names": ("center_x", "theta_initial"),
            "solve_fit": solve_fit,
            "solver_inputs": solver_inputs,
            "stamp": 20260420,
            "log_path": "fit.log",
        }
    )

    kwargs = context.build_solver_phase_kwargs_for_worker(prepared_run)

    assert kwargs["prepared_run"] is prepared_run
    assert kwargs["var_names"] == ["center_x", "theta_initial"]
    assert kwargs["solve_fit"] is solve_fit
    assert kwargs["solver_inputs"] is solver_inputs
    assert kwargs["stamp"] == "20260420"
    assert kwargs["log_path"] == "fit.log"


def test_worker_solver_phase_kwargs_applies_mosaic_params_copy() -> None:
    mosaic_params = {"profile": {"sigma": 0.25}}
    prepared_run = SimpleNamespace(fit_params={"existing": True})
    context = GeometryFitWorkerContext.from_job(
        {"fit_solver_mosaic_params": mosaic_params}
    )

    context.build_solver_phase_kwargs_for_worker(prepared_run)

    assert prepared_run.fit_params["mosaic_params"] == {"profile": {"sigma": 0.25}}
    assert prepared_run.fit_params["mosaic_params"] is not mosaic_params
    mosaic_params["profile"]["sigma"] = 0.5
    assert prepared_run.fit_params["mosaic_params"]["profile"]["sigma"] == 0.25
    assert prepared_run.fit_params["existing"] is True


def test_worker_solver_phase_kwargs_event_callback_emits_worker_event() -> None:
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job(
        {"job_id": "7", "event_queue": queue}
    )
    kwargs = context.build_solver_phase_kwargs_for_worker(
        SimpleNamespace(fit_params={})
    )
    payload = {"message": {"text": "running"}}

    kwargs["event_callback"]("cmd_line", payload)
    payload["message"]["text"] = "mutated"

    assert queue.items == [
        {
            "job_id": 7,
            "kind": "cmd_line",
            "payload": {"message": {"text": "running"}},
        }
    ]


def test_worker_solver_phase_kwargs_live_update_callback_emits_live_cache_update() -> None:
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job(
        {"job_id": 8, "event_queue": queue, "enable_live_update_events": True}
    )
    kwargs = context.build_solver_phase_kwargs_for_worker(
        SimpleNamespace(fit_params={})
    )
    payload = {"background_index": 1, "rows": [{"row_id": "r1"}]}

    kwargs["live_update_callback"](payload)
    payload["rows"][0]["row_id"] = "mutated"

    assert queue.items == [
        {
            "job_id": 8,
            "kind": "live_cache_update",
            "payload": {"background_index": 1, "rows": [{"row_id": "r1"}]},
        }
    ]


def test_worker_solver_phase_kwargs_disables_live_update_callback_when_requested() -> None:
    context = GeometryFitWorkerContext.from_job({"enable_live_update_events": False})

    kwargs = context.build_solver_phase_kwargs_for_worker(
        SimpleNamespace(fit_params={})
    )

    assert kwargs["live_update_callback"] is None


def test_worker_execute_solver_phase_calls_injected_executor_with_kwargs() -> None:
    fake_deps = FakeSolverDeps()
    prepared_run = SimpleNamespace(fit_params={})
    solver_inputs = object()
    context = GeometryFitWorkerContext.from_job(
        {
            "var_names": ("gamma", "Gamma"),
            "solver_inputs": solver_inputs,
            "solve_fit": "solve-fit",
            "stamp": "run-1",
            "log_path": "run.log",
        }
    )
    context.solver_deps = fake_deps.deps

    context.execute_solver_phase_for_worker(prepared_run)

    call = fake_deps.calls[0]
    assert call["prepared_run"] is prepared_run
    assert call["var_names"] == ["gamma", "Gamma"]
    assert call["solver_inputs"] is solver_inputs
    assert call["solve_fit"] == "solve-fit"
    assert call["stamp"] == "run-1"
    assert call["log_path"] == "run.log"


def test_worker_execute_solver_phase_returns_executor_result() -> None:
    fake_deps = FakeSolverDeps()
    context = GeometryFitWorkerContext.from_job({})
    context.solver_deps = fake_deps.deps

    result = context.execute_solver_phase_for_worker(SimpleNamespace(fit_params={}))

    assert result is fake_deps.result


def test_worker_execute_solver_phase_preserves_event_callback_shape() -> None:
    fake_deps = FakeSolverDeps()
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job({"job_id": 12, "event_queue": queue})
    context.solver_deps = fake_deps.deps

    context.execute_solver_phase_for_worker(SimpleNamespace(fit_params={}))
    payload = {"message": {"text": "running"}}
    fake_deps.calls[0]["event_callback"]("progress_text", payload)
    payload["message"]["text"] = "mutated"

    assert queue.items == [
        {
            "job_id": 12,
            "kind": "progress_text",
            "payload": {"message": {"text": "running"}},
        }
    ]


def test_worker_execute_solver_phase_preserves_live_update_callback_shape() -> None:
    fake_deps = FakeSolverDeps()
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job(
        {"job_id": 13, "event_queue": queue, "enable_live_update_events": True}
    )
    context.solver_deps = fake_deps.deps

    context.execute_solver_phase_for_worker(SimpleNamespace(fit_params={}))
    payload = {"background_index": 2, "rows": [{"row_id": "r1"}]}
    fake_deps.calls[0]["live_update_callback"](payload)
    payload["rows"][0]["row_id"] = "mutated"

    assert queue.items == [
        {
            "job_id": 13,
            "kind": "live_cache_update",
            "payload": {"background_index": 2, "rows": [{"row_id": "r1"}]},
        }
    ]


def test_worker_execute_solver_phase_does_not_package_action_result() -> None:
    fake_deps = FakeSolverDeps()
    fake_deps.result = {"execution_result": True}
    context = GeometryFitWorkerContext.from_job({})
    context.solver_deps = fake_deps.deps

    result = context.execute_solver_phase_for_worker(SimpleNamespace(fit_params={}))

    assert result == {"execution_result": True}


def test_worker_action_result_for_worker_passes_job_and_prepare_result() -> None:
    fake_deps = FakeResultDeps()
    prepare_result = object()
    context = GeometryFitWorkerContext.from_job({"job_id": 5, "params": {"a": 1.0}})
    context.result_deps = fake_deps.deps

    context.build_action_result_for_worker(prepare_result=prepare_result)

    call = fake_deps.calls[0]
    assert call["job"] is context.job_data
    assert call["kwargs"] == {
        "prepare_result": prepare_result,
        "execution_result": None,
        "error_text": None,
    }


def test_worker_action_result_for_worker_passes_execution_result_and_error_text() -> None:
    fake_deps = FakeResultDeps()
    execution_result = object()
    context = GeometryFitWorkerContext.from_job({"job_id": 6})
    context.result_deps = fake_deps.deps

    context.build_action_result_for_worker(
        execution_result=execution_result,
        error_text="solver failed",
    )

    call = fake_deps.calls[0]
    assert call["kwargs"] == {
        "prepare_result": None,
        "execution_result": execution_result,
        "error_text": "solver failed",
    }


def test_worker_action_result_for_worker_returns_builder_result() -> None:
    fake_deps = FakeResultDeps()
    fake_deps.result = {"worker_action_result": "ready"}
    context = GeometryFitWorkerContext.from_job({})
    context.result_deps = fake_deps.deps

    result = context.build_action_result_for_worker()

    assert result == {"worker_action_result": "ready"}


def test_worker_action_result_for_worker_does_not_clear_cache_or_touch_events() -> None:
    fake_deps = FakeResultDeps()
    queue = RecordingQueue()
    context = GeometryFitWorkerContext.from_job({"job_id": 7, "event_queue": queue})
    context.result_deps = fake_deps.deps
    context.worker_background_cache_by_index[0] = {"cached": True}

    context.build_action_result_for_worker(error_text="preflight failed")

    assert context.worker_background_cache_by_index == {0: {"cached": True}}
    assert queue.items == []


def test_worker_manual_pairs_for_background_returns_copied_pairs() -> None:
    pairs_by_background = {0: [{"pair_id": "pair-0", "value": {"nested": True}}]}
    context, _deps = _context_with_manual_fit_space_deps(
        {"manual_pairs_by_background": pairs_by_background}
    )

    pairs = context.worker_manual_pairs_for_background(0)
    pairs[0]["pair_id"] = "mutated"

    assert pairs == [{"pair_id": "mutated", "value": {"nested": True}}]
    assert pairs_by_background[0][0]["pair_id"] == "pair-0"


def test_worker_manual_pairs_for_background_missing_map_returns_empty() -> None:
    context, _deps = _context_with_manual_fit_space_deps(
        {"manual_pairs_by_background": None}
    )

    assert context.worker_manual_pairs_for_background(0) == []


def test_worker_manual_fit_space_by_background_uses_complete_stored_map() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0, 1],
            "manual_fit_space_by_background": {0: "caked", "1": "mixed"},
            "pick_uses_caked_space": True,
        }
    )

    assert context.worker_manual_fit_space_by_background() == {0: "caked", 1: "mixed"}
    assert fake_deps.fit_space_calls == []


def test_worker_manual_fit_space_by_background_defaults_missing_stored_entries_to_detector() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0, 1],
            "manual_fit_space_by_background": {0: "caked"},
            "pick_uses_caked_space": True,
            "current_background_index": 1,
        }
    )
    fake_deps.space_map = {0: "detector", 1: "caked"}

    assert context.worker_manual_fit_space_by_background() == {0: "caked", 1: "detector"}
    assert fake_deps.fit_space_calls == []


def test_worker_manual_fit_space_by_background_falls_back_without_stored_map() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0, 1],
            "manual_fit_space_by_background": None,
            "pick_uses_caked_space": True,
            "current_background_index": 1,
        }
    )
    fake_deps.space_map = {0: "detector", 1: "caked"}

    assert context.worker_manual_fit_space_by_background() == {0: "detector", 1: "caked"}
    assert fake_deps.fit_space_calls == [
        {
            "required_indices": [0, 1],
            "pairs": {
                0: [{"pair_id": "pair-0", "background_index": 0}],
                1: [{"pair_id": "pair-1", "background_index": 1}],
            },
            "pick_uses_caked_space": True,
            "current_background_index": 1,
        }
    ]


def test_worker_manual_fit_space_by_background_normalizes_unknown_to_detector() -> None:
    context, _deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0, 1],
            "manual_fit_space_by_background": {0: "unknown", 1: ""},
        }
    )

    assert context.worker_manual_fit_space_by_background() == {0: "detector", 1: "detector"}


def test_worker_manual_caked_fit_space_required_uses_explicit_override() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {"manual_caked_fit_space_required_by_background": {"1": True}}
    )

    assert context.worker_manual_caked_fit_space_required_for_background(1) is True
    assert fake_deps.caked_required_calls == []


def test_worker_manual_caked_fit_space_required_true_for_caked_space() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0],
            "manual_fit_space_by_background": {0: "caked"},
        }
    )

    assert context.worker_manual_caked_fit_space_required_for_background(0) is True
    assert fake_deps.caked_required_calls == []


def test_worker_manual_caked_fit_space_required_calls_context_fallback() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0],
            "manual_fit_space_by_background": {0: "detector"},
            "projection_view_mode": "caked",
            "pick_uses_caked_space": True,
            "current_background_index": 0,
        }
    )
    fake_deps.caked_required_result = True

    assert context.worker_manual_caked_fit_space_required_for_background(0) is True
    assert fake_deps.caked_required_calls == [
        {
            "pairs": [{"pair_id": "pair-0", "background_index": 0}],
            "manual_fit_space_kind": "detector",
            "projection_view_mode": "caked",
            "pick_uses_caked_space": True,
            "pick_applies_to_background": True,
        }
    ]


def test_worker_manual_caked_fit_space_required_computes_pick_applies_to_background() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [1],
            "manual_fit_space_by_background": {1: "detector"},
            "current_background_index": 0,
        }
    )

    assert context.worker_manual_caked_fit_space_required_for_background(1) is False
    assert fake_deps.caked_required_calls[0]["pick_applies_to_background"] is False


def test_worker_validate_required_source_rows_adds_caked_required_flag() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0],
            "manual_fit_space_by_background": {0: "detector"},
        }
    )

    validation = context.worker_validate_required_source_rows_for_fit_space(
        [_source_row()],
        required_pairs=[{"pair_id": "pair-0"}],
        background_index=0,
    )

    assert validation["manual_caked_fit_space_required"] is False
    assert fake_deps.validation_calls[0]["require_caked_fit_space"] is False


def test_worker_validate_required_source_rows_adds_required_pair_count_when_caked() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0],
            "manual_fit_space_by_background": {0: "caked"},
        }
    )
    fake_deps.validation_result = {"valid": False}
    fake_deps.required_targets = [{"pair_id": "target-0"}, {"pair_id": "target-1"}]

    validation = context.worker_validate_required_source_rows_for_fit_space(
        [_source_row()],
        required_pairs=[{"pair_id": "pair-0"}],
        background_index=0,
    )

    assert validation["manual_caked_fit_space_required"] is True
    assert validation["manual_caked_required_pair_count"] == 2
    assert fake_deps.validation_calls[0]["require_caked_fit_space"] is True
    assert fake_deps.required_target_calls[0]["background_index"] == 0


def test_worker_validate_required_source_rows_preserves_validator_payload() -> None:
    context, fake_deps = _context_with_manual_fit_space_deps(
        {
            "required_indices": [0],
            "manual_fit_space_by_background": {0: "detector"},
        }
    )
    fake_deps.validation_result = {"valid": False, "required_pair_count": 5}

    validation = context.worker_validate_required_source_rows_for_fit_space(
        [],
        required_pairs=[],
        background_index=0,
    )

    assert validation["valid"] is False
    assert validation["required_pair_count"] == 5
    assert validation["manual_caked_fit_space_required"] is False


def test_reject_worker_mixed_manual_fit_spaces_noops_without_mixed() -> None:
    context, _deps = _context_with_manual_fit_space_deps({})

    context.reject_worker_mixed_manual_fit_spaces({0: "detector", 1: "caked"})


def test_reject_worker_mixed_manual_fit_spaces_allows_mixed_when_caked_required() -> None:
    context, _deps = _context_with_manual_fit_space_deps(
        {"manual_caked_fit_space_required_by_background": {1: True}}
    )

    context.reject_worker_mixed_manual_fit_spaces({1: "mixed"})


def test_reject_worker_mixed_manual_fit_spaces_raises_with_one_based_background_labels() -> None:
    context, _deps = _context_with_manual_fit_space_deps({})

    try:
        context.reject_worker_mixed_manual_fit_spaces({1: "mixed", 3: "mixed"})
    except RuntimeError as exc:
        assert str(exc) == (
            "mixed detector/caked manual fit spaces are not supported "
            "for background(s) 2, 4; rebuild manual pairs in one fit space"
        )
    else:
        raise AssertionError("expected mixed manual fit-space rejection")


def test_worker_context_from_job_copies_diagnostics() -> None:
    source_diagnostics = {"rows": ["source"]}
    simulation_diagnostics = {"rows": ["simulation"]}

    context = GeometryFitWorkerContext.from_job(
        {
            "source_snapshot_diagnostics": source_diagnostics,
            "simulation_diagnostics": simulation_diagnostics,
        }
    )

    source_diagnostics["rows"].append("mutated")
    simulation_diagnostics["rows"].append("mutated")
    assert context.worker_source_snapshot_diagnostics == {"rows": ["source"]}
    assert context.worker_simulation_diagnostics == {"rows": ["simulation"]}


def test_worker_context_defaults_job_id_to_minus_one() -> None:
    context = GeometryFitWorkerContext.from_job({})

    assert context.job_id == -1


def test_worker_context_from_job_writes_normalized_generation_map_to_job_data() -> None:
    context = GeometryFitWorkerContext.from_job(
        {"source_cache_generation_by_background": {"1": 2}}
    )

    assert context.job_data["source_cache_generation_by_background"] == {"1": 2}


def test_worker_context_emit_event_deep_copies_payload() -> None:
    queue = RecordingQueue()
    payload = {"rows": [{"value": 1}]}
    context = GeometryFitWorkerContext.from_job({"job_id": "7", "event_queue": queue})

    context.emit_event("source_cache_ready", payload)
    payload["rows"][0]["value"] = 99

    assert queue.items == [
        {
            "job_id": 7,
            "kind": "source_cache_ready",
            "payload": {"rows": [{"value": 1}]},
        }
    ]


def test_worker_context_emit_event_swallows_queue_failure() -> None:
    context = GeometryFitWorkerContext.from_job({"event_queue": FailingQueue()})

    context.emit_event("source_cache_ready", {"ok": True})


def test_worker_context_emit_event_no_queue_is_noop() -> None:
    context = GeometryFitWorkerContext.from_job({"event_queue": None})

    context.emit_event("source_cache_ready", {"ok": True})


def test_worker_context_emit_event_preserves_manual_pair_backgrounds_skipped_shape() -> None:
    queue = RecordingQueue()
    payload = {
        "skipped_manual_pair_backgrounds": {1: "bg2.osc"},
        "skipped_background_indices": [1],
        "message": "preflight: skipping selected backgrounds without saved manual Qr/Qz pairs",
    }
    context = GeometryFitWorkerContext.from_job({"job_id": 3, "event_queue": queue})

    context.emit_event("manual_pair_backgrounds_skipped", payload)
    payload["skipped_background_indices"].append(2)

    assert queue.items == [
        {
            "job_id": 3,
            "kind": "manual_pair_backgrounds_skipped",
            "payload": {
                "skipped_manual_pair_backgrounds": {1: "bg2.osc"},
                "skipped_background_indices": [1],
                "message": (
                    "preflight: skipping selected backgrounds without saved manual Qr/Qz pairs"
                ),
            },
        }
    ]


def test_worker_context_current_source_cache_generation_defaults_zero() -> None:
    context = GeometryFitWorkerContext.from_job({})

    assert context.current_source_cache_generation(4) == 0


def test_worker_context_advance_source_cache_generation_updates_job_data() -> None:
    context = GeometryFitWorkerContext.from_job(
        {"source_cache_generation_by_background": {1: 3, 2: 5}}
    )

    assert context.advance_source_cache_generation(2) == 6
    assert context.source_cache_generation_by_background == {1: 3, 2: 6}
    assert context.job_data["source_cache_generation_by_background"] == {1: 3, 2: 6}


def test_worker_context_source_cache_generation_matches_none_generation() -> None:
    context = GeometryFitWorkerContext.from_job({})

    assert context.source_cache_generation_matches(2, None) is True


def test_worker_context_source_cache_generation_matches_current_generation() -> None:
    context = GeometryFitWorkerContext.from_job(
        {"source_cache_generation_by_background": {2: 5}}
    )

    assert context.source_cache_generation_matches(2, 5) is True


def test_worker_context_source_cache_generation_rejects_stale_generation() -> None:
    context = GeometryFitWorkerContext.from_job(
        {"source_cache_generation_by_background": {2: 5}}
    )

    assert context.source_cache_generation_matches(2, 4) is False


def test_worker_context_source_snapshot_diagnostics_replace_existing_values() -> None:
    context = GeometryFitWorkerContext.from_job(
        {"source_snapshot_diagnostics": {"old": True}}
    )

    context.set_worker_source_snapshot_diagnostics(new=True)

    assert context.worker_source_snapshot_diagnostics == {"new": True}


def test_worker_context_last_source_snapshot_diagnostics_returns_copy() -> None:
    context = GeometryFitWorkerContext.from_job(
        {"source_snapshot_diagnostics": {"value": 1}}
    )

    diagnostics = context.last_worker_source_snapshot_diagnostics()
    diagnostics["value"] = 2

    assert context.worker_source_snapshot_diagnostics == {"value": 1}


def test_worker_context_last_simulation_diagnostics_returns_copy() -> None:
    context = GeometryFitWorkerContext.from_job({"simulation_diagnostics": {"value": 1}})

    diagnostics = context.last_worker_simulation_diagnostics()
    diagnostics["value"] = 2

    assert context.worker_simulation_diagnostics == {"value": 1}


def test_worker_context_load_background_by_index_snapshot_returns_float64_copies() -> None:
    native = np.asarray([[1]], dtype=np.int32)
    display = np.asarray([[2]], dtype=np.int32)
    context = GeometryFitWorkerContext.from_job(
        {"background_images": {0: {"native": native, "display": display}}}
    )

    native_copy, display_copy = context.load_background_by_index_snapshot(0)
    native[0, 0] = 99
    display[0, 0] = 88

    assert native_copy.dtype == np.float64
    assert display_copy.dtype == np.float64
    assert native_copy[0, 0] == 1.0
    assert display_copy[0, 0] == 2.0


def test_worker_context_load_background_by_index_snapshot_uses_int_index() -> None:
    context = GeometryFitWorkerContext.from_job(
        {"background_images": {2: {"native": [[1.0]], "display": [[2.0]]}}}
    )

    native_copy, display_copy = context.load_background_by_index_snapshot("2")

    assert native_copy[0, 0] == 1.0
    assert display_copy[0, 0] == 2.0


def test_worker_context_load_background_by_index_snapshot_missing_payload_matches_inline_behavior() -> None:
    context = GeometryFitWorkerContext.from_job({})

    native_copy, display_copy = context.load_background_by_index_snapshot(0)

    assert native_copy.shape == ()
    assert display_copy.shape == ()
    assert native_copy.dtype == np.float64
    assert display_copy.dtype == np.float64
    assert np.isnan(native_copy)
    assert np.isnan(display_copy)


def test_project_source_rows_for_background_preserves_order_and_identity_fields() -> None:
    context, _fake_deps = _context_with_source_projection_deps()
    rows = [
        _source_row(
            q_group_key=("q_group", "primary", 3, 10),
            hkl=(-2, 1, 5),
            normalized_hkl=(-2, 1, 5),
            source_reflection_index=11,
            source_row_index=99,
            locked_qr=1.5,
            locked_qz=2.75,
        ),
        _source_row(
            q_group_key=("q_group", "primary", 3, 11),
            hkl=(-2, 1, 6),
            normalized_hkl=(-2, 1, 6),
            source_reflection_index=12,
            source_row_index=100,
            locked_qr=1.75,
            locked_qz=2.95,
        ),
    ]

    projected = context.project_source_rows_for_background(0, rows)

    assert [row["source_row_index"] for row in projected] == [99, 100]
    assert all(row["projected"] is True for row in projected)
    assert projected[0]["q_group_key"] == ("q_group", "primary", 3, 10)
    assert projected[0]["hkl"] == (-2, 1, 5)
    assert projected[0]["normalized_hkl"] == (-2, 1, 5)
    assert projected[0]["source_reflection_index"] == 11
    assert projected[0]["source_reflection_namespace"] == "full_reflection"
    assert projected[0]["source_reflection_is_full"] is True
    assert (projected[0]["locked_qr"], projected[0]["locked_qz"]) == (1.5, 2.75)


def test_project_source_rows_for_background_empty_input_matches_current_shape() -> None:
    context, _fake_deps = _context_with_source_projection_deps()

    assert context.project_source_rows_for_background(0, []) == []
    assert context.project_source_rows_for_background(0, None) == []


def test_project_source_rows_for_background_detector_empty_projection_preserves_rows() -> None:
    context, fake_deps = _context_with_source_projection_deps()
    fake_deps.callbacks.projected_rows = []
    rows = [_source_row(source_row_index=4, caked_x=1.0, caked_y=2.0)]

    projected = context.project_source_rows_for_background(0, rows)

    assert projected == rows


def test_project_source_rows_for_background_detector_empty_projection_rejects_stored_caked_rows() -> None:
    context, fake_deps = _context_with_source_projection_deps()
    fake_deps.callbacks.projected_rows = []
    row = _source_row(source_row_index=4, caked_x=1.0, caked_y=2.0)
    row.pop("source_label")

    projected = context.project_source_rows_for_background(0, [row])

    assert projected == []


def test_project_source_rows_by_row_background_preserves_grouping_order_and_identity() -> None:
    context, fake_deps = _context_with_source_projection_deps()
    rows = [
        _source_row(background_index=1, source_row_index=1, locked_qr=4.0, locked_qz=5.0),
        _source_row(background_index=0, source_row_index=2, locked_qr=6.0, locked_qz=7.0),
        _source_row(background_index=1, source_row_index=3, locked_qr=8.0, locked_qz=9.0),
    ]

    projected = context.project_source_rows_by_row_background(rows)

    assert [row["background_index"] for row in projected] == [1, 0, 1]
    assert [row["source_row_index"] for row in projected] == [1, 2, 3]
    assert [(row["locked_qr"], row["locked_qz"]) for row in projected] == [
        (4.0, 5.0),
        (6.0, 7.0),
        (8.0, 9.0),
    ]
    assert [
        [row["background_index"] for row in call]
        for call in fake_deps.callbacks.calls
    ] == [[1, 1], [0]]


def test_project_source_rows_by_row_background_missing_row_background_matches_current_fallback() -> None:
    context, _fake_deps = _context_with_source_projection_deps(
        {"current_background_index": 3}
    )
    missing_background_row = _source_row(source_row_index=1)
    missing_background_row.pop("background_index")
    rows = [
        missing_background_row,
        _source_row(background_index=2, source_row_index=2),
    ]

    projected = context.project_source_rows_by_row_background(rows)

    assert [row["background_index"] for row in projected] == [3, 2]
    assert [row["source_row_index"] for row in projected] == [1, 2]


def test_worker_native_detector_to_display_returns_none_for_missing_background() -> None:
    context, _fake_deps = _context_with_source_projection_deps(
        {"background_images": {}}
    )

    assert (
        context.worker_native_detector_coords_to_detector_display_coords_for_background(5)
        is None
    )


def test_worker_native_detector_to_display_returns_none_before_requiring_deps() -> None:
    context = GeometryFitWorkerContext.from_job({})

    assert (
        context.worker_native_detector_coords_to_detector_display_coords_for_background(
            "bad"
        )
        is None
    )


def test_worker_native_detector_to_display_uses_background_shape() -> None:
    context, fake_deps = _context_with_source_projection_deps(
        {
            "background_images": {
                2: {"native": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
            }
        }
    )

    to_display = (
        context.worker_native_detector_coords_to_detector_display_coords_for_background(2)
    )

    assert callable(to_display)
    assert to_display(7.0, 8.0) == (17.0, 28.0)
    assert fake_deps.rotate_calls[-1]["shape"] == (2, 3)


def test_worker_native_detector_to_display_uses_job_rotate_k() -> None:
    context, fake_deps = _context_with_source_projection_deps(
        {
            "display_rotate_k": 3,
            "background_images": {0: {"native": [[1.0, 2.0], [3.0, 4.0]]}},
        }
    )

    to_display = (
        context.worker_native_detector_coords_to_detector_display_coords_for_background(0)
    )

    assert callable(to_display)
    to_display(1.0, 2.0)
    assert fake_deps.rotate_calls[-1]["rotate_k"] == 3


def test_worker_native_detector_to_display_sets_stable_name() -> None:
    context, _fake_deps = _context_with_source_projection_deps()

    to_display = (
        context.worker_native_detector_coords_to_detector_display_coords_for_background(1)
    )

    assert callable(to_display)
    assert (
        to_display.__name__
        == "_worker_native_detector_coords_to_detector_display_coords_bg_1"
    )


def test_worker_geometry_manual_entry_display_coords_returns_none_for_non_mapping() -> None:
    context, _fake_deps = _context_with_source_projection_deps()

    assert context.worker_geometry_manual_entry_display_coords(None) is None
    assert context.worker_geometry_manual_entry_display_coords(["x", "y"]) is None


def test_worker_geometry_manual_entry_display_coords_returns_none_before_requiring_deps() -> None:
    context = GeometryFitWorkerContext.from_job({})

    assert context.worker_geometry_manual_entry_display_coords(None) is None


def test_worker_geometry_manual_entry_display_coords_returns_caked_fields_before_requiring_deps() -> None:
    context = GeometryFitWorkerContext.from_job({"pick_uses_caked_space": True})

    point = context.worker_geometry_manual_entry_display_coords(
        {"caked_x": 11.0, "caked_y": 12.0}
    )

    assert point == (11.0, 12.0)


def test_worker_geometry_manual_entry_display_coords_prefers_caked_fields_when_pick_uses_caked_space() -> None:
    context, _fake_deps = _context_with_source_projection_deps(
        {"pick_uses_caked_space": True}
    )

    point = context.worker_geometry_manual_entry_display_coords(
        {
            "caked_x": 11.0,
            "caked_y": 12.0,
            "x": 1.0,
            "y": 2.0,
        }
    )

    assert point == (11.0, 12.0)


def test_worker_geometry_manual_entry_display_coords_prefers_detector_display_tuple_fields() -> None:
    context, _fake_deps = _context_with_source_projection_deps()

    point = context.worker_geometry_manual_entry_display_coords(
        {
            "geometry_detector_display_px": (21.0, 22.0),
            "x": 1.0,
            "y": 2.0,
        }
    )

    assert point == (21.0, 22.0)


def test_worker_geometry_manual_entry_display_coords_falls_back_to_native_projection() -> None:
    context, fake_deps = _context_with_source_projection_deps(
        {
            "background_images": {
                1: {"native": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
            }
        }
    )

    point = context.worker_geometry_manual_entry_display_coords(
        {
            "background_index": 1,
            "geometry_detector_native_px": (3.0, 4.0),
        }
    )

    assert point == (13.0, 24.0)
    assert fake_deps.rotate_calls[-1]["shape"] == (2, 3)


def test_worker_geometry_manual_entry_display_coords_uses_current_background_fallback() -> None:
    context, fake_deps = _context_with_source_projection_deps(
        {
            "current_background_index": 1,
            "background_images": {
                1: {"native": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
            },
        }
    )

    point = context.worker_geometry_manual_entry_display_coords(
        {"geometry_detector_native_px": (5.0, 6.0)}
    )

    assert point == (15.0, 26.0)
    assert fake_deps.rotate_calls[-1]["shape"] == (2, 3)


def test_project_source_rows_for_background_view_reuses_matching_cached_rows() -> None:
    context, fake_deps = _context_with_source_projection_deps(
        {"projection_view_mode": "caked"}
    )
    rows = context.mark_worker_cached_projection_rows(
        [_source_row(background_index=0, source_row_index=1)],
        background_index=0,
        mode="caked",
    )

    projected = context.project_source_rows_for_background_view_worker(0, rows)

    assert projected is not rows
    assert projected == rows
    assert fake_deps.callbacks.calls == []


def test_project_source_rows_for_background_view_projects_when_cache_mismatch() -> None:
    context, fake_deps = _context_with_source_projection_deps(
        {
            "projection_view_mode": "caked",
            "background_images": {
                0: {"native": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
            },
        }
    )
    context.caked_payload_deps = FakeCakedPayloadDeps().deps
    rows = context.mark_worker_cached_projection_rows(
        [_source_row(background_index=1, source_row_index=1)],
        background_index=1,
        mode="caked",
    )
    context.job_data["projection_payload_by_background"] = {
        0: _ready_projection_payload()
    }

    projected = context.project_source_rows_for_background_view_worker(0, rows)

    assert projected[0]["projected"] is True
    assert len(fake_deps.callbacks.calls) == 1


def test_project_source_rows_for_background_view_uses_mode_override() -> None:
    context, _fake_deps = _context_with_source_projection_deps(
        {"projection_view_mode": "detector"}
    )
    rows = context.mark_worker_cached_projection_rows(
        [_source_row(background_index=0, source_row_index=1)],
        background_index=0,
        mode="q_space",
    )

    projected = context.project_source_rows_for_background_view_worker(
        0,
        rows,
        mode_override="q_space",
    )

    assert projected == rows


def test_project_source_rows_for_background_view_defaults_strict_caked_projection_true() -> None:
    context, _fake_deps = _context_with_source_projection_deps(
        {"projection_view_mode": "caked"}
    )
    context.caked_payload_deps = FakeCakedPayloadDeps().deps

    try:
        context.project_source_rows_for_background_view_worker(
            0,
            [_source_row(background_index=0, source_row_index=1)],
        )
    except RuntimeError as exc:
        assert str(exc) == "exact caked projector unavailable for background 1"
    else:
        raise AssertionError("expected strict caked projection failure")


def test_mark_worker_cached_projection_rows_noops_for_detector_mode() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = [_source_row(source_row_index=1)]

    marked = context.mark_worker_cached_projection_rows(
        rows,
        background_index=0,
        mode="detector",
    )

    assert marked is rows
    assert "_geometry_fit_worker_cached_projection" not in rows[0]


def test_mark_worker_cached_projection_rows_marks_caked_rows_in_place() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = [_source_row(source_row_index=1)]

    marked = context.mark_worker_cached_projection_rows(
        rows,
        background_index="2",
        mode="caked",
    )

    assert marked is rows
    assert rows[0]["_geometry_fit_worker_cached_projection"] is True
    assert rows[0]["_geometry_fit_worker_projection_mode"] == "caked"
    assert rows[0]["_geometry_fit_worker_projection_background_index"] == 2


def test_mark_worker_cached_projection_rows_marks_q_space_rows_in_place() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = [_source_row(source_row_index=1)]

    marked = context.mark_worker_cached_projection_rows(
        rows,
        background_index=3,
        mode="q_space",
    )

    assert marked is rows
    assert rows[0]["_geometry_fit_worker_cached_projection"] is True
    assert rows[0]["_geometry_fit_worker_projection_mode"] == "q_space"
    assert rows[0]["_geometry_fit_worker_projection_background_index"] == 3


def test_worker_cached_projection_rows_match_rejects_empty_rows() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()

    assert (
        context.worker_cached_projection_rows_match(
            [],
            background_index=0,
            mode="caked",
        )
        is False
    )


def test_worker_cached_projection_rows_match_rejects_detector_mode() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = context.mark_worker_cached_projection_rows(
        [_source_row()],
        background_index=0,
        mode="caked",
    )

    assert (
        context.worker_cached_projection_rows_match(
            rows,
            background_index=0,
            mode="detector",
        )
        is False
    )


def test_worker_cached_projection_rows_match_rejects_missing_marker() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()

    assert (
        context.worker_cached_projection_rows_match(
            [_source_row()],
            background_index=0,
            mode="caked",
        )
        is False
    )


def test_worker_cached_projection_rows_match_rejects_background_mismatch() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = context.mark_worker_cached_projection_rows(
        [_source_row()],
        background_index=1,
        mode="caked",
    )

    assert (
        context.worker_cached_projection_rows_match(
            rows,
            background_index=0,
            mode="caked",
        )
        is False
    )


def test_worker_cached_projection_rows_match_rejects_mode_mismatch() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = context.mark_worker_cached_projection_rows(
        [_source_row()],
        background_index=0,
        mode="q_space",
    )

    assert (
        context.worker_cached_projection_rows_match(
            rows,
            background_index=0,
            mode="caked",
        )
        is False
    )


def test_worker_cached_projection_rows_match_accepts_matching_caked_rows() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = context.mark_worker_cached_projection_rows(
        [_source_row()],
        background_index=0,
        mode="caked",
    )

    assert (
        context.worker_cached_projection_rows_match(
            rows,
            background_index=0,
            mode="caked",
        )
        is True
    )


def test_worker_cached_projection_rows_match_accepts_matching_q_space_rows() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    rows = context.mark_worker_cached_projection_rows(
        [_source_row()],
        background_index=0,
        mode="q_space",
    )

    assert (
        context.worker_cached_projection_rows_match(
            rows,
            background_index=0,
            mode="q_space",
        )
        is True
    )


def test_bundle_rows_returns_empty_for_non_bundle() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()

    assert context.bundle_rows(object()) == []


def test_bundle_rows_reuses_matching_cached_projected_rows_as_copies() -> None:
    context, _source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "caked"}
    )
    projected_rows = context.mark_worker_cached_projection_rows(
        [_source_row(source_row_index=1, projected=True)],
        background_index=0,
        mode="caked",
    )
    bundle = FakeBackgroundCacheBundle(projected_rows=projected_rows)

    rows = context.bundle_rows(bundle)
    rows[0]["source_row_index"] = 99

    assert rows[0]["projected"] is True
    assert bundle.projected_rows[0]["source_row_index"] == 1


def test_bundle_rows_marks_unmarked_projected_rows_when_mode_matches_and_no_params_override() -> None:
    context, _source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "caked"}
    )
    bundle = FakeBackgroundCacheBundle(
        projected_rows=[_source_row(source_row_index=1, projected=True)]
    )

    rows = context.bundle_rows(bundle)

    assert rows[0]["_geometry_fit_worker_cached_projection"] is True
    assert rows[0]["_geometry_fit_worker_projection_mode"] == "caked"
    assert rows[0]["_geometry_fit_worker_projection_background_index"] == 0


def test_bundle_rows_remarks_stale_projected_rows_when_mode_matches() -> None:
    context, source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "caked"}
    )
    stale_rows = context.mark_worker_cached_projection_rows(
        [_source_row(source_row_index=1, projected=True)],
        background_index=2,
        mode="caked",
    )
    bundle = FakeBackgroundCacheBundle(
        stored_rows=[_source_row(source_row_index=2)],
        projected_rows=stale_rows,
    )

    rows = context.bundle_rows(bundle)

    assert rows[0]["source_row_index"] == 1
    assert rows[0]["_geometry_fit_worker_projection_background_index"] == 0
    assert not source_deps.callbacks.calls


def test_bundle_rows_remarks_incompatible_projection_modes_when_mode_matches() -> None:
    context, source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "caked"}
    )
    q_space_rows = context.mark_worker_cached_projection_rows(
        [_source_row(source_row_index=1, projected=True)],
        background_index=0,
        mode="q_space",
    )
    bundle = FakeBackgroundCacheBundle(
        stored_rows=[_source_row(source_row_index=2)],
        projected_rows=q_space_rows,
    )

    rows = context.bundle_rows(bundle)

    assert rows[0]["source_row_index"] == 1
    assert rows[0]["_geometry_fit_worker_projection_mode"] == "caked"
    assert not source_deps.callbacks.calls


def test_bundle_rows_matching_cached_rows_win_before_params_override() -> None:
    context, source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "caked"}
    )
    projected_rows = context.mark_worker_cached_projection_rows(
        [_source_row(source_row_index=1, projected=True)],
        background_index=0,
        mode="caked",
    )
    bundle = FakeBackgroundCacheBundle(
        stored_rows=[_source_row(source_row_index=2)],
        projected_rows=projected_rows,
    )

    rows = context.bundle_rows(bundle, params_override={"center": (1.0, 2.0)})

    assert rows[0]["source_row_index"] == 1
    assert not source_deps.callbacks.calls


def test_bundle_rows_detector_mode_returns_stored_rows_when_no_projected_rows() -> None:
    context, _source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "detector"}
    )
    stored_rows = [_source_row(source_row_index=3)]
    bundle = FakeBackgroundCacheBundle(stored_rows=stored_rows, projected_rows=[])

    rows = context.bundle_rows(bundle)
    rows[0]["source_row_index"] = 99

    assert bundle.stored_rows[0]["source_row_index"] == 3


def test_store_worker_background_cache_bundle_updates_cache_map() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    bundle = FakeBackgroundCacheBundle(background_index=2)

    context.store_worker_background_cache_bundle(bundle)

    assert context.worker_background_cache_by_index[2] is bundle


def test_store_worker_background_cache_bundle_writes_source_snapshot_shape() -> None:
    context, _source_deps = _context_with_cache_bundle_deps(
        {
            "projection_view_signature_by_background": {
                2: {"mode": "caked", "digest": "sig"}
            }
        }
    )
    bundle = FakeBackgroundCacheBundle(
        background_index=2,
        stored_rows=[_source_row(source_row_index=1)],
        projected_rows=[_source_row(source_row_index=2, projected=True)],
        diagnostics={"status": "ready"},
    )

    context.store_worker_background_cache_bundle(bundle)
    snapshot = context.worker_source_row_snapshots[2]

    assert set(snapshot) == {
        "background_index",
        "created_from",
        "diagnostics",
        "projected_row_count",
        "projected_rows",
        "projection_view_signature",
        "requested_signature",
        "requested_signature_summary",
        "row_count",
        "rows",
        "simulation_signature",
        "stored_rows",
        "valid_for_geometry_fit_dataset",
        "valid_for_picker",
    }
    assert snapshot["projection_view_signature"] == {"mode": "caked", "digest": "sig"}


def test_store_worker_background_cache_bundle_deep_copies_rows_and_diagnostics() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    bundle = FakeBackgroundCacheBundle(
        stored_rows=[_source_row(source_row_index=1)],
        projected_rows=[_source_row(source_row_index=2, projected=True)],
        diagnostics={"rows": [{"value": 1}]},
    )

    context.store_worker_background_cache_bundle(bundle)
    snapshot = context.worker_source_row_snapshots[0]
    snapshot["rows"][0]["source_row_index"] = 99
    snapshot["projected_rows"][0]["source_row_index"] = 98
    snapshot["diagnostics"]["rows"][0]["value"] = 2

    assert bundle.stored_rows[0]["source_row_index"] == 1
    assert bundle.projected_rows[0]["source_row_index"] == 2
    assert bundle.diagnostics["rows"][0]["value"] == 1


def test_store_worker_background_cache_bundle_writes_projection_view_signature() -> None:
    context, _source_deps = _context_with_cache_bundle_deps(
        {"projection_view_signature_by_background": {0: {"mode": "q_space"}}}
    )
    bundle = FakeBackgroundCacheBundle()

    context.store_worker_background_cache_bundle(bundle)

    assert (
        context.worker_source_row_snapshots[0]["projection_view_signature"]
        == {"mode": "q_space"}
    )


def test_store_worker_background_cache_bundle_advances_generation_and_job_data() -> None:
    context, _source_deps = _context_with_cache_bundle_deps(
        {"source_cache_generation_by_background": {0: 4}}
    )
    bundle = FakeBackgroundCacheBundle()

    generation = context.store_worker_background_cache_bundle(bundle)

    assert generation == 5
    assert context.source_cache_generation_by_background[0] == 5
    assert context.job_data["source_cache_generation_by_background"][0] == 5


def _caked_storage_result_keys() -> set[str]:
    return {
        "background_index",
        "background_label",
        "source_cache_generation_id",
        "row_count",
        "caked_view_stored",
        "caked_view_status",
        "roi_enabled",
        "roi_used_restricted_cake",
        "roi_pixel_count",
        "roi_fraction",
        "roi_fallback_reason",
        "roi_half_width_px",
        "elapsed_s",
    }


def test_store_worker_caked_view_for_background_missing_native_background_result_shape() -> None:
    context, _storage_deps = _context_with_caked_view_storage_deps({})
    bundle = FakeBackgroundCacheBundle(
        background_index=2,
        background_label="other background",
        stored_rows=[{"row": 1}],
    )

    result = context.store_worker_caked_view_for_background(
        bundle,
        source_cache_generation_id=7,
    )

    assert set(result) == _caked_storage_result_keys()
    assert result["background_index"] == 2
    assert result["background_label"] == "other background"
    assert result["source_cache_generation_id"] == 7
    assert result["row_count"] == 1
    assert result["caked_view_stored"] is False
    assert result["caked_view_status"] == "missing_native_background"
    assert result["roi_fallback_reason"] == "missing_native_background"


def test_store_worker_caked_view_for_background_missing_integrator_result_shape() -> None:
    caked_payload_deps = FakeCakedPayloadDeps()
    caked_payload_deps.integrator = None
    context, _storage_deps = _context_with_caked_view_storage_deps(
        {"background_images": {0: {"native": np.ones((2, 2), dtype=np.float64)}}},
        caked_payload_deps=caked_payload_deps,
    )

    result = context.store_worker_caked_view_for_background(
        FakeBackgroundCacheBundle(background_index=0, stored_rows=[{"row": 1}])
    )

    assert set(result) == _caked_storage_result_keys()
    assert result["caked_view_stored"] is False
    assert result["caked_view_status"] == "missing_integrator"
    assert result["roi_fallback_reason"] == "missing_integrator"


def test_store_worker_caked_view_for_background_invalid_caked_payload_result_shape() -> None:
    context, storage_deps = _context_with_caked_view_storage_deps(
        {"background_images": {0: {"native": np.ones((2, 2), dtype=np.float64)}}}
    )
    storage_deps.prepared_payload = None

    result = context.store_worker_caked_view_for_background(
        FakeBackgroundCacheBundle(background_index=0, stored_rows=[{"row": 1}])
    )

    assert set(result) == _caked_storage_result_keys()
    assert result["caked_view_stored"] is False
    assert result["caked_view_status"] == "invalid_caked_payload"
    assert result["roi_fallback_reason"] == "invalid_caked_payload"


def test_store_worker_caked_view_for_background_stale_generation_rejects_storage() -> None:
    context, _storage_deps = _context_with_caked_view_storage_deps(
        {
            "background_images": {0: {"native": np.ones((2, 2), dtype=np.float64)}},
            "source_cache_generation_by_background": {0: 2},
        }
    )

    result = context.store_worker_caked_view_for_background(
        FakeBackgroundCacheBundle(background_index=0, stored_rows=[{"row": 1}]),
        source_cache_generation_id=1,
    )

    assert result["caked_view_stored"] is False
    assert result["caked_view_status"] == "stale_generation"
    assert context.job_data["caked_views_by_background"] == {}
    assert context.job_data["projection_payload_by_background"] == {}


def test_store_worker_caked_view_for_background_writes_payloads_and_events() -> None:
    context, storage_deps = _context_with_caked_view_storage_deps(
        {
            "background_images": {0: {"native": np.ones((2, 2), dtype=np.float64)}},
            "source_cache_generation_by_background": {0: 3},
        }
    )

    result = context.store_worker_caked_view_for_background(
        FakeBackgroundCacheBundle(background_index=0, stored_rows=[{"row": 1}]),
        source_cache_generation_id=3,
        stage_callback="stage-callback",
    )

    assert result["caked_view_stored"] is True
    assert result["caked_view_status"] == "stored"
    stored_payload = context.job_data["caked_views_by_background"][0]
    assert set(stored_payload) == {
        "background",
        "radial_axis",
        "azimuth_axis",
        "raw_azimuth_axis",
        "raw_to_gui_row_permutation",
        "transform_bundle",
        "detector_shape",
        "roi_enabled",
        "roi_used_restricted_cake",
        "roi_pixel_count",
        "roi_fraction",
        "roi_fallback_reason",
        "roi_half_width_px",
        "caked_display_sanitize_status",
        "sanitized_empty_bin_count",
        "nonfinite_supported_bin_count",
    }
    assert context.job_data["projection_payload_by_background"][0]["digest"] == (
        "projection-digest"
    )
    event_names = [event[1] for event in storage_deps.stage_events]
    assert event_names == [
        "source_cache_caked_roi_selection_start",
        "source_cache_caked_roi_selection_ready",
        "source_cache_full_cake_start",
        "source_cache_full_cake_ready",
        "projection_payload_ready",
    ]
    assert all(event[0] == "stage-callback" for event in storage_deps.stage_events)
    assert all(
        {"background_index", "background_label", "source_cache_generation_id", "row_count"}
        <= set(event[2])
        for event in storage_deps.stage_events
    )


def test_store_worker_caked_view_for_background_refreshes_caked_projection_rows() -> None:
    context, _storage_deps = _context_with_caked_view_storage_deps(
        {
            "background_images": {0: {"native": np.ones((2, 2), dtype=np.float64)}},
            "projection_view_mode": "caked",
            "current_background_index": 0,
            "analysis_bins": (4, 5),
            "source_cache_generation_by_background": {0: 3},
        }
    )

    result = context.store_worker_caked_view_for_background(
        FakeBackgroundCacheBundle(
            background_index=0,
            stored_rows=[{"background_index": 0, "source_label": "manual"}],
            diagnostics={"status": "cached"},
        ),
        source_cache_generation_id=3,
    )

    assert result["caked_view_stored"] is True
    signature = context.job_data["projection_view_signature_by_background"][0]
    assert signature["mode_override"] == "caked"
    assert context.job_data["projection_view_signature"]["mode_override"] == "caked"
    assert (
        context.job_data["projection_view_signature"]["normalized_background_index"]
        == 0
    )
    assert context.worker_background_cache_by_index[0].projected_rows
    assert context.worker_source_row_snapshots[0]["projected_row_count"] == 1
    assert (
        context.worker_source_row_snapshots[0]["diagnostics"][
            "projection_view_signature_digest"
        ][0]
        == "digest"
    )


def test_build_geometry_fit_background_cache_bundle_copies_stored_and_projected_rows() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    stored_rows = [_source_row(source_row_index=1)]
    projected_rows = [_source_row(source_row_index=2, projected=True)]

    bundle = context.build_geometry_fit_background_cache_bundle(
        background_index=0,
        background_label="bg1",
        requested_signature=("sig", 0),
        requested_signature_summary={"digest": "sig"},
        theta_base=1.0,
        theta_initial=2.0,
        stored_rows=stored_rows,
        projected_rows=projected_rows,
        cache_source="unit",
    )
    stored_rows[0]["source_row_index"] = 99
    projected_rows[0]["source_row_index"] = 98

    assert bundle.stored_rows[0]["source_row_index"] == 1
    assert bundle.projected_rows[0]["source_row_index"] == 2


def test_build_geometry_fit_background_cache_bundle_detector_mode_projects_missing_projected_rows() -> None:
    context, source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "detector"}
    )

    bundle = context.build_geometry_fit_background_cache_bundle(
        background_index=0,
        background_label="bg1",
        requested_signature=("sig", 0),
        requested_signature_summary={"digest": "sig"},
        theta_base=1.0,
        theta_initial=2.0,
        stored_rows=[_source_row(source_row_index=3)],
        cache_source="unit",
    )

    assert bundle.projected_rows[0]["projected"] is True
    assert bundle.diagnostics["projected_peak_count"] == 1
    assert "projected_rows_generated_from_stored_rows" not in bundle.diagnostics
    assert len(source_deps.callbacks.calls) == 1


def test_build_geometry_fit_background_cache_bundle_caked_projection_failure_records_reason() -> None:
    context, source_deps = _context_with_cache_bundle_deps(
        {"projection_view_mode": "caked"}
    )
    context.caked_payload_deps = FakeCakedPayloadDeps().deps
    source_deps.callbacks.error = RuntimeError("projection unavailable")

    bundle = context.build_geometry_fit_background_cache_bundle(
        background_index=0,
        background_label="bg1",
        requested_signature=("sig", 0),
        requested_signature_summary={"digest": "sig"},
        theta_base=1.0,
        theta_initial=2.0,
        stored_rows=[_source_row(source_row_index=4)],
        cache_source="unit",
    )

    assert bundle.projected_rows == []
    assert bundle.diagnostics["projection_failure_reason"] == "projection_error:RuntimeError"
    assert bundle.diagnostics["projected_peak_count"] == 0


def test_build_geometry_fit_background_cache_bundle_sets_default_diagnostics() -> None:
    context, _source_deps = _context_with_cache_bundle_deps(
        {"live_cache_inventory": {"ready": 2}}
    )

    bundle = context.build_geometry_fit_background_cache_bundle(
        background_index=2,
        background_label="bg3",
        requested_signature=("sig", 2),
        requested_signature_summary={"digest": "sig2"},
        theta_base=3.0,
        theta_initial=4.0,
        stored_rows=[_source_row(background_index=2, source_row_index=5)],
        projected_rows=[],
        cache_source="unit",
        diagnostics={"status": "custom_status"},
    )

    assert bundle.diagnostics["source"] == "geometry_fit_background_cache"
    assert bundle.diagnostics["cache_family"] == "geometry_fit_background_cache"
    assert bundle.diagnostics["action"] == "prepare"
    assert bundle.diagnostics["status"] == "custom_status"
    assert bundle.diagnostics["background_index"] == 2
    assert bundle.diagnostics["background_label"] == "bg3"
    assert bundle.diagnostics["theta_base"] == 3.0
    assert bundle.diagnostics["theta_initial"] == 4.0
    assert bundle.diagnostics["raw_peak_count"] == 1
    assert bundle.diagnostics["cache_source"] == "unit"
    assert bundle.diagnostics["signature_match"] is True
    assert bundle.diagnostics["live_cache_inventory"] == {"ready": 2}


def test_build_geometry_fit_background_cache_bundle_preserves_optional_tables_and_metadata() -> None:
    context, _source_deps = _context_with_cache_bundle_deps()
    peak_table_lattice = [{"phase": "primary"}]
    hit_tables = [{"hits": [1]}]
    intersection_cache = [{"intersections": [2]}]
    cache_metadata = {"source": "logged"}

    bundle = context.build_geometry_fit_background_cache_bundle(
        background_index=0,
        background_label="bg1",
        requested_signature=("sig", 0),
        requested_signature_summary={"digest": "sig"},
        theta_base=1.0,
        theta_initial=2.0,
        stored_rows=[],
        projected_rows=[],
        cache_source="unit",
        peak_table_lattice=peak_table_lattice,
        hit_tables=hit_tables,
        intersection_cache=intersection_cache,
        cache_metadata=cache_metadata,
    )
    peak_table_lattice[0]["phase"] = "mutated"
    hit_tables[0]["hits"].append(99)
    intersection_cache[0]["intersections"].append(98)
    cache_metadata["source"] = "mutated"

    assert bundle.peak_table_lattice == [{"phase": "primary"}]
    assert bundle.hit_tables == [{"hits": [1]}]
    assert bundle.intersection_cache == [{"intersections": [2]}]
    assert bundle.cache_metadata == {"source": "logged"}


def test_prebuild_background_cache_bundle_worker_uses_matching_snapshot_without_rebuild() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "params": {"theta_initial": 1.0, "distance": 2.0},
            "requested_signatures": {0: ("sig", 0)},
            "requested_signature_summaries": {0: {"digest": "sig"}},
            "source_snapshots": {
                0: {
                    "simulation_signature": ("sig", 0),
                    "rows": [_source_row(source_row_index=11)],
                    "created_from": "source_snapshot",
                }
            },
            "background_labels": {0: "bg1"},
        }
    )

    bundle = context.prebuild_background_cache_bundle_worker(
        0,
        theta_base=2.0,
    )

    assert isinstance(bundle, FakeBackgroundCacheBundle)
    assert bundle.cache_source == "source_snapshot"
    assert bundle.stored_rows[0]["source_row_index"] == 11
    assert prebuild_deps.rebuild_calls == []
    assert context.worker_background_cache_by_index[0] is bundle
    assert context.worker_source_snapshot_diagnostics["status"] == "background_cache_ready"
    assert context.worker_simulation_diagnostics["status"] == "background_cache_ready"


def test_prebuild_background_cache_bundle_worker_records_pair_validation_failure() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "params": {"theta_initial": 1.0},
            "requested_signatures": {0: ("sig", 0)},
            "requested_signature_summaries": {0: {"digest": "sig"}},
            "source_snapshots": {
                0: {
                    "simulation_signature": ("sig", 0),
                    "rows": [_source_row(source_row_index=12)],
                    "created_from": "source_snapshot",
                }
            },
            "background_labels": {0: "bg1"},
        }
    )
    prebuild_deps.validation_result = {"valid": False, "reason": "missing_pair"}
    prebuild_deps.rebuild_result = FakeRebuildResult(
        stored_rows=[],
        diagnostics={"status": "empty_rebuild"},
    )

    bundle = context.prebuild_background_cache_bundle_worker(
        0,
        theta_base=2.0,
        required_pairs=[{"manual_pair_id": "pair-1"}],
    )

    assert bundle is None
    assert context.worker_source_snapshot_diagnostics["status"] == "empty_rebuild"
    assert prebuild_deps.rebuild_calls
    rebuild_call = prebuild_deps.rebuild_calls[0]
    assert rebuild_call["preflight_mode"] == "manual_geometry_targeted"
    assert rebuild_call["prior_diagnostics"]["status"] == (
        "background_cache_pair_validation_failed"
    )
    assert rebuild_call["prior_diagnostics"]["live_runtime_cache_validation"]["valid"] is False


def test_prebuild_background_cache_bundle_worker_live_rows_payload_requires_signature_match() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "params": {"theta_initial": 1.0},
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
            "live_rows_signature_by_background": {0: ("stale", 0)},
            "live_rows_by_background": {0: [_source_row(source_row_index=13)]},
            "live_rows_cache_metadata_by_background": {0: {"cache_source": "live"}},
        }
    )
    prebuild_deps.rebuild_result = FakeRebuildResult(
        stored_rows=[],
        diagnostics={"status": "empty_rebuild"},
    )

    context.prebuild_background_cache_bundle_worker(0, theta_base=2.0)

    assert prebuild_deps.live_rows_payloads
    payload = prebuild_deps.live_rows_payloads[0]
    assert payload["rows"] == []
    assert payload["cache_metadata"]["live_rows_signature_match"] is False
    assert payload["cache_metadata"]["reason"] == "requested_signature_mismatch"
    assert payload["cache_metadata"]["live_rows_payload_count"] == 0


def test_prebuild_background_cache_bundle_worker_rebuild_result_stores_bundle() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "params": {"theta_initial": 1.0},
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
        }
    )
    prebuild_deps.rebuild_result = FakeRebuildResult(
        requested_signature=("rebuilt", 0),
        requested_signature_summary={"digest": "rebuilt"},
        stored_rows=[_source_row(source_row_index=14)],
        projected_rows=[_source_row(source_row_index=15, projected=True)],
        rebuild_source="fresh_rebuild",
        diagnostics={"status": "fresh_rebuild_ready"},
        metadata={"cache": "metadata"},
    )

    bundle = context.prebuild_background_cache_bundle_worker(0, theta_base=2.0)

    assert isinstance(bundle, FakeBackgroundCacheBundle)
    assert bundle.requested_signature == ("rebuilt", 0)
    assert bundle.stored_rows[0]["source_row_index"] == 14
    assert bundle.projected_rows[0]["source_row_index"] == 15
    assert bundle.cache_source == "fresh_rebuild"
    assert context.worker_background_cache_by_index[0] is bundle
    assert context.source_cache_generation_by_background[0] == 1
    assert context.job_data["source_cache_generation_by_background"][0] == 1
    assert context.worker_source_snapshot_diagnostics["status"] == "fresh_rebuild_ready"
    assert context.worker_simulation_diagnostics["status"] == "fresh_rebuild_ready"


def test_prebuild_background_cache_bundle_worker_empty_rebuild_returns_none() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "params": {"theta_initial": 1.0},
            "requested_signatures": {0: ("requested", 0)},
        }
    )
    prebuild_deps.rebuild_result = FakeRebuildResult(
        stored_rows=[],
        projected_rows=[_source_row(source_row_index=16, projected=True)],
        diagnostics={"status": "empty_rebuild", "reason": "no_rows"},
    )

    bundle = context.prebuild_background_cache_bundle_worker(0, theta_base=2.0)

    assert bundle is None
    assert context.worker_background_cache_by_index == {}
    assert context.worker_source_snapshot_diagnostics == {
        "status": "empty_rebuild",
        "reason": "no_rows",
    }
    assert context.worker_simulation_diagnostics == {
        "status": "empty_rebuild",
        "reason": "no_rows",
    }


def _assert_diagnostic_keys(
    diagnostics: Mapping[str, object],
    *,
    status: str,
    keys: set[str],
) -> None:
    assert set(diagnostics) == keys
    assert diagnostics["status"] == status


def test_source_rows_for_background_worker_missing_cache_records_diagnostics() -> None:
    context, _prebuild_deps = _context_with_prebuild_deps(
        {
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
            "background_labels": {0: "bg1"},
            "live_cache_inventory": {"ready": 1},
        }
    )

    rows = context.source_rows_for_background_worker(0, consumer="dataset")

    assert rows == []
    diagnostics = context.worker_source_snapshot_diagnostics
    _assert_diagnostic_keys(
        diagnostics,
        status="background_cache_missing",
        keys={
            "source",
            "cache_family",
            "action",
            "consumer",
            "status",
            "background_index",
            "background_label",
            "requested_signature",
            "requested_signature_summary",
            "raw_peak_count",
            "projected_peak_count",
            "signature_match",
            "live_cache_inventory",
        },
    )
    assert diagnostics["background_index"] == 0
    assert diagnostics["background_label"] == "bg1"
    assert diagnostics["requested_signature"] == ("requested", 0)
    assert diagnostics["requested_signature_summary"] == {"digest": "requested"}
    assert diagnostics["raw_peak_count"] == 0
    assert diagnostics["projected_peak_count"] == 0
    assert diagnostics["signature_match"] is False
    assert diagnostics["live_cache_inventory"] == {"ready": 1}


def test_source_rows_for_background_worker_signature_mismatch_records_diagnostics() -> None:
    context, _prebuild_deps = _context_with_prebuild_deps(
        {
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
            "background_labels": {0: "bg1"},
            "live_cache_inventory": {"ready": 2},
        }
    )
    context.worker_background_cache_by_index[0] = FakeBackgroundCacheBundle(
        requested_signature=("stale", 0),
        stored_rows=[_source_row(source_row_index=21)],
        projected_rows=[_source_row(source_row_index=22, projected=True)],
        cache_source="prebuilt",
    )

    rows = context.source_rows_for_background_worker(0, consumer="dataset")

    assert rows == []
    diagnostics = context.worker_source_snapshot_diagnostics
    _assert_diagnostic_keys(
        diagnostics,
        status="background_cache_signature_mismatch",
        keys={
            "source",
            "cache_family",
            "action",
            "consumer",
            "status",
            "background_index",
            "background_label",
            "requested_signature",
            "requested_signature_summary",
            "snapshot_signature",
            "stored_signature_summary",
            "raw_peak_count",
            "projected_peak_count",
            "created_from",
            "signature_match",
            "live_cache_inventory",
        },
    )
    assert diagnostics["snapshot_signature"] == ("stale", 0)
    assert diagnostics["stored_signature_summary"] == {"signature": ("stale", 0)}
    assert diagnostics["raw_peak_count"] == 1
    assert diagnostics["projected_peak_count"] == 0
    assert diagnostics["created_from"] == "prebuilt"
    assert diagnostics["signature_match"] is False


def test_source_rows_for_background_worker_empty_projected_rows_records_diagnostics() -> None:
    context, _prebuild_deps = _context_with_prebuild_deps(
        {
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
            "background_labels": {0: "bg1"},
            "live_cache_inventory": {"ready": 3},
        }
    )
    context.worker_background_cache_by_index[0] = FakeBackgroundCacheBundle(
        requested_signature=("requested", 0),
        stored_rows=[],
        projected_rows=[],
        cache_source="prebuilt",
    )

    rows = context.source_rows_for_background_worker(0, consumer="dataset")

    assert rows == []
    diagnostics = context.worker_source_snapshot_diagnostics
    _assert_diagnostic_keys(
        diagnostics,
        status="background_cache_empty",
        keys={
            "source",
            "cache_family",
            "action",
            "consumer",
            "status",
            "background_index",
            "background_label",
            "requested_signature",
            "requested_signature_summary",
            "snapshot_signature",
            "stored_signature_summary",
            "raw_peak_count",
            "projected_peak_count",
            "created_from",
            "signature_match",
            "live_cache_inventory",
        },
    )
    assert diagnostics["snapshot_signature"] == ("requested", 0)
    assert diagnostics["stored_signature_summary"] == {"signature": ("requested", 0)}
    assert diagnostics["raw_peak_count"] == 0
    assert diagnostics["projected_peak_count"] == 0
    assert diagnostics["signature_match"] is True


def test_source_rows_for_background_worker_cache_hit_returns_projected_rows() -> None:
    context, _prebuild_deps = _context_with_prebuild_deps(
        {
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
            "background_labels": {0: "bg1"},
            "live_cache_inventory": {"ready": 4},
        }
    )
    projected_row = _source_row(source_row_index=24, projected=True)
    context.worker_background_cache_by_index[0] = FakeBackgroundCacheBundle(
        requested_signature=("requested", 0),
        stored_rows=[_source_row(source_row_index=25)],
        projected_rows=[projected_row],
        cache_source="prebuilt",
        theta_base=2.0,
        theta_initial=3.0,
    )

    rows = context.source_rows_for_background_worker(0, consumer="dataset")
    rows[0]["mutated"] = True

    assert rows[0]["source_row_index"] == 24
    assert "mutated" not in projected_row
    diagnostics = context.worker_source_snapshot_diagnostics
    _assert_diagnostic_keys(
        diagnostics,
        status="background_cache_hit",
        keys={
            "source",
            "cache_family",
            "action",
            "consumer",
            "status",
            "background_index",
            "background_label",
            "requested_signature",
            "requested_signature_summary",
            "snapshot_signature",
            "stored_signature_summary",
            "raw_peak_count",
            "projected_peak_count",
            "created_from",
            "cache_source",
            "signature_match",
            "theta_base",
            "theta_initial",
            "live_cache_inventory",
            "live_runtime_cache_validation",
        },
    )
    assert diagnostics["projected_peak_count"] == 1
    assert diagnostics["theta_base"] == 2.0
    assert diagnostics["theta_initial"] == 3.0
    assert diagnostics["live_runtime_cache_validation"] == {}


def test_source_rows_for_background_worker_generation_map_does_not_reject_cache_hit() -> None:
    context, _prebuild_deps = _context_with_prebuild_deps(
        {
            "requested_signatures": {0: ("requested", 0)},
            "source_cache_generation_by_background": {0: 7},
        }
    )
    context.worker_background_cache_by_index[0] = FakeBackgroundCacheBundle(
        requested_signature=("requested", 0),
        stored_rows=[_source_row(source_row_index=26)],
        projected_rows=[_source_row(source_row_index=27, projected=True)],
    )

    rows = context.source_rows_for_background_worker(0, consumer="dataset")

    assert [row["source_row_index"] for row in rows] == [27]
    assert context.source_cache_generation_by_background[0] == 7
    assert context.job_data["source_cache_generation_by_background"][0] == 7


def test_source_rows_for_background_worker_pair_validation_failure_rebuilds() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
            "background_labels": {0: "bg1"},
        }
    )
    context.worker_background_cache_by_index[0] = FakeBackgroundCacheBundle(
        requested_signature=("requested", 0),
        stored_rows=[_source_row(source_row_index=28)],
        projected_rows=[_source_row(source_row_index=29, projected=True)],
        cache_source="prebuilt",
        theta_base=4.0,
        theta_initial=5.0,
    )
    prebuild_deps.validation_result = {"valid": False, "reason": "missing_pair"}
    prebuild_deps.rebuild_result = FakeRebuildResult(
        stored_rows=[],
        diagnostics={"status": "empty_rebuild"},
    )

    rows = context.source_rows_for_background_worker(
        0,
        consumer="dataset",
        required_pairs=[{"manual_pair_id": "pair-1"}],
    )

    assert rows == []
    assert prebuild_deps.rebuild_calls
    prior_diagnostics = prebuild_deps.rebuild_calls[0]["prior_diagnostics"]
    _assert_diagnostic_keys(
        prior_diagnostics,
        status="background_cache_pair_validation_failed",
        keys={
            "source",
            "cache_family",
            "action",
            "consumer",
            "status",
            "background_index",
            "background_label",
            "requested_signature",
            "requested_signature_summary",
            "snapshot_signature",
            "stored_signature_summary",
            "raw_peak_count",
            "projected_peak_count",
            "created_from",
            "cache_source",
            "signature_match",
            "theta_base",
            "theta_initial",
            "live_cache_inventory",
            "live_runtime_cache_validation",
        },
    )
    assert prior_diagnostics["live_runtime_cache_validation"]["valid"] is False
    assert prior_diagnostics["live_runtime_cache_validation"]["reason"] == "missing_pair"


def test_rebuild_source_rows_for_background_worker_uses_prebuilt_trial_rows() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "projection_view_mode": "detector",
            "requested_signatures": {0: ("requested", 0)},
        }
    )
    context.worker_background_cache_by_index[0] = FakeBackgroundCacheBundle(
        requested_signature=("requested", 0),
        stored_rows=[_source_row(source_row_index=30)],
        projected_rows=[],
    )

    rows = context.rebuild_source_rows_for_background_worker(
        0,
        {"distance": 3.0},
        consumer="geometry_fit_trial_source_rows",
    )

    assert [row["source_row_index"] for row in rows] == [30]
    assert rows[0]["projected"] is True
    assert prebuild_deps.rebuild_calls == []


def test_rebuild_source_rows_for_background_worker_forwards_prebuild_call_arguments() -> None:
    context, prebuild_deps = _context_with_prebuild_deps(
        {
            "params": {"theta_initial": 1.0, "distance": 2.0},
            "requested_signatures": {0: ("requested", 0)},
            "requested_signature_summaries": {0: {"digest": "requested"}},
            "background_labels": {0: "bg1"},
        }
    )
    prebuild_deps.rebuild_result = FakeRebuildResult(
        stored_rows=[],
        diagnostics={"status": "empty_rebuild"},
    )

    rows = context.rebuild_source_rows_for_background_worker(
        0,
        {"distance": 3.0},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "prior"},
        required_pairs=[{"manual_pair_id": "pair-1"}],
    )

    assert rows == []
    rebuild_call = prebuild_deps.rebuild_calls[0]
    assert rebuild_call["background_index"] == 0
    assert rebuild_call["background_label"] == "bg1"
    assert rebuild_call["params_local"]["distance"] == 3.0
    assert rebuild_call["params_local"]["theta_initial"] == 10.0
    assert rebuild_call["consumer"] == "geometry_fit_dataset"
    assert rebuild_call["prior_diagnostics"] == {"status": "prior"}
    assert rebuild_call["requested_signature"][:2] == (
        "geometry_fit_worker_trial_source_rows",
        0,
    )
    assert rebuild_call["requested_signature_summary"] == {
        "signature": rebuild_call["requested_signature"]
    }
    assert rebuild_call["projection_view_mode"] == "detector"
    assert rebuild_call["required_pairs"] == [{"manual_pair_id": "pair-1"}]
    assert rebuild_call["preflight_mode"] == "manual_geometry_targeted"


def test_caked_projection_payload_status_reports_ready_payload() -> None:
    context, _fake_deps = _context_with_caked_deps()

    assert (
        context.caked_projection_payload_status(_ready_projection_payload())
        == "projection_payload_ready"
    )


def test_caked_projection_payload_status_reports_missing_axes() -> None:
    context, _fake_deps = _context_with_caked_deps()

    assert context.caked_projection_payload_status(None) == "projection_payload_missing_axes"


def test_caked_projection_payload_status_reports_axis_mismatch() -> None:
    context, _fake_deps = _context_with_caked_deps()
    payload = _ready_projection_payload(raw_azimuth_axis=[0.0])

    assert (
        context.caked_projection_payload_status(payload)
        == "projection_payload_axis_mismatch"
    )


def test_caked_projection_payload_status_reports_missing_bundle_or_digest() -> None:
    context, _fake_deps = _context_with_caked_deps()

    assert (
        context.caked_projection_payload_status(
            _ready_projection_payload(transform_bundle=None)
        )
        == "missing_exact_caked_bundle"
    )
    assert (
        context.caked_projection_payload_status(_ready_projection_payload(digest=None))
        == "missing_exact_caked_bundle"
    )


def test_projection_candidate_state_returns_projection_copy() -> None:
    context, _fake_deps = _context_with_caked_deps()
    candidate = _ready_projection_payload()

    projection, state = context.projection_candidate_state(
        candidate,
        detector_shape=(2, 3),
    )
    assert state == "ready"
    assert isinstance(projection, dict)
    projection["digest"] = "mutated"

    assert candidate["digest"] == "digest"


def test_projection_candidate_state_rejects_shape_mismatch() -> None:
    context, _fake_deps = _context_with_caked_deps()

    projection, state = context.projection_candidate_state(
        _ready_projection_payload(detector_shape=(8, 9)),
        detector_shape=(2, 3),
    )

    assert isinstance(projection, dict)
    assert state == "invalid"


def test_load_caked_view_snapshot_missing_payload_creates_no_job_entries() -> None:
    context, _fake_deps = _context_with_caked_deps()

    assert context.load_caked_view_by_index_snapshot(0) is None
    assert "caked_views_by_background" not in context.job_data
    assert "projection_payload_by_background" not in context.job_data


def test_load_caked_view_snapshot_stores_hydrated_view() -> None:
    context, fake_deps = _context_with_caked_deps(
        {"caked_views_by_background": {0: {"detector_shape": (2, 3)}}}
    )

    payload = context.load_caked_view_by_index_snapshot(0)

    assert isinstance(payload, dict)
    assert context.job_data["caked_views_by_background"][0] == payload
    assert fake_deps.hydrate_calls[-1]["require_background"] is True


def test_load_caked_view_snapshot_stores_projection_only_in_caked_mode() -> None:
    context, _fake_deps = _context_with_caked_deps(
        {
            "projection_view_mode": "caked",
            "caked_views_by_background": {0: {"detector_shape": (2, 3)}},
        }
    )

    context.load_caked_view_by_index_snapshot(0)

    assert 0 in context.job_data["projection_payload_by_background"]


def test_load_caked_projection_existing_ready_avoids_generated_resolver() -> None:
    context, fake_deps = _context_with_caked_deps(
        {"projection_payload_by_background": {0: _ready_projection_payload()}}
    )
    fake_deps.generated_payload = _ready_projection_payload(digest="generated")

    payload = context.load_caked_projection_by_index_snapshot(
        0,
        detector_shape=(2, 3),
        allow_generated_payload=True,
    )

    assert isinstance(payload, dict)
    assert payload["digest"] == "digest"
    assert fake_deps.generated_calls == []


def test_load_caked_projection_invalid_existing_avoids_generated_resolver() -> None:
    context, fake_deps = _context_with_caked_deps(
        {"projection_payload_by_background": {0: _ready_projection_payload(digest=None)}}
    )
    fake_deps.generated_payload = _ready_projection_payload(digest="generated")

    assert (
        context.load_caked_projection_by_index_snapshot(
            0,
            detector_shape=(2, 3),
            allow_generated_payload=True,
        )
        is None
    )
    assert fake_deps.generated_calls == []


def test_load_caked_projection_generated_disallowed_leaves_maps_unchanged() -> None:
    projection_map = {5: _ready_projection_payload(digest="existing")}
    context, fake_deps = _context_with_caked_deps(
        {"projection_payload_by_background": projection_map}
    )
    fake_deps.generated_payload = _ready_projection_payload(digest="generated")

    assert (
        context.load_caked_projection_by_index_snapshot(
            0,
            detector_shape=(2, 3),
            allow_generated_payload=False,
        )
        is None
    )

    assert context.job_data["projection_payload_by_background"] == projection_map
    assert fake_deps.generated_calls == []


def test_load_caked_projection_uses_generated_payload_only_when_allowed() -> None:
    context, fake_deps = _context_with_caked_deps({"analysis_bins": (4, 5)})
    fake_deps.generated_payload = _ready_projection_payload(digest="generated")

    payload = context.load_caked_projection_by_index_snapshot(
        0,
        detector_shape=(2, 3),
        allow_generated_payload=True,
    )

    assert isinstance(payload, dict)
    assert payload["digest"] == "generated"
    assert fake_deps.generated_calls == [
        {
            "background_index": 0,
            "detector_shape": (2, 3),
            "ai": "fake-ai",
            "analysis_preview_bins": (4, 5),
            "allow_generated_payload": True,
        }
    ]


def test_ensure_caked_projection_ready_emits_no_failure_event() -> None:
    context, fake_deps = _context_with_caked_deps(
        {
            "background_labels": {0: "bg0"},
            "projection_payload_by_background": {0: _ready_projection_payload()},
        }
    )

    payload = context.ensure_worker_caked_projection_payload(
        0,
        detector_shape=(2, 3),
        stage_callback="stage",
        emit_event=True,
    )

    assert isinstance(payload, dict)
    assert fake_deps.stage_events == [
        (
            "stage",
            "projection_payload_ready",
            {
                "background_index": 0,
                "background_label": "bg0",
                "status": "projection_payload_ready",
                "payload_kind": "projection",
                "message": (
                    "preflight: exact caked projection payload ready for background 1"
                ),
            },
        )
    ]


def test_ensure_caked_projection_failure_event_payload_fields() -> None:
    context, fake_deps = _context_with_caked_deps({})

    assert (
        context.ensure_worker_caked_projection_payload(
            1,
            detector_shape=(2, 3),
            stage_callback="stage",
            emit_event=True,
        )
        is None
    )

    assert fake_deps.stage_events == [
        (
            "stage",
            "projection_payload_missing_axes",
            {
                "background_index": 1,
                "background_label": "background 2",
                "status": "projection_payload_missing_axes",
                "payload_kind": "projection",
                "message": (
                    "preflight: exact caked projection payload failed for background 2 "
                    "(status=projection_payload_missing_axes)"
                ),
            },
        )
    ]


def test_worker_caked_view_payload_ready_true_for_exact_bundle() -> None:
    context, fake_deps = _context_with_caked_deps(
        {
            "background_images": {
                0: {"native": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
            },
            "projection_payload_by_background": {0: _ready_projection_payload()},
            "params": {"center_x": 10.0},
        }
    )

    assert context.worker_caked_view_payload_ready(0) is True
    assert fake_deps.hydrate_calls[-1] == {
        "detector_shape": (2, 3),
        "params": {"center_x": 10.0},
        "require_background": False,
    }


def test_worker_caked_view_payload_ready_false_for_missing_payload() -> None:
    context, _fake_deps = _context_with_caked_deps(
        {"background_images": {0: {"native": [[1.0, 2.0], [3.0, 4.0]]}}}
    )

    assert context.worker_caked_view_payload_ready(0) is False


def test_worker_caked_view_payload_ready_uses_background_detector_shape() -> None:
    context, fake_deps = _context_with_caked_deps(
        {
            "background_images": {
                2: {"native": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
            },
            "analysis_bins": (4, 5),
        }
    )
    fake_deps.generated_payload = _ready_projection_payload(detector_shape=(2, 3))

    assert context.worker_caked_view_payload_ready(2) is True
    assert fake_deps.generated_calls == [
        {
            "background_index": 2,
            "detector_shape": (2, 3),
            "ai": "fake-ai",
            "analysis_preview_bins": (4, 5),
            "allow_generated_payload": True,
        }
    ]


def test_worker_caked_view_payload_ready_requests_generated_payload() -> None:
    context, fake_deps = _context_with_caked_deps(
        {"background_images": {1: {"native": [[1.0]]}}}
    )
    fake_deps.generated_payload = _ready_projection_payload(detector_shape=(1, 1))

    assert context.worker_caked_view_payload_ready(1) is True
    assert fake_deps.generated_calls[0]["allow_generated_payload"] is True


def test_ensure_worker_geometry_fit_caked_view_noops_without_required_caked_backgrounds() -> None:
    context, _fake_deps = _context_with_caked_deps(
        {
            "required_indices": [0, 1],
            "manual_fit_space_by_background": {0: "detector", 1: "detector"},
        }
    )
    manual_deps = FakeManualFitSpaceDeps()
    context.manual_fit_space_deps = manual_deps.deps

    context.ensure_worker_geometry_fit_caked_view()


def test_ensure_worker_geometry_fit_caked_view_calls_mixed_space_rejection_first() -> None:
    context, _fake_deps = _context_with_caked_deps(
        {
            "required_indices": [0],
            "manual_fit_space_by_background": {0: "mixed"},
        }
    )
    manual_deps = FakeManualFitSpaceDeps()
    context.manual_fit_space_deps = manual_deps.deps

    try:
        context.ensure_worker_geometry_fit_caked_view()
    except RuntimeError as exc:
        assert str(exc) == (
            "mixed detector/caked manual fit spaces are not supported "
            "for background(s) 1; rebuild manual pairs in one fit space"
        )
    else:
        raise AssertionError("expected mixed manual fit-space rejection")
    assert manual_deps.caked_required_calls == [
        {
            "pairs": [],
            "manual_fit_space_kind": "mixed",
            "projection_view_mode": None,
            "pick_uses_caked_space": False,
            "pick_applies_to_background": True,
        }
    ]


def test_ensure_worker_geometry_fit_caked_view_checks_only_caked_required_backgrounds() -> None:
    context, _fake_deps = _context_with_caked_deps(
        {
            "required_indices": [0, 1],
            "manual_caked_fit_space_required_by_background": {0: False, 1: True},
            "background_images": {1: {"native": [[1.0]]}},
            "projection_payload_by_background": {
                1: _ready_projection_payload(detector_shape=(1, 1))
            },
        }
    )
    manual_deps = FakeManualFitSpaceDeps()
    context.manual_fit_space_deps = manual_deps.deps

    context.ensure_worker_geometry_fit_caked_view()


def test_ensure_worker_geometry_fit_caked_view_raises_with_one_based_background_label() -> None:
    context, _fake_deps = _context_with_caked_deps(
        {
            "required_indices": [2],
            "manual_caked_fit_space_required_by_background": {2: True},
            "background_images": {2: {"native": [[1.0]]}},
        }
    )
    manual_deps = FakeManualFitSpaceDeps()
    context.manual_fit_space_deps = manual_deps.deps

    try:
        context.ensure_worker_geometry_fit_caked_view()
    except RuntimeError as exc:
        assert str(exc) == "exact caked projector unavailable for background 3"
    else:
        raise AssertionError("expected exact caked projector failure")


def test_prebuild_required_background_caches_emits_start_and_bundle_start() -> None:
    queue = RecordingQueue()
    context, _deps = _context_with_required_cache_deps({"event_queue": queue})

    context.prebuild_required_background_caches(stage_callback="stage")

    assert _event_payloads(queue, "source_cache_build_start") == [
        {
            "background_index": 0,
            "background_label": "background 1",
            "elapsed_s": 0.0,
            "message": "preflight: building source cache for background 1",
        }
    ]
    bundle_start = _event_payloads(queue, "source_cache_bundle_start")
    assert len(bundle_start) == 1
    assert bundle_start[0]["background_index"] == 0
    assert bundle_start[0]["background_label"] == "background 1"
    assert bundle_start[0]["status"] == "starting"
    assert bundle_start[0]["required_pair_count"] == 1


def test_prebuild_required_background_caches_failed_bundle_emits_failure_status() -> None:
    queue = RecordingQueue()
    context, _deps = _context_with_required_cache_deps({"event_queue": queue})
    context.set_worker_source_snapshot_diagnostics(status="background_cache_empty")
    context.prebuild_background_cache_bundle_worker = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )

    context.prebuild_required_background_caches(stage_callback="stage")

    failures = _event_payloads(queue, "source_cache_bundle_failed")
    assert len(failures) == 1
    assert failures[0]["background_index"] == 0
    assert failures[0]["background_label"] == "background 1"
    assert failures[0]["status"] == "background_cache_empty"
    assert failures[0]["message"] == (
        "preflight: source cache bundle failed for background 1 "
        "(status=background_cache_empty)"
    )


def test_prebuild_required_background_caches_success_emits_ready_status() -> None:
    queue = RecordingQueue()
    context, _deps = _context_with_required_cache_deps({"event_queue": queue})

    context.prebuild_required_background_caches(stage_callback="stage")

    assert _event_payloads(queue, "source_cache_bundle_ready")[0]["status"] == "ready"
    assert _event_payloads(queue, "source_cache_rows_ready")[0]["status"] == "rows_ready"
    assert _event_payloads(queue, "source_cache_build_ready")[0]["status"] == "ready"


def test_prebuild_required_background_caches_preserves_source_cache_generation_id() -> None:
    queue = RecordingQueue()
    context, _deps = _context_with_required_cache_deps(
        {
            "event_queue": queue,
            "source_cache_generation_by_background": {0: 9},
        }
    )

    context.prebuild_required_background_caches(stage_callback="stage")

    payload = _event_payloads(queue, "source_cache_bundle_ready")[0]
    assert payload["source_cache_generation_id"] == 9


def test_prebuild_required_background_caches_locked_qr_readiness_event_fields() -> None:
    queue = RecordingQueue()
    context, deps = _context_with_required_cache_deps({"event_queue": queue})
    deps.readiness = {
        "expected_locked_qr_rows": 1,
        "projected_locked_qr_rows": 1,
        "finite_locked_qr_rows": 1,
        "fit_space_projection_ready": True,
        "projection_degenerate": False,
        "caked_view_storage_required_for_fit": True,
        "locked_qr_row_keys": [{"pair_id": "pair-1", "source_exists": True}],
    }

    context.prebuild_required_background_caches(stage_callback="stage")

    readiness = _event_payloads(queue, "locked_qr_projection_readiness")
    assert len(readiness) == 1
    assert readiness[0]["background_index"] == 0
    assert readiness[0]["projection_ready"] is True
    assert readiness[0]["caked_view_storage_status"] == "ready"
    assert readiness[0]["storage_timeout_fatal"] is False
    assert readiness[0]["full_cake_started"] is True
    assert readiness[0]["exact_caked_projection_payload_status"] == "missing"


def test_prebuild_required_background_caches_caked_timeout_event_fields() -> None:
    queue = RecordingQueue()
    context, deps = _context_with_required_cache_deps({"event_queue": queue})
    deps.caked_outcome = None

    def _start_non_gating_caked_view_task(
        _bundle: object,
        *,
        source_cache_generation_id: int,
        started_at: float,
        stage_callback: object = None,
    ) -> tuple[object, object]:
        def _await_result(_timeout_s: float) -> None:
            return None

        def _announce_timeout() -> None:
            deps.announced_timeouts += 1

        return _await_result, _announce_timeout

    context.start_non_gating_caked_view_task = (  # type: ignore[method-assign]
        _start_non_gating_caked_view_task
    )

    context.prebuild_required_background_caches(stage_callback="stage")

    assert deps.announced_timeouts == 1
    payload = _event_payloads(queue, "source_cache_caked_view_timeout")[0]
    assert payload["background_index"] == 0
    assert payload["background_label"] == "background 1"
    assert payload["source_cache_generation_id"] == 4
    assert payload["row_count"] == 1
    assert payload["status"] == "timeout"
    assert payload["message"] == (
        "preflight: caked view storage timeout for background 1 (waited 5.0s)"
    )
