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
