"""Internal geometry-fit worker context helpers."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


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
