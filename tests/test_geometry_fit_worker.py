from __future__ import annotations

import numpy as np

from ra_sim.gui._runtime.geometry_fit_worker import GeometryFitWorkerContext


class RecordingQueue:
    def __init__(self) -> None:
        self.items: list[object] = []

    def put(self, item: object) -> None:
        self.items.append(item)


class FailingQueue:
    def put(self, _item: object) -> None:
        raise RuntimeError("queue unavailable")


def test_worker_context_from_job_copies_source_snapshots() -> None:
    source_snapshots = {0: {"rows": [{"source": "cached"}]}}

    context = GeometryFitWorkerContext.from_job({"source_snapshots": source_snapshots})

    source_snapshots[0]["rows"][0]["source"] = "mutated"
    assert context.worker_source_row_snapshots[0]["rows"][0]["source"] == "cached"


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
        {"source_cache_generation_by_background": {2: 5}}
    )

    assert context.advance_source_cache_generation(2) == 6
    assert context.source_cache_generation_by_background[2] == 6
    assert context.job_data["source_cache_generation_by_background"] == {2: 6}


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
