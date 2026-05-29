from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ra_sim.gui._runtime.geometry_fit_worker import (
    GeometryFitWorkerCakedPayloadDeps,
    GeometryFitWorkerContext,
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


class FakeCakedPayloadDeps:
    def __init__(self) -> None:
        self.generated_payload: dict[str, object] | None = None
        self.generated_calls: list[dict[str, object]] = []
        self.stage_events: list[tuple[object, str, dict[str, object]]] = []
        self.hydrate_calls: list[dict[str, object]] = []
        self.normalize_calls: list[dict[str, object]] = []

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
        return "fake-ai"

    def emit_stage_event(
        self,
        stage_callback: object,
        kind: str,
        **payload: object,
    ) -> None:
        self.stage_events.append((stage_callback, kind, payload))

    def is_transform_bundle(self, value: object) -> bool:
        return isinstance(value, FakeBundle)


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


# D2 contract map: caked payload helpers return ready/invalid/absent status
# strings, mutate only job-local caked/projection payload maps, short-circuit
# invalid existing payloads before generated fallback, and emit stage diagnostics
# with the same status/kind/message fields as the old nested helpers.


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


def test_worker_context_advance_source_cache_generation_preserves_other_backgrounds() -> None:
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
