from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ra_sim.gui._runtime.geometry_fit_worker import (
    GeometryFitWorkerCakedPayloadDeps,
    GeometryFitWorkerContext,
    GeometryFitWorkerSourceProjectionDeps,
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


class FakeProjectionCallbacks:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, object]]] = []
        self.projected_rows: list[dict[str, object]] | None = None

    def project_peaks_to_current_view(
        self,
        rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        self.calls.append([dict(row) for row in rows])
        if self.projected_rows is not None:
            return [dict(row) for row in self.projected_rows]
        return [dict(row, projected=True) for row in rows]


class FakeSourceProjectionDeps:
    def __init__(self) -> None:
        self.callbacks = FakeProjectionCallbacks()

    @property
    def deps(self) -> GeometryFitWorkerSourceProjectionDeps:
        return GeometryFitWorkerSourceProjectionDeps(
            rows_for_background=self.rows_for_background,
            group_rows_by_background=self.group_rows_by_background,
            make_projection_callbacks=self.make_projection_callbacks,
            native_detector_coords_to_bundle_detector_coords=(
                self.native_detector_coords_to_bundle_detector_coords
            ),
            raw_phi_to_gui_phi=lambda value: float(value),
            rotate_point_for_display=lambda col, row: (float(col), float(row)),
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

    def native_detector_coords_to_bundle_detector_coords(
        self,
        col: float,
        row: float,
        _detector_shape: object,
    ) -> tuple[float, float]:
        return float(col), float(row)


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


def test_project_source_rows_for_background_preserves_row_order() -> None:
    context, _fake_deps = _context_with_source_projection_deps()
    rows = [
        _source_row(source_row_index=1, hkl=(1, 0, 0)),
        _source_row(source_row_index=2, hkl=(2, 0, 0)),
    ]

    projected = context.project_source_rows_for_background(0, rows)

    assert [row["source_row_index"] for row in projected] == [1, 2]
    assert all(row["projected"] is True for row in projected)


def test_project_source_rows_for_background_preserves_source_locator_identity() -> None:
    context, _fake_deps = _context_with_source_projection_deps()
    row = _source_row(
        q_group_key=("q_group", "primary", 3, 10),
        source_row_index=99,
    )

    projected = context.project_source_rows_for_background(0, [row])

    assert projected[0]["q_group_key"] == ("q_group", "primary", 3, 10)
    assert projected[0]["source_row_index"] == 99


def test_project_source_rows_for_background_preserves_reflection_identity() -> None:
    context, _fake_deps = _context_with_source_projection_deps()
    row = _source_row(
        hkl=(-2, 1, 5),
        normalized_hkl=(-2, 1, 5),
        source_reflection_index=11,
        source_reflection_namespace="full_reflection",
        source_reflection_is_full=True,
    )

    projected = context.project_source_rows_for_background(0, [row])

    assert projected[0]["hkl"] == (-2, 1, 5)
    assert projected[0]["normalized_hkl"] == (-2, 1, 5)
    assert projected[0]["source_reflection_index"] == 11
    assert projected[0]["source_reflection_namespace"] == "full_reflection"
    assert projected[0]["source_reflection_is_full"] is True


def test_project_source_rows_for_background_preserves_locked_qr_qz_identity() -> None:
    context, _fake_deps = _context_with_source_projection_deps()
    row = _source_row(locked_qr=1.5, locked_qz=2.75)

    projected = context.project_source_rows_for_background(0, [row])

    assert projected[0]["locked_qr"] == 1.5
    assert projected[0]["locked_qz"] == 2.75


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


def test_project_source_rows_by_row_background_groups_by_same_background_key() -> None:
    context, fake_deps = _context_with_source_projection_deps()
    rows = [
        _source_row(background_index=0, source_row_index=1),
        _source_row(background_index=1, source_row_index=2),
    ]

    projected = context.project_source_rows_by_row_background(rows)

    assert [row["background_index"] for row in projected] == [0, 1]
    assert [row["source_row_index"] for row in projected] == [1, 2]
    assert len(fake_deps.callbacks.calls) == 2


def test_project_source_rows_by_row_background_preserves_group_order() -> None:
    context, _fake_deps = _context_with_source_projection_deps()
    rows = [
        _source_row(background_index=1, source_row_index=1),
        _source_row(background_index=0, source_row_index=2),
        _source_row(background_index=1, source_row_index=3),
    ]

    projected = context.project_source_rows_by_row_background(rows)

    assert [row["source_row_index"] for row in projected] == [1, 2, 3]


def test_project_source_rows_by_row_background_preserves_locked_qr_qz_fields() -> None:
    context, _fake_deps = _context_with_source_projection_deps()
    rows = [
        _source_row(background_index=1, source_row_index=1, locked_qr=4.0, locked_qz=5.0),
        _source_row(background_index=0, source_row_index=2, locked_qr=6.0, locked_qz=7.0),
    ]

    projected = context.project_source_rows_by_row_background(rows)

    assert [(row["locked_qr"], row["locked_qz"]) for row in projected] == [
        (4.0, 5.0),
        (6.0, 7.0),
    ]


def test_project_source_rows_by_row_background_does_not_merge_distinct_background_groups() -> None:
    context, fake_deps = _context_with_source_projection_deps()
    rows = [
        _source_row(background_index=1, source_row_index=1),
        _source_row(background_index=0, source_row_index=2),
    ]

    context.project_source_rows_by_row_background(rows)

    assert [
        [row["background_index"] for row in call]
        for call in fake_deps.callbacks.calls
    ] == [[1], [0]]


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
