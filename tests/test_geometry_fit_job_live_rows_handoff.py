from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np

from ra_sim.gui import geometry_fit


def _live_row(source_label: str = "primary", *, x: float = 10.0) -> dict[str, object]:
    return {
        "hkl": (1, 0, 0),
        "q_group_key": ("q_group", source_label, 1, 0),
        "source_label": source_label,
        "phase_label": "Disordered phase" if source_label == "disordered_phase" else "Primary",
        "structure_role": "disordered" if source_label == "disordered_phase" else "primary",
        "source_reflection_index": 7 if source_label == "primary" else 8,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_table_index": 0 if source_label == "primary" else 1,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": float(x),
        "sim_row": float(x) + 1.0,
        "display_col": float(x),
        "display_row": float(x) + 1.0,
        "detector_display_x": float(x),
        "detector_display_y": float(x) + 1.0,
        "qr": 1.2,
        "qz": 0.4,
        "gz_index": 0,
    }


def _required_pair(source_label: str = "primary") -> dict[str, object]:
    row = _live_row(source_label)
    row.update(
        {
            "pair_id": f"bg0:{source_label}:pair0",
            "overlay_match_index": 0,
        }
    )
    return row


def _worker_job(runtime_session) -> dict[str, object]:
    return {
        "job_id": 201,
        "event_queue": runtime_session.queue.Queue(),
        "params": {
            "center": [1.0, 1.0],
            "corto_detector": 0.5,
            "pixel_size_m": 1.0e-4,
            "lambda": 1.54e-10,
            "theta_initial": 0.0,
        },
        "var_names": [],
        "preserve_live_theta": False,
        "source_snapshots": {},
        "source_snapshot_diagnostics": {},
        "simulation_diagnostics": {},
        "background_images": {
            0: {
                "native": np.ones((4, 4), dtype=np.float64),
                "display": np.ones((4, 4), dtype=np.float64),
            }
        },
        "requested_signatures": {0: ("requested", 0)},
        "requested_signature_summaries": {0: "requested-0"},
        "background_labels": {0: "bg0.osc"},
        "live_rows_signature": ("global", "stale"),
        "live_rows_signature_by_background": {0: ("requested", 0)},
        "live_rows_by_background": {0: []},
        "live_rows_cache_metadata_by_background": {
            0: {
                "cache_source": "empty",
                "geometry_fit_live_handoff_patch_marker": "phase4d1",
            }
        },
        "memory_intersection_cache": [],
        "memory_intersection_cache_signature": ("requested", 0),
        "manual_pairs_by_background": {0: [_required_pair("primary")]},
        "required_indices": [0],
        "current_background_index": 0,
        "image_size": 4,
        "theta_initial": 0.0,
        "theta_initial_by_background": {0: 0.0},
        "theta_base_by_background": {0: 0.0},
        "geometry_runtime_cfg": {},
        "fit_config": {},
        "live_cache_inventory": {"source_snapshot_count": 1},
        "solver_inputs": SimpleNamespace(miller=[], intensities=[], image_size=4),
        "selected_background_indices": [0],
        "joint_background_mode": False,
        "osc_files": ["bg0.osc"],
        "selection_applied": True,
        "theta_metadata_applied": True,
        "background_theta_values": [0.0],
        "theta_offset": 0.0,
        "uses_shared_theta": False,
        "projection_view_mode": "detector",
        "projection_view_signature": {"mode": "detector", "detector_shape": [4, 4]},
        "projection_view_signature_by_background": {
            0: {"mode": "detector", "detector_shape": [4, 4]}
        },
        "projection_payload_by_background": {},
        "stamp": "20260420_000000",
        "log_path": "artifacts/geometry_fit_worker_test.log",
        "enable_live_update_events": False,
    }


def _drain_events(event_queue) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while True:
        try:
            events.append(event_queue.get_nowait())
        except Exception:
            return events


def _patch_worker_prepare(monkeypatch, runtime_session) -> None:
    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            geometry_fit_targeted_projected_cache_by_background={},
            geometry_fit_caking_ai_cache={},
            analysis_preview_bins=(4, 4),
            source_row_snapshots={},
            last_simulation_signature=None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session.gui_geometry_fit,
        "prepare_geometry_fit_run",
        lambda **_kwargs: runtime_session.gui_geometry_fit.GeometryFitPreparationResult(),
    )


def _bindings(
    monkeypatch,
    runtime_session,
    tmp_path,
    *,
    q_group_entries=None,
    picker_cache=None,
    selected_indices=None,
    shared_theta=False,
):
    resolved_indices = [int(idx) for idx in (selected_indices or [0])]
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=[f"bg{idx}.osc" for idx in resolved_indices],
        current_background_index=0,
        image_size=4,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda _idx: [_required_pair("primary")],
        load_background_by_index=lambda _idx: (
            np.ones((4, 4), dtype=np.float64),
            np.ones((4, 4), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        geometry_manual_simulated_lookup=lambda _rows: {},
        geometry_manual_entry_display_coords=lambda _entry: None,
        unrotate_display_peaks=lambda entries, shape, *, k: list(entries),
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda *_args, **_kwargs: ({}, {"pairs": 0}),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: False,
    )
    prepare_bindings = geometry_fit.GeometryFitRuntimePreparationBindings(
        fit_config={"geometry": {"solver": {"dynamic_point_geometry_fit": False}}},
        theta_initial=0.0,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: list(resolved_indices),
        geometry_fit_uses_shared_theta_offset=lambda _indices: bool(shared_theta),
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0 for _idx in resolved_indices],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        ensure_geometry_fit_caked_view=lambda: None,
        manual_dataset_bindings=manual_dataset_bindings,
        build_runtime_config=lambda _params: {"solver": {"dynamic_point_geometry_fit": False}},
    )
    execution_bindings = SimpleNamespace(
        downloads_dir=tmp_path,
        log_dir=tmp_path,
        simulation_runtime_state=SimpleNamespace(
            geometry_fit_job_counter=0,
            geometry_fit_event_queue=runtime_session.queue.Queue(),
            source_row_snapshots={},
            last_simulation_signature=("stale", "global"),
            analysis_preview_bins=(4, 4),
        ),
        solver_inputs=SimpleNamespace(miller=[], intensities=[], image_size=4),
        background_runtime_state=SimpleNamespace(current_background_index=0),
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_q_group_state",
        SimpleNamespace(cached_entries=list(q_group_entries or ())),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "geometry_runtime_state",
        SimpleNamespace(
            manual_pick_cache_data=dict(picker_cache or {}),
            manual_pick_cache_signature=None,
        ),
        raising=False,
    )
    return SimpleNamespace(
        value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
            current_var_names=lambda: [],
            current_params=lambda: {
                "theta_initial": 0.0,
                "center": [1.0, 1.0],
                "corto_detector": 0.5,
                "lambda": 1.54e-10,
            },
            current_ui_params=lambda: {},
            var_map={},
            build_mosaic_params=lambda **_kwargs: {},
        ),
        prepare_bindings_factory=lambda _var_names: prepare_bindings,
        execution_bindings=execution_bindings,
        solve_fit=lambda *_args, **_kwargs: None,
        stamp_factory=lambda: "20260422_000000",
    )


def _patch_job_build(monkeypatch, runtime_session, *, live_rows=None) -> list[tuple[str, dict]]:
    traces: list[tuple[str, dict]] = []
    monkeypatch.setattr(runtime_session, "_geometry_fit_background_label", lambda idx: "bg0.osc")
    monkeypatch.setattr(
        runtime_session,
        "_geometry_source_snapshot_signature_for_background",
        lambda idx, params: ("requested", int(idx)),
    )
    monkeypatch.setattr(runtime_session, "_live_cache_signature_summary", lambda sig: repr(sig))
    monkeypatch.setattr(
        runtime_session,
        "_geometry_fit_targeted_projection_view_signature",
        lambda background_index, **kwargs: {
            "background_index": int(background_index),
            "mode": kwargs.get("mode_override") or "detector",
            "available": True,
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_build_live_preview_simulated_peaks_from_cache",
        lambda: [dict(row) for row in (live_rows or ())],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_last_live_preview_cache_metadata",
        lambda: {"cache_source": "empty" if not live_rows else "stored_hit_tables"},
        raising=False,
    )
    monkeypatch.setattr(runtime_session, "_live_cache_inventory_snapshot", lambda: {}, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_last_source_snapshot_diagnostics",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_last_simulation_diagnostics",
        lambda: {},
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: traces.append((str(event), dict(fields))),
        raising=False,
    )
    return traces


def test_active_preflight_marker_appears_in_job_trace(monkeypatch, tmp_path) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    traces = _patch_job_build(monkeypatch, runtime_session, live_rows=[_live_row("primary")])

    job = runtime_session._build_geometry_fit_async_job(
        _bindings(monkeypatch, runtime_session, tmp_path)
    )

    assert job["live_rows_handoff_diagnostics"]["geometry_fit_live_handoff_patch_marker"] == (
        "phase4d1"
    )
    assert any(
        event == "geometry_fit_job_live_rows_build"
        and fields.get("geometry_fit_live_handoff_patch_marker") == "phase4d1"
        for event, fields in traces
    )


def test_geometry_fit_job_logs_live_rows_build_counts(monkeypatch, tmp_path) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    traces = _patch_job_build(monkeypatch, runtime_session, live_rows=[_live_row("primary")])

    runtime_session._build_geometry_fit_async_job(
        _bindings(monkeypatch, runtime_session, tmp_path)
    )
    fields = next(
        fields for event, fields in traces if event == "geometry_fit_job_live_rows_build"
    )

    assert fields["current_background"] == 0
    assert fields["live_preview_rows_count"] == 1
    assert fields["live_rows_by_background_current_count"] == 1
    assert fields["live_rows_by_background_keys"] == [0]
    assert fields["requested_signature_keys"] == [0]
    assert fields["live_rows_signature_by_background_keys"] == [0]


def test_geometry_fit_job_uses_q_group_snapshot_when_live_rows_empty(
    monkeypatch,
    tmp_path,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_job_build(monkeypatch, runtime_session, live_rows=[])
    q_group_entries = [
        {"key": ("q_group", "primary", 1, 0), "source_label": "primary", "rows": [_live_row()]},
    ]

    job = runtime_session._build_geometry_fit_async_job(
        _bindings(monkeypatch, runtime_session, tmp_path, q_group_entries=q_group_entries)
    )

    assert len(job["live_rows_by_background"][0]) == 1
    assert job["live_rows_cache_metadata_by_background"][0]["job_local_fallback_source"] == (
        "q_group_snapshot"
    )
    assert job["live_rows_signature_by_background"][0] == ("requested", 0)


def test_geometry_fit_job_builds_q_group_rows_for_noncurrent_required_background(
    monkeypatch,
    tmp_path,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    _patch_job_build(monkeypatch, runtime_session, live_rows=[])
    q_group_entries = [
        {
            "key": ("q_group", "primary", 1, 0),
            "source_label": "primary",
            "rows": [_live_row("primary", x=25.0)],
        },
    ]

    job = runtime_session._build_geometry_fit_async_job(
        _bindings(
            monkeypatch,
            runtime_session,
            tmp_path,
            q_group_entries=q_group_entries,
            selected_indices=[0, 1],
            shared_theta=True,
        )
    )

    assert sorted(job["live_rows_by_background"]) == [0, 1]
    assert len(job["live_rows_by_background"][1]) == 1
    assert job["live_rows_cache_metadata_by_background"][1]["job_local_fallback_source"] == (
        "q_group_snapshot"
    )
    assert job["live_rows_signature_by_background"][1] == ("requested", 1)


def test_geometry_fit_job_snapshot_preserves_disordered_source_rows(
    monkeypatch,
    tmp_path,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    traces = _patch_job_build(monkeypatch, runtime_session, live_rows=[])
    q_group_entries = [
        {
            "key": ("q_group", "primary", 1, 0),
            "source_label": "primary",
            "rows": [_live_row("primary", x=10.0)],
        },
        {
            "key": ("q_group", "disordered_phase", 1, 0),
            "source_label": "disordered_phase",
            "phase_label": "Disordered phase",
            "structure_role": "disordered",
            "rows": [_live_row("disordered_phase", x=50.0)],
        },
    ]

    job = runtime_session._build_geometry_fit_async_job(
        _bindings(monkeypatch, runtime_session, tmp_path, q_group_entries=q_group_entries)
    )

    rows = job["live_rows_by_background"][0]
    assert {row["source_label"] for row in rows} == {"primary", "disordered_phase"}
    assert any(row["q_group_key"] == ("q_group", "disordered_phase", 1, 0) for row in rows)
    assert any(
        event == "geometry_fit_job_live_rows_from_q_group_snapshot"
        and fields.get("sources") == {"disordered_phase": 1, "primary": 1}
        for event, fields in traces
    )


def test_actual_fresh_fallback_call_dedupes_consumer_kwarg(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _worker_job(runtime_session)
    _patch_worker_prepare(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: [np.zeros((1, 7), dtype=float)],
        raising=False,
    )

    calls: list[str | None] = []

    def _fake_build_rows(hit_tables, **kwargs):
        calls.append(kwargs.get("consumer"))
        return [_live_row("primary")], [], list(hit_tables or ()), [7]

    monkeypatch.setattr(
        runtime_session,
        "_geometry_manual_build_source_rows_from_hit_tables",
        _fake_build_rows,
        raising=False,
    )

    result = runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_events(job["event_queue"])
    kinds = [str(event.get("kind")) for event in events]

    assert "got multiple values for keyword argument 'consumer'" not in str(result.error_text)
    assert calls == [None]
    assert "source_cache_fresh_rebuild_consumer_wrapper" in kinds
    wrapper_payload = next(
        dict(event.get("payload") or {})
        for event in events
        if str(event.get("kind")) == "source_cache_fresh_rebuild_consumer_wrapper"
    )
    assert wrapper_payload["fresh_rebuild_consumer_wrapper"] == "deduped"
