from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def _live_row(source_label: str = "primary") -> dict[str, object]:
    return {
        "hkl": (1, 0, 0),
        "q_group_key": ("q_group", source_label, 1, 0),
        "source_label": source_label,
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }


def _required_pair(source_label: str = "primary") -> dict[str, object]:
    return {
        "pair_id": f"bg0:{source_label}:pair0",
        "overlay_match_index": 0,
        "hkl": (1, 0, 0),
        "q_group_key": ("q_group", source_label, 1, 0),
        "source_label": source_label,
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }


def _worker_job(runtime_session) -> dict[str, object]:
    return {
        "job_id": 101,
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
        "live_rows_by_background": {0: [_live_row("primary")]},
        "live_rows_cache_metadata_by_background": {
            0: {
                "cache_source": "stored_hit_tables",
                "live_rows_source_counts": {"primary": 1},
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


def test_geometry_fit_job_includes_live_rows_for_current_background(monkeypatch, tmp_path) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_fit = runtime_session.gui_geometry_fit
    live_rows = [_live_row("primary"), _live_row("disordered_phase")]
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["bg0.osc"],
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
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda _indices: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0],
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
    bindings = SimpleNamespace(
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
        lambda: [dict(row) for row in live_rows],
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "_last_live_preview_cache_metadata",
        lambda: {"cache_source": "stored_hit_tables"},
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

    job = runtime_session._build_geometry_fit_async_job(bindings)

    assert list(job["live_rows_by_background"]) == [0]
    assert len(job["live_rows_by_background"][0]) == 2
    assert job["live_rows_signature_by_background"][0] == ("requested", 0)
    assert job["live_rows_handoff_diagnostics"]["live_rows_current_background_count"] == 2


def test_worker_live_rows_payload_uses_background_requested_signature(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _worker_job(runtime_session)
    _patch_worker_prepare(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: pytest.fail("accepted live rows must avoid fresh simulation"),
        raising=False,
    )

    result = runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_events(job["event_queue"])
    kinds = [str(event.get("kind")) for event in events]

    assert result.error_text is None
    assert "source_cache_live_runtime_cache_accepted" in kinds
    assert "source_cache_targeted_fresh_simulation_start" not in kinds
    ready_payload = next(
        dict(event.get("payload") or {})
        for event in events
        if str(event.get("kind")) == "source_cache_live_runtime_cache_validation_ready"
    )
    assert ready_payload["live_rows_signature_match"] is True
    assert ready_payload["live_rows_raw_count"] == 1


def test_worker_live_rows_payload_reports_signature_mismatch_before_dropping_rows(
    monkeypatch,
) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _worker_job(runtime_session)
    job["live_rows_signature_by_background"] = {0: ("other", 0)}
    _patch_worker_prepare(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: [],
        raising=False,
    )

    runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_events(job["event_queue"])
    ready_payload = next(
        dict(event.get("payload") or {})
        for event in events
        if str(event.get("kind")) == "source_cache_live_runtime_cache_validation_ready"
    )

    assert ready_payload["row_count"] == 0
    assert ready_payload["live_rows_raw_count"] == 1
    assert ready_payload["live_rows_signature_match"] is False
    assert ready_payload["reason"] == "requested_signature_mismatch"


def test_geometry_fit_uses_live_preview_rows_before_fresh_simulation(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _worker_job(runtime_session)
    _patch_worker_prepare(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: pytest.fail("fresh simulation should not run"),
        raising=False,
    )

    runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_events(job["event_queue"])
    kinds = [str(event.get("kind")) for event in events]

    assert "source_cache_live_runtime_cache_accepted" in kinds
    assert not any("fresh_simulation_start" in kind for kind in kinds)


def test_live_rows_accepted_when_job_local_fallback_populates_rows(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _worker_job(runtime_session)
    job["live_rows_by_background"] = {0: [_live_row("primary")]}
    job["live_rows_signature_by_background"] = {0: ("requested", 0)}
    job["live_rows_cache_metadata_by_background"] = {
        0: {
            "cache_source": "q_group_snapshot",
            "geometry_fit_live_handoff_patch_marker": "phase4d1",
            "job_local_fallback_rows": 1,
        }
    }
    _patch_worker_prepare(monkeypatch, runtime_session)
    monkeypatch.setattr(
        runtime_session,
        "_simulate_hit_tables_for_fit",
        lambda *_args, **_kwargs: pytest.fail("job-local live rows should avoid fresh simulation"),
        raising=False,
    )

    runtime_session._run_async_geometry_fit_worker_job(job)
    events = _drain_events(job["event_queue"])
    kinds = [str(event.get("kind")) for event in events]
    ready_payload = next(
        dict(event.get("payload") or {})
        for event in events
        if str(event.get("kind")) == "source_cache_live_runtime_cache_validation_ready"
    )

    assert "source_cache_live_runtime_cache_accepted" in kinds
    assert ready_payload["live_rows_raw_count"] == 1
    assert ready_payload["live_rows_signature_match"] is True
    assert ready_payload["geometry_fit_live_handoff_patch_marker"] == "phase4d1"
