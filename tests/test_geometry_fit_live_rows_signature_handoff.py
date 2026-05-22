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
    monkeypatch.setattr(
        runtime_session, "_live_cache_inventory_snapshot", lambda: {}, raising=False
    )
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
    assert job["live_rows_cache_metadata_by_background"][0]["live_rows_raw_count"] == 2
    assert job["live_rows_cache_metadata_by_background"][0]["live_rows_source_counts"] == {
        "disordered_phase": 1,
        "primary": 1,
    }
    assert job["live_rows_signature_by_background"][0] == ("requested", 0)
    assert job["live_rows_handoff_diagnostics"]["live_rows_current_background_count"] == 2


def test_worker_live_rows_payload_uses_background_requested_signature(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    job = _worker_job(runtime_session)
    _patch_worker_prepare(monkeypatch, runtime_session)
    job["live_rows_by_background"] = {
        0: [
            {
                **_live_row("primary"),
                "background_two_theta_deg": 1.0,
                "background_phi_deg": 2.0,
                "caked_x": 1.0,
                "caked_y": 2.0,
            }
        ]
    }
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


def test_runtime_session_preserves_locked_qr_dynamic_payload_fields() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_fit = runtime_session.gui_geometry_fit
    hkl = (-1, 0, 10)
    q_group_key = ("q_group", "primary", 1, 10)
    observed_native = (1083.734, 1152.380)
    observed_caked = (37.566, 39.750)
    clean_prediction_native = (1080.0, 1139.0)
    clean_prediction_caked = (38.168, 39.250)
    stale_prediction_native = (1862.136, 1077.417)
    stale_prediction_caked = (41.225, -38.250)
    identity = {
        "normalized_hkl": hkl,
        "source_table_index": 0,
        "source_reflection_index": 0,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 12,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "q_group_key": q_group_key,
        "source_label": "primary",
    }
    provider_pair = {
        "pair_index": 0,
        "provider_pair_index": 0,
        "dataset_pair_index": 0,
        "background_index": 0,
        "q_group_key": q_group_key,
        "hkl": hkl,
        "normalized_hkl": hkl,
        "source_branch_index": 1,
        "selected_source_identity_canonical": dict(identity),
        "background_point": observed_native,
        "background_frame": "detector_native",
        "solver_measured_point": observed_native,
        "solver_measured_frame": "detector_native",
        "simulated_point": clean_prediction_caked,
        "simulated_frame": "caked_2theta_phi",
        "simulated_point_source": "manual_picker_saved",
    }
    measured_row = {
        "x": observed_native[0],
        "y": observed_native[1],
        "hkl": hkl,
        "q_group_key": q_group_key,
        "background_two_theta_deg": observed_caked[0],
        "background_phi_deg": observed_caked[1],
        "fit_observed_detector_native_px": observed_native,
        "fit_observed_caked_deg": observed_caked,
        **identity,
    }
    initial_row = {
        "hkl": hkl,
        "q_group_key": q_group_key,
        "sim_native": stale_prediction_native,
        "sim_native_source": "stale_hit_table_display_mislabeled_native",
        "simulated_two_theta_deg": stale_prediction_caked[0],
        "simulated_phi_deg": stale_prediction_caked[1],
        **identity,
    }
    audit_row = {
        "pair_index": 0,
        "hkl": hkl,
        "q_group_key": q_group_key,
        "source_branch_index": 1,
        "fit_observed_detector_native_px": observed_native,
        "fit_observed_caked_deg": observed_caked,
        "observed_caked_authority": "dynamic_trial_projection_from_observed_native",
        "fit_prediction_detector_native_px": clean_prediction_native,
        "fit_prediction_caked_deg": clean_prediction_caked,
        "predicted_caked_deg": clean_prediction_caked,
        "sim_refined_caked_deg": clean_prediction_caked,
        "fit_prediction_caked_authority": "dynamic_trial_projection_from_prediction_native",
        "sim_refined_caked_authority": "dynamic_trial_projection_from_prediction_native",
    }
    dataset = {
        "dataset_index": 0,
        "pair_count": 1,
        "provider_pairs": [provider_pair],
        "manual_point_pairs": [dict(provider_pair)],
        "measured_for_fit": [measured_row],
        "initial_pairs_display": [initial_row],
        "fit_handoff_audit_rows": [audit_row],
    }

    rows, summary = geometry_fit._build_geometry_fit_optimizer_request_rows(
        prepared_run=SimpleNamespace(current_dataset=dataset),
        solver_inputs=SimpleNamespace(
            miller=np.asarray([hkl], dtype=np.int64),
            intensities=np.ones(1, dtype=np.float64),
            image_size=4000,
        ),
    )

    assert summary["fixed_source_pair_count"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["fit_observed_detector_native_px"] == pytest.approx(observed_native)
    assert row["fit_observed_caked_deg"] == pytest.approx(observed_caked)
    assert row["fit_prediction_detector_native_px"] == pytest.approx(clean_prediction_native)
    assert row["fit_prediction_caked_deg"] == pytest.approx(clean_prediction_caked)
    assert row["sim_refined_caked_deg"] == pytest.approx(clean_prediction_caked)
    assert row["fit_prediction_caked_authority"] == (
        "dynamic_trial_projection_from_prediction_native"
    )
    assert row["observed_caked_authority"] == "dynamic_trial_projection_from_observed_native"


def test_optimizer_request_preserves_handoff_payload_for_multiple_locked_qr_groups() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    geometry_fit = runtime_session.gui_geometry_fit
    groups = [
        (("q_group", "primary", 1, 5), (-1, 0, 5)),
        (("q_group", "primary", 1, 10), (-1, 0, 10)),
    ]
    provider_pairs: list[dict[str, object]] = []
    manual_pairs: list[dict[str, object]] = []
    measured_rows: list[dict[str, object]] = []
    initial_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    expected_clean_caked: dict[tuple[object, ...], tuple[float, float]] = {}
    expected_native: dict[tuple[object, ...], tuple[float, float]] = {}

    def identity_key(
        q_group_key: tuple[object, ...],
        hkl: tuple[int, int, int],
        branch: int,
        source_row_index: int,
    ) -> tuple[object, ...]:
        return tuple(q_group_key), tuple(hkl), int(branch), int(source_row_index)

    for group_index, (q_group_key, hkl) in enumerate(groups):
        for branch in (0, 1):
            row_index = group_index * 10 + branch
            identity = {
                "normalized_hkl": hkl,
                "hkl": hkl,
                "source_table_index": group_index,
                "source_reflection_index": group_index,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": row_index,
                "source_branch_index": branch,
                "source_peak_index": branch,
                "q_group_key": q_group_key,
                "source_label": "primary",
            }
            observed_native = (1080.0 + row_index, 1100.0 + row_index)
            observed_caked = (30.0 + group_index, 40.0 + branch)
            clean_prediction_native = (observed_native[0] + 1.0, observed_native[1] + 2.0)
            clean_prediction_caked = (observed_caked[0] + 0.5, observed_caked[1] + 0.25)
            stale_prediction_native = (1800.0 + row_index, 900.0 + row_index)
            stale_prediction_caked = (80.0 + group_index * 20.0, -120.0 + branch * 70.0)
            key = identity_key(q_group_key, hkl, branch, row_index)
            expected_clean_caked[key] = clean_prediction_caked
            expected_native[key] = clean_prediction_native
            provider_pair = {
                "pair_index": len(provider_pairs),
                "provider_pair_index": len(provider_pairs),
                "dataset_pair_index": len(provider_pairs),
                "background_index": 0,
                "selected_source_identity_canonical": dict(identity),
                "background_point": observed_native,
                "background_frame": "detector_native",
                "solver_measured_point": observed_native,
                "solver_measured_frame": "detector_native",
                "simulated_point": clean_prediction_caked,
                "simulated_frame": "caked_2theta_phi",
                "simulated_point_source": "manual_picker_saved",
                "fit_source_resolution_kind": "provider_fixed_source_local",
                "optimizer_request_has_fixed_source": True,
                **identity,
            }
            provider_pairs.append(provider_pair)
            manual_pairs.append(dict(provider_pair))
            measured_rows.append(
                {
                    "x": observed_native[0],
                    "y": observed_native[1],
                    "background_two_theta_deg": observed_caked[0],
                    "background_phi_deg": observed_caked[1],
                    "fit_observed_detector_native_px": observed_native,
                    "fit_observed_caked_deg": observed_caked,
                    **identity,
                }
            )
            initial_rows.append(
                {
                    "sim_native": stale_prediction_native,
                    "sim_native_source": "stale_hit_table_display_mislabeled_native",
                    "simulated_two_theta_deg": stale_prediction_caked[0],
                    "simulated_phi_deg": stale_prediction_caked[1],
                    **identity,
                }
            )
            audit_rows.append(
                {
                    "hkl": hkl,
                    "q_group_key": q_group_key,
                    "source_table_index": group_index,
                    "source_reflection_index": group_index,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_row_index": row_index,
                    "source_branch_index": branch,
                    "source_peak_index": branch,
                    "source_label": "primary",
                    "fit_observed_detector_native_px": observed_native,
                    "fit_observed_caked_deg": observed_caked,
                    "observed_caked_authority": "dynamic_trial_projection_from_observed_native",
                    "fit_prediction_detector_native_px": clean_prediction_native,
                    "fit_prediction_caked_deg": clean_prediction_caked,
                    "predicted_caked_deg": clean_prediction_caked,
                    "sim_refined_caked_deg": clean_prediction_caked,
                    "fit_prediction_caked_authority": (
                        "dynamic_trial_projection_from_prediction_native"
                    ),
                    "sim_refined_caked_authority": (
                        "dynamic_trial_projection_from_prediction_native"
                    ),
                    "sim_nominal_caked_deg": stale_prediction_caked,
                    "simulated_two_theta_deg": stale_prediction_caked[0],
                    "simulated_phi_deg": stale_prediction_caked[1],
                }
            )

    dataset = {
        "dataset_index": 0,
        "pair_count": len(provider_pairs),
        "provider_pairs": provider_pairs,
        "manual_point_pairs": manual_pairs,
        "measured_for_fit": measured_rows,
        "initial_pairs_display": initial_rows,
        "fit_handoff_audit_rows": list(reversed(audit_rows)),
    }

    rows, summary = geometry_fit._build_geometry_fit_optimizer_request_rows(
        prepared_run=SimpleNamespace(current_dataset=dataset),
        solver_inputs=SimpleNamespace(
            miller=np.asarray([group[1] for group in groups], dtype=np.int64),
            intensities=np.ones(len(groups), dtype=np.float64),
            image_size=4000,
        ),
    )

    assert summary["fixed_source_pair_count"] == 4
    assert len(rows) == 4
    for row in rows:
        key = identity_key(
            tuple(row["q_group_key"]),
            tuple(row["hkl"]),
            int(row["source_branch_index"]),
            int(row["source_row_index"]),
        )
        assert row["fit_prediction_detector_native_px"] == pytest.approx(expected_native[key])
        assert row["fit_prediction_caked_deg"] == pytest.approx(expected_clean_caked[key])
        assert row["fit_prediction_caked_authority"] in {
            "dynamic_trial_projection_from_prediction_native",
            "exact_projector_from_native",
            "sim_refined_caked_matching_prediction_native",
        }
        assert row["fit_prediction_caked_authority"] not in {
            "sim_nominal_caked",
            "simulated_two_theta_phi",
            "saved_handoff_caked",
            "unknown",
        }
