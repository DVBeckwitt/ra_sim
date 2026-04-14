from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def _load_baseline_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug"
        / "run_geometry_fit_quality_baseline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_geometry_fit_quality_baseline",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_quality_report_extracts_decision_row_and_overlay_preview(tmp_path: Path) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new2_fresh_all.json"
    out_state_path = tmp_path / "new2_fresh_all_fit.json"
    log_path = tmp_path / "geometry_fit_log_20260414_120000.txt"
    trace_path = tmp_path / "geometry_fit_trace_20260414_120000.jsonl"
    matched_peaks_path = tmp_path / "matched_peaks_20260414_120000.npy"
    cli_stdout_path = tmp_path / "cli_stdout.txt"
    cli_stderr_path = tmp_path / "cli_stderr.txt"

    state_path.write_text(
        json.dumps(
            {
                "type": "ra_sim.gui_state",
                "state": {
                    "files": {
                        "background_files": ["bg0.osc", "bg1.osc"],
                        "current_background_index": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out_state_path.write_text(
        json.dumps({"type": "ra_sim.gui_state", "state": {"geometry": {"fit_result": "ok"}}}),
        encoding="utf-8",
    )
    log_path.write_text(
        "\n".join(
            [
                "success=True",
                "display_rms_px=3.500000",
                "final_metric_name=full_beam_fixed_correspondence",
                "bound_hits=[gamma, Gamma]",
                "boundary_warning=Possible identifiability issue",
                (
                    "param[theta_initial] group=tilt start=6.000000 final=5.500000 "
                    "delta=-0.500000 bounds=[5.000000, 7.000000] scale=1.000000"
                ),
                (
                    "param[gamma] group=tilt start=1.000000 final=0.800000 "
                    "delta=-0.200000 bounds=[-1.000000, 2.000000] scale=1.000000"
                ),
                (
                    "final metric=full_beam_fixed_correspondence cost=12.000000 "
                    "robust_cost=11.000000 weighted_rms_px=2.000000 "
                    "final_full_beam_rms_px=3.500000"
                ),
            ]
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "run",
                        "accepted": True,
                        "rejection_reason": None,
                        "final_metric_name": "full_beam_fixed_correspondence",
                        "weighted_residual_rms_px": 2.0,
                        "detector_rms_px": 3.5,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "seed_correspondence",
                        "pair_id": "pair[0]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [1, 1, 1],
                        "match_status": "matched",
                        "detector_residual_px": 8.0,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "seed_correspondence",
                        "pair_id": "pair[1]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [2, 0, 0],
                        "match_status": "matched",
                        "detector_residual_px": 4.0,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "acceptance_residuals",
                        "pair_id": "pair[0]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [1, 1, 1],
                        "match_status": "matched",
                        "detector_residual_px": 3.0,
                        "source_branch_index": 1,
                        "source_peak_index": 1,
                        "resolved_peak_index": 1,
                        "resolution_kind": "fixed_source",
                        "resolution_reason": "resolved",
                    }
                ),
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "acceptance_residuals",
                        "pair_id": "pair[1]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [2, 0, 0],
                        "match_status": "matched",
                        "detector_residual_px": 1.0,
                        "source_branch_index": 0,
                        "source_peak_index": 0,
                        "resolved_peak_index": 0,
                        "resolution_kind": "fixed_source",
                        "resolution_reason": "resolved",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    np.save(
        matched_peaks_path,
        np.asarray(
            [
                {"pair_id": "pair[0]", "measured_point": [100.0, 200.0]},
                {"pair_id": "pair[1]", "measured_point": [120.0, 220.0]},
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )
    cli_stdout_path.write_text("", encoding="utf-8")
    cli_stderr_path.write_text("", encoding="utf-8")

    artifacts = module.RunArtifacts(
        state_path=state_path,
        run_dir=tmp_path,
        out_state_path=out_state_path,
        log_path=log_path,
        trace_path=trace_path,
        matched_peaks_path=matched_peaks_path,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        cli_returncode=0,
    )

    report = module.build_quality_report(artifacts)

    assert report["decision_row"] == {
        "accepted": True,
        "rejection_reason": None,
        "final_metric_name": "full_beam_fixed_correspondence",
        "bound_hits": ["gamma", "Gamma"],
        "boundary_warning": "Possible identifiability issue",
        "matched_count_before": 2,
        "matched_count_after": 2,
        "rms_before_px": (80.0 / 2.0) ** 0.5,
        "rms_after_px": (10.0 / 2.0) ** 0.5,
        "max_residual_before_px": 8.0,
        "max_residual_after_px": 3.0,
        "worst_5_pair_ids_before": ["pair[0]", "pair[1]"],
        "worst_5_pair_ids_after": ["pair[0]", "pair[1]"],
    }
    assert report["start_parameters"] == [
        {
            "name": "theta_initial",
            "group": "tilt",
            "value": 6.0,
            "lower_bound": 5.0,
            "upper_bound": 7.0,
        },
        {
            "name": "gamma",
            "group": "tilt",
            "value": 1.0,
            "lower_bound": -1.0,
            "upper_bound": 2.0,
        },
    ]
    assert report["fitted_parameters"] == [
        {
            "name": "theta_initial",
            "group": "tilt",
            "value": 5.5,
            "delta": -0.5,
            "lower_bound": 5.0,
            "upper_bound": 7.0,
        },
        {
            "name": "gamma",
            "group": "tilt",
            "value": 0.8,
            "delta": -0.2,
            "lower_bound": -1.0,
            "upper_bound": 2.0,
        },
    ]
    assert report["final_metric"] == {
        "name": "full_beam_fixed_correspondence",
        "weighted_residual_rms_px": 2.0,
        "detector_rms_px": 3.5,
        "cost": 12.0,
        "robust_cost": 11.0,
    }
    assert report["overlay_evidence"]["matched_peaks"]["record_count"] == 2
    assert report["overlay_evidence"]["matched_peaks"]["preview"] == [
        {"pair_id": "pair[0]", "measured_point": [100.0, 200.0]},
        {"pair_id": "pair[1]", "measured_point": [120.0, 220.0]},
    ]
    assert report["identity_retention_after_fit"] == {
        "phase": "acceptance_residuals",
        "matched_statuses": ["matched"],
        "pair_record_count": 2,
        "matched_pair_count": 2,
        "source_branch_count": 2,
        "source_peak_count": 2,
        "resolved_peak_count": 2,
        "matched_with_resolved_peak_count": 2,
        "branch_retained_count": 2,
        "branch_mismatch_count": 0,
        "branch_unresolved_count": 0,
        "peak_retained_count": 2,
        "peak_mismatch_count": 0,
        "peak_unresolved_count": 0,
        "matched_resolution_kind_counts": {"fixed_source": 2},
        "matched_resolution_reason_counts": {"resolved": 2},
        "unmatched_resolution_reason_counts": {},
        "issue_preview": [],
    }


def test_build_quality_report_falls_back_to_preflight_pairs_and_stderr_reason(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new3.json"
    out_state_path = tmp_path / "new3_fit.json"
    trace_path = tmp_path / "geometry_fit_trace_new3.jsonl"
    cli_stdout_path = tmp_path / "cli_stdout.txt"
    cli_stderr_path = tmp_path / "cli_stderr.txt"

    state_path.write_text(
        json.dumps(
            {
                "type": "ra_sim.gui_state",
                "state": {
                    "files": {
                        "background_files": ["bg0.osc", "bg1.osc", "bg2.osc"],
                        "current_background_index": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "preflight_normalized_pairs",
                        "pair_id": "pair[0]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [1, 0, 0],
                        "match_status": "selected",
                        "optimizer_residual_px": 8.0,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "preflight_normalized_pairs",
                        "pair_id": "pair[1]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [2, 0, 0],
                        "match_status": "selected",
                        "optimizer_residual_px": 4.0,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "acceptance_residuals",
                        "pair_id": "pair[0]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [1, 0, 0],
                        "match_status": "missing_pair",
                        "source_branch_index": 1,
                        "source_peak_index": 1,
                        "resolution_kind": "fixed_source",
                        "resolution_reason": "source_table_out_of_range",
                    }
                ),
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "acceptance_residuals",
                        "pair_id": "pair[1]",
                        "dataset_index": 0,
                        "background_label": "bg0.osc",
                        "hkl": [2, 0, 0],
                        "match_status": "missing_pair",
                        "source_branch_index": 0,
                        "source_peak_index": 0,
                        "resolution_kind": "fixed_source",
                        "resolution_reason": "source_table_out_of_range",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    cli_stdout_path.write_text("", encoding="utf-8")
    cli_stderr_path.write_text(
        "Geometry fit unavailable: save manual Qr/Qz pairs first for bg1.osc, bg2.osc.\n",
        encoding="utf-8",
    )

    artifacts = module.RunArtifacts(
        state_path=state_path,
        run_dir=tmp_path,
        out_state_path=out_state_path,
        log_path=None,
        trace_path=trace_path,
        matched_peaks_path=None,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        cli_returncode=1,
    )

    report = module.build_quality_report(artifacts)

    assert report["before_fit"]["phase"] == "preflight_normalized_pairs"
    assert report["decision_row"]["rejection_reason"] == (
        "Geometry fit unavailable: save manual Qr/Qz pairs first for bg1.osc, bg2.osc."
    )
    assert report["decision_row"]["matched_count_before"] == 2
    assert report["decision_row"]["matched_count_after"] == 0
    assert report["decision_row"]["rms_before_px"] == (80.0 / 2.0) ** 0.5
    assert report["decision_row"]["rms_after_px"] is None
    assert report["identity_retention_after_fit"]["branch_retained_count"] == 0
    assert report["identity_retention_after_fit"]["branch_unresolved_count"] == 2
    assert report["identity_retention_after_fit"]["peak_retained_count"] == 0
    assert report["identity_retention_after_fit"]["peak_unresolved_count"] == 2
    assert report["identity_retention_after_fit"]["unmatched_resolution_reason_counts"] == {
        "source_table_out_of_range": 2
    }
    assert report["decision_row"]["worst_5_pair_ids_before"] == ["pair[0]", "pair[1]"]
    assert report["decision_row"]["worst_5_pair_ids_after"] == ["pair[0]", "pair[1]"]


def test_active_phase_label_prefers_trace_phase(tmp_path: Path) -> None:
    module = _load_baseline_module()

    log_path = tmp_path / "geometry_fit_log_state.txt"
    trace_path = tmp_path / "geometry_fit_trace_state.jsonl"
    log_path.write_text("Fit rejected:\n", encoding="utf-8")
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "pair",
                        "phase": "acceptance_residuals",
                        "pair_id": "bg0:pair3",
                        "match_status": "matched",
                    }
                ),
                json.dumps(
                    {
                        "record_type": "run",
                        "accepted": False,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert (
        module._active_phase_label(
            log_path=log_path,
            trace_path=trace_path,
        )
        == "run_complete_rejected"
    )


def test_active_phase_label_falls_back_to_log_marker(tmp_path: Path) -> None:
    module = _load_baseline_module()

    log_path = tmp_path / "geometry_fit_log_state.txt"
    log_path.write_text(
        "\n".join(
            [
                "Point-match diagnostics:",
                "Fit-space calibration:",
                "Fit rejected:",
            ]
        ),
        encoding="utf-8",
    )

    assert (
        module._active_phase_label(
            log_path=log_path,
            trace_path=None,
        )
        == "fit_rejected"
    )


def test_resolve_live_artifact_paths_ignores_stale_logs(tmp_path: Path) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new3_fresh_all.json"
    state_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir()
    cache_dir.mkdir()
    old_log = run_dir / "geometry_fit_log_old.txt"
    old_trace = cache_dir / "geometry_fit_trace_new3_fresh_all.jsonl"
    old_log.write_text("Fit rejected:\n", encoding="utf-8")
    old_trace.write_text("", encoding="utf-8")
    stale_timestamp = time.time() - 300.0
    os.utime(old_log, (stale_timestamp, stale_timestamp))
    os.utime(old_trace, (stale_timestamp, stale_timestamp))
    cli_stdout_path = run_dir / "cli_stdout.txt"
    cli_stderr_path = run_dir / "cli_stderr.txt"
    cli_stdout_path.write_text("", encoding="utf-8")
    cli_stderr_path.write_text("", encoding="utf-8")

    original_cache_dir = module.DEFAULT_CACHE_LOG_DIR
    module.DEFAULT_CACHE_LOG_DIR = cache_dir
    try:
        log_path, trace_path = module._resolve_live_artifact_paths(
            state_path=state_path,
            run_dir=run_dir,
            cli_stdout_path=cli_stdout_path,
            cli_stderr_path=cli_stderr_path,
            started_at_wall_time=time.time(),
        )
    finally:
        module.DEFAULT_CACHE_LOG_DIR = original_cache_dir

    assert log_path is None
    assert trace_path is None
