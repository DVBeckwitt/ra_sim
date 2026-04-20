from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest


RUN_SLOW_BASELINE_FITS = os.environ.get("RA_SIM_RUN_SLOW_BASELINE_FITS") == "1"

requires_slow_baseline_fit = pytest.mark.skipif(
    not RUN_SLOW_BASELINE_FITS,
    reason=(
        "slow/hanging real geometry baseline fit is opt-in for now; "
        "set RA_SIM_RUN_SLOW_BASELINE_FITS=1 to run"
    ),
)


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

    state_path = tmp_path / "new4_fresh_all.json"
    preflight_report_path = tmp_path / "new4_preflight_report.json"
    out_state_path = tmp_path / "new4_fresh_all_fit.json"
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
                    },
                    "geometry": {
                        "manual_pairs": [
                            {
                                "background_index": 0,
                                "entries": [
                                    {"hkl": [1, 1, 1]},
                                    {"hkl": [2, 0, 0]},
                                ],
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    preflight_report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "classification": "pass",
                "bound_manual_entry_count": 2,
                "processed_manual_entry_count": 2,
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
                        "fit_quality_passed": True,
                        "final_metric_name": "full_beam_fixed_correspondence",
                        "selection_status": "accepted",
                        "selected_candidate_name": "full_beam_polish_result",
                        "selected_candidate_source": "full_beam_polish",
                        "best_valid_raw_detector_candidate_name": "full_beam_polish_result",
                        "best_valid_raw_detector_candidate_source": "full_beam_polish",
                        "constraint_count": 4,
                        "active_fit_variable_count": 2,
                        "active_fit_variables": ["theta_initial", "gamma"],
                        "candidate_ledger": [
                            {
                                "candidate_name": "requested_start",
                                "x_vector_source": "requested_x0",
                                "matched_pair_count": 2,
                                "missing_pair_count": 0,
                                "branch_mismatch_count": 0,
                                "rms_px": (80.0 / 2.0) ** 0.5,
                                "median_px": 6.0,
                                "max_px": 8.0,
                                "outside_radius_count": 0,
                                "weighted_objective": 13.0,
                                "accepted_or_rejected": "accepted",
                                "rejection_reason": None,
                                "valid_raw_detector_candidate": True,
                                "selected": False,
                            },
                            {
                                "candidate_name": "full_beam_polish_result",
                                "x_vector_source": "full_beam_polish",
                                "matched_pair_count": 2,
                                "missing_pair_count": 0,
                                "branch_mismatch_count": 0,
                                "rms_px": (10.0 / 2.0) ** 0.5,
                                "median_px": 2.0,
                                "max_px": 3.0,
                                "outside_radius_count": 1,
                                "weighted_objective": 11.0,
                                "accepted_or_rejected": "accepted",
                                "rejection_reason": None,
                                "valid_raw_detector_candidate": True,
                                "selected": True,
                            },
                        ],
                        "weighted_residual_rms_px": 2.0,
                        "detector_rms_px": 3.5,
                        "dynamic_point_geometry_fit": True,
                        "full_beam_polish_enabled": True,
                        "full_beam_polish_accepted": True,
                        "full_beam_start_vector_source": "requested_x0",
                        "seed_correspondence_count": 2,
                        "nfev": 7,
                        "stage_timing_s": {
                            "preflight_rebind": 1.25,
                            "dynamic_seed": 2.5,
                            "central_solve": 3.75,
                            "full_beam_polish": 4.0,
                            "acceptance_diagnostics": 0.5,
                        },
                        "full_beam_polish_summary": {
                            "matched_pair_count_before": 2,
                            "matched_pair_count_after": 2,
                            "missing_pair_count_after": 0,
                            "start_outside_radius_count": 0,
                            "outside_radius_count_after": 0,
                            "branch_mismatch_count_after": 0,
                            "start_vector_source": "requested_x0",
                            "seed_correspondence_count": 2,
                        },
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
                        "dx_px": 8.0,
                        "dy_px": 0.0,
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
                        "dx_px": 4.0,
                        "dy_px": 0.0,
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
                        "dx_px": 3.0,
                        "dy_px": 0.0,
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
                        "dx_px": 1.0,
                        "dy_px": 0.0,
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
        "median_before_px": 6.0,
        "median_after_px": 2.0,
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
    assert report["before_fit"]["outside_radius_count"] == 0
    assert report["after_fit"]["outside_radius_count"] == 0
    assert report["run_summary"] == {
        "dynamic_point_geometry_fit": True,
        "fit_quality_passed": True,
        "selection_status": "accepted",
        "selected_candidate_name": "full_beam_polish_result",
        "selected_candidate_source": "full_beam_polish",
        "best_valid_raw_detector_candidate_name": "full_beam_polish_result",
        "best_valid_raw_detector_candidate_source": "full_beam_polish",
        "constraint_count": 4,
        "active_fit_variable_count": 2,
        "active_fit_variables": ["theta_initial", "gamma"],
        "candidate_ledger": [
            {
                "candidate_name": "requested_start",
                "x_vector_source": "requested_x0",
                "matched_pair_count": 2,
                "missing_pair_count": 0,
                "branch_mismatch_count": 0,
                "rms_px": (80.0 / 2.0) ** 0.5,
                "median_px": 6.0,
                "max_px": 8.0,
                "outside_radius_count": 0,
                "weighted_objective": 13.0,
                "accepted_or_rejected": "accepted",
                "rejection_reason": None,
                "valid_raw_detector_candidate": True,
                "selected": False,
            },
            {
                "candidate_name": "full_beam_polish_result",
                "x_vector_source": "full_beam_polish",
                "matched_pair_count": 2,
                "missing_pair_count": 0,
                "branch_mismatch_count": 0,
                "rms_px": (10.0 / 2.0) ** 0.5,
                "median_px": 2.0,
                "max_px": 3.0,
                "outside_radius_count": 1,
                "weighted_objective": 11.0,
                "accepted_or_rejected": "accepted",
                "rejection_reason": None,
                "valid_raw_detector_candidate": True,
                "selected": True,
            },
        ],
        "full_beam_polish_enabled": True,
        "full_beam_polish_accepted": True,
        "full_beam_start_vector_source": "requested_x0",
        "seed_correspondence_count": 2,
        "nfev": 7,
        "matched_fixed_pair_count_before": 2,
        "matched_fixed_pair_count_after": 2,
        "missing_fixed_pair_count_after": 0,
        "outside_radius_count_before": 0,
        "outside_radius_count_after": 0,
        "branch_mismatch_count": 0,
        "stage_timing_s": {
            "preflight_rebind": 1.25,
            "dynamic_seed": 2.5,
            "central_solve": 3.75,
            "full_beam_polish": 4.0,
            "acceptance_diagnostics": 0.5,
        },
    }
    assert report["saved_state_gate"] == {
        "ok": True,
        "failures": [],
        "timed_out": False,
        "timeout_reason": None,
        "preflight_valid_count": 2,
        "preflight_valid_count_source": "preflight_report",
        "matched_fixed_pair_count_before": 2,
        "matched_fixed_pair_count_after": 2,
        "missing_fixed_pair_count_before": 0,
        "missing_fixed_pair_count_after": 0,
        "branch_mismatch_count": 0,
        "before_rms_px": (80.0 / 2.0) ** 0.5,
        "after_rms_px": (10.0 / 2.0) ** 0.5,
        "before_median_px": 6.0,
        "after_median_px": 2.0,
        "before_max_px": 8.0,
        "after_max_px": 3.0,
        "outside_radius_count_before": 0,
        "outside_radius_count_after": 0,
        "fit_quality_passed": True,
        "selection_status": "accepted",
        "selected_candidate_name": "full_beam_polish_result",
        "selected_candidate_source": "full_beam_polish",
        "best_valid_raw_detector_candidate_name": "full_beam_polish_result",
        "best_valid_raw_detector_candidate_source": "full_beam_polish",
        "selected_candidate_matches_best_valid_raw": True,
        "rms_tolerance_px": 0.25,
        "max_error_tolerance_px": 1.0,
    }
    assert len(report["pair_alignment_rows"]) == 2
    assert report["pair_alignment_rows"][0] | {
        "pair_id": "pair[0]",
        "pair_index": 0,
        "hkl": [1, 1, 1],
        "source_branch_index": 1,
        "source_peak_index": 1,
        "before_dx_px": 8.0,
        "before_dy_px": 0.0,
        "before_distance_px": 8.0,
        "after_dx_px": 3.0,
        "after_dy_px": 0.0,
        "after_distance_px": 3.0,
        "improved": True,
    } == report["pair_alignment_rows"][0]
    assert report["pair_alignment_rows"][1] | {
        "pair_id": "pair[1]",
        "pair_index": 1,
        "hkl": [2, 0, 0],
        "source_branch_index": 0,
        "source_peak_index": 0,
        "before_dx_px": 4.0,
        "before_dy_px": 0.0,
        "before_distance_px": 4.0,
        "after_dx_px": 1.0,
        "after_dy_px": 0.0,
        "after_distance_px": 1.0,
        "improved": True,
    } == report["pair_alignment_rows"][1]
    for row in report["pair_alignment_rows"]:
        assert "measured_fit_space_source" in row
        assert "simulated_fit_space_source" in row
        assert "fit_space_projector_kind" in row
        assert "cake_bundle_signature" in row
        assert "valid" in row
        assert "invalid_projection_reason" in row
    assert report["worst_start_pair"] | {
        "pair_id": "pair[0]",
        "pair_index": 0,
        "hkl": [1, 1, 1],
        "source_branch_index": 1,
        "source_peak_index": 1,
        "before_dx_px": 8.0,
        "before_dy_px": 0.0,
        "before_distance_px": 8.0,
        "after_dx_px": 3.0,
        "after_dy_px": 0.0,
        "after_distance_px": 3.0,
        "improved": True,
    } == report["worst_start_pair"]
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
    compact = module._compact_summary_from_report(
        report,
        elapsed_s=12.5,
        baseline_report_writing_s=0.125,
    )
    assert compact | {
        "state": "new4_fresh_all",
        "accepted": True,
        "rejection_reason": None,
        "before_rms_px": (80.0 / 2.0) ** 0.5,
        "after_rms_px": (10.0 / 2.0) ** 0.5,
        "before_median_px": 6.0,
        "after_median_px": 2.0,
        "before_max_px": 8.0,
        "after_max_px": 3.0,
        "preflight_valid_count": 2,
        "matched_fixed_pair_count_before": 2,
        "missing_fixed_pair_count_before": 0,
        "matched_fixed_pair_count_after": 2,
        "missing_fixed_pair_count_after": 0,
        "outside_radius_count_before": 0,
        "outside_radius_count_after": 0,
        "branch_mismatch_count": 0,
        "fit_quality_passed": True,
        "selection_status": "accepted",
        "selected_candidate_name": "full_beam_polish_result",
        "selected_candidate_source": "full_beam_polish",
        "best_valid_raw_detector_candidate_name": "full_beam_polish_result",
        "best_valid_raw_detector_candidate_source": "full_beam_polish",
        "dynamic_point_geometry_fit": True,
        "full_beam_polish_enabled": True,
        "full_beam_polish_accepted": True,
        "full_beam_start_vector_source": "requested_x0",
        "seed_correspondence_count": 2,
        "constraint_count": 4,
        "active_fit_variable_count": 2,
        "active_fit_variables": ["theta_initial", "gamma"],
        "nfev": 7,
        "saved_state_gate_ok": True,
        "saved_state_gate_failures": [],
        "elapsed_s": 12.5,
        "stage_timing_s": {
            "preflight_rebind": 1.25,
            "dynamic_seed": 2.5,
            "central_solve": 3.75,
            "full_beam_polish": 4.0,
            "acceptance_diagnostics": 0.5,
            "baseline_report_writing": 0.125,
        },
    } == compact


def test_validate_manual_caked_fit_space_provenance_rejects_empty_new4_exact_proof() -> None:
    module = _load_baseline_module()

    violations = module.validate_manual_caked_fit_space_provenance(
        {
            "state_name": "new4_fresh_all",
            "manual_caked_residual_row_count": 0,
            "dataset_fit_space_projector_row_count": 0,
            "invalid_dataset_fit_space_projector_row_count": 0,
            "analytic_detector_fit_space_row_count": 0,
            "exact_fit_space_projector_available": False,
            "exact_fit_space_projection_reason": "missing_projector",
            "pair_alignment_rows": [],
        }
    )

    assert "new4 manual_caked_residual_row_count is empty" in violations
    assert "new4 exact_fit_space_projector_available is false" in violations
    assert "new4 dataset_fit_space_projector_row_count is empty" in violations


def test_validate_manual_caked_fit_space_provenance_accepts_exact_projector_row() -> None:
    module = _load_baseline_module()

    violations = module.validate_manual_caked_fit_space_provenance(
        {
            "state_name": "new4_fresh_all",
            "manual_caked_residual_row_count": 1,
            "dataset_fit_space_projector_row_count": 1,
            "invalid_dataset_fit_space_projector_row_count": 0,
            "analytic_detector_fit_space_row_count": 0,
            "exact_fit_space_projector_available": True,
            "exact_fit_space_projection_reason": None,
            "pair_alignment_rows": [
                {
                    "pair_id": "pair[0]",
                    "selected_is_minimum_background_distance": True,
                    "fit_space_projector_kind": "exact_caked_bundle",
                    "fit_space_anchor_override": False,
                    "measured_fit_space_source": "dataset_fit_space_projector",
                    "simulated_fit_space_source": "dataset_fit_space_projector",
                    "measured_detector_input_frame": "native_detector",
                    "simulated_detector_input_frame": "fit_detector",
                    "measured_native_frame_conversion_count": 0,
                    "simulated_native_frame_conversion_count": 1,
                    "measured_invalid_projection_reason": None,
                    "simulated_invalid_projection_reason": None,
                    "invalid_projection_reason": None,
                    "cake_bundle_signature": "sig-1",
                }
            ],
        }
    )

    assert violations == []


def test_validate_manual_caked_fit_space_provenance_accepts_rows_only_exact_row() -> None:
    module = _load_baseline_module()

    violations = module.validate_manual_caked_fit_space_provenance(
        [
            {
                "pair_id": "pair[0]",
                "selected_is_minimum_background_distance": True,
                "fit_space_projector_kind": "exact_caked_bundle",
                "fit_space_anchor_override": False,
                "measured_fit_space_source": "dataset_fit_space_projector",
                "simulated_fit_space_source": "dataset_fit_space_projector",
                "measured_detector_input_frame": "native_detector",
                "simulated_detector_input_frame": "fit_detector",
                "measured_native_frame_conversion_count": 0,
                "simulated_native_frame_conversion_count": 1,
                "measured_invalid_projection_reason": None,
                "simulated_invalid_projection_reason": None,
                "invalid_projection_reason": None,
                "cake_bundle_signature": "sig-rows-only",
            }
        ]
    )

    assert violations == []


def test_validate_manual_caked_fit_space_provenance_rejects_override_without_explicit_frame() -> None:
    module = _load_baseline_module()

    violations = module.validate_manual_caked_fit_space_provenance(
        [
            {
                "pair_id": "pair[override-bad-frame]",
                "selected_is_minimum_background_distance": True,
                "fit_space_projector_kind": "exact_caked_bundle",
                "fit_space_anchor_override": True,
                "measured_fit_space_source": "cached_fit_space_anchor",
                "simulated_fit_space_source": "dataset_fit_space_projector",
                "measured_detector_input_frame": "native_detector",
                "simulated_detector_input_frame": "fit_detector",
                "measured_native_frame_conversion_count": 0,
                "simulated_native_frame_conversion_count": 1,
                "measured_invalid_projection_reason": None,
                "simulated_invalid_projection_reason": None,
                "invalid_projection_reason": None,
                "cake_bundle_signature": "sig-override",
            }
        ]
    )

    assert (
        "pair[override-bad-frame]: fit_space_anchor_override requires explicit_override frame"
        in violations
    )


def test_validate_manual_caked_fit_space_provenance_rejects_explicit_frame_without_override() -> None:
    module = _load_baseline_module()

    violations = module.validate_manual_caked_fit_space_provenance(
        [
            {
                "pair_id": "pair[override-bad-flag]",
                "selected_is_minimum_background_distance": True,
                "fit_space_projector_kind": "exact_caked_bundle",
                "fit_space_anchor_override": False,
                "measured_fit_space_source": "cached_fit_space_anchor",
                "simulated_fit_space_source": "dataset_fit_space_projector",
                "measured_detector_input_frame": "explicit_override",
                "simulated_detector_input_frame": "fit_detector",
                "measured_native_frame_conversion_count": 0,
                "simulated_native_frame_conversion_count": 1,
                "measured_invalid_projection_reason": None,
                "simulated_invalid_projection_reason": None,
                "invalid_projection_reason": None,
                "cake_bundle_signature": "sig-explicit-frame",
            }
        ]
    )

    assert (
        "pair[override-bad-flag]: explicit_override frame requires fit_space_anchor_override"
        in violations
    )


def test_build_quality_report_falls_back_to_preflight_pairs_and_stderr_reason(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "baseline_state.json"
    out_state_path = tmp_path / "baseline_state_fit.json"
    trace_path = tmp_path / "geometry_fit_trace_baseline_state.jsonl"
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


def test_saved_state_gate_uses_preflight_report_count_without_requiring_before_fit_match() -> None:
    module = _load_baseline_module()

    gate = module._saved_state_gate_summary(
        {
            "decision_row": {
                "rms_before_px": 5.0,
                "rms_after_px": 4.5,
                "median_before_px": 4.0,
                "median_after_px": 3.5,
                "max_residual_before_px": 8.0,
                "max_residual_after_px": 7.0,
                "rejection_reason": None,
            },
            "run_summary": {
                "matched_fixed_pair_count_before": 1,
                "matched_fixed_pair_count_after": 2,
                "missing_fixed_pair_count_after": 0,
                "outside_radius_count_before": 1,
                "outside_radius_count_after": 1,
                "branch_mismatch_count": 0,
                "fit_quality_passed": True,
                "selection_status": "accepted",
                "selected_candidate_name": "full_beam_polish_result",
                "selected_candidate_source": "full_beam_polish",
                "best_valid_raw_detector_candidate_name": "full_beam_polish_result",
                "best_valid_raw_detector_candidate_source": "full_beam_polish",
            },
            "before_fit": {
                "matched_count": 1,
                "unresolved_count": 1,
                "outside_radius_count": 1,
                "rms_px": 5.0,
                "median_px": 4.0,
                "max_residual_px": 8.0,
            },
            "after_fit": {
                "matched_count": 2,
                "unresolved_count": 0,
                "outside_radius_count": 1,
                "rms_px": 4.5,
                "median_px": 3.5,
                "max_residual_px": 7.0,
            },
            "background_context": {
                "preflight_report_path": "C:/tmp/new4_preflight_report.json",
                "preflight_ok": True,
                "preflight_valid_count": 2,
            },
        }
    )

    assert gate == {
        "ok": True,
        "failures": [],
        "timed_out": False,
        "timeout_reason": None,
        "preflight_valid_count": 2,
        "preflight_valid_count_source": "preflight_report",
        "matched_fixed_pair_count_before": 1,
        "matched_fixed_pair_count_after": 2,
        "missing_fixed_pair_count_before": 1,
        "missing_fixed_pair_count_after": 0,
        "branch_mismatch_count": 0,
        "before_rms_px": 5.0,
        "after_rms_px": 4.5,
        "before_median_px": 4.0,
        "after_median_px": 3.5,
        "before_max_px": 8.0,
        "after_max_px": 7.0,
        "outside_radius_count_before": 1,
        "outside_radius_count_after": 1,
        "fit_quality_passed": True,
        "selection_status": "accepted",
        "selected_candidate_name": "full_beam_polish_result",
        "selected_candidate_source": "full_beam_polish",
        "best_valid_raw_detector_candidate_name": "full_beam_polish_result",
        "best_valid_raw_detector_candidate_source": "full_beam_polish",
        "selected_candidate_matches_best_valid_raw": True,
        "rms_tolerance_px": 0.25,
        "max_error_tolerance_px": 1.0,
    }


def test_saved_state_gate_allows_retained_start_safe_fallback_with_valid_raw_alignment() -> None:
    module = _load_baseline_module()

    gate = module._saved_state_gate_summary(
        {
            "decision_row": {
                "rms_before_px": 5.0,
                "rms_after_px": 5.2,
                "median_before_px": 4.0,
                "median_after_px": 4.1,
                "max_residual_before_px": 8.0,
                "max_residual_after_px": 8.5,
                "rejection_reason": None,
            },
            "run_summary": {
                "matched_fixed_pair_count_before": 6,
                "matched_fixed_pair_count_after": 7,
                "missing_fixed_pair_count_after": 0,
                "outside_radius_count_before": 1,
                "outside_radius_count_after": 1,
                "branch_mismatch_count": 0,
                "fit_quality_passed": False,
                "selection_status": "retained_start_safe_fallback",
                "selected_candidate_name": "requested_start",
                "selected_candidate_source": "requested_x0",
                "best_valid_raw_detector_candidate_name": "requested_start",
                "best_valid_raw_detector_candidate_source": "requested_x0",
            },
            "before_fit": {
                "matched_count": 6,
                "unresolved_count": 1,
                "outside_radius_count": 1,
                "rms_px": 5.0,
                "median_px": 4.0,
                "max_residual_px": 8.0,
            },
            "after_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 1,
                "rms_px": 5.2,
                "median_px": 4.1,
                "max_residual_px": 8.5,
            },
            "background_context": {
                "preflight_report_path": "C:/tmp/new4_preflight_report.json",
                "preflight_ok": True,
                "preflight_valid_count": 7,
            },
        }
    )

    assert gate["ok"] is True
    assert gate["failures"] == []
    assert gate["fit_quality_passed"] is False
    assert gate["selection_status"] == "retained_start_safe_fallback"


def test_saved_state_gate_rejects_retained_start_safe_fallback_when_raw_alignment_regresses() -> None:
    module = _load_baseline_module()

    gate = module._saved_state_gate_summary(
        {
            "decision_row": {
                "rms_before_px": 5.0,
                "rms_after_px": 5.6,
                "median_before_px": 4.0,
                "median_after_px": 4.4,
                "max_residual_before_px": 8.0,
                "max_residual_after_px": 9.5,
                "rejection_reason": None,
            },
            "run_summary": {
                "matched_fixed_pair_count_before": 7,
                "matched_fixed_pair_count_after": 6,
                "missing_fixed_pair_count_after": 1,
                "outside_radius_count_before": 0,
                "outside_radius_count_after": 1,
                "branch_mismatch_count": 0,
                "fit_quality_passed": False,
                "selection_status": "retained_start_safe_fallback",
                "selected_candidate_name": "requested_start",
                "selected_candidate_source": "requested_x0",
                "best_valid_raw_detector_candidate_name": "requested_start",
                "best_valid_raw_detector_candidate_source": "requested_x0",
            },
            "before_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 0,
                "rms_px": 5.0,
                "median_px": 4.0,
                "max_residual_px": 8.0,
            },
            "after_fit": {
                "matched_count": 6,
                "unresolved_count": 1,
                "outside_radius_count": 1,
                "rms_px": 5.6,
                "median_px": 4.4,
                "max_residual_px": 9.5,
            },
            "background_context": {
                "preflight_report_path": "C:/tmp/new4_preflight_report.json",
                "preflight_ok": True,
                "preflight_valid_count": 7,
            },
        }
    )

    assert gate["ok"] is False
    assert (
        "fit quality did not pass raw detector gate (selection_status=retained_start_safe_fallback)"
        in gate["failures"]
    )


def test_saved_state_gate_rejects_non_retained_failed_result_even_when_other_metrics_look_valid() -> None:
    module = _load_baseline_module()

    gate = module._saved_state_gate_summary(
        {
            "decision_row": {
                "rms_before_px": 5.0,
                "rms_after_px": 5.1,
                "median_before_px": 4.0,
                "median_after_px": 4.0,
                "max_residual_before_px": 8.0,
                "max_residual_after_px": 8.2,
                "rejection_reason": None,
            },
            "run_summary": {
                "matched_fixed_pair_count_before": 7,
                "matched_fixed_pair_count_after": 7,
                "missing_fixed_pair_count_after": 0,
                "outside_radius_count_before": 0,
                "outside_radius_count_after": 0,
                "branch_mismatch_count": 0,
                "fit_quality_passed": False,
                "selection_status": "rejected",
                "selected_candidate_name": "requested_start",
                "selected_candidate_source": "requested_x0",
                "best_valid_raw_detector_candidate_name": "requested_start",
                "best_valid_raw_detector_candidate_source": "requested_x0",
            },
            "before_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 0,
                "rms_px": 5.0,
                "median_px": 4.0,
                "max_residual_px": 8.0,
            },
            "after_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 0,
                "rms_px": 5.1,
                "median_px": 4.0,
                "max_residual_px": 8.2,
            },
            "background_context": {
                "preflight_report_path": "C:/tmp/new4_preflight_report.json",
                "preflight_ok": True,
                "preflight_valid_count": 7,
            },
        }
    )

    assert gate["ok"] is False
    assert "fit quality did not pass raw detector gate (selection_status=rejected)" in gate[
        "failures"
    ]


def test_saved_state_gate_rejects_retained_start_safe_fallback_with_missing_candidate_ids() -> None:
    module = _load_baseline_module()

    gate = module._saved_state_gate_summary(
        {
            "decision_row": {
                "rms_before_px": 5.0,
                "rms_after_px": 5.2,
                "median_before_px": 4.0,
                "median_after_px": 4.1,
                "max_residual_before_px": 8.0,
                "max_residual_after_px": 8.5,
                "rejection_reason": None,
            },
            "run_summary": {
                "matched_fixed_pair_count_before": 6,
                "matched_fixed_pair_count_after": 7,
                "missing_fixed_pair_count_after": 0,
                "outside_radius_count_before": 1,
                "outside_radius_count_after": 1,
                "branch_mismatch_count": 0,
                "fit_quality_passed": False,
                "selection_status": "retained_start_safe_fallback",
                "selected_candidate_name": None,
                "selected_candidate_source": None,
                "best_valid_raw_detector_candidate_name": None,
                "best_valid_raw_detector_candidate_source": None,
            },
            "before_fit": {
                "matched_count": 6,
                "unresolved_count": 1,
                "outside_radius_count": 1,
                "rms_px": 5.0,
                "median_px": 4.0,
                "max_residual_px": 8.0,
            },
            "after_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 1,
                "rms_px": 5.2,
                "median_px": 4.1,
                "max_residual_px": 8.5,
            },
            "background_context": {
                "preflight_report_path": "C:/tmp/new4_preflight_report.json",
                "preflight_ok": True,
                "preflight_valid_count": 7,
            },
        }
    )

    assert gate["ok"] is False
    assert gate["selected_candidate_matches_best_valid_raw"] is False
    assert "selected candidate was not the best valid raw detector candidate" in gate[
        "failures"
    ]


def test_saved_state_gate_rejects_retained_start_safe_fallback_with_partial_candidate_ids() -> None:
    module = _load_baseline_module()

    gate = module._saved_state_gate_summary(
        {
            "decision_row": {
                "rms_before_px": 5.0,
                "rms_after_px": 5.2,
                "median_before_px": 4.0,
                "median_after_px": 4.1,
                "max_residual_before_px": 8.0,
                "max_residual_after_px": 8.5,
                "rejection_reason": None,
            },
            "run_summary": {
                "matched_fixed_pair_count_before": 6,
                "matched_fixed_pair_count_after": 7,
                "missing_fixed_pair_count_after": 0,
                "outside_radius_count_before": 1,
                "outside_radius_count_after": 1,
                "branch_mismatch_count": 0,
                "fit_quality_passed": False,
                "selection_status": "retained_start_safe_fallback",
                "selected_candidate_name": "requested_start",
                "selected_candidate_source": "",
                "best_valid_raw_detector_candidate_name": "requested_start",
                "best_valid_raw_detector_candidate_source": "requested_x0",
            },
            "before_fit": {
                "matched_count": 6,
                "unresolved_count": 1,
                "outside_radius_count": 1,
                "rms_px": 5.0,
                "median_px": 4.0,
                "max_residual_px": 8.0,
            },
            "after_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 1,
                "rms_px": 5.2,
                "median_px": 4.1,
                "max_residual_px": 8.5,
            },
            "background_context": {
                "preflight_report_path": "C:/tmp/new4_preflight_report.json",
                "preflight_ok": True,
                "preflight_valid_count": 7,
            },
        }
    )

    assert gate["ok"] is False
    assert gate["selected_candidate_matches_best_valid_raw"] is False
    assert "selected candidate was not the best valid raw detector candidate" in gate[
        "failures"
    ]


def test_saved_state_gate_requires_preflight_report() -> None:
    module = _load_baseline_module()

    gate = module._saved_state_gate_summary(
        {
            "decision_row": {
                "rms_before_px": 5.0,
                "rms_after_px": 4.5,
                "max_residual_before_px": 8.0,
                "max_residual_after_px": 7.0,
            },
            "run_summary": {
                "matched_fixed_pair_count_before": 7,
                "matched_fixed_pair_count_after": 7,
                "missing_fixed_pair_count_after": 0,
                "outside_radius_count_before": 0,
                "outside_radius_count_after": 0,
                "branch_mismatch_count": 0,
                "fit_quality_passed": True,
                "selection_status": "accepted",
                "selected_candidate_name": "full_beam_polish_result",
                "selected_candidate_source": "full_beam_polish",
                "best_valid_raw_detector_candidate_name": "full_beam_polish_result",
                "best_valid_raw_detector_candidate_source": "full_beam_polish",
            },
            "before_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 0,
                "rms_px": 5.0,
                "max_residual_px": 8.0,
            },
            "after_fit": {
                "matched_count": 7,
                "unresolved_count": 0,
                "outside_radius_count": 0,
                "rms_px": 4.5,
                "max_residual_px": 7.0,
            },
            "background_context": {
                "current_background_manual_pair_count": 7,
            },
        }
    )

    assert gate["ok"] is False
    assert gate["preflight_valid_count"] == 0
    assert gate["preflight_valid_count_source"] == "missing_preflight_report"
    assert "preflight report was unavailable" in gate["failures"]
    assert "preflight report was not ok" in gate["failures"]
    assert "preflight valid pair count was 0" in gate["failures"]


def test_candidate_preflight_report_paths_prefer_base_state_report() -> None:
    module = _load_baseline_module()

    state_path = Path("C:/tmp/new4_fresh_all.json")

    assert module._candidate_preflight_report_paths(state_path) == (
        Path("C:/tmp/new4_preflight_report.json"),
        Path("C:/tmp/new4_fresh_all_preflight_report.json"),
    )


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


def test_build_quality_report_timeout_surfaces_last_phase_and_pair_in_gate(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new4_fresh_all.json"
    preflight_report_path = tmp_path / "new4_preflight_report.json"
    out_state_path = tmp_path / "new4_fresh_all_fit.json"
    trace_path = tmp_path / "geometry_fit_trace_new4_fresh_all.jsonl"
    cli_stdout_path = tmp_path / "cli_stdout.txt"
    cli_stderr_path = tmp_path / "cli_stderr.txt"
    timeout_reason = (
        "new4_fresh_all exceeded 600.0s last_phase=acceptance_residuals "
        "last_pair_id=bg0:pair6"
    )

    state_path.write_text(
        json.dumps(
            {
                "type": "ra_sim.gui_state",
                "state": {
                    "files": {
                        "background_files": ["bg0.osc"],
                        "current_background_index": 0,
                    },
                    "geometry": {
                        "manual_pairs": [
                            {
                                "background_index": 0,
                                "entries": [{"hkl": [1, 0, 0]} for _ in range(7)],
                            }
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    preflight_report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "classification": "pass",
                "bound_manual_entry_count": 7,
                "processed_manual_entry_count": 7,
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
                        "phase": "acceptance_residuals",
                        "pair_id": "bg0:pair6",
                        "match_status": "matched",
                        "detector_residual_px": 6.0,
                    }
                )
            ]
        ),
        encoding="utf-8",
    )
    cli_stdout_path.write_text("", encoding="utf-8")
    cli_stderr_path.write_text("", encoding="utf-8")

    artifacts = module.RunArtifacts(
        state_path=state_path,
        run_dir=tmp_path,
        out_state_path=out_state_path,
        log_path=None,
        trace_path=trace_path,
        matched_peaks_path=None,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        cli_returncode=-1,
        timed_out=True,
        timeout_reason=timeout_reason,
        timeout_last_phase="acceptance_residuals",
        timeout_last_pair_id="bg0:pair6",
    )

    report = module.build_quality_report(artifacts)

    assert report["artifacts"]["timed_out"] is True
    assert report["decision_row"]["rejection_reason"] == timeout_reason
    assert report["saved_state_gate"]["ok"] is False
    assert report["saved_state_gate"]["timed_out"] is True
    assert report["saved_state_gate"]["timeout_reason"] == timeout_reason
    assert timeout_reason in report["saved_state_gate"]["failures"]


def test_write_report_files_keeps_quality_report_canonical_on_timeout(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    quality_json = tmp_path / "new4_fresh_all_quality_report.json"
    quality_md = tmp_path / "new4_fresh_all_quality_report.md"
    quality_json.write_text('{"status":"good"}', encoding="utf-8")
    quality_md.write_text("good\n", encoding="utf-8")

    json_path, md_path = module._write_report_files(
        {"status": "timeout"},
        tmp_path,
        "new4_fresh_all",
        timed_out=True,
    )

    timeout_json = tmp_path / "new4_fresh_all_timeout_report.json"
    timeout_md = tmp_path / "new4_fresh_all_timeout_report.md"
    assert json_path == timeout_json
    assert md_path == timeout_md
    assert json.loads(quality_json.read_text(encoding="utf-8")) == {"status": "good"}
    assert quality_md.read_text(encoding="utf-8") == "good\n"
    assert json.loads(timeout_json.read_text(encoding="utf-8")) == {"status": "timeout"}
    assert timeout_md.is_file()


def test_write_report_files_removes_stale_timeout_sidecars_on_success(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    timeout_json = tmp_path / "new4_fresh_all_timeout_report.json"
    timeout_md = tmp_path / "new4_fresh_all_timeout_report.md"
    timeout_json.write_text('{"status":"timeout"}', encoding="utf-8")
    timeout_md.write_text("timeout\n", encoding="utf-8")

    json_path, md_path = module._write_report_files(
        {"status": "ok"},
        tmp_path,
        "new4_fresh_all",
        timed_out=False,
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"status": "ok"}
    assert md_path.is_file()
    assert timeout_json.exists() is False
    assert timeout_md.exists() is False


def test_resolve_live_artifact_paths_ignores_stale_logs(tmp_path: Path) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new4_fresh_all.json"
    state_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir()
    cache_dir.mkdir()
    old_log = run_dir / "geometry_fit_log_old.txt"
    old_trace = cache_dir / "geometry_fit_trace_new4_fresh_all.jsonl"
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


def test_is_current_run_artifact_rejects_near_launch_older_file(tmp_path: Path) -> None:
    module = _load_baseline_module()

    artifact_path = tmp_path / "geometry_fit_log_new4_fresh_all.txt"
    artifact_path.write_text("Fit rejected:\n", encoding="utf-8")
    started_at_wall_time = time.time()
    stale_timestamp = started_at_wall_time - 0.25
    os.utime(artifact_path, (stale_timestamp, stale_timestamp))

    assert (
        module._is_current_run_artifact(
            artifact_path,
            started_at_wall_time=started_at_wall_time,
        )
        is False
    )


def test_resolve_artifact_paths_falls_back_to_default_live_cache_logs(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new4_fresh_all.json"
    state_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir()
    cache_dir.mkdir()
    cli_stdout_path = run_dir / "cli_stdout.txt"
    cli_stderr_path = run_dir / "cli_stderr.txt"
    cli_stdout_path.write_text("", encoding="utf-8")
    cli_stderr_path.write_text("", encoding="utf-8")
    default_log_path = cache_dir / "geometry_fit_log_new4_fresh_all.txt"
    default_trace_path = cache_dir / "geometry_fit_trace_new4_fresh_all.jsonl"
    default_log_path.write_text(
        "\n".join(
            [
                "Fit rejected:",
                f"trace_path={default_trace_path}",
            ]
        ),
        encoding="utf-8",
    )
    default_trace_path.write_text(
        json.dumps(
            {
                "record_type": "run",
                "accepted": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    original_cache_dir = module.DEFAULT_CACHE_LOG_DIR
    module.DEFAULT_CACHE_LOG_DIR = cache_dir
    try:
        log_path, trace_path, matched_peaks_path = module._resolve_artifact_paths(
            state_path=state_path,
            run_dir=run_dir,
            cli_stdout_path=cli_stdout_path,
            cli_stderr_path=cli_stderr_path,
            allow_default_live_cache=True,
        )
    finally:
        module.DEFAULT_CACHE_LOG_DIR = original_cache_dir

    assert matched_peaks_path is None
    assert log_path == run_dir / default_log_path.name
    assert trace_path == run_dir / default_trace_path.name
    assert log_path.is_file()
    assert trace_path.is_file()


def test_resolve_artifact_paths_ignores_stale_default_live_cache_for_fresh_run(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new4_fresh_all.json"
    state_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir()
    cache_dir.mkdir()
    cli_stdout_path = run_dir / "cli_stdout.txt"
    cli_stderr_path = run_dir / "cli_stderr.txt"
    cli_stdout_path.write_text("", encoding="utf-8")
    cli_stderr_path.write_text("", encoding="utf-8")
    default_log_path = cache_dir / "geometry_fit_log_new4_fresh_all.txt"
    default_trace_path = cache_dir / "geometry_fit_trace_new4_fresh_all.jsonl"
    default_log_path.write_text(
        "\n".join(
            [
                "Fit rejected:",
                f"trace_path={default_trace_path}",
            ]
        ),
        encoding="utf-8",
    )
    default_trace_path.write_text("", encoding="utf-8")
    stale_timestamp = time.time() - 300.0
    os.utime(default_log_path, (stale_timestamp, stale_timestamp))
    os.utime(default_trace_path, (stale_timestamp, stale_timestamp))

    original_cache_dir = module.DEFAULT_CACHE_LOG_DIR
    module.DEFAULT_CACHE_LOG_DIR = cache_dir
    try:
        log_path, trace_path, matched_peaks_path = module._resolve_artifact_paths(
            state_path=state_path,
            run_dir=run_dir,
            cli_stdout_path=cli_stdout_path,
            cli_stderr_path=cli_stderr_path,
            allow_default_live_cache=True,
            started_at_wall_time=time.time(),
        )
    finally:
        module.DEFAULT_CACHE_LOG_DIR = original_cache_dir

    assert log_path is None
    assert trace_path is None
    assert matched_peaks_path is None


def test_reuse_existing_run_ignores_default_live_cache_artifacts(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new4_fresh_all.json"
    state_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    run_dir.mkdir()
    cache_dir.mkdir()

    (run_dir / "new4_fresh_all_fit.json").write_text("{}", encoding="utf-8")
    (run_dir / "cli_stdout.txt").write_text("", encoding="utf-8")
    (run_dir / "cli_stderr.txt").write_text("", encoding="utf-8")

    default_log_path = cache_dir / "geometry_fit_log_new4_fresh_all.txt"
    default_trace_path = cache_dir / "geometry_fit_trace_new4_fresh_all.jsonl"
    default_log_path.write_text(
        "\n".join(
            [
                "Fit rejected:",
                f"trace_path={default_trace_path}",
            ]
        ),
        encoding="utf-8",
    )
    default_trace_path.write_text(
        json.dumps(
            {
                "record_type": "run",
                "accepted": False,
                "phase": "stale_phase",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    original_cache_dir = module.DEFAULT_CACHE_LOG_DIR
    module.DEFAULT_CACHE_LOG_DIR = cache_dir
    try:
        artifacts = module._reuse_existing_run(state_path, run_dir)
    finally:
        module.DEFAULT_CACHE_LOG_DIR = original_cache_dir

    assert artifacts is not None
    assert artifacts.log_path is None
    assert artifacts.trace_path is None
    assert artifacts.matched_peaks_path is None
    assert (run_dir / default_log_path.name).exists() is False
    assert (run_dir / default_trace_path.name).exists() is False


def test_resolve_artifact_paths_rejects_stale_sidecars_reported_by_fresh_log(
    tmp_path: Path,
) -> None:
    module = _load_baseline_module()

    state_path = tmp_path / "new4_fresh_all.json"
    state_path.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    artifact_dir = tmp_path / "artifacts"
    run_dir.mkdir()
    artifact_dir.mkdir()
    cli_stdout_path = run_dir / "cli_stdout.txt"
    cli_stderr_path = run_dir / "cli_stderr.txt"
    reported_log_path = artifact_dir / "geometry_fit_log_new4_fresh_all.txt"
    reported_trace_path = artifact_dir / "geometry_fit_trace_new4_fresh_all.jsonl"
    reported_matched_peaks_path = artifact_dir / "matched_peaks_new4_fresh_all.npy"
    reported_log_path.write_text(
        "\n".join(
            [
                "Fit rejected:",
                f"trace_path={reported_trace_path}",
            ]
        ),
        encoding="utf-8",
    )
    reported_trace_path.write_text("", encoding="utf-8")
    np.save(reported_matched_peaks_path, np.asarray([[1.0, 2.0]], dtype=float))
    stale_timestamp = time.time() - 300.0
    os.utime(reported_trace_path, (stale_timestamp, stale_timestamp))
    os.utime(reported_matched_peaks_path, (stale_timestamp, stale_timestamp))
    started_at_wall_time = time.time()
    os.utime(reported_log_path, (started_at_wall_time, started_at_wall_time))
    cli_stdout_path.write_text(
        "\n".join(
            [
                f"Geometry fit log: {reported_log_path}",
                f"Matched peaks: {reported_matched_peaks_path}",
            ]
        ),
        encoding="utf-8",
    )
    cli_stderr_path.write_text("", encoding="utf-8")

    log_path, trace_path, matched_peaks_path = module._resolve_artifact_paths(
        state_path=state_path,
        run_dir=run_dir,
        cli_stdout_path=cli_stdout_path,
        cli_stderr_path=cli_stderr_path,
        started_at_wall_time=started_at_wall_time - 1.0,
    )

    assert log_path == run_dir / reported_log_path.name
    assert trace_path is None
    assert matched_peaks_path is None


@pytest.mark.slow_baseline_fit
@requires_slow_baseline_fit
def test_new4_preflight_and_baseline_stop_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    validate_script = (
        repo_root / "scripts" / "debug" / "validate_geometry_preflight_rebind.py"
    )
    baseline_script = (
        repo_root / "scripts" / "debug" / "run_geometry_fit_quality_baseline.py"
    )
    state_path = (
        repo_root / "artifacts" / "geometry_fit_gui_states" / "new4.json"
    )
    fresh_state_path = tmp_path / "new4_fresh_all.json"
    preflight_report_path = tmp_path / "new4_preflight_report.json"
    baseline_output_root = tmp_path / "baseline"
    slow_timeout_s = int(os.environ.get("RA_SIM_SLOW_BASELINE_TIMEOUT", "900"))

    try:
        subprocess.run(
            [
                sys.executable,
                str(validate_script),
                "--state",
                str(state_path),
                "--background-index",
                "0",
                "--mode",
                "full",
                "--export-fresh-state",
                str(fresh_state_path),
                "--report-path",
                str(preflight_report_path),
            ],
            check=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=slow_timeout_s,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "new4 slow baseline fit timed out; this remains an optimizer/runtime issue"
        )

    preflight_report = json.loads(preflight_report_path.read_text(encoding="utf-8"))
    targeted_gate = dict(preflight_report.get("targeted_performance_gate", {}))

    assert preflight_report["processed_manual_entry_count"] == 7
    assert preflight_report["bound_manual_entry_count"] == 7
    assert preflight_report["missing_manual_entry_count"] == 0
    assert preflight_report["branch_mismatch_count"] == 0
    assert preflight_report["background_distance_gate_ok"] is True
    assert preflight_report["runtime_prepare_ok"] is True
    assert preflight_report["fresh_export_ok"] is True
    assert preflight_report["resolved_source_pair_count"] == 7
    assert targeted_gate
    assert targeted_gate["ok"] is True

    try:
        subprocess.run(
            [
                sys.executable,
                str(baseline_script),
                str(fresh_state_path),
                "--output-root",
                str(baseline_output_root),
            ],
            check=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=slow_timeout_s,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "new4 slow baseline fit timed out; this remains an optimizer/runtime issue"
        )

    baseline_report_path = (
        baseline_output_root
        / fresh_state_path.stem
        / f"{fresh_state_path.stem}_quality_report.json"
    )
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    saved_state_gate = dict(baseline_report.get("saved_state_gate", {}))

    assert baseline_report["matched_fixed_pair_count_after"] == 7
    assert baseline_report["missing_fixed_pair_count_after"] == 0
    assert baseline_report["branch_mismatch_count"] == 0
    assert baseline_report["rejection_reason"] != (
        "No matched peak pairs were available for the fitted solution."
    )
    assert (
        baseline_report["after_rms_px"]
        <= baseline_report["before_rms_px"] + 0.25
    )
    assert (
        baseline_report["after_max_error_px"]
        <= baseline_report["before_max_error_px"] + 1.0
    )
    assert saved_state_gate["ok"] is True
