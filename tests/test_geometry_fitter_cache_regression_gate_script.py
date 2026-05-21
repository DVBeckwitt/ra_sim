from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/debug/run_geometry_fitter_cache_regression_gate.py"
HANDOFF_CHECK_SCRIPT_PATH = REPO_ROOT / "scripts/diagnostics/check_geometry_fit_handoff.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "run_geometry_fitter_cache_regression_gate",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_handoff_check_module():
    spec = importlib.util.spec_from_file_location(
        "check_geometry_fit_handoff",
        HANDOFF_CHECK_SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _command_names(commands) -> list[str]:
    return [command.name for command in commands]


def test_gate_local_mode_builds_fast_commands_without_slow_geometry(tmp_path) -> None:
    gate = _load_gate_module()

    commands = gate.build_gate_commands(mode="local", repo_root=tmp_path)

    names = _command_names(commands)
    assert "compile" in names
    assert "fast local pytest gate" in names
    assert "manual identity gate" in names
    assert "workflow slice" in names
    fast_gate = next(command for command in commands if command.name == "fast local pytest gate")
    assert "tests/test_gui_runtime_mixed_update_regressions.py" in fast_gate.command
    slow_gate = next(command for command in commands if command.name == "slow_geometry gate")
    assert slow_gate.skipped
    assert slow_gate.skip_reason == "skipped local mode"
    bi_gate = next(command for command in commands if command.name == "bi geometry baseline")
    assert not bi_gate.skipped
    assert "scripts/debug/run_geometry_fit_quality_baseline.py" in bi_gate.command


def test_gate_strict_mode_includes_slow_geometry(tmp_path) -> None:
    gate = _load_gate_module()

    commands = gate.build_gate_commands(mode="strict", repo_root=tmp_path)

    slow_gate = next(command for command in commands if command.name == "slow_geometry gate")
    assert not slow_gate.skipped
    assert "tests/test_manual_geometry_selection_helpers.py" in slow_gate.command
    assert gate.SLOW_GEOMETRY_SLICE in slow_gate.command


def test_gate_has_no_active_new4_commands(tmp_path) -> None:
    gate = _load_gate_module()

    commands = gate.build_gate_commands(mode="strict", repo_root=tmp_path)

    names = _command_names(commands)
    assert "new4 preflight" not in names
    assert "new4 ladder" not in names
    assert "bi geometry baseline" in names


def test_gate_uses_sys_executable_for_python_commands(tmp_path) -> None:
    gate = _load_gate_module()

    commands = gate.build_gate_commands(mode="strict", repo_root=tmp_path)

    runnable = [command for command in commands if not command.skipped]
    assert runnable
    assert all(command.command[0] == sys.executable for command in runnable)


def test_geometry_fit_handoff_checker_rejects_broken_caked_signature(tmp_path) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_trace_bad.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"record_type":"summary","live_runtime_cache_validation":'
                '[["manual_caked_fit_space_required",false],'
                '["validator_finite_caked_rows",0]]}',
                '{"record_type":"run","final_metric_name":"central_point_match",'
                '"point_match_summary":[["metric_unit","px"]],"nfev":1}',
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.violations_for_trace(trace_path)

    assert any("broken signature present" in violation for violation in violations)


def test_geometry_fit_handoff_checker_rejects_split_caked_objective_pixel_metric(
    tmp_path,
) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_trace_split_bad.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"record_type":"audit","objective_space":"caked_deg"}',
                '{"record_type":"run","final_metric_name":"central_point_match",'
                '"point_match_summary":[["metric_unit","px"]],"nfev":1}',
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.violations_for_trace(trace_path)

    assert any("objective_space=caked_deg used central_point_match" in v for v in violations)
    assert any("objective_space=caked_deg used metric_unit=px" in v for v in violations)


def test_geometry_fit_handoff_checker_rejects_split_caked_objective_zero_matches(
    tmp_path,
) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_trace_split_matched_zero.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                '{"record_type":"audit","objective_space":"caked_deg"}',
                '{"record_type":"run","point_match_summary":[["matched_pair_count",0]],"nfev":1}',
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.violations_for_trace(trace_path)

    assert any(
        "objective_space=caked_deg reached optimizer with matched=0" in v for v in violations
    )


def test_geometry_fit_handoff_checker_rejects_live_text_log_signature(tmp_path) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_bad_caked_handoff_17f00f25.log"
    trace_path.write_text(
        "\n".join(
            [
                "[geometry] manual_geometry_live_code_path git_commit=17f00f25",
                "[geometry] Placed peak 1 of 2 detector_to_caked_unavailable=true",
                "[geometry-fit] fit_observed_caked_deg=<unavailable reason=missing>",
                "[geometry-fit] fit_prediction_caked_deg=(32.931, 129.250)",
                "[geometry-fit] objective_space=caked_deg",
                "[geometry] preflight: ready to solve geometry fit (1 dataset)",
                "[geometry-fit] Geometry fit: identity fixed manual pairs eval=1 "
                "cost=433.320000 best_cost=433.320000 weighted_rms=12.0183 px",
                "[geometry-fit] Geometry fit: complete "
                "(cost=433.320000, rms=12.0183px, metric=central_point_match, matched=0)",
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.violations_for_trace(trace_path)

    assert any(
        "missing observed caked coordinates reached preflight ready" in v for v in violations
    )
    assert any("objective_space=caked_deg used central_point_match" in v for v in violations)
    assert any("objective_space=caked_deg used weighted_rms in px" in v for v in violations)


def test_geometry_fit_handoff_checker_rejects_finite_anchor_pixel_fallback(
    tmp_path,
) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_bad_caked_routing_e4a3e15f.log"
    trace_path.write_text(
        "\n".join(
            [
                "[geometry] manual_geometry_live_code_path git_commit=e4a3e15f",
                "[geometry-fit] fit_observed_caked_deg=(33.063, 130.754)",
                "[geometry-fit] fit_prediction_caked_deg=(32.773, 129.750)",
                "[geometry-fit] fit_observed_caked_deg=(37.566, 39.750)",
                "[geometry-fit] fit_prediction_caked_deg=(38.041, 41.750)",
                "[geometry-fit] objective_space=caked_deg",
                "[geometry-fit] Geometry fit: identity fixed manual pairs eval=1 "
                "cost=433.320000 best_cost=433.320000 weighted_rms=12.0183 px",
                "[geometry-fit] Geometry fit: complete "
                "(cost=433.320000, rms=12.0183px, metric=central_point_match, matched=0)",
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.violations_for_trace(trace_path)

    assert any("finite caked manual anchors used central_point_match" in v for v in violations)
    assert any("finite caked manual anchors used weighted_rms in px" in v for v in violations)


def test_checker_rejects_manual_tk_finite_caked_anchors_that_fall_back_to_pixel_match(
    tmp_path,
) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_bad_manual_tk_f761e78f.log"
    trace_path.write_text(
        "\n".join(
            [
                "git_commit=f761e78f",
                "fit_observed_caked_deg=(33.063, 130.754)",
                "fit_prediction_caked_deg=(32.696, 129.750)",
                "fit_observed_caked_deg=(37.566, 39.750)",
                "fit_prediction_caked_deg=(38.124, 41.281)",
                "objective_space=caked_deg",
                "objective_residual_units=deg",
                "Geometry fit: setup mode=point-match",
                "Geometry fit: running fixed-manual-pair direct least-squares solve",
                "weighted_rms=12.0183 px",
                "Geometry fit: complete "
                "(cost=433.320000, rms=12.0183px, metric=central_point_match, matched=0)",
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.violations_for_trace(trace_path)

    assert checker.main([str(trace_path)]) == 1
    assert any("caked objective used detector-pixel fallback" in v for v in violations)
    assert any("objective_space=caked_deg used central_point_match" in v for v in violations)
    assert any("objective_space=caked_deg used weighted_rms in px" in v for v in violations)


def test_geometry_fit_handoff_checker_rejects_split_text_dynamic_px_rms(tmp_path) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_bad_dynamic_px_rms.log"
    trace_path.write_text(
        "\n".join(
            [
                "Geometry fit: setup mode=dynamic-angle datasets=1",
                "qr_fit_point_surface_contract={'objective_space': 'caked_deg'}",
                "worst_angular_residual_rows=[{'observed_caked_deg': [33.063, 130.754], "
                "'predicted_caked_deg': [32.773, 129.750]}]",
                "Geometry fit: identity seed 1 eval=40 cost=112.578512 "
                "best_cost=112.578512 weighted_rms=3.1288px",
                "final metric=dynamic_angular_point_match cost=112.578512 "
                "weighted_rms_px=3.128807 final_full_beam_rms_px=914.494855",
            ]
        ),
        encoding="utf-8",
    )

    violations = checker.violations_for_trace(trace_path)

    assert any("objective_space=caked_deg used weighted_rms in px" in v for v in violations)


def test_geometry_fit_handoff_checker_allows_clean_manual_caked_route_check(
    tmp_path,
) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_clean_manual_caked_route.log"
    trace_path.write_text(
        "\n".join(
            [
                "manual_caked_route_check objective_space=caked_deg required=true "
                "ready=true observed_caked=2 predicted_caked=2 "
                "evaluator=dynamic_angular_point_match unit=deg",
                "Geometry fit: setup mode=dynamic-angle datasets=1",
                "Geometry fit: complete "
                "(cost=0.500000, rms=0.5000deg, "
                "metric=dynamic_angular_point_match, matched=2)",
            ]
        ),
        encoding="utf-8",
    )

    assert checker.violations_for_trace(trace_path) == []


def test_geometry_fit_handoff_checker_allows_caked_preflight_failure(tmp_path) -> None:
    checker = _load_handoff_check_module()
    trace_path = tmp_path / "geometry_fit_trace_preflight.jsonl"
    trace_path.write_text(
        '{"record_type":"summary","objective_space":"caked_deg",'
        '"manual_caked_fit_space_required":true,'
        '"validator_finite_caked_rows":0,'
        '"preflight_error":"manual_caked_fit_space_missing"}\n',
        encoding="utf-8",
    )

    assert checker.violations_for_trace(trace_path) == []
