"""Shared test-tier manifest for dev tooling and pytest collection."""

from __future__ import annotations

from pathlib import Path


FAST_TEST_FILES = (
    "test_background_peak_matching.py",
    "test_background_theta_helpers.py",
    "test_cli_cif_parse.py",
    "test_cli_geometry_fit.py",
    "test_compare_intensity.py",
    "test_config_loader.py",
    "test_ctr_fast_attenuation.py",
    "test_data_loading_parameters.py",
    "test_debug_controls.py",
    "test_debug_utils.py",
    "test_dependency_metadata.py",
    "test_dev_cli.py",
    "test_import_smoke.py",
)
INTEGRATION_TEST_FILES = (
    "test_cli_headless.py",
    "test_geometry_fit_landscape.py",
    "test_geometry_fitting.py",
    "test_gui_geometry_fit_workflow.py",
    "test_hbn_fitter_bundle_export.py",
    "test_simulation.py",
    "test_simulation_engine.py",
)
BENCHMARK_TEST_DIR = "tests/benchmarks"


def fast_test_paths(root: Path) -> list[str]:
    return [str(root / "tests" / name) for name in FAST_TEST_FILES]


def integration_test_paths(root: Path) -> list[str]:
    return [str(root / "tests" / name) for name in INTEGRATION_TEST_FILES]
