import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import least_squares, nnls

from scripts.diagnostics.run_all_background_peak_fits import _output_dir_for_state, _safe_run_name


NOTEBOOK_PATH = Path("scripts/diagnostics/all_background_peak_fits.ipynb")
RUNNER_PATH = Path("scripts/diagnostics/run_all_background_peak_fits.py")


def _notebook_source() -> str:
    if not NOTEBOOK_PATH.exists():
        pytest.skip(f"{NOTEBOOK_PATH} is not present in this checkout")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _notebook_functions(*names: str) -> dict[str, object]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    wanted = set(names)
    selected: list[ast.FunctionDef] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        tree = ast.parse(source, filename=f"{NOTEBOOK_PATH}:cell{index}")
        selected.extend(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
        )
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "np": np,
        "least_squares": least_squares,
        "_rod_qz_nnls": nnls,
        "ROD_QZ_TAIL_POWER_GRID": np.asarray([0.75, 1.0, 1.35, 2.0, 3.5, 7.0], dtype=np.float64),
        "ROD_QZ_TAIL_WIDTH_SCALE_GRID": np.asarray([0.55, 0.80, 1.15, 1.70, 2.50], dtype=np.float64),
        "ROD_QZ_TAIL_CENTER_SEARCH_BINS": 5.0,
        "ROD_QZ_TAIL_MIN_HALFWIDTH_BINS": 0.55,
        "ROD_QZ_TAIL_MAX_HALFWIDTH_FRACTION": 0.70,
        "ROD_QZ_TAIL_MAX_AUTO_PEAKS": 10,
        "ROD_QZ_TAIL_COMPONENT_MIN_RELATIVE": 1.0e-5,
    }
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def test_all_background_peak_fits_limits_qz_profiles_to_sixty_deg_two_theta() -> None:
    source = _notebook_source()

    assert "ROD_PROFILE_MAX_TWO_THETA_DEG = 60.0" in source
    assert "detector_two_theta_map=profile_detector_two_theta_map" in source
    assert "theta_map <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)" in source
    assert "two_theta_values <= float(ROD_PROFILE_MAX_TWO_THETA_DEG)" in source
    assert "branch_mask = branch_mask & theta_region_within_profile_limit[None, :]" in source
    assert "ax.set_xlim(caked_region_theta_min, caked_region_theta_max)" in source


def test_background_peak_fits_notebook_uses_density_profiles() -> None:
    source = _notebook_source()

    for token in (
        "background_density",
        "fit_density",
        "caked_sum_signal",
        "caked_sum_normalization",
    ):
        assert token in source

    calls = re.findall(r"(?<!def )normalized_profile_payload\([^\)]*\)", source)
    assert calls
    assert not any("background_sum" in call or "fit_sum" in call for call in calls)


def test_background_peak_fits_notebook_uses_density_caked_figures() -> None:
    source = _notebook_source()

    for token in (
        'CAKED_FIGURE_INTENSITY_MODE = "density"',
        "def caked_image_for_intensity_mode(",
        '"caked_density_image"',
        '"caked_raw_sum_image"',
        '"caked_display_image"',
        '"caked_count"',
        'profile_bg.get("caked_display_image"',
    ):
        assert token in source

    assert 'caked_image = bg.get("caked_display_image"' in source
    assert 'caked_image = bg["caked_image"]' not in source
    assert 'caked_region_bg = caked_log_image(np.asarray(profile_bg["caked_image"]' not in source


def test_background_peak_fits_notebook_uses_rotated_gaussian_peak_fits() -> None:
    source = _notebook_source()

    for token in (
        "GAUSSIAN_TAIL_DISTANCE_WEIGHT = 1.25",
        "GAUSSIAN_CORE_SIGNAL_DOWNSCALE = 0.06",
        "GAUSSIAN_TAIL_OVERPREDICTION_START = 0.55",
        "GAUSSIAN_TAIL_OVERPREDICTION_WEIGHT = 1.75",
        "def _rotated_gaussian_value_numba",
        "peak = math.exp(-0.5 * r2)",
        "_rotated_gaussian_residual_points_numba",
        "tail_overprediction_weight",
        "if residual > 0.0:",
        "residual *= tail_overprediction_weight[idx]",
        '"fit_model": "rotated_gaussian_plane"',
        "Fitted tail-aware peak sum",
        "def fit_joint_qz_peak_sum",
        "def gaussian_sum_qz_model",
        "def add_joint_qz_fit_columns",
        '"joint_fit_density"',
        '"joint_fit_peak_count"',
        "projected branch-point peaks in each rod/branch are fit together",
        "sub.get(\"joint_fit_density\", sub[\"fit_density\"])",
    ):
        assert token in source

    for removed in ("PSEUDO_VOIGT", "_pseudo_voigt", "Fitted pseudo-Voigt"):
        assert removed not in source


def test_joint_qz_fit_keeps_close_peak_valley_low() -> None:
    namespace = _notebook_functions(
        "rolling_lower_envelope",
        "gaussian_sum_qz_model",
        "_unique_sorted_markers",
        "_qz_grid_step",
        "_nanfilled_profile",
        "_smooth_qz_profile",
        "_nearest_marker_spacing",
        "_refine_centers_to_local_maxima",
        "_fallback_markers_from_profile",
        "_estimate_peak_hwhm",
        "_pearson_vii_profile",
        "_tail_aware_basis",
        "_weighted_nonnegative_amplitudes",
        "_aggregate_tail_components",
        "fit_joint_qz_peak_sum",
    )
    fit_joint_qz_peak_sum = namespace["fit_joint_qz_peak_sum"]
    x = np.linspace(0.82, 1.24, 180, dtype=np.float64)
    y = (
        0.16 * np.exp(-0.5 * ((x - 0.94) / 0.018) ** 2)
        + 1.0 * np.exp(-0.5 * ((x - 1.05) / 0.018) ** 2)
        + 0.015
    )

    payload = fit_joint_qz_peak_sum(x, y, np.array([0.94, 1.05], dtype=np.float64))

    assert payload["success"] is True
    model = np.asarray(payload["model_density"], dtype=np.float64)
    peak_height = float(np.nanmax(model))
    valley = float(model[int(np.nanargmin(np.abs(x - 0.995)))])
    assert valley < 0.12 * peak_height


def test_background_peak_fits_notebook_has_fast_state_parameters() -> None:
    source = _notebook_source()

    for token in (
        '"tags": [\n     "parameters"',
        'GUI_STATE_PATH = ""',
        'OUTPUT_DIR = ""',
        'RUN_NAME = ""',
        "RA_SIM_ALL_BACKGROUND_STATE",
        "RA_SIM_ALL_BACKGROUND_RUN_NAME",
        "RA_SIM_ALL_BACKGROUND_OUT_DIR",
        'f"{STATE_RUN_NAME}_state"',
        'print(f"run_name={STATE_RUN_NAME}")',
    ):
        assert token in source or token in NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_background_peak_fits_notebook_handles_background_reference_labels() -> None:
    source = _notebook_source()

    for token in (
        "def _is_specular_roi_candidate(",
        "abs(float(params[2])) <= float(phi_limit_deg)",
        "if not specs:",
        "skipped specular ROI examples: no fitted specular-like ROI entries were available",
        "for m_value, q_group in q_group_by_m.items():",
        "SAMPLE_LABEL = _sample_label_from_name(SAMPLE_NAME)",
    ):
        assert token in source


def test_background_peak_fits_runner_exists_for_batch_state_reruns() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    for token in (
        "def _execute_notebook(",
        "RA_SIM_ALL_BACKGROUND_STATE",
        "RA_SIM_ALL_BACKGROUND_RUN_NAME",
        "RA_SIM_ALL_BACKGROUND_OUT_DIR",
        "--keep-going",
        "state_count > 1",
    ):
        assert token in source


def test_background_peak_fits_runner_sanitizes_run_names() -> None:
    assert _safe_run_name(r"C:\data\New 4 fitted.json") == "New_4_fitted"
    assert _safe_run_name("../all.json") == "all"
    assert _safe_run_name("!!!") == "state"


def test_background_peak_fits_runner_separates_batch_outputs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    base = Path("artifacts/background_peak_fits")

    assert (
        _output_dir_for_state(
            base_out_dir=base,
            state_count=1,
            run_name="new4",
            repo_root=repo_root,
        )
        == repo_root / base
    )
    assert (
        _output_dir_for_state(
            base_out_dir=base,
            state_count=2,
            run_name="new4",
            repo_root=repo_root,
        )
        == repo_root / base / "new4_state"
    )
