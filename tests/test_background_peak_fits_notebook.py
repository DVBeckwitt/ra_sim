import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import re
from pathlib import Path
from textwrap import dedent

import matplotlib as mpl
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares, nnls

from scripts.diagnostics import background_peak_fit_worker as peak_fit_worker
from scripts.diagnostics.run_all_background_peak_fits import (
    _execute_notebook,
    _notebook_code_cells,
    _output_dir_for_state,
    _safe_run_name,
)


NOTEBOOK_PATH = Path("scripts/diagnostics/all_background_peak_fits.ipynb")
PARALLEL_NOTEBOOK_PATH = Path(
    "scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.ipynb"
)
PARALLEL_SCRIPT_PATH = Path(
    "scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py"
)
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
        "ROD_QZ_TAIL_WIDTH_SCALE_GRID": np.asarray(
            [0.55, 0.80, 1.15, 1.70, 2.50], dtype=np.float64
        ),
        "ROD_QZ_TAIL_CENTER_SEARCH_BINS": 5.0,
        "ROD_QZ_TAIL_MIN_HALFWIDTH_BINS": 0.55,
        "ROD_QZ_TAIL_MAX_HALFWIDTH_FRACTION": 0.70,
        "ROD_QZ_TAIL_MAX_AUTO_PEAKS": 10,
        "ROD_QZ_TAIL_COMPONENT_MIN_RELATIVE": 1.0e-5,
    }
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


def _script_functions(*names: str) -> dict[str, object]:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    wanted = set(names)
    tree = ast.parse(PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(PARALLEL_SCRIPT_PATH))
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    missing = wanted - {node.name for node in selected}
    assert not missing
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "mpl": mpl,
        "np": np,
        "pd": pd,
        "Path": Path,
        "plt": plt,
        "LogNorm": LogNorm,
        "JOURNAL_DETECTOR_CMAP": "magma",
        "least_squares": least_squares,
        "_rod_qz_nnls": nnls,
        "ROD_QZ_TAIL_POWER_GRID": np.asarray([0.75, 1.0, 1.35, 2.0, 3.5, 7.0], dtype=np.float64),
        "ROD_QZ_TAIL_WIDTH_SCALE_GRID": np.asarray(
            [0.55, 0.80, 1.15, 1.70, 2.50], dtype=np.float64
        ),
        "ROD_QZ_TAIL_CENTER_SEARCH_BINS": 5.0,
        "ROD_QZ_TAIL_MIN_HALFWIDTH_BINS": 0.55,
        "ROD_QZ_TAIL_MAX_HALFWIDTH_FRACTION": 0.70,
        "ROD_QZ_TAIL_MAX_AUTO_PEAKS": 10,
        "ROD_QZ_TAIL_COMPONENT_MIN_RELATIVE": 1.0e-5,
        "ROD_QZ_NONLINEAR_REFINEMENT_ENABLED": True,
        "ROD_QZ_NONLINEAR_MAX_COMPONENTS": 14,
        "ROD_QZ_NONLINEAR_CENTER_BOUND_BINS": 4.0,
        "ROD_QZ_NONLINEAR_TAIL_POWER_BOUNDS": (0.55, 12.0),
        "ROD_QZ_NONLINEAR_MAX_NFEV": 1200,
        "ROD_QZ_SHARED_LINEAR_BASELINE_ENABLED": True,
        "QR_ROD_FINAL_FIT_CACHE_SIGNATURE": "joint_qz_labeled_marker_fit_v2",
    }
    exec(compile(module, str(PARALLEL_SCRIPT_PATH), "exec"), namespace)
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
        'sub.get("joint_fit_density", sub["fit_density"])',
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


def test_parallel_script_joint_qz_fit_keeps_labeled_weak_hk0_marker() -> None:
    namespace = _script_functions(
        "_unique_sorted_markers",
        "_qz_grid_step",
        "_nanfilled_profile",
        "_center_search_profile",
        "_nearest_marker_spacing",
        "_refine_centers_to_local_maxima",
        "_fallback_markers_from_profile",
        "_estimate_peak_hwhm",
        "_pearson_vii_profile",
        "_tail_aware_basis",
        "_shared_linear_qz_baseline",
        "_weighted_nonnegative_amplitudes",
        "_aggregate_tail_components",
        "_pearson_vii_component_sum",
        "_component_density_to_sorted",
        "_refine_pearson_vii_components",
        "_empty_joint_qz_payload",
        "fit_joint_qz_peak_sum",
    )
    fit_joint_qz_peak_sum = namespace["fit_joint_qz_peak_sum"]
    x = np.linspace(1.0, 2.0, 260, dtype=np.float64)
    strong_004 = 1.0 * np.exp(-0.5 * ((x - 1.30) / 0.018) ** 2)
    weak_006 = 0.006 * np.exp(-0.5 * ((x - 1.72) / 0.018) ** 2)
    baseline = 0.018 + 0.004 * (x - 1.5)
    y = strong_004 + weak_006 + baseline

    payload = fit_joint_qz_peak_sum(x, y, np.array([1.30, 1.72], dtype=np.float64))

    assert payload["success"] is True
    markers = sorted(float(component["marker"]) for component in payload["components"])
    assert len(markers) == 2
    assert markers == pytest.approx([1.30, 1.72], abs=0.025)


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


def test_parallel_script_has_sample_name_override_parameter() -> None:
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    for token in (
        'SAMPLE_NAME_OVERRIDE = ""',
        "RA_SIM_ALL_BACKGROUND_SAMPLE_NAME",
        'SAMPLE_NAME_OVERRIDE_TEXT = _setting_text("SAMPLE_NAME_OVERRIDE", "RA_SIM_ALL_BACKGROUND_SAMPLE_NAME", "")',
        "SAMPLE_NAME_OVERRIDE_TEXT or _sample_name_from_path(primary_cif_path)",
    ):
        assert token in source


def test_background_peak_fits_runner_exists_for_batch_state_reruns() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    for token in (
        "def _execute_notebook(",
        "RA_SIM_ALL_BACKGROUND_PROCESS_GUARD",
        "RA_SIM_ALL_BACKGROUND_STATE",
        "RA_SIM_ALL_BACKGROUND_RUN_NAME",
        "RA_SIM_ALL_BACKGROUND_OUT_DIR",
        "BACKGROUND_FIT_BACKEND",
        "BACKGROUND_PROCESS_NUMBA_THREADS",
        "--fit-backend",
        "--process-numba-threads",
        "notebook={notebook_path}",
        "--keep-going",
        "state_count > 1",
    ):
        assert token in source


def test_background_peak_fits_runner_reads_python_diagnostic_source(tmp_path: Path) -> None:
    diagnostic_path = tmp_path / "diagnostic.py"
    diagnostic_path.write_text("print('script source executed')\n", encoding="utf-8")

    assert _notebook_code_cells(diagnostic_path) == [(0, "print('script source executed')\n")]


def test_background_peak_fits_runner_executes_python_diagnostic_with_process_guard(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "out"
    diagnostic_path = tmp_path / "diagnostic.py"
    diagnostic_path.write_text(
        dedent(
            """
            import json
            import os
            from pathlib import Path

            out_dir = Path(os.environ['RA_SIM_ALL_BACKGROUND_OUT_DIR'])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / 'capture.json').write_text(json.dumps({
                'guard': os.environ.get('RA_SIM_ALL_BACKGROUND_PROCESS_GUARD'),
                'state': os.environ.get('RA_SIM_ALL_BACKGROUND_STATE'),
                'run_name': os.environ.get('RA_SIM_ALL_BACKGROUND_RUN_NAME'),
                'backend': os.environ.get('BACKGROUND_FIT_BACKEND'),
                'process_numba_threads': os.environ.get('BACKGROUND_PROCESS_NUMBA_THREADS'),
                'file': __file__,
            }), encoding='utf-8')
            """
        ).lstrip(),
        encoding="utf-8",
    )

    _execute_notebook(
        notebook_path=diagnostic_path,
        state_path=state_path,
        run_name="state",
        out_dir=out_dir,
        repo_root=repo_root,
        fit_workers=2,
        numba_threads=3,
        fit_backend="process",
        process_numba_threads=1,
    )

    capture = json.loads((out_dir / "capture.json").read_text(encoding="utf-8"))
    assert capture == {
        "guard": "1",
        "state": str(state_path),
        "run_name": "state",
        "backend": "process",
        "process_numba_threads": "1",
        "file": str(diagnostic_path),
    }
    assert os.environ.get("RA_SIM_ALL_BACKGROUND_PROCESS_GUARD") is None


def test_parallel_script_windows_process_backend_requires_runner_guard() -> None:
    namespace = _script_functions("normalize_fit_backend")
    normalize_fit_backend = namespace["normalize_fit_backend"]

    assert normalize_fit_backend("process", workers=28, platform_name="nt") == "thread"
    assert (
        normalize_fit_backend(
            "process",
            workers=28,
            platform_name="nt",
            process_guard_enabled=True,
        )
        == "process"
    )
    assert (
        normalize_fit_backend(
            "auto",
            workers=28,
            platform_name="nt",
            process_guard_enabled=True,
        )
        == "process"
    )


def test_parallel_script_hk0_l3_star_crop_bounds_contains_beam_and_peak() -> None:
    namespace = _script_functions("hk0_star_crop_bounds")
    crop_bounds = namespace["hk0_star_crop_bounds"]

    bounds = crop_bounds(
        (200, 300),
        (150.0, 160.0),
        (148.0, 60.0),
        lateral_half_width_px=12,
        above_peak_padding_px=20,
        below_beam_padding_px=5,
    )

    assert bounds is not None
    row_slice, col_slice = bounds
    assert row_slice.start <= 40
    assert row_slice.stop >= 166
    assert col_slice.start <= 136
    assert col_slice.stop >= 163


def test_parallel_script_hk0_l3_star_crop_bounds_clips_and_rejects_invalid() -> None:
    namespace = _script_functions("hk0_star_crop_bounds")
    crop_bounds = namespace["hk0_star_crop_bounds"]

    row_slice, col_slice = crop_bounds(
        (80, 70),
        (4.0, 78.0),
        (2.0, 1.0),
        lateral_half_width_px=20,
        above_peak_padding_px=20,
        below_beam_padding_px=10,
    )

    assert row_slice.start == 0
    assert row_slice.stop == 80
    assert col_slice.start == 0
    assert col_slice.stop <= 70
    assert crop_bounds((80, 70), (np.nan, 78.0), (2.0, 1.0)) is None
    assert crop_bounds((80, 70), (4.0, 78.0), (np.inf, 1.0)) is None


def test_parallel_script_hk0_l3_star_selects_l3_marker_center() -> None:
    namespace = _script_functions("hk0_star_marker_center")
    marker_center = namespace["hk0_star_marker_center"]
    markers = pd.DataFrame(
        [
            {
                "m": 0,
                "l": 1,
                "refined_two_theta_deg": 20.0,
                "refined_phi_deg": 0.0,
            },
            {
                "m": 0,
                "l": 3,
                "refined_two_theta_deg": 12.0,
                "refined_phi_deg": 3.0,
            },
            {
                "m": 1,
                "l": 3,
                "refined_two_theta_deg": 99.0,
                "refined_phi_deg": 9.0,
            },
        ]
    )

    center = marker_center(markers, projector=lambda theta, phi: (theta + 1.0, phi + 2.0))

    assert center == (13.0, 5.0)


def test_parallel_script_hk0_l3_star_save_writes_png(tmp_path) -> None:
    namespace = _script_functions(
        "detector_display_cmap",
        "detector_intensity_display",
        "detector_log_norm",
        "hk0_star_crop_bounds",
        "save_hk0_star_crop",
    )
    save_crop = namespace["save_hk0_star_crop"]
    image = np.arange(10_000, dtype=np.float64).reshape(100, 100)
    out_path = tmp_path / "hk0_l3_star.png"

    result = save_crop(
        image,
        out_path,
        beam_center=(50.0, 82.0),
        peak_center=(52.0, 28.0),
        lateral_half_width_px=18,
        above_peak_padding_px=16,
        below_beam_padding_px=8,
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    rendered = plt.imread(out_path)
    assert rendered.shape[-1] in {3, 4}
    assert not np.allclose(rendered[..., 0], rendered[..., 1])


def test_parallel_script_hk0_l3_star_save_uses_detector_log_color() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def save_hk0_star_crop(") : source.index("\ndef label_from_entry(")
    ]

    assert "detector_display_cmap()" in function_source
    assert "detector_intensity_display(crop)" in function_source
    assert "detector_log_norm([display_crop]" in function_source
    assert 'cmap="gray"' not in function_source


def test_parallel_script_hk0_l3_star_output_is_wired() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")

    for token in (
        '"hk0_l3_star.png"',
        "hk0_star_marker_center(",
        "save_hk0_star_crop(",
        "detector_xy_from_caked_angles(profile_bg",
        "skipped hk0_l3_star.png",
        "HK=0, L=3",
    ):
        assert token in source


def test_parallel_script_qr_rod_marker_group_edits_replace_only_one_group() -> None:
    namespace = _script_functions("replace_qr_rod_marker_group_qz")
    replace_group = namespace["replace_qr_rod_marker_group_qz"]
    markers = pd.DataFrame(
        [
            {"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0, "display_l": 2.0},
            {"m": 1, "branch": "+", "qz_marker": 2.0, "fit_l": 4.0, "display_l": 4.0},
            {"m": 3, "branch": "-", "qz_marker": 5.0, "fit_l": 8.0, "display_l": 8.0},
        ]
    )

    edited = replace_group(markers, m_value=1, branch_value="+", qz_values=[2.25, 0.75, 0.75])

    edited_group = edited[(edited["m"] == 1) & (edited["branch"] == "+")]
    other_group = edited[(edited["m"] == 3) & (edited["branch"] == "-")]
    assert edited_group["qz_marker"].tolist() == [0.75, 2.25]
    assert edited_group["marker_source"].tolist() == ["manual_edit", "manual_edit"]
    assert edited_group["fit_l"].tolist() == pytest.approx([1.5, 4.5])
    assert other_group["qz_marker"].tolist() == [5.0]


def test_parallel_script_qr_rod_marker_table_includes_specular_hk0_markers() -> None:
    namespace = _script_functions("marker_table_with_specular_l_markers")
    include_specular = namespace["marker_table_with_specular_l_markers"]
    markers = pd.DataFrame(
        [{"m": 1, "branch": "+", "hkl": "-1,0,2", "qz_marker": 1.0}]
    )
    specular = pd.DataFrame(
        [
            {"m": 0, "branch": "qz", "hkl": "0,0,1", "qz_marker": 0.5},
            {"m": 0, "branch": "qz", "hkl": "0,0,1", "qz_marker": 0.5},
            {"m": 0, "branch": "qz", "hkl": "0,0,3", "qz_marker": 1.5},
        ]
    )

    merged = include_specular(markers, specular)

    assert merged[merged["m"] == 0]["hkl"].tolist() == ["0,0,1", "0,0,3"]
    assert merged[merged["m"] == 1]["hkl"].tolist() == ["-1,0,2"]


def test_parallel_script_qr_rod_editor_qz_l_axis_coefficients() -> None:
    namespace = _script_functions("qz_l_linear_coeff_from_marker_rows")
    qz_l_coeff = namespace["qz_l_linear_coeff_from_marker_rows"]
    markers = pd.DataFrame(
        [
            {"qz_marker": 0.75, "fit_l": 3},
            {"qz_marker": 1.50, "fit_l": 6},
        ]
    )

    slope, intercept = qz_l_coeff(markers)

    assert slope == pytest.approx(4.0)
    assert intercept == pytest.approx(0.0)


def test_parallel_script_qr_rod_snap_moves_all_panel_markers_to_local_peaks() -> None:
    namespace = _script_functions("snap_qr_rod_markers_to_profile_peaks")
    snap_markers = namespace["snap_qr_rod_markers_to_profile_peaks"]
    x = np.linspace(0.0, 3.0, 301, dtype=np.float64)
    y = (
        np.exp(-0.5 * ((x - 0.95) / 0.025) ** 2)
        + 0.7 * np.exp(-0.5 * ((x - 2.05) / 0.030) ** 2)
    )

    snapped = snap_markers([0.90, 2.12], x, y)

    assert snapped.tolist() == pytest.approx([0.95, 2.05], abs=0.011)


def test_parallel_script_qr_rod_marker_hash_changes_cache_key() -> None:
    namespace = _script_functions("qr_rod_peak_edit_cache_key")
    cache_key = namespace["qr_rod_peak_edit_cache_key"]
    markers = pd.DataFrame(
        [{"m": 1, "branch": "+", "qz_marker": 1.0, "fit_l": 2.0, "display_l": 2.0}]
    )
    shifted = markers.copy()
    shifted.loc[0, "qz_marker"] = 1.25

    assert cache_key(None) == {
        "mode": "last_cached",
        "fit_signature": "joint_qz_labeled_marker_fit_v2",
    }
    assert cache_key(None, marker_table=markers, mode="popup") != cache_key(
        None, marker_table=shifted, mode="popup"
    )
    assert cache_key(None, marker_table=markers, mode="popup")["mode"] == "popup"


def test_parallel_script_marker_title_changes_cache_key() -> None:
    namespace = _script_functions("qr_rod_peak_edit_cache_key")
    cache_key = namespace["qr_rod_peak_edit_cache_key"]
    markers = pd.DataFrame(
        [
            {
                "m": 1,
                "branch": "+",
                "qz_marker": 1.0,
                "fit_l": 2.0,
                "display_l": 2.0,
                "marker_title": "L=2",
            }
        ]
    )
    relabeled = markers.copy()
    relabeled.loc[0, "marker_title"] = "L=2 shoulder"

    assert cache_key(None, marker_table=markers, mode="popup") != cache_key(
        None, marker_table=relabeled, mode="popup"
    )


def test_parallel_script_qr_rod_final_cache_requires_fit_signature() -> None:
    namespace = _script_functions("qr_rod_profile_cache_has_final_fit")
    cache_has_final_fit = namespace["qr_rod_profile_cache_has_final_fit"]
    payload = {
        "final_rod_profile_table": pd.DataFrame(
            {"qz_center": [1.0, 2.0], "joint_fit_density": [0.1, 0.2]}
        ),
        "final_marker_table": pd.DataFrame(
            {"qz_marker": [1.0], "fit_l": [6.0], "display_l": [6.0]}
        ),
        "final_rod_component_table": pd.DataFrame(),
        "final_peak_edit_cache_key": {"mode": "last_cached"},
    }

    assert not cache_has_final_fit(
        payload,
        {"mode": "last_cached", "fit_signature": "joint_qz_labeled_marker_fit_v2"},
    )
    payload["final_peak_edit_cache_key"] = {
        "mode": "last_cached",
        "fit_signature": "joint_qz_labeled_marker_fit_v2",
    }
    assert cache_has_final_fit(
        payload,
        {"mode": "last_cached", "fit_signature": "joint_qz_labeled_marker_fit_v2"},
    )


def test_parallel_script_qr_rod_peak_edit_runtime_mode_respects_headless() -> None:
    namespace = _script_functions("qr_rod_peak_edit_runtime_mode")
    runtime_mode = namespace["qr_rod_peak_edit_runtime_mode"]

    assert runtime_mode("auto", backend_name="agg", env={}) == "skip"
    assert runtime_mode("auto", backend_name="TkAgg", env={}) == "popup"
    assert runtime_mode("auto", backend_name="TkAgg", env={"CI": "1"}) == "skip"
    assert runtime_mode("popup", backend_name="agg", env={"CI": "1"}) == "popup"
    assert runtime_mode("skip", backend_name="TkAgg", env={}) == "skip"


def test_parallel_script_qr_rod_peak_edits_round_trip_json(tmp_path: Path) -> None:
    namespace = _script_functions(
        "clean_marker_title",
        "load_qr_rod_peak_edits",
        "write_qr_rod_peak_edits",
    )
    write_edits = namespace["write_qr_rod_peak_edits"]
    load_edits = namespace["load_qr_rod_peak_edits"]
    edit_path = tmp_path / "qr_rod_peak_edits.json"
    markers = pd.DataFrame(
        [
            {
                "m": 1,
                "branch": "+",
                "qz_marker": 1.0,
                "fit_l": 2.0,
                "display_l": 2.0,
                "marker_title": "L=2",
            }
        ]
    )

    write_edits(edit_path, markers)
    loaded = load_edits(edit_path)

    assert loaded[
        ["m", "branch", "qz_marker", "fit_l", "display_l", "marker_title"]
    ].to_dict("records") == [
        {
            "m": 1,
            "branch": "+",
            "qz_marker": 1.0,
            "fit_l": 2.0,
            "display_l": 2.0,
            "marker_title": "L=2",
        }
    ]


def test_parallel_script_marker_title_overrides_final_rod_label() -> None:
    namespace = _script_functions(
        "clean_marker_title",
        "marker_row_title",
        "l_tick_label",
        "hk_display_label",
        "rod_marker_annotation_label",
    )
    marker_label = namespace["rod_marker_annotation_label"]

    assert marker_label({"display_l": 2.0}, 1) == "L=2"
    assert marker_label({"display_l": 2.49}, 1) == "L=2"
    assert marker_label({"display_l": 2.51}, 1) == "L=3"
    assert marker_label({"display_l": 2.0, "marker_title": "L=2 shoulder"}, 1) == "L=2 shoulder"
    assert marker_label({"display_l": 2.51, "marker_title": "L=2.51 custom"}, 1) == "L=2.51 custom"
    assert marker_label({"display_l": 2.0, "marker_title": "  "}, 1) == "L=2"


def test_parallel_script_final_rod_labels_point_from_upper_right() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def annotate_rod_profile_hk_locations(") : source.index(
            "\ndef qz_to_l_linear_coeff("
        )
    ]

    assert "xytext=(12.0, 12.0 + 5.0 * (marker_index % 3))" in function_source
    assert 'ha="left"' in function_source
    assert 'va="bottom"' in function_source
    assert '"arrowstyle": "->"' in function_source


def test_parallel_script_qr_rod_peak_editor_is_wired_before_joint_fit_cache() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    marker_table_index = source.index("marker_table = pd.DataFrame(marker_rows)")
    specular_call = source.index(
        "specular_l_marker_table = specular_l_marker_rows_for_background",
        marker_table_index,
    )
    specular_merge = source.index(
        "marker_table = marker_table_with_specular_l_markers(",
        specular_call,
    )
    editor_call = source.index("edit_qr_rod_peak_markers(", marker_table_index)
    cache_key_call = source.index("qr_rod_peak_edit_cache_key(", marker_table_index)
    final_fit_call = source.index("add_joint_qz_fit_columns(", marker_table_index)

    assert specular_call < specular_merge < editor_call < cache_key_call < final_fit_call
    assert "RA_SIM_QR_ROD_PEAK_EDIT_MODE" in source
    assert '"QR_ROD_PEAK_EDIT_MODE_OVERRIDE", "RA_SIM_QR_ROD_PEAK_EDIT_MODE", "popup"' in source
    assert "TextBox(" in source
    assert "marker_title" in source
    assert 'getattr(event, "inaxes", None) is getattr(box, "ax", None)' in source


def test_parallel_script_qr_rod_peak_editor_uses_l_axis() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    source = PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8")
    function_source = source[
        source.index("def show_qr_rod_peak_marker_popup(") : source.index(
            "\ndef edit_qr_rod_peak_markers("
        )
    ]

    assert "qz_l_linear_coeff_from_marker_rows(" in function_source
    assert "qz_to_editor_l(" in function_source
    assert "editor_l_to_qz(" in function_source
    assert 'ax.set_xlabel("L"' in function_source
    assert "MaxNLocator(integer=True)" in function_source
    assert 'ax.set_xlabel("Qz"' not in function_source


def test_parallel_script_defines_rod_marker_label_before_first_call() -> None:
    if not PARALLEL_SCRIPT_PATH.exists():
        pytest.skip(f"{PARALLEL_SCRIPT_PATH} is not present in this checkout")
    tree = ast.parse(PARALLEL_SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(PARALLEL_SCRIPT_PATH))
    definition_line = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "rod_marker_annotation_label"
    )
    call_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "rod_marker_annotation_label"
    ]

    assert call_lines
    assert definition_line < min(call_lines)


def test_parallel_background_peak_fits_notebook_uses_process_pool_worker() -> None:
    if not PARALLEL_NOTEBOOK_PATH.exists():
        pytest.skip(f"{PARALLEL_NOTEBOOK_PATH} is not present in this checkout")
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(PARALLEL_NOTEBOOK_PATH.read_text(encoding="utf-8")).get(
            "cells",
            [],
        )
    )

    for token in (
        "ProcessPoolExecutor",
        "fit_peak_from_process_job",
        "save_peak_fit_background_arrays",
        "BACKGROUND_FIT_BACKEND",
        "BACKGROUND_PROCESS_NUMBA_THREADS",
        "TemporaryDirectory",
        "backend={active_fit_backend}",
        "pids={len(fit_pid_counts)}",
        "_attach_fit_detector_projection(item, prep)",
        "optimizer_success_count",
        "success_count != expected_fit_count",
    ):
        assert token in source


def test_parallel_background_peak_fits_notebook_uses_gaussian_core_lorentzian_tail_model() -> None:
    if not PARALLEL_NOTEBOOK_PATH.exists():
        pytest.skip(f"{PARALLEL_NOTEBOOK_PATH} is not present in this checkout")
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(PARALLEL_NOTEBOOK_PATH.read_text(encoding="utf-8")).get(
            "cells",
            [],
        )
    )

    for token in (
        'FIT_MODEL_NAME = "rotated_gaussian_core_lorentzian_tail_shared_center"',
        "GAUSSIAN_FWHM_TO_SIGMA = 2.3548200450309493",
        "gamma_l_u = max(0.5 * params[9], 1.0e-12)",
        "return params[6] + params[7] * dt + params[8] * dp + peak",
        "fwhm_l_u_max = max(4.0 * theta_half_window, fwhm_g_u_max)",
        "eta_tail_seeds = (0.0, 0.15, 0.35, 0.60)",
        "0.65 * weighted_score",
        "fit_parameter_warnings",
        '"fit_fwhm_lorentzian_u_deg"',
        "3.0 * float(p[9])",
    ):
        assert token in source

    for removed in (
        "rotated_gaussian_peak_only_output_with_shared_linear_two_theta_baseline",
        "baseline_equation\": \"density = b + m*(two_theta_deg - seed_two_theta_deg)",
    ):
        assert removed not in source


def _synthetic_peak_fit_case(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    theta_axis = np.linspace(8.8, 11.2, 72, dtype=np.float64)
    phi_axis = np.linspace(-3.2, 3.2, 80, dtype=np.float64)
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    image = (
        12.0
        + 0.8 * (theta_grid - 10.0)
        + 90.0
        * np.exp(-0.5 * (((theta_grid - 10.05) / 0.17) ** 2 + ((phi_grid - 0.25) / 0.55) ** 2))
    )
    entry = {
        "_background_index": 0,
        "_background_name": "synthetic",
        "_label": "1,0,0",
        "_branch": "+",
        "_theta_seed_deg": 10.0,
        "_phi_seed_deg": 0.2,
    }
    paths = peak_fit_worker.save_peak_fit_background_arrays(
        tmp_path,
        background_index=0,
        caked_image=image,
        theta_axis=theta_axis,
        phi_axis=phi_axis,
    )
    return entry, paths, image, theta_axis, phi_axis


def _mixed_peak_image(
    params: np.ndarray,
    theta_axis: np.ndarray,
    phi_axis: np.ndarray,
    *,
    noise_sigma: float = 0.0,
) -> np.ndarray:
    theta_grid, phi_grid = np.meshgrid(theta_axis, phi_axis)
    amp, theta0, phi0 = float(params[0]), float(params[1]), float(params[2])
    fwhm_g_u, fwhm_g_v = float(params[3]), float(params[4])
    angle = float(params[5])
    baseline, theta_slope, phi_slope = float(params[6]), float(params[7]), float(params[8])
    fwhm_l_u, fwhm_l_v, eta_tail = float(params[9]), float(params[10]), float(params[11])
    dt = theta_grid - theta0
    dp = peak_fit_worker.wrapped_delta_deg(phi_grid, phi0)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    u = cos_a * dt + sin_a * dp
    v = -sin_a * dt + cos_a * dp
    sigma_g_u = fwhm_g_u / 2.3548200450309493
    sigma_g_v = fwhm_g_v / 2.3548200450309493
    gamma_l_u = 0.5 * fwhm_l_u
    gamma_l_v = 0.5 * fwhm_l_v
    gaussian_core = np.exp(-0.5 * ((u / sigma_g_u) ** 2 + (v / sigma_g_v) ** 2))
    lorentzian_tail = 1.0 / (1.0 + (u / gamma_l_u) ** 2 + (v / gamma_l_v) ** 2)
    image = (
        baseline
        + theta_slope * dt
        + phi_slope * dp
        + amp * ((1.0 - eta_tail) * gaussian_core + eta_tail * lorentzian_tail)
    )
    if noise_sigma:
        image = image + np.random.default_rng(4).normal(0.0, noise_sigma, image.shape)
    return np.asarray(image, dtype=np.float64)


def test_background_peak_fit_worker_memmap_job_matches_direct_fit(tmp_path: Path) -> None:
    settings = peak_fit_worker.peak_fit_settings_from_values(PEAK_FIT_MAX_NFEV=250)
    peak_fit_worker.configure_peak_fit_worker(settings, numba_threads=1)
    entry, paths, image, theta_axis, phi_axis = _synthetic_peak_fit_case(tmp_path)

    direct = peak_fit_worker.fit_one_peak(entry, image, theta_axis, phi_axis)
    job = peak_fit_worker.make_peak_fit_job(
        background_index=0,
        local_index=0,
        entry=entry,
        **paths,
    )
    try:
        via_job = peak_fit_worker.fit_peak_from_job(job)["item"]
    finally:
        peak_fit_worker.clear_peak_fit_worker_array_cache()

    direct_params = np.asarray(direct["params"], dtype=np.float64)
    job_params = np.asarray(via_job["params"], dtype=np.float64)
    assert direct_params.size == 12
    assert job_params.size == 12
    assert np.allclose(job_params, direct_params, rtol=1.0e-7, atol=1.0e-7)
    assert direct["fit_model"] == "rotated_gaussian_core_lorentzian_tail_shared_center"
    assert abs(float(job_params[1]) - 10.05) < 0.08
    assert abs(float(job_params[2]) - 0.25) < 0.20


def test_background_peak_fit_worker_runs_in_process_pool(tmp_path: Path) -> None:
    settings = peak_fit_worker.peak_fit_settings_from_values(PEAK_FIT_MAX_NFEV=250)
    entry, paths, _image, _theta_axis, _phi_axis = _synthetic_peak_fit_case(tmp_path)
    job = peak_fit_worker.make_peak_fit_job(
        background_index=0,
        local_index=0,
        entry=entry,
        **paths,
    )

    with ProcessPoolExecutor(
        max_workers=1,
        initializer=peak_fit_worker.configure_peak_fit_worker,
        initargs=(settings, 1),
    ) as executor:
        payload = executor.submit(peak_fit_worker.fit_peak_from_job, job).result(timeout=45)

    assert int(payload["background_index"]) == 0
    assert int(payload["local_index"]) == 0
    assert int(payload["pid"]) != os.getpid()
    params = np.asarray(payload["item"]["params"], dtype=np.float64)
    assert params.size == 12
    assert abs(float(params[1]) - 10.05) < 0.08
    assert abs(float(params[2]) - 0.25) < 0.20


def test_background_peak_fit_worker_recovers_gaussian_core_lorentzian_tail() -> None:
    theta_axis = np.linspace(8.8, 11.2, 90, dtype=np.float64)
    phi_axis = np.linspace(-4.0, 4.0, 96, dtype=np.float64)
    true_params = np.asarray(
        [
            85.0,
            10.06,
            0.32,
            0.42,
            1.05,
            np.deg2rad(20.0),
            14.0,
            0.9,
            -0.45,
            1.25,
            3.20,
            0.35,
        ],
        dtype=np.float64,
    )
    image = _mixed_peak_image(true_params, theta_axis, phi_axis, noise_sigma=0.15)
    entry = {
        "_background_index": 0,
        "_background_name": "synthetic",
        "_label": "1,0,1",
        "_branch": "+",
        "_theta_seed_deg": 10.0,
        "_phi_seed_deg": 0.25,
    }
    settings = peak_fit_worker.peak_fit_settings_from_values(
        PEAK_FIT_MAX_NFEV=500,
        GAUSSIAN_CORE_SIGNAL_DOWNSCALE=0.0,
    )
    peak_fit_worker.configure_peak_fit_worker(settings, numba_threads=1)

    item = peak_fit_worker.fit_one_peak(entry, image, theta_axis, phi_axis)
    params = np.asarray(item["params"], dtype=np.float64)

    assert item["fit_model"] == "rotated_gaussian_core_lorentzian_tail_shared_center"
    assert item["fit_parameter_warnings"] == []
    assert abs(float(params[1]) - float(true_params[1])) < 0.25 * np.median(np.diff(theta_axis))
    assert abs(float(params[2]) - float(true_params[2])) < 0.25 * np.median(np.diff(phi_axis))
    assert abs(np.rad2deg(float(params[5] - true_params[5]))) < 5.0
    assert np.allclose(params[[3, 4]], true_params[[3, 4]], rtol=0.20)
    assert np.allclose(params[[9, 10]], true_params[[9, 10]], rtol=0.30)
    assert abs(float(params[11]) - float(true_params[11])) < 0.10


def test_background_peak_fit_worker_distinguishes_fit_from_optimizer_success(
    tmp_path: Path,
) -> None:
    settings = peak_fit_worker.peak_fit_settings_from_values(PEAK_FIT_MAX_NFEV=1)
    peak_fit_worker.configure_peak_fit_worker(settings, numba_threads=1)
    entry, _paths, image, theta_axis, phi_axis = _synthetic_peak_fit_case(tmp_path)

    item = peak_fit_worker.fit_one_peak(entry, image, theta_axis, phi_axis)

    assert item["success"] is True
    assert "optimizer_success" in item


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
